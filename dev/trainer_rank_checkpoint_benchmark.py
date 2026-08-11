"""Profile portable TrainerRank checkpoint save/load under ``torchrun``.

The JSON schema is intentionally stable so results from different commits and
topologies can be compared. Caladan augments ``post_queue_upload_seconds``.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Literal

import torch
import torch.distributed as dist

from art.megatron.model_support.lora_disk import load_adapter_config
from art.megatron.weights import lora_publish
from art.trainer_rank import AdamParams, TrainerRank

Operation = Literal["load", "save", "roundtrip"]


def _tensor_bytes(shape: Sequence[int], dtype_name: str) -> int:
    dtype = getattr(torch, dtype_name)
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


class _CheckpointProbe:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.gather_seconds = 0.0
        self.sent_bytes = 0
        self._exchange = lora_publish._exchange_tensors

    def install(self) -> None:
        def exchange(metadata, **kwargs):
            started = time.perf_counter()
            try:
                return self._exchange(metadata, **kwargs)
            finally:
                self.gather_seconds += time.perf_counter() - started
                self.sent_bytes += sum(
                    _tensor_bytes(meta.shape, meta.dtype_name)
                    for meta in metadata
                    if meta.owner_rank == self.rank and self.rank != 0
                )

        setattr(lora_publish, "_exchange_tensors", exchange)

    def restore(self) -> None:
        setattr(lora_publish, "_exchange_tensors", self._exchange)


class _RssSampler:
    def __init__(self) -> None:
        self.initial = self._rss()
        self.peak = self.initial
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    @staticmethod
    def _rss() -> int:
        with Path("/proc/self/statm").open() as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")

    def _sample(self) -> None:
        while not self._stop.wait(0.01):
            self.peak = max(self.peak, self._rss())

    def __enter__(self) -> _RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._rss())


def _artifact_size(path: str | None) -> int | None:
    if path is None or not Path(path).is_dir():
        return None
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def _topology(args: argparse.Namespace, world_size: int) -> dict[str, int]:
    model_parallel = args.tp * args.pp * args.cp
    if world_size % model_parallel:
        raise ValueError(
            f"world_size={world_size} is not divisible by tp*pp*cp={model_parallel}"
        )
    return {
        "world_size": world_size,
        "tp": args.tp,
        "pp": args.pp,
        "cp": args.cp,
        "ep": args.ep,
        "etp": args.etp,
        "dp": world_size // model_parallel,
    }


def _initialize_optimizer_state(
    trainer: TrainerRank, checkpoint: str, learning_rate: float
) -> int:
    dynamic = trainer._dynamic_optimizers.get(checkpoint)
    if dynamic is None:
        dynamic = trainer._new_dynamic_optimizer(
            checkpoint, AdamParams(learning_rate=learning_rate)
        )
        trainer._dynamic_optimizers[checkpoint] = dynamic
    for index, master in enumerate(dynamic.master_params, start=1):
        master.grad = torch.full_like(master, 1e-3 * index)
    dynamic.optimizer.step()
    dynamic.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for model, master in zip(
            trainer._checkpoint_slot_params_by_name[checkpoint],
            dynamic.master_params,
            strict=True,
        ):
            model.copy_(master)
            model.grad = None

    resident_bytes = 0
    for master in dynamic.master_params:
        state = dynamic.optimizer.state[master]
        moments = (state.get("exp_avg"), state.get("exp_avg_sq"))
        if not all(
            isinstance(moment, torch.Tensor) and bool(torch.count_nonzero(moment))
            for moment in moments
        ):
            raise RuntimeError("benchmark optimizer moments were not initialized")
        step = state.get("step")
        if not isinstance(step, torch.Tensor) or float(step.item()) < 1:
            raise RuntimeError("benchmark optimizer step was not initialized")
        resident_bytes += master.numel() * master.element_size()
        resident_bytes += sum(
            moment.numel() * moment.element_size()
            for moment in moments
            if isinstance(moment, torch.Tensor)
        )
    return resident_bytes


async def _exercise(
    trainer: TrainerRank,
    *,
    operation: Operation,
    source: str,
    output: str | None,
    learning_rate: float,
) -> dict[str, float | int | None]:
    started = time.perf_counter()
    await trainer.load_checkpoint(source)
    input_load = time.perf_counter() - started
    save_total: float | None = None
    queue_pause: float | None = None
    post_queue_serialization: float | None = None
    restored_load: float | None = None
    resident_optimizer_bytes: int | None = None
    if operation in {"save", "roundtrip"}:
        if output is None:
            raise ValueError("--output is required for save and roundtrip")
        resident_optimizer_bytes = _initialize_optimizer_state(
            trainer, source, learning_rate
        )
        started = time.perf_counter()
        trainer._prepare_checkpoint_save(output, source)
        queue_pause = time.perf_counter() - started
        started = time.perf_counter()
        trainer._finish_checkpoint_save(output)
        post_queue_serialization = time.perf_counter() - started
        save_total = queue_pause + post_queue_serialization
    if operation == "roundtrip":
        assert output is not None
        started = time.perf_counter()
        await trainer.load_checkpoint(output)
        restored_load = time.perf_counter() - started
    return {
        "input_load_seconds": input_load,
        "save_total_seconds": save_total,
        "queue_pause_seconds": queue_pause,
        "post_queue_serialization_seconds": post_queue_serialization,
        "load_total_seconds": restored_load
        if restored_load is not None
        else input_load,
        "ready_seconds": restored_load if restored_load is not None else input_load,
        "resident_optimizer_state_bytes": resident_optimizer_bytes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--operation", choices=("load", "save", "roundtrip"), default="roundtrip"
    )
    parser.add_argument("--model")
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint benchmark requires CUDA")
    device = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    topology = _topology(args, world_size)
    for key, value in (
        ("TENSOR_MODEL", args.tp),
        ("PIPELINE_MODEL", args.pp),
        ("CONTEXT", args.cp),
        ("EXPERT_MODEL", args.ep),
        ("EXPERT_TENSOR", args.etp),
    ):
        os.environ[f"ART_MEGATRON_{key}_PARALLEL_SIZE"] = str(value)

    try:
        from art.megatron import train as megatron_train

        config = load_adapter_config(args.source)
        model = args.model or config.get("base_model_name_or_path")
        if not isinstance(model, str) or not model:
            raise ValueError(
                "--model or checkpoint base_model_name_or_path is required"
            )
        runtime = megatron_train.build_training_runtime(
            model_identifier=model,
            provider_configure=(
                (lambda provider: setattr(provider, "num_layers", args.layers))
                if args.layers > 0
                else None
            ),
            print_env=rank == 0,
        )
        trainer = TrainerRank(runtime)
        probe = _CheckpointProbe(rank)
        torch.cuda.reset_peak_memory_stats(device)
        initial_gpu = torch.cuda.memory_allocated(device)
        probe.install()
        try:
            with _RssSampler() as rss:
                timings = asyncio.run(
                    _exercise(
                        trainer,
                        operation=args.operation,
                        source=args.source,
                        output=args.output,
                        learning_rate=args.learning_rate,
                    )
                )
        finally:
            probe.restore()
        rank_metrics = {
            **timings,
            "rank": rank,
            "snapshot_gather_seconds": probe.gather_seconds,
            "communication_sent_bytes": probe.sent_bytes,
            "peak_cpu_rss_bytes": rss.peak,
            "peak_cpu_rss_delta_bytes": max(rss.peak - rss.initial, 0),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_gpu_allocated_delta_bytes": max(
                torch.cuda.max_memory_allocated(device) - initial_gpu, 0
            ),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        all_metrics: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(all_metrics, rank_metrics)
        if rank == 0:
            ranks = [value for value in all_metrics if value is not None]

            def maximum(key: str) -> float | int | None:
                values = [value[key] for value in ranks if value[key] is not None]
                return max(values) if values else None

            aggregate = {
                "queue_pause_seconds": maximum("queue_pause_seconds"),
                "snapshot_gather_seconds": maximum("snapshot_gather_seconds"),
                "save_total_seconds": maximum("save_total_seconds"),
                "load_total_seconds": maximum("load_total_seconds"),
                "post_queue_serialization_seconds": maximum(
                    "post_queue_serialization_seconds"
                ),
                "post_queue_upload_seconds": None,
                "distributed_communication_bytes": sum(
                    value["communication_sent_bytes"] for value in ranks
                ),
                "artifact_size_bytes": _artifact_size(args.output or args.source),
                "resident_optimizer_state_bytes": maximum(
                    "resident_optimizer_state_bytes"
                ),
                "peak_cpu_rss_bytes": maximum("peak_cpu_rss_bytes"),
                "peak_cpu_rss_delta_bytes": maximum("peak_cpu_rss_delta_bytes"),
                "peak_gpu_allocated_bytes": maximum("peak_gpu_allocated_bytes"),
                "peak_gpu_allocated_delta_bytes": maximum(
                    "peak_gpu_allocated_delta_bytes"
                ),
                "peak_gpu_reserved_bytes": maximum("peak_gpu_reserved_bytes"),
                "ready_seconds": maximum("ready_seconds"),
            }
            payload = {
                "schema_version": 1,
                "operation": args.operation,
                "model": model,
                "source": args.source,
                "output": args.output,
                "topology": topology,
                "aggregate": aggregate,
                "ranks": ranks,
            }
            encoded = json.dumps(payload, indent=2, sort_keys=True)
            if args.output_json:
                Path(args.output_json).write_text(encoded + "\n")
            print(encoded, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
