"""Exercise canonical TrainerRank checkpoints under ``torchrun``.

The driver runs this module repeatedly with different rank counts. ``step-save``
loads a LoRA/checkpoint, applies one deterministic optimizer step, and saves a
canonical checkpoint. ``step-export`` applies the same step and exports LoRA
weights for cross-topology comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from art.trainer_rank import AdamParams, TrainerRank


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("step-save", "step-export", "load"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--grad", type=float, default=1e-4)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(path).as_posix().encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


async def _load(trainer: TrainerRank, source: str) -> None:
    await trainer.load_checkpoint(source)


def main() -> None:
    args = _args()
    os.environ.setdefault("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_CONTEXT_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", "1")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    started = time.perf_counter()
    try:
        from art.megatron import train as megatron_train

        runtime = megatron_train.build_training_runtime(
            model_identifier=args.model,
            provider_configure=lambda provider: setattr(
                provider, "num_layers", args.layers
            ),
            print_env=rank == 0,
        )
        trainer = TrainerRank(runtime)
        asyncio.run(_load(trainer, args.source))
        loaded = time.perf_counter()
        if args.operation != "load":
            slot = trainer._checkpoint_slots[args.source]
            for parameter in slot.params:
                parameter.grad = torch.full_like(parameter, args.grad)
            metrics = trainer.optim_step(
                params=AdamParams(learning_rate=3e-4, grad_clip_norm=0),
                scale_grads=1 / dist.get_world_size(),
            )
            if metrics["update_successful"] != 1:
                raise RuntimeError(f"optimizer step failed: {metrics}")
            if args.output is None:
                raise ValueError("--output is required")
            if args.operation == "step-save":
                trainer.save_checkpoint(args.output)
            else:
                trainer.export_lora(args.output)
        dist.barrier()
        if rank == 0:
            output = None if args.output is None else Path(args.output)
            payload = {
                "world_size": dist.get_world_size(),
                "load_seconds": loaded - started,
                "total_seconds": time.perf_counter() - started,
                "output_digest": None if output is None else _digest(output),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            }
            encoded = json.dumps(payload, sort_keys=True)
            print(encoded, flush=True)
            if args.output_json:
                Path(args.output_json).write_text(encoded + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
