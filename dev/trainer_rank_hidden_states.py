"""Equal-shape, cold/warm TrainerRank hidden-state benchmark on one GPU.

Run with the managed Megatron runtime under single-process torchrun. For example:
ART_MEGATRON_LORA_RANK=1 megatron_runtime/.venv/bin/python -m torch.distributed.run \
    --standalone --nproc-per-node=1 dev/trainer_rank_hidden_states.py \
    --tokens 16384 32768 --output /tmp/hidden-states.jsonl

Random base weights preserve the full model architecture without downloading a
checkpoint. Every mode uses the same token IDs, LoRA slot and packing geometry.
Forward-only uses no_grad; forward+backward uses a nonzero FP32 linear head for
hidden outputs, and target logprobs for target-only. The target+hidden mode also
runs a head-only loss, like Caladan 065 (its returned logprobs are unused).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import torch
import torch.distributed as dist
from trainer_rank_support import load_random_checkpoints

from art.megatron.train import build_training_runtime
from art.trainer_rank import ForwardInput, TrainerRank

MODES = ("target_hidden", "target", "hidden")


class PhaseCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("ART_TRAINER_EVENT "):
            self.events.append(json.loads(message.removeprefix("ART_TRAINER_EVENT ")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--tokens", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.warmup >= 2 and args.repeat > 0 and min(args.tokens) > 0
    torch.cuda.set_device(0)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 1, "This benchmark reserves exactly one GPU"
    torch.manual_seed(137)
    capture = PhaseCapture()
    logger = logging.getLogger("art.trainer_rank.telemetry")
    logger.setLevel(logging.INFO)
    logger.addHandler(capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with args.output.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        print(
            "HIDDEN_BENCH " + json.dumps(row, sort_keys=True, default=str), flush=True
        )

    try:
        runtime = build_training_runtime(
            model_identifier=args.model,
            model_initialization="random",
            provider_configure=(
                (lambda provider: setattr(provider, "num_layers", args.layers))
                if args.layers
                else None
            ),
            print_env=False,
            build_optimizer=False,
        )
        for chunk in runtime.model:
            chunk.train()
        rank = TrainerRank(runtime)
        [slot] = load_random_checkpoints(
            runtime, rank, 1, base_model=args.model, lora_rank=1
        )
        hidden_size = int(runtime.provider.hidden_size)
        head = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        head = (head / hidden_size).requires_grad_()
        emit(
            {
                "kind": "metadata",
                "model": args.model,
                "model_initialization": "random",
                "layers": runtime.provider.num_layers,
                "hidden_size": hidden_size,
                "lora_rank": 1,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "compiled": runtime.transformer_layers_compiled,
                "tokens": args.tokens,
                "warmup": args.warmup,
                "repeat": args.repeat,
            }
        )
        packing_by_tokens = {}
        for count in args.tokens:
            generator = torch.Generator().manual_seed(531)
            tokens = torch.randint(10, 32000, (count,), generator=generator)
            targets = tokens.roll(-1)
            targets[-1] = -100
            requests = {
                mode: ForwardInput(
                    input_tokens=tokens,
                    target_tokens=targets if mode != "hidden" else None,
                    hidden_states=mode != "target",
                    checkpoint=slot,
                )
                for mode in MODES
            }
            for backward in (False, True):
                for iteration in range(args.warmup + args.repeat):
                    # Rotate order so later variants do not systematically
                    # benefit from a warm device, clock or allocator.
                    shift = iteration % len(MODES)
                    for mode in MODES[shift:] + MODES[:shift]:
                        rank.zero_grad()
                        head.grad = None
                        capture.events.clear()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        begin = time.perf_counter()
                        forward_seconds = backward_seconds = 0.0
                        batches = 0
                        shapes = []
                        with torch.set_grad_enabled(backward):
                            iterator = rank.forward_micro_batches([requests[mode]])
                            while True:
                                before = time.perf_counter()
                                try:
                                    batch = next(iterator)
                                except StopIteration:
                                    break
                                torch.cuda.synchronize()
                                forward_seconds += time.perf_counter() - before
                                batches += 1
                                output = batch.outputs[0]
                                if output.hidden_states is not None:
                                    shapes.append(
                                        {
                                            "shape": list(output.hidden_states.shape),
                                            "dtype": str(output.hidden_states.dtype),
                                        }
                                    )
                                before = time.perf_counter()
                                if backward:
                                    if output.hidden_states is not None:
                                        loss = (
                                            (output.hidden_states.float() @ head)
                                            .square()
                                            .mean()
                                        )
                                    else:
                                        assert output.target_logprobs is not None
                                        loss = -output.target_logprobs[:-1].mean()
                                    loss.backward()
                                    del loss
                                torch.cuda.synchronize()
                                backward_seconds += time.perf_counter() - before
                                del output, batch
                        torch.cuda.synchronize()
                        total_seconds = time.perf_counter() - begin
                        phases = [
                            e for e in capture.events if e.get("event") == "phase"
                        ]
                        plan = rank.last_forward_telemetry()
                        assert batches == 1 and plan["subforward_count"] == 1
                        packing = plan["packing_plan_sha256"]
                        assert packing_by_tokens.setdefault(count, packing) == packing
                        emit(
                            {
                                "kind": "timing",
                                "mode": mode,
                                "backward": backward,
                                "tokens": count,
                                "iteration": iteration,
                                "warmup": iteration < args.warmup,
                                "forward_seconds": forward_seconds,
                                "backward_seconds": backward_seconds,
                                "total_seconds": total_seconds,
                                "batches": batches,
                                "peak_bytes": torch.cuda.max_memory_allocated(),
                                "hidden": shapes,
                                "plan": plan,
                                "phases": phases,
                            }
                        )
                        rank.zero_grad()
    finally:
        logger.removeHandler(capture)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
