from __future__ import annotations

import argparse
import gc
import json
import statistics
from typing import Any

import torch

from art.megatron.kernels.cute_grouped_lora_quack import (
    quack_grouped_lora,
    quack_grouped_lora_shared_outer,
)

COUNTS = [1024, 896, 768, 640, 512, 384, 256, 128]
IN_FEATURES = 2048
OUT_FEATURES = 4096
RANK = 32
SCALE = 1.25
MEMORY_LIMIT_BYTES = 4 * 1024**3


def _leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().requires_grad_(True)


def _data(mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260819)
    tokens = sum(COUNTS)
    experts = len(COUNTS)
    x = torch.rand(
        tokens,
        IN_FEATURES,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    if mode == "fc1":
        a_t = torch.rand(
            IN_FEATURES,
            RANK,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.02)
        b_t = torch.rand(
            experts,
            RANK,
            OUT_FEATURES,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.02)
    else:
        a_t = torch.rand(
            experts,
            IN_FEATURES,
            RANK,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.02)
        b_t = torch.rand(
            RANK,
            OUT_FEATURES,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.02)
    grad_out = torch.rand(
        tokens,
        OUT_FEATURES,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    return x, a_t, b_t, grad_out


def _mean_abs_pct(candidate: torch.Tensor, target: torch.Tensor) -> float:
    candidate = candidate.detach().float()
    target = target.detach().float()
    nonzero = target != 0
    if not torch.equal(candidate[~nonzero], target[~nonzero]):
        return float("inf")
    return float(
        ((candidate[nonzero] - target[nonzero]).abs() / target[nonzero].abs()).mean()
        * 100
    )


def _correctness(mode: str) -> dict[str, float]:
    x, a_t, b_t, grad_out = _data(mode)
    x_shared, a_shared, b_shared = map(_leaf, (x.clone(), a_t.clone(), b_t.clone()))
    x_expert = _leaf(x.clone())
    if mode == "fc1":
        a_expert = _leaf(a_t.unsqueeze(0).repeat(len(COUNTS), 1, 1))
        b_expert = _leaf(b_t.clone())
    else:
        a_expert = _leaf(a_t.clone())
        b_expert = _leaf(b_t.unsqueeze(0).repeat(len(COUNTS), 1, 1))

    shared_out = quack_grouped_lora_shared_outer(
        x_shared, a_shared, b_shared, COUNTS, scale=SCALE
    )
    expert_out = quack_grouped_lora(x_expert, a_expert, b_expert, COUNTS, scale=SCALE)
    shared_out.backward(grad_out)
    expert_out.backward(grad_out)

    if mode == "fc1":
        a_expert_grad = a_expert.grad.sum(dim=0)
        b_expert_grad = b_expert.grad
    else:
        a_expert_grad = a_expert.grad
        b_expert_grad = b_expert.grad.sum(dim=0)
    pairs = {
        "out": (shared_out, expert_out),
        "x_grad": (x_shared.grad, x_expert.grad),
        "a_grad": (a_shared.grad, a_expert_grad),
        "b_grad": (b_shared.grad, b_expert_grad),
    }
    errors: dict[str, float] = {}
    for name, (candidate, target) in pairs.items():
        assert candidate is not None and target is not None
        errors[name] = _mean_abs_pct(candidate, target)
        limit = 3.0 if name == "out" else 5.0
        if errors[name] > limit:
            raise AssertionError(f"{mode} {name} mean_abs_pct={errors[name]:.6f}%")
    return errors


def _variant(
    mode: str, baseline: bool
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
]:
    x, a_t, b_t, grad_out = _data(mode)
    if baseline and mode == "fc1":
        a_t = a_t.unsqueeze(0).repeat(len(COUNTS), 1, 1)
    elif baseline:
        b_t = b_t.unsqueeze(0).repeat(len(COUNTS), 1, 1)
    x, a_t, b_t = map(_leaf, (x, a_t, b_t))
    return x, a_t, b_t, grad_out, [x, a_t, b_t]


def _benchmark(mode: str, baseline: bool, warmup: int, iters: int) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    x, a_t, b_t, grad_out, leaves = _variant(mode, baseline)
    fn = quack_grouped_lora if baseline else quack_grouped_lora_shared_outer

    def step() -> None:
        for leaf in leaves:
            leaf.grad = None
        fn(x, a_t, b_t, COUNTS, scale=SCALE).backward(grad_out)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    for leaf in leaves:
        leaf.grad = None
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    end.synchronize()
    peak = torch.cuda.max_memory_allocated()
    if peak >= MEMORY_LIMIT_BYTES:
        raise AssertionError(f"peak allocation {peak / 1024**3:.3f} GiB exceeds 4 GiB")
    return {
        "fwd_bwd_ms": start.elapsed_time(end) / iters,
        "allocated_before_mib": allocated_before / 1024**2,
        "peak_allocated_mib": peak / 1024**2,
        "incremental_peak_mib": (peak - allocated_before) / 1024**2,
        "factor_storage_mib": (a_t.nbytes + b_t.nbytes) / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    result: dict[str, Any] = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "bfloat16",
        },
        "shape": {
            "experts": len(COUNTS),
            "counts": COUNTS,
            "tokens": sum(COUNTS),
            "in_features": IN_FEATURES,
            "rank": RANK,
            "out_features": OUT_FEATURES,
        },
        "correctness_vs_per_expert_quack": {},
        "benchmarks": {},
    }
    modes = ("fc1", "fc2")
    cases = tuple((mode, baseline) for mode in modes for baseline in (True, False))
    for mode in modes:
        result["correctness_vs_per_expert_quack"][mode] = _correctness(mode)
    samples: dict[str, list[dict[str, Any]]] = {
        f"{mode}_{'per_expert' if baseline else 'shared_outer'}": []
        for mode, baseline in cases
    }
    for repeat in range(args.repeats):
        ordered_cases = cases if repeat % 2 == 0 else tuple(reversed(cases))
        for mode, baseline in ordered_cases:
            label = f"{mode}_{'per_expert' if baseline else 'shared_outer'}"
            samples[label].append(_benchmark(mode, baseline, args.warmup, args.iters))
    for label, runs in samples.items():
        timings = [run["fwd_bwd_ms"] for run in runs]
        result["benchmarks"][label] = {
            "fwd_bwd_ms_median": statistics.median(timings),
            "fwd_bwd_ms_samples": timings,
            "peak_allocated_mib_max": max(run["peak_allocated_mib"] for run in runs),
            "incremental_peak_mib_max": max(
                run["incremental_peak_mib"] for run in runs
            ),
            "factor_storage_mib": runs[0]["factor_storage_mib"],
        }
    for mode in modes:
        baseline = result["benchmarks"][f"{mode}_per_expert"]["fwd_bwd_ms_median"]
        shared = result["benchmarks"][f"{mode}_shared_outer"]["fwd_bwd_ms_median"]
        result["benchmarks"][f"{mode}_speedup"] = baseline / shared
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
