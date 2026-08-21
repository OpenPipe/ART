from __future__ import annotations

import argparse
import gc
import statistics
import time
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from art.trainer_rank._impl import (
    _compact_optimizer_valid_ranges,
    _distributed_grad_norm,
    _OptimizerTensorValidRanges,
    _zero_optimizer_padding,
)


def _reference(
    params: tuple[torch.nn.Parameter, ...],
    grads: tuple[torch.Tensor, ...],
    masks: tuple[torch.Tensor, ...],
) -> tuple[float, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    masked = tuple(
        grad.masked_fill(mask, 0) for grad, mask in zip(grads, masks, strict=True)
    )
    squared = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for grad in masked:
        squared.add_(grad.float().square().sum())
    norm = float(torch.sqrt(squared).item())
    clip = min(1.0, 0.1 / (norm + 1.0e-6))
    return norm, masked, tuple(grad.mul(clip) for grad in masked)


def _candidate(
    params: tuple[torch.nn.Parameter, ...],
    grads: tuple[torch.Tensor, ...],
    ranges: tuple[_OptimizerTensorValidRanges, ...],
) -> tuple[float, tuple[torch.Tensor, ...]]:
    _zero_optimizer_padding(zip(grads, ranges, strict=True))
    norm = _distributed_grad_norm(params, grads)
    clip = min(1.0, 0.1 / (norm + 1.0e-6))
    torch._foreach_mul_(grads, clip)
    return norm, grads


def _median_ms(
    source: tuple[torch.Tensor, ...],
    operation: Callable[[tuple[torch.Tensor, ...]], object],
) -> float:
    samples = []
    for _ in range(3):
        owned = tuple(tensor.clone() for tensor in source)
        started = time.perf_counter()
        output = operation(owned)
        samples.append(time.perf_counter() - started)
        del output, owned
        gc.collect()
    return statistics.median(samples) * 1.0e3


def _cpu() -> None:
    torch.set_num_threads(1)
    print("cpu_threads=1")
    elements = 4096
    for count in (64, 256, 1024, 4096):
        torch.manual_seed(1)
        source = tuple(torch.randn(elements) for _ in range(count))
        masks = tuple(torch.arange(elements) >= elements - 64 for _ in range(count))
        ranges = tuple(_compact_optimizer_valid_ranges(mask) for mask in masks)
        param = torch.nn.Parameter(torch.empty(0))
        params = (param,) * count
        old_ms = _median_ms(source, lambda grads: _reference(params, grads, masks))
        new_ms = _median_ms(source, lambda grads: _candidate(params, grads, ranges))
        grad_bytes = count * elements * 4
        print(
            f"count={count:4d} grad_mib={grad_bytes / 2**20:6.1f} "
            f"old_ms={old_ms:8.3f} new_ms={new_ms:8.3f} "
            f"speedup={old_ms / new_ms:5.2f}x "
            f"old_mask_mib={grad_bytes / 4 / 2**20:6.1f} "
            f"old_grad_images_mib={2 * grad_bytes / 2**20:6.1f} "
            f"new_norm_kib={(count * 8 + 4) / 2**10:6.1f}"
        )

    count = 256
    source = tuple(torch.randn(elements) for _ in range(count))
    masks = tuple(torch.arange(elements) >= elements - 64 for _ in range(count))
    ranges = tuple(_compact_optimizer_valid_ranges(mask) for mask in masks)
    params = (torch.nn.Parameter(torch.empty(0)),) * count
    profiles = {}
    for name, operation in (
        ("old", lambda grads: _reference(params, grads, masks)),
        ("new", lambda grads: _candidate(params, grads, ranges)),
    ):
        grads = tuple(tensor.clone() for tensor in source)
        with profile(activities=[ProfilerActivity.CPU]) as profiler:
            output = operation(grads)
        profiles[name] = {row.key: row.count for row in profiler.key_averages()}
        del output, grads
    keys = (
        "aten::masked_fill",
        "aten::square",
        "aten::sum",
        "aten::add_",
        "aten::mul",
        "aten::_foreach_zero_",
        "aten::_foreach_norm",
        "aten::dot",
        "aten::_foreach_mul_",
    )
    for name, counts in profiles.items():
        print(name, " ".join(f"{key}={counts.get(key, 0)}" for key in keys))


def _cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; kernel probe has no CPU fallback.")
    device = torch.device("cuda")
    count, elements = 512, 65536
    source = tuple(torch.randn(elements, device=device) for _ in range(count))
    masks = tuple(
        torch.arange(elements, device=device) >= elements - 256 for _ in range(count)
    )
    ranges = tuple(_compact_optimizer_valid_ranges(mask) for mask in masks)
    params = (torch.nn.Parameter(torch.empty(0, device=device)),) * count
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    for name, operation in (
        ("old", lambda grads: _reference(params, grads, masks)),
        ("new", lambda grads: _candidate(params, grads, ranges)),
    ):
        grads = tuple(tensor.clone() for tensor in source)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with profile(activities=activities) as profiler:
            output = operation(grads)
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - baseline
        kernels = sum(
            event.device_type == torch.autograd.DeviceType.CUDA
            for event in profiler.events()
        )
        print(
            f"{name} count={count} grad_mib={count * elements * 4 / 2**20:.1f} "
            f"incremental_peak_mib={peak / 2**20:.3f} cuda_kernels={kernels}"
        )
        del output, grads


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    _cuda() if args.cuda else _cpu()
