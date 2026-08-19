import argparse
import json
from statistics import median
import time

import torch

from art.megatron.tensor_snapshot import (
    PendingCpuSnapshot,
    PinnedCpuSnapshotStager,
)
from art.megatron.weights.lora_publish import _stage_published_tensors

_GIB = 1024**3


def _stage_with_concat(
    tensors: dict[str, torch.Tensor], stager: PinnedCpuSnapshotStager
) -> PendingCpuSnapshot[dict[str, torch.Tensor]]:
    builder = stager.begin()
    ordered = sorted(tensors.items())
    flat = torch.cat([tensor.detach().contiguous().view(-1) for _, tensor in ordered])
    staged_flat = builder.stage(flat)
    staged: dict[str, torch.Tensor] = {}
    offset = 0
    for key, tensor in ordered:
        staged[key] = staged_flat.narrow(0, offset, tensor.numel()).view(tensor.shape)
        offset += tensor.numel()
    return builder.finish(staged)


def _stage_grouped(
    tensors: dict[str, torch.Tensor], stager: PinnedCpuSnapshotStager
) -> PendingCpuSnapshot[dict[str, torch.Tensor]]:
    builder = stager.begin()
    return builder.finish(_stage_published_tensors(tensors, builder))


def _measure(
    tensors: dict[str, torch.Tensor],
    stager: PinnedCpuSnapshotStager,
    stage,
    iterations: int,
) -> dict[str, float | int]:
    device = next(iter(tensors.values())).device
    elapsed: list[float] = []
    peak_extra: list[int] = []
    result: dict[str, torch.Tensor] = {}
    for _ in range(iterations):
        stager.reset()
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        pending = stage(tensors, stager)
        result = pending.resolve()
        elapsed.append(time.perf_counter() - started)
        peak_extra.append(torch.cuda.max_memory_allocated(device) - allocated)
    for index, tensor in enumerate(result.values()):
        assert tensor.flatten()[0].item() == index
    logical_bytes = sum(tensor.nbytes for tensor in tensors.values())
    median_s = median(elapsed)
    return {
        "median_s": median_s,
        "throughput_gib_s": logical_bytes / _GIB / median_s,
        "max_cuda_peak_extra_bytes": max(peak_extra),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-mib", type=int, default=512)
    parser.add_argument("--tensors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    logical_bytes = args.total_mib * 1024**2
    estimated_peak_bytes = 4 * logical_bytes
    if estimated_peak_bytes > 4 * _GIB:
        raise ValueError("estimated source, temp, and pinned memory exceeds 4 GiB")
    element_size = torch.empty((), dtype=torch.bfloat16).element_size()
    elements = logical_bytes // args.tensors // element_size
    device = torch.device("cuda", 0)
    tensors = {
        f"tensor_{index:04d}": torch.full(
            (elements,), index, dtype=torch.bfloat16, device=device
        )
        for index in range(args.tensors)
    }
    logical_bytes = sum(tensor.nbytes for tensor in tensors.values())
    concat_stager = PinnedCpuSnapshotStager(reusable=True)
    grouped_stager = PinnedCpuSnapshotStager(reusable=True)
    _measure(tensors, concat_stager, _stage_with_concat, 2)
    _measure(tensors, grouped_stager, _stage_grouped, 2)
    concat = _measure(tensors, concat_stager, _stage_with_concat, args.iterations)
    grouped = _measure(tensors, grouped_stager, _stage_grouped, args.iterations)
    if concat["max_cuda_peak_extra_bytes"] < logical_bytes:
        raise RuntimeError("concat baseline did not expose its aggregate CUDA tensor")
    if grouped["max_cuda_peak_extra_bytes"] != 0:
        raise RuntimeError("grouped staging allocated an aggregate CUDA tensor")
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "dtype": "bfloat16",
                "tensor_count": args.tensors,
                "logical_bytes": logical_bytes,
                "estimated_peak_bytes": estimated_peak_bytes,
                "concat": concat,
                "grouped": grouped,
                "speedup": concat["median_s"] / grouped["median_s"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
