"""Fixed scheduling; descriptors are included in the training invocation cost."""

from typing import Any, cast

import torch


def configuration(n, k, dtype):
    # Frozen from N/K/dtype only, never from M or expert occupancy. GPU9 does
    # not tune this table on correctness cells. Autotuning is a later proposal.
    if min(n, k) <= 32:
        return 32, 32, 32
    return 128, 128, 64 if dtype != torch.float32 else 32


def persistent(a, b):
    if a.device.type != "cuda":
        from .ops import _cpu_mm

        return _cpu_mm(a, b)
    from .fast_kernels import persistent_mm

    m, k = a.shape
    n = b.shape[1]
    bm, bn, bk = configuration(n, k, a.dtype)
    out = a.new_empty((m, n))
    if out.numel():
        sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        # Triton's launch object accepts compiler kwargs absent from its Python signature.
        cast(Any, persistent_mm)[
            (min(sms, ((m + bm - 1) // bm) * ((n + bn - 1) // bn)),)
        ](
            a,
            b,
            out,
            m,
            n,
            k,
            *a.stride(),
            *b.stride(),
            sms,
            bm,
            bn,
            bk,
            "tf32x3" if a.dtype == torch.float32 and min(n, k) > 32 else "ieee",
            FP32_OPERANDS=min(n, k) <= 32,
            num_warps=4 if bn == 32 else 8,
            num_stages=3,
        )
    return out


def group_metadata(counts, device):
    offsets, experts, rows = [0], [], []
    for e, c in enumerate(counts):
        offsets.append(offsets[-1] + c)
        for row in range(0, c, 32):
            experts.append(e)
            rows.append(row)
    return (
        torch.tensor(offsets, device=device, dtype=torch.int32),
        torch.tensor(experts, device=device, dtype=torch.int32),
        torch.tensor(rows, device=device, dtype=torch.int32),
        len(rows),
    )


def grouped(a, b, counts, metadata=None):
    """BF16 forward/dgrad; counts are CPU metadata, b is [E,K,N]."""
    if len(counts) != len(b) or sum(counts) != len(a) or any(c < 0 for c in counts):
        raise ValueError("expert counts must cover the input")
    if a.device.type != "cuda":
        from .ops import _cpu_mm

        return torch.cat(
            [_cpu_mm(x, w) for x, w in zip(a.split(counts), b, strict=True)]
        )
    if a.dtype != torch.bfloat16 or b.dtype != a.dtype:
        raise ValueError("GPU9 grouped candidate requires BF16")
    from .fast_kernels import grouped_mm

    out = a.new_empty((len(a), b.shape[-1]))
    if out.numel():
        offsets, tile_experts, tile_rows, tiles = (
            group_metadata(counts, a.device) if metadata is None else metadata
        )
        cast(Any, grouped_mm)[(tiles, (b.shape[-1] + 63) // 64)](
            a.contiguous(),
            b,
            out,
            offsets,
            tile_experts,
            tile_rows,
            len(counts),
            b.shape[-1],
            a.shape[-1],
            b.stride(1),
            b.stride(2),
            b.stride(0),
            BM=32,
            NT=64,
            KT=32,
            num_warps=4,
            num_stages=3,
        )
    return out


def experts(a, weights, counts):
    if (
        len(weights) != len(counts)
        or sum(counts) != len(a)
        or any(c < 0 for c in counts)
    ):
        raise ValueError("expert counts must cover the input")
    if a.device.type != "cuda" or a.dtype == torch.float64:
        from .ops import _cpu_mm

        return torch.cat(
            [_cpu_mm(x, w) for x, w in zip(a.split(counts), weights, strict=True)]
        )
    from .fast_kernels import expert_mm

    strides = weights[0].stride()
    if any(
        w.shape != weights[0].shape
        or w.stride() != strides
        or w.dtype != a.dtype
        or w.device != a.device
        for w in weights
    ):
        raise ValueError(
            "expert weights require matching shape, stride, dtype and device"
        )
    out = a.new_empty((len(a), weights[0].shape[1]))
    if out.numel():
        offsets, ids, rows, tiles = group_metadata(counts, a.device)
        pointers = torch.tensor(
            [w.data_ptr() for w in weights], dtype=torch.int64, device=a.device
        )
        cast(Any, expert_mm)[(tiles, (out.shape[1] + 63) // 64)](
            a.contiguous(),
            pointers,
            out,
            offsets,
            ids,
            rows,
            out.shape[1],
            a.shape[1],
            *strides,
            num_warps=4,
            num_stages=3,
        )
    return out
