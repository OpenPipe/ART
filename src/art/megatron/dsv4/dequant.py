from __future__ import annotations

import torch
import triton
import triton.language as tl

_DSV4_FP4_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_TABLE_CACHE: dict[int, torch.Tensor] = {}


@triton.jit
def _mxfp4_dequant_kernel(
    weight_ptr,
    scale_ptr,
    table_ptr,
    out_ptr,
    total: tl.constexpr,
    in_dim: tl.constexpr,
    in_bytes: tl.constexpr,
    scale_cols: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < total
    col = offsets % in_dim
    row = offsets // in_dim
    packed = tl.load(weight_ptr + row * in_bytes + col // 2, mask=mask, other=0)
    nibble = tl.where((col & 1) == 0, packed & 0x0F, (packed >> 4) & 0x0F)
    fp4 = tl.load(table_ptr + nibble)
    raw_scale = tl.load(scale_ptr + row * scale_cols + col // 32, mask=mask, other=127)
    scale = tl.where(
        raw_scale == 255,
        float("nan"),
        tl.exp2(raw_scale.to(tl.float32) - 127.0),
    )
    tl.store(out_ptr + offsets, (fp4 * scale).to(tl.bfloat16), mask=mask)


def _fp4_table(device: torch.device) -> torch.Tensor:
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    table = _TABLE_CACHE.get(index)
    if table is None:
        table = torch.tensor(_DSV4_FP4_TABLE, dtype=torch.float32, device=device)
        _TABLE_CACHE[index] = table
    return table


def dequant_mxfp4_cuda(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    device = (
        weight.device
        if weight.device.type == "cuda"
        else torch.device("cuda", torch.cuda.current_device())
    )
    weight = weight.contiguous().to(device=device, non_blocking=True).view(torch.uint8)
    scale = scale.contiguous().to(device=device, non_blocking=True).view(torch.uint8)
    out_dim, in_bytes = weight.shape
    in_dim = in_bytes * 2
    out = torch.empty((out_dim, in_dim), dtype=torch.bfloat16, device=device)
    block = 256
    _mxfp4_dequant_kernel[(triton.cdiv(out.numel(), block),)](
        weight,
        scale,
        _fp4_table(device),
        out,
        out.numel(),
        in_dim,
        in_bytes,
        scale.shape[1],
        block,
        num_warps=4,
    )
    return out


@triton.jit
def _block_fp8_dequant_kernel(
    weight_ptr,
    scale_ptr,
    out_ptr,
    total: tl.constexpr,
    n_cols: tl.constexpr,
    scale_cols: tl.constexpr,
    block_elems: tl.constexpr,
):
    offsets = tl.program_id(0) * block_elems + tl.arange(0, block_elems)
    mask = offsets < total
    col = offsets % n_cols
    row = offsets // n_cols
    value = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    raw_scale = tl.load(
        scale_ptr + (row // 128) * scale_cols + col // 128,
        mask=mask,
        other=127,
    )
    scale = tl.where(
        raw_scale == 255,
        float("nan"),
        tl.exp2(raw_scale.to(tl.float32) - 127.0),
    )
    tl.store(out_ptr + offsets, (value * scale).to(tl.bfloat16), mask=mask)


def dequant_block_fp8_cuda(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    device = (
        weight.device
        if weight.device.type == "cuda"
        else torch.device("cuda", torch.cuda.current_device())
    )
    weight = weight.contiguous().to(device=device, non_blocking=True)
    scale = scale.contiguous().to(device=device, non_blocking=True).view(torch.uint8)
    out = torch.empty_like(weight, dtype=torch.bfloat16)
    block_elems = 256
    _block_fp8_dequant_kernel[(triton.cdiv(out.numel(), block_elems),)](
        weight,
        scale,
        out,
        out.numel(),
        weight.shape[1],
        scale.shape[1],
        block_elems,
        num_warps=4,
    )
    return out
