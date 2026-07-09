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
        out.numel(),  # ty: ignore[invalid-argument-type]
        in_dim,  # ty: ignore[invalid-argument-type]
        in_bytes,  # ty: ignore[invalid-argument-type]
        scale.shape[1],  # ty: ignore[invalid-argument-type]
        block,  # ty: ignore[invalid-argument-type]
        num_warps=4,  # ty: ignore[unknown-argument]
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
        out.numel(),  # ty: ignore[invalid-argument-type]
        weight.shape[1],  # ty: ignore[invalid-argument-type]
        scale.shape[1],  # ty: ignore[invalid-argument-type]
        block_elems,  # ty: ignore[invalid-argument-type]
        num_warps=4,  # ty: ignore[unknown-argument]
    )
    return out


@triton.jit
def _mxfp4_quant_kernel(
    weight_ptr,
    packed_ptr,
    scale_ptr,
    total_blocks: tl.constexpr,
    cols: tl.constexpr,
    blocks_per_row: tl.constexpr,
):
    block_id = tl.program_id(0)
    pair_offsets = tl.arange(0, 16)
    row = block_id // blocks_per_row
    block_col = block_id - row * blocks_per_row
    even = tl.load(weight_ptr + row * cols + block_col * 32 + pair_offsets * 2).to(
        tl.float32
    )
    odd = tl.load(weight_ptr + row * cols + block_col * 32 + pair_offsets * 2 + 1).to(
        tl.float32
    )
    amax = tl.maximum(
        tl.maximum(tl.max(tl.abs(even), axis=0), tl.max(tl.abs(odd), axis=0)),
        1.0e-4,
    )
    exponent = tl.ceil(tl.log2(amax / 6.0))
    raw_scale = (exponent + 127.0).to(tl.uint8)
    scale = tl.exp2(exponent)
    scaled_even = even / scale
    scaled_odd = odd / scale
    abs_even = tl.abs(scaled_even)
    abs_odd = tl.abs(scaled_odd)
    code_even = tl.zeros((16,), dtype=tl.uint8)
    code_odd = tl.zeros((16,), dtype=tl.uint8)
    code_even += (abs_even > 0.25).to(tl.uint8)
    code_even += (abs_even > 0.75).to(tl.uint8)
    code_even += (abs_even > 1.25).to(tl.uint8)
    code_even += (abs_even > 1.75).to(tl.uint8)
    code_even += (abs_even > 2.5).to(tl.uint8)
    code_even += (abs_even > 3.5).to(tl.uint8)
    code_even += (abs_even > 5.0).to(tl.uint8)
    code_even += ((scaled_even < 0.0) & (code_even != 0)).to(tl.uint8) * 8
    code_odd += (abs_odd > 0.25).to(tl.uint8)
    code_odd += (abs_odd > 0.75).to(tl.uint8)
    code_odd += (abs_odd > 1.25).to(tl.uint8)
    code_odd += (abs_odd > 1.75).to(tl.uint8)
    code_odd += (abs_odd > 2.5).to(tl.uint8)
    code_odd += (abs_odd > 3.5).to(tl.uint8)
    code_odd += (abs_odd > 5.0).to(tl.uint8)
    code_odd += ((scaled_odd < 0.0) & (code_odd != 0)).to(tl.uint8) * 8
    tl.store(
        packed_ptr + row * (cols // 2) + block_col * 16 + pair_offsets,
        code_even | (code_odd << 4),
    )
    tl.store(scale_ptr + block_id, raw_scale, mask=block_id < total_blocks)


def quant_mxfp4_cuda(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.shape[1] % 32 != 0:
        raise ValueError(
            f"Expected 2-D MXFP4 weight with K % 32 == 0, got {weight.shape}."
        )
    if weight.device.type != "cuda":
        raise ValueError("quant_mxfp4_cuda expects a CUDA tensor")
    weight = weight.contiguous()
    rows, cols = weight.shape
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scale_raw = torch.empty((rows, cols // 32), dtype=torch.uint8, device=weight.device)
    blocks_per_row = cols // 32
    total_blocks = rows * blocks_per_row
    _mxfp4_quant_kernel[(total_blocks,)](
        weight,
        packed,
        scale_raw,
        total_blocks,  # ty: ignore[invalid-argument-type]
        cols,  # ty: ignore[invalid-argument-type]
        blocks_per_row,  # ty: ignore[invalid-argument-type]
        num_warps=1,  # ty: ignore[unknown-argument]
    )
    return packed, scale_raw.view(torch.float8_e8m0fnu)
