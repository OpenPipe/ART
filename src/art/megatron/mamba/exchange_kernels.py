# ty: ignore[invalid-argument-type, invalid-method-override, unknown-argument]

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["tokens"])
def _pack_projected_kernel(
    source,
    output,
    tokens,
    inner: tl.constexpr,
    heads: tl.constexpr,
    groups: tl.constexpr,
    state_dim: tl.constexpr,
    cp_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rank = tl.program_id(1)
    d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    local_inner: tl.constexpr = inner // cp_size
    local_heads: tl.constexpr = heads // cp_size
    local_group_width: tl.constexpr = (
        groups // cp_size * state_dim if groups >= cp_size else state_dim
    )
    local_width: tl.constexpr = 2 * local_inner + 2 * local_group_width + local_heads
    input_width: tl.constexpr = 2 * inner + 2 * groups * state_dim + heads
    group = rank if groups >= cp_size else rank // (cp_size // groups)
    src_d = tl.where(
        d < local_inner,
        rank * local_inner + d,
        tl.where(
            d < 2 * local_inner,
            inner + rank * local_inner + d - local_inner,
            tl.where(
                d < 2 * local_inner + local_group_width,
                2 * inner + group * local_group_width + d - 2 * local_inner,
                tl.where(
                    d < 2 * local_inner + 2 * local_group_width,
                    2 * inner
                    + groups * state_dim
                    + group * local_group_width
                    + d
                    - 2 * local_inner
                    - local_group_width,
                    2 * inner
                    + 2 * groups * state_dim
                    + rank * local_heads
                    + d
                    - 2 * local_inner
                    - 2 * local_group_width,
                ),
            ),
        ),
    ).to(tl.int64)
    n64 = n.to(tl.int64)
    mask = (n[:, None] < tokens) & (d[None, :] < local_width)
    values = tl.load(
        source + n64[:, None] * input_width + src_d[None, :],
        mask=mask,
        other=0.0,
    )
    out_row = rank.to(tl.int64) * tokens + n64
    tl.store(
        output + out_row[:, None] * local_width + d[None, :].to(tl.int64),
        values,
        mask=mask,
    )


@triton.jit(do_not_specialize=["tokens"])
def _unpack_projected_grad_kernel(
    grad_output,
    grad_source,
    tokens,
    inner: tl.constexpr,
    heads: tl.constexpr,
    groups: tl.constexpr,
    state_dim: tl.constexpr,
    cp_size: tl.constexpr,
    REPLICATED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rank = tl.program_id(1)
    d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    local_inner: tl.constexpr = inner // cp_size
    local_heads: tl.constexpr = heads // cp_size
    local_group_width: tl.constexpr = (
        groups // cp_size * state_dim if groups >= cp_size else state_dim
    )
    local_width: tl.constexpr = 2 * local_inner + 2 * local_group_width + local_heads
    input_width: tl.constexpr = 2 * inner + 2 * groups * state_dim + heads
    group = rank if groups >= cp_size else rank // (cp_size // groups)
    src_d = tl.where(
        d < local_inner,
        rank * local_inner + d,
        tl.where(
            d < 2 * local_inner,
            inner + rank * local_inner + d - local_inner,
            tl.where(
                d < 2 * local_inner + local_group_width,
                2 * inner + group * local_group_width + d - 2 * local_inner,
                tl.where(
                    d < 2 * local_inner + 2 * local_group_width,
                    2 * inner
                    + groups * state_dim
                    + group * local_group_width
                    + d
                    - 2 * local_inner
                    - local_group_width,
                    2 * inner
                    + 2 * groups * state_dim
                    + rank * local_heads
                    + d
                    - 2 * local_inner
                    - 2 * local_group_width,
                ),
            ),
        ),
    ).to(tl.int64)
    n64 = n.to(tl.int64)
    mask = (n[:, None] < tokens) & (d[None, :] < local_width)
    out_row = rank.to(tl.int64) * tokens + n64
    values = tl.load(
        grad_output + out_row[:, None] * local_width + d[None, :].to(tl.int64),
        mask=mask,
        other=0.0,
    )
    destination = grad_source + n64[:, None] * input_width + src_d[None, :]
    if REPLICATED:
        tl.atomic_add(destination, values, sem="relaxed", mask=mask)
    else:
        tl.store(destination, values, mask=mask)


@triton.jit(do_not_specialize=["tokens", "flat_tokens"])
def _assemble_head_shards_kernel(
    received,
    output,
    tokens,
    flat_tokens,
    cp_size: tl.constexpr,
    components: tl.constexpr,
    local_inner: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    inner: tl.constexpr = cp_size * local_inner
    local_width: tl.constexpr = components * local_inner
    output_width: tl.constexpr = components * inner
    component = d // inner
    within = d % inner
    rank = within // local_inner
    local_d = component * local_inner + within % local_inner
    n64 = n.to(tl.int64)
    source_row = rank[None, :].to(tl.int64) * tokens + n64[:, None]
    valid = n[:, None] < tokens
    mask = (n[:, None] < flat_tokens) & (d[None, :] < output_width)
    values = tl.load(
        received + source_row * local_width + local_d[None, :].to(tl.int64),
        mask=mask & valid,
        other=0.0,
    )
    tl.store(
        output + n64[:, None] * output_width + d[None, :].to(tl.int64),
        values,
        mask=mask,
    )


@triton.jit(do_not_specialize=["tokens"])
def _disassemble_head_shards_grad_kernel(
    grad_output,
    grad_received,
    tokens,
    cp_size: tl.constexpr,
    components: tl.constexpr,
    local_inner: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rank = tl.program_id(1)
    d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    inner: tl.constexpr = cp_size * local_inner
    local_width: tl.constexpr = components * local_inner
    output_width: tl.constexpr = components * inner
    component = d // local_inner
    output_d = component * inner + rank * local_inner + d % local_inner
    n64 = n.to(tl.int64)
    mask = (n[:, None] < tokens) & (d[None, :] < local_width)
    values = tl.load(
        grad_output + n64[:, None] * output_width + output_d[None, :].to(tl.int64),
        mask=mask,
        other=0.0,
    )
    destination_row = rank.to(tl.int64) * tokens + n64
    tl.store(
        grad_received
        + destination_row[:, None] * local_width
        + d[None, :].to(tl.int64),
        values,
        mask=mask,
    )


@triton.jit(do_not_specialize=["tokens"])
def _pack_recurrent_pair_kernel(
    first,
    second,
    output,
    tokens,
    width: tl.constexpr,
    first_stride: tl.constexpr,
    second_stride: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    source_row = n.to(tl.int64)
    first_component = d[None, :] < width
    first_value = tl.load(
        first + source_row[:, None] * first_stride + d[None, :].to(tl.int64),
        mask=(n[:, None] < tokens) & first_component,
        other=0.0,
    )
    second_d = d[None, :] - width
    second_value = tl.load(
        second + source_row[:, None] * second_stride + second_d.to(tl.int64),
        mask=(n[:, None] < tokens) & ~first_component & (second_d < width),
        other=0.0,
    )
    tl.store(
        output + n[:, None].to(tl.int64) * (2 * width) + d[None, :].to(tl.int64),
        tl.where(first_component, first_value, second_value),
        mask=(n[:, None] < tokens) & (d[None, :] < 2 * width),
    )


@triton.jit(do_not_specialize=["tokens"])
def _unpack_recurrent_pair_grad_kernel(
    grad,
    grad_first,
    grad_second,
    tokens,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    destination = n.to(tl.int64)
    mask = (n[:, None] < tokens) & (d[None, :] < width)
    first_value = tl.load(
        grad + n[:, None].to(tl.int64) * (2 * width) + d[None, :].to(tl.int64),
        mask=mask,
    )
    second_value = tl.load(
        grad + n[:, None].to(tl.int64) * (2 * width) + width + d[None, :].to(tl.int64),
        mask=mask,
    )
    tl.store(
        grad_first + destination[:, None] * width + d[None, :].to(tl.int64),
        first_value,
        mask=mask,
    )
    tl.store(
        grad_second + destination[:, None] * width + d[None, :].to(tl.int64),
        second_value,
        mask=mask,
    )


_BLOCK_N = 8
_BLOCK_D = 256


class _PackProjected(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        source: torch.Tensor,
        inner: int,
        heads: int,
        groups: int,
        state_dim: int,
        cp_size: int,
    ) -> torch.Tensor:
        tokens = int(source.shape[0])
        local_groups = groups // cp_size if groups >= cp_size else 1
        local_width = (
            2 * inner // cp_size + 2 * local_groups * state_dim + heads // cp_size
        )
        output = source.new_empty((cp_size * tokens, local_width))
        _pack_projected_kernel[
            (triton.cdiv(tokens, _BLOCK_N), cp_size, triton.cdiv(local_width, _BLOCK_D))
        ](
            source,
            output,
            tokens,
            inner,
            heads,
            groups,
            state_dim,
            cp_size,
            _BLOCK_N,
            _BLOCK_D,
            num_warps=8,
        )
        ctx.geometry = (inner, heads, groups, state_dim, cp_size)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        inner, heads, groups, state_dim, cp_size = ctx.geometry
        tokens = int(grad.shape[0]) // cp_size
        input_width = 2 * inner + 2 * groups * state_dim + heads
        replicated = groups < cp_size
        output = (
            grad.new_zeros((tokens, input_width))
            if replicated
            else grad.new_empty((tokens, input_width))
        )
        local_width = int(grad.shape[1])
        _unpack_projected_grad_kernel[
            (triton.cdiv(tokens, _BLOCK_N), cp_size, triton.cdiv(local_width, _BLOCK_D))
        ](
            grad,
            output,
            tokens,
            inner,
            heads,
            groups,
            state_dim,
            cp_size,
            replicated,
            _BLOCK_N,
            _BLOCK_D,
            num_warps=8,
        )
        return output, None, None, None, None, None


class _AssembleHeadShards(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        received: torch.Tensor,
        flat_tokens: int,
        tokens: int,
        cp_size: int,
        components: int,
        local_inner: int,
    ) -> torch.Tensor:
        output_width = components * cp_size * local_inner
        output = received.new_empty((flat_tokens, output_width))
        _assemble_head_shards_kernel[
            (triton.cdiv(flat_tokens, _BLOCK_N), triton.cdiv(output_width, _BLOCK_D))
        ](
            received,
            output,
            tokens,
            flat_tokens,
            cp_size,
            components,
            local_inner,
            _BLOCK_N,
            _BLOCK_D,
            num_warps=8,
        )
        ctx.geometry = (tokens, cp_size, components, local_inner)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        tokens, cp_size, components, local_inner = ctx.geometry
        local_width = components * local_inner
        output = grad.new_empty((cp_size * tokens, local_width))
        _disassemble_head_shards_grad_kernel[
            (triton.cdiv(tokens, _BLOCK_N), cp_size, triton.cdiv(local_width, _BLOCK_D))
        ](
            grad,
            output,
            tokens,
            cp_size,
            components,
            local_inner,
            _BLOCK_N,
            _BLOCK_D,
            num_warps=8,
        )
        return output.flatten(), None, None, None, None, None


class _PackRecurrentPair(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        tokens, width = first.shape
        output = first.new_empty((tokens, 2 * width))
        _pack_recurrent_pair_kernel[
            (triton.cdiv(tokens, _BLOCK_N), triton.cdiv(2 * width, _BLOCK_D))
        ](
            first,
            second,
            output,
            tokens,
            width,
            first.stride(0),
            second.stride(0),
            _BLOCK_N,
            _BLOCK_D,
            num_warps=8,
        )
        ctx.shape = (tokens, width)
        return output

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, width = ctx.shape
        grad = grad.contiguous()
        first = grad.new_empty((tokens, width))
        second = grad.new_empty((tokens, width))
        _unpack_recurrent_pair_grad_kernel[
            (triton.cdiv(tokens, _BLOCK_N), triton.cdiv(width, _BLOCK_D))
        ](
            grad,
            first,
            second,
            tokens,
            width,
            _BLOCK_N,
            _BLOCK_D,
            num_warps=8,
        )
        return first, second


def pack_projected(
    source: torch.Tensor,
    *,
    inner: int,
    heads: int,
    groups: int,
    state_dim: int,
    cp_size: int,
) -> torch.Tensor:
    return _PackProjected.apply(source, inner, heads, groups, state_dim, cp_size)


def assemble_head_shards(
    received: torch.Tensor,
    *,
    flat_tokens: int,
    tokens: int,
    cp_size: int,
    components: int,
    local_inner: int,
) -> torch.Tensor:
    return _AssembleHeadShards.apply(
        received, flat_tokens, tokens, cp_size, components, local_inner
    )


def pack_recurrent_pair(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Mamba recurrent components must have equal 2D shapes")
    return _PackRecurrentPair.apply(first, second)
