from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["token_count", "sequence_length"])
def _gather_compact_qkv_kernel(
    qkv_flat,
    row_indices,
    position_indices,
    out,
    token_count,
    sequence_length,
    channels: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = n < token_count
    row = tl.load(row_indices + n, mask=token_mask, other=0)
    pos = tl.load(position_indices + n, mask=token_mask, other=0)
    src = row * sequence_length + pos
    n64 = n.to(tl.int64)
    d64 = d.to(tl.int64)
    src64 = src.to(tl.int64)
    mask = token_mask[:, None] & (d[None, :] < channels)
    values = tl.load(
        qkv_flat + src64[:, None] * channels + d64[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(out + n64[:, None] * channels + d64[None, :], values, mask=mask)


@triton.jit(do_not_specialize=["token_count", "sequence_length"])
def _gather_compact_aux_kernel(
    x_flat,
    row_indices,
    position_indices,
    out,
    token_count,
    sequence_length,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = n < token_count
    row = tl.load(row_indices + n, mask=token_mask, other=0)
    pos = tl.load(position_indices + n, mask=token_mask, other=0)
    src = row * sequence_length + pos
    n64 = n.to(tl.int64)
    d64 = d.to(tl.int64)
    src64 = src.to(tl.int64)
    mask = token_mask[:, None] & (d[None, :] < width)
    values = tl.load(
        x_flat + src64[:, None] * width + d64[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(out + n64[:, None] * width + d64[None, :], values, mask=mask)


@triton.jit(do_not_specialize=["token_count", "sequence_length"])
def _scatter_compact_qkv_grad_kernel(
    grad_out,
    row_indices,
    position_indices,
    grad_flat,
    token_count,
    sequence_length,
    channels: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = n < token_count
    row = tl.load(row_indices + n, mask=token_mask, other=0)
    pos = tl.load(position_indices + n, mask=token_mask, other=0)
    dst = row * sequence_length + pos
    n64 = n.to(tl.int64)
    d64 = d.to(tl.int64)
    dst64 = dst.to(tl.int64)
    mask = token_mask[:, None] & (d[None, :] < channels)
    values = tl.load(
        grad_out + n64[:, None] * channels + d64[None, :],
        mask=mask,
        other=0.0,
    )
    tl.atomic_add(
        grad_flat + dst64[:, None] * channels + d64[None, :],
        values,
        sem="relaxed",
        mask=mask,
    )


@triton.jit(do_not_specialize=["token_count", "sequence_length"])
def _scatter_compact_aux_grad_kernel(
    grad_out,
    row_indices,
    position_indices,
    grad_flat,
    token_count,
    sequence_length,
    width: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = n < token_count
    row = tl.load(row_indices + n, mask=token_mask, other=0)
    pos = tl.load(position_indices + n, mask=token_mask, other=0)
    dst = row * sequence_length + pos
    n64 = n.to(tl.int64)
    d64 = d.to(tl.int64)
    dst64 = dst.to(tl.int64)
    mask = token_mask[:, None] & (d[None, :] < width)
    values = tl.load(
        grad_out + n64[:, None] * width + d64[None, :],
        mask=mask,
        other=0.0,
    )
    tl.atomic_add(
        grad_flat + dst64[:, None] * width + d64[None, :],
        values,
        sem="relaxed",
        mask=mask,
    )


@triton.jit(do_not_specialize=["token_count"])
def _prepare_packed_qkv_kernel(
    qkv,
    query,
    key,
    value,
    token_count,
    channels: tl.constexpr,
    key_heads: tl.constexpr,
    value_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    repeat: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    vh = tl.program_id(1)
    kind = tl.program_id(2)
    d = tl.arange(0, BLOCK_D)
    token_mask = n < token_count
    n64 = n.to(tl.int64)
    d64 = d.to(tl.int64)
    kh = vh // repeat
    if kind == 0:
        mask = d < key_dim
        channel = kh * key_dim + d
        channel64 = channel.to(tl.int64)
        values = tl.load(
            qkv + n64[:, None] * channels + channel64[None, :],
            mask=token_mask[:, None] & mask[None, :],
            other=0.0,
        )
        tl.store(
            query + (n64[:, None] * value_heads + vh) * key_dim + d64[None, :],
            values,
            mask=token_mask[:, None] & mask[None, :],
        )
    elif kind == 1:
        mask = d < key_dim
        base = key_heads * key_dim
        channel = base + kh * key_dim + d
        channel64 = channel.to(tl.int64)
        values = tl.load(
            qkv + n64[:, None] * channels + channel64[None, :],
            mask=token_mask[:, None] & mask[None, :],
            other=0.0,
        )
        tl.store(
            key + (n64[:, None] * value_heads + vh) * key_dim + d64[None, :],
            values,
            mask=token_mask[:, None] & mask[None, :],
        )
    else:
        mask = d < value_dim
        base = 2 * key_heads * key_dim
        channel = base + vh * value_dim + d
        channel64 = channel.to(tl.int64)
        values = tl.load(
            qkv + n64[:, None] * channels + channel64[None, :],
            mask=token_mask[:, None] & mask[None, :],
            other=0.0,
        )
        tl.store(
            value + (n64[:, None] * value_heads + vh) * value_dim + d64[None, :],
            values,
            mask=token_mask[:, None] & mask[None, :],
        )


@triton.jit(do_not_specialize=["token_count"])
def _prepare_packed_qkv_backward_kernel(
    grad_query,
    grad_key,
    grad_value,
    grad_qkv,
    token_count,
    channels: tl.constexpr,
    key_heads: tl.constexpr,
    value_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    repeat: tl.constexpr,
    HAS_QUERY: tl.constexpr,
    HAS_KEY: tl.constexpr,
    HAS_VALUE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    q_channels: tl.constexpr = key_heads * key_dim
    k_channels: tl.constexpr = q_channels
    v_base: tl.constexpr = q_channels + k_channels
    n64 = n.to(tl.int64)
    c64 = c.to(tl.int64)
    mask = (n[:, None] < token_count) & (c[None, :] < channels)
    is_query = c < q_channels
    is_key = (c >= q_channels) & (c < v_base)
    is_value = c >= v_base
    values = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)

    if HAS_QUERY:
        q_kh = c // key_dim
        q_d = c - q_kh * key_dim
        q_mask = mask & is_query[None, :]
        q_values = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
        for r in tl.static_range(0, repeat):
            vh = q_kh * repeat + r
            q_values += tl.load(
                grad_query
                + (n64[:, None] * value_heads + vh[None, :].to(tl.int64)) * key_dim
                + q_d[None, :],
                mask=q_mask,
                other=0.0,
            )
        values = tl.where(q_mask, q_values, values)

    if HAS_KEY:
        k_channel = c - q_channels
        k_kh = k_channel // key_dim
        k_d = k_channel - k_kh * key_dim
        k_mask = mask & is_key[None, :]
        k_values = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
        for r in tl.static_range(0, repeat):
            vh = k_kh * repeat + r
            k_values += tl.load(
                grad_key
                + (n64[:, None] * value_heads + vh[None, :].to(tl.int64)) * key_dim
                + k_d[None, :],
                mask=k_mask,
                other=0.0,
            )
        values = tl.where(k_mask, k_values, values)

    if HAS_VALUE:
        v_channel = c - v_base
        vh = v_channel // value_dim
        v_d = v_channel - vh * value_dim
        v_mask = mask & is_value[None, :]
        v_values = tl.load(
            grad_value
            + (n64[:, None] * value_heads + vh[None, :].to(tl.int64)) * value_dim
            + v_d[None, :],
            mask=v_mask,
            other=0.0,
        )
        values = tl.where(v_mask, v_values, values)

    tl.store(grad_qkv + n64[:, None] * channels + c64[None, :], values, mask=mask)


@triton.jit(do_not_specialize=["token_count", "output_sequence_length"])
def _scatter_bucket_output_compact_forward_kernel(
    output,
    bucket_output,
    row_indices,
    position_indices,
    output_mask,
    token_count,
    output_sequence_length,
    heads: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    hd = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = n < token_count
    write = tl.load(output_mask + n, mask=token_mask, other=0).to(tl.int1)
    row = tl.load(row_indices + n, mask=token_mask, other=0)
    pos = tl.load(position_indices + n, mask=token_mask, other=0)
    h = hd // dim
    d = hd - h * dim
    n64 = n.to(tl.int64)
    row64 = row.to(tl.int64)
    pos64 = pos.to(tl.int64)
    h64 = h.to(tl.int64)
    d64 = d.to(tl.int64)
    mask = token_mask[:, None] & (hd[None, :] < heads * dim) & write[:, None]
    values = tl.load(
        bucket_output + (n64[:, None] * heads + h64[None, :]) * dim + d64[None, :],
        mask=mask,
        other=0.0,
    )
    tl.store(
        output
        + ((row64[:, None] * output_sequence_length + pos64[:, None]) * heads + h64)
        * dim
        + d64,
        values,
        mask=mask,
    )


@triton.jit(do_not_specialize=["token_count", "output_sequence_length"])
def _scatter_bucket_output_compact_backward_kernel(
    grad_output,
    grad_base,
    grad_bucket_output,
    row_indices,
    position_indices,
    output_mask,
    token_count,
    output_sequence_length,
    heads: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    hd = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = n < token_count
    write = tl.load(output_mask + n, mask=token_mask, other=0).to(tl.int1)
    row = tl.load(row_indices + n, mask=token_mask, other=0)
    pos = tl.load(position_indices + n, mask=token_mask, other=0)
    h = hd // dim
    d = hd - h * dim
    n64 = n.to(tl.int64)
    row64 = row.to(tl.int64)
    pos64 = pos.to(tl.int64)
    h64 = h.to(tl.int64)
    d64 = d.to(tl.int64)
    mask = token_mask[:, None] & (hd[None, :] < heads * dim) & write[:, None]
    output_offset = (
        (row64[:, None] * output_sequence_length + pos64[:, None]) * heads + h64
    ) * dim + d64
    values = tl.load(grad_output + output_offset, mask=mask, other=0.0)
    tl.store(
        grad_bucket_output + (n64[:, None] * heads + h64[None, :]) * dim + d64[None, :],
        values,
        mask=mask,
    )
    tl.store(
        grad_base + output_offset,
        tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32),
        mask=mask,
    )


class _CompactBucketStreamGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        qkv_flat: Tensor,
        beta_flat: Tensor,
        recurrent_g_flat: Tensor,
        row_indices: Tensor,
        position_indices: Tensor,
        token_count: int,
        sequence_length: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _validate_cuda("qkv_flat", qkv_flat)
        qkv_flat = qkv_flat.contiguous()
        beta_flat = beta_flat.contiguous()
        recurrent_g_flat = recurrent_g_flat.contiguous()
        row_indices = row_indices.contiguous()
        position_indices = position_indices.contiguous()
        token_count = int(token_count)
        sequence_length = int(sequence_length)
        if tuple(row_indices.shape) != (token_count,):
            raise ValueError(
                "packed row_indices must be [tokens], got "
                f"{tuple(row_indices.shape)} for {token_count}"
            )
        if tuple(position_indices.shape) != (token_count,):
            raise ValueError(
                "packed position_indices must be [tokens], got "
                f"{tuple(position_indices.shape)} for {token_count}"
            )
        qkv_channels = int(qkv_flat.shape[-1])
        aux_width = int(beta_flat.shape[-1])
        qkv = torch.empty(
            (token_count, qkv_channels),
            device=qkv_flat.device,
            dtype=qkv_flat.dtype,
        )
        beta = torch.empty(
            (token_count, aux_width), device=beta_flat.device, dtype=beta_flat.dtype
        )
        recurrent_g = torch.empty(
            (token_count, aux_width),
            device=recurrent_g_flat.device,
            dtype=recurrent_g_flat.dtype,
        )
        block_n, block_qkv, block_aux = 32, 64, 32
        _gather_compact_qkv_kernel[
            (triton.cdiv(token_count, block_n), triton.cdiv(qkv_channels, block_qkv))
        ](
            qkv_flat,
            row_indices,
            position_indices,
            qkv,
            token_count,
            sequence_length,
            qkv_channels,
            BLOCK_N=block_n,
            BLOCK_D=block_qkv,
            num_warps=4,
        )
        grid_aux = (
            triton.cdiv(token_count, block_n),
            triton.cdiv(aux_width, block_aux),
        )
        _gather_compact_aux_kernel[grid_aux](
            beta_flat,
            row_indices,
            position_indices,
            beta,
            token_count,
            sequence_length,
            aux_width,
            BLOCK_N=block_n,
            BLOCK_D=block_aux,
            num_warps=4,
        )
        _gather_compact_aux_kernel[grid_aux](
            recurrent_g_flat,
            row_indices,
            position_indices,
            recurrent_g,
            token_count,
            sequence_length,
            aux_width,
            BLOCK_N=block_n,
            BLOCK_D=block_aux,
            num_warps=4,
        )
        ctx.save_for_backward(row_indices, position_indices)
        ctx.token_count = token_count
        ctx.sequence_length = sequence_length
        ctx.qkv_flat_count = int(qkv_flat.shape[0])
        ctx.beta_flat_count = int(beta_flat.shape[0])
        ctx.recurrent_g_flat_count = int(recurrent_g_flat.shape[0])
        ctx.qkv_channels = qkv_channels
        ctx.aux_width = aux_width
        return qkv, beta, recurrent_g

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor | None
    ) -> tuple[
        Tensor | None,
        Tensor | None,
        Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_qkv_bucket, grad_beta_bucket, grad_g_bucket = grad_outputs
        row_indices, position_indices = ctx.saved_tensors
        block_n, block_qkv, block_aux = 32, 64, 32
        grad_qkv = None
        if ctx.needs_input_grad[0] and grad_qkv_bucket is not None:
            grad_qkv_bucket = grad_qkv_bucket.contiguous()
            grad_qkv = grad_qkv_bucket.new_zeros(ctx.qkv_flat_count, ctx.qkv_channels)
            _scatter_compact_qkv_grad_kernel[
                (
                    triton.cdiv(ctx.token_count, block_n),
                    triton.cdiv(ctx.qkv_channels, block_qkv),
                )
            ](
                grad_qkv_bucket,
                row_indices,
                position_indices,
                grad_qkv,
                ctx.token_count,
                ctx.sequence_length,
                ctx.qkv_channels,
                BLOCK_N=block_n,
                BLOCK_D=block_qkv,
                num_warps=4,
            )
        grad_beta = None
        if ctx.needs_input_grad[1] and grad_beta_bucket is not None:
            grad_beta_bucket = grad_beta_bucket.contiguous()
            grad_beta = grad_beta_bucket.new_zeros(ctx.beta_flat_count, ctx.aux_width)
            _scatter_compact_aux_grad_kernel[
                (
                    triton.cdiv(ctx.token_count, block_n),
                    triton.cdiv(ctx.aux_width, block_aux),
                )
            ](
                grad_beta_bucket,
                row_indices,
                position_indices,
                grad_beta,
                ctx.token_count,
                ctx.sequence_length,
                ctx.aux_width,
                BLOCK_N=block_n,
                BLOCK_D=block_aux,
                num_warps=4,
            )
        grad_g = None
        if ctx.needs_input_grad[2] and grad_g_bucket is not None:
            grad_g_bucket = grad_g_bucket.contiguous()
            grad_g = grad_g_bucket.new_zeros(ctx.recurrent_g_flat_count, ctx.aux_width)
            _scatter_compact_aux_grad_kernel[
                (
                    triton.cdiv(ctx.token_count, block_n),
                    triton.cdiv(ctx.aux_width, block_aux),
                )
            ](
                grad_g_bucket,
                row_indices,
                position_indices,
                grad_g,
                ctx.token_count,
                ctx.sequence_length,
                ctx.aux_width,
                BLOCK_N=block_n,
                BLOCK_D=block_aux,
                num_warps=4,
            )
        return grad_qkv, grad_beta, grad_g, None, None, None, None, None, None


class _PreparePackedRecurrentInputs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        qkv: Tensor,
        beta: Tensor,
        recurrent_g: Tensor,
        key_heads: int,
        value_heads: int,
        key_dim: int,
        value_dim: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        _validate_cuda("qkv", qkv)
        qkv = qkv.contiguous()
        beta = beta.contiguous()
        recurrent_g = recurrent_g.contiguous()
        token_count, channels = qkv.shape
        key_heads = int(key_heads)
        value_heads = int(value_heads)
        key_dim = int(key_dim)
        value_dim = int(value_dim)
        if value_heads % key_heads != 0:
            raise ValueError(
                f"value_heads must be divisible by key_heads, got {value_heads} and {key_heads}"
            )
        expected_channels = 2 * key_heads * key_dim + value_heads * value_dim
        if int(channels) != expected_channels:
            raise ValueError(
                "packed qkv channel count mismatch, got "
                f"{channels} and expected {expected_channels}"
            )
        if tuple(beta.shape) != (token_count, value_heads):
            raise ValueError(
                f"beta must be [tokens, value_heads], got {tuple(beta.shape)}"
            )
        if tuple(recurrent_g.shape) != tuple(beta.shape):
            raise ValueError(
                "recurrent_g shape must match beta, got "
                f"{tuple(recurrent_g.shape)} and {tuple(beta.shape)}"
            )
        repeat = value_heads // key_heads
        query = torch.empty(
            (1, token_count, value_heads, key_dim), device=qkv.device, dtype=qkv.dtype
        )
        key = torch.empty_like(query)
        value = torch.empty(
            (1, token_count, value_heads, value_dim), device=qkv.device, dtype=qkv.dtype
        )
        block_n = 16
        block_d = triton.next_power_of_2(max(key_dim, value_dim))
        if block_d > 128:
            raise ValueError(
                f"unsupported GDN head dimension {block_d}; expected <= 128"
            )
        _prepare_packed_qkv_kernel[(triton.cdiv(token_count, block_n), value_heads, 3)](
            qkv,
            query,
            key,
            value,
            token_count,
            channels,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            repeat,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=1,
        )
        ctx.input_shape = tuple(qkv.shape)
        ctx.beta_shape = tuple(beta.shape)
        ctx.device = qkv.device
        ctx.input_dtype = qkv.dtype
        ctx.beta_dtype = beta.dtype
        ctx.g_dtype = recurrent_g.dtype
        ctx.key_heads = key_heads
        ctx.value_heads = value_heads
        ctx.key_dim = key_dim
        ctx.value_dim = value_dim
        ctx.repeat = repeat
        return query, key, value, beta.unsqueeze(0), recurrent_g.unsqueeze(0)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Tensor | None,
    ) -> tuple[
        Tensor | None,
        Tensor | None,
        Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        if len(grad_outputs) != 5:
            raise RuntimeError("expected five packed QKV output gradients")
        grad_query, grad_key, grad_value, grad_beta_out, grad_g_out = grad_outputs
        token_count, channels = ctx.input_shape
        grad_qkv = None
        device = None
        if grad_query is not None:
            device = grad_query.device
        elif grad_key is not None:
            device = grad_key.device
        elif grad_value is not None:
            device = grad_value.device
        elif grad_beta_out is not None:
            device = grad_beta_out.device
        elif grad_g_out is not None:
            device = grad_g_out.device
        if ctx.needs_input_grad[0]:
            if device is None:
                raise RuntimeError("missing device for packed qkv gradient")
            grad_qkv = torch.empty(
                (token_count, channels), device=device, dtype=ctx.input_dtype
            )
            grad_query_arg = (
                grad_query.contiguous() if grad_query is not None else grad_qkv
            )
            grad_key_arg = grad_key.contiguous() if grad_key is not None else grad_qkv
            grad_value_arg = (
                grad_value.contiguous() if grad_value is not None else grad_qkv
            )
            block_n, block_c = 16, 64
            _prepare_packed_qkv_backward_kernel[
                (triton.cdiv(token_count, block_n), triton.cdiv(channels, block_c))
            ](
                grad_query_arg,
                grad_key_arg,
                grad_value_arg,
                grad_qkv,
                token_count,
                channels,
                ctx.key_heads,
                ctx.value_heads,
                ctx.key_dim,
                ctx.value_dim,
                ctx.repeat,
                HAS_QUERY=grad_query is not None,
                HAS_KEY=grad_key is not None,
                HAS_VALUE=grad_value is not None,
                BLOCK_N=block_n,
                BLOCK_C=block_c,
                num_warps=4,
            )
        grad_beta = None
        if ctx.needs_input_grad[1]:
            grad_beta = (
                grad_beta_out.reshape(ctx.beta_shape).contiguous()
                if grad_beta_out is not None
                else torch.zeros(
                    ctx.beta_shape, device=ctx.device, dtype=ctx.beta_dtype
                )
            )
        grad_g = None
        if ctx.needs_input_grad[2]:
            grad_g = (
                grad_g_out.reshape(ctx.beta_shape).contiguous()
                if grad_g_out is not None
                else torch.zeros(ctx.beta_shape, device=ctx.device, dtype=ctx.g_dtype)
            )
        return grad_qkv, grad_beta, grad_g, None, None, None, None


class _CompactScatterBucketOutput(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        output: Tensor,
        bucket_output: Tensor,
        row_indices: Tensor,
        position_indices: Tensor,
        output_mask: Tensor,
    ) -> Tensor:
        _validate_cuda("output", output)
        output = output.contiguous()
        bucket_output = bucket_output.contiguous()
        row_indices = row_indices.contiguous()
        position_indices = position_indices.contiguous()
        output_mask = output_mask.contiguous()
        if bucket_output.ndim != 4 or int(bucket_output.shape[0]) != 1:
            raise ValueError(
                "bucket_output must have shape [1, tokens, heads, dim], got "
                f"{tuple(bucket_output.shape)}"
            )
        output_batch, output_sequence_length, heads, dim = output.shape
        del output_batch
        token_count = int(bucket_output.shape[1])
        if tuple(row_indices.shape) != (token_count,):
            raise ValueError(
                "packed row_indices must be [tokens], got "
                f"{tuple(row_indices.shape)} for {token_count}"
            )
        if tuple(position_indices.shape) != (token_count,):
            raise ValueError(
                "packed position_indices must be [tokens], got "
                f"{tuple(position_indices.shape)} for {token_count}"
            )
        if tuple(output_mask.shape) != (token_count,):
            raise ValueError(
                "packed output_mask must be [tokens], got "
                f"{tuple(output_mask.shape)} for {token_count}"
            )
        out = output.clone()
        block_n, block_d = 16, 64
        _scatter_bucket_output_compact_forward_kernel[
            (triton.cdiv(token_count, block_n), triton.cdiv(heads * dim, block_d))
        ](
            out,
            bucket_output,
            row_indices,
            position_indices,
            output_mask,
            token_count,
            output_sequence_length,
            heads,
            dim,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(row_indices, position_indices, output_mask)
        ctx.output_shape = tuple(output.shape)
        ctx.bucket_output_shape = tuple(bucket_output.shape)
        ctx.token_count = token_count
        return out

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Tensor | None
    ) -> tuple[Tensor, Tensor, None, None, None, None]:
        if len(grad_outputs) != 1 or grad_outputs[0] is None:
            raise RuntimeError("expected compact scatter output gradient")
        grad_out = grad_outputs[0]
        row_indices, position_indices, output_mask = ctx.saved_tensors
        _, output_sequence_length, heads, dim = ctx.output_shape
        grad_out = grad_out.contiguous()
        grad_base = grad_out.clone()
        grad_bucket = grad_out.new_zeros(ctx.bucket_output_shape)
        block_n, block_d = 16, 64
        _scatter_bucket_output_compact_backward_kernel[
            (
                triton.cdiv(ctx.token_count, block_n),
                triton.cdiv(heads * dim, block_d),
            )
        ](
            grad_out,
            grad_base,
            grad_bucket,
            row_indices,
            position_indices,
            output_mask,
            ctx.token_count,
            output_sequence_length,
            heads,
            dim,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return grad_base, grad_bucket, None, None, None, None


def gather_bucket_streams_compact(
    qkv_flat: Tensor,
    beta_flat: Tensor,
    recurrent_g_flat: Tensor,
    row_indices: Tensor,
    position_indices: Tensor,
    *,
    token_count: int,
    sequence_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return _CompactBucketStreamGather.apply(
        qkv_flat,
        beta_flat,
        recurrent_g_flat,
        row_indices,
        position_indices,
        token_count,
        sequence_length,
    )


def scatter_bucket_output_compact(
    output: Tensor,
    bucket_output: Tensor,
    row_indices: Tensor,
    position_indices: Tensor,
    output_mask: Tensor,
) -> Tensor:
    return _CompactScatterBucketOutput.apply(
        output,
        bucket_output,
        row_indices,
        position_indices,
        output_mask,
    )


def prepare_packed_recurrent_inputs(
    qkv: Tensor,
    beta: Tensor,
    recurrent_g: Tensor,
    *,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _PreparePackedRecurrentInputs.apply(
        qkv,
        beta,
        recurrent_g,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
    )


def _validate_cuda(name: str, tensor: Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
