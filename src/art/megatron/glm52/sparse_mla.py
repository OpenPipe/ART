from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

_LATENT_DIM = 512
_ROPE_DIM = 64
_HEAD_BLOCK = 16
_INDEX_BLOCK = 32


@triton.jit
def _sparse_mla_fwd_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    out_ptr,
    lse_ptr,
    q_tokens,
    kv_tokens,
    heads,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvs,
    stride_kvd,
    stride_ib,
    stride_is,
    stride_ik,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    stride_lb,
    stride_ls,
    stride_lh,
    scale: tl.constexpr,
    topk: tl.constexpr,
    latent_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_h: tl.constexpr,
    block_i: tl.constexpr,
):
    row = tl.program_id(0)
    h_block = tl.program_id(1)
    batch = row // q_tokens
    token = row - batch * q_tokens
    h = h_block * block_h + tl.arange(0, block_h)
    latent = tl.arange(0, latent_dim)
    rope = tl.arange(0, rope_dim)

    q_latent = tl.load(
        q_ptr
        + batch * stride_qb
        + token * stride_qs
        + h[:, None] * stride_qh
        + latent[None, :] * stride_qd,
        mask=h[:, None] < heads,
        other=0.0,
    )
    q_rope = tl.load(
        q_ptr
        + batch * stride_qb
        + token * stride_qs
        + h[:, None] * stride_qh
        + (latent_dim + rope[None, :]) * stride_qd,
        mask=h[:, None] < heads,
        other=0.0,
    )
    acc = tl.zeros((block_h, latent_dim), tl.float32)
    row_max = tl.full((block_h,), float("-inf"), tl.float32)
    row_sum = tl.zeros((block_h,), tl.float32)

    for index_start in range(0, topk, block_i):
        index_offsets = index_start + tl.arange(0, block_i)
        indices = tl.load(
            indices_ptr
            + batch * stride_ib
            + token * stride_is
            + index_offsets * stride_ik,
        )
        valid_indices = (indices >= 0) & (indices < kv_tokens)
        safe_indices = tl.maximum(indices, 0)
        kv_latent = tl.load(
            kv_ptr
            + batch * stride_kvb
            + safe_indices[:, None] * stride_kvs
            + latent[None, :] * stride_kvd,
            mask=valid_indices[:, None],
            other=0.0,
        )
        kv_rope = tl.load(
            kv_ptr
            + batch * stride_kvb
            + safe_indices[:, None] * stride_kvs
            + (latent_dim + rope[None, :]) * stride_kvd,
            mask=valid_indices[:, None],
            other=0.0,
        )
        scores = tl.dot(q_latent, tl.trans(kv_latent))
        scores += tl.dot(q_rope, tl.trans(kv_rope))
        valid = (h[:, None] < heads) & valid_indices[None, :]
        scores = tl.where(valid, scores * scale, float("-inf"))
        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(row_max, block_max)
        alpha = tl.exp(row_max - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        acc *= alpha[:, None]
        acc += tl.dot(probabilities.to(tl.bfloat16), kv_latent)
        row_max = next_max

    has_keys = row_sum > 0.0
    output = tl.where(has_keys[:, None], acc / row_sum[:, None], 0.0)
    lse = tl.where(has_keys, tl.log(row_sum) + row_max, float("-inf"))
    tl.store(
        out_ptr
        + batch * stride_ob
        + token * stride_os
        + h[:, None] * stride_oh
        + latent[None, :] * stride_od,
        output,
        mask=h[:, None] < heads,
    )
    tl.store(
        lse_ptr + batch * stride_lb + token * stride_ls + h * stride_lh,
        lse,
        mask=h < heads,
    )


@triton.jit
def _sparse_mla_bwd_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    out_ptr,
    grad_out_ptr,
    lse_ptr,
    grad_q_ptr,
    grad_kv_ptr,
    q_tokens,
    kv_tokens,
    heads,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvs,
    stride_kvd,
    stride_ib,
    stride_is,
    stride_ik,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    stride_gob,
    stride_gos,
    stride_goh,
    stride_god,
    stride_lb,
    stride_ls,
    stride_lh,
    stride_gqb,
    stride_gqs,
    stride_gqh,
    stride_gqd,
    stride_gkb,
    stride_gks,
    stride_gkd,
    scale: tl.constexpr,
    topk: tl.constexpr,
    latent_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    block_h: tl.constexpr,
    block_i: tl.constexpr,
):
    row = tl.program_id(0)
    h_block = tl.program_id(1)
    batch = row // q_tokens
    token = row - batch * q_tokens
    h = h_block * block_h + tl.arange(0, block_h)
    latent = tl.arange(0, latent_dim)
    rope = tl.arange(0, rope_dim)
    valid_heads = h < heads

    q_latent = tl.load(
        q_ptr
        + batch * stride_qb
        + token * stride_qs
        + h[:, None] * stride_qh
        + latent[None, :] * stride_qd,
        mask=valid_heads[:, None],
        other=0.0,
    )
    q_rope = tl.load(
        q_ptr
        + batch * stride_qb
        + token * stride_qs
        + h[:, None] * stride_qh
        + (latent_dim + rope[None, :]) * stride_qd,
        mask=valid_heads[:, None],
        other=0.0,
    )
    out = tl.load(
        out_ptr
        + batch * stride_ob
        + token * stride_os
        + h[:, None] * stride_oh
        + latent[None, :] * stride_od,
        mask=valid_heads[:, None],
        other=0.0,
    )
    grad_out = tl.load(
        grad_out_ptr
        + batch * stride_gob
        + token * stride_gos
        + h[:, None] * stride_goh
        + latent[None, :] * stride_god,
        mask=valid_heads[:, None],
        other=0.0,
    )
    lse = tl.load(
        lse_ptr + batch * stride_lb + token * stride_ls + h * stride_lh,
        mask=valid_heads,
        other=float("-inf"),
    )
    delta = tl.sum(out.to(tl.float32) * grad_out.to(tl.float32), axis=1)
    grad_q_latent = tl.zeros((block_h, latent_dim), tl.float32)
    grad_q_rope = tl.zeros((block_h, rope_dim), tl.float32)

    for index_start in range(0, topk, block_i):
        index_offsets = index_start + tl.arange(0, block_i)
        indices = tl.load(
            indices_ptr
            + batch * stride_ib
            + token * stride_is
            + index_offsets * stride_ik,
        )
        valid_indices = (indices >= 0) & (indices < kv_tokens)
        safe_indices = tl.maximum(indices, 0)
        kv_latent = tl.load(
            kv_ptr
            + batch * stride_kvb
            + safe_indices[:, None] * stride_kvs
            + latent[None, :] * stride_kvd,
            mask=valid_indices[:, None],
            other=0.0,
        )
        kv_rope = tl.load(
            kv_ptr
            + batch * stride_kvb
            + safe_indices[:, None] * stride_kvs
            + (latent_dim + rope[None, :]) * stride_kvd,
            mask=valid_indices[:, None],
            other=0.0,
        )
        scores = tl.dot(q_latent, tl.trans(kv_latent))
        scores += tl.dot(q_rope, tl.trans(kv_rope))
        valid = valid_heads[:, None] & valid_indices[None, :]
        probabilities = tl.exp(scores * scale - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_out, tl.trans(kv_latent))
        grad_scores = probabilities * (grad_probabilities - delta[:, None]) * scale

        grad_q_latent += tl.dot(grad_scores.to(tl.bfloat16), kv_latent)
        grad_q_rope += tl.dot(grad_scores.to(tl.bfloat16), kv_rope)
        grad_key_latent = tl.dot(tl.trans(grad_scores.to(tl.bfloat16)), q_latent)
        grad_value = tl.dot(tl.trans(probabilities.to(tl.bfloat16)), grad_out)
        grad_key_rope = tl.dot(tl.trans(grad_scores.to(tl.bfloat16)), q_rope)
        tl.atomic_add(
            grad_kv_ptr
            + batch * stride_gkb
            + safe_indices[:, None] * stride_gks
            + latent[None, :] * stride_gkd,
            grad_key_latent + grad_value,
            mask=valid_indices[:, None],
        )
        tl.atomic_add(
            grad_kv_ptr
            + batch * stride_gkb
            + safe_indices[:, None] * stride_gks
            + (latent_dim + rope[None, :]) * stride_gkd,
            grad_key_rope,
            mask=valid_indices[:, None],
        )

    tl.store(
        grad_q_ptr
        + batch * stride_gqb
        + token * stride_gqs
        + h[:, None] * stride_gqh
        + latent[None, :] * stride_gqd,
        grad_q_latent,
        mask=valid_heads[:, None],
    )
    tl.store(
        grad_q_ptr
        + batch * stride_gqb
        + token * stride_gqs
        + h[:, None] * stride_gqh
        + (latent_dim + rope[None, :]) * stride_gqd,
        grad_q_rope,
        mask=valid_heads[:, None],
    )


def sparse_mla_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(q, kv, indices)
    batch, q_tokens, heads, _ = q.shape
    kv_tokens = int(kv.shape[1])
    topk = int(indices.shape[-1])
    if topk % _INDEX_BLOCK:
        raise ValueError(f"GLM-5.2 sparse topk must be divisible by {_INDEX_BLOCK}.")
    out = torch.empty(
        (batch, q_tokens, heads, _LATENT_DIM), device=q.device, dtype=q.dtype
    )
    lse = torch.empty((batch, q_tokens, heads), device=q.device, dtype=torch.float32)
    _sparse_mla_fwd_kernel[(batch * q_tokens, triton.cdiv(heads, _HEAD_BLOCK))](
        q,
        kv,
        indices,
        out,
        lse,
        q_tokens,
        kv_tokens,
        heads,
        *q.stride(),
        *kv.stride(),
        *indices.stride(),
        *out.stride(),
        *lse.stride(),
        scale=scale,  # ty: ignore[invalid-argument-type]
        topk=topk,  # ty: ignore[invalid-argument-type]
        latent_dim=_LATENT_DIM,  # ty: ignore[invalid-argument-type]
        rope_dim=_ROPE_DIM,  # ty: ignore[invalid-argument-type]
        block_h=_HEAD_BLOCK,  # ty: ignore[invalid-argument-type]
        block_i=_INDEX_BLOCK,  # ty: ignore[invalid-argument-type]
        num_warps=8,  # ty: ignore[unknown-argument]
        num_stages=2,  # ty: ignore[unknown-argument]
    )
    return out, lse


def sparse_mla_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, q_tokens, heads, _ = q.shape
    kv_tokens = int(kv.shape[1])
    topk = int(indices.shape[-1])
    grad_q = torch.empty_like(q)
    grad_kv_fp32 = torch.zeros_like(kv, dtype=torch.float32)
    _sparse_mla_bwd_kernel[(batch * q_tokens, triton.cdiv(heads, _HEAD_BLOCK))](
        q,
        kv,
        indices,
        out,
        grad_out.contiguous(),
        lse,
        grad_q,
        grad_kv_fp32,
        q_tokens,
        kv_tokens,
        heads,
        *q.stride(),
        *kv.stride(),
        *indices.stride(),
        *out.stride(),
        *grad_out.stride(),
        *lse.stride(),
        *grad_q.stride(),
        *grad_kv_fp32.stride(),
        scale=scale,  # ty: ignore[invalid-argument-type]
        topk=topk,  # ty: ignore[invalid-argument-type]
        latent_dim=_LATENT_DIM,  # ty: ignore[invalid-argument-type]
        rope_dim=_ROPE_DIM,  # ty: ignore[invalid-argument-type]
        block_h=_HEAD_BLOCK,  # ty: ignore[invalid-argument-type]
        block_i=_INDEX_BLOCK,  # ty: ignore[invalid-argument-type]
        num_warps=8,  # ty: ignore[unknown-argument]
        num_stages=1,  # ty: ignore[unknown-argument]
    )
    return grad_q, grad_kv_fp32.to(kv.dtype)


class _SparseMla(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        out, lse = sparse_mla_forward(q, kv, indices, scale=scale)
        ctx.save_for_backward(q, kv, indices, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        q, kv, indices, out, lse = ctx.saved_tensors
        grad_out = grad_outputs[0]
        grad_q, grad_kv = sparse_mla_backward(
            q,
            kv,
            indices,
            out,
            lse,
            grad_out,
            scale=ctx.scale,
        )
        return grad_q, grad_kv, None, None


def sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Run GLM-5.2 list-sparse absorbed MLA with BF16 forward/backward."""
    return _SparseMla.apply(
        q.contiguous(), kv.contiguous(), indices.contiguous(), scale
    )


def _validate_inputs(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> None:
    if q.dtype is not torch.bfloat16 or kv.dtype is not torch.bfloat16:
        raise TypeError(f"GLM-5.2 sparse MLA requires bf16, got {q.dtype}/{kv.dtype}.")
    if indices.dtype is not torch.int32:
        raise TypeError(
            f"GLM-5.2 sparse MLA indices must be int32, got {indices.dtype}."
        )
    if not q.is_cuda or q.device != kv.device or q.device != indices.device:
        raise RuntimeError("GLM-5.2 sparse MLA requires colocated CUDA tensors.")
    if q.ndim != 4 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            "GLM-5.2 sparse MLA expects q[B,S,H,576], kv[B,K,576], ids[B,S,T]."
        )
    if q.shape[-1] != _LATENT_DIM + _ROPE_DIM or kv.shape[-1] != q.shape[-1]:
        raise ValueError(
            f"GLM-5.2 sparse MLA requires absorbed dim {_LATENT_DIM + _ROPE_DIM}."
        )
    if q.shape[:2] != indices.shape[:2] or q.shape[0] != kv.shape[0]:
        raise ValueError("GLM-5.2 sparse MLA batch/query shapes do not match.")
