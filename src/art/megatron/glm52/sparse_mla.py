from __future__ import annotations

from typing import Any

import torch

from art.megatron.glm52 import tilelang_sparse_mla

_LATENT_DIM = 512
_ROPE_DIM = 64
_TOPK_BLOCK = 64


def sparse_mla_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(q, kv, indices)
    return tilelang_sparse_mla.forward(
        q.contiguous(), kv.contiguous(), indices.contiguous(), float(scale)
    )


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
    _validate_inputs(q, kv, indices)
    expected_out = (*q.shape[:-1], _LATENT_DIM)
    if out.shape != expected_out or grad_out.shape != expected_out:
        raise ValueError(
            f"GLM-5.2 sparse MLA output and gradient must have shape {expected_out}."
        )
    if lse.shape != q.shape[:-1] or lse.dtype is not torch.float32:
        raise ValueError("GLM-5.2 sparse MLA LSE must be fp32 with shape [B,S,H].")
    return tilelang_sparse_mla.backward(
        q.contiguous(),
        kv.contiguous(),
        indices.contiguous(),
        out.contiguous(),
        lse.contiguous(),
        grad_out.contiguous(),
        float(scale),
    )


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
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        q, kv, indices, out, lse = ctx.saved_tensors
        grad_q, grad_kv = sparse_mla_backward(
            q,
            kv,
            indices,
            out,
            lse,
            grad_outputs[0],
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
        q.contiguous(), kv.contiguous(), indices.contiguous(), float(scale)
    )


def _validate_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
) -> None:
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
    if not 0 < q.shape[2] <= 64 or q.shape[3] != _LATENT_DIM + _ROPE_DIM:
        raise ValueError("GLM-5.2 sparse MLA requires positive 576-dimensional heads.")
    if kv.shape[-1] != q.shape[-1] or q.shape[:2] != indices.shape[:2]:
        raise ValueError("GLM-5.2 sparse MLA tensor shapes do not match.")
    if q.shape[0] != kv.shape[0]:
        raise ValueError("GLM-5.2 sparse MLA batch dimensions do not match.")
    if indices.shape[-1] % _TOPK_BLOCK:
        raise ValueError(
            f"GLM-5.2 sparse MLA top-k must be divisible by {_TOPK_BLOCK}."
        )
