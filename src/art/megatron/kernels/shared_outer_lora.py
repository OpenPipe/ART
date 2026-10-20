"""Expert LoRA with a dense hidden-side factor and a grouped expert-side factor.

No expert-sized copy of a shared weight or its gradient is created. Inputs are
already dispatched, expert-major rows; this does not avoid top-k row duplication.
The shared gradient is a single GEMM, rather than rounded per-expert gradients
followed by a sum. Consequently BF16 backward need not match the expanded path.
"""

from typing import Any

import torch

from .cute_grouped_lora_quack import (
    _build_expert_offsets,
    _grad_a_tile_m,
    _grad_b_tile_m,
    _matmul_tile_n,
    _pad_a_t,
    _pad_b_t,
    _proj_tile_n,
    _tokens_per_expert_to_tensor,
    _varlen_quack_gemm,
    _varlen_quack_gemm_k,
)


def _grouped(x, w, offsets, out_features, *, alpha=1.0, projection=False):
    return _varlen_quack_gemm(
        x.contiguous(),
        w,
        out_features=out_features,
        expert_offsets=offsets,
        tile_m=64,
        tile_n=_proj_tile_n(out_features)
        if projection
        else _matmul_tile_n(out_features),
        alpha=alpha,
    )


class _SharedOuter(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, a, b, counts, scale, shared_a):
        rank = a.shape[-1]
        effective = ((rank + 7) // 8) * 8
        a = _pad_a_t(a, effective)
        b = _pad_b_t(b, effective)
        offsets = _build_expert_offsets(counts, device=x.device)
        if shared_a:
            tmp = x @ a[0]
            out = _grouped(
                tmp, b.transpose(1, 2).contiguous(), offsets, b.shape[-1], alpha=scale
            )
        else:
            tmp = _grouped(
                x, a.transpose(1, 2).contiguous(), offsets, effective, projection=True
            )
            out = torch.addmm(b.new_empty(b.shape[-1]), tmp, b[0], beta=0, alpha=scale)
        ctx.save_for_backward(x, a, b, tmp, offsets)
        ctx.rank, ctx.scale, ctx.shared_a = rank, scale, shared_a
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_out,) = grad_outputs
        x, a, b, tmp, offsets = ctx.saved_tensors
        effective, scale = a.shape[-1], ctx.scale
        grad_out = grad_out.contiguous()
        if ctx.shared_a:
            grad_tmp = _grouped(
                grad_out, b, offsets, effective, alpha=scale, projection=True
            )
            grad_x = grad_tmp @ a[0].T
            grad_a = (x.T @ grad_tmp).unsqueeze(0)
            grad_b = _varlen_quack_gemm_k(
                tmp.T,
                grad_out.T,
                batch_count=b.shape[0],
                out_shape_m=effective,
                out_shape_n=b.shape[-1],
                expert_offsets=offsets,
                tile_m=_grad_b_tile_m(effective),
                tile_n=_matmul_tile_n(b.shape[-1]),
                alpha=scale,
            )
        else:
            grad_tmp = torch.addmm(
                b.new_empty(effective), grad_out, b[0].T, beta=0, alpha=scale
            )
            grad_x = _grouped(grad_tmp, a, offsets, x.shape[-1])
            grad_a = _varlen_quack_gemm_k(
                x.T,
                grad_tmp.T,
                batch_count=a.shape[0],
                out_shape_m=a.shape[1],
                out_shape_n=effective,
                expert_offsets=offsets,
                tile_m=_grad_a_tile_m(effective),
                tile_n=_proj_tile_n(effective),
            )
            grad_b = torch.addmm(
                b.new_empty((effective, b.shape[-1])),
                tmp.T,
                grad_out,
                beta=0,
                alpha=scale,
            ).unsqueeze(0)
        return (
            grad_x,
            grad_a[..., : ctx.rank].contiguous(),
            grad_b[:, : ctx.rank].contiguous(),
            None,
            None,
            None,
        )


def shared_outer_lora(x, a_t, b_t, counts, *, scale=1.0, shared_factor):
    """Consume singleton-expert factors in ART's transposed storage order."""
    counts = _tokens_per_expert_to_tensor(counts)
    if shared_factor not in ("A", "B"):
        raise ValueError("shared_factor must be A or B")
    shared, other = (a_t, b_t) if shared_factor == "A" else (b_t, a_t)
    if shared.shape[0] != 1 or other.shape[0] != counts.numel():
        raise ValueError("expected one shared factor and one factor per local expert")
    if a_t.shape[-1] != b_t.shape[1] or a_t.shape[1] != x.shape[1]:
        raise ValueError("LoRA factor dimensions disagree")
    if x.dtype not in (torch.bfloat16, torch.float16) or any(
        t.dtype != x.dtype or t.device != x.device for t in (a_t, b_t)
    ):
        raise ValueError("shared-outer requires colocated BF16/FP16 tensors")
    if x.shape[0] == 0:
        # Keep input and both factors connected for empty local expert ranks.
        # Reading one element avoids a reduction across a large idle factor.
        zero = (x.sum() + a_t.flatten()[:1].sum() + b_t.flatten()[:1].sum()) * 0
        return x.new_zeros((0, b_t.shape[-1])) + zero
    return _SharedOuter.apply(x, a_t, b_t, counts, float(scale), shared_factor == "A")
