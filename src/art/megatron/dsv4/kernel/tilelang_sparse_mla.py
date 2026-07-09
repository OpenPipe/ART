from typing import Any

import torch

from art.megatron.dsv4.kernel.tilelang_import import preserve_tilelang_env


def _sparse_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale):
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    bsz, seqlen, _, dim = q.shape
    safe_idxs = topk_idxs.clamp_min(0)
    selected_kv = torch.gather(
        kv[:, None].expand(-1, seqlen, -1, -1),
        2,
        safe_idxs[..., None].expand(-1, -1, -1, dim),
    )
    scores = torch.einsum("bshd,bskd->bshk", q.float(), selected_kv.float())
    scores = scores * float(sm_scale)
    scores = scores.masked_fill(topk_idxs[:, :, None, :] < 0, float("-inf"))
    sinks = attn_sink.view(1, 1, -1, 1).expand(bsz, seqlen, -1, -1)
    probs = torch.softmax(torch.cat([scores, sinks], dim=-1), dim=-1)
    attn_probs = probs[..., :-1]
    return torch.einsum("bshk,bskd->bshd", attn_probs, selected_kv.float())


def _pad_topk_idxs(topk_idxs: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    topk = int(topk_idxs.shape[-1])
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk == topk:
        return topk_idxs
    pad = torch.full(
        (*topk_idxs.shape[:-1], padded_topk - topk),
        -1,
        device=topk_idxs.device,
        dtype=topk_idxs.dtype,
    )
    return torch.cat([topk_idxs, pad], dim=-1).contiguous()


class DeepSeekV4SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, sm_scale=None, output_dtype=None):
        with preserve_tilelang_env():
            from art.megatron.dsv4.kernel import (
                tilelang_sparse_mla_fwd as sparse_mla_fwd,
            )

            o, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(
                q, kv, attn_sink, topk_idxs, sm_scale=sm_scale
            )

        output = o if output_dtype is None else o.to(output_dtype)
        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, output.clone(), lse)
        ctx.sm_scale = sm_scale

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        do = grad_outputs[0]
        q, kv, attn_sink, topk_idxs, output, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        with preserve_tilelang_env():
            from art.megatron.dsv4.kernel import (
                tilelang_sparse_mla_bwd as sparse_mla_bwd,
            )

            dq, dkv, d_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
                q,
                kv,
                attn_sink,
                output.to(q.dtype),
                do.to(q.dtype),
                topk_idxs,
                lse,
                sm_scale=sm_scale,
            )

        return dq, dkv, d_attn_sink, None, None, None


@torch.compiler.disable
def sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Run TileLang sparse MLA outside TorchDynamo tracing.

    TileLang's TVM FFI adapter uses non-literal string objects internally, which
    Dynamo cannot represent as constants. Keep only this kernel boundary eager
    while allowing the surrounding DSV4 transformer layer to compile.
    """
    output_dtype = q.dtype
    if q.dtype is torch.float32:
        return _sparse_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale)
    if kv.dtype != q.dtype:
        kv = kv.to(q.dtype)
    q = q.contiguous()
    kv = kv.contiguous()
    topk_idxs = _pad_topk_idxs(topk_idxs.contiguous())
    head_count = int(q.shape[2])
    if head_count < 16:
        pad_heads = 16 - head_count
        q = torch.cat(
            [
                q,
                q.new_zeros((*q.shape[:2], pad_heads, q.shape[3])),
            ],
            dim=2,
        ).contiguous()
        attn_sink = torch.cat(
            [attn_sink, attn_sink.new_zeros(pad_heads)],
            dim=0,
        ).contiguous()
    out = DeepSeekV4SparseAttention.apply(
        q, kv, attn_sink, topk_idxs, sm_scale, output_dtype
    )
    return out[:, :, :head_count, :]
