from typing import Any

import torch

from art.megatron.dsv4.kernel import tilelang_sparse_mla_bwd as sparse_mla_bwd
from art.megatron.dsv4.kernel import tilelang_sparse_mla_fwd as sparse_mla_fwd


def _exact_sink_grad(q, kv, attn_sink, topk_idxs, do, sm_scale):
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
    p_sink = probs[..., -1]
    output = torch.einsum("bshk,bskd->bshd", attn_probs, selected_kv.float())
    return -((output * do.float()).sum(dim=-1) * p_sink).sum(dim=(0, 1))


class DeepSeekV4SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        attn_sink,
        topk_idxs,
        sm_scale=None,
        output_dtype=None,
        exact_sink_q=None,
        exact_sink_kv=None,
    ):
        o, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(
            q, kv, attn_sink, topk_idxs, sm_scale=sm_scale
        )

        output = o if output_dtype is None else o.to(output_dtype)
        if exact_sink_q is None:
            exact_sink_q = q.new_empty(0)
            exact_sink_kv = kv.new_empty(0)
        ctx.save_for_backward(
            q,
            kv,
            attn_sink,
            topk_idxs,
            output.clone(),
            lse,
            exact_sink_q,
            exact_sink_kv,
        )
        ctx.sm_scale = sm_scale

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        do = grad_outputs[0]
        q, kv, attn_sink, topk_idxs, output, lse, exact_sink_q, exact_sink_kv = (
            ctx.saved_tensors
        )
        sm_scale = ctx.sm_scale

        dq, dkv, _ = sparse_mla_bwd.sparse_mqa_bwd_interface(
            q,
            kv,
            attn_sink,
            output.to(q.dtype),
            do.to(q.dtype),
            topk_idxs,
            lse,
            sm_scale=sm_scale,
        )
        if exact_sink_q.numel() > 0:
            d_attn_sink = _exact_sink_grad(
                exact_sink_q, exact_sink_kv, attn_sink, topk_idxs, do, sm_scale
            )
        else:
            p_sink = torch.exp2(attn_sink.view(1, 1, -1) * 1.4426950408889634 - lse)
            d_attn_sink = -((output.float() * do.float()).sum(dim=-1) * p_sink).sum(
                dim=(0, 1)
            )

        return dq, dkv, d_attn_sink, None, None, None, None, None


@torch.compiler.disable
def sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Run TileLang sparse MLA outside TorchDynamo tracing.

    TileLang's TVM FFI adapter uses non-literal string objects internally, which
    Dynamo cannot represent as constants. Keep only this kernel boundary eager
    while allowing the surrounding DSV4 transformer layer to compile.
    """
    output_dtype = q.dtype
    if q.dtype is torch.float32:
        exact_sink_q = q.detach()
        exact_sink_kv = kv.detach()
        q = q.to(torch.bfloat16)
        kv = kv.to(torch.bfloat16)
    else:
        exact_sink_q = None
        exact_sink_kv = None
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
        if exact_sink_q is not None:
            exact_sink_q = torch.cat(
                [
                    exact_sink_q,
                    exact_sink_q.new_zeros(
                        (*exact_sink_q.shape[:2], pad_heads, exact_sink_q.shape[3])
                    ),
                ],
                dim=2,
            ).contiguous()
        attn_sink = torch.cat(
            [attn_sink, attn_sink.new_zeros(pad_heads)],
            dim=0,
        ).contiguous()
    out = DeepSeekV4SparseAttention.apply(
        q,
        kv,
        attn_sink,
        topk_idxs,
        sm_scale,
        output_dtype,
        exact_sink_q,
        exact_sink_kv,
    )
    return out[:, :, :head_count, :]
