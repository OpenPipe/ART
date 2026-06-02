from typing import Any

import torch

from art.megatron.dsv4.kernel import tilelang_sparse_mla_bwd as sparse_mla_bwd
from art.megatron.dsv4.kernel import tilelang_sparse_mla_fwd as sparse_mla_fwd


class DeepSeekV4SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, sm_scale=None):
        o, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(
            q, kv, attn_sink, topk_idxs, sm_scale=sm_scale
        )

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, o.clone(), lse)
        ctx.sm_scale = sm_scale

        return o

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        do = grad_outputs[0]
        q, kv, attn_sink, topk_idxs, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        dq, dkv, d_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
            q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=sm_scale
        )

        return dq, dkv, d_attn_sink, None, None


@torch.compiler.disable
def sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """Run TileLang sparse MLA outside TorchDynamo tracing.

    TileLang's TVM FFI adapter uses non-literal string objects internally, which
    Dynamo cannot represent as constants. Keep only this kernel boundary eager
    while allowing the surrounding DSV4 transformer layer to compile.
    """
    output_dtype = q.dtype
    if q.dtype is torch.float32:
        q = q.to(torch.bfloat16)
        kv = kv.to(torch.bfloat16)
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
    out = DeepSeekV4SparseAttention.apply(q, kv, attn_sink, topk_idxs, sm_scale)
    return out[:, :, :head_count, :].to(output_dtype)
