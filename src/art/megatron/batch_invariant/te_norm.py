"""TE 2.11 pad-to-four RMSNorm, retaining TE's native VJP and Parameter owner."""

import importlib

import torch
import torch.nn.functional as F


class _Pad4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, zero_centered):
        TE_DType = importlib.import_module(
            "transformer_engine.pytorch.constants"
        ).TE_DType
        tex = importlib.import_module("transformer_engine_torch")

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        padded = F.pad(flat, (0, 0, 0, (-len(flat)) % 4))
        gamma = weight.to(x.dtype).contiguous()
        y, _, r = tex.rmsnorm_fwd(
            padded, gamma, eps, None, None, TE_DType[x.dtype], 0, zero_centered
        )
        ctx.save_for_backward(padded, gamma, r)
        ctx.shape, ctx.dtype, ctx.zero_centered = x.shape, weight.dtype, zero_centered
        return y[: len(flat)].reshape_as(x)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        tex = importlib.import_module("transformer_engine_torch")

        x, gamma, r = ctx.saved_tensors
        flat = dy.reshape(-1, dy.shape[-1]).contiguous()
        dy = F.pad(flat, (0, 0, 0, len(x) - len(flat)))
        dx, dg = tex.rmsnorm_bwd(dy, x, r, gamma, 0, ctx.zero_centered)
        return dx[: len(flat)].reshape(ctx.shape), dg.to(ctx.dtype), None, None


def pad4(x, weight, eps, zero_centered=False):
    return _Pad4.apply(x, weight, eps, zero_centered)
