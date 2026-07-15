from __future__ import annotations

from typing import Any

from fla.modules.l2norm import (
    l2norm_bwd_kernel,
    l2norm_bwd_kernel1,
    l2norm_fwd_kernel,
    l2norm_fwd_kernel1,
)
import torch
from torch import Tensor
import triton


class _DynamicL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, eps: float) -> Tensor:
        shape = x.shape
        x = x.view(-1, shape[-1])
        rows, width = x.shape
        block_width = triton.next_power_of_2(width)
        y = torch.empty_like(x)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        if width <= 512:
            l2norm_fwd_kernel[lambda meta: (triton.cdiv(rows, meta["BT"]),)](
                x=x,
                y=y,
                rstd=rstd,
                eps=eps,
                T=rows,
                D=width,
                BD=block_width,
                # NB is only an FLA autotune key; a fixed value keeps dynamic
                # token counts from compiling a new configuration set.
                NB=1,
            )
        else:
            l2norm_fwd_kernel1[(rows,)](
                x=x, y=y, rstd=rstd, eps=eps, D=width, BD=block_width
            )
        ctx.eps = eps
        ctx.input_shape = shape
        ctx.save_for_backward(y, rstd)
        return y.view(shape)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (dy,) = grad_outputs
        y, rstd = ctx.saved_tensors
        y = y.view(-1, ctx.input_shape[-1])
        dy = dy.contiguous().view_as(y)
        rows, width = y.shape
        block_width = triton.next_power_of_2(width)
        dx = torch.empty_like(y)
        if width <= 512:
            l2norm_bwd_kernel[lambda meta: (triton.cdiv(rows, meta["BT"]),)](
                y=y,
                rstd=rstd,
                dy=dy,
                dx=dx,
                eps=ctx.eps,
                T=rows,
                D=width,
                BD=block_width,
                NB=1,
            )
        else:
            l2norm_bwd_kernel1[(rows,)](
                y=y,
                rstd=rstd,
                dy=dy,
                dx=dx,
                eps=ctx.eps,
                D=width,
                BD=block_width,
            )
        return dx.view(ctx.input_shape), None


def dynamic_l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return _DynamicL2Norm.apply(x, eps)
