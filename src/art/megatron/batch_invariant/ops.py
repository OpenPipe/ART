"""Differentiable row-invariant primitives, including a CPU contract oracle."""

from typing import Any, cast

import torch
import torch.nn.functional as F

POLICY_VERSION = "fixed-tree-rows-v2"
ROW_TILE = 128
K_TILE = 32


def fixed_sum(x: torch.Tensor) -> torch.Tensor:
    """Adjacent binary tree on the last axis, padded only from its width."""
    width = x.shape[-1]
    if width < 1:
        raise ValueError("a reduction needs a nonempty channel dimension")
    x = F.pad(x, (0, (1 << (width - 1).bit_length()) - width))
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x


def _cpu_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Constant physical M/N/K even for a tail. This is an executable contract,
    # not a fast CPU implementation. Preserve FP64 for gradient references.
    m, k = a.shape
    n = b.shape[1]
    dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
    rows = []
    for start in range(0, m, ROW_TILE):
        columns = []
        for col in range(0, n, 64):
            acc = a.new_zeros((ROW_TILE, 64), dtype=dtype)
            for kk in range(0, k, K_TILE):
                aa = a[start : start + ROW_TILE, kk : kk + K_TILE]
                bb = b[kk : kk + K_TILE, col : col + 64]
                aa = F.pad(aa, (0, K_TILE - aa.shape[1], 0, ROW_TILE - aa.shape[0]))
                bb = F.pad(bb, (0, 64 - bb.shape[1], 0, K_TILE - bb.shape[0]))
                acc = acc + aa.to(dtype) @ bb.to(dtype)
            columns.append(
                acc[: min(ROW_TILE, m - start), : min(64, n - col)].to(a.dtype)
            )
        rows.append(torch.cat(columns, dim=1))
    return torch.cat(rows) if rows else a.new_empty((m, n))


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.device.type != "cuda" or a.dtype == torch.float64:
        return _cpu_mm(a, b)
    from .fast_ops import persistent

    return persistent(a, b)


class _Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _mm(a, b)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        a, b = ctx.saved_tensors
        return (
            _mm(dy, b.T) if ctx.needs_input_grad[0] else None,
            _mm(a.T.float(), dy.float()).to(b.dtype)
            if ctx.needs_input_grad[1] and b.dtype != torch.float64
            else _mm(a.T, dy)
            if ctx.needs_input_grad[1]
            else None,
        )


def matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x[..., K] @ weight[K, N], with the same geometry in forward and dgrad.

    Wgrad has an inherently batch-dependent reduction domain; its K traversal
    is fixed, with no shape-selected split. Parameters keep their identities.
    """
    if x.ndim < 2 or weight.ndim != 2 or x.shape[-1] != weight.shape[0]:
        raise ValueError("expected [..., K] @ [K, N]")
    if x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError("matmul requires matching dtype/device")
    flat = x.reshape(-1, x.shape[-1])
    return _Matmul.apply(flat, weight).reshape(*x.shape[:-1], weight.shape[1])


class _Bias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        ctx.save_for_backward(bias)
        return x + bias

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        (bias,) = ctx.saved_tensors
        flat = dy.reshape(-1, dy.shape[-1])
        # The same fixed K schedule as wgrad; accumulate low-precision grads
        # in FP32 and cast once at the parameter boundary.
        dtype = torch.float64 if dy.dtype == torch.float64 else torch.float32
        grad = _mm(flat.T.to(dtype), flat.new_ones((flat.shape[0], 1), dtype=dtype))
        return dy, grad[:, 0].to(bias.dtype)


def linear(x, weight, bias=None):
    out = matmul(x, weight.T)
    return out if bias is None else _Bias.apply(out, bias)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, zero_centered):
        from .kernels import rms_forward

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.empty_like(flat)
        r = torch.empty(flat.shape[0], dtype=torch.float32, device=x.device)
        width = x.shape[-1]
        if flat.shape[0]:
            cast(Any, rms_forward)[(flat.shape[0],)](
                flat,
                weight,
                out,
                r,
                width,
                eps,
                int(zero_centered),
                1 << (width - 1).bit_length(),
                num_warps=4,
                enable_fp_fusion=False,
            )
        ctx.save_for_backward(flat, weight, r)
        ctx.zero_centered = zero_centered
        ctx.shape = x.shape
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        from .kernels import rms_backward, rms_weight_backward

        x, weight, r = ctx.saved_tensors
        dy = dy.reshape_as(x).contiguous()
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        width = x.shape[-1]
        if x.shape[0]:
            if ctx.needs_input_grad[0]:
                cast(Any, rms_backward)[(x.shape[0],)](
                    x,
                    weight,
                    r,
                    dy,
                    dx,
                    width,
                    int(ctx.zero_centered),
                    1 << (width - 1).bit_length(),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            if ctx.needs_input_grad[1]:
                cast(Any, rms_weight_backward)[((width + 31) // 32,)](
                    x,
                    r,
                    dy,
                    dw,
                    x.shape[0],
                    width,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        else:
            dw.zero_()
        return (
            dx.reshape(ctx.shape) if ctx.needs_input_grad[0] else None,
            dw,
            None,
            None,
        )


def rms_norm(x, weight, eps, *, zero_centered=False):
    if x.shape[-1] != weight.numel() or weight.ndim != 1:
        raise ValueError("RMSNorm requires one weight per channel")
    if x.device.type == "cuda" and x.dtype != torch.float64:
        return _RMSNorm.apply(x, weight, eps, zero_centered)
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    xx = x.to(dtype)
    r = torch.rsqrt(fixed_sum(xx * xx) / x.shape[-1] + eps)
    return ((xx * r) * (weight.to(dtype) + int(zero_centered))).to(x.dtype)


class _GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, counts, *weights):
        from .fast_ops import experts

        ctx.counts = counts
        ctx.save_for_backward(x, *weights)
        return experts(x, weights, counts)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        from .fast_ops import experts

        x, *weights = ctx.saved_tensors
        dx = (
            experts(dy, [w.T for w in weights], ctx.counts)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = []
        for i, (part, grad, w) in enumerate(
            zip(x.split(ctx.counts), dy.split(ctx.counts), weights, strict=True)
        ):
            dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            dw.append(
                _mm(part.to(dtype).T, grad.to(dtype)).to(w.dtype)
                if ctx.needs_input_grad[i + 2]
                else None
            )
        return dx, None, *dw


def grouped_matmul(x, weights, counts):
    return _GroupedMatmul.apply(x, tuple(counts), *weights)


def _project(x, weight, counts):
    if counts is None:
        return _mm(x, weight)
    if weight.shape[0] == 1:
        return _mm(x, weight[0])
    from .fast_ops import experts

    return experts(x, tuple(weight.unbind()), counts)


def _project_wgrad(x, dy, counts, shared):
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    if counts is None:
        return _mm(x.to(dtype).T, dy.to(dtype))
    if shared:
        return _mm(x.to(dtype).T, dy.to(dtype)).unsqueeze(0)
    return torch.stack(
        [
            _mm(a.to(dtype).T, b.to(dtype))
            for a, b in zip(x.split(counts), dy.split(counts), strict=True)
        ]
    )


def _parameter_gradient(parameter, gradient):
    if getattr(parameter, "_stark_fp32_gradients", False):
        if getattr(parameter, "main_grad", None) is None:
            parameter.main_grad = gradient.float().clone()
        else:
            parameter.main_grad.add_(gradient.float())
        return None
    return gradient.to(parameter.dtype)


class _LoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, counts, scale):
        flat = x.reshape(-1, x.shape[-1])
        ctx.save_for_backward(flat, a, b)
        ctx.parameters = a, b
        ctx.counts, ctx.scale, ctx.shape = counts, scale, x.shape
        out = _project(_project(flat, a, counts), b, counts)
        return (out if scale == 1 else out * scale).reshape(*x.shape[:-1], b.shape[-1])

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        x, a, b = ctx.saved_tensors
        counts, scale = ctx.counts, ctx.scale
        dy = dy.reshape(-1, dy.shape[-1])
        # Scale once in the cotangent, matching the two rounded forward GEMMs.
        scaled = dy if scale == 1 else dy * scale
        tmp = _project(x, a, counts)
        dt = _project(scaled, b.transpose(-1, -2), counts)
        dx = (
            _project(dt, a.transpose(-1, -2), counts)
            if ctx.needs_input_grad[0]
            else None
        )
        da = _project_wgrad(x, dt, counts, counts is not None and a.shape[0] == 1)
        db = _project_wgrad(tmp, scaled, counts, counts is not None and b.shape[0] == 1)
        return (
            None if dx is None else dx.reshape(ctx.shape),
            _parameter_gradient(ctx.parameters[0], da)
            if ctx.needs_input_grad[1]
            else None,
            _parameter_gradient(ctx.parameters[1], db)
            if ctx.needs_input_grad[2]
            else None,
            None,
            None,
        )


def lora(x, a, b, counts=None, *, scale=1.0):
    if counts is not None:
        if isinstance(counts, torch.Tensor):
            if counts.device.type != "cpu":
                raise ValueError("expert counts must be prepared on CPU")
            counts = counts.tolist()
        counts = tuple(counts)
        if sum(counts) != x.reshape(-1, x.shape[-1]).shape[0] or any(
            c < 0 for c in counts
        ):
            raise ValueError("expert counts do not cover the input rows")
        if a.shape[0] not in (1, len(counts)) or b.shape[0] not in (1, len(counts)):
            raise ValueError("expert factors differ from counts")
    return _LoRA.apply(x, a, b, counts, scale)


def grouped_lora(x, a, b, counts, *, scale=1.0):
    return lora(x, a, b, counts, scale=scale)


class _ParameterFloat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter):
        ctx.parameter = parameter
        return parameter.float()

    @staticmethod
    def backward(ctx, *grad_outputs):
        (gradient,) = grad_outputs
        return _parameter_gradient(ctx.parameter, gradient)


def parameter_float(parameter):
    return None if parameter is None else _ParameterFloat.apply(parameter)


class _LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        from .kernels import lse_forward

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.empty((flat.shape[0], 1), dtype=torch.float32, device=x.device)
        if flat.shape[0]:
            cast(Any, lse_forward)[(flat.shape[0],)](
                flat, out, flat.shape[-1], num_warps=4, enable_fp_fusion=False
            )
        ctx.save_for_backward(x, out.reshape(*x.shape[:-1], 1))
        return out.reshape(*x.shape[:-1], 1)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        x, out = ctx.saved_tensors
        return ((x.float() - out).exp() * dy).to(x.dtype)


def logsumexp(x):
    """Fixed per-row vocabulary reduction, retaining the last dimension."""
    if x.device.type == "cuda" and x.dtype != torch.float64:
        return _LogSumExp.apply(x)
    maximum = x.amax(-1, keepdim=True)
    return maximum + fixed_sum((x - maximum).exp()).log()


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        from .kernels import sum_forward

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.empty((flat.shape[0], 1), dtype=torch.float32, device=x.device)
        if flat.shape[0]:
            cast(Any, sum_forward)[(flat.shape[0],)](
                flat, out, flat.shape[-1], num_warps=4, enable_fp_fusion=False
            )
        ctx.shape, ctx.dtype = x.shape, x.dtype
        return out.reshape(*x.shape[:-1], 1)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (dy,) = grad_outputs
        return dy.expand(ctx.shape).to(ctx.dtype)


def sum_last(x):
    """Bounded-storage fixed row reduction for head statistics fallbacks."""
    if x.device.type == "cuda" and x.dtype != torch.float64:
        return _Sum.apply(x)
    return fixed_sum(x)
