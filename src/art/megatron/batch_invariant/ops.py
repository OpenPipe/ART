"""Differentiable row-invariant primitives, including a CPU contract oracle."""

import torch
import torch.nn.functional as F

POLICY_VERSION = "fixed-tree-rows-v1"
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
    from .kernels import mm_kernel

    out = a.new_empty((a.shape[0], b.shape[1]))
    if out.numel():
        mm_kernel[((a.shape[0] + 127) // 128, (b.shape[1] + 63) // 64)](
            a,
            b,
            out,
            a.shape[0],
            b.shape[1],
            a.shape[1],
            *a.stride(),
            *b.stride(),
            num_warps=4,
            num_stages=2,
        )
    return out


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
            _mm(a.T, dy) if ctx.needs_input_grad[1] else None,
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
            rms_forward[(flat.shape[0],)](
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
                rms_backward[(x.shape[0],)](
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
                rms_weight_backward[((width + 31) // 32,)](
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


def grouped_lora(x, a, b, counts, *, scale=1.0):
    """Expert/adapter rows stay local; no grouped shape-dependent GEMM choice."""
    if isinstance(counts, torch.Tensor):
        if counts.device.type != "cpu":
            raise ValueError("expert counts must be prepared on CPU")
        counts = counts.tolist()
    if sum(counts) != x.shape[0]:
        raise ValueError("expert counts do not cover the input rows")
    out = [
        matmul(
            matmul(part, a[0 if a.shape[0] == 1 else i]), b[0 if b.shape[0] == 1 else i]
        )
        for i, part in enumerate(x.split(counts))
    ]
    return torch.cat(out) * scale


class _LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        from .kernels import lse_forward

        flat = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.empty((flat.shape[0], 1), dtype=torch.float32, device=x.device)
        if flat.shape[0]:
            lse_forward[(flat.shape[0],)](
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
            sum_forward[(flat.shape[0],)](
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
