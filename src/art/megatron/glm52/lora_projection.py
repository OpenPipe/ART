from typing import Any, cast

import torch
import triton
import triton.language as tl

_MAX_RANK = 512


@triton.jit
def _rank_one_kernel(
    x,
    a,
    out,
    m,
    k: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    acc = tl.zeros((block_m,), tl.float32)
    for k_start in range(0, k, block_k):
        inner = k_start + tl.arange(0, block_k)
        x_tile = tl.load(
            x + rows[:, None] * k + inner[None, :],
            mask=(rows[:, None] < m) & (inner[None, :] < k),
            other=0.0,
        ).to(tl.float32)
        a_tile = tl.load(a + inner, mask=inner < k, other=0.0).to(tl.float32)
        acc += tl.sum(x_tile * a_tile[None, :], axis=1)
    tl.store(out + rows, acc, mask=rows < m)


@triton.jit
def _matrix_kernel(
    x,
    a,
    out,
    m,
    k: tl.constexpr,
    n: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    acc = tl.zeros((block_m, block_n), tl.float32)
    for k_start in range(0, k, block_k):
        inner = k_start + tl.arange(0, block_k)
        x_tile = tl.load(
            x + rows[:, None] * k + inner[None, :],
            mask=(rows[:, None] < m) & (inner[None, :] < k),
            other=0.0,
        ).to(tl.float32)
        a_tile = tl.load(
            a + inner[:, None] * n + cols[None, :],
            mask=(inner[:, None] < k) & (cols[None, :] < n),
            other=0.0,
        ).to(tl.float32)
        acc = tl.dot(x_tile, a_tile, acc, input_precision="tf32x3")
    tl.store(
        out + rows[:, None] * n + cols[None, :],
        acc,
        mask=(rows[:, None] < m) & (cols[None, :] < n),
    )


def _validate(x: torch.Tensor, a: torch.Tensor) -> None:
    if not x.is_cuda or not a.is_cuda or x.device != a.device:
        raise ValueError("GLM-5.2 LoRA projection requires tensors on one CUDA device.")
    if x.dtype != torch.bfloat16 or a.dtype != torch.bfloat16:
        raise ValueError("GLM-5.2 LoRA projection requires BF16 tensors.")
    if x.ndim < 2 or a.ndim != 2 or x.shape[-1] != a.shape[0]:
        raise ValueError(
            f"GLM-5.2 LoRA projection shape mismatch: x={tuple(x.shape)}, "
            f"A_T={tuple(a.shape)}."
        )
    if not x.is_contiguous() or not a.is_contiguous():
        raise ValueError("GLM-5.2 LoRA projection requires contiguous tensors.")
    if not 1 <= a.shape[1] <= _MAX_RANK:
        raise ValueError(
            f"GLM-5.2 LoRA rank must be in [1, {_MAX_RANK}], got {a.shape[1]}."
        )


def _forward(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    _validate(x, a)
    x_2d = x.view(-1, x.shape[-1])
    m, k = x_2d.shape
    n = a.shape[1]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out.view(*x.shape[:-1], n)
    if n == 1:
        _rank_one_kernel[(triton.cdiv(m, 8),)](
            x_2d,
            a,
            out,
            m,
            k=k,  # ty: ignore[invalid-argument-type]
            block_m=8,  # ty: ignore[invalid-argument-type]
            block_k=512,  # ty: ignore[invalid-argument-type]
            num_warps=4,  # ty: ignore[unknown-argument]
            num_stages=1,  # ty: ignore[unknown-argument]
        )
    else:
        block_n = 16 if n <= 16 else 32
        _matrix_kernel[(triton.cdiv(m, 64), triton.cdiv(n, block_n))](
            x_2d,
            a,
            out,
            m,
            k=k,  # ty: ignore[invalid-argument-type]
            n=n,  # ty: ignore[invalid-argument-type]
            block_m=64,  # ty: ignore[invalid-argument-type]
            block_k=64,  # ty: ignore[invalid-argument-type]
            block_n=block_n,  # ty: ignore[invalid-argument-type]
            num_warps=4,  # ty: ignore[unknown-argument]
            num_stages=3,  # ty: ignore[unknown-argument]
        )
    return out.view(*x.shape[:-1], n)


class _Glm52LoraA(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, a)
        return _forward(x, a)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        x, a = ctx.saved_tensors
        grad_out = cast(torch.Tensor, grad_outputs[0])
        grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x = grad_a = None
        if ctx.needs_input_grad[0]:
            grad_x = (grad_2d @ a.T).view_as(x)
        if ctx.needs_input_grad[1]:
            grad_a = x.view(-1, x.shape[-1]).T @ grad_2d
        return grad_x, grad_a


def glm52_lora_a(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return _Glm52LoraA.apply(x, a)
