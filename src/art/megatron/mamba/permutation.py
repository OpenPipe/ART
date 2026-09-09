# ty: ignore[invalid-argument-type, invalid-method-override, unknown-argument, unresolved-attribute]

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _permute_rows_kernel(
    source,
    order,
    output,
    rows,
    width: tl.constexpr,
    source_stride: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    ordered = tl.load(order + n, mask=n < rows, other=0).to(tl.int64)
    source_row = tl.where(INVERSE, n, ordered).to(tl.int64)
    output_row = tl.where(INVERSE, ordered, n).to(tl.int64)
    mask = (n[:, None] < rows) & (d[None, :] < width)
    value = tl.load(
        source + source_row[:, None] * source_stride + d[None, :].to(tl.int64),
        mask=mask,
    )
    tl.store(
        output + output_row[:, None] * width + d[None, :].to(tl.int64),
        value,
        mask=mask,
    )


_BLOCK_N = 8
_BLOCK_D = 256


def _launch(
    source: torch.Tensor, order: torch.Tensor, *, inverse: bool
) -> torch.Tensor:
    rows, width = source.shape
    if source.stride(1) != 1:
        raise ValueError("row permutation requires a contiguous feature dimension")
    output = torch.empty_like(source)
    _permute_rows_kernel[(triton.cdiv(rows, _BLOCK_N), triton.cdiv(width, _BLOCK_D))](
        source,
        order,
        output,
        rows,
        width,
        source.stride(0),
        inverse,
        _BLOCK_N,
        _BLOCK_D,
        num_warps=8,
    )
    return output


class _PermuteRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(order)  # type: ignore[attr-defined]
        return _launch(tensor, order, inverse=False)

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        (order,) = ctx.saved_tensors  # type: ignore[attr-defined]
        return _launch(grad, order, inverse=True), None


def permute_rows(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Permute every row exactly once, with a non-atomic inverse backward."""

    if tensor.shape[0] != order.numel():
        raise ValueError("row permutation must contain every input row")
    return _PermuteRows.apply(tensor, order)
