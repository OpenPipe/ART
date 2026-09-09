# ty: ignore[invalid-argument-type, invalid-method-override, unknown-argument]

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _gather_scan_inputs_kernel(
    convolved,
    dt,
    positions,
    output,
    rows,
    conv_width: tl.constexpr,
    dt_width: tl.constexpr,
    conv_stride: tl.constexpr,
    dt_stride: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    position = tl.load(positions + n, mask=n < rows, other=-1).to(tl.int64)
    valid = (n[:, None] < rows) & (position[:, None] >= 0)
    conv_feature = d[None, :] < conv_width
    conv_value = tl.load(
        convolved + position[:, None] * conv_stride + d[None, :].to(tl.int64),
        mask=valid & conv_feature,
        other=0.0,
    )
    dt_feature = d[None, :] - conv_width
    dt_value = tl.load(
        dt + position[:, None] * dt_stride + dt_feature.to(tl.int64),
        mask=valid & ~conv_feature & (dt_feature < dt_width),
        other=0.0,
    )
    value = tl.where(conv_feature, conv_value, dt_value)
    value = tl.where(valid, value, tl.where(conv_feature, 0.0, -float("inf")))
    width: tl.constexpr = conv_width + dt_width
    tl.store(
        output + n[:, None].to(tl.int64) * width + d[None, :].to(tl.int64),
        value,
        mask=(n[:, None] < rows) & (d[None, :] < width),
    )


@triton.jit
def _reduce_scan_input_grad_kernel(
    grad,
    occurrences,
    grad_convolved,
    grad_dt,
    tokens,
    conv_width: tl.constexpr,
    dt_width: tl.constexpr,
    max_occurrences: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    width: tl.constexpr = conv_width + dt_width
    total = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    for slot in range(max_occurrences):
        row = tl.load(
            occurrences + n * max_occurrences + slot,
            mask=n < tokens,
            other=-1,
        ).to(tl.int64)
        total += tl.load(
            grad + row[:, None] * width + d[None, :].to(tl.int64),
            mask=(row[:, None] >= 0) & (d[None, :] < width),
            other=0.0,
        ).to(tl.float32)
    valid = n[:, None] < tokens
    conv_feature = d[None, :] < conv_width
    tl.store(
        grad_convolved + n[:, None].to(tl.int64) * conv_width + d[None, :].to(tl.int64),
        total,
        mask=valid & conv_feature,
    )
    dt_feature = d[None, :] - conv_width
    tl.store(
        grad_dt + n[:, None].to(tl.int64) * dt_width + dt_feature.to(tl.int64),
        total,
        mask=valid & ~conv_feature & (dt_feature < dt_width),
    )


@triton.jit(do_not_specialize=["rows"])
def _scatter_rows_kernel(
    source,
    source_rows,
    destinations,
    output,
    rows,
    width: tl.constexpr,
    source_stride: tl.constexpr,
    IDENTITY_SOURCE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    source_row = n.to(tl.int64)
    if not IDENTITY_SOURCE:
        source_row = tl.load(source_rows + n, mask=n < rows, other=0).to(tl.int64)
    destination = tl.load(destinations + n, mask=n < rows, other=0).to(tl.int64)
    mask = (n[:, None] < rows) & (d[None, :] < width)
    value = tl.load(
        source + source_row[:, None] * source_stride + d[None, :].to(tl.int64),
        mask=mask,
    )
    tl.store(
        output + destination[:, None] * width + d[None, :].to(tl.int64),
        value,
        mask=mask,
    )


@triton.jit(do_not_specialize=["rows"])
def _gather_rows_grad_kernel(
    grad,
    destinations,
    source_rows,
    output,
    rows,
    width: tl.constexpr,
    IDENTITY_SOURCE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    destination = tl.load(destinations + n, mask=n < rows, other=0).to(tl.int64)
    source_row = n.to(tl.int64)
    if not IDENTITY_SOURCE:
        source_row = tl.load(source_rows + n, mask=n < rows, other=0).to(tl.int64)
    mask = (n[:, None] < rows) & (d[None, :] < width)
    value = tl.load(
        grad + destination[:, None] * width + d[None, :].to(tl.int64),
        mask=mask,
    )
    tl.store(
        output + source_row[:, None] * width + d[None, :].to(tl.int64),
        value,
        mask=mask,
    )


_BLOCK_N = 8
_BLOCK_D = 256


class _GatherScanInputs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        convolved: torch.Tensor,
        dt: torch.Tensor,
        positions: torch.Tensor,
        occurrences: torch.Tensor,
    ) -> torch.Tensor:
        rows = int(positions.numel())
        conv_width = int(convolved.shape[1])
        dt_width = int(dt.shape[1])
        width = conv_width + dt_width
        output = convolved.new_empty((rows, width))
        _gather_scan_inputs_kernel[
            (triton.cdiv(rows, _BLOCK_N), triton.cdiv(width, _BLOCK_D))
        ](
            convolved,
            dt,
            positions,
            output,
            rows,
            conv_width,
            dt_width,
            convolved.stride(0),
            dt.stride(0),
            _BLOCK_N,
            _BLOCK_D,
            num_warps=8,
        )
        ctx.save_for_backward(occurrences)
        ctx.geometry = (int(convolved.shape[0]), conv_width, dt_width)
        return output

    @staticmethod
    def backward(
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        (occurrences,) = ctx.saved_tensors
        tokens, conv_width, dt_width = ctx.geometry
        grad = grad.contiguous()
        grad_convolved = grad.new_empty((tokens, conv_width))
        grad_dt = grad.new_empty((tokens, dt_width))
        _reduce_scan_input_grad_kernel[
            (
                triton.cdiv(tokens, _BLOCK_N),
                triton.cdiv(conv_width + dt_width, _BLOCK_D),
            )
        ](
            grad,
            occurrences,
            grad_convolved,
            grad_dt,
            tokens,
            conv_width,
            dt_width,
            int(occurrences.shape[1]),
            _BLOCK_N,
            _BLOCK_D,
            num_warps=8,
        )
        return grad_convolved, grad_dt, None, None


class _AssembleScanOutputs(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, token_count: int, *args: torch.Tensor) -> torch.Tensor:
        if not args or len(args) % 3:
            raise ValueError(
                "scan output assembly requires output/row/position triples"
            )
        outputs = args[0::3]
        source_rows = args[1::3]
        destinations = args[2::3]
        width = int(outputs[0].shape[-1])
        result = outputs[0].new_empty((token_count, width))
        for output, rows, positions in zip(
            outputs, source_rows, destinations, strict=True
        ):
            count = int(rows.numel())
            _scatter_rows_kernel[
                (triton.cdiv(count, _BLOCK_N), triton.cdiv(width, _BLOCK_D))
            ](
                output,
                rows,
                positions,
                result,
                count,
                width,
                output.stride(-2),
                False,
                _BLOCK_N,
                _BLOCK_D,
                num_warps=8,
            )
        ctx.save_for_backward(*source_rows, *destinations)
        ctx.shapes = tuple(tuple(output.shape) for output in outputs)
        ctx.width = width
        return result

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
        count = len(ctx.shapes)
        source_rows = ctx.saved_tensors[:count]
        destinations = ctx.saved_tensors[count:]
        result: list[Any] = [None]
        for shape, rows, positions in zip(
            ctx.shapes, source_rows, destinations, strict=True
        ):
            output = grad.new_zeros(shape)
            row_count = int(rows.numel())
            _gather_rows_grad_kernel[
                (triton.cdiv(row_count, _BLOCK_N), triton.cdiv(ctx.width, _BLOCK_D))
            ](
                grad,
                positions,
                rows,
                output,
                row_count,
                ctx.width,
                False,
                _BLOCK_N,
                _BLOCK_D,
                num_warps=8,
            )
            result.extend((output, None, None))
        return tuple(result)


class _AssembleRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, token_count: int, *args: torch.Tensor) -> torch.Tensor:
        outputs = args[0::2]
        destinations = args[1::2]
        width = int(outputs[0].shape[-1])
        result = outputs[0].new_empty((token_count, width))
        for output, positions in zip(outputs, destinations, strict=True):
            rows = int(positions.numel())
            _scatter_rows_kernel[
                (triton.cdiv(rows, _BLOCK_N), triton.cdiv(width, _BLOCK_D))
            ](
                output,
                positions,
                positions,
                result,
                rows,
                width,
                output.stride(0),
                True,
                _BLOCK_N,
                _BLOCK_D,
                num_warps=8,
            )
        ctx.save_for_backward(*destinations)
        ctx.shapes = tuple(tuple(output.shape) for output in outputs)
        ctx.width = width
        return result

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
        result: list[Any] = [None]
        for shape, positions in zip(ctx.shapes, ctx.saved_tensors, strict=True):
            output = grad.new_empty(shape)
            rows = int(positions.numel())
            _gather_rows_grad_kernel[
                (triton.cdiv(rows, _BLOCK_N), triton.cdiv(ctx.width, _BLOCK_D))
            ](
                grad,
                positions,
                positions,
                output,
                rows,
                ctx.width,
                True,
                _BLOCK_N,
                _BLOCK_D,
                num_warps=8,
            )
            result.extend((output, None))
        return tuple(result)


def gather_scan_inputs(
    convolved: torch.Tensor,
    dt: torch.Tensor,
    positions: torch.Tensor,
    occurrences: torch.Tensor,
) -> torch.Tensor:
    return _GatherScanInputs.apply(convolved, dt, positions, occurrences)


def assemble_rows(
    outputs: list[torch.Tensor],
    destinations: list[torch.Tensor],
    token_count: int,
) -> torch.Tensor:
    args = tuple(
        item for pair in zip(outputs, destinations, strict=True) for item in pair
    )
    return _AssembleRows.apply(token_count, *args)


def assemble_scan_outputs(
    outputs: list[torch.Tensor],
    output_rows: list[torch.Tensor],
    output_positions: list[torch.Tensor],
    token_count: int,
) -> torch.Tensor:
    args = tuple(
        item
        for triple in zip(outputs, output_rows, output_positions, strict=True)
        for item in triple
    )
    return _AssembleScanOutputs.apply(token_count, *args)
