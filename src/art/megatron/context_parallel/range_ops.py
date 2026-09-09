from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from .types import TokenRange

_WARMED_HEAD_MAJOR_KERNELS: set[tuple[str, int | None, torch.dtype, int]] = set()


def _single_range(ranges: Sequence[TokenRange]) -> TokenRange | None:
    compact = [range_ for range_ in ranges if range_.size() > 0]
    if len(compact) != 1:
        return None
    return compact[0]


@triton.jit
def _range_gather_per_row_kernel(
    input_ptr,
    output_ptr,
    ranges_ptr,
    cu_range_sizes_ptr,
    row_map_ptr,
    input_stride,
    output_stride,
    n_cols: tl.constexpr,
    n_col_blocks: tl.constexpr,
    elem_per_block: tl.constexpr,
):
    out_row = tl.program_id(0)
    block_idx = tl.program_id(1)

    range_idx = tl.load(row_map_ptr + out_row)
    range_base = tl.load(cu_range_sizes_ptr + range_idx)
    range_row = out_row - range_base
    input_row = tl.load(ranges_ptr + range_idx * 2) + range_row

    cols = block_idx * elem_per_block + tl.arange(0, elem_per_block)
    mask = cols < n_cols

    input_offsets = input_row * input_stride + cols
    output_offsets = out_row * output_stride + cols

    values = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + output_offsets, values, mask=mask)


@triton.jit
def _range_reduce_sum_per_row_kernel(
    input_ptr,
    output_ptr,
    ranges_ptr,
    cu_range_sizes_ptr,
    row_map_ptr,
    input_stride,
    output_stride,
    n_cols: tl.constexpr,
    n_col_blocks: tl.constexpr,
    elem_per_block: tl.constexpr,
):
    in_row = tl.program_id(0)
    block_idx = tl.program_id(1)

    range_idx = tl.load(row_map_ptr + in_row)
    range_base = tl.load(cu_range_sizes_ptr + range_idx)
    range_row = in_row - range_base
    output_row = tl.load(ranges_ptr + range_idx * 2) + range_row

    cols = block_idx * elem_per_block + tl.arange(0, elem_per_block)
    mask = cols < n_cols

    input_offsets = in_row * input_stride + cols
    output_offsets = output_row * output_stride + cols

    update = tl.load(input_ptr + input_offsets, mask=mask)
    tl.atomic_add(output_ptr + output_offsets, update, mask=mask)


@triton.jit
def _range_gather_head_major_kernel(
    input_ptr,
    output_ptr,
    ranges_ptr,
    cu_range_sizes_ptr,
    row_map_ptr,
    input_head_stride,
    input_token_stride,
    output_head_stride,
    output_token_stride,
    inner_size: tl.constexpr,
    n_cols,
    elem_per_block: tl.constexpr,
):
    out_row = tl.program_id(0)
    block_idx = tl.program_id(1)

    range_idx = tl.load(row_map_ptr + out_row)
    range_base = tl.load(cu_range_sizes_ptr + range_idx)
    range_row = out_row - range_base
    input_row = tl.load(ranges_ptr + range_idx * 2) + range_row

    cols = block_idx * elem_per_block + tl.arange(0, elem_per_block)
    mask = cols < n_cols
    head_idx = cols // inner_size
    inner_idx = cols % inner_size

    input_offsets = (
        head_idx * input_head_stride + input_row * input_token_stride + inner_idx
    )
    output_offsets = (
        head_idx * output_head_stride + out_row * output_token_stride + inner_idx
    )
    values = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + output_offsets, values, mask=mask)


@triton.jit
def _range_reduce_sum_head_major_kernel(
    input_ptr,
    output_ptr,
    ranges_ptr,
    cu_range_sizes_ptr,
    row_map_ptr,
    input_head_stride,
    input_token_stride,
    output_head_stride,
    output_token_stride,
    inner_size: tl.constexpr,
    n_cols,
    elem_per_block: tl.constexpr,
):
    in_row = tl.program_id(0)
    block_idx = tl.program_id(1)

    range_idx = tl.load(row_map_ptr + in_row)
    range_base = tl.load(cu_range_sizes_ptr + range_idx)
    range_row = in_row - range_base
    output_row = tl.load(ranges_ptr + range_idx * 2) + range_row

    cols = block_idx * elem_per_block + tl.arange(0, elem_per_block)
    mask = cols < n_cols
    head_idx = cols // inner_size
    inner_idx = cols % inner_size

    input_offsets = (
        head_idx * input_head_stride + in_row * input_token_stride + inner_idx
    )
    output_offsets = (
        head_idx * output_head_stride + output_row * output_token_stride + inner_idx
    )
    update = tl.load(input_ptr + input_offsets, mask=mask)
    tl.atomic_add(output_ptr + output_offsets, update, mask=mask)


def _range_key(ranges: Sequence[TokenRange]) -> tuple[tuple[int, int], ...]:
    return tuple((range_.start, range_.end) for range_ in ranges if range_.size() > 0)


def _range_meta(
    ranges: Sequence[TokenRange],
    *,
    device: torch.device,
    range_meta_cache: dict[
        tuple[tuple[tuple[int, int], ...], str, int | None],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    ]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    key = (_range_key(ranges), device.type, device.index)
    if range_meta_cache is not None:
        cached = range_meta_cache.get(key)
        if cached is not None:
            return cached

    compact = [range_ for range_ in ranges if range_.size() > 0]
    if not compact:
        empty_i64 = torch.empty((0,), device=device, dtype=torch.int64)
        empty_ranges = torch.empty((0, 2), device=device, dtype=torch.int64)
        cached = (empty_ranges, empty_i64, empty_i64, 0)
        if range_meta_cache is not None:
            range_meta_cache[key] = cached
        return cached

    ranges_tensor = torch.tensor(
        [(range_.start, range_.end) for range_ in compact],
        device=device,
        dtype=torch.int64,
    )
    range_sizes = torch.tensor(
        [0, *[range_.size() for range_ in compact]],
        device=device,
        dtype=torch.int64,
    )
    cu_range_sizes = torch.cumsum(range_sizes, dim=0)
    total_size = sum(range_.size() for range_ in compact)
    row_map = torch.repeat_interleave(
        torch.arange(len(compact), device=device, dtype=torch.int64),
        range_sizes[1:],
        output_size=total_size,
    )
    cached = (ranges_tensor, cu_range_sizes, row_map, total_size)
    if range_meta_cache is not None:
        range_meta_cache[key] = cached
    return cached


def _python_range_gather(
    input_tensor: torch.Tensor,
    ranges: Sequence[TokenRange],
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [
        input_tensor[range_.start : range_.end]
        for range_ in ranges
        if range_.size() > 0
    ]
    if output is None:
        if not parts:
            return input_tensor.new_empty((0, *input_tensor.shape[1:]))
        if len(parts) == 1:
            return parts[0].contiguous()
        return torch.cat(parts, dim=0).contiguous()
    if not parts:
        return output
    cursor = 0
    for part in parts:
        next_cursor = cursor + int(part.shape[0])
        output[cursor:next_cursor].copy_(part)
        cursor = next_cursor
    return output


def _python_range_gather_head_major(
    input_tensor: torch.Tensor,
    ranges: Sequence[TokenRange],
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [
        input_tensor[:, range_.start : range_.end]
        for range_ in ranges
        if range_.size() > 0
    ]
    if output is None:
        if not parts:
            return input_tensor.new_empty(
                (input_tensor.shape[0], 0, *input_tensor.shape[2:])
            )
        if len(parts) == 1:
            return parts[0].contiguous()
        return torch.cat(parts, dim=1).contiguous()
    if not parts:
        return output
    cursor = 0
    for part in parts:
        next_cursor = cursor + int(part.shape[1])
        output[:, cursor:next_cursor].copy_(part)
        cursor = next_cursor
    return output


def _range_gather_impl(
    input_tensor: torch.Tensor,
    ranges: Sequence[TokenRange],
    *,
    output: torch.Tensor | None = None,
    range_meta_cache: dict[
        tuple[tuple[tuple[int, int], ...], str, int | None],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    ]
    | None = None,
) -> torch.Tensor:
    if input_tensor.ndim < 1:
        raise RuntimeError(
            f"Expected tensor with dim>=1, got {tuple(input_tensor.shape)}"
        )
    if not ranges:
        if output is not None:
            return output
        return input_tensor.new_empty((0, *input_tensor.shape[1:]))
    if not input_tensor.is_cuda:
        return _python_range_gather(input_tensor, ranges, output=output)

    ranges_tensor, cu_range_sizes, row_map, total_size = _range_meta(
        ranges,
        device=input_tensor.device,
        range_meta_cache=range_meta_cache,
    )
    if output is None:
        output = input_tensor.new_empty((total_size, *input_tensor.shape[1:]))
    else:
        if int(output.shape[0]) != total_size:
            raise RuntimeError(
                f"range_gather output has wrong first dim: expected {total_size}, got {int(output.shape[0])}"
            )
        output = output.contiguous()
    if total_size == 0 or input_tensor.numel() == 0:
        return output

    n_cols = input_tensor.numel() // max(int(input_tensor.shape[0]), 1)
    elem_per_block = max(1, 2048 // input_tensor.element_size())
    n_col_blocks = triton.cdiv(n_cols, elem_per_block)
    _range_gather_per_row_kernel[(total_size, n_col_blocks)](
        input_tensor,
        output,
        ranges_tensor,
        cu_range_sizes,
        row_map,
        input_tensor.stride(0),
        output.stride(0),
        n_cols=n_cols,  # ty: ignore[invalid-argument-type]
        n_col_blocks=n_col_blocks,
        elem_per_block=elem_per_block,  # ty: ignore[invalid-argument-type]
        num_warps=4,  # ty: ignore[unknown-argument]
    )
    return output


def _range_gather_head_major_impl(
    input_tensor: torch.Tensor,
    ranges: Sequence[TokenRange],
    *,
    output: torch.Tensor | None = None,
    range_meta_cache: dict[
        tuple[tuple[tuple[int, int], ...], str, int | None],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    ]
    | None = None,
) -> torch.Tensor:
    if input_tensor.ndim < 2:
        raise RuntimeError(
            f"Expected tensor with dim>=2, got {tuple(input_tensor.shape)}"
        )
    if not ranges:
        if output is not None:
            return output
        return input_tensor.new_empty(
            (input_tensor.shape[0], 0, *input_tensor.shape[2:])
        )
    if not input_tensor.is_cuda:
        return _python_range_gather_head_major(input_tensor, ranges, output=output)

    input_tensor = input_tensor.contiguous()
    ranges_tensor, cu_range_sizes, row_map, total_size = _range_meta(
        ranges,
        device=input_tensor.device,
        range_meta_cache=range_meta_cache,
    )
    if output is None:
        output = input_tensor.new_empty(
            (input_tensor.shape[0], total_size, *input_tensor.shape[2:])
        )
    else:
        if int(output.shape[1]) != total_size:
            raise RuntimeError(
                "range_gather_head_major output has wrong token dim: "
                f"expected {total_size}, got {int(output.shape[1])}"
            )
        output = output.contiguous()
    if total_size == 0 or input_tensor.numel() == 0:
        return output

    inner_size = input_tensor.numel() // max(
        int(input_tensor.shape[0] * input_tensor.shape[1]), 1
    )
    n_cols = int(input_tensor.shape[0]) * inner_size
    elem_per_block = max(1, 2048 // input_tensor.element_size())
    n_col_blocks = triton.cdiv(n_cols, elem_per_block)
    _range_gather_head_major_kernel[(total_size, n_col_blocks)](
        input_tensor,
        output,
        ranges_tensor,
        cu_range_sizes,
        row_map,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output.stride(0),
        output.stride(1),
        inner_size=inner_size,  # ty: ignore[invalid-argument-type]
        n_cols=n_cols,
        elem_per_block=elem_per_block,  # ty: ignore[invalid-argument-type]
        num_warps=4,  # ty: ignore[unknown-argument]
    )
    return output


def prepare_range_head_major_kernels(input_tensor: torch.Tensor) -> None:
    """Compile one shape-polymorphic gather/reduce pair before variable CP plans.

    Prefix trees may not require multi-range Q/KV movement until a late batch.
    Warm once per device, dtype, and inner dimension so that workload variation
    cannot expose compilation in a later optimizer step. This enqueues two tiny
    kernels on the current stream and must remain free of CUDA synchronization.
    """
    if not input_tensor.is_cuda:
        return
    if input_tensor.ndim < 3:
        raise RuntimeError(
            "Head-major range kernel preparation expects [H, T, ...], "
            f"got {tuple(input_tensor.shape)}"
        )
    inner_size = input_tensor.numel() // max(
        int(input_tensor.shape[0] * input_tensor.shape[1]), 1
    )
    key = (
        input_tensor.device.type,
        input_tensor.device.index,
        input_tensor.dtype,
        inner_size,
    )
    if key in _WARMED_HEAD_MAJOR_KERNELS:
        return
    probe = input_tensor.new_empty((input_tensor.shape[0], 2, *input_tensor.shape[2:]))
    ranges = (TokenRange(start=0, end=1), TokenRange(start=1, end=2))
    gathered = _range_gather_head_major_impl(probe, ranges)
    range_reduce_sum_head_major_(gathered, output_tensor=probe, ranges=ranges)
    _WARMED_HEAD_MAJOR_KERNELS.add(key)


class _RangeGatherFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        ranges: tuple[TokenRange, ...],
        range_meta_cache: dict[
            tuple[tuple[tuple[int, int], ...], str, int | None],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
        ]
        | None,
    ) -> torch.Tensor:
        ctx.ranges = ranges
        ctx.input_shape = tuple(input_tensor.shape)
        ctx.range_meta_cache = range_meta_cache
        return _range_gather_impl(
            input_tensor,
            ranges,
            range_meta_cache=range_meta_cache,
        )

    @staticmethod
    def backward(
        ctx, *grad_outputs: torch.Tensor | None
    ) -> tuple[torch.Tensor, None, None]:
        grad_output = grad_outputs[0]
        if grad_output is None:
            raise RuntimeError("_RangeGatherFn.backward expected one grad output")
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_zeros(ctx.input_shape)
        range_reduce_sum_(
            grad_output,
            output_tensor=grad_input,
            ranges=ctx.ranges,
            range_meta_cache=ctx.range_meta_cache,
        )
        return grad_input, None, None


class _RangeGatherHeadMajorFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        ranges: tuple[TokenRange, ...],
        range_meta_cache: dict[
            tuple[tuple[tuple[int, int], ...], str, int | None],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
        ]
        | None,
    ) -> torch.Tensor:
        ctx.ranges = ranges
        ctx.input_shape = tuple(input_tensor.shape)
        ctx.range_meta_cache = range_meta_cache
        return _range_gather_head_major_impl(
            input_tensor,
            ranges,
            range_meta_cache=range_meta_cache,
        )

    @staticmethod
    def backward(
        ctx, *grad_outputs: torch.Tensor | None
    ) -> tuple[torch.Tensor, None, None]:
        grad_output = grad_outputs[0]
        if grad_output is None:
            raise RuntimeError(
                "_RangeGatherHeadMajorFn.backward expected one grad output"
            )
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_zeros(ctx.input_shape)
        range_reduce_sum_head_major_(
            grad_output,
            output_tensor=grad_input,
            ranges=ctx.ranges,
            range_meta_cache=ctx.range_meta_cache,
        )
        return grad_input, None, None


def range_gather(
    input_tensor: torch.Tensor,
    ranges: Sequence[TokenRange],
    *,
    output: torch.Tensor | None = None,
    range_meta_cache: dict[
        tuple[tuple[tuple[int, int], ...], str, int | None],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    ]
    | None = None,
) -> torch.Tensor:
    normalized_ranges = tuple(range_ for range_ in ranges if range_.size() > 0)
    single_range = _single_range(normalized_ranges)
    if single_range is not None:
        gathered = input_tensor[single_range.start : single_range.end]
        if output is None:
            return gathered.contiguous()
        output.copy_(gathered)
        return output
    if output is not None:
        return _range_gather_impl(
            input_tensor,
            normalized_ranges,
            output=output,
            range_meta_cache=range_meta_cache,
        )
    if input_tensor.requires_grad:
        return _RangeGatherFn.apply(input_tensor, normalized_ranges, range_meta_cache)
    return _range_gather_impl(
        input_tensor,
        normalized_ranges,
        range_meta_cache=range_meta_cache,
    )


def range_gather_head_major(
    input_tensor: torch.Tensor,
    ranges: Sequence[TokenRange],
    *,
    output: torch.Tensor | None = None,
    range_meta_cache: dict[
        tuple[tuple[tuple[int, int], ...], str, int | None],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    ]
    | None = None,
) -> torch.Tensor:
    normalized_ranges = tuple(range_ for range_ in ranges if range_.size() > 0)
    single_range = _single_range(normalized_ranges)
    if single_range is not None:
        gathered = input_tensor[:, single_range.start : single_range.end]
        if output is None:
            return gathered.contiguous()
        output.copy_(gathered)
        return output
    if output is not None:
        return _range_gather_head_major_impl(
            input_tensor,
            normalized_ranges,
            output=output,
            range_meta_cache=range_meta_cache,
        )
    if input_tensor.requires_grad:
        return _RangeGatherHeadMajorFn.apply(
            input_tensor,
            normalized_ranges,
            range_meta_cache,
        )
    return _range_gather_head_major_impl(
        input_tensor,
        normalized_ranges,
        range_meta_cache=range_meta_cache,
    )


def range_reduce_sum_(
    input_tensor: torch.Tensor,
    *,
    output_tensor: torch.Tensor,
    ranges: Sequence[TokenRange],
    range_meta_cache: dict[
        tuple[tuple[tuple[int, int], ...], str, int | None],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    ]
    | None = None,
) -> torch.Tensor:
    expected_rows = sum(range_.size() for range_ in ranges)
    if int(input_tensor.shape[0]) != expected_rows:
        raise RuntimeError(
            "range_reduce_sum_ consumed the wrong number of rows: "
            f"consumed={int(input_tensor.shape[0])}, expected={expected_rows}"
        )
    if expected_rows == 0:
        return output_tensor
    single_range = _single_range(ranges)
    if single_range is not None:
        updated = output_tensor[single_range.start : single_range.end] + input_tensor
        output_tensor[single_range.start : single_range.end].copy_(updated)
        return output_tensor
    if not input_tensor.is_cuda or not output_tensor.is_cuda:
        cursor = 0
        for range_ in ranges:
            size = range_.size()
            if size <= 0:
                continue
            output_tensor[range_.start : range_.end].add_(
                input_tensor[cursor : cursor + size]
            )
            cursor += size
        return output_tensor

    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()
    ranges_tensor, cu_range_sizes, row_map, total_size = _range_meta(
        ranges,
        device=input_tensor.device,
        range_meta_cache=range_meta_cache,
    )
    if total_size != expected_rows:
        raise RuntimeError(
            f"range_reduce_sum_ range metadata mismatch: expected {expected_rows}, got {total_size}"
        )
    n_cols = input_tensor.numel() // max(int(input_tensor.shape[0]), 1)
    elem_per_block = max(1, 2048 // input_tensor.element_size())
    n_col_blocks = triton.cdiv(n_cols, elem_per_block)
    _range_reduce_sum_per_row_kernel[(total_size, n_col_blocks)](
        input_tensor,
        output_tensor,
        ranges_tensor,
        cu_range_sizes,
        row_map,
        input_tensor.stride(0),
        output_tensor.stride(0),
        n_cols=n_cols,  # ty: ignore[invalid-argument-type]
        n_col_blocks=n_col_blocks,
        elem_per_block=elem_per_block,  # ty: ignore[invalid-argument-type]
        num_warps=4,  # ty: ignore[unknown-argument]
    )
    return output_tensor


def range_reduce_sum_head_major_(
    input_tensor: torch.Tensor,
    *,
    output_tensor: torch.Tensor,
    ranges: Sequence[TokenRange],
    range_meta_cache: dict[
        tuple[tuple[tuple[int, int], ...], str, int | None],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    ]
    | None = None,
) -> torch.Tensor:
    expected_rows = sum(range_.size() for range_ in ranges)
    if int(input_tensor.shape[1]) != expected_rows:
        raise RuntimeError(
            "range_reduce_sum_head_major_ consumed the wrong number of rows: "
            f"consumed={int(input_tensor.shape[1])}, expected={expected_rows}"
        )
    if expected_rows == 0:
        return output_tensor
    single_range = _single_range(ranges)
    if single_range is not None:
        updated = output_tensor[:, single_range.start : single_range.end] + input_tensor
        output_tensor[:, single_range.start : single_range.end].copy_(updated)
        return output_tensor
    if not input_tensor.is_cuda or not output_tensor.is_cuda:
        cursor = 0
        for range_ in ranges:
            size = range_.size()
            if size <= 0:
                continue
            output_tensor[:, range_.start : range_.end].add_(
                input_tensor[:, cursor : cursor + size]
            )
            cursor += size
        return output_tensor

    input_tensor = input_tensor.contiguous()
    output_tensor = output_tensor.contiguous()
    ranges_tensor, cu_range_sizes, row_map, total_size = _range_meta(
        ranges,
        device=input_tensor.device,
        range_meta_cache=range_meta_cache,
    )
    if total_size != expected_rows:
        raise RuntimeError(
            "range_reduce_sum_head_major_ range metadata mismatch: "
            f"expected {expected_rows}, got {total_size}"
        )
    inner_size = input_tensor.numel() // max(
        int(input_tensor.shape[0] * input_tensor.shape[1]), 1
    )
    n_cols = int(input_tensor.shape[0]) * inner_size
    elem_per_block = max(1, 2048 // input_tensor.element_size())
    n_col_blocks = triton.cdiv(n_cols, elem_per_block)
    _range_reduce_sum_head_major_kernel[(total_size, n_col_blocks)](
        input_tensor,
        output_tensor,
        ranges_tensor,
        cu_range_sizes,
        row_map,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor.stride(0),
        output_tensor.stride(1),
        inner_size=inner_size,  # ty: ignore[invalid-argument-type]
        n_cols=n_cols,
        elem_per_block=elem_per_block,  # ty: ignore[invalid-argument-type]
        num_warps=4,  # ty: ignore[unknown-argument]
    )
    return output_tensor
