from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.flex_attn.compiled import normalize_sparse_block_size
from art.megatron.prefix_tree import parse_prefix_tree_row

from .types import AttnMaskKind, FlexMaskSpec

_INVALID_ABS = -(1 << 63)
_INVALID_ENTER = -1
_INVALID_EXIT = -1
_INVALID_POS = -(1 << 63)


@dataclass(frozen=True, slots=True)
class PreparedBlockMaskContext:
    source_len: int
    group_enter_np: np.ndarray
    group_exit_np: np.ndarray
    input_pos_np: np.ndarray | None = None


def _build_interval_mask_mod(
    *,
    q_abs: np.ndarray,
    k_abs: np.ndarray,
    q_enter: np.ndarray,
    k_enter: np.ndarray,
    k_exit: np.ndarray,
    q_pos: np.ndarray | None,
    k_pos: np.ndarray | None,
    sliding_window: int | None,
    device: torch.device,
):
    metadata_dtype = _smallest_mask_metadata_dtype(
        q_abs,
        k_abs,
        q_enter,
        k_enter,
        k_exit,
        *((q_pos, k_pos) if sliding_window is not None else ()),
    )
    q_abs_tensor = torch.as_tensor(q_abs, device=device, dtype=metadata_dtype)
    k_abs_tensor = torch.as_tensor(k_abs, device=device, dtype=metadata_dtype)
    q_enter_tensor = torch.as_tensor(q_enter, device=device, dtype=metadata_dtype)
    k_enter_tensor = torch.as_tensor(k_enter, device=device, dtype=metadata_dtype)
    k_exit_tensor = torch.as_tensor(k_exit, device=device, dtype=metadata_dtype)
    q_pos_tensor_or_none = (
        None
        if q_pos is None
        else torch.as_tensor(q_pos, device=device, dtype=metadata_dtype)
    )
    k_pos_tensor_or_none = (
        None
        if k_pos is None
        else torch.as_tensor(k_pos, device=device, dtype=metadata_dtype)
    )
    if sliding_window is not None and (
        q_pos_tensor_or_none is None or k_pos_tensor_or_none is None
    ):
        raise RuntimeError("Sliding-window prefix-tree masks require input_pos.")
    q_pos_tensor = q_pos_tensor_or_none
    k_pos_tensor = k_pos_tensor_or_none

    def mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del batch_idx, head_idx
        q_abs_local = q_abs_tensor[query_idx]
        k_abs_local = k_abs_tensor[kv_idx]
        q_enter_local = q_enter_tensor[query_idx]
        k_enter_local = k_enter_tensor[kv_idx]
        k_exit_local = k_exit_tensor[kv_idx]
        in_key_subtree = (k_enter_local <= q_enter_local) & (
            q_enter_local < k_exit_local
        )
        allowed = (q_abs_local >= k_abs_local) & in_key_subtree
        if sliding_window is None:
            return allowed
        assert q_pos_tensor is not None and k_pos_tensor is not None
        delta = q_pos_tensor[query_idx] - k_pos_tensor[kv_idx]
        return allowed & (delta >= 0) & (delta < int(sliding_window))

    return mask_mod


def _smallest_mask_metadata_dtype(*arrays: np.ndarray | None) -> torch.dtype:
    int32 = np.iinfo(np.int32)
    for array in arrays:
        if array is None or int(array.size) == 0:
            continue
        if int(array.min()) < int32.min or int(array.max()) > int32.max:
            return torch.int64
    return torch.int32


def _dense_blocks_to_ordered(
    blocks: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_indices, column_indices = np.nonzero(blocks)
    counts_np = np.bincount(row_indices, minlength=blocks.shape[0]).astype(np.int32)
    indices_np = np.zeros(blocks.shape, dtype=np.int32)
    if int(row_indices.size) > 0:
        starts = np.concatenate(([0], np.cumsum(counts_np[:-1], dtype=np.int64)))
        offsets = np.arange(int(row_indices.size), dtype=np.int64) - starts[row_indices]
        indices_np[row_indices, offsets] = column_indices
    counts = torch.from_numpy(counts_np)
    indices = torch.from_numpy(indices_np)
    return (
        counts.view(1, 1, -1).to(device=device),
        indices.view(1, 1, blocks.shape[0], blocks.shape[1]).to(device=device),
    )


def _select_with_invalid_np(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    invalid_value: int,
) -> np.ndarray:
    selected = np.full(indices.shape, invalid_value, dtype=np.int64)
    valid = indices >= 0
    if bool(valid.any()):
        selected[valid] = values[indices[valid]]
    return selected


def _refine_interval_blocks(
    *,
    partial_blocks: np.ndarray,
    full_blocks: np.ndarray,
    q_abs: np.ndarray,
    k_abs: np.ndarray,
    q_enter: np.ndarray,
    k_enter: np.ndarray,
    k_exit: np.ndarray,
    q_block: int,
    k_block: int,
) -> None:
    if not bool((partial_blocks | full_blocks).any()):
        return

    q_abs_blocks = _block_matrix(
        q_abs,
        block_size=q_block,
        block_count=int(partial_blocks.shape[0]),
        fill_value=_INVALID_ABS,
    )
    q_enter_blocks = _block_matrix(
        q_enter,
        block_size=q_block,
        block_count=int(partial_blocks.shape[0]),
        fill_value=_INVALID_ENTER,
    )
    k_abs_blocks = _block_matrix(
        k_abs,
        block_size=k_block,
        block_count=int(partial_blocks.shape[1]),
        fill_value=_INVALID_ABS,
    )
    k_enter_blocks = _block_matrix(
        k_enter,
        block_size=k_block,
        block_count=int(partial_blocks.shape[1]),
        fill_value=_INVALID_ENTER,
    )
    k_exit_blocks = _block_matrix(
        k_exit,
        block_size=k_block,
        block_count=int(partial_blocks.shape[1]),
        fill_value=_INVALID_EXIT,
    )

    q_valid = (q_abs_blocks >= 0) & (q_enter_blocks >= 0)
    k_valid = (
        (k_abs_blocks >= 0) & (k_enter_blocks >= 0) & (k_exit_blocks > k_enter_blocks)
    )
    q_all_valid = q_valid.all(axis=1)
    k_all_valid = k_valid.all(axis=1)
    q_min_abs = np.where(q_valid, q_abs_blocks, np.iinfo(np.int64).max).min(axis=1)
    q_min_enter = np.where(
        q_valid,
        q_enter_blocks,
        np.iinfo(np.int64).max,
    ).min(axis=1)
    q_max_enter = np.where(q_valid, q_enter_blocks, _INVALID_ENTER).max(axis=1)
    k_max_abs = np.where(k_valid, k_abs_blocks, _INVALID_ABS).max(axis=1)
    k_max_enter = np.where(k_valid, k_enter_blocks, _INVALID_ENTER).max(axis=1)
    k_min_exit = np.where(k_valid, k_exit_blocks, np.iinfo(np.int64).max).min(axis=1)
    safe_full = (
        q_all_valid[:, None]
        & k_all_valid[None, :]
        & (q_min_abs[:, None] >= k_max_abs[None, :])
        & (k_max_enter[None, :] <= q_min_enter[:, None])
        & (q_max_enter[:, None] < k_min_exit[None, :])
    )
    candidate_blocks = partial_blocks | (full_blocks & ~safe_full)
    q_indices, k_indices = np.nonzero(candidate_blocks)
    if int(q_indices.size) == 0:
        return

    rows = np.arange(int(k_valid.shape[0]))
    first_valid_offsets = k_valid.argmax(axis=1)
    first_enter = k_enter_blocks[rows, first_valid_offsets]
    first_exit = k_exit_blocks[rows, first_valid_offsets]
    k_single_interval = k_valid.any(axis=1) & (
        (~k_valid)
        | (
            (k_enter_blocks == first_enter[:, None])
            & (k_exit_blocks == first_exit[:, None])
        )
    ).all(axis=1)

    single_pair = k_single_interval[k_indices]
    if bool(single_pair.any()):
        single_q = q_indices[single_pair]
        single_k = k_indices[single_pair]
        q_abs_selected = q_abs_blocks[single_q]
        q_enter_selected = q_enter_blocks[single_q]
        in_subtree = (
            q_valid[single_q]
            & (q_enter_selected >= first_enter[single_k, None])
            & (q_enter_selected < first_exit[single_k, None])
        )
        max_abs_in_subtree = np.where(
            in_subtree,
            q_abs_selected,
            _INVALID_ABS,
        ).max(axis=1)
        k_min_abs = np.where(k_valid, k_abs_blocks, np.iinfo(np.int64).max).min(axis=1)
        has_any = max_abs_in_subtree >= k_min_abs[single_k]

        is_full = (
            has_any
            & q_all_valid[single_q]
            & k_all_valid[single_k]
            & (q_min_abs[single_q] >= k_max_abs[single_k])
            & (first_enter[single_k] <= q_min_enter[single_q])
            & (q_max_enter[single_q] < first_exit[single_k])
        )
        partial_blocks[single_q, single_k] = has_any & ~is_full
        full_blocks[single_q, single_k] = is_full

    intervals_by_k: dict[int, tuple[tuple[int, int, int], ...]] = {}

    def k_intervals(k_idx: int) -> tuple[tuple[int, int, int], ...]:
        cached = intervals_by_k.get(k_idx)
        if cached is not None:
            return cached
        min_abs_by_interval: dict[tuple[int, int], int] = {}
        for abs_value, enter_value, exit_value in zip(
            k_abs_blocks[k_idx, k_valid[k_idx]],
            k_enter_blocks[k_idx, k_valid[k_idx]],
            k_exit_blocks[k_idx, k_valid[k_idx]],
            strict=True,
        ):
            key = (int(enter_value), int(exit_value))
            prior = min_abs_by_interval.get(key)
            min_abs_by_interval[key] = (
                int(abs_value) if prior is None else min(prior, int(abs_value))
            )
        cached = tuple(
            (enter, exit, min_abs)
            for (enter, exit), min_abs in min_abs_by_interval.items()
        )
        intervals_by_k[k_idx] = cached
        return cached

    for q_idx, k_idx in zip(
        q_indices[~single_pair],
        k_indices[~single_pair],
        strict=True,
    ):
        q_valid_row = q_valid[q_idx]
        intervals = k_intervals(int(k_idx))
        has_any = False
        if bool(q_valid_row.any()) and intervals:
            q_abs_row = q_abs_blocks[q_idx]
            q_enter_row = q_enter_blocks[q_idx]
            for enter, exit, min_abs in intervals:
                in_subtree = q_valid_row & (q_enter_row >= enter) & (q_enter_row < exit)
                if bool(in_subtree.any()) and int(q_abs_row[in_subtree].max()) >= int(
                    min_abs
                ):
                    has_any = True
                    break
        is_full = (
            has_any
            and bool(q_all_valid[q_idx])
            and bool(k_all_valid[k_idx])
            and int(q_min_abs[q_idx]) >= int(k_max_abs[k_idx])
            and int(k_max_enter[k_idx]) <= int(q_min_enter[q_idx])
            and int(q_max_enter[q_idx]) < int(k_min_exit[k_idx])
        )
        partial_blocks[q_idx, k_idx] = has_any and not is_full
        full_blocks[q_idx, k_idx] = bool(is_full)


def _refine_sliding_interval_blocks(
    *,
    partial_blocks: np.ndarray,
    full_blocks: np.ndarray,
    q_abs: np.ndarray,
    k_abs: np.ndarray,
    q_enter: np.ndarray,
    k_enter: np.ndarray,
    k_exit: np.ndarray,
    q_pos: np.ndarray,
    k_pos: np.ndarray,
    q_block: int,
    k_block: int,
    sliding_window: int,
) -> None:
    candidates = partial_blocks | full_blocks
    if not bool(candidates.any()):
        return

    q_abs_blocks = _block_matrix(
        q_abs,
        block_size=q_block,
        block_count=int(partial_blocks.shape[0]),
        fill_value=_INVALID_ABS,
    )
    q_enter_blocks = _block_matrix(
        q_enter,
        block_size=q_block,
        block_count=int(partial_blocks.shape[0]),
        fill_value=_INVALID_ENTER,
    )
    q_pos_blocks = _block_matrix(
        q_pos,
        block_size=q_block,
        block_count=int(partial_blocks.shape[0]),
        fill_value=_INVALID_POS,
    )
    k_abs_blocks = _block_matrix(
        k_abs,
        block_size=k_block,
        block_count=int(partial_blocks.shape[1]),
        fill_value=_INVALID_ABS,
    )
    k_enter_blocks = _block_matrix(
        k_enter,
        block_size=k_block,
        block_count=int(partial_blocks.shape[1]),
        fill_value=_INVALID_ENTER,
    )
    k_exit_blocks = _block_matrix(
        k_exit,
        block_size=k_block,
        block_count=int(partial_blocks.shape[1]),
        fill_value=_INVALID_EXIT,
    )
    k_pos_blocks = _block_matrix(
        k_pos,
        block_size=k_block,
        block_count=int(partial_blocks.shape[1]),
        fill_value=_INVALID_POS,
    )

    q_valid = (
        (q_abs_blocks >= 0) & (q_enter_blocks >= 0) & (q_pos_blocks != _INVALID_POS)
    )
    k_valid = (
        (k_abs_blocks >= 0)
        & (k_enter_blocks >= 0)
        & (k_exit_blocks > k_enter_blocks)
        & (k_pos_blocks != _INVALID_POS)
    )

    q_indices, k_indices = np.nonzero(candidates)
    partial_blocks[q_indices, k_indices] = False
    full_blocks[q_indices, k_indices] = False
    for q_idx, k_idx in zip(q_indices, k_indices, strict=True):
        q_valid_row = q_valid[q_idx]
        k_valid_row = k_valid[k_idx]
        if not bool(q_valid_row.any()) or not bool(k_valid_row.any()):
            continue

        q_abs_row = q_abs_blocks[q_idx][:, None]
        q_enter_row = q_enter_blocks[q_idx][:, None]
        q_pos_row = q_pos_blocks[q_idx][:, None]
        k_abs_row = k_abs_blocks[k_idx][None, :]
        k_enter_row = k_enter_blocks[k_idx][None, :]
        k_exit_row = k_exit_blocks[k_idx][None, :]
        k_pos_row = k_pos_blocks[k_idx][None, :]
        delta = q_pos_row - k_pos_row
        allowed = (
            q_valid_row[:, None]
            & k_valid_row[None, :]
            & (q_abs_row >= k_abs_row)
            & (k_enter_row <= q_enter_row)
            & (q_enter_row < k_exit_row)
            & (delta >= 0)
            & (delta < int(sliding_window))
        )
        if not bool(allowed.any()):
            continue
        if bool(q_valid_row.all()) and bool(k_valid_row.all()) and bool(allowed.all()):
            full_blocks[q_idx, k_idx] = True
        else:
            partial_blocks[q_idx, k_idx] = True


def _is_strictly_increasing(values: np.ndarray) -> bool:
    return int(values.size) <= 1 or bool(np.all(values[1:] > values[:-1]))


def _block_min_max(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mins = np.empty(starts.shape, dtype=values.dtype)
    maxes = np.empty(starts.shape, dtype=values.dtype)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        block = values[int(start) : int(end)]
        mins[index] = block.min()
        maxes[index] = block.max()
    return mins, maxes


def _block_matrix(
    values: np.ndarray,
    *,
    block_size: int,
    block_count: int,
    fill_value: int,
) -> np.ndarray:
    padded = np.full(block_count * block_size, fill_value, dtype=np.int64)
    padded[: int(values.size)] = values
    return padded.reshape(block_count, block_size)


def _build_group_interval_arrays(
    *,
    row_tree,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    enter_by_group: dict[int, int] = {}
    exit_by_group: dict[int, int] = {}
    segment_by_group = {segment.group_id: segment for segment in row_tree.segments}
    children_by_group: dict[int, list[int]] = {}
    roots: list[int] = []
    for segment in row_tree.segments:
        if segment.ancestors:
            children_by_group.setdefault(segment.parent_id, []).append(segment.group_id)
        else:
            roots.append(segment.group_id)

    next_enter = 0

    def visit(group_id: int) -> None:
        nonlocal next_enter
        enter_by_group[group_id] = next_enter
        next_enter += 1
        children = children_by_group.get(group_id, [])
        children.sort(key=lambda child: segment_by_group[child].start)
        for child_group_id in children:
            visit(child_group_id)
        exit_by_group[group_id] = next_enter

    roots.sort(key=lambda root: segment_by_group[root].start)
    for root_group_id in roots:
        visit(root_group_id)

    enter_by_token = np.full((length,), _INVALID_ENTER, dtype=np.int64)
    exit_by_token = np.full((length,), _INVALID_EXIT, dtype=np.int64)
    for segment in row_tree.segments:
        enter_by_token[segment.start : segment.end] = enter_by_group[segment.group_id]
        exit_by_token[segment.start : segment.end] = exit_by_group[segment.group_id]
    if int(row_tree.valid_tokens) < int(length):
        enter_by_token[int(row_tree.valid_tokens) :] = next_enter
        exit_by_token[int(row_tree.valid_tokens) :] = next_enter + 1
    return enter_by_token, exit_by_token


def _build_sparse_block_mask(
    spec: FlexMaskSpec,
    *,
    device: torch.device,
    context: PreparedBlockMaskContext,
    sliding_window: int | None,
    block_size: tuple[int, int],
) -> BlockMask:
    q_block, k_block = block_size
    q_blocks = (int(spec.q_len) + q_block - 1) // q_block
    k_blocks = (int(spec.k_len) + k_block - 1) // k_block
    partial_blocks = np.zeros((q_blocks, k_blocks), dtype=bool)
    full_blocks = np.zeros((q_blocks, k_blocks), dtype=bool)
    touch_counts = np.zeros((q_blocks, k_blocks), dtype=np.int16)
    q_abs_tensor = spec.exact_mask.q_token_indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    k_abs_tensor = spec.exact_mask.k_token_indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    q_abs = q_abs_tensor.numpy()
    k_abs = k_abs_tensor.numpy()
    q_abs_sorted = _is_strictly_increasing(q_abs[q_abs >= 0])
    k_abs_sorted = _is_strictly_increasing(k_abs[k_abs >= 0])
    q_enter = _select_with_invalid_np(
        context.group_enter_np,
        q_abs,
        invalid_value=_INVALID_ENTER,
    )
    k_enter = _select_with_invalid_np(
        context.group_enter_np,
        k_abs,
        invalid_value=_INVALID_ENTER,
    )
    k_exit = _select_with_invalid_np(
        context.group_exit_np,
        k_abs,
        invalid_value=_INVALID_EXIT,
    )
    q_pos = (
        _select_with_invalid_np(
            _require_input_pos(context),
            q_abs,
            invalid_value=_INVALID_POS,
        )
        if sliding_window is not None
        else None
    )
    k_pos = (
        _select_with_invalid_np(
            _require_input_pos(context),
            k_abs,
            invalid_value=_INVALID_POS,
        )
        if sliding_window is not None
        else None
    )
    mask_mod = _build_interval_mask_mod(
        q_abs=q_abs,
        k_abs=k_abs,
        q_enter=q_enter,
        k_enter=k_enter,
        k_exit=k_exit,
        q_pos=q_pos,
        k_pos=k_pos,
        sliding_window=sliding_window,
        device=device,
    )
    if not spec.slices:
        raise RuntimeError(
            "Cannot build a CP attention block mask without stage slices"
        )

    for slice_ in spec.slices:
        q_start = max(0, int(slice_.q_range.start))
        q_end = min(int(spec.q_len), int(slice_.q_range.end))
        k_start = max(0, int(slice_.k_range.start))
        k_end = min(int(spec.k_len), int(slice_.k_range.end))
        q_block_indices = np.arange(
            q_start // q_block,
            (q_end + q_block - 1) // q_block,
            dtype=np.int64,
        )
        k_block_indices = np.arange(
            k_start // k_block,
            (k_end + k_block - 1) // k_block,
            dtype=np.int64,
        )
        if int(q_block_indices.size) == 0 or int(k_block_indices.size) == 0:
            continue
        q_block_start = q_block_indices * q_block
        q_block_end_raw = (q_block_indices + 1) * q_block
        q_block_end = np.minimum(q_block_end_raw, int(spec.q_len))
        k_block_start = k_block_indices * k_block
        k_block_end_raw = (k_block_indices + 1) * k_block
        k_block_end = np.minimum(k_block_end_raw, int(spec.k_len))
        q_overlap_start = np.maximum(
            q_block_start,
            q_start,
        )
        q_overlap_end = np.minimum(
            q_block_end,
            q_end,
        )
        k_overlap_start = np.maximum(
            k_block_start,
            k_start,
        )
        k_overlap_end = np.minimum(
            k_block_end,
            k_end,
        )
        q_is_full = (q_overlap_start == q_block_start) & (
            q_overlap_end == q_block_end_raw
        )
        k_is_full = (k_overlap_start == k_block_start) & (
            k_overlap_end == k_block_end_raw
        )
        covers_block = q_is_full[:, None] & k_is_full[None, :]
        if sliding_window is None:
            window_has_any = None
            window_is_full = None
        else:
            assert q_pos is not None and k_pos is not None
            q_pos_min, q_pos_max = _block_min_max(
                q_pos,
                q_overlap_start,
                q_overlap_end,
            )
            k_pos_min, k_pos_max = _block_min_max(
                k_pos,
                k_overlap_start,
                k_overlap_end,
            )
            window_has_any = (q_pos_max[:, None] >= k_pos_min[None, :]) & (
                q_pos_min[:, None] - k_pos_max[None, :] < int(sliding_window)
            )
            window_is_full = (q_pos_min[:, None] >= k_pos_max[None, :]) & (
                q_pos_max[:, None] - k_pos_min[None, :] < int(sliding_window)
            )
        if slice_.mask_kind == AttnMaskKind.FULL:
            has_any = np.ones(
                (int(q_block_indices.size), int(k_block_indices.size)), dtype=bool
            )
            is_full = covers_block
        else:
            q_min, q_max = (
                (q_abs[q_overlap_start], q_abs[q_overlap_end - 1])
                if q_abs_sorted
                else _block_min_max(q_abs, q_overlap_start, q_overlap_end)
            )
            k_min, k_max = (
                (k_abs[k_overlap_start], k_abs[k_overlap_end - 1])
                if k_abs_sorted
                else _block_min_max(k_abs, k_overlap_start, k_overlap_end)
            )
            has_any = q_max[:, None] >= k_min[None, :]
            is_full = covers_block & (q_min[:, None] >= k_max[None, :])
        if sliding_window is not None:
            assert window_has_any is not None and window_is_full is not None
            has_any &= window_has_any
            is_full &= window_is_full

        q_slice = slice(int(q_block_indices[0]), int(q_block_indices[-1]) + 1)
        k_slice = slice(int(k_block_indices[0]), int(k_block_indices[-1]) + 1)
        touch_counts[q_slice, k_slice] += has_any.astype(np.int16)
        partial_blocks[q_slice, k_slice] |= has_any
        full_blocks[q_slice, k_slice] |= is_full

    partial_blocks &= ~full_blocks
    needs_refine = full_blocks | ((touch_counts > 1) & partial_blocks)
    if bool(needs_refine.any()):
        refined_partial = partial_blocks & needs_refine
        refined_full = full_blocks & needs_refine
        _refine_interval_blocks(
            partial_blocks=refined_partial,
            full_blocks=refined_full,
            q_abs=q_abs,
            k_abs=k_abs,
            q_enter=q_enter,
            k_enter=k_enter,
            k_exit=k_exit,
            q_block=q_block,
            k_block=k_block,
        )
        partial_blocks = (partial_blocks & ~needs_refine) | refined_partial
        full_blocks = (full_blocks & ~needs_refine) | refined_full
    if sliding_window is not None:
        assert q_pos is not None and k_pos is not None
        _refine_sliding_interval_blocks(
            partial_blocks=partial_blocks,
            full_blocks=full_blocks,
            q_abs=q_abs,
            k_abs=k_abs,
            q_enter=q_enter,
            k_enter=k_enter,
            k_exit=k_exit,
            q_pos=q_pos,
            k_pos=k_pos,
            q_block=q_block,
            k_block=k_block,
            sliding_window=int(sliding_window),
        )
    kv_num_blocks, kv_indices = _dense_blocks_to_ordered(
        partial_blocks,
        device=device,
    )
    full_kv_num_blocks, full_kv_indices = _dense_blocks_to_ordered(
        full_blocks,
        device=device,
    )
    q_num_blocks, q_indices = _dense_blocks_to_ordered(
        partial_blocks.T,
        device=device,
    )
    full_q_num_blocks, full_q_indices = _dense_blocks_to_ordered(
        full_blocks.T,
        device=device,
    )
    return BlockMask(
        seq_lengths=(int(spec.q_len), int(spec.k_len)),
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num_blocks,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num_blocks,
        full_q_indices=full_q_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
    )


def prepare_block_mask_context(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    input_pos: torch.Tensor | None = None,
) -> PreparedBlockMaskContext:
    if group_ids.ndim != 1 or parent_ids.ndim != 1:
        raise RuntimeError(
            "Shared-prefix sparse block masks require rank-1 group_ids and parent_ids."
        )
    if int(group_ids.numel()) != int(parent_ids.numel()):
        raise RuntimeError(
            "Shared-prefix sparse block masks require equal group_ids and parent_ids lengths."
        )
    if input_pos is not None and int(input_pos.numel()) != int(group_ids.numel()):
        raise RuntimeError(
            "Shared-prefix sparse block masks require input_pos to match group_ids length."
        )
    flat_group_ids = group_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    flat_parent_ids = (
        parent_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    )
    flat_input_pos = (
        None
        if input_pos is None
        else input_pos.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    )
    row_tree = parse_prefix_tree_row(
        group_ids=flat_group_ids,
        parent_ids=flat_parent_ids,
    )
    group_enter_np, group_exit_np = _build_group_interval_arrays(
        row_tree=row_tree,
        length=int(flat_group_ids.numel()),
    )
    return PreparedBlockMaskContext(
        source_len=int(flat_group_ids.numel()),
        group_enter_np=group_enter_np,
        group_exit_np=group_exit_np,
        input_pos_np=None if flat_input_pos is None else flat_input_pos.numpy(),
    )


def _require_input_pos(context: PreparedBlockMaskContext) -> np.ndarray:
    if context.input_pos_np is None:
        raise RuntimeError("Sliding-window prefix-tree block masks require input_pos.")
    return context.input_pos_np


def _validate_exact_indices(
    indices: torch.Tensor,
    *,
    name: str,
    source_len: int,
) -> int:
    if indices.ndim != 1:
        raise RuntimeError(f"{name} exact token indices must be rank 1.")
    if indices.dtype != torch.int64:
        raise RuntimeError(f"{name} exact token indices must be int64.")
    indices_cpu = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    invalid = indices_cpu < 0
    if bool(invalid.any().item()):
        first_invalid = int(torch.nonzero(invalid, as_tuple=False)[0].item())
        if bool((indices_cpu[first_invalid:] >= 0).any().item()):
            raise RuntimeError(
                f"{name} exact token indices must use only contiguous tail padding."
            )
        indices_cpu = indices_cpu[:first_invalid]
    if int(indices_cpu.numel()) == 0:
        return 0
    if int(indices_cpu.unique().numel()) != int(indices_cpu.numel()):
        raise RuntimeError(f"{name} exact token indices must not contain duplicates.")
    max_index = int(indices_cpu.max().item())
    if max_index >= int(source_len):
        raise RuntimeError(
            f"{name} exact token index {max_index} exceeds source metadata length {int(source_len)}."
        )
    return int(indices_cpu.numel())


def _validate_supported_mask_spec(
    spec: FlexMaskSpec,
    *,
    source_len: int,
) -> None:
    q_valid_len = _validate_exact_indices(
        spec.exact_mask.q_token_indices,
        name="q",
        source_len=source_len,
    )
    k_valid_len = _validate_exact_indices(
        spec.exact_mask.k_token_indices,
        name="k",
        source_len=source_len,
    )
    for slice_ in spec.slices:
        if int(slice_.row_index) != 0:
            raise RuntimeError(
                "Shared-prefix sparse block masks support exactly one packed row."
            )
        if slice_.mask_kind not in {AttnMaskKind.FULL, AttnMaskKind.CAUSAL}:
            raise RuntimeError(f"Unsupported attention mask kind: {slice_.mask_kind}")
        if (
            slice_.q_range.start < 0
            or slice_.q_range.end > int(spec.q_len)
            or slice_.k_range.start < 0
            or slice_.k_range.end > int(spec.k_len)
            or slice_.q_range.end < slice_.q_range.start
            or slice_.k_range.end < slice_.k_range.start
        ):
            raise RuntimeError(f"Attention slice is outside mask bounds: {slice_}")
        if slice_.q_range.end > q_valid_len or slice_.k_range.end > k_valid_len:
            raise RuntimeError(
                "Attention slices may not cover exact-index tail padding: "
                f"slice={slice_}, q_valid_len={q_valid_len}, k_valid_len={k_valid_len}"
            )


def build_block_mask_from_context(
    spec: FlexMaskSpec,
    *,
    context: PreparedBlockMaskContext,
    device: torch.device,
    sliding_window: int | None = None,
    validate: bool = True,
) -> BlockMask | None:
    if spec.q_len <= 0 or spec.k_len <= 0:
        return None
    if int(spec.exact_mask.q_token_indices.numel()) != int(spec.q_len):
        raise RuntimeError(
            "Exact stage q-token metadata length mismatch: "
            f"{int(spec.exact_mask.q_token_indices.numel())} != {int(spec.q_len)}"
        )
    if int(spec.exact_mask.k_token_indices.numel()) != int(spec.k_len):
        raise RuntimeError(
            "Exact stage k-token metadata length mismatch: "
            f"{int(spec.exact_mask.k_token_indices.numel())} != {int(spec.k_len)}"
        )
    if validate:
        _validate_supported_mask_spec(
            spec,
            source_len=context.source_len,
        )
    block_size = normalize_sparse_block_size(spec.block_size)
    return _build_sparse_block_mask(
        spec,
        device=device,
        context=context,
        sliding_window=sliding_window,
        block_size=block_size,
    )
