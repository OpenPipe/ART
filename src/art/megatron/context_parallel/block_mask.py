from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask

from art.megatron.flex_attn.compiled import normalize_sparse_block_size

from .types import AttnMaskKind, FlexMaskSpec

_INVALID_Q_GROUP = -(1 << 63)
_INVALID_Q_PARENT = _INVALID_Q_GROUP + 1
_INVALID_K_GROUP = _INVALID_Q_GROUP + 2
_INVALID_POSITION = _INVALID_Q_GROUP + 3


def _build_exact_mask_mod(
    *,
    q_abs: np.ndarray,
    k_abs: np.ndarray,
    q_group: np.ndarray,
    q_parent: np.ndarray,
    k_group: np.ndarray,
    q_pos: np.ndarray | None,
    k_pos: np.ndarray | None,
    sliding_window: int | None,
    device: torch.device,
):
    q_group_tensor = torch.as_tensor(q_group, device=device, dtype=torch.int64)
    q_parent_tensor = torch.as_tensor(q_parent, device=device, dtype=torch.int64)
    k_group_tensor = torch.as_tensor(k_group, device=device, dtype=torch.int64)

    if sliding_window is not None:
        q_pos_tensor = torch.as_tensor(q_pos, device=device, dtype=torch.int64)
        k_pos_tensor = torch.as_tensor(k_pos, device=device, dtype=torch.int64)

        def sliding_mask_mod(
            batch_idx: torch.Tensor,
            head_idx: torch.Tensor,
            query_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            del batch_idx, head_idx
            same_group = q_group_tensor[query_idx] == k_group_tensor[kv_idx]
            parent_prefix = q_parent_tensor[query_idx] == k_group_tensor[kv_idx]
            delta = q_pos_tensor[query_idx] - k_pos_tensor[kv_idx]
            return (
                (same_group | parent_prefix)
                & (delta >= 0)
                & (delta < int(sliding_window))
            )

        return sliding_mask_mod

    q_abs_tensor = torch.as_tensor(q_abs, device=device, dtype=torch.int64)
    k_abs_tensor = torch.as_tensor(k_abs, device=device, dtype=torch.int64)

    def mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del batch_idx, head_idx
        q_abs_local = q_abs_tensor[query_idx]
        k_abs_local = k_abs_tensor[kv_idx]
        same_group = q_group_tensor[query_idx] == k_group_tensor[kv_idx]
        parent_prefix = q_parent_tensor[query_idx] == k_group_tensor[kv_idx]
        return (q_abs_local >= k_abs_local) & (same_group | parent_prefix)

    return mask_mod


def _dense_blocks_to_ordered(
    blocks: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.from_numpy(blocks.sum(axis=-1).astype(np.int32))
    indices = torch.from_numpy(
        np.argsort(-blocks.astype(np.int32), axis=-1, kind="stable").astype(np.int32)
    )
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


def _build_q_block_group_state(
    *,
    q_abs: np.ndarray,
    q_group: np.ndarray,
    q_parent: np.ndarray,
    q_block: int,
    q_blocks: int,
) -> tuple[np.ndarray, list[dict[int, int]], list[frozenset[int]]]:
    q_min_by_block = np.empty((q_blocks,), dtype=np.int64)
    q_allowed_max_by_group: list[dict[int, int]] = []
    q_all_allowed_groups: list[frozenset[int]] = []
    for block_idx in range(q_blocks):
        start = block_idx * q_block
        end = min((block_idx + 1) * q_block, int(q_abs.size))
        q = q_abs[start:end]
        q_group_block = q_group[start:end]
        q_parent_block = q_parent[start:end]
        q_min_by_block[block_idx] = int(q.min()) if int(q.size) else 0
        max_by_group: dict[int, int] = {}
        all_groups: list[int] = []
        for group_value in np.unique(np.concatenate((q_group_block, q_parent_block))):
            allowed = (q_group_block == group_value) | (q_parent_block == group_value)
            if bool(allowed.any()):
                max_by_group[int(group_value)] = int(q[allowed].max())
            if bool(allowed.all()):
                all_groups.append(int(group_value))
        q_allowed_max_by_group.append(max_by_group)
        q_all_allowed_groups.append(frozenset(all_groups))
    return q_min_by_block, q_allowed_max_by_group, q_all_allowed_groups


def _build_k_block_group_state(
    *,
    k_abs: np.ndarray,
    k_group: np.ndarray,
    k_block: int,
    k_blocks: int,
) -> tuple[np.ndarray, list[dict[int, int]], list[tuple[int, ...]]]:
    k_max_by_block = np.empty((k_blocks,), dtype=np.int64)
    k_min_by_group: list[dict[int, int]] = []
    k_groups_by_block: list[tuple[int, ...]] = []
    for block_idx in range(k_blocks):
        start = block_idx * k_block
        end = min((block_idx + 1) * k_block, int(k_abs.size))
        k = k_abs[start:end]
        k_group_block = k_group[start:end]
        k_max_by_block[block_idx] = int(k.max()) if int(k.size) else 0
        min_by_group: dict[int, int] = {}
        for group_value in np.unique(k_group_block):
            min_by_group[int(group_value)] = int(k[k_group_block == group_value].min())
        k_min_by_group.append(min_by_group)
        k_groups_by_block.append(tuple(min_by_group))
    return k_max_by_block, k_min_by_group, k_groups_by_block


def _exact_block_state(
    *,
    q_idx: int,
    k_idx: int,
    q_min_by_block: np.ndarray,
    q_allowed_max_by_group: list[dict[int, int]],
    q_all_allowed_groups: list[frozenset[int]],
    k_max_by_block: np.ndarray,
    k_min_by_group: list[dict[int, int]],
    k_groups_by_block: list[tuple[int, ...]],
) -> tuple[bool, bool]:
    q_allowed_max = q_allowed_max_by_group[q_idx]
    k_min = k_min_by_group[k_idx]
    if not any(
        q_allowed_max.get(k_group_value, _INVALID_Q_GROUP) >= min_k
        for k_group_value, min_k in k_min.items()
    ):
        return False, False
    if int(q_min_by_block[q_idx]) < int(k_max_by_block[k_idx]):
        return True, False
    q_all_allowed = q_all_allowed_groups[q_idx]
    return True, all(
        k_group_value in q_all_allowed for k_group_value in k_groups_by_block[k_idx]
    )


def _range_min_max(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mins = np.empty((int(starts.size),), dtype=np.int64)
    maxes = np.empty((int(starts.size),), dtype=np.int64)
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        block = values[int(start) : int(end)]
        mins[index] = int(block.min()) if int(block.size) else 0
        maxes[index] = int(block.max()) if int(block.size) else 0
    return mins, maxes


def _exact_sliding_block_state(
    *,
    q_idx: int,
    k_idx: int,
    q_block: int,
    k_block: int,
    q_len: int,
    k_len: int,
    q_group: np.ndarray,
    q_parent: np.ndarray,
    k_group: np.ndarray,
    q_pos: np.ndarray,
    k_pos: np.ndarray,
    sliding_window: int,
) -> tuple[bool, bool]:
    q_start = int(q_idx) * int(q_block)
    q_end = min(q_start + int(q_block), int(q_len))
    k_start = int(k_idx) * int(k_block)
    k_end = min(k_start + int(k_block), int(k_len))
    q_group_block = q_group[q_start:q_end]
    q_parent_block = q_parent[q_start:q_end]
    k_group_block = k_group[k_start:k_end]
    delta = q_pos[q_start:q_end, None] - k_pos[None, k_start:k_end]
    allowed = (
        (
            (q_group_block[:, None] == k_group_block[None, :])
            | (q_parent_block[:, None] == k_group_block[None, :])
        )
        & (delta >= 0)
        & (delta < int(sliding_window))
    )
    return bool(allowed.any()), bool(allowed.all())


def _build_sparse_block_mask(
    spec: FlexMaskSpec,
    *,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    input_pos: torch.Tensor | None,
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
    flat_group_ids = group_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    flat_parent_ids = (
        parent_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    )
    flat_group_ids_np = flat_group_ids.numpy()
    flat_parent_ids_np = flat_parent_ids.numpy()
    input_pos_for_window = cast(torch.Tensor, input_pos)
    flat_input_pos_np = (
        input_pos_for_window.detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1)
        .numpy()
        if sliding_window is not None
        else None
    )
    q_group = _select_with_invalid_np(
        flat_group_ids_np,
        q_abs,
        invalid_value=_INVALID_Q_GROUP,
    )
    q_parent = _select_with_invalid_np(
        flat_parent_ids_np,
        q_abs,
        invalid_value=_INVALID_Q_PARENT,
    )
    k_group = _select_with_invalid_np(
        flat_group_ids_np,
        k_abs,
        invalid_value=_INVALID_K_GROUP,
    )
    q_pos = (
        _select_with_invalid_np(
            cast(np.ndarray, flat_input_pos_np),
            q_abs,
            invalid_value=_INVALID_POSITION,
        )
        if sliding_window is not None
        else None
    )
    k_pos = (
        _select_with_invalid_np(
            cast(np.ndarray, flat_input_pos_np),
            k_abs,
            invalid_value=_INVALID_POSITION,
        )
        if sliding_window is not None
        else None
    )
    mask_mod = _build_exact_mask_mod(
        q_abs=q_abs,
        k_abs=k_abs,
        q_group=q_group,
        q_parent=q_parent,
        k_group=k_group,
        q_pos=q_pos,
        k_pos=k_pos,
        sliding_window=sliding_window,
        device=device,
    )
    q_min_by_block, q_allowed_max_by_group, q_all_allowed_groups = (
        _build_q_block_group_state(
            q_abs=q_abs,
            q_group=q_group,
            q_parent=q_parent,
            q_block=q_block,
            q_blocks=q_blocks,
        )
    )
    k_max_by_block, k_min_by_group, k_groups_by_block = _build_k_block_group_state(
        k_abs=k_abs,
        k_group=k_group,
        k_block=k_block,
        k_blocks=k_blocks,
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
        q_block_end = np.minimum(
            (q_block_indices + 1) * q_block,
            int(spec.q_len),
        )
        k_block_start = k_block_indices * k_block
        k_block_end = np.minimum(
            (k_block_indices + 1) * k_block,
            int(spec.k_len),
        )
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
        q_min = q_abs[q_overlap_start]
        q_max = q_abs[q_overlap_end - 1]
        k_min = k_abs[k_overlap_start]
        k_max = k_abs[k_overlap_end - 1]
        if sliding_window is not None:
            q_pos_for_window = cast(np.ndarray, q_pos)
            k_pos_for_window = cast(np.ndarray, k_pos)
            q_pos_min, q_pos_max = _range_min_max(
                q_pos_for_window,
                q_overlap_start,
                q_overlap_end,
            )
            k_pos_min, k_pos_max = _range_min_max(
                k_pos_for_window,
                k_overlap_start,
                k_overlap_end,
            )
            window_has_any = (q_pos_max[:, None] >= k_pos_min[None, :]) & (
                q_pos_min[:, None] - k_pos_max[None, :] < int(sliding_window)
            )
            window_is_full = (q_pos_min[:, None] >= k_pos_max[None, :]) & (
                q_pos_max[:, None] - k_pos_min[None, :] < int(sliding_window)
            )
        q_is_full = (q_overlap_start == q_block_start) & (q_overlap_end == q_block_end)
        k_is_full = (k_overlap_start == k_block_start) & (k_overlap_end == k_block_end)
        covers_block = q_is_full[:, None] & k_is_full[None, :]
        if sliding_window is not None:
            has_any = window_has_any
            is_full = covers_block & window_is_full
        elif slice_.mask_kind == AttnMaskKind.FULL:
            has_any = np.ones(
                (int(q_block_indices.size), int(k_block_indices.size)), dtype=bool
            )
            is_full = covers_block
        else:
            has_any = q_max[:, None] >= k_min[None, :]
            is_full = covers_block & (q_min[:, None] >= k_max[None, :])

        q_slice = slice(int(q_block_indices[0]), int(q_block_indices[-1]) + 1)
        k_slice = slice(int(k_block_indices[0]), int(k_block_indices[-1]) + 1)
        touch_counts[q_slice, k_slice] += has_any.astype(np.int16)
        partial_blocks[q_slice, k_slice] |= has_any
        full_blocks[q_slice, k_slice] |= is_full

    ambiguous = (touch_counts > 1) & partial_blocks & ~full_blocks
    for q_idx, k_idx in np.argwhere(ambiguous):
        if sliding_window is None:
            has_any, is_full = _exact_block_state(
                q_idx=int(q_idx),
                k_idx=int(k_idx),
                q_min_by_block=q_min_by_block,
                q_allowed_max_by_group=q_allowed_max_by_group,
                q_all_allowed_groups=q_all_allowed_groups,
                k_max_by_block=k_max_by_block,
                k_min_by_group=k_min_by_group,
                k_groups_by_block=k_groups_by_block,
            )
        else:
            has_any, is_full = _exact_sliding_block_state(
                q_idx=int(q_idx),
                k_idx=int(k_idx),
                q_block=q_block,
                k_block=k_block,
                q_len=int(spec.q_len),
                k_len=int(spec.k_len),
                q_group=q_group,
                q_parent=q_parent,
                k_group=k_group,
                q_pos=cast(np.ndarray, q_pos),
                k_pos=cast(np.ndarray, k_pos),
                sliding_window=int(sliding_window),
            )
        partial_blocks[q_idx, k_idx] = False
        full_blocks[q_idx, k_idx] = False
        if is_full:
            full_blocks[q_idx, k_idx] = True
        elif has_any:
            partial_blocks[q_idx, k_idx] = True

    partial_blocks &= ~full_blocks
    kv_num_blocks, kv_indices = _dense_blocks_to_ordered(
        partial_blocks,
        device=device,
    )
    full_kv_num_blocks, full_kv_indices = _dense_blocks_to_ordered(
        full_blocks,
        device=device,
    )
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(int(spec.q_len), int(spec.k_len)),
    )


def build_block_mask(
    spec: FlexMaskSpec,
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    input_pos: torch.Tensor | None = None,
    sliding_window: int | None = None,
    device: torch.device,
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
    block_size = normalize_sparse_block_size(spec.block_size)
    return _build_sparse_block_mask(
        spec,
        device=device,
        group_ids=group_ids,
        parent_ids=parent_ids,
        input_pos=input_pos,
        sliding_window=sliding_window,
        block_size=block_size,
    )
