from __future__ import annotations

import torch

from art.megatron.shared_prefix_tree import parse_shared_prefix_tree

from .types import (
    AttnMaskKind,
    AttnSlice,
    PackedBatchAttentionSpec,
    PackedRowAttentionSpec,
    TokenRange,
)


def _sort_and_dedupe_slices(slices: list[AttnSlice]) -> tuple[AttnSlice, ...]:
    sorted_slices = sorted(
        slices,
        key=lambda slice_: (
            int(slice_.row_index),
            int(slice_.q_range.start),
            int(slice_.q_range.end),
            int(slice_.k_range.start),
            int(slice_.k_range.end),
            str(slice_.mask_kind),
            -1 if slice_.family_index is None else int(slice_.family_index),
        ),
    )
    deduped: list[AttnSlice] = []
    last_key: tuple[int, int, int, int, int, str, int] | None = None
    for slice_ in sorted_slices:
        key = (
            int(slice_.row_index),
            int(slice_.q_range.start),
            int(slice_.q_range.end),
            int(slice_.k_range.start),
            int(slice_.k_range.end),
            str(slice_.mask_kind),
            -1 if slice_.family_index is None else int(slice_.family_index),
        )
        if key == last_key:
            continue
        deduped.append(slice_)
        last_key = key
    return tuple(deduped)


def build_shared_prefix_attention_spec(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    ignore_padding_group_id: int = -1,
) -> PackedBatchAttentionSpec:
    if group_ids.shape != parent_ids.shape:
        raise RuntimeError(
            "group_ids and parent_ids must share shape, got "
            f"{tuple(group_ids.shape)} vs {tuple(parent_ids.shape)}"
        )
    if group_ids.ndim != 2:
        raise RuntimeError(
            "group_ids and parent_ids must be rank-2 packed tensors, got "
            f"{group_ids.ndim}"
        )
    rows: list[PackedRowAttentionSpec] = []
    for row in parse_shared_prefix_tree(
        group_ids=group_ids,
        parent_ids=parent_ids,
        ignore_padding_group_id=ignore_padding_group_id,
    ):
        if row.valid_tokens == 0:
            rows.append(
                PackedRowAttentionSpec(
                    row_index=row.row_index, valid_tokens=0, slices=()
                )
            )
            continue

        segment_by_group_id = {segment.group_id: segment for segment in row.segments}
        row_slices: list[AttnSlice] = []
        for segment in row.segments:
            q_range = TokenRange(start=segment.start, end=segment.end)
            for ancestor_group_id in segment.ancestors:
                ancestor = segment_by_group_id[ancestor_group_id]
                row_slices.append(
                    AttnSlice(
                        q_range=q_range,
                        k_range=TokenRange(start=ancestor.start, end=ancestor.end),
                        mask_kind=AttnMaskKind.FULL,
                        row_index=row.row_index,
                        family_index=segment.family_index,
                    )
                )
            row_slices.append(
                AttnSlice(
                    q_range=q_range,
                    k_range=q_range,
                    mask_kind=AttnMaskKind.CAUSAL,
                    row_index=row.row_index,
                    family_index=segment.family_index,
                )
            )

        rows.append(
            PackedRowAttentionSpec(
                row_index=row.row_index,
                valid_tokens=row.valid_tokens,
                slices=_sort_and_dedupe_slices(row_slices),
            )
        )

    return PackedBatchAttentionSpec(rows=tuple(rows))


def build_dense_reference_mask(
    *,
    row_spec: PackedRowAttentionSpec,
) -> torch.Tensor:
    dense = torch.zeros(
        (row_spec.valid_tokens, row_spec.valid_tokens),
        dtype=torch.bool,
    )
    for slice_ in row_spec.slices:
        q = slice_.q_range
        k = slice_.k_range
        if slice_.mask_kind is AttnMaskKind.FULL:
            dense[q.start : q.end, k.start : k.end] = True
            continue
        for q_idx in range(q.start, q.end):
            rel_q = q_idx - q.start
            max_k = k.start + rel_q
            if max_k < k.start:
                continue
            dense[q_idx, k.start : min(k.end, max_k + 1)] = True
    return dense
