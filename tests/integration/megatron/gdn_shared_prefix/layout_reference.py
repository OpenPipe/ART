from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch import Tensor

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn.gdn_prefix_tree import (
    GdnPackedExecutionSpec,
    parse_gdn_prefix_tree_segments,
)
from art.megatron.gdn.layout import (
    GdnCpExchangePlan,
    GdnCpPeerTransfer,
    build_local_rank_cp_exchange_plan_from_dest_ranges,
)


class TestGdnCpLayoutPlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    batch_size: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    cp_size: int = Field(ge=1)
    attention_token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    gdn_token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...]
    attention_to_gdn: GdnCpExchangePlan
    gdn_to_attention: GdnCpExchangePlan


def build_test_gdn_cp_layout_plan(
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None = None,
    gdn_token_ranges_by_rank: Sequence[Sequence[tuple[int, int, int]]] | None = None,
    device: torch.device | str | None = None,
) -> TestGdnCpLayoutPlan:
    spec = parse_gdn_prefix_tree_segments(group_ids, parent_ids)
    gdn_ranges = (
        _normalize_rank_ranges(gdn_token_ranges_by_rank, cp_size=cp_size)
        if gdn_token_ranges_by_rank is not None
        else _split_gdn_token_ranges_by_rank(spec, cp_size=cp_size)
    )
    source_layout = attention_token_layout_index or _token_layout_from_rank_ranges(
        _split_attention_token_ranges_by_rank(spec, cp_size=cp_size)
    )
    attention_to_gdn = _build_full_exchange_plan(
        source_layout=source_layout,
        dest_ranges_by_rank=gdn_ranges,
        device=device,
    )
    gdn_layout = _token_layout_from_rank_ranges(gdn_ranges)
    gdn_to_attention = _build_full_exchange_plan(
        source_layout=gdn_layout,
        dest_ranges_by_rank=source_layout.ownership_ranges_by_rank,
        device=device,
    )
    return TestGdnCpLayoutPlan(
        batch_size=spec.batch_size,
        sequence_length=spec.sequence_length,
        cp_size=cp_size,
        attention_token_ranges_by_rank=source_layout.ownership_ranges_by_rank,
        gdn_token_ranges_by_rank=gdn_ranges,
        attention_to_gdn=attention_to_gdn,
        gdn_to_attention=gdn_to_attention,
    )


def _build_full_exchange_plan(
    *,
    source_layout: TokenLayoutIndex,
    dest_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...],
    device: torch.device | str | None,
) -> GdnCpExchangePlan:
    transfers: dict[tuple[int, int], GdnCpPeerTransfer] = {}
    for local_rank in range(len(source_layout.token_counts_by_rank)):
        local_plan = build_local_rank_cp_exchange_plan_from_dest_ranges(
            source_layout=source_layout,
            dest_ranges_by_rank=dest_ranges_by_rank,
            device=device,
            local_rank=local_rank,
            cross_rank_token_count=0,
        )
        for transfer in local_plan.transfers:
            transfers.setdefault((transfer.source_rank, transfer.dest_rank), transfer)
    return GdnCpExchangePlan(
        cp_size=len(source_layout.token_counts_by_rank),
        source_token_counts_by_rank=source_layout.token_counts_by_rank,
        dest_token_counts_by_rank=tuple(
            sum(end - start for start, end, _ in ranges)
            for ranges in dest_ranges_by_rank
        ),
        transfers=tuple(
            sorted(
                transfers.values(), key=lambda item: (item.source_rank, item.dest_rank)
            )
        ),
    )


def _split_attention_token_ranges_by_rank(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    return _split_ordered_ranges_by_rank(
        tuple(
            (
                row_index * spec.sequence_length,
                row_index * spec.sequence_length + valid_length,
            )
            for row_index, valid_length in enumerate(spec.valid_lengths)
            if valid_length
        ),
        cp_size=cp_size,
    )


def _split_gdn_token_ranges_by_rank(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    return _split_ordered_ranges_by_rank(
        tuple(
            (
                _segment_token_start(segment, spec.sequence_length),
                _segment_token_start(segment, spec.sequence_length) + segment.length,
            )
            for segment in spec.tree_segments
        ),
        cp_size=cp_size,
    )


def _split_ordered_ranges_by_rank(
    ordered_ranges: Sequence[tuple[int, int]],
    *,
    cp_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    total_tokens = sum(end - start for start, end in ordered_ranges)
    ranks: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    rank_positions = [0] * cp_size
    rank = 0
    rank_end = (total_tokens * (rank + 1)) // cp_size
    consumed = 0
    for start, end in ordered_ranges:
        cursor = start
        while cursor < end:
            while rank + 1 < cp_size and consumed >= rank_end:
                rank += 1
                rank_end = (total_tokens * (rank + 1)) // cp_size
            piece_end = end
            if rank + 1 < cp_size:
                piece_end = min(piece_end, cursor + rank_end - consumed)
            ranks[rank].append((cursor, piece_end, rank_positions[rank]))
            piece_length = piece_end - cursor
            rank_positions[rank] += piece_length
            consumed += piece_length
            cursor = piece_end
    return tuple(tuple(ranges) for ranges in ranks)


def _token_layout_from_rank_ranges(
    ranges_by_rank: Sequence[Sequence[tuple[int, int, int]]],
) -> TokenLayoutIndex:
    ranges = _normalize_rank_ranges(ranges_by_rank, cp_size=len(ranges_by_rank))
    return TokenLayoutIndex(
        ownership_ranges_by_rank=ranges,
        token_counts_by_rank=tuple(
            sum(end - start for start, end, _ in rank_ranges) for rank_ranges in ranges
        ),
    )


def _normalize_rank_ranges(
    ranges_by_rank: Sequence[Sequence[tuple[int, int, int]]],
    *,
    cp_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    if len(ranges_by_rank) != cp_size:
        raise ValueError("rank range count must equal cp_size")
    return tuple(
        tuple((int(start), int(end), int(position)) for start, end, position in ranges)
        for ranges in ranges_by_rank
    )


def _segment_token_start(segment: object, sequence_length: int) -> int:
    return int(getattr(segment, "row_index")) * int(sequence_length) + int(
        getattr(segment, "start")
    )
