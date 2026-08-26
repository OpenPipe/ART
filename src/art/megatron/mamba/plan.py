from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.recurrent import RecurrentPrefixTree


class MambaConvBucket(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    segment_indices: tuple[int, ...]
    parent_indices: tuple[int, ...]
    parent_rows: torch.Tensor
    token_indices: torch.Tensor


class MambaScanBucket(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    state_indices: tuple[int, ...]
    parent_state_indices: tuple[int, ...]
    parent_rows: torch.Tensor
    token_indices: torch.Tensor
    output_rows: torch.Tensor
    output_positions: torch.Tensor
    needs_final_state: bool

    @property
    def batch_size(self) -> int:
        return len(self.state_indices)

    @property
    def length(self) -> int:
        return int(self.token_indices.shape[1])


class MambaTokenExchangePlan(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cp_rank: int = Field(ge=0)
    cp_size: int = Field(gt=0)
    source_token_counts: tuple[int, ...]
    global_positions_by_rank: tuple[torch.Tensor, ...]
    canonical_to_received: torch.Tensor
    physical_token_positions: torch.Tensor

    @property
    def token_count(self) -> int:
        return sum(self.source_token_counts)

    @property
    def local_token_count(self) -> int:
        return self.source_token_counts[self.cp_rank]


class MambaExecutionPlan(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tree: RecurrentPrefixTree
    conv_buckets: tuple[MambaConvBucket, ...]
    scan_phases: tuple[tuple[MambaScanBucket, ...], ...]
    conv_token_positions: torch.Tensor
    scan_token_positions: torch.Tensor
    scan_token_occurrences: torch.Tensor
    exchange: MambaTokenExchangePlan
    chunk_size: int = Field(gt=0)


class _ScanColumn(BaseModel):
    model_config = ConfigDict(frozen=True)

    state_index: int
    parent_state_index: int
    positions: tuple[int, ...]
    output_mask: tuple[bool, ...]
    needs_final_state: bool


@lru_cache(maxsize=4)
def build_mamba_execution_plan(
    tree: RecurrentPrefixTree,
    *,
    device: torch.device,
    cp_rank: int,
    cp_size: int,
    token_layout: TokenLayoutIndex | None,
    chunk_size: int = 128,
) -> MambaExecutionPlan:
    """Reuse fixed-shape tree, scan, and CP metadata across identical packed rows."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    exchange = _build_exchange_plan(
        tree,
        device=device,
        cp_rank=cp_rank,
        cp_size=cp_size,
        token_layout=token_layout,
    )
    canonical_to_received = exchange.canonical_to_received
    conv_buckets = tuple(
        bucket.model_copy(
            update={"token_indices": canonical_to_received[bucket.token_indices]}
        )
        for bucket in _build_conv_buckets(tree, device)
    )
    scan_phases = tuple(
        tuple(
            bucket.model_copy(
                update={
                    "token_indices": _remap_positions(
                        bucket.token_indices, canonical_to_received
                    ),
                    "output_positions": canonical_to_received[bucket.output_positions],
                }
            )
            for bucket in phase
        )
        for phase in _build_scan_phases(tree, chunk_size, device)
    )
    conv_positions = torch.cat(tuple(bucket.token_indices for bucket in conv_buckets))
    scan_positions = torch.cat(
        tuple(
            bucket.token_indices.flatten() for phase in scan_phases for bucket in phase
        )
    )
    scan_output_positions = torch.cat(
        tuple(bucket.output_positions for phase in scan_phases for bucket in phase)
    )
    _canonical_order(scan_output_positions, tree.token_count)
    scan_occurrences = _token_occurrences(scan_positions, tree.token_count)
    return MambaExecutionPlan(
        tree=tree,
        conv_buckets=conv_buckets,
        scan_phases=scan_phases,
        conv_token_positions=conv_positions,
        scan_token_positions=scan_positions,
        scan_token_occurrences=scan_occurrences,
        exchange=exchange,
        chunk_size=chunk_size,
    )


def _build_conv_buckets(
    tree: RecurrentPrefixTree,
    device: torch.device,
) -> tuple[MambaConvBucket, ...]:
    buckets = []
    state_rows: dict[int, int] = {}
    for depth in range(
        max((segment.depth for segment in tree.segments), default=-1) + 1
    ):
        lengths = sorted(
            {segment.length for segment in tree.segments if segment.depth == depth}
        )
        for length in lengths:
            segments = tuple(
                segment
                for segment in tree.segments
                if segment.depth == depth and segment.length == length
            )
            segment_indices = tuple(segment.index for segment in segments)
            parent_indices = tuple(segment.parent_index for segment in segments)
            buckets.append(
                MambaConvBucket(
                    segment_indices=segment_indices,
                    parent_indices=parent_indices,
                    parent_rows=torch.tensor(
                        tuple(
                            -1 if parent < 0 else state_rows[parent]
                            for parent in parent_indices
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                    token_indices=torch.tensor(
                        tuple(
                            position
                            for segment in segments
                            for position in range(segment.start, segment.end)
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                )
            )
            for segment_index in segment_indices:
                state_rows[segment_index] = len(state_rows)
    return tuple(buckets)


def _build_scan_phases(
    tree: RecurrentPrefixTree,
    chunk_size: int,
    device: torch.device,
) -> tuple[tuple[MambaScanBucket, ...], ...]:
    children: list[list[int]] = [[] for _ in tree.segments]
    for segment in tree.segments:
        if segment.parent_index >= 0:
            children[segment.parent_index].append(segment.index)
    phases: list[list[_ScanColumn]] = []
    next_state_index = 0

    def emit(
        positions: tuple[int, ...],
        output_mask: tuple[bool, ...],
        parent_state_index: int,
        phase: int,
        needs_final_state: bool,
    ) -> int:
        nonlocal next_state_index
        if not positions or len(positions) != len(output_mask):
            raise ValueError("Mamba scan columns require aligned non-empty metadata")
        while len(phases) <= phase:
            phases.append([])
        state_index = next_state_index
        next_state_index += 1
        phases[phase].append(
            _ScanColumn(
                state_index=state_index,
                parent_state_index=parent_state_index,
                positions=positions,
                output_mask=output_mask,
                needs_final_state=needs_final_state,
            )
        )
        return state_index

    def visit(
        segment_index: int,
        inherited_positions: tuple[int, ...],
        inherited_output_mask: tuple[bool, ...],
        parent_state_index: int,
        phase: int,
    ) -> None:
        segment = tree.segments[segment_index]
        positions = inherited_positions + tuple(range(segment.start, segment.end))
        output_mask = inherited_output_mask + (True,) * segment.length
        segment_children = children[segment_index]
        complete_length = len(positions) // chunk_size * chunk_size
        if segment_children and complete_length:
            parent_state_index = emit(
                positions[:complete_length],
                output_mask[:complete_length],
                parent_state_index,
                phase,
                True,
            )
            positions = positions[complete_length:]
            output_mask = output_mask[complete_length:]
            phase += 1
        if segment_children:
            for child_offset, child in enumerate(segment_children):
                visit(
                    child,
                    positions,
                    output_mask if child_offset == 0 else (False,) * len(output_mask),
                    parent_state_index,
                    phase,
                )
        else:
            emit(positions, output_mask, parent_state_index, phase, False)

    for segment in tree.segments:
        if segment.parent_index < 0:
            visit(segment.index, (), (), -1, 0)

    materialized = []
    state_rows: dict[int, int] = {}
    for columns in phases:
        buckets = _materialize_scan_phase(columns, chunk_size, state_rows, device)
        materialized.append(buckets)
        for bucket in buckets:
            if bucket.needs_final_state:
                for state_index in bucket.state_indices:
                    state_rows[state_index] = len(state_rows)
    return tuple(materialized)


def _materialize_scan_phase(
    columns: list[_ScanColumn],
    chunk_size: int,
    state_rows: dict[int, int],
    device: torch.device,
) -> tuple[MambaScanBucket, ...]:
    grouped: dict[tuple[bool, int], list[_ScanColumn]] = {}
    for column in columns:
        padded_length = (
            len(column.positions)
            if column.needs_final_state
            else (len(column.positions) + chunk_size - 1) // chunk_size * chunk_size
        )
        grouped.setdefault((column.needs_final_state, padded_length), []).append(column)
    return tuple(
        _materialize_scan_bucket(
            group,
            max(len(column.positions) for column in group),
            needs_state,
            state_rows,
            device,
        )
        for (needs_state, _), group in grouped.items()
    )


def _materialize_scan_bucket(
    columns: list[_ScanColumn],
    length: int,
    needs_final_state: bool,
    state_rows: dict[int, int],
    device: torch.device,
) -> MambaScanBucket:
    token_indices = torch.zeros((len(columns), length), dtype=torch.long)
    real_mask = torch.zeros((len(columns), length), dtype=torch.bool)
    output_mask = torch.zeros((len(columns), length), dtype=torch.bool)
    for row, column in enumerate(columns):
        count = len(column.positions)
        token_indices[row, :count] = torch.tensor(column.positions, dtype=torch.long)
        real_mask[row, :count] = True
        output_mask[row, :count] = torch.tensor(column.output_mask, dtype=torch.bool)
    output_rows = output_mask.flatten().nonzero().flatten()
    output_positions = token_indices.flatten()[output_rows]
    token_indices = torch.where(real_mask, token_indices, -1)
    return MambaScanBucket(
        state_indices=tuple(column.state_index for column in columns),
        parent_state_indices=tuple(column.parent_state_index for column in columns),
        parent_rows=torch.tensor(
            tuple(
                -1
                if column.parent_state_index < 0
                else state_rows[column.parent_state_index]
                for column in columns
            ),
            dtype=torch.long,
            device=device,
        ),
        token_indices=token_indices.to(device),
        output_rows=output_rows.to(device),
        output_positions=output_positions.to(device),
        needs_final_state=needs_final_state,
    )


def _build_exchange_plan(
    tree: RecurrentPrefixTree,
    *,
    device: torch.device,
    cp_rank: int,
    cp_size: int,
    token_layout: TokenLayoutIndex | None,
) -> MambaTokenExchangePlan:
    if cp_size == 1:
        if token_layout is not None and token_layout.token_counts_by_rank != (
            tree.token_count,
        ):
            raise ValueError(
                "CP1 recurrent token layout disagrees with the prefix tree"
            )
        positions_by_rank = (tuple(range(tree.token_count)),)
    else:
        if token_layout is None:
            raise ValueError("Mamba CP requires the ART attention token layout")
        if len(token_layout.token_counts_by_rank) != cp_size:
            raise ValueError("Mamba and attention CP sizes differ")
        if sum(token_layout.token_counts_by_rank) != tree.token_count:
            raise ValueError("Mamba and attention token counts differ")
        positions_by_rank = tuple(
            _local_to_global_positions(ranges, token_layout.token_counts_by_rank[rank])
            for rank, ranges in enumerate(token_layout.ownership_ranges_by_rank)
        )
    flattened = tuple(position for rank in positions_by_rank for position in rank)
    if tuple(sorted(flattened)) != tuple(range(tree.token_count)):
        raise ValueError(
            "attention token ownership must cover each recurrent token once"
        )
    tensors = tuple(
        torch.tensor(positions, dtype=torch.long, device=device)
        for positions in positions_by_rank
    )
    received_global_positions = torch.tensor(flattened, dtype=torch.long, device=device)
    return MambaTokenExchangePlan(
        cp_rank=cp_rank,
        cp_size=cp_size,
        source_token_counts=tuple(len(positions) for positions in positions_by_rank),
        global_positions_by_rank=tensors,
        canonical_to_received=_canonical_order(
            received_global_positions, tree.token_count
        ),
        physical_token_positions=torch.tensor(
            tree.physical_token_positions, dtype=torch.long, device=device
        ),
    )


def _remap_positions(
    positions: torch.Tensor, canonical_to_received: torch.Tensor
) -> torch.Tensor:
    valid = positions >= 0
    return torch.where(
        valid,
        canonical_to_received[positions.clamp_min(0)],
        -1,
    )


def _canonical_order(positions: torch.Tensor, token_count: int) -> torch.Tensor:
    if positions.numel() != token_count or not torch.equal(
        positions.sort().values,
        torch.arange(token_count, dtype=torch.long, device=positions.device),
    ):
        raise ValueError("Mamba row positions must be a complete permutation")
    return positions.argsort()


def _token_occurrences(positions: torch.Tensor, token_count: int) -> torch.Tensor:
    valid_occurrences = (positions >= 0).nonzero().flatten()
    valid_positions = positions[valid_occurrences]
    order = valid_positions.argsort(stable=True)
    sorted_positions = valid_positions[order]
    counts = torch.bincount(sorted_positions, minlength=token_count)
    if torch.any(counts == 0):
        raise ValueError("Mamba scan plan must contain every recurrent token")
    starts = counts.cumsum(0) - counts
    slots = (
        torch.arange(order.numel(), device=positions.device) - starts[sorted_positions]
    )
    occurrences = positions.new_full((token_count, int(counts.max())), -1)
    occurrences[sorted_positions, slots] = valid_occurrences[order]
    return occurrences


def _local_to_global_positions(
    ranges: tuple[tuple[int, int, int], ...],
    token_count: int,
) -> tuple[int, ...]:
    positions = [-1] * int(token_count)
    for start, end, local_start in ranges:
        for offset, global_position in enumerate(range(int(start), int(end))):
            positions[int(local_start) + offset] = global_position
    if any(position < 0 for position in positions):
        raise ValueError("attention token layout has a gap in local positions")
    return tuple(positions)
