from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.recurrent import RecurrentPrefixTree


class MambaConvBucket(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    segment_indices: tuple[int, ...]
    parent_indices: tuple[int, ...]
    token_indices: torch.Tensor
    cu_seqlens: torch.Tensor


class MambaScanBucket(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    state_indices: tuple[int, ...]
    parent_state_indices: tuple[int, ...]
    token_indices: torch.Tensor
    real_mask: torch.Tensor
    output_mask: torch.Tensor
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
    received_global_positions: torch.Tensor
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
    exchange: MambaTokenExchangePlan
    chunk_size: int = Field(gt=0)


class _ScanColumn(BaseModel):
    model_config = ConfigDict(frozen=True)

    state_index: int
    parent_state_index: int
    positions: tuple[int, ...]
    output_mask: tuple[bool, ...]
    needs_final_state: bool


def build_mamba_execution_plan(
    tree: RecurrentPrefixTree,
    *,
    device: torch.device,
    cp_rank: int,
    cp_size: int,
    token_layout: TokenLayoutIndex | None,
    chunk_size: int = 128,
) -> MambaExecutionPlan:
    """Materialize fixed-shape tree, scan, and CP metadata once per packed row."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    exchange = _build_exchange_plan(
        tree,
        device=device,
        cp_rank=cp_rank,
        cp_size=cp_size,
        token_layout=token_layout,
    )
    return MambaExecutionPlan(
        tree=tree,
        conv_buckets=_build_conv_buckets(tree, device),
        scan_phases=_build_scan_phases(tree, chunk_size, device),
        exchange=exchange,
        chunk_size=chunk_size,
    )


def _build_conv_buckets(
    tree: RecurrentPrefixTree,
    device: torch.device,
) -> tuple[MambaConvBucket, ...]:
    buckets = []
    for depth in range(
        max((segment.depth for segment in tree.segments), default=-1) + 1
    ):
        segments = tuple(
            sorted(
                (segment for segment in tree.segments if segment.depth == depth),
                key=lambda segment: (segment.length, segment.index),
            )
        )
        if not segments:
            continue
        lengths = tuple(segment.length for segment in segments)
        cu_seqlens = torch.tensor(
            (0, *torch.tensor(lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        buckets.append(
            MambaConvBucket(
                segment_indices=tuple(segment.index for segment in segments),
                parent_indices=tuple(segment.parent_index for segment in segments),
                token_indices=torch.tensor(
                    tuple(
                        position
                        for segment in segments
                        for position in range(segment.start, segment.end)
                    ),
                    dtype=torch.long,
                    device=device,
                ),
                cu_seqlens=cu_seqlens,
            )
        )
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

    return tuple(
        _materialize_scan_phase(columns, chunk_size, device) for columns in phases
    )


def _materialize_scan_phase(
    columns: list[_ScanColumn],
    chunk_size: int,
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
            device,
        )
        for (needs_state, _), group in grouped.items()
    )


def _materialize_scan_bucket(
    columns: list[_ScanColumn],
    length: int,
    needs_final_state: bool,
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
    return MambaScanBucket(
        state_indices=tuple(column.state_index for column in columns),
        parent_state_indices=tuple(column.parent_state_index for column in columns),
        token_indices=token_indices.to(device),
        real_mask=real_mask.to(device),
        output_mask=output_mask.to(device),
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
    return MambaTokenExchangePlan(
        cp_rank=cp_rank,
        cp_size=cp_size,
        source_token_counts=tuple(len(positions) for positions in positions_by_rank),
        global_positions_by_rank=tensors,
        received_global_positions=torch.tensor(
            flattened, dtype=torch.long, device=device
        ),
        physical_token_positions=torch.tensor(
            tree.physical_token_positions, dtype=torch.long, device=device
        ),
    )


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
