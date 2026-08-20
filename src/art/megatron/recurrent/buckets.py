from __future__ import annotations

from bisect import bisect_left

from pydantic import BaseModel, ConfigDict
import torch

from .prefix_tree import RecurrentSegmentSpec

TokenRange = tuple[int, int, int]


class RecurrentSegmentBucketPlan(BaseModel):
    """Immutable indices for one variable-length recurrent segment batch."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    length: int
    lengths: torch.Tensor
    lengths_cpu: torch.Tensor
    real_mask: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    row_indices: torch.Tensor
    position_indices: torch.Tensor
    family_indices: torch.Tensor
    real_token_count_static: int
    lengths_by_rank_cpu: torch.Tensor | None = None
    family_indices_cpu: torch.Tensor | None = None
    parent_indices: torch.Tensor | None = None
    parent_indices_cpu: torch.Tensor | None = None
    dense_real_mask: torch.Tensor | None = None
    dense_token_indices: torch.Tensor | None = None
    flat_token_indices: torch.Tensor | None = None
    dense_row_indices: torch.Tensor | None = None
    dense_position_indices: torch.Tensor | None = None
    family_indices_cpu_tuple: tuple[int, ...] = ()
    parent_indices_cpu_tuple: tuple[int, ...] | None = None
    needs_final_state: bool = True
    output_mask: torch.Tensor | None = None

    @property
    def segment_count(self) -> int:
        return int(self.family_indices.numel())

    @property
    def real_token_count(self) -> int:
        return self.real_token_count_static


def build_recurrent_tree_bucket_plans(
    segments: tuple[RecurrentSegmentSpec, ...],
    tree_parent_indices: tuple[int, ...],
    tree_has_children: tuple[bool, ...],
    *,
    sequence_length: int,
    device: torch.device | str,
    local_token_ranges: tuple[TokenRange, ...] | None = None,
    token_ranges_by_rank: tuple[tuple[TokenRange, ...], ...] | None = None,
    split_by_final_state: bool = True,
    max_padded_tokens: int | None = None,
    build_dense_rows: bool = False,
) -> tuple[RecurrentSegmentBucketPlan, ...]:
    """Build deterministic length buckets for one tree depth."""

    del split_by_final_state
    segment_buckets = _batch_segments(segments, max_padded_tokens=max_padded_tokens)
    return tuple(
        _with_tree_parent_indices(
            (
                build_recurrent_segment_bucket_plan(
                    bucket,
                    sequence_length=sequence_length,
                    device=device,
                    build_dense_rows=build_dense_rows,
                )
                if local_token_ranges is None
                else _build_position_bucket_plan(
                    bucket,
                    local_token_ranges,
                    sequence_length=sequence_length,
                    device=device,
                    token_ranges_by_rank=token_ranges_by_rank,
                    build_dense_rows=build_dense_rows,
                )
            ),
            bucket,
            tree_parent_indices,
            tree_has_children,
            device=device,
        )
        for bucket in segment_buckets
    )


def build_recurrent_segment_bucket_plan(
    segments: tuple[RecurrentSegmentSpec, ...],
    *,
    sequence_length: int | None = None,
    device: torch.device | str,
    build_dense_rows: bool = False,
) -> RecurrentSegmentBucketPlan:
    """Build a packed and dense-row index plan for contiguous segments."""

    lengths_cpu = torch.tensor(
        [segment.length for segment in segments], dtype=torch.long
    )
    max_length = int(lengths_cpu.max().item())
    starts_cpu = torch.tensor([segment.start for segment in segments], dtype=torch.long)
    rows_cpu = torch.tensor(
        [segment.row_index for segment in segments], dtype=torch.long
    )
    offsets_cpu = torch.arange(max_length, dtype=torch.long).unsqueeze(1)
    return build_recurrent_bucket_plan(
        segments,
        lengths_cpu=lengths_cpu,
        row_indices_cpu=rows_cpu.unsqueeze(0).expand(max_length, -1).contiguous(),
        position_indices_cpu=starts_cpu.unsqueeze(0) + offsets_cpu,
        flat_sequence_length=sequence_length,
        device=device,
        build_dense_rows=build_dense_rows,
    )


def build_recurrent_bucket_plan(
    segments: tuple[RecurrentSegmentSpec, ...],
    *,
    lengths_cpu: torch.Tensor,
    row_indices_cpu: torch.Tensor,
    position_indices_cpu: torch.Tensor,
    device: torch.device | str,
    lengths_by_rank_cpu: torch.Tensor | None = None,
    flat_sequence_length: int | None = None,
    build_dense_rows: bool = False,
) -> RecurrentSegmentBucketPlan:
    """Build compact metadata and optional dense rows from CPU column indices."""

    max_length = int(lengths_cpu.max().item())
    if (
        int(row_indices_cpu.shape[0]) < max_length
        or int(position_indices_cpu.shape[0]) < max_length
    ):
        raise ValueError("bucket index tensors are shorter than max segment length")
    dense_real_mask_cpu = None
    dense_token_indices_cpu = None
    flat_token_indices_cpu = None
    dense_rows_cpu = None
    dense_positions_cpu = None
    if build_dense_rows:
        offsets = torch.arange(max_length, dtype=torch.long).unsqueeze(1)
        dense_real_mask_columns = offsets < lengths_cpu.unsqueeze(0)
        dense_real_mask_cpu = dense_real_mask_columns.T.contiguous()
        dense_rows_cpu = torch.where(
            dense_real_mask_columns,
            row_indices_cpu[:max_length],
            torch.zeros_like(offsets),
        ).T.contiguous()
        dense_positions_cpu = torch.where(
            dense_real_mask_columns,
            position_indices_cpu[:max_length],
            torch.zeros_like(offsets),
        ).T.contiguous()
        dense_token_indices_cpu = _pack_column_major(
            torch.arange(max_length * len(segments), dtype=torch.long)
            .view(len(segments), max_length)
            .T,
            lengths_cpu,
        )
    row_indices_cpu = _pack_column_major(row_indices_cpu, lengths_cpu)
    position_indices_cpu = _pack_column_major(position_indices_cpu, lengths_cpu)
    if build_dense_rows and flat_sequence_length is not None:
        flat_token_indices_cpu = (
            row_indices_cpu * flat_sequence_length + position_indices_cpu
        )
    real_mask_cpu = torch.ones(int(lengths_cpu.sum().item()), dtype=torch.bool)
    cu_seqlens_cpu = torch.cat(
        [lengths_cpu.new_zeros(1), torch.cumsum(lengths_cpu, dim=0)]
    )
    family_indices_cpu = torch.tensor(
        [segment.family_index for segment in segments], dtype=torch.long
    )
    return RecurrentSegmentBucketPlan(
        length=max_length,
        lengths=_move_tensor(lengths_cpu, device),
        lengths_cpu=lengths_cpu,
        lengths_by_rank_cpu=lengths_by_rank_cpu,
        real_mask=_move_tensor(real_mask_cpu, device),
        dense_real_mask=(
            _move_tensor(dense_real_mask_cpu, device)
            if dense_real_mask_cpu is not None
            else None
        ),
        dense_token_indices=(
            _move_tensor(dense_token_indices_cpu, device)
            if dense_token_indices_cpu is not None
            else None
        ),
        cu_seqlens=_move_tensor(cu_seqlens_cpu, device),
        cu_seqlens_cpu=cu_seqlens_cpu,
        row_indices=_move_tensor(row_indices_cpu, device),
        position_indices=_move_tensor(position_indices_cpu, device),
        flat_token_indices=(
            _move_tensor(flat_token_indices_cpu, device)
            if flat_token_indices_cpu is not None
            else None
        ),
        dense_row_indices=(
            _move_tensor(dense_rows_cpu, device) if dense_rows_cpu is not None else None
        ),
        dense_position_indices=(
            _move_tensor(dense_positions_cpu, device)
            if dense_positions_cpu is not None
            else None
        ),
        family_indices=_move_tensor(family_indices_cpu, device),
        family_indices_cpu=family_indices_cpu,
        family_indices_cpu_tuple=tuple(segment.family_index for segment in segments),
        real_token_count_static=int(lengths_cpu.sum().item()),
    )


def move_recurrent_segment_bucket_plans(
    buckets: tuple[RecurrentSegmentBucketPlan, ...],
    device: torch.device | str,
) -> tuple[RecurrentSegmentBucketPlan, ...]:
    """Materialize execution tensors while retaining CPU planning metadata."""

    tensor_fields = (
        "lengths",
        "real_mask",
        "dense_real_mask",
        "dense_token_indices",
        "cu_seqlens",
        "row_indices",
        "position_indices",
        "flat_token_indices",
        "dense_row_indices",
        "dense_position_indices",
        "family_indices",
        "parent_indices",
        "output_mask",
    )
    return tuple(
        bucket.model_copy(
            update={
                name: _move_tensor(value, device)
                for name in tensor_fields
                if (value := getattr(bucket, name)) is not None
            }
        )
        for bucket in buckets
    )


def _with_tree_parent_indices(
    plan: RecurrentSegmentBucketPlan,
    segments: tuple[RecurrentSegmentSpec, ...],
    tree_parent_indices: tuple[int, ...],
    tree_has_children: tuple[bool, ...],
    *,
    device: torch.device | str,
) -> RecurrentSegmentBucketPlan:
    parent_indices_cpu = torch.tensor(
        [tree_parent_indices[segment.family_index] for segment in segments],
        dtype=torch.long,
    )
    return plan.model_copy(
        update={
            "parent_indices": _move_tensor(parent_indices_cpu, device),
            "parent_indices_cpu": parent_indices_cpu,
            "parent_indices_cpu_tuple": tuple(
                tree_parent_indices[segment.family_index] for segment in segments
            ),
            "needs_final_state": any(
                tree_has_children[segment.family_index] for segment in segments
            ),
        }
    )


def _build_position_bucket_plan(
    segments: tuple[RecurrentSegmentSpec, ...],
    local_token_ranges: tuple[TokenRange, ...],
    *,
    sequence_length: int,
    device: torch.device | str,
    token_ranges_by_rank: tuple[tuple[TokenRange, ...], ...] | None,
    build_dense_rows: bool,
) -> RecurrentSegmentBucketPlan:
    range_positions = {
        (start, end): position for start, end, position in local_token_ranges
    }
    starts: list[int] = []
    lengths: list[int] = []
    for segment in segments:
        token_start = _segment_token_start(segment, sequence_length)
        position_start = range_positions.get(
            (token_start, token_start + segment.length)
        )
        if position_start is None:
            break
        starts.append(position_start)
        lengths.append(segment.length)
    else:
        starts_cpu = torch.tensor(starts, dtype=torch.long)
        lengths_cpu = torch.tensor(lengths, dtype=torch.long)
        offsets_cpu = torch.arange(max(lengths), dtype=torch.long).unsqueeze(1)
        position_indices_cpu = torch.where(
            offsets_cpu < lengths_cpu.unsqueeze(0),
            starts_cpu.unsqueeze(0) + offsets_cpu,
            torch.zeros_like(offsets_cpu),
        )
        return build_recurrent_bucket_plan(
            segments,
            lengths_cpu=lengths_cpu,
            row_indices_cpu=torch.zeros_like(position_indices_cpu),
            position_indices_cpu=position_indices_cpu,
            flat_sequence_length=0,
            lengths_by_rank_cpu=_bucket_lengths_by_rank(
                segments, token_ranges_by_rank, sequence_length=sequence_length
            ),
            device=device,
            build_dense_rows=build_dense_rows,
        )

    local_positions = tuple(
        _local_positions_for_segment(
            segment,
            sequence_length=sequence_length,
            local_token_ranges=local_token_ranges,
        )
        for segment in segments
    )
    for segment, positions in zip(segments, local_positions, strict=True):
        if not positions.numel():
            raise ValueError(
                "planned recurrent bucket contains a segment with no local tokens; "
                f"family={segment.family_index} kind={segment.kind}"
            )
    lengths_cpu = torch.tensor([positions.numel() for positions in local_positions])
    position_indices_cpu = torch.zeros(
        int(lengths_cpu.max().item()), len(segments), dtype=torch.long
    )
    for column, positions in enumerate(local_positions):
        position_indices_cpu[: positions.numel(), column] = positions
    return build_recurrent_bucket_plan(
        segments,
        lengths_cpu=lengths_cpu,
        row_indices_cpu=torch.zeros_like(position_indices_cpu),
        position_indices_cpu=position_indices_cpu,
        flat_sequence_length=0,
        lengths_by_rank_cpu=_bucket_lengths_by_rank(
            segments, token_ranges_by_rank, sequence_length=sequence_length
        ),
        device=device,
        build_dense_rows=build_dense_rows,
    )


def _bucket_lengths_by_rank(
    segments: tuple[RecurrentSegmentSpec, ...],
    token_ranges_by_rank: tuple[tuple[TokenRange, ...], ...] | None,
    *,
    sequence_length: int,
) -> torch.Tensor | None:
    if token_ranges_by_rank is None:
        return None
    return torch.tensor(
        [
            [
                sum(
                    max(0, min(end, range_end) - max(start, range_start))
                    for range_start, range_end, _ in rank_ranges
                )
                for segment in segments
                for start, end in (
                    (
                        _segment_token_start(segment, sequence_length),
                        _segment_token_start(segment, sequence_length) + segment.length,
                    ),
                )
            ]
            for rank_ranges in token_ranges_by_rank
        ],
        dtype=torch.long,
    )


def _batch_segments(
    segments: tuple[RecurrentSegmentSpec, ...],
    *,
    max_padded_tokens: int | None,
) -> tuple[tuple[RecurrentSegmentSpec, ...], ...]:
    ordered = tuple(
        sorted(segments, key=lambda segment: (segment.length, segment.family_index))
    )
    if not ordered or max_padded_tokens is None:
        return (ordered,) if ordered else ()
    if max_padded_tokens < 1:
        raise ValueError("max_padded_tokens must be positive")
    batches: list[tuple[RecurrentSegmentSpec, ...]] = []
    current: list[RecurrentSegmentSpec] = []
    for segment in ordered:
        if current and segment.length * (len(current) + 1) > max_padded_tokens:
            batches.append(tuple(current))
            current = []
        current.append(segment)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _local_positions_for_segment(
    segment: RecurrentSegmentSpec,
    *,
    sequence_length: int,
    local_token_ranges: tuple[TokenRange, ...],
) -> torch.Tensor:
    segment_start = _segment_token_start(segment, sequence_length)
    segment_end = segment_start + segment.length
    range_ends = tuple(token_end for _, token_end, _ in local_token_ranges)
    pieces = []
    for token_start, token_end, position_start in local_token_ranges[
        bisect_left(range_ends, segment_start + 1) :
    ]:
        if token_start >= segment_end:
            break
        overlap_start = max(segment_start, token_start)
        overlap_end = min(segment_end, token_end)
        if overlap_start < overlap_end:
            pieces.append(
                torch.arange(
                    position_start + overlap_start - token_start,
                    position_start + overlap_end - token_start,
                    dtype=torch.long,
                )
            )
    return torch.cat(pieces) if pieces else torch.empty((0,), dtype=torch.long)


def _pack_column_major(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    pieces = [
        values[: int(length), column]
        for column, length in enumerate(lengths.tolist())
        if int(length) > 0
    ]
    return torch.cat(pieces).contiguous() if pieces else values.new_empty((0,))


def _move_tensor(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    target = torch.device(device)
    return tensor if target.type == "cpu" else tensor.to(device=target)


def _segment_token_start(segment: RecurrentSegmentSpec, sequence_length: int) -> int:
    return segment.row_index * sequence_length + segment.start
