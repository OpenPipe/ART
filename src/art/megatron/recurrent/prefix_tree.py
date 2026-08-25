from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.megatron.prefix_tree import parse_prefix_tree


class RecurrentSegment(BaseModel):
    """One contiguous prefix-tree node in compact packed-token order."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    parent_index: int = Field(ge=-1)
    row_index: int = Field(ge=0)
    group_id: int
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    depth: int = Field(ge=0)

    @property
    def length(self) -> int:
        return self.end - self.start


class RecurrentPrefixTree(BaseModel):
    """Model-neutral recurrent execution topology for a packed prefix tree."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    valid_lengths: tuple[int, ...]
    segments: tuple[RecurrentSegment, ...]

    @property
    def token_count(self) -> int:
        return sum(self.valid_lengths)

    @property
    def physical_token_positions(self) -> tuple[int, ...]:
        """Compact-token gather indices into a flattened [sequence, batch] tensor."""

        return tuple(
            position * self.batch_size + row
            for row, length in enumerate(self.valid_lengths)
            for position in range(length)
        )


def parse_recurrent_prefix_tree(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> RecurrentPrefixTree:
    """Parse ART metadata once for any stateful linear-recurrent family."""

    groups = _rank2_long_cpu("group_ids", group_ids)
    parents = _rank2_long_cpu("parent_ids", parent_ids)
    if groups.shape != parents.shape:
        raise ValueError(
            "group_ids and parent_ids must share shape, got "
            f"{tuple(groups.shape)} and {tuple(parents.shape)}"
        )
    rows = parse_prefix_tree(group_ids=groups, parent_ids=parents)
    valid_lengths = tuple(row.valid_tokens for row in rows)
    row_offsets = []
    cursor = 0
    for length in valid_lengths:
        row_offsets.append(cursor)
        cursor += length

    segments = []
    segment_by_group: dict[tuple[int, int], int] = {}
    for row in rows:
        for segment in row.segments:
            index = len(segments)
            parent_index = (
                -1
                if segment.depth == 0
                else segment_by_group[(row.row_index, segment.parent_id)]
            )
            segments.append(
                RecurrentSegment(
                    index=index,
                    parent_index=parent_index,
                    row_index=row.row_index,
                    group_id=segment.group_id,
                    start=row_offsets[row.row_index] + segment.start,
                    end=row_offsets[row.row_index] + segment.end,
                    depth=segment.depth,
                )
            )
            segment_by_group[(row.row_index, segment.group_id)] = index

    return RecurrentPrefixTree(
        batch_size=int(groups.shape[0]),
        sequence_length=int(groups.shape[1]),
        valid_lengths=valid_lengths,
        segments=tuple(segments),
    )


def _rank2_long_cpu(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 1 or 2, got {tensor.ndim}")
    return tensor.detach().to(device="cpu", dtype=torch.long).contiguous()
