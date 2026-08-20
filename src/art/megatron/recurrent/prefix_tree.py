from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from art.megatron.prefix_tree import parse_prefix_tree

SegmentKind = Literal["prefix", "completion"]


class RecurrentSegmentSpec(BaseModel):
    """One contiguous physical segment in a packed prefix-tree row."""

    model_config = ConfigDict(frozen=True)

    row_index: int = Field(ge=0)
    family_index: int = Field(ge=0)
    group_id: int
    parent_id: int
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    kind: SegmentKind
    child_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_range(self) -> RecurrentSegmentSpec:
        if self.end <= self.start:
            raise ValueError("recurrent tree segments must be non-empty")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start


class RecurrentPackedExecutionSpec(BaseModel):
    """Immutable CPU prefix-tree DAG shared by recurrent layer families."""

    model_config = ConfigDict(frozen=True)

    schema_identity: Literal["art.linear_recurrent.prefix_tree.v1"] = (
        "art.linear_recurrent.prefix_tree.v1"
    )
    batch_size: int = Field(ge=0)
    sequence_length: int = Field(ge=0)
    valid_lengths: tuple[int, ...]
    tree_segments: tuple[RecurrentSegmentSpec, ...]
    tree_parent_indices: tuple[int, ...]
    tree_depths: tuple[int, ...]

    @model_validator(mode="after")
    def _validate_dag(self) -> RecurrentPackedExecutionSpec:
        node_count = len(self.tree_segments)
        if len(self.valid_lengths) != self.batch_size:
            raise ValueError("valid_lengths must contain one entry per packed row")
        if any(
            length < 0 or length > self.sequence_length for length in self.valid_lengths
        ):
            raise ValueError("valid lengths must lie within the packed sequence length")
        if (
            len(self.tree_parent_indices) != node_count
            or len(self.tree_depths) != node_count
        ):
            raise ValueError("tree parent and depth metadata must match tree segments")
        for node_index, (segment, parent_index, depth) in enumerate(
            zip(
                self.tree_segments,
                self.tree_parent_indices,
                self.tree_depths,
                strict=True,
            )
        ):
            if segment.family_index != node_index:
                raise ValueError(
                    "recurrent family indices must be contiguous DAG indices"
                )
            if (
                segment.row_index >= self.batch_size
                or segment.end > self.valid_lengths[segment.row_index]
            ):
                raise ValueError("recurrent segment lies outside its packed row")
            if parent_index < 0:
                if (
                    depth != 0
                    or segment.kind != "prefix"
                    or segment.child_index is not None
                ):
                    raise ValueError("root recurrent segments require prefix metadata")
            elif (
                parent_index >= node_index
                or depth != self.tree_depths[parent_index] + 1
                or segment.kind != "completion"
                or segment.child_index is None
            ):
                raise ValueError(
                    "child recurrent segments require an earlier direct parent"
                )
        return self

    @property
    def family_count(self) -> int:
        return len(self.tree_segments)

    @property
    def real_token_count(self) -> int:
        return sum(self.valid_lengths)


def parse_recurrent_prefix_tree_segments(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> RecurrentPackedExecutionSpec:
    """Parse packed ART prefix metadata once for every recurrent family."""

    groups = _rank2_long_cpu("group_ids", group_ids)
    parents = _rank2_long_cpu("parent_ids", parent_ids)
    if tuple(groups.shape) != tuple(parents.shape):
        raise ValueError(
            "group_ids and parent_ids must have the same shape, got "
            f"{tuple(groups.shape)} and {tuple(parents.shape)}"
        )

    batch_size, sequence_length = map(int, groups.shape)
    rows = parse_prefix_tree(group_ids=groups, parent_ids=parents)
    tree_segments: list[RecurrentSegmentSpec] = []
    tree_parent_indices: list[int] = []
    tree_depths: list[int] = []
    node_by_row_group: dict[tuple[int, int], int] = {}
    child_counts_by_parent: dict[int, int] = {}

    for row in rows:
        for segment in row.segments:
            node_index = len(tree_segments)
            parent_index = (
                -1
                if segment.depth == 0
                else node_by_row_group[(row.row_index, segment.parent_id)]
            )
            child_index = None
            if parent_index >= 0:
                child_index = child_counts_by_parent.get(parent_index, 0)
                child_counts_by_parent[parent_index] = child_index + 1
            tree_segments.append(
                RecurrentSegmentSpec(
                    row_index=row.row_index,
                    family_index=node_index,
                    group_id=segment.group_id,
                    parent_id=segment.parent_id,
                    start=segment.start,
                    end=segment.end,
                    kind="prefix" if parent_index < 0 else "completion",
                    child_index=child_index,
                )
            )
            tree_parent_indices.append(parent_index)
            tree_depths.append(segment.depth)
            node_by_row_group[(row.row_index, segment.group_id)] = node_index

    return RecurrentPackedExecutionSpec(
        batch_size=batch_size,
        sequence_length=sequence_length,
        valid_lengths=tuple(row.valid_tokens for row in rows),
        tree_segments=tuple(tree_segments),
        tree_parent_indices=tuple(tree_parent_indices),
        tree_depths=tuple(tree_depths),
    )


def _rank2_long_cpu(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 2 [batch, sequence], got {tensor.ndim}")
    if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"{name} must contain integer ids, got dtype={tensor.dtype}")
    return tensor.detach().to(device="cpu", dtype=torch.long)
