from collections.abc import Iterable, Sequence
import os
import random
import time
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
from typing_extensions import NotRequired, TypedDict, Unpack

from ..megatron.prefix_tree_packing import (
    PrefixTreePackSegment,
)
from ..megatron.prefix_tree_packing import (
    prefix_tree_pack_segments as _prefix_tree_pack_segments,
)
from ..types import Verbosity
from .moe_routing import (
    MoeRouteArray,
    MoeRouteSegments,
    MoeRoutingPackStats,
    PackedMoeRoutingReplay,
    deterministic_moe_routes,
    moe_route_dtype,
)
from .tokenize import TokenizedResult

DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH = 64


class PrefixTreePackingStats(TypedDict):
    logical_tokens: int
    physical_tokens: int


class PackedTensors(TypedDict):
    tokens: torch.Tensor
    group_ids: torch.Tensor
    parent_ids: torch.Tensor
    input_pos: torch.Tensor
    assistant_mask: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor
    pixel_values: list[torch.Tensor | None]
    image_grid_thw: list[torch.Tensor | None]
    moe_routing_replay: PackedMoeRoutingReplay | None
    prefix_tree_packing_stats: NotRequired[PrefixTreePackingStats]
    original_logprobs: NotRequired[torch.Tensor]


class DiskPackedTensors(TypedDict):
    dir: str
    num_sequences: int
    sequence_length: int
    pixel_values: NotRequired[tuple[int, list[int]]]
    image_grid_thw: NotRequired[tuple[int, list[int]]]


class _PrefixTreePackItem(NamedTuple):
    token_ids: tuple[int, ...]
    input_pos: np.ndarray
    assistant_mask: np.ndarray
    logprobs: np.ndarray
    advantage: float
    weight: float
    prompt_id: int
    shareable_length: int
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    moe_routes: MoeRouteArray | MoeRouteSegments | None


class _PrefixTreeRowPlan(NamedTuple):
    segments: tuple[PrefixTreePackSegment, ...]
    length: int


class _PrefixTreeLeaf(NamedTuple):
    item_index: int
    packing_group_id: int
    segment_path: tuple[tuple[int, int], ...]
    empty_bin_cost: int


class _PrefixTreeBin:
    __slots__ = ("leaves", "occupied_segments", "token_count")

    def __init__(self) -> None:
        self.leaves: list[_PrefixTreeLeaf] = []
        self.occupied_segments: set[int] = set()
        self.token_count = 0

    def insertion_delta(self, leaf: _PrefixTreeLeaf) -> int:
        return sum(
            length
            for segment_id, length in leaf.segment_path
            if segment_id not in self.occupied_segments
        )

    def add(self, leaf: _PrefixTreeLeaf) -> None:
        self.token_count += self.insertion_delta(leaf)
        self.occupied_segments.update(segment_id for segment_id, _ in leaf.segment_path)
        self.leaves.append(leaf)


class PrefixTreePackingEstimate(NamedTuple):
    packed_sequences: int
    non_padding_tokens: int


class PrefixTreePackingPool:
    __slots__ = ("group_costs", "groups")

    def __init__(
        self,
        groups: Sequence[Sequence[tuple[Sequence[int], int]]],
        *,
        min_shared_segment_length: int = DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH,
    ) -> None:
        prefixes: list[Sequence[int]] = []
        leaf_specs: list[tuple[int, int | None, int]] = []
        for group_id, group in enumerate(groups):
            group_prefixes: list[tuple[Sequence[int], int]] = []
            for tokens, shareable_length in group:
                prefix = tokens[:shareable_length]
                prefix_index = next(
                    (
                        index
                        for candidate, index in group_prefixes
                        if candidate == prefix
                    ),
                    None,
                )
                if prefix_index is None and shareable_length > 0:
                    prefix_index = len(prefixes)
                    prefixes.append(prefix)
                    group_prefixes.append((prefix, prefix_index))
                leaf_specs.append(
                    (group_id, prefix_index, len(tokens) - shareable_length)
                )
        if not leaf_specs:
            raise ValueError("Prefix-tree packing pool requires at least one leaf")
        segments = (
            _prefix_tree_pack_segments(
                prefixes,
                max_depth=max(map(len, prefixes)),
                shareable_lengths=map(len, prefixes),
                min_shared_segment_length=min_shared_segment_length,
            )
            if prefixes
            else ()
        )
        paths: list[list[tuple[int, int]]] = [[] for _ in prefixes]
        for segment_id, segment in enumerate(segments):
            path_segment = (segment_id, segment.length)
            for prefix_index in segment.sequence_indices:
                paths[prefix_index].append(path_segment)
        next_segment_id = len(segments)
        leaves = []
        for index, (group_id, prefix_index, tail_length) in enumerate(leaf_specs):
            segment_path = tuple(
                () if prefix_index is None else paths[prefix_index]
            ) + (((next_segment_id + index, tail_length),) if tail_length > 0 else ())
            leaves.append(
                _PrefixTreeLeaf(
                    item_index=index,
                    packing_group_id=group_id,
                    segment_path=segment_path,
                    empty_bin_cost=sum(length for _, length in segment_path),
                ),
            )
        grouped_leaves: list[list[_PrefixTreeLeaf]] = [[] for _ in groups]
        for leaf in leaves:
            grouped_leaves[leaf.packing_group_id].append(leaf)
        self.groups = tuple(tuple(group) for group in grouped_leaves)
        self.group_costs = tuple(
            max(leaf.empty_bin_cost for leaf in group) for group in self.groups
        )

    def estimate(
        self, group_indices: Sequence[int], *, seq_len: int
    ) -> PrefixTreePackingEstimate:
        ordered = sorted(
            group_indices,
            key=lambda index: self.group_costs[index],
            reverse=True,
        )
        bins = _place_prefix_tree_leaves(
            (self.groups[index] for index in ordered),
            seq_len=seq_len,
            groups_are_ordered=True,
        )
        return PrefixTreePackingEstimate(
            packed_sequences=len(bins),
            non_padding_tokens=sum(packed_bin.token_count for packed_bin in bins),
        )


def packed_tensors_from_tokenized_results(
    tokenized_results: list[TokenizedResult],
    seq_len: int,
    pad_token_id: int = -100,
    truncate_long_results: bool = True,
    advantage_balance: float = 0.0,
    verbosity: Verbosity = 1,
    pack_results: bool = True,
    include_moe_routing: bool = False,
    min_prefix_tree_shared_segment_length: int = (
        DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH
    ),
) -> PackedTensors:
    return prefix_tree_pack(
        tokenized_results=tokenized_results,
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        truncate_long_results=truncate_long_results,
        advantage_balance=advantage_balance,
        verbosity=verbosity,
        pack_results=pack_results,
        include_moe_routing=include_moe_routing,
        min_prefix_tree_shared_segment_length=(min_prefix_tree_shared_segment_length),
    )


def prefix_tree_pack(
    *,
    tokenized_results: list[TokenizedResult],
    seq_len: int,
    pad_token_id: int = -100,
    truncate_long_results: bool = True,
    advantage_balance: float = 0.0,
    verbosity: Verbosity = 1,
    pack_results: bool = True,
    include_moe_routing: bool = False,
    min_prefix_tree_shared_segment_length: int = (
        DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH
    ),
) -> PackedTensors:
    if min_prefix_tree_shared_segment_length < 0:
        raise ValueError("min_prefix_tree_shared_segment_length must be >= 0")
    items: list[_PrefixTreePackItem] = []
    moe_routing_pack_stats = MoeRoutingPackStats()

    for result in tokenized_results:
        if len(result.token_ids) > seq_len and not truncate_long_results:
            if verbosity > 1:
                print("Result is too long, skipping")
            continue
        if include_moe_routing and result.moe_routed_experts is None:
            raise RuntimeError(
                "MoE routing replay from trajectories was requested, but a "
                "tokenized result has no aligned routed experts"
            )
        if sum(result.assistant_mask[result.prompt_length :]) == 0:
            if verbosity > 1:
                print("Result has no unique completion tokens, skipping")
            continue
        item = _prefix_tree_pack_item(result, seq_len=seq_len)
        if truncate_long_results:
            item = _truncate_prefix_tree_pack_item(item, seq_len)
        items.append(item)

    planned_rows = _prefix_tree_pack_rows(
        items,
        seq_len=seq_len,
        pack_results=pack_results,
        min_shared_segment_length=min_prefix_tree_shared_segment_length,
    )
    if not planned_rows:
        raise RuntimeError("No tokenized results were packable")
    random.shuffle(planned_rows)
    rows = [row for row, _ in planned_rows]
    row_plans = [plan for _, plan in planned_rows]

    num_sequences = len(rows)
    tokens_np = np.full((num_sequences, seq_len), pad_token_id, dtype=np.int64)
    group_ids_np = np.full((num_sequences, seq_len), -1, dtype=np.int64)
    parent_ids_np = np.full((num_sequences, seq_len), -1, dtype=np.int64)
    input_pos_np = np.zeros((num_sequences, seq_len), dtype=np.int64)
    assistant_mask_np = np.zeros((num_sequences, seq_len), dtype=np.bool_)
    logprobs_np = np.full((num_sequences, seq_len), np.nan, dtype=np.float32)
    advantages_np = np.zeros((num_sequences, seq_len), dtype=np.float32)
    weights_np = np.zeros((num_sequences, seq_len), dtype=np.float32)
    pixel_values: list[torch.Tensor | None] = []
    image_grid_thw: list[torch.Tensor | None] = []
    route_contract = _moe_route_contract(rows) if include_moe_routing else None
    route_tensor_np: np.ndarray | None = None
    if include_moe_routing:
        if route_contract is None:
            raise RuntimeError("No MoE routes were packed")
        num_experts, num_layers, topk = route_contract
        padding = deterministic_moe_routes(
            np.arange(seq_len, dtype=np.int64),
            route_shape=(num_layers, topk),
            num_experts=num_experts,
        )
        route_tensor_np = np.broadcast_to(
            np.moveaxis(padding, 1, 0)[:, None],
            (num_layers, num_sequences, seq_len, topk),
        ).copy()

    for index, (row, plan) in enumerate(zip(rows, row_plans, strict=True)):
        row_route_tensor = (
            route_tensor_np[:, index] if route_tensor_np is not None else None
        )
        _materialize_prefix_tree_row(
            row,
            plan=plan,
            token_ids=tokens_np[index],
            group_ids=group_ids_np[index],
            parent_ids=parent_ids_np[index],
            input_pos=input_pos_np[index],
            assistant_mask=assistant_mask_np[index],
            logprobs=logprobs_np[index],
            advantages=advantages_np[index],
            weights=weights_np[index],
            route_tensor=row_route_tensor,
            route_shape=(None if route_contract is None else route_contract[1:]),
            include_moe_routing=include_moe_routing,
        )
        pixel_values.append(_packed_row_tensor_list(row, "pixel_values"))
        image_grid_thw.append(_packed_row_tensor_list(row, "image_grid_thw"))
    assistant_mask_tensor = torch.from_numpy(assistant_mask_np)
    weights_tensor = torch.from_numpy(weights_np)
    weights_tensor = torch.where(
        assistant_mask_tensor, weights_tensor, torch.zeros_like(weights_tensor)
    )
    if bool(assistant_mask_tensor.any()):
        weights_tensor[assistant_mask_tensor] /= weights_tensor[
            assistant_mask_tensor
        ].mean()
    advantages_tensor = torch.from_numpy(advantages_np)
    advantages_tensor = torch.where(
        assistant_mask_tensor, advantages_tensor, torch.zeros_like(advantages_tensor)
    )
    if advantage_balance > 0.0:
        advantages_tensor = torch.where(
            advantages_tensor > 0,
            advantages_tensor,
            advantages_tensor * (1 - advantage_balance),
        )
    elif advantage_balance < 0.0:
        advantages_tensor = torch.where(
            advantages_tensor < 0,
            advantages_tensor,
            advantages_tensor * (1 + advantage_balance),
        )
    if bool(assistant_mask_tensor.any()):
        advantages_tensor[assistant_mask_tensor] /= (
            advantages_tensor[assistant_mask_tensor].abs()
            * weights_tensor[assistant_mask_tensor]
        ).mean()

    packed_tensors: PackedTensors = {
        "tokens": torch.from_numpy(tokens_np),
        "group_ids": torch.from_numpy(group_ids_np),
        "parent_ids": torch.from_numpy(parent_ids_np),
        "input_pos": torch.from_numpy(input_pos_np),
        "assistant_mask": assistant_mask_tensor,
        "logprobs": torch.from_numpy(logprobs_np),
        "advantages": advantages_tensor,
        "weights": weights_tensor,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "moe_routing_replay": None,
        "prefix_tree_packing_stats": {
            "logical_tokens": sum(len(item.token_ids) for item in items),
            "physical_tokens": sum(plan.length for plan in row_plans),
        },
    }
    if include_moe_routing:
        assert route_tensor_np is not None and route_contract is not None
        num_experts, _num_layers, _topk = route_contract
        moe_routing_pack_stats.packed_tokens = sum(plan.length for plan in row_plans)
        packed_tensors["moe_routing_replay"] = PackedMoeRoutingReplay(
            expert_indices=torch.from_numpy(route_tensor_np),
            num_experts=num_experts,
            pack_stats=moe_routing_pack_stats,
        )
    return packed_tensors


def _prefix_tree_pack_rows(
    items: list[_PrefixTreePackItem],
    *,
    seq_len: int,
    pack_results: bool,
    min_shared_segment_length: int,
) -> list[tuple[list[_PrefixTreePackItem], _PrefixTreeRowPlan]]:
    if not items:
        return []
    if not pack_results:
        return [
            (
                [item],
                _prefix_tree_row_plan(
                    [item],
                    seq_len=seq_len,
                    pack_results=False,
                    min_shared_segment_length=min_shared_segment_length,
                ),
            )
            for item in items
        ]

    segments = _prefix_tree_pack_segments(
        (item.token_ids for item in items),
        max_depth=max(len(item.token_ids) for item in items),
        shareable_lengths=(item.shareable_length for item in items),
        min_shared_segment_length=min_shared_segment_length,
    )
    paths: list[list[tuple[int, int]]] = [[] for _ in items]
    for segment_id, segment in enumerate(segments):
        path_segment = (segment_id, segment.length)
        for item_index in segment.sequence_indices:
            paths[item_index].append(path_segment)
    leaves = [
        _PrefixTreeLeaf(
            item_index=index,
            packing_group_id=item.prompt_id,
            segment_path=tuple(paths[index]),
            empty_bin_cost=sum(length for _, length in paths[index]),
        )
        for index, item in enumerate(items)
    ]
    for leaf in leaves:
        if leaf.empty_bin_cost > seq_len:
            raise RuntimeError(
                "Prefix-tree pack item exceeds sequence length: "
                f"cost={leaf.empty_bin_cost}, seq_len={seq_len}"
            )
    grouped: dict[int, list[_PrefixTreeLeaf]] = {}
    for leaf in leaves:
        grouped.setdefault(leaf.packing_group_id, []).append(leaf)
    bins = _place_prefix_tree_leaves(grouped.values(), seq_len=seq_len)

    planned_rows = []
    for packed_bin in bins:
        row = [items[leaf.item_index] for leaf in packed_bin.leaves]
        occupancy_plan = _filtered_prefix_tree_plan(
            segments,
            item_indices=tuple(leaf.item_index for leaf in packed_bin.leaves),
        )
        if occupancy_plan.length != packed_bin.token_count:
            raise RuntimeError(
                "Global prefix-tree occupancy disagrees with final bin plan: "
                f"occupancy={packed_bin.token_count}, plan={occupancy_plan.length}"
            )
        # Rebuild only after placement so bin-local paths compress without putting
        # repeated tree construction in the best-fit search.
        plan = _prefix_tree_row_plan(
            row,
            seq_len=seq_len,
            pack_results=True,
            min_shared_segment_length=min_shared_segment_length,
        )
        if plan.length > occupancy_plan.length:
            raise RuntimeError(
                "Final prefix-tree rebuild increased bin occupancy: "
                f"global={occupancy_plan.length}, rebuilt={plan.length}"
            )
        planned_rows.append((row, plan))
    return planned_rows


def _place_prefix_tree_leaves(
    groups: Iterable[Sequence[_PrefixTreeLeaf]],
    *,
    seq_len: int,
    groups_are_ordered: bool = False,
) -> list[_PrefixTreeBin]:
    ordered_groups = (
        groups
        if groups_are_ordered
        else sorted(
            groups,
            key=lambda group: max(leaf.empty_bin_cost for leaf in group),
            reverse=True,
        )
    )
    bins: list[_PrefixTreeBin] = []
    for leaf in (leaf for group in ordered_groups for leaf in group):
        if leaf.empty_bin_cost > seq_len:
            raise RuntimeError(
                "Prefix-tree pack item exceeds sequence length: "
                f"cost={leaf.empty_bin_cost}, seq_len={seq_len}"
            )
        best_bin = None
        best_remaining = seq_len + 1
        for candidate in bins:
            count = candidate.token_count + candidate.insertion_delta(leaf)
            if count <= seq_len and seq_len - count < best_remaining:
                best_bin = candidate
                best_remaining = seq_len - count
        if best_bin is None:
            best_bin = _PrefixTreeBin()
            bins.append(best_bin)
        best_bin.add(leaf)
    return bins


def _filtered_prefix_tree_plan(
    segments: tuple[PrefixTreePackSegment, ...],
    *,
    item_indices: tuple[int, ...],
) -> _PrefixTreeRowPlan:
    """Restrict the global plan to one bin without rerunning sharing decisions."""
    local_index = {item_index: index for index, item_index in enumerate(item_indices)}
    aliases: dict[int, int] = {}
    group_positions: dict[int, int] = {}
    planned: list[PrefixTreePackSegment] = []
    cursor = 0

    def resolve(group_id: int) -> int:
        while group_id in aliases:
            group_id = aliases[group_id]
        return group_id

    for segment in segments:
        sequence_indices = tuple(
            local_index[index]
            for index in segment.sequence_indices
            if index in local_index
        )
        if not sequence_indices:
            continue
        parent_id = resolve(segment.parent_id)
        parent_position = group_positions.get(parent_id)
        if parent_position is not None:
            parent = planned[parent_position]
            if (
                parent_position == len(planned) - 1
                and parent.sequence_indices == sequence_indices
                and parent.end == segment.start
            ):
                planned[parent_position] = PrefixTreePackSegment(
                    sequence_indices=sequence_indices,
                    start=parent.start,
                    end=segment.end,
                    packed_start=parent.packed_start,
                    group_id=parent.group_id,
                    parent_id=parent.parent_id,
                )
                aliases[segment.group_id] = parent.group_id
                cursor += segment.length
                continue
        group_id = segment.group_id
        if segment.parent_id == segment.group_id:
            parent_id = group_id
        group_positions[group_id] = len(planned)
        planned.append(
            PrefixTreePackSegment(
                sequence_indices=sequence_indices,
                start=segment.start,
                end=segment.end,
                packed_start=cursor,
                group_id=group_id,
                parent_id=parent_id,
            )
        )
        cursor += segment.length
    return _PrefixTreeRowPlan(segments=tuple(planned), length=cursor)


def _prefix_tree_pack_item(
    result: TokenizedResult,
    *,
    seq_len: int,
) -> _PrefixTreePackItem:
    assistant_mask = np.asarray(result.assistant_mask, dtype=np.bool_)
    logprobs = np.asarray(result.logprobs, dtype=np.float32)
    shareable_length = prefix_tree_shareable_length(
        result,
        assistant_mask=assistant_mask,
        logprobs=logprobs,
    )
    item = _PrefixTreePackItem(
        token_ids=tuple(result.token_ids),
        input_pos=np.asarray(result.input_pos, dtype=np.int64),
        assistant_mask=assistant_mask,
        logprobs=logprobs,
        advantage=float(result.advantage),
        weight=float(result.weight),
        prompt_id=int(result.prompt_id),
        shareable_length=shareable_length,
        pixel_values=result.pixel_values,
        image_grid_thw=result.image_grid_thw,
        moe_routes=result.moe_routed_experts,
    )
    return _truncate_prefix_tree_pack_item(item, seq_len)


def prefix_tree_shareable_length(
    result: TokenizedResult,
    *,
    assistant_mask: np.ndarray | None = None,
    logprobs: np.ndarray | None = None,
) -> int:
    assistant_mask = (
        np.asarray(result.assistant_mask, dtype=np.bool_)
        if assistant_mask is None
        else assistant_mask
    )
    logprobs = (
        np.asarray(result.logprobs, dtype=np.float32) if logprobs is None else logprobs
    )
    return min(
        int(result.prompt_length),
        max(
            _first_trainable_token_index(
                assistant_mask=assistant_mask,
                logprobs=logprobs,
            )
            - 1,
            0,
        ),
    )


def _truncate_prefix_tree_pack_item(
    item: _PrefixTreePackItem,
    seq_len: int,
) -> _PrefixTreePackItem:
    if len(item.token_ids) <= seq_len:
        return item
    return _PrefixTreePackItem(
        token_ids=item.token_ids[:seq_len],
        input_pos=item.input_pos[:seq_len],
        assistant_mask=item.assistant_mask[:seq_len],
        logprobs=item.logprobs[:seq_len],
        advantage=item.advantage,
        weight=item.weight,
        prompt_id=item.prompt_id,
        shareable_length=min(item.shareable_length, seq_len),
        pixel_values=item.pixel_values,
        image_grid_thw=item.image_grid_thw,
        moe_routes=item.moe_routes,
    )


def _first_trainable_token_index(
    *,
    assistant_mask: np.ndarray,
    logprobs: np.ndarray,
) -> int:
    trainable = assistant_mask | ~np.isnan(logprobs)
    indices = np.flatnonzero(trainable)
    return int(indices[0]) if int(indices.size) > 0 else int(assistant_mask.shape[0])


def _prefix_tree_row_plan(
    row: list[_PrefixTreePackItem],
    *,
    seq_len: int,
    pack_results: bool,
    min_shared_segment_length: int,
) -> _PrefixTreeRowPlan:
    segments = _prefix_tree_pack_segments(
        (item.token_ids for item in row),
        max_depth=seq_len if pack_results else 0,
        shareable_lengths=(
            item.shareable_length if pack_results else 0 for item in row
        ),
        min_shared_segment_length=min_shared_segment_length,
    )
    return _PrefixTreeRowPlan(
        segments=segments,
        length=min(sum(segment.length for segment in segments), seq_len),
    )


def _materialize_prefix_tree_row(
    row: list[_PrefixTreePackItem],
    *,
    plan: _PrefixTreeRowPlan,
    token_ids: np.ndarray,
    group_ids: np.ndarray,
    parent_ids: np.ndarray,
    input_pos: np.ndarray,
    assistant_mask: np.ndarray,
    logprobs: np.ndarray,
    advantages: np.ndarray,
    weights: np.ndarray,
    route_tensor: np.ndarray | None,
    route_shape: tuple[int, int] | None,
    include_moe_routing: bool,
) -> None:
    for segment in plan.segments:
        dst_start = int(segment.packed_start)
        if dst_start >= plan.length:
            continue
        segment_length = min(int(segment.length), plan.length - dst_start)
        dst_end = dst_start + segment_length
        src_start = int(segment.start)
        src_end = src_start + segment_length
        item = row[segment.sequence_indices[0]]
        token_ids[dst_start:dst_end] = item.token_ids[src_start:src_end]
        group_ids[dst_start:dst_end] = int(segment.group_id)
        parent_ids[dst_start:dst_end] = int(segment.parent_id)
        input_pos[dst_start:dst_end] = item.input_pos[src_start:src_end]
        assistant_mask[dst_start:dst_end] = item.assistant_mask[src_start:src_end]
        logprobs[dst_start:dst_end] = item.logprobs[src_start:src_end]
        advantages[dst_start:dst_end] = item.advantage
        weights[dst_start:dst_end] = item.weight
        if len(segment.sequence_indices) > 1:
            _validate_shared_prefix_tree_segment(
                row,
                sequence_indices=segment.sequence_indices,
                src_start=src_start,
                src_end=src_end,
            )
        if include_moe_routing:
            assert route_tensor is not None
            assert route_shape is not None
            assert item.moe_routes is not None
            _copy_moe_route_slice(
                route_tensor=route_tensor,
                dst_start=dst_start,
                src_start=src_start,
                src_end=src_end,
                raw_routes=item.moe_routes,
                route_shape=route_shape,
            )


def _validate_shared_prefix_tree_segment(
    row: list[_PrefixTreePackItem],
    *,
    sequence_indices: tuple[int, ...],
    src_start: int,
    src_end: int,
) -> None:
    reference = row[sequence_indices[0]]
    reference_input_pos = reference.input_pos[src_start:src_end]
    for sequence_index in sequence_indices:
        item = row[sequence_index]
        if src_end > item.shareable_length:
            raise RuntimeError("Prefix-tree pack attempted to share a trainable token")
        if not np.array_equal(item.input_pos[src_start:src_end], reference_input_pos):
            raise RuntimeError(
                "Prefix-tree pack cannot share mismatched input positions"
            )
        if (item.moe_routes is None) != (reference.moe_routes is None):
            raise RuntimeError("Prefix-tree shared routes are incomplete")


def _packed_row_tensor_list(
    row: list[_PrefixTreePackItem],
    attr: Literal["pixel_values", "image_grid_thw"],
) -> torch.Tensor | None:
    tensors: list[torch.Tensor] = []
    seen_shared_prompts: set[int] = set()
    for item in row:
        tensor = getattr(item, attr)
        if tensor is None:
            continue
        if item.shareable_length > 0:
            if item.prompt_id in seen_shared_prompts:
                continue
            seen_shared_prompts.add(item.prompt_id)
        tensors.append(tensor)
    return torch.concat(tensors) if tensors else None


def _moe_route_contract(
    rows: list[list[_PrefixTreePackItem]],
) -> tuple[int, int, int] | None:
    contracts = {
        (
            routes.num_experts,
            int(routes.shape[1]),
            int(routes.shape[2]),
        )
        for row in rows
        for item in row
        if (routes := item.moe_routes) is not None and routes.shape[0] > 0
    }
    if len(contracts) > 1:
        raise RuntimeError("Packed MoE routes must share one exact contract")
    return next(iter(contracts), None)


def _coerce_moe_routes(raw: MoeRouteArray | MoeRouteSegments) -> MoeRouteArray:
    if not isinstance(raw, MoeRouteArray):
        raise RuntimeError(f"Expected MoE routes array, got {type(raw)}")
    if raw.dtype != moe_route_dtype(raw.num_experts):
        raise RuntimeError("Packed MoE routes use the wrong exact ID dtype")
    return raw


def _copy_moe_route_slice(
    *,
    route_tensor: np.ndarray,
    dst_start: int,
    src_start: int,
    src_end: int,
    raw_routes: MoeRouteArray | MoeRouteSegments,
    route_shape: tuple[int, int],
) -> None:
    if src_end <= src_start:
        return
    if isinstance(raw_routes, MoeRouteSegments):
        covered_until = src_start
        for segment_start, segment in raw_routes.iter_slices(src_start, src_end):
            if segment_start != covered_until:
                raise RuntimeError(
                    "Segmented MoE routes did not cover packed source slice"
                )
            if tuple(segment.shape[1:]) != route_shape:
                raise RuntimeError("Packed MoE routes must have one rectangular shape")
            segment_dst_start = dst_start + segment_start - src_start
            segment_dst_end = segment_dst_start + int(segment.shape[0])
            route_tensor[:, segment_dst_start:segment_dst_end] = np.moveaxis(
                segment, 1, 0
            )
            covered_until = segment_start + int(segment.shape[0])
        if covered_until != src_end:
            raise RuntimeError("Segmented MoE routes did not cover packed source slice")
        return

    routes = _coerce_moe_routes(raw_routes)
    route_slice = routes[src_start:src_end]
    if tuple(route_slice.shape[1:]) != route_shape:
        raise RuntimeError("Packed MoE routes must have one rectangular shape")
    dst_end = dst_start + int(route_slice.shape[0])
    route_tensor[:, dst_start:dst_end] = np.moveaxis(
        route_slice,
        1,
        0,
    )


def packed_tensors_from_dir(**kwargs: Unpack[DiskPackedTensors]) -> PackedTensors:
    os.makedirs(kwargs["dir"], exist_ok=True)
    packed_tensors = {
        key: torch.from_file(
            f"{kwargs['dir']}/{key}.pt",
            shared=True,
            size=kwargs["num_sequences"] * kwargs["sequence_length"],
            dtype=dtype,
        ).view(kwargs["num_sequences"], kwargs["sequence_length"])
        for key, dtype in {
            "tokens": torch.long,
            "group_ids": torch.long,
            "parent_ids": torch.long,
            "input_pos": torch.long,
            "assistant_mask": torch.bool,
            "logprobs": torch.float32,
            "advantages": torch.float32,
            "weights": torch.float32,
        }.items()
    }
    _add_tensor_list(packed_tensors, kwargs, "pixel_values", torch.float32)  # ty:ignore[invalid-argument-type]
    _add_tensor_list(packed_tensors, kwargs, "image_grid_thw", torch.long)  # ty:ignore[invalid-argument-type]
    return cast(PackedTensors, packed_tensors)


def _add_tensor_list(
    packed_tensors: dict[str, Any],
    disk_packed_tensors: DiskPackedTensors,
    key: str,
    dtype: torch.dtype,
) -> None:
    if info := disk_packed_tensors.get(key):
        packed_tensors[key] = []
        inner_dim, offsets = cast(tuple[int, list[int]], info)
        packed_pixel_values = torch.from_file(
            f"{disk_packed_tensors['dir']}/{key}.pt",
            shared=True,
            size=offsets[-1] * inner_dim,
            dtype=dtype,
        ).view(-1, inner_dim)
        for start, end in zip(offsets[:-1], offsets[1:]):
            packed_tensors[key].append(
                packed_pixel_values[start:end] if start < end else None
            )
    else:
        packed_tensors[key] = [None] * disk_packed_tensors["num_sequences"]


def packed_tensors_to_dir(tensors: PackedTensors, dir: str) -> DiskPackedTensors:
    os.makedirs(dir, exist_ok=True)
    disk_packed_tensors: DiskPackedTensors = {
        "dir": dir,
        "num_sequences": tensors["tokens"].shape[0],
        "sequence_length": tensors["tokens"].shape[1],
    }
    if info := _get_tensor_list_info(tensors["pixel_values"]):
        disk_packed_tensors["pixel_values"] = info
    if info := _get_tensor_list_info(tensors["image_grid_thw"]):
        disk_packed_tensors["image_grid_thw"] = info
    for key, tensor in packed_tensors_from_dir(**disk_packed_tensors).items():
        if isinstance(tensor, list):
            for i, t in enumerate(tensor):
                if t is not None:
                    t.copy_(tensors[key][i])  # ty:ignore[invalid-key, unresolved-attribute]
        else:
            tensor.copy_(tensors[key])  # type: ignore
    return disk_packed_tensors


def _get_tensor_list_info(
    tensors: list[torch.Tensor | None],
) -> tuple[int, list[int]] | None:
    inner_dims = {tensor.shape[1] for tensor in tensors if tensor is not None}
    if len(inner_dims) == 0:
        return None
    assert len(inner_dims) == 1, f"Inner dimensions of {tensors} are not the same"
    offsets = [0]
    for tensor in tensors:
        if tensor is not None:
            offsets.append(offsets[-1] + tensor.shape[0])
        else:
            offsets.append(offsets[-1])
    return inner_dims.pop(), offsets


def plot_packed_tensors(
    packed_tensors: PackedTensors, output_dir: str | None = None
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Plotting dependencies are not installed. Please install them with: "
            "pip install openpipe-art[plotting]"
        )

    plt.figure(figsize=(15, 24))

    for tensor, label, title, subplot_idx in (
        (packed_tensors["tokens"], "Token IDs", "Token IDs", 1),
        (packed_tensors["logprobs"], "Log Probabilities", "Token Log Probs", 2),
        (packed_tensors["group_ids"], "Group IDs", "Token Groups", 3),
        (packed_tensors["parent_ids"], "Parent IDs", "Parent IDs", 4),
        (packed_tensors["input_pos"], "Position", "Input Position", 5),
        (packed_tensors["assistant_mask"], "Assistant Mask", "Assistant Mask", 6),
        (packed_tensors["advantages"], "Advantages", "Token Advantages", 7),
        (packed_tensors["weights"], "Weights", "Token Weights", 8),
    ):
        plt.subplot(4, 2, subplot_idx)
        sns.heatmap(
            tensor.numpy(),
            cmap="viridis",
            cbar_kws={"label": label},
            xticklabels=False,
        )
        plt.title(title)
        plt.xlabel("Sequence Position")
        plt.ylabel("Batch")

    plt.tight_layout()
    plt.show()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = f"{output_dir}/packed_tensors_plot_{int(time.time())}.png"
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
    else:
        print("No output directory specified, plot not saved")
