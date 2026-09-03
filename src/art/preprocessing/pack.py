from array import array
from collections.abc import Iterable, Sequence
import os
import random
import time
from typing import Any, Literal, NamedTuple, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch
from typing_extensions import NotRequired, TypedDict, Unpack

from ..megatron.prefix_tree_packing import (
    PrefixTreePackSegment,
)
from ..megatron.prefix_tree_packing import (
    prefix_tree_pack_segments as _prefix_tree_pack_segments,
)
from ..pipeline_tuner.config import PackedGroupShape, PackingLeafShape
from ..training.contracts import PolicyTokenCount, TrainingOutcome
from ..training.token_matrix import (
    MAX_MATRIX_VALUES,
    MAX_TARGET_CANDIDATES,
    NamedLossRequest,
    TokenMatrix,
    TokenMatrixBatch,
    active_loss_positions,
    validate_token_matrix_batch,
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
    target_tokens: NotRequired[torch.Tensor]
    projection_ids: NotRequired[torch.Tensor]
    logical_projection_ids: NotRequired[torch.Tensor]
    logical_matrix_indices: NotRequired[torch.Tensor]
    logical_target_indices: NotRequired[torch.Tensor]
    logical_value_mask: NotRequired[torch.Tensor]
    logical_loss_weights: NotRequired[torch.Tensor]
    logical_behavior_logprobs: NotRequired[torch.Tensor]
    logical_advantages: NotRequired[torch.Tensor]
    token_matrix_output_map: NotRequired["TokenMatrixOutputMap"]
    token_matrix_training_outcome: NotRequired[TrainingOutcome]


class TokenMatrixOutputMap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matrix_ids: tuple[str, ...] = Field(min_length=1)
    target_shapes: tuple[tuple[int, int], ...] = Field(min_length=1)


# Exact tokenized inputs have no model-specific pad token. Padding positions are
# attention-masked, but their input IDs are still embedded by the model.
TOKENIZED_INPUT_PADDING_ID = 0
TOKENIZED_TARGET_PADDING_ID = -100


class DiskPackedTensors(TypedDict):
    dir: str
    num_sequences: int
    sequence_length: int
    pixel_values: NotRequired[tuple[int, list[int]]]
    image_grid_thw: NotRequired[tuple[int, list[int]]]


class _PrefixTreePackItem(NamedTuple):
    token_ids: tuple[int, ...]
    sharing_ids: tuple[int, ...]
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
    matrix_index: int | None = None
    target_tokens: np.ndarray | None = None
    loss_weights: np.ndarray | None = None
    behavior_logprobs: np.ndarray | None = None
    token_advantages: np.ndarray | None = None
    placement_group_id: int | None = None


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

    def insertion_delta_group(self, leaves: Sequence[_PrefixTreeLeaf]) -> int:
        segments = {
            segment_id: length
            for leaf in leaves
            for segment_id, length in leaf.segment_path
            if segment_id not in self.occupied_segments
        }
        return sum(segments.values())

    def add_group(self, leaves: Sequence[_PrefixTreeLeaf]) -> None:
        self.token_count += self.insertion_delta_group(leaves)
        self.occupied_segments.update(
            segment_id for leaf in leaves for segment_id, _ in leaf.segment_path
        )
        self.leaves.extend(leaves)


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


def packed_tensors_from_token_matrices(
    batch: TokenMatrixBatch,
    *,
    loss: NamedLossRequest | None,
    seq_len: int,
    pad_token_id: int = TOKENIZED_INPUT_PADDING_ID,
    pack_results: bool = True,
    return_token_logprobs: bool = True,
    resolved_routes: dict[str, MoeRouteArray | MoeRouteSegments] | None = None,
    min_prefix_tree_shared_segment_length: int = (
        DEFAULT_MIN_PREFIX_TREE_SHARED_SEGMENT_LENGTH
    ),
) -> PackedTensors:
    """Build one physical prefix tree while retaining every logical occurrence."""

    validate_token_matrix_batch(
        batch,
        loss,
        output_rows=("learner_logprobs",) if return_token_logprobs else (),
    )
    component_ids = _matrix_component_ids(batch, loss)
    routes = _resolved_matrix_routes(batch, resolved_routes)
    items = [
        _token_matrix_pack_item(
            matrix,
            matrix_index=index,
            loss=loss,
            seq_len=seq_len,
            route=routes.get(matrix.matrix_id),
            placement_group_id=component_ids.get(matrix.matrix_id),
        )
        for index, matrix in enumerate(batch.matrices)
    ]
    return prefix_tree_pack(
        tokenized_results=[],
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        truncate_long_results=False,
        advantage_balance=0.0,
        verbosity=0,
        pack_results=pack_results,
        include_moe_routing=bool(routes),
        min_prefix_tree_shared_segment_length=(min_prefix_tree_shared_segment_length),
        _items=items,
        _token_matrix_batch=batch,
        _return_token_logprobs=return_token_logprobs,
    )


def token_matrix_packing_shapes(
    batch: TokenMatrixBatch,
    loss: NamedLossRequest,
) -> tuple[PackedGroupShape, ...]:
    """Expose bounded logical placement units with stable matrix identities."""

    by_id = {matrix.matrix_id: matrix for matrix in batch.matrices}
    assigned = {
        matrix_id
        for component in loss.placement_components()
        for matrix_id in component
    }
    units = [*loss.placement_components()]
    units.extend(
        (matrix.matrix_id,)
        for matrix in batch.matrices
        if matrix.matrix_id not in assigned
    )
    return tuple(
        PackedGroupShape(
            leaves=tuple(
                PackingLeafShape(
                    matrix_id=matrix_id,
                    token_ids=array(
                        "I", by_id[matrix_id].row("token_ids").dense_values()
                    ),
                    shareable_length=by_id[matrix_id].token_count,
                )
                for matrix_id in unit
            )
        )
        for unit in units
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
    _items: list[_PrefixTreePackItem] | None = None,
    _token_matrix_batch: TokenMatrixBatch | None = None,
    _return_token_logprobs: bool = True,
) -> PackedTensors:
    if min_prefix_tree_shared_segment_length < 0:
        raise ValueError("min_prefix_tree_shared_segment_length must be >= 0")
    if _items is not None and tokenized_results:
        raise ValueError("prefix_tree_pack accepts results or prebuilt items")
    if _token_matrix_batch is not None and _items is None:
        raise ValueError("TokenMatrix packing requires prebuilt items")
    items: list[_PrefixTreePackItem] = [] if _items is None else _items
    moe_routing_pack_stats = MoeRoutingPackStats()

    if _items is None:
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
    random.Random(len(planned_rows)).shuffle(planned_rows)
    rows = [row for row, _ in planned_rows]
    row_plans = [plan for _, plan in planned_rows]

    num_sequences = len(rows)
    if _token_matrix_batch is not None and num_sequences * seq_len > MAX_MATRIX_VALUES:
        raise ValueError("TokenMatrix physical plan exceeds the configured value limit")
    tokens_np = np.full((num_sequences, seq_len), pad_token_id, dtype=np.int64)
    group_ids_np = np.full((num_sequences, seq_len), -1, dtype=np.int64)
    parent_ids_np = np.full((num_sequences, seq_len), -1, dtype=np.int64)
    input_pos_np = np.zeros((num_sequences, seq_len), dtype=np.int64)
    assistant_mask_np = np.zeros((num_sequences, seq_len), dtype=np.bool_)
    logprobs_np = np.full((num_sequences, seq_len), np.nan, dtype=np.float32)
    advantages_np = np.zeros((num_sequences, seq_len), dtype=np.float32)
    weights_np = np.zeros((num_sequences, seq_len), dtype=np.float32)
    packed_positions = (
        [np.full(len(item.token_ids), -1, dtype=np.int64) for item in items]
        if _token_matrix_batch is not None
        else None
    )
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
            packed_positions=packed_positions,
            packed_row=index,
            packed_sequence_length=seq_len,
        )
        pixel_values.append(_packed_row_tensor_list(row, "pixel_values"))
        image_grid_thw.append(_packed_row_tensor_list(row, "image_grid_thw"))
    token_matrix_tensors = None
    if _token_matrix_batch is not None:
        if packed_positions is None:
            raise RuntimeError("TokenMatrix packing did not retain logical positions")
        token_matrix_tensors = _materialize_token_matrix_logical_tensors(
            batch=_token_matrix_batch,
            items=items,
            packed_positions=packed_positions,
            num_sequences=num_sequences,
            sequence_length=seq_len,
            assistant_mask=assistant_mask_np,
            return_token_logprobs=_return_token_logprobs,
        )
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
    if token_matrix_tensors is not None:
        packed_tensors.update(token_matrix_tensors)
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
        grouped: dict[tuple[str, int], list[_PrefixTreePackItem]] = {}
        for index, item in enumerate(items):
            key = (
                ("item", index)
                if item.placement_group_id is None
                else ("component", item.placement_group_id)
            )
            grouped.setdefault(key, []).append(item)
        planned_rows = []
        for row in grouped.values():
            required_tokens = sum(len(item.token_ids) for item in row)
            if required_tokens > seq_len:
                raise RuntimeError(
                    "TokenMatrix placement component exceeds sequence length "
                    "without prefix sharing: "
                    f"cost={required_tokens}, seq_len={seq_len}"
                )
            plan = _prefix_tree_row_plan(
                row,
                seq_len=seq_len,
                pack_results=False,
                min_shared_segment_length=min_shared_segment_length,
            )
            if plan.length != required_tokens:
                raise RuntimeError(
                    "unpacked TokenMatrix component did not retain every token"
                )
            planned_rows.append((row, plan))
        return planned_rows

    segments = _prefix_tree_pack_segments(
        (item.sharing_ids for item in items),
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
            packing_group_id=(
                item.placement_group_id
                if item.placement_group_id is not None
                else len(items) + index
            ),
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
    bins = _place_prefix_tree_groups(grouped.values(), seq_len=seq_len)

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


def _place_prefix_tree_groups(
    groups: Iterable[Sequence[_PrefixTreeLeaf]],
    *,
    seq_len: int,
) -> list[_PrefixTreeBin]:
    """Best-fit placement where every loss-connected group is one hard unit."""

    units = []
    for group in groups:
        leaves = tuple(group)
        segment_lengths = {
            segment_id: length
            for leaf in leaves
            for segment_id, length in leaf.segment_path
        }
        empty_cost = sum(segment_lengths.values())
        if empty_cost > seq_len:
            raise RuntimeError(
                "TokenMatrix placement component exceeds sequence length: "
                f"cost={empty_cost}, seq_len={seq_len}"
            )
        units.append((empty_cost, leaves))
    units.sort(key=lambda item: item[0], reverse=True)

    bins: list[_PrefixTreeBin] = []
    for _empty_cost, leaves in units:
        best_bin = None
        best_remaining = seq_len + 1
        for candidate in bins:
            count = candidate.token_count + candidate.insertion_delta_group(leaves)
            if count <= seq_len and seq_len - count < best_remaining:
                best_bin = candidate
                best_remaining = seq_len - count
        if best_bin is None:
            best_bin = _PrefixTreeBin()
            bins.append(best_bin)
        best_bin.add_group(leaves)
    return bins


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
        sharing_ids=tuple(result.token_ids),
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
    _validate_prefix_tree_pack_item(item)
    return _truncate_prefix_tree_pack_item(item, seq_len)


def _token_matrix_pack_item(
    matrix: TokenMatrix,
    *,
    matrix_index: int,
    loss: NamedLossRequest | None,
    seq_len: int,
    route: MoeRouteArray | MoeRouteSegments | None,
    placement_group_id: int | None,
) -> _PrefixTreePackItem:
    if matrix.token_count > seq_len:
        raise ValueError(
            "TokenMatrix exceeds packed sequence length: "
            f"{matrix.token_count} > {seq_len}"
        )
    tokens = tuple(int(value) for value in matrix.row("token_ids").dense_values())
    target_row = matrix.row("target_token_ids")
    targets = np.asarray(target_row.dense_values(), dtype=np.int64).reshape(
        target_row.shape
    )
    behavior = _matrix_float_row(matrix, "behavior_logprobs", target_row.shape)
    advantages = _matrix_float_row(matrix, "advantages", target_row.shape)
    active = np.asarray(active_loss_positions(matrix, loss), dtype=np.bool_)
    default_weight = (
        1.0
        if loss is not None
        and loss.name in {"importance_sampling", "cispo"}
        and matrix.optional_row("loss_weights") is None
        else 0.0
    )
    weights = _matrix_float_row(
        matrix,
        "loss_weights",
        target_row.shape,
        default=default_weight,
    )
    weights = np.where(active[:, None], weights, 0.0).astype(np.float32, copy=False)
    if loss is None or loss.name not in {"importance_sampling", "cispo"}:
        # Losses without an advantage row use this otherwise-unused tensor as a
        # one-per-position activity marker for topology-invariant normalization.
        advantages = np.zeros_like(weights)
        for position in np.flatnonzero(active):
            candidates = np.flatnonzero(weights[position] != 0.0)
            if candidates.size:
                advantages[position, int(candidates[0])] = 1.0
    item = _PrefixTreePackItem(
        token_ids=tokens,
        sharing_ids=tokens,
        input_pos=np.arange(matrix.token_count, dtype=np.int64),
        assistant_mask=active,
        logprobs=np.full(matrix.token_count, np.nan, dtype=np.float32),
        advantage=1.0,
        weight=1.0,
        prompt_id=matrix_index,
        shareable_length=matrix.token_count,
        pixel_values=None,
        image_grid_thw=None,
        moe_routes=route,
        matrix_index=matrix_index,
        target_tokens=targets,
        loss_weights=weights,
        behavior_logprobs=behavior,
        token_advantages=advantages,
        placement_group_id=placement_group_id,
    )
    _validate_prefix_tree_pack_item(item)
    return item


def _matrix_float_row(
    matrix: TokenMatrix,
    name: str,
    shape: tuple[int, ...],
    *,
    default: float = 0.0,
) -> np.ndarray:
    row = matrix.optional_row(name)
    if row is None:
        return np.full(shape, default, dtype=np.float32)
    return np.asarray(row.dense_values(), dtype=np.float32).reshape(shape)


def _matrix_component_ids(
    batch: TokenMatrixBatch,
    loss: NamedLossRequest | None,
) -> dict[str, int]:
    if loss is None:
        return {}
    return {
        matrix_id: component_id
        for component_id, component in enumerate(loss.placement_components())
        for matrix_id in component
    }


def _resolved_matrix_routes(
    batch: TokenMatrixBatch,
    resolved: dict[str, MoeRouteArray | MoeRouteSegments] | None,
) -> dict[str, MoeRouteArray | MoeRouteSegments]:
    from .token_matrix import inline_routes_array

    routes = dict(resolved or {})
    for route in batch.routes:
        if route.kind == "captured":
            raise ValueError(
                "captured TokenMatrix routes must be resolved before packing"
            )
        if route.kind == "retained":
            if route.matrix_id not in routes:
                raise ValueError(
                    "retained TokenMatrix routes must be resolved before packing"
                )
            continue
        inline = inline_routes_array(route)
        previous = routes.setdefault(route.matrix_id, inline)
        if previous is not inline:
            raise ValueError("TokenMatrix routes were supplied twice")
    if routes and set(routes) != {matrix.matrix_id for matrix in batch.matrices}:
        raise ValueError("routing replay requires routes for every TokenMatrix")
    return routes


def _validate_prefix_tree_pack_item(item: _PrefixTreePackItem) -> None:
    token_count = len(item.token_ids)
    if len(item.sharing_ids) != token_count:
        raise RuntimeError("Prefix-tree sharing IDs must match token IDs")
    for name in ("input_pos", "assistant_mask", "logprobs"):
        value = getattr(item, name)
        if value.ndim != 1 or len(value) != token_count:
            raise RuntimeError(
                f"Prefix-tree packing {name} must have shape ({token_count},), got "
                f"{value.shape}"
            )
    if item.shareable_length > token_count:
        raise RuntimeError("Prefix-tree shareable length exceeds token count")
    if item.moe_routes is not None and item.moe_routes.shape[0] != token_count:
        raise RuntimeError(
            "Prefix-tree MoE route token count does not match token IDs: "
            f"{item.moe_routes.shape[0]} != {token_count}"
        )
    logical = (
        item.target_tokens,
        item.loss_weights,
        item.behavior_logprobs,
        item.token_advantages,
    )
    if item.matrix_index is None:
        if any(value is not None for value in logical):
            raise RuntimeError("ART pack items cannot carry TokenMatrix tensors")
        return
    if item.target_tokens is None or any(
        value is not None and value.shape != item.target_tokens.shape
        for value in logical[1:]
    ):
        raise RuntimeError("TokenMatrix logical tensors must share one exact shape")
    if item.target_tokens.shape[0] != token_count:
        raise RuntimeError("TokenMatrix logical tensors must align with input tokens")


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
        sharing_ids=item.sharing_ids[:seq_len],
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
        (item.sharing_ids for item in row),
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
    packed_positions: list[np.ndarray] | None = None,
    packed_row: int = 0,
    packed_sequence_length: int = 0,
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
        if packed_positions is not None:
            for sequence_index in segment.sequence_indices:
                source = row[sequence_index]
                if source.matrix_index is None:
                    raise RuntimeError("TokenMatrix occurrence mapping is incomplete")
                packed_positions[source.matrix_index][src_start:src_end] = (
                    packed_row * packed_sequence_length
                    + np.arange(dst_start, dst_end, dtype=np.int64)
                )
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


def _materialize_token_matrix_logical_tensors(
    *,
    batch: TokenMatrixBatch,
    items: list[_PrefixTreePackItem],
    packed_positions: list[np.ndarray],
    num_sequences: int,
    sequence_length: int,
    assistant_mask: np.ndarray,
    return_token_logprobs: bool,
) -> dict[str, Any]:
    if len(items) != len(batch.matrices) or len(packed_positions) != len(items):
        raise RuntimeError("TokenMatrix logical inputs are not aligned")

    packed_rows: list[int] = []
    logical_entry_counts = [0] * num_sequences
    for item, positions in zip(items, packed_positions, strict=True):
        if np.any(positions < 0):
            raise RuntimeError("TokenMatrix occurrence map is incomplete")
        rows = {int(position) // sequence_length for position in positions}
        if len(rows) != 1:
            raise RuntimeError("one TokenMatrix must remain within one packed row")
        packed_row = next(iter(rows))
        packed_rows.append(packed_row)
        targets = item.target_tokens
        weights = item.loss_weights
        advantages = item.token_advantages
        if targets is None or weights is None or advantages is None:
            raise RuntimeError("TokenMatrix item is missing logical loss rows")
        logical_entry_counts[packed_row] += (
            int(targets.size)
            if return_token_logprobs
            else int(np.count_nonzero((weights != 0.0) | (advantages != 0.0)))
        )
    logical_width = max(max(logical_entry_counts, default=0), 1)
    padded_logical_value_count = num_sequences * logical_width
    if padded_logical_value_count > MAX_MATRIX_VALUES:
        raise ValueError(
            "TokenMatrix padded physical or logical plan exceeds the configured "
            "value limit"
        )

    # The physical target union is deduplicated by (physical token, target
    # token), while the logical entries below retain every caller occurrence.
    # Projection IDs are microbatch-global because the gather flattens rows.
    position_targets: list[dict[int, tuple[int, int]]] = [
        {} for _ in range(num_sequences * sequence_length)
    ]
    next_projection_id = 0
    entries: list[list[tuple[int, int, int, float, float, float]]] = [
        [] for _ in range(num_sequences)
    ]

    accepted_positions = 0
    policy_counts: dict[int, int] = {}
    policy_applicable = any(
        matrix.optional_row("policy_version") is not None for matrix in batch.matrices
    )
    candidate_capacity = 1
    for matrix_index, (item, packed_row) in enumerate(
        zip(items, packed_rows, strict=True)
    ):
        targets = item.target_tokens
        weights = item.loss_weights
        behavior = item.behavior_logprobs
        advantages = item.token_advantages
        if any(value is None for value in (targets, weights, behavior, advantages)):
            raise RuntimeError("TokenMatrix item is missing logical loss rows")
        assert targets is not None
        assert weights is not None
        assert behavior is not None
        assert advantages is not None
        positions = packed_positions[matrix_index]
        candidate_count = int(targets.shape[1])
        policy_row = batch.matrices[matrix_index].optional_row("policy_version")
        policy_values = None if policy_row is None else policy_row.dense_values()
        for logical_position, physical_position in enumerate(positions):
            active_position = bool(item.assistant_mask[logical_position])
            if active_position:
                assistant_mask.reshape(-1)[int(physical_position)] = True
                accepted_positions += 1
                if policy_values is not None:
                    policy_version = int(
                        policy_values[logical_position * policy_row.trailing_width]
                    )
                    policy_counts[policy_version] = (
                        policy_counts.get(policy_version, 0) + 1
                    )
            for candidate in range(candidate_count):
                weight = float(weights[logical_position, candidate])
                advantage = float(advantages[logical_position, candidate])
                if not return_token_logprobs and weight == 0.0 and advantage == 0.0:
                    continue
                target = int(targets[logical_position, candidate])
                target_slots = position_targets[int(physical_position)]
                slot_projection = target_slots.get(target)
                if slot_projection is None:
                    slot = len(target_slots)
                    if slot >= MAX_TARGET_CANDIDATES:
                        raise ValueError(
                            "shared TokenMatrix target union exceeds the candidate limit"
                        )
                    candidate_capacity = max(candidate_capacity, slot + 1)
                    if (
                        num_sequences * sequence_length * candidate_capacity
                        > MAX_MATRIX_VALUES
                    ):
                        raise ValueError(
                            "TokenMatrix padded physical or logical plan exceeds the "
                            "configured value limit"
                        )
                    projection_id = next_projection_id
                    next_projection_id += 1
                    slot_projection = (slot, projection_id)
                    target_slots[target] = slot_projection
                _slot, projection_id = slot_projection
                entries[packed_row].append(
                    (
                        matrix_index,
                        logical_position * candidate_count + candidate,
                        projection_id,
                        weight,
                        float(behavior[logical_position, candidate]),
                        advantage,
                    )
                )

    physical_value_count = num_sequences * sequence_length * candidate_capacity
    if physical_value_count > MAX_MATRIX_VALUES:
        raise ValueError(
            "TokenMatrix padded physical or logical plan exceeds the configured "
            "value limit"
        )

    physical_shape = (num_sequences, sequence_length, candidate_capacity)
    target_tokens = np.full(physical_shape, TOKENIZED_TARGET_PADDING_ID, dtype=np.int64)
    projection_ids = np.full(physical_shape, -1, dtype=np.int64)
    for flat_position, targets in enumerate(position_targets):
        row, position = divmod(flat_position, sequence_length)
        for target, (slot, projection_id) in targets.items():
            target_tokens[row, position, slot] = target
            projection_ids[row, position, slot] = projection_id

    logical_shape = (num_sequences, logical_width)
    logical_projection_ids = np.full(logical_shape, -1, dtype=np.int64)
    logical_matrix_indices = np.full(logical_shape, -1, dtype=np.int32)
    logical_target_indices = np.full(logical_shape, -1, dtype=np.int64)
    logical_value_mask = np.zeros(logical_shape, dtype=np.bool_)
    logical_loss_weights = np.zeros(logical_shape, dtype=np.float32)
    logical_behavior_logprobs = np.zeros(logical_shape, dtype=np.float32)
    logical_advantages = np.zeros(logical_shape, dtype=np.float32)
    for row_index, row_entries in enumerate(entries):
        for index, (
            matrix_index,
            target_index,
            projection_id,
            weight,
            behavior,
            advantage,
        ) in enumerate(row_entries):
            logical_projection_ids[row_index, index] = projection_id
            logical_matrix_indices[row_index, index] = matrix_index
            logical_target_indices[row_index, index] = target_index
            logical_value_mask[row_index, index] = True
            logical_loss_weights[row_index, index] = weight
            logical_behavior_logprobs[row_index, index] = behavior
            logical_advantages[row_index, index] = advantage

    return {
        "target_tokens": torch.from_numpy(target_tokens),
        "projection_ids": torch.from_numpy(projection_ids),
        "logical_projection_ids": torch.from_numpy(logical_projection_ids),
        "logical_matrix_indices": torch.from_numpy(logical_matrix_indices),
        "logical_target_indices": torch.from_numpy(logical_target_indices),
        "logical_value_mask": torch.from_numpy(logical_value_mask),
        "logical_loss_weights": torch.from_numpy(logical_loss_weights),
        "logical_behavior_logprobs": torch.from_numpy(logical_behavior_logprobs),
        "logical_advantages": torch.from_numpy(logical_advantages),
        "token_matrix_output_map": TokenMatrixOutputMap(
            matrix_ids=tuple(matrix.matrix_id for matrix in batch.matrices),
            target_shapes=tuple(
                cast(tuple[int, int], matrix.row("target_token_ids").shape)
                for matrix in batch.matrices
            ),
        ),
        "token_matrix_training_outcome": TrainingOutcome(
            accepted_trainable_tokens=accepted_positions,
            policy_token_counts=(
                tuple(
                    PolicyTokenCount(
                        policy_version=version,
                        accepted_trainable_tokens=count,
                    )
                    for version, count in sorted(policy_counts.items())
                )
                if policy_applicable
                else None
            ),
        ),
    }


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
        if not np.array_equal(item.input_pos[src_start:src_end], reference_input_pos):
            raise RuntimeError(
                "Prefix-tree pack cannot share mismatched input positions"
            )


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
