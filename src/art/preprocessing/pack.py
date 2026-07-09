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
    MISSING_EXPERT_ID,
    MoeRouteArray,
    MoeRouteSegments,
    MoeRoutingPackStats,
    PackedMoeRoutingReplay,
)
from .tokenize import TokenizedResult


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


class DiskPackedTensors(TypedDict):
    dir: str
    num_sequences: int
    sequence_length: int
    pixel_values: NotRequired[tuple[int, list[int]]]
    image_grid_thw: NotRequired[tuple[int, list[int]]]


class _PackedPrefixTreeRow(NamedTuple):
    token_ids: np.ndarray
    group_ids: np.ndarray
    parent_ids: np.ndarray
    input_pos: np.ndarray
    assistant_mask: np.ndarray
    logprobs: np.ndarray
    advantages: np.ndarray
    weights: np.ndarray
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    route_tensor: np.ndarray | None = None
    route_mask: np.ndarray | None = None
    max_expert_id: int = 0


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
    moe_routes: Any | None


class _PrefixTreeRowPlan(NamedTuple):
    segments: tuple[PrefixTreePackSegment, ...]
    length: int


class _PrefixTrieNode:
    __slots__ = ("children",)

    def __init__(self) -> None:
        self.children: dict[int, _PrefixTrieEdge] = {}


class _PrefixTrieEdge:
    __slots__ = ("child", "edge_id", "tokens")

    def __init__(
        self,
        *,
        edge_id: int,
        tokens: tuple[int, ...],
        child: _PrefixTrieNode,
    ) -> None:
        self.edge_id = edge_id
        self.tokens = tokens
        self.child = child


class _PrefixTrie:
    __slots__ = ("_next_edge_id", "root")

    def __init__(self) -> None:
        self.root = _PrefixTrieNode()
        self._next_edge_id = 1

    def insert(self, tokens: tuple[int, ...]) -> None:
        node = self.root
        pos = 0
        while pos < len(tokens):
            token = tokens[pos]
            edge = node.children.get(token)
            if edge is None:
                node.children[token] = self._new_edge(tokens[pos:])
                return
            common = _prefix_trie_common_length(edge.tokens, tokens, pos)
            if common == len(edge.tokens):
                node = edge.child
                pos += common
                continue
            split = _PrefixTrieNode()
            old_suffix = edge.tokens[common:]
            split.children[old_suffix[0]] = self._new_edge(old_suffix, child=edge.child)
            node.children[token] = self._new_edge(edge.tokens[:common], child=split)
            pos += common
            if pos < len(tokens):
                split.children[tokens[pos]] = self._new_edge(tokens[pos:])
            return

    def edge_path(self, tokens: tuple[int, ...]) -> tuple[_PrefixTrieEdge, ...]:
        node = self.root
        pos = 0
        path: list[_PrefixTrieEdge] = []
        while pos < len(tokens):
            edge = node.children[tokens[pos]]
            if not _prefix_trie_edge_matches(edge.tokens, tokens, pos):
                raise RuntimeError("Prefix trie path mismatch")
            path.append(edge)
            pos += len(edge.tokens)
            node = edge.child
        return tuple(path)

    def _new_edge(
        self,
        tokens: tuple[int, ...],
        *,
        child: _PrefixTrieNode | None = None,
    ) -> _PrefixTrieEdge:
        edge = _PrefixTrieEdge(
            edge_id=self._next_edge_id,
            tokens=tokens,
            child=child or _PrefixTrieNode(),
        )
        self._next_edge_id += 1
        return edge


class _PrefixTrieLeaf(NamedTuple):
    item: _PrefixTreePackItem
    edge_path: tuple[_PrefixTrieEdge, ...]
    suffix_len: int

    @property
    def empty_bin_cost(self) -> int:
        return self.suffix_len + sum(len(edge.tokens) for edge in self.edge_path)


class _PrefixTrieBin:
    __slots__ = ("edge_ids", "items", "token_count")

    def __init__(self) -> None:
        self.edge_ids: set[int] = set()
        self.items: list[_PrefixTreePackItem] = []
        self.token_count = 0

    def insertion_delta(self, leaf: _PrefixTrieLeaf) -> int:
        return leaf.suffix_len + sum(
            len(edge.tokens)
            for edge in leaf.edge_path
            if edge.edge_id not in self.edge_ids
        )

    def add(self, leaf: _PrefixTrieLeaf) -> None:
        self.token_count += self.insertion_delta(leaf)
        self.edge_ids.update(edge.edge_id for edge in leaf.edge_path)
        self.items.append(leaf.item)


def packed_tensors_from_tokenized_results(
    tokenized_results: list[TokenizedResult],
    seq_len: int,
    pad_token_id: int = -100,
    truncate_long_results: bool = True,
    advantage_balance: float = 0.0,
    verbosity: Verbosity = 1,
    pack_results: bool = True,
    include_moe_routing: bool = False,
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
) -> PackedTensors:
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

    rows = _prefix_tree_pack_rows(
        items,
        seq_len=seq_len,
        pack_results=pack_results,
    )
    if not rows:
        raise RuntimeError("No tokenized results were packable")
    random.shuffle(rows)
    row_plans = [
        _prefix_tree_row_plan(
            row,
            seq_len=seq_len,
            pack_results=pack_results,
        )
        for row in rows
    ]

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
    route_shape = next(
        (
            shape
            for row in rows
            if (shape := _first_item_moe_route_shape(row)) is not None
        ),
        None,
    )
    route_tensor_np: np.ndarray | None = None
    route_mask_np: np.ndarray | None = None
    max_expert_id = 0
    if include_moe_routing:
        if route_shape is None:
            raise RuntimeError("No MoE routes were packed")
        num_layers, topk = route_shape
        route_tensor_np = np.zeros(
            (num_sequences, seq_len, num_layers, topk), dtype=np.int32
        )
        route_mask_np = np.zeros((num_sequences, seq_len), dtype=np.bool_)

    for index, (row, plan) in enumerate(zip(rows, row_plans, strict=True)):
        row_route_tensor = (
            route_tensor_np[index] if route_tensor_np is not None else None
        )
        row_route_mask = route_mask_np[index] if route_mask_np is not None else None
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
            route_mask=row_route_mask,
            route_shape=route_shape,
            include_moe_routing=include_moe_routing,
        )
        pixel_values.append(_packed_row_tensor_list(row, "pixel_values"))
        image_grid_thw.append(_packed_row_tensor_list(row, "image_grid_thw"))
    if include_moe_routing:
        assert route_tensor_np is not None and route_mask_np is not None
        if bool(route_mask_np.any()):
            max_expert_id = int(route_tensor_np.max())

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
    }
    if include_moe_routing:
        assert route_tensor_np is not None and route_mask_np is not None
        assert route_shape is not None
        num_layers, topk = route_shape
        if not bool(route_mask_np.any()):
            raise RuntimeError("No MoE routes were packed")
        moe_routing_pack_stats.packed_tokens = int(route_mask_np.sum())
        packed_tensors["moe_routing_replay"] = PackedMoeRoutingReplay(
            expert_indices=torch.from_numpy(route_tensor_np),
            token_mask=torch.from_numpy(route_mask_np),
            num_layers=num_layers,
            topk=topk,
            num_experts=max(topk, max_expert_id + 1),
            pack_stats=moe_routing_pack_stats,
        )
    return packed_tensors


def _prefix_tree_pack_rows(
    items: list[_PrefixTreePackItem],
    *,
    seq_len: int,
    pack_results: bool,
) -> list[list[_PrefixTreePackItem]]:
    if not items:
        return []
    if not pack_results:
        return [[item] for item in items]

    trie = _PrefixTrie()
    prefixes = [item.token_ids[: item.shareable_length] for item in items]
    for prefix in prefixes:
        trie.insert(prefix)
    leaves = [
        _PrefixTrieLeaf(
            item=item,
            edge_path=trie.edge_path(prefix),
            suffix_len=len(item.token_ids) - item.shareable_length,
        )
        for item, prefix in zip(items, prefixes, strict=True)
    ]
    grouped: dict[int, list[_PrefixTrieLeaf]] = {}
    for leaf in leaves:
        grouped.setdefault(leaf.item.prompt_id, []).append(leaf)
    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: max(leaf.empty_bin_cost for leaf in group),
        reverse=True,
    )
    bins: list[_PrefixTrieBin] = []
    for leaf in (leaf for group in ordered_groups for leaf in group):
        if leaf.empty_bin_cost > seq_len:
            raise RuntimeError(
                "Prefix-tree pack item exceeds sequence length: "
                f"cost={leaf.empty_bin_cost}, seq_len={seq_len}"
            )
        best_bin: _PrefixTrieBin | None = None
        best_remaining = seq_len + 1
        for candidate in bins:
            new_count = candidate.token_count + candidate.insertion_delta(leaf)
            if new_count <= seq_len:
                remaining = seq_len - new_count
                if remaining < best_remaining:
                    best_bin = candidate
                    best_remaining = remaining
        if best_bin is None:
            best_bin = _PrefixTrieBin()
            bins.append(best_bin)
        best_bin.add(leaf)
    return [packed_bin.items for packed_bin in bins]


def _prefix_trie_common_length(
    edge_tokens: tuple[int, ...],
    tokens: tuple[int, ...],
    start: int,
) -> int:
    limit = min(len(edge_tokens), len(tokens) - start)
    if limit == len(edge_tokens) and edge_tokens == tokens[start : start + limit]:
        return limit
    index = 0
    while index < limit and edge_tokens[index] == tokens[start + index]:
        index += 1
    return index


def _prefix_trie_edge_matches(
    edge_tokens: tuple[int, ...],
    tokens: tuple[int, ...],
    start: int,
) -> bool:
    if start + len(edge_tokens) > len(tokens):
        return False
    return edge_tokens == tokens[start : start + len(edge_tokens)]


def _prefix_tree_pack_item(
    result: TokenizedResult,
    *,
    seq_len: int,
) -> _PrefixTreePackItem:
    assistant_mask = np.asarray(result.assistant_mask, dtype=np.bool_)
    logprobs = np.asarray(result.logprobs, dtype=np.float32)
    shareable_length = min(
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
) -> _PrefixTreeRowPlan:
    segments = _prefix_tree_pack_segments(
        (item.token_ids for item in row),
        max_depth=seq_len if pack_results else 0,
        shareable_lengths=(
            item.shareable_length if pack_results else 0 for item in row
        ),
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
    route_mask: np.ndarray | None,
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
            assert route_tensor is not None and route_mask is not None
            assert route_shape is not None
            _copy_moe_route_slice(
                route_tensor=route_tensor,
                route_mask=route_mask,
                dst_start=dst_start,
                src_start=src_start,
                src_end=src_end,
                raw_routes=item.moe_routes,
                route_shape=route_shape,
            )


def _pack_prefix_tree_row(
    row: list[_PrefixTreePackItem],
    *,
    seq_len: int,
    pack_results: bool,
    include_moe_routing: bool,
) -> _PackedPrefixTreeRow:
    if not row:
        empty_i64 = np.empty((0,), dtype=np.int64)
        empty_f32 = np.empty((0,), dtype=np.float32)
        return _PackedPrefixTreeRow(
            token_ids=empty_i64,
            group_ids=empty_i64,
            parent_ids=empty_i64,
            input_pos=empty_i64,
            assistant_mask=np.empty((0,), dtype=np.bool_),
            logprobs=empty_f32,
            advantages=empty_f32,
            weights=empty_f32,
            pixel_values=None,
            image_grid_thw=None,
        )
    plan = _prefix_tree_row_plan(row, seq_len=seq_len, pack_results=pack_results)
    length = plan.length
    token_ids = np.empty(length, dtype=np.int64)
    group_ids = np.empty(length, dtype=np.int64)
    parent_ids = np.empty(length, dtype=np.int64)
    input_pos = np.zeros(length, dtype=np.int64)
    assistant_mask = np.zeros(length, dtype=np.bool_)
    logprobs = np.full(length, np.nan, dtype=np.float32)
    advantages = np.zeros(length, dtype=np.float32)
    weights = np.zeros(length, dtype=np.float32)
    route_shape = _first_item_moe_route_shape(row) if include_moe_routing else None
    route_tensor: np.ndarray | None = None
    route_mask: np.ndarray | None = None
    max_expert_id = 0
    if route_shape is not None:
        route_tensor = np.zeros(
            (length, route_shape[0], route_shape[1]), dtype=np.int32
        )
        route_mask = np.zeros(length, dtype=np.bool_)
    _materialize_prefix_tree_row(
        row,
        plan=plan,
        token_ids=token_ids,
        group_ids=group_ids,
        parent_ids=parent_ids,
        input_pos=input_pos,
        assistant_mask=assistant_mask,
        logprobs=logprobs,
        advantages=advantages,
        weights=weights,
        route_tensor=route_tensor,
        route_mask=route_mask,
        route_shape=route_shape,
        include_moe_routing=include_moe_routing,
    )
    max_expert_id = (
        int(route_tensor.max())
        if route_tensor is not None
        and route_mask is not None
        and bool(route_mask.any())
        else 0
    )
    return _PackedPrefixTreeRow(
        token_ids=token_ids[:length],
        group_ids=group_ids[:length],
        parent_ids=parent_ids[:length],
        input_pos=input_pos,
        assistant_mask=assistant_mask,
        logprobs=logprobs,
        advantages=advantages,
        weights=weights,
        pixel_values=_packed_row_tensor_list(row, "pixel_values"),
        image_grid_thw=_packed_row_tensor_list(row, "image_grid_thw"),
        route_tensor=route_tensor,
        route_mask=route_mask,
        max_expert_id=max_expert_id,
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


def _first_item_moe_route_shape(
    row: list[_PrefixTreePackItem],
) -> tuple[int, int] | None:
    for item in row:
        if item.moe_routes is not None:
            shape = _moe_route_shape(item.moe_routes)
            if shape is not None:
                return shape
    return None


def _moe_route_shape(raw: Any) -> tuple[int, int] | None:
    if isinstance(raw, MoeRouteSegments):
        return int(raw.shape[1]), int(raw.shape[2])
    routes = _coerce_moe_routes(raw)
    if routes.shape[0] == 0:
        return None
    return int(routes.shape[1]), int(routes.shape[2])


def _coerce_moe_routes(raw: Any) -> MoeRouteArray:
    if isinstance(raw, np.ndarray):
        routes = raw.astype(np.int32, copy=False)
    elif isinstance(raw, list):
        first = next((route for route in raw if route is not None), None)
        if first is None:
            raise RuntimeError("No MoE routes were packed")
        routes = np.full(
            (len(raw), len(first), len(first[0])), MISSING_EXPERT_ID, dtype=np.int32
        )
        for index, route in enumerate(raw):
            if route is not None:
                routes[index] = route
    else:
        raise RuntimeError(f"Expected MoE routes array, got {type(raw)}")
    if routes.ndim != 3 or routes.shape[1] <= 0 or routes.shape[2] <= 0:
        raise RuntimeError(f"Packed MoE routes must be rank 3, got {routes.shape}")
    return routes


def _copy_moe_route_slice(
    *,
    route_tensor: np.ndarray,
    route_mask: np.ndarray,
    dst_start: int,
    src_start: int,
    src_end: int,
    raw_routes: Any,
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
            _copy_valid_moe_route_chunk(
                route_tensor=route_tensor,
                route_mask=route_mask,
                dst_start=segment_dst_start,
                routes=segment,
                assume_valid=True,
            )
            covered_until = segment_start + int(segment.shape[0])
        if covered_until != src_end:
            raise RuntimeError("Segmented MoE routes did not cover packed source slice")
        return

    routes = _coerce_moe_routes(raw_routes)
    route_slice = routes[src_start:src_end]
    if tuple(route_slice.shape[1:]) != route_shape:
        raise RuntimeError("Packed MoE routes must have one rectangular shape")
    _copy_valid_moe_route_chunk(
        route_tensor=route_tensor,
        route_mask=route_mask,
        dst_start=dst_start,
        routes=route_slice,
    )


def _copy_valid_moe_route_chunk(
    *,
    route_tensor: np.ndarray,
    route_mask: np.ndarray,
    dst_start: int,
    routes: np.ndarray,
    assume_valid: bool = False,
) -> None:
    if int(routes.shape[0]) == 0:
        return
    if assume_valid:
        dst_end = dst_start + int(routes.shape[0])
        route_tensor[dst_start:dst_end] = routes
        route_mask[dst_start:dst_end] = True
        return
    valid = np.all(routes != MISSING_EXPERT_ID, axis=(1, 2))
    if not bool(valid.any()):
        return
    if bool(valid.all()):
        dst_end = dst_start + int(routes.shape[0])
        route_tensor[dst_start:dst_end] = routes
        route_mask[dst_start:dst_end] = True
        return
    valid_offsets = np.nonzero(valid)[0]
    route_tensor[dst_start + valid_offsets] = routes[valid_offsets]
    route_mask[dst_start + valid_offsets] = True


def _copy_source_moe_route(
    *,
    route_tensor: np.ndarray,
    route_mask: np.ndarray,
    dst_index: int,
    source_index: int,
    raw_routes: Any,
    route_shape: tuple[int, int],
) -> int:
    if isinstance(raw_routes, MoeRouteSegments):
        for segment_start, segment in raw_routes.iter_slices(
            source_index, source_index + 1
        ):
            if tuple(segment.shape[1:]) != route_shape:
                raise RuntimeError("Packed MoE routes must have one rectangular shape")
            route = segment[source_index - segment_start]
            return _copy_valid_moe_route(
                route_tensor=route_tensor,
                route_mask=route_mask,
                dst_index=dst_index,
                route=route,
            )
        raise RuntimeError(f"Segmented MoE routes did not cover row {source_index}")

    routes = _coerce_moe_routes(raw_routes)
    route = routes[source_index]
    if tuple(route.shape) != route_shape:
        raise RuntimeError("Packed MoE routes must have one rectangular shape")
    return _copy_valid_moe_route(
        route_tensor=route_tensor,
        route_mask=route_mask,
        dst_index=dst_index,
        route=route,
    )


def _copy_valid_moe_route(
    *,
    route_tensor: np.ndarray,
    route_mask: np.ndarray,
    dst_index: int,
    route: np.ndarray,
) -> int:
    valid = bool(np.all(route != MISSING_EXPERT_ID))
    if not valid:
        return 0
    route_tensor[dst_index] = route
    route_mask[dst_index] = True
    return int(route.max()) if route.size else 0


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
