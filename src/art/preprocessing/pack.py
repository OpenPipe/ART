import math
import os
import random
import time
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import torch
from typing_extensions import NotRequired, TypedDict, Unpack

from ..megatron.prefix_tree_packing import (
    prefix_tree_pack as _prefix_tree_pack_sequences,
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
    input_pos: tuple[int, ...]
    assistant_mask: tuple[int, ...]
    logprobs: tuple[float, ...]
    advantage: float
    weight: float
    prompt_id: int
    shareable_length: int
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    moe_routes: Any | None


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
    packed_rows = [
        _pack_prefix_tree_row(
            row,
            seq_len=seq_len,
            pack_results=pack_results,
            include_moe_routing=include_moe_routing,
        )
        for row in rows
    ]

    num_sequences = len(packed_rows)
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
            tuple(row.route_tensor.shape[1:])
            for row in packed_rows
            if row.route_tensor is not None
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

    for index, row in enumerate(packed_rows):
        length = min(int(row.token_ids.shape[0]), seq_len)
        tokens_np[index, :length] = row.token_ids[:length]
        group_ids_np[index, :length] = row.group_ids[:length]
        parent_ids_np[index, :length] = row.parent_ids[:length]
        input_pos_np[index, :length] = row.input_pos[:length]
        assistant_mask_np[index, :length] = row.assistant_mask[:length]
        logprobs_np[index, :length] = row.logprobs[:length]
        advantages_np[index, :length] = row.advantages[:length]
        weights_np[index, :length] = row.weights[:length]
        pixel_values.append(row.pixel_values)
        image_grid_thw.append(row.image_grid_thw)
        if include_moe_routing:
            assert route_tensor_np is not None and route_mask_np is not None
            assert row.route_tensor is not None and row.route_mask is not None
            route_tensor_np[index, :length] = row.route_tensor[:length]
            route_mask_np[index, :length] = row.route_mask[:length]
            max_expert_id = max(max_expert_id, row.max_expert_id)

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
    for index, token in enumerate(edge_tokens):
        if tokens[start + index] != token:
            return False
    return True


def _prefix_tree_pack_item(
    result: TokenizedResult,
    *,
    seq_len: int,
) -> _PrefixTreePackItem:
    shareable_length = min(
        int(result.prompt_length),
        max(_first_trainable_token_index(result) - 1, 0),
    )
    item = _PrefixTreePackItem(
        token_ids=tuple(int(value) for value in result.token_ids),
        input_pos=tuple(int(value) for value in result.input_pos),
        assistant_mask=tuple(int(value) for value in result.assistant_mask),
        logprobs=tuple(float(value) for value in result.logprobs),
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


def _first_trainable_token_index(result: TokenizedResult) -> int:
    return next(
        (
            index
            for index, (is_assistant, logprob) in enumerate(
                zip(result.assistant_mask, result.logprobs, strict=True)
            )
            if bool(is_assistant) or not math.isnan(float(logprob))
        ),
        len(result.token_ids),
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
    tree = _prefix_tree_pack_sequences(
        (torch.tensor(item.token_ids, dtype=torch.long) for item in row),
        max_depth=seq_len if pack_results else 0,
        shareable_lengths=(
            item.shareable_length if pack_results else 0 for item in row
        ),
    )
    token_ids = np.asarray(tree.tokens.reshape(-1).numpy(), dtype=np.int64)
    length = min(int(token_ids.shape[0]), seq_len)
    assigned = np.zeros(length, dtype=np.bool_)
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
    for item_index, item in enumerate(row):
        packed_positions = tree.positions_by_sequence[item_index].cpu().numpy()
        for source_index, packed_index in enumerate(packed_positions):
            packed_index = int(packed_index)
            if packed_index >= length:
                continue
            _validate_prefix_tree_assignment(
                item,
                source_index=source_index,
                packed_index=packed_index,
                token_ids=token_ids,
                input_pos=input_pos,
                assigned=assigned,
            )
            if assigned[packed_index]:
                continue
            assigned[packed_index] = True
            input_pos[packed_index] = item.input_pos[source_index]
            assistant_mask[packed_index] = item.assistant_mask[source_index]
            logprobs[packed_index] = item.logprobs[source_index]
            advantages[packed_index] = item.advantage
            weights[packed_index] = item.weight
            if include_moe_routing:
                assert route_tensor is not None and route_mask is not None
                assert route_shape is not None
                max_expert_id = max(
                    max_expert_id,
                    _copy_source_moe_route(
                        route_tensor=route_tensor,
                        route_mask=route_mask,
                        dst_index=packed_index,
                        source_index=source_index,
                        raw_routes=item.moe_routes,
                        route_shape=route_shape,
                    ),
                )
    return _PackedPrefixTreeRow(
        token_ids=token_ids[:length],
        group_ids=np.asarray(tree.group_ids.reshape(-1).numpy(), dtype=np.int64)[
            :length
        ],
        parent_ids=np.asarray(tree.parent_ids.reshape(-1).numpy(), dtype=np.int64)[
            :length
        ],
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


def _validate_prefix_tree_assignment(
    item: _PrefixTreePackItem,
    *,
    source_index: int,
    packed_index: int,
    token_ids: np.ndarray,
    input_pos: np.ndarray,
    assigned: np.ndarray,
) -> None:
    if token_ids[packed_index] != item.token_ids[source_index]:
        raise RuntimeError("Prefix-tree pack token assignment mismatch")
    if not assigned[packed_index]:
        return
    if input_pos[packed_index] != item.input_pos[source_index]:
        raise RuntimeError("Prefix-tree pack cannot share mismatched input positions")
    if item.assistant_mask[source_index] or not math.isnan(item.logprobs[source_index]):
        raise RuntimeError("Prefix-tree pack attempted to share a trainable token")


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
