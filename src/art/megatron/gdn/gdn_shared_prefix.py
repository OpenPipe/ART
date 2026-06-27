from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.shared_prefix_tree import parse_shared_prefix_tree

GdnSegmentKind = Literal["prefix", "completion"]
# FLA's public chunk_gated_delta_rule hard-codes 64-token WY chunks.
FLA_CHUNK_SIZE = 64


@dataclass(frozen=True)
class GdnSegmentSpec:
    """Contiguous logical GDN segment in one packed row."""

    row_index: int
    family_index: int
    group_id: int
    parent_id: int
    start: int
    end: int
    kind: GdnSegmentKind
    child_index: int | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class GdnPackedExecutionSpec:
    """Parsed shared-prefix GDN execution metadata for a packed batch."""

    batch_size: int
    sequence_length: int
    valid_lengths: tuple[int, ...]
    tree_segments: tuple[GdnSegmentSpec, ...]
    tree_parent_indices: tuple[int, ...]
    tree_depths: tuple[int, ...]

    @property
    def family_count(self) -> int:
        return len(self.tree_segments)

    @property
    def real_token_count(self) -> int:
        return sum(self.valid_lengths)


@dataclass(frozen=True)
class GdnSegmentBucketPlan:
    """Device-local index tensors for a variable-length GDN segment batch."""

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
    needs_final_state: bool = True
    output_mask: torch.Tensor | None = None

    @property
    def segment_count(self) -> int:
        return int(self.family_indices.numel())

    @property
    def real_token_count(self) -> int:
        return self.real_token_count_static


@dataclass(frozen=True)
class GdnStateExchangePlan:
    """Sparse CP exchange for tree parent states needed by remote children."""

    source_family_indices: tuple[int, ...]
    dest_family_indices: tuple[int, ...]
    exchange: Any
    reverse_exchange: Any


@dataclass(frozen=True)
class GdnPlannerConfig:
    """Tunable cost coefficients for one packed-row GDN execution plan."""

    max_padding_ratio: float = 2.0
    max_segments_per_batch: int = 4096
    cp_chain_min_tokens_per_rank: int = 32
    cp_chain_min_total_tokens: int = 32768
    cp_chain_min_prefix_only_tokens: int = 32768
    cp_tree_chain_min_total_tokens: int = 8192
    cp_tree_chain_min_prefix_only_tokens: int = 8192
    rank_idle_token_cost: float = 1.0
    max_zero_exchange_load_imbalance: float = 1.5
    planner_local_token_ms: float = 0.00065
    planner_layout_cross_rank_token_ms: float = 0.00008
    planner_empty_rank_ms: float = 32.0


@dataclass(frozen=True)
class GdnRankExecutionPlan:
    """Rank-local planned execution metadata for shared-prefix GDN."""

    cp_rank: int
    cp_size: int
    batch_size: int
    sequence_length: int
    real_token_mask: torch.Tensor
    packed_batch_size: int | None = None
    packed_sequence_length: int | None = None
    attention_to_gdn: Any | None = None
    gdn_to_attention: Any | None = None
    attention_token_ranges: tuple[tuple[int, int, int], ...] = ()
    gdn_token_ranges: tuple[tuple[int, int, int], ...] = ()
    attention_token_count: int = 0
    gdn_token_count: int = 0
    tree_segment_buckets_by_depth: tuple[tuple[GdnSegmentBucketPlan, ...], ...] = ()
    tree_chain_buckets_by_depth: tuple[tuple[GdnSegmentBucketPlan, ...], ...] = ()
    tree_state_exchanges_by_depth: tuple[GdnStateExchangePlan | None, ...] = ()

    @property
    def attention_token_indices(self) -> tuple[int, ...]:
        return _tokens_from_rank_ranges(self.attention_token_ranges)

    @property
    def gdn_token_indices(self) -> tuple[int, ...]:
        return _tokens_from_rank_ranges(self.gdn_token_ranges)


@dataclass(frozen=True)
class _AttentionLayoutIndex:
    """Counting index for CP attention token ownership."""

    token_ranges_by_rank: tuple[tuple[tuple[int, int], ...], ...]
    token_range_ends_by_rank: tuple[tuple[int, ...], ...]


def _tokens_from_rank_ranges(
    ranges: tuple[tuple[int, int, int], ...],
) -> tuple[int, ...]:
    return tuple(token for start, end, _ in ranges for token in range(start, end))


def build_gdn_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int = 0,
    cp_size: int = 1,
    attention_token_layout_index: TokenLayoutIndex | None = None,
    planner_config: GdnPlannerConfig | None = None,
) -> GdnRankExecutionPlan:
    """Build rank-local tensor metadata from a parsed shared-prefix DAG.

    Planning is CPU-bound and must run once per packed training sequence. CP>1
    emits mixed work: native FLA CP chain buckets for long segments and local
    fork buckets for short work where CP collectives would be inefficient.
    """

    planner_config = planner_config or GdnPlannerConfig()
    target_device = torch.device(device)
    if target_device.type != "cpu":
        cpu_plan = build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=cp_rank,
            cp_size=cp_size,
            attention_token_layout_index=attention_token_layout_index,
            planner_config=planner_config,
        )
        return move_gdn_rank_execution_plan_to_device(cpu_plan, target_device)
    return _build_tree_rank_execution_plan(
        spec,
        device=device,
        cp_rank=cp_rank,
        cp_size=cp_size,
        attention_token_layout_index=attention_token_layout_index,
        planner_config=planner_config,
    )


def _build_tree_rank_execution_plan(
    spec: GdnPackedExecutionSpec,
    *,
    device: torch.device | str,
    cp_rank: int,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None,
    planner_config: GdnPlannerConfig,
) -> GdnRankExecutionPlan:
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")
    if not spec.tree_segments:
        raise ValueError("tree GDN planning requires tree segments")
    if len(spec.tree_parent_indices) != len(spec.tree_segments):
        raise ValueError("tree parent metadata length must match tree segments")
    if len(spec.tree_depths) != len(spec.tree_segments):
        raise ValueError("tree depth metadata length must match tree segments")

    from art.megatron.gdn.layout import (
        _reverse_exchange_plan,
        build_local_rank_cp_exchange_plan_from_dest_ranges,
    )

    source_layout = _attention_source_layout(
        spec,
        cp_size=cp_size,
        attention_token_layout_index=attention_token_layout_index,
        planner_config=planner_config,
    )
    attention_layout_index = _build_attention_layout_index_from_token_layout(
        source_layout
    )
    segment_attention_counts = _segment_attention_rank_counts(
        spec,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )

    depth_count = max(spec.tree_depths, default=0) + 1
    rank_loads = [0] * cp_size
    owner_by_node = [-1] * len(spec.tree_segments)
    chained_nodes = [False] * len(spec.tree_segments)
    tree_has_children = [False] * len(spec.tree_segments)
    for parent_index in spec.tree_parent_indices:
        if parent_index >= 0:
            tree_has_children[parent_index] = True
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    segments_by_rank_depth: list[list[list[GdnSegmentSpec]]] = [
        [[] for _ in range(depth_count)] for _ in range(cp_size)
    ]
    chain_segments_by_depth: list[list[GdnSegmentSpec]] = [
        [] for _ in range(depth_count)
    ]
    cross_rank_token_count = 0

    children_by_node: list[list[int]] = [[] for _ in spec.tree_segments]
    root_indices: list[int] = []
    for node_index, parent_index in enumerate(spec.tree_parent_indices):
        if parent_index < 0:
            root_indices.append(node_index)
        else:
            children_by_node[parent_index].append(node_index)

    def subtree_indices(root_index: int) -> tuple[int, ...]:
        ordered: list[int] = []
        stack = [root_index]
        while stack:
            node_index = stack.pop()
            ordered.append(node_index)
            stack.extend(reversed(children_by_node[node_index]))
        return tuple(ordered)

    def assign_local_group(node_indices: tuple[int, ...]) -> None:
        nonlocal cross_rank_token_count
        segments = tuple(spec.tree_segments[index] for index in node_indices)
        owner = _best_segment_owner(
            segments,
            rank_loads,
            segment_attention_counts=segment_attention_counts,
            planner_config=planner_config,
        )
        for segment in segments:
            owner_by_node[segment.family_index] = owner
            segments_by_rank_depth[owner][
                spec.tree_depths[segment.family_index]
            ].append(segment)
            cross_rank_token_count += _append_local_segment(
                gdn_ranges_by_rank,
                rank_loads,
                owner,
                segment,
                spec,
                segment_attention_counts=segment_attention_counts,
            )

    subtree_token_counts = [segment.length for segment in spec.tree_segments]
    for node_index in reversed(range(len(spec.tree_segments))):
        for child_index in children_by_node[node_index]:
            subtree_token_counts[node_index] += subtree_token_counts[child_index]
    target_rank_load = spec.real_token_count / max(1, cp_size)
    max_local_group_tokens = max(1, int(target_rank_load))

    def assign_tree(root_index: int) -> None:
        nonlocal cross_rank_token_count
        root = spec.tree_segments[root_index]
        if (
            spec.tree_parent_indices[root_index] < 0
            and cp_size > 1
            and _can_chain_tree_segment(
                root,
                cp_size=cp_size,
                planner_config=planner_config,
            )
        ):
            chained_nodes[root.family_index] = True
            chain_segments_by_depth[spec.tree_depths[root.family_index]].append(root)
            cross_rank_token_count += _append_chain_segment(
                gdn_ranges_by_rank,
                rank_loads,
                root,
                spec,
                attention_layout_index=attention_layout_index,
            )
            for child_index in children_by_node[root_index]:
                assign_tree(child_index)
            return

        if subtree_token_counts[root_index] <= max_local_group_tokens:
            assign_local_group(subtree_indices(root_index))
            return

        assign_local_group((root_index,))
        for child_index in children_by_node[root_index]:
            assign_tree(child_index)

    for root_index in root_indices:
        assign_tree(root_index)

    gdn_ranges_by_rank_by_position = tuple(
        tuple(ranges) for ranges in gdn_ranges_by_rank
    )
    gdn_ranges_by_rank_by_source = tuple(
        tuple(sorted(ranges)) for ranges in gdn_ranges_by_rank
    )

    attention_to_gdn = build_local_rank_cp_exchange_plan_from_dest_ranges(
        source_layout=source_layout,
        device=device,
        local_rank=cp_rank,
        dest_ranges_by_rank=gdn_ranges_by_rank_by_position,
        cross_rank_token_count=cross_rank_token_count,
    )
    local_token_ranges = gdn_ranges_by_rank_by_source[cp_rank]
    tree_segment_buckets_by_depth = tuple(
        _build_tree_bucket_plans(
            tuple(segments_by_rank_depth[cp_rank][depth]),
            spec.tree_parent_indices,
            tuple(tree_has_children),
            local_token_ranges=None if cp_size == 1 else local_token_ranges,
            sequence_length=spec.sequence_length,
            device=device,
            planner_config=planner_config,
        )
        for depth in range(depth_count)
    )
    tree_chain_buckets_by_depth = (
        tuple(
            _build_tree_bucket_plans(
                tuple(chain_segments_by_depth[depth]),
                spec.tree_parent_indices,
                tuple(tree_has_children),
                local_token_ranges=local_token_ranges,
                sequence_length=spec.sequence_length,
                device=device,
                planner_config=planner_config,
                token_ranges_by_rank=tuple(
                    tuple(ranges) for ranges in gdn_ranges_by_rank_by_source
                ),
                split_by_final_state=False,
            )
            for depth in range(depth_count)
        )
        if cp_size > 1
        else tuple(() for _ in range(depth_count))
    )
    tree_state_exchanges_by_depth = _build_tree_state_exchanges_by_depth(
        spec,
        owner_by_node=tuple(owner_by_node),
        chained_nodes=tuple(chained_nodes),
        cp_rank=cp_rank,
        cp_size=cp_size,
        depth_count=depth_count,
        device=device,
    )
    if cp_size == 1:
        valid_lengths = torch.tensor(
            spec.valid_lengths, device=device, dtype=torch.long
        )
        positions = torch.arange(spec.sequence_length, device=device, dtype=torch.long)
        real_token_mask = positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
    else:
        real_token_mask = torch.ones(
            1,
            rank_loads[cp_rank],
            device=device,
            dtype=torch.bool,
        )

    return GdnRankExecutionPlan(
        cp_rank=cp_rank,
        cp_size=cp_size,
        batch_size=1 if cp_size > 1 else spec.batch_size,
        sequence_length=rank_loads[cp_rank] if cp_size > 1 else spec.sequence_length,
        packed_batch_size=spec.batch_size,
        packed_sequence_length=spec.sequence_length,
        real_token_mask=real_token_mask,
        attention_to_gdn=attention_to_gdn,
        gdn_to_attention=_reverse_exchange_plan(attention_to_gdn),
        attention_token_ranges=source_layout.ownership_ranges_by_rank[cp_rank],
        gdn_token_ranges=gdn_ranges_by_rank_by_position[cp_rank],
        attention_token_count=source_layout.token_counts_by_rank[cp_rank],
        gdn_token_count=rank_loads[cp_rank],
        tree_segment_buckets_by_depth=tree_segment_buckets_by_depth,
        tree_chain_buckets_by_depth=tree_chain_buckets_by_depth,
        tree_state_exchanges_by_depth=tree_state_exchanges_by_depth,
    )


def move_gdn_rank_execution_plan_to_device(
    plan: GdnRankExecutionPlan,
    device: torch.device | str,
) -> GdnRankExecutionPlan:
    """Move planner tensors to the execution device after CPU planning."""

    from art.megatron.gdn.layout import move_cp_exchange_plan_to_device

    return replace(
        plan,
        real_token_mask=_move_planner_tensor(plan.real_token_mask, device),
        attention_to_gdn=move_cp_exchange_plan_to_device(plan.attention_to_gdn, device),
        gdn_to_attention=move_cp_exchange_plan_to_device(plan.gdn_to_attention, device),
        tree_segment_buckets_by_depth=tuple(
            _move_bucket_plans(buckets, device)
            for buckets in plan.tree_segment_buckets_by_depth
        ),
        tree_chain_buckets_by_depth=tuple(
            _move_bucket_plans(buckets, device)
            for buckets in plan.tree_chain_buckets_by_depth
        ),
        tree_state_exchanges_by_depth=tuple(
            _move_state_exchange_plan(exchange, device)
            for exchange in plan.tree_state_exchanges_by_depth
        ),
    )


def _move_state_exchange_plan(
    exchange: GdnStateExchangePlan | None,
    device: torch.device | str,
) -> GdnStateExchangePlan | None:
    if exchange is None:
        return None
    from art.megatron.gdn.layout import move_cp_exchange_plan_to_device

    return replace(
        exchange,
        exchange=move_cp_exchange_plan_to_device(exchange.exchange, device),
        reverse_exchange=move_cp_exchange_plan_to_device(
            exchange.reverse_exchange, device
        ),
    )


def _move_bucket_plans(
    buckets: tuple[GdnSegmentBucketPlan, ...],
    device: torch.device | str,
) -> tuple[GdnSegmentBucketPlan, ...]:
    return tuple(
        replace(
            bucket,
            lengths=_move_planner_tensor(bucket.lengths, device),
            real_mask=_move_planner_tensor(bucket.real_mask, device),
            cu_seqlens=_move_planner_tensor(bucket.cu_seqlens, device),
            row_indices=_move_planner_tensor(bucket.row_indices, device),
            position_indices=_move_planner_tensor(bucket.position_indices, device),
            family_indices=_move_planner_tensor(bucket.family_indices, device),
            parent_indices=(
                _move_planner_tensor(bucket.parent_indices, device)
                if bucket.parent_indices is not None
                else None
            ),
            output_mask=(
                _move_planner_tensor(bucket.output_mask, device)
                if bucket.output_mask is not None
                else None
            ),
        )
        for bucket in buckets
    )


def parse_gdn_shared_prefix_segments(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> GdnPackedExecutionSpec:
    """Parse ART packed shared-prefix metadata into generic GDN tree nodes."""

    groups = _rank2_long_cpu("group_ids", group_ids)
    parents = _rank2_long_cpu("parent_ids", parent_ids)
    if tuple(groups.shape) != tuple(parents.shape):
        raise ValueError(
            "group_ids and parent_ids must have the same shape, got "
            f"{tuple(groups.shape)} and {tuple(parents.shape)}"
        )

    batch_size, sequence_length = (int(groups.shape[0]), int(groups.shape[1]))
    rows = parse_shared_prefix_tree(group_ids=groups, parent_ids=parents)
    tree_segments: list[GdnSegmentSpec] = []
    tree_parent_indices: list[int] = []
    tree_depths: list[int] = []
    valid_lengths: list[int] = []
    node_by_row_group: dict[tuple[int, int], int] = {}
    child_counts_by_parent: dict[int, int] = {}

    for row in rows:
        valid_lengths.append(row.valid_tokens)
        for segment in row.segments:
            node_index = len(tree_segments)
            is_root = segment.depth == 0
            parent_node_index = (
                -1 if is_root else node_by_row_group[(row.row_index, segment.parent_id)]
            )
            child_index = None
            if not is_root:
                child_index = child_counts_by_parent.get(parent_node_index, 0)
                child_counts_by_parent[parent_node_index] = child_index + 1
            tree_segments.append(
                GdnSegmentSpec(
                    row_index=row.row_index,
                    family_index=node_index,
                    group_id=segment.group_id,
                    parent_id=segment.parent_id,
                    start=segment.start,
                    end=segment.end,
                    kind="prefix" if is_root else "completion",
                    child_index=child_index,
                )
            )
            tree_parent_indices.append(parent_node_index)
            tree_depths.append(segment.depth)
            node_by_row_group[(row.row_index, segment.group_id)] = node_index

    return GdnPackedExecutionSpec(
        batch_size=batch_size,
        sequence_length=sequence_length,
        valid_lengths=tuple(valid_lengths),
        tree_segments=tuple(tree_segments),
        tree_parent_indices=tuple(tree_parent_indices),
        tree_depths=tuple(tree_depths),
    )


def _attention_source_layout(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None,
    planner_config: GdnPlannerConfig,
) -> TokenLayoutIndex:
    if attention_token_layout_index is not None:
        layout_cp_size = len(attention_token_layout_index.token_counts_by_rank)
        layout_token_count = sum(
            int(count) for count in attention_token_layout_index.token_counts_by_rank
        )
        if layout_cp_size != cp_size:
            raise ValueError(
                "attention token layout index cp_size must match GDN cp_size, got "
                f"{layout_cp_size} and {cp_size}"
            )
        if layout_token_count != spec.real_token_count:
            raise ValueError(
                "attention token layout index token count must match GDN real token "
                f"count, got {layout_token_count} and {spec.real_token_count}"
            )
        return attention_token_layout_index
    ranges_by_rank = _default_attention_layout_ranges(
        spec,
        cp_size=cp_size,
        planner_config=planner_config,
    )
    return TokenLayoutIndex(
        ownership_ranges_by_rank=ranges_by_rank,
        token_counts_by_rank=tuple(
            sum(int(end) - int(start) for start, end, _ in ranges)
            for ranges in ranges_by_rank
        ),
    )


def _can_chain_tree_segment(
    segment: GdnSegmentSpec,
    *,
    cp_size: int,
    planner_config: GdnPlannerConfig,
) -> bool:
    min_total_tokens = (
        min(
            planner_config.cp_tree_chain_min_prefix_only_tokens,
            planner_config.cp_chain_min_prefix_only_tokens,
        )
        if segment.kind == "prefix"
        else min(
            planner_config.cp_tree_chain_min_total_tokens,
            planner_config.cp_chain_min_total_tokens,
        )
    )
    return (
        segment.length >= min_total_tokens
        and segment.length >= cp_size
        and segment.length // FLA_CHUNK_SIZE >= cp_size
        and segment.length / cp_size >= planner_config.cp_chain_min_tokens_per_rank
    )


def _best_segment_owner(
    segments: tuple[GdnSegmentSpec, ...],
    rank_loads: list[int],
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    planner_config: GdnPlannerConfig,
) -> int:
    segment_length = sum(segment.length for segment in segments)
    if len(segments) == 1:
        on_rank_tokens = segment_attention_counts[_segment_key(segments[0])]
    else:
        rank_count = len(rank_loads)
        counts_by_rank = [0] * rank_count
        for segment in segments:
            segment_counts = segment_attention_counts[_segment_key(segment)]
            for rank in range(rank_count):
                counts_by_rank[rank] += segment_counts[rank]
        on_rank_tokens = tuple(counts_by_rank)
    best: tuple[float, float, int, int, int, int] | None = None
    for rank, tokens in enumerate(on_rank_tokens):
        projected_loads = list(rank_loads)
        projected_loads[rank] += segment_length
        max_load = max(projected_loads, default=0)
        target_load = sum(projected_loads) / max(1, len(projected_loads))
        overload = max(
            0.0,
            max_load - planner_config.max_zero_exchange_load_imbalance * target_load,
        )
        idle_tokens = sum(max_load - load for load in projected_loads)
        cross_rank_tokens = segment_length - int(tokens)
        empty_rank_count = sum(1 for load in projected_loads if load == 0)
        score = (
            max_load * planner_config.planner_local_token_ms
            + idle_tokens
            * planner_config.rank_idle_token_cost
            * planner_config.planner_local_token_ms
            + cross_rank_tokens * planner_config.planner_layout_cross_rank_token_ms
            + empty_rank_count * planner_config.planner_empty_rank_ms
        )
        candidate = (
            overload,
            score,
            max_load,
            cross_rank_tokens,
            -int(tokens),
            rank,
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return _least_loaded_rank(rank_loads)
    return best[-1]


def _build_tree_state_exchanges_by_depth(
    spec: GdnPackedExecutionSpec,
    *,
    owner_by_node: tuple[int, ...],
    chained_nodes: tuple[bool, ...],
    cp_rank: int,
    cp_size: int,
    depth_count: int,
    device: torch.device | str,
) -> tuple[GdnStateExchangePlan | None, ...]:
    if cp_size <= 1:
        return tuple(None for _ in range(depth_count))

    from art.megatron.gdn.layout import (
        GdnCpExchangePlan,
        _make_peer_transfer,
        _reverse_exchange_plan,
    )

    families_by_depth_pair: list[dict[tuple[int, int], set[int]]] = [
        {} for _ in range(depth_count)
    ]
    for child_index, parent_index in enumerate(spec.tree_parent_indices):
        if parent_index < 0 or chained_nodes[parent_index]:
            continue
        source_rank = owner_by_node[parent_index]
        dest_rank = owner_by_node[child_index]
        if source_rank < 0 or dest_rank < 0:
            raise ValueError("tree state exchange requires every node to have an owner")
        if source_rank == dest_rank:
            continue
        depth = spec.tree_depths[child_index]
        families_by_depth_pair[depth].setdefault((source_rank, dest_rank), set()).add(
            parent_index
        )

    state_exchanges: list[GdnStateExchangePlan | None] = []
    for pair_families in families_by_depth_pair:
        if not pair_families:
            state_exchanges.append(None)
            continue
        source_families_by_rank = [set[int]() for _ in range(cp_size)]
        dest_families_by_rank = [set[int]() for _ in range(cp_size)]
        for (source_rank, dest_rank), parent_indices in pair_families.items():
            source_families_by_rank[source_rank].update(parent_indices)
            dest_families_by_rank[dest_rank].update(parent_indices)
        source_families = tuple(
            tuple(sorted(families)) for families in source_families_by_rank
        )
        dest_families = tuple(
            tuple(sorted(families)) for families in dest_families_by_rank
        )
        source_positions = (
            {family: index for index, family in enumerate(families)}
            for families in source_families
        )
        dest_positions = (
            {family: index for index, family in enumerate(families)}
            for families in dest_families
        )
        source_position_by_rank = tuple(source_positions)
        dest_position_by_rank = tuple(dest_positions)
        transfers = []
        transfer_count = 0
        for (source_rank, dest_rank), parent_indices in sorted(pair_families.items()):
            ordered = tuple(sorted(parent_indices))
            transfer_count += len(ordered)
            transfers.append(
                _make_peer_transfer(
                    source_rank=source_rank,
                    dest_rank=dest_rank,
                    source_positions=torch.tensor(
                        [
                            source_position_by_rank[source_rank][family]
                            for family in ordered
                        ],
                        dtype=torch.long,
                    ),
                    dest_positions=torch.tensor(
                        [
                            dest_position_by_rank[dest_rank][family]
                            for family in ordered
                        ],
                        dtype=torch.long,
                    ),
                    source_count=len(source_families[source_rank]),
                    dest_count=len(dest_families[dest_rank]),
                    device=device,
                )
            )
        exchange = GdnCpExchangePlan(
            cp_size=cp_size,
            source_token_counts_by_rank=tuple(
                len(families) for families in source_families
            ),
            dest_token_counts_by_rank=tuple(
                len(families) for families in dest_families
            ),
            transfers=tuple(transfers),
            cross_rank_token_count_override=transfer_count,
        )
        state_exchanges.append(
            GdnStateExchangePlan(
                source_family_indices=source_families[cp_rank],
                dest_family_indices=dest_families[cp_rank],
                exchange=exchange,
                reverse_exchange=_reverse_exchange_plan(exchange),
            )
        )
    return tuple(state_exchanges)


def _build_attention_layout_index_from_token_layout(
    layout: TokenLayoutIndex,
) -> _AttentionLayoutIndex:
    ranges_by_rank = tuple(
        tuple(sorted((int(start), int(end)) for start, end, _ in rank_ranges))
        for rank_ranges in layout.ownership_ranges_by_rank
    )
    return _AttentionLayoutIndex(
        token_ranges_by_rank=ranges_by_rank,
        token_range_ends_by_rank=tuple(
            tuple(end for _, end in ranges) for ranges in ranges_by_rank
        ),
    )


def _segment_attention_rank_counts(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex,
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    del cp_size
    segments = spec.tree_segments
    if not segments:
        return {}
    starts = torch.tensor(
        [_segment_token_start(segment, spec.sequence_length) for segment in segments],
        dtype=torch.long,
    )
    lengths = torch.tensor([segment.length for segment in segments], dtype=torch.long)
    ends = starts + lengths
    counts_by_rank = []
    for ranges in attention_layout_index.token_ranges_by_rank:
        counts_by_rank.append(_rank_range_overlap_counts(starts, ends, ranges))
    counts_tensor = torch.stack(counts_by_rank, dim=1)
    totals = counts_tensor.sum(dim=1)
    if not torch.equal(totals, lengths):
        bad_index = int(torch.nonzero(totals != lengths, as_tuple=False)[0].item())
        raise ValueError(
            "attention layout is missing a real token required by GDN; "
            f"segment={_segment_key(segments[bad_index])}"
        )
    counts = counts_tensor.tolist()
    return {
        _segment_key(segment): tuple(int(value) for value in counts[index])
        for index, segment in enumerate(segments)
    }


def _rank_range_overlap_counts(
    starts: torch.Tensor,
    ends: torch.Tensor,
    ranges: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    if not ranges:
        return torch.zeros_like(starts)
    range_starts = torch.tensor([start for start, _ in ranges], dtype=torch.long)
    range_ends = torch.tensor([end for _, end in ranges], dtype=torch.long)
    range_lengths = range_ends - range_starts
    prefix = torch.cat((range_lengths.new_zeros(1), torch.cumsum(range_lengths, dim=0)))

    def owned_before(points: torch.Tensor) -> torch.Tensor:
        indices = torch.searchsorted(range_ends, points, right=False)
        counts = prefix.index_select(0, indices)
        active = indices < int(range_starts.numel())
        if bool(active.any().item()):
            active_indices = indices[active]
            active_starts = range_starts.index_select(0, active_indices)
            active_ends = range_ends.index_select(0, active_indices)
            counts[active] += torch.minimum(
                torch.clamp(points[active] - active_starts, min=0),
                active_ends - active_starts,
            )
        return counts

    return owned_before(ends) - owned_before(starts)


def _segment_key(segment: GdnSegmentSpec) -> tuple[int, int, int]:
    return (segment.row_index, segment.start, segment.end)


def _default_attention_layout_ranges(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    planner_config: GdnPlannerConfig,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    ranks: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    loads = [0] * cp_size
    target_rank_load = spec.real_token_count / cp_size

    def append_segment(rank: int, token_start: int, token_count: int) -> None:
        ranks[rank].append((token_start, token_start + token_count, loads[rank]))
        loads[rank] += token_count

    def should_split_segment(segment: GdnSegmentSpec) -> bool:
        if segment.length <= planner_config.max_zero_exchange_load_imbalance * (
            target_rank_load
        ):
            return False
        return _can_chain_tree_segment(
            segment, cp_size=cp_size, planner_config=planner_config
        )

    for segment in spec.tree_segments:
        token_start = _segment_token_start(segment, spec.sequence_length)
        if should_split_segment(segment):
            for rank in range(cp_size):
                start = (segment.length * rank) // cp_size
                end = (segment.length * (rank + 1)) // cp_size
                ranks[rank].append(
                    (token_start + start, token_start + end, loads[rank])
                )
                loads[rank] += end - start
            continue
        owner = _least_loaded_rank(loads)
        append_segment(owner, token_start, segment.length)
    return tuple(tuple(ranges) for ranges in ranks)


def _append_chain_segment(
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]],
    rank_loads: list[int],
    segment: GdnSegmentSpec,
    spec: GdnPackedExecutionSpec,
    *,
    attention_layout_index: _AttentionLayoutIndex | None = None,
) -> int:
    token_start = _segment_token_start(segment, spec.sequence_length)
    cp_size = len(gdn_ranges_by_rank)
    attention_shards = _attention_contiguous_chain_shards(
        token_start,
        segment.length,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )
    if attention_shards is not None:
        for rank, shard in enumerate(attention_shards):
            position_start = rank_loads[rank]
            gdn_ranges_by_rank[rank].append((shard.start, shard.stop, position_start))
            rank_loads[rank] += len(shard)
        return 0
    cross_rank_tokens = 0
    shard_lengths = _fla_aligned_chain_shard_lengths(segment.length, cp_size=cp_size)
    start = 0
    for rank, shard_length in enumerate(shard_lengths):
        end = start + shard_length
        if start >= end:
            raise ValueError(
                "CP chain planning requires non-empty shards; "
                f"segment={segment.kind}:{segment.family_index} "
                f"length={segment.length} cp_size={cp_size}"
            )
        shard_start = token_start + start
        position_start = rank_loads[rank]
        gdn_ranges_by_rank[rank].append(
            (shard_start, shard_start + shard_length, position_start)
        )
        rank_loads[rank] += shard_length
        if attention_layout_index is not None:
            cross_rank_tokens += shard_length - _range_overlap_count(
                shard_start,
                shard_start + shard_length,
                attention_layout_index.token_ranges_by_rank[rank],
                attention_layout_index.token_range_ends_by_rank[rank],
            )
        start = end
    return cross_rank_tokens


def _fla_aligned_chain_shard_lengths(length: int, *, cp_size: int) -> tuple[int, ...]:
    full_chunks = int(length) // FLA_CHUNK_SIZE
    if full_chunks < int(cp_size):
        raise ValueError(
            "CP chain planning requires at least one full FLA chunk per rank; "
            f"length={length} cp_size={cp_size}"
        )
    base_chunks = full_chunks // int(cp_size)
    extra_chunks = full_chunks % int(cp_size)
    chunk_counts = tuple(
        base_chunks + (1 if rank < extra_chunks else 0) for rank in range(int(cp_size))
    )
    lengths = [count * FLA_CHUNK_SIZE for count in chunk_counts]
    lengths[-1] += int(length) - full_chunks * FLA_CHUNK_SIZE
    return tuple(lengths)


def _attention_contiguous_chain_shards(
    token_start: int,
    token_count: int,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex | None,
) -> tuple[range, ...] | None:
    if attention_layout_index is None:
        return None
    segment_end = token_start + token_count
    shards: list[range] = []
    cursor = token_start
    for rank in range(cp_size):
        overlaps = _range_overlaps(
            token_start,
            segment_end,
            attention_layout_index.token_ranges_by_rank[rank],
        )
        if len(overlaps) != 1:
            return None
        start, end = overlaps[0]
        if start != cursor or end <= start:
            return None
        shards.append(range(start, end))
        cursor = end
    if cursor != segment_end:
        return None
    if any(len(shard) % FLA_CHUNK_SIZE != 0 for shard in shards[:-1]):
        return None
    return tuple(shards)


def _append_local_segment(
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]],
    rank_loads: list[int],
    rank: int,
    segment: GdnSegmentSpec,
    spec: GdnPackedExecutionSpec,
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
) -> int:
    token_start = _segment_token_start(segment, spec.sequence_length)
    position_start = rank_loads[rank]
    gdn_ranges_by_rank[rank].append(
        (token_start, token_start + segment.length, position_start)
    )
    rank_loads[rank] += segment.length
    return segment.length - segment_attention_counts[_segment_key(segment)][rank]


def _least_loaded_rank(rank_loads: list[int]) -> int:
    return min(range(len(rank_loads)), key=lambda rank: (rank_loads[rank], rank))


def _build_tree_bucket_plans(
    segments: tuple[GdnSegmentSpec, ...],
    tree_parent_indices: tuple[int, ...],
    tree_has_children: tuple[bool, ...],
    *,
    local_token_ranges: tuple[tuple[int, int, int], ...] | None,
    sequence_length: int,
    device: torch.device | str,
    planner_config: GdnPlannerConfig,
    token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] | None = None,
    split_by_final_state: bool = True,
) -> tuple[GdnSegmentBucketPlan, ...]:
    segment_buckets = (
        _batch_tree_segments_by_padded_work(
            segments,
            tree_has_children,
            max_padding_ratio=planner_config.max_padding_ratio,
            max_segments_per_batch=planner_config.max_segments_per_batch,
        )
        if split_by_final_state
        else _batch_segments_by_padded_work(
            segments,
            max_padding_ratio=planner_config.max_padding_ratio,
            max_segments_per_batch=planner_config.max_segments_per_batch,
        )
    )
    return tuple(
        _bucket_with_tree_parent_indices(
            (
                _build_segment_bucket_plan(bucket, device=device)
                if local_token_ranges is None
                else _build_position_bucket_plan(
                    bucket,
                    local_token_ranges,
                    sequence_length=sequence_length,
                    device=device,
                    token_ranges_by_rank=token_ranges_by_rank,
                )
            ),
            bucket,
            tree_parent_indices,
            tree_has_children,
            device=device,
        )
        for bucket in segment_buckets
    )


def _bucket_with_tree_parent_indices(
    plan: GdnSegmentBucketPlan,
    segments: tuple[GdnSegmentSpec, ...],
    tree_parent_indices: tuple[int, ...],
    tree_has_children: tuple[bool, ...],
    *,
    device: torch.device | str,
) -> GdnSegmentBucketPlan:
    parent_indices = torch.tensor(
        [tree_parent_indices[segment.family_index] for segment in segments],
        dtype=torch.long,
    )
    return replace(
        plan,
        parent_indices=_move_planner_tensor(parent_indices, device),
        parent_indices_cpu=parent_indices,
        needs_final_state=any(
            tree_has_children[segment.family_index] for segment in segments
        ),
    )


def _build_position_bucket_plan(
    segments: tuple[GdnSegmentSpec, ...],
    local_token_ranges: tuple[tuple[int, int, int], ...],
    *,
    sequence_length: int,
    device: torch.device | str,
    token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] | None = None,
) -> GdnSegmentBucketPlan:
    range_positions = {
        (start, end): position for start, end, position in local_token_ranges
    }
    starts: list[int] = []
    lengths: list[int] = []
    for segment in segments:
        token_start = _segment_token_start(segment, sequence_length)
        token_end = token_start + segment.length
        position_start = range_positions.get((token_start, token_end))
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
        return _build_bucket_plan(
            segments,
            lengths_cpu=lengths_cpu,
            row_indices_cpu=torch.zeros_like(position_indices_cpu),
            position_indices_cpu=position_indices_cpu,
            lengths_by_rank_cpu=_bucket_lengths_by_rank_cpu(
                segments,
                token_ranges_by_rank,
                sequence_length=sequence_length,
            ),
            device=device,
        )

    local_positions_by_segment: list[torch.Tensor] = []
    local_range_ends = tuple(token_end for _, token_end, _ in local_token_ranges)
    for segment in segments:
        positions = _local_positions_for_segment(
            segment,
            sequence_length=sequence_length,
            local_token_ranges=local_token_ranges,
            local_range_ends=local_range_ends,
        )
        if not int(positions.numel()):
            raise ValueError(
                "planned GDN bucket contains a segment with no local tokens; "
                f"family={segment.family_index} kind={segment.kind}"
            )
        local_positions_by_segment.append(positions)

    lengths_cpu = torch.tensor(
        [int(positions.numel()) for positions in local_positions_by_segment],
        dtype=torch.long,
    )
    max_length = int(lengths_cpu.max().item())
    position_indices_cpu = torch.zeros(max_length, len(segments), dtype=torch.long)
    for column, positions in enumerate(local_positions_by_segment):
        position_indices_cpu[: int(positions.numel()), column] = positions
    return _build_bucket_plan(
        segments,
        lengths_cpu=lengths_cpu,
        row_indices_cpu=torch.zeros_like(position_indices_cpu),
        position_indices_cpu=position_indices_cpu,
        lengths_by_rank_cpu=_bucket_lengths_by_rank_cpu(
            segments,
            token_ranges_by_rank,
            sequence_length=sequence_length,
        ),
        device=device,
    )


def _bucket_lengths_by_rank_cpu(
    segments: tuple[GdnSegmentSpec, ...],
    token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] | None,
    *,
    sequence_length: int,
) -> torch.Tensor | None:
    if token_ranges_by_rank is None:
        return None
    lengths_by_rank = []
    for rank_ranges_with_positions in token_ranges_by_rank:
        rank_ranges = tuple(
            (start, end) for start, end, _position in rank_ranges_with_positions
        )
        rank_lengths = []
        for segment in segments:
            start = _segment_token_start(segment, sequence_length)
            end = start + segment.length
            rank_lengths.append(
                sum(
                    max(0, min(end, range_end) - max(start, range_start))
                    for range_start, range_end in rank_ranges
                )
            )
        lengths_by_rank.append(rank_lengths)
    return torch.tensor(lengths_by_rank, dtype=torch.long)


def _move_planner_tensor(
    tensor: torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    target = torch.device(device)
    if target.type == "cpu":
        return tensor
    return tensor.to(device=target)


def _batch_segments_by_padded_work(
    segments: tuple[GdnSegmentSpec, ...],
    *,
    max_padding_ratio: float = 1.25,
    max_segments_per_batch: int = 128,
) -> tuple[tuple[GdnSegmentSpec, ...], ...]:
    if not segments:
        return ()
    ordered = sorted(
        segments, key=lambda segment: (segment.length, segment.family_index)
    )
    batches: list[list[GdnSegmentSpec]] = []
    current: list[GdnSegmentSpec] = []
    current_tokens = 0
    current_max = 0
    for segment in ordered:
        next_count = len(current) + 1
        next_tokens = current_tokens + segment.length
        next_max = max(current_max, segment.length)
        padded = next_max * next_count
        can_extend = not current or (
            next_count <= max_segments_per_batch
            and padded <= max_padding_ratio * next_tokens
        )
        if not can_extend:
            batches.append(current)
            current = []
            current_tokens = 0
            current_max = 0
        current.append(segment)
        current_tokens += segment.length
        current_max = max(current_max, segment.length)
    if current:
        batches.append(current)
    return tuple(tuple(batch) for batch in batches)


def _batch_tree_segments_by_padded_work(
    segments: tuple[GdnSegmentSpec, ...],
    tree_has_children: tuple[bool, ...],
    *,
    max_padding_ratio: float = 1.25,
    max_segments_per_batch: int = 128,
) -> tuple[tuple[GdnSegmentSpec, ...], ...]:
    stateful = tuple(
        segment for segment in segments if tree_has_children[segment.family_index]
    )
    stateless = tuple(
        segment for segment in segments if not tree_has_children[segment.family_index]
    )
    return (
        *_batch_segments_by_padded_work(
            stateful,
            max_padding_ratio=max_padding_ratio,
            max_segments_per_batch=max_segments_per_batch,
        ),
        *_batch_segments_by_padded_work(
            stateless,
            max_padding_ratio=max_padding_ratio,
            max_segments_per_batch=max_segments_per_batch,
        ),
    )


def _build_segment_bucket_plan(
    segments: tuple[GdnSegmentSpec, ...], *, device: torch.device | str
) -> GdnSegmentBucketPlan:
    lengths_cpu = torch.tensor(
        [segment.length for segment in segments], dtype=torch.long
    )
    max_length = int(lengths_cpu.max().item())
    starts_cpu = torch.tensor([segment.start for segment in segments], dtype=torch.long)
    rows_cpu = torch.tensor(
        [segment.row_index for segment in segments], dtype=torch.long
    )
    offsets_cpu = torch.arange(max_length, dtype=torch.long).unsqueeze(1)
    return _build_bucket_plan(
        segments,
        lengths_cpu=lengths_cpu,
        row_indices_cpu=rows_cpu.unsqueeze(0).expand(max_length, -1).contiguous(),
        position_indices_cpu=starts_cpu.unsqueeze(0) + offsets_cpu,
        device=device,
    )


def _build_bucket_plan(
    segments: tuple[GdnSegmentSpec, ...],
    *,
    lengths_cpu: torch.Tensor,
    row_indices_cpu: torch.Tensor,
    position_indices_cpu: torch.Tensor,
    device: torch.device | str,
    lengths_by_rank_cpu: torch.Tensor | None = None,
) -> GdnSegmentBucketPlan:
    max_length = int(lengths_cpu.max().item())
    offsets_cpu = torch.arange(max_length, dtype=torch.long).unsqueeze(1)
    real_mask_cpu = offsets_cpu < lengths_cpu.unsqueeze(0)
    cu_seqlens_cpu = torch.cat(
        [lengths_cpu.new_zeros(1), torch.cumsum(lengths_cpu, dim=0)]
    )
    family_indices_cpu = torch.tensor(
        [segment.family_index for segment in segments],
        dtype=torch.long,
    )
    return GdnSegmentBucketPlan(
        length=max_length,
        lengths=_move_planner_tensor(lengths_cpu, device),
        lengths_cpu=lengths_cpu,
        lengths_by_rank_cpu=lengths_by_rank_cpu,
        real_mask=_move_planner_tensor(real_mask_cpu, device),
        cu_seqlens=_move_planner_tensor(cu_seqlens_cpu, device),
        cu_seqlens_cpu=cu_seqlens_cpu,
        row_indices=_move_planner_tensor(row_indices_cpu, device),
        position_indices=_move_planner_tensor(position_indices_cpu, device),
        family_indices=_move_planner_tensor(family_indices_cpu, device),
        family_indices_cpu=family_indices_cpu,
        real_token_count_static=int(lengths_cpu.sum().item()),
    )


def _segment_token_start(segment: GdnSegmentSpec, sequence_length: int) -> int:
    return segment.row_index * sequence_length + segment.start


def _range_overlap_count(
    start: int,
    end: int,
    ranges: tuple[tuple[int, int], ...],
    range_ends: tuple[int, ...],
) -> int:
    count = 0
    range_index = bisect_left(range_ends, start + 1)
    for range_start, range_end in ranges[range_index:]:
        if range_start >= end:
            break
        count += min(end, range_end) - max(start, range_start)
    return count


def _range_overlaps(
    start: int,
    end: int,
    ranges: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    overlaps = [
        (max(start, range_start), min(end, range_end))
        for range_start, range_end in ranges
        if max(start, range_start) < min(end, range_end)
    ]
    overlaps.sort()
    return overlaps


def _local_positions_for_segment(
    segment: GdnSegmentSpec,
    *,
    sequence_length: int,
    local_token_ranges: tuple[tuple[int, int, int], ...],
    local_range_ends: tuple[int, ...],
) -> torch.Tensor:
    segment_start = _segment_token_start(segment, sequence_length)
    segment_end = segment_start + segment.length
    pieces = []
    range_index = bisect_left(local_range_ends, segment_start + 1)
    for token_start, token_end, position_start in local_token_ranges[range_index:]:
        if token_start >= segment_end:
            break
        overlap_start = max(segment_start, token_start)
        overlap_end = min(segment_end, token_end)
        if overlap_start >= overlap_end:
            continue
        pieces.append(
            torch.arange(
                position_start + overlap_start - token_start,
                position_start + overlap_end - token_start,
                dtype=torch.long,
            )
        )
    if not pieces:
        return torch.empty((0,), dtype=torch.long)
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces)


def _rank2_long_cpu(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 2 [batch, sequence], got {tensor.ndim}")
    if tensor.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    ):
        raise TypeError(f"{name} must contain integer ids, got dtype={tensor.dtype}")
    return tensor.detach().to(device="cpu", dtype=torch.long)
