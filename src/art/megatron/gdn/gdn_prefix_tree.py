from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, cast

import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.prefix_tree import parse_prefix_tree

GdnSegmentKind = Literal["prefix", "completion"]
# FLA's public chunk_gated_delta_rule hard-codes 64-token WY chunks.
FLA_CHUNK_SIZE = 64
# Fitted from Qwen3.5 35B/397B GDN lab points. Doubling state elements
# increases exposed recurrent runtime by about 1.7x, not 2x, because the
# kernels recover some parallelism at larger state shapes. Communication terms
# below still use exact bytes moved.
_RUNTIME_STATE_THROUGHPUT_EXPONENT = 0.75


def _dtype_bytes(dtype: Any) -> int:
    if dtype is None:
        return 2
    if isinstance(dtype, str):
        dtype_by_name = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype not in dtype_by_name:
            raise ValueError(f"unsupported GDN planner dtype {dtype!r}")
        dtype = dtype_by_name[dtype]
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    return torch.tensor([], dtype=dtype).element_size()


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
    """Parsed prefix-tree GDN execution metadata for a packed batch."""

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
class _ExplicitBucketColumn:
    row_index: int
    family_index: int
    parent_index: int
    positions: tuple[int, ...]
    output_mask: tuple[bool, ...]
    needs_final_state: bool

    @property
    def length(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class GdnPlannerConfig:
    """Runtime model for one packed-row GDN execution plan.

    The defaults reproduce the Qwen3.5-35B reference shape used to fit the
    GDN CP lab: hidden=2048, BF16 hidden exchange, 32 local value heads, and
    key/value dim 128. Production callers should construct this with
    ``from_model_shape`` so the same fitted hardware model scales by the actual
    GDN state size instead of acting as a per-model profile.
    """

    cp_chain_beam_width: int = 2
    cp_chain_beam_branch_factor: int = 4
    cp_chain_beam_candidate_limit: int = 16
    cp_chain_beam_max_steps: int = 4
    # Chain buckets add extra collectives and kernel shapes; require a
    # measurable runtime win before selecting them over local execution.
    cp_chain_min_runtime_delta_ms: float = 4.0
    runtime_hidden_bytes_per_token: int = 4096
    runtime_layout_exchange_count: int = 4
    # Global all-to-all token counts are priced against aggregate CP bandwidth.
    runtime_layout_bandwidth_bytes_per_ms: float = 448_000_000.0
    runtime_layout_collective_latency_ms: float = 0.0
    # Recurrent rates are fitted fwd+bwd exposed runtime, including the CP
    # recurrent communication work that is not captured by token counts alone.
    runtime_local_recurrent_tokens_per_ms: float = 1_500.0
    runtime_chain_recurrent_tokens_per_ms: float = 1_400.0
    runtime_local_bucket_launch_ms: float = 0.20
    runtime_chain_bucket_launch_ms: float = 0.20
    runtime_local_segment_launch_ms: float = 0.005
    runtime_cp_summary_bytes_per_segment: int = 4_194_304
    runtime_cp_summary_exchange_count_per_bucket: int = 8
    # Summary collectives move small state tensors and do not sustain the large
    # hidden-state all-to-all bandwidth used by the layout exchange term.
    runtime_cp_summary_bandwidth_bytes_per_ms: float = 80_000_000.0
    runtime_cp_summary_collective_latency_ms: float = 0.0
    runtime_cp_summary_compute_segments_per_ms: float = 320.0
    runtime_cp_suffix_scan_latency_ms: float = 2.0
    runtime_cp_suffix_scan_segments_per_ms: float = 15.0
    runtime_parent_state_bytes_per_exchange: int = 262_144
    runtime_parent_state_bandwidth_bytes_per_ms: float = 56_000_000.0
    runtime_parent_state_latency_ms: float = 0.0

    @classmethod
    def from_provider(
        cls,
        provider: Any,
        *,
        dtype_bytes: int | None = None,
        **overrides: Any,
    ) -> GdnPlannerConfig:
        return cls.from_model_shape(
            hidden_size=int(getattr(provider, "hidden_size")),
            tensor_model_parallel_size=int(
                getattr(provider, "tensor_model_parallel_size", 1) or 1
            ),
            linear_num_key_heads=int(getattr(provider, "linear_num_key_heads")),
            linear_num_value_heads=int(getattr(provider, "linear_num_value_heads")),
            linear_key_head_dim=int(getattr(provider, "linear_key_head_dim")),
            linear_value_head_dim=int(getattr(provider, "linear_value_head_dim")),
            linear_conv_kernel_dim=int(
                getattr(provider, "linear_conv_kernel_dim", 4) or 4
            ),
            dtype_bytes=(
                dtype_bytes
                if dtype_bytes is not None
                else _dtype_bytes(getattr(provider, "params_dtype", None))
            ),
            **overrides,
        )

    @classmethod
    def from_model_shape(
        cls,
        *,
        hidden_size: int,
        tensor_model_parallel_size: int,
        linear_num_key_heads: int,
        linear_num_value_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int = 4,
        dtype_bytes: int = 2,
        **overrides: Any,
    ) -> GdnPlannerConfig:
        """Build the fitted runtime model for an arbitrary GDN tensor shape."""

        tp_size = max(1, int(tensor_model_parallel_size))
        if linear_num_key_heads % tp_size != 0:
            raise ValueError(
                "linear_num_key_heads must be divisible by tensor parallel size, "
                f"got {linear_num_key_heads} and {tp_size}"
            )
        if linear_num_value_heads % tp_size != 0:
            raise ValueError(
                "linear_num_value_heads must be divisible by tensor parallel size, "
                f"got {linear_num_value_heads} and {tp_size}"
            )
        local_key_heads = int(linear_num_key_heads) // tp_size
        local_value_heads = int(linear_num_value_heads) // tp_size
        key_dim = int(linear_key_head_dim)
        value_dim = int(linear_value_head_dim)
        recurrent_state_elements = local_value_heads * key_dim * value_dim
        summary_state_elements = local_value_heads * key_dim * (value_dim + key_dim)
        ref_recurrent_elements = 32 * 128 * 128
        ref_summary_elements = 32 * 128 * (128 + 128)
        recurrent_scale = (
            ref_recurrent_elements / max(1, recurrent_state_elements)
        ) ** _RUNTIME_STATE_THROUGHPUT_EXPONENT
        summary_scale = (
            ref_summary_elements / max(1, summary_state_elements)
        ) ** _RUNTIME_STATE_THROUGHPUT_EXPONENT
        qkv_channels = 2 * local_key_heads * key_dim + local_value_heads * value_dim
        conv_state_bytes = (
            qkv_channels * max(0, int(linear_conv_kernel_dim) - 1) * int(dtype_bytes)
        )
        recurrent_state_bytes = recurrent_state_elements * 4
        ref_parent_state_bytes = (
            (2 * 16 * 128 + 32 * 128) * (4 - 1) * 2
        ) + ref_recurrent_elements * 4
        parent_state_bandwidth = 56_000_000.0 * ref_parent_state_bytes / 262_144.0
        config = cls(
            runtime_hidden_bytes_per_token=int(hidden_size) * int(dtype_bytes),
            runtime_local_recurrent_tokens_per_ms=1_500.0 * recurrent_scale,
            runtime_chain_recurrent_tokens_per_ms=1_400.0 * recurrent_scale,
            runtime_cp_summary_bytes_per_segment=summary_state_elements * 4,
            runtime_cp_summary_compute_segments_per_ms=320.0 * summary_scale,
            runtime_cp_suffix_scan_segments_per_ms=15.0 * summary_scale,
            runtime_parent_state_bytes_per_exchange=(
                conv_state_bytes + recurrent_state_bytes
            ),
            runtime_parent_state_bandwidth_bytes_per_ms=parent_state_bandwidth,
        )
        return (
            cast(GdnPlannerConfig, replace(config, **overrides))
            if overrides
            else config
        )


@dataclass(frozen=True)
class GdnRankExecutionPlan:
    """Rank-local planned execution metadata for prefix-tree GDN."""

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


GdnSegmentDecisionKey = tuple[int, int, int]


class _GdnChainSearchDecision(NamedTuple):
    chain_segment_keys: frozenset[GdnSegmentDecisionKey]
    score: float


class _GdnChainSearchMetadata(NamedTuple):
    depth_count: int
    children_by_node: tuple[tuple[int, ...], ...]
    root_indices: tuple[int, ...]
    subtree_token_counts: tuple[int, ...]
    subtree_indices_by_node: tuple[tuple[int, ...], ...]
    subtree_attention_counts: tuple[tuple[int, ...], ...]
    segment_keys: tuple[GdnSegmentDecisionKey, ...]
    segment_lengths: tuple[int, ...]
    segment_depths: tuple[int, ...]
    segment_attention_counts: tuple[tuple[int, ...], ...]


class _GdnLocalRuntimeEstimate(NamedTuple):
    rank_work: tuple[int, ...]
    rank_bucket_counts: tuple[int, ...]
    rank_segment_counts: tuple[int, ...]


class _GdnChainRuntimeEstimate(NamedTuple):
    rank_work: tuple[int, ...]
    bucket_count: int
    segment_count: int


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
    """Build rank-local tensor metadata from a parsed prefix-tree DAG.

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
    )
    attention_layout_index = _build_attention_layout_index_from_token_layout(
        source_layout
    )
    segment_attention_counts = _segment_attention_rank_counts(
        spec,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )
    chain_segment_keys = _select_chain_segment_keys(
        spec,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
        segment_attention_counts=segment_attention_counts,
        planner_config=planner_config,
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
    chain_nodes = [
        _segment_key(segment) in chain_segment_keys for segment in spec.tree_segments
    ]
    subtree_has_chained_node = list(chain_nodes)
    for node_index in reversed(range(len(spec.tree_segments))):
        subtree_has_chained_node[node_index] = subtree_has_chained_node[
            node_index
        ] or any(
            subtree_has_chained_node[child_index]
            for child_index in children_by_node[node_index]
        )
    target_rank_load = spec.real_token_count / max(1, cp_size)
    max_local_group_tokens = max(1, int(target_rank_load))

    def assign_tree(node_index: int) -> None:
        nonlocal cross_rank_token_count
        segment = spec.tree_segments[node_index]
        if chain_nodes[node_index]:
            chained_nodes[segment.family_index] = True
            chain_segments_by_depth[spec.tree_depths[segment.family_index]].append(
                segment
            )
            cross_rank_token_count += _append_chain_segment(
                gdn_ranges_by_rank,
                rank_loads,
                segment,
                spec,
                attention_layout_index=attention_layout_index,
            )
            for child_index in children_by_node[node_index]:
                assign_tree(child_index)
            return

        if (
            not subtree_has_chained_node[node_index]
            and subtree_token_counts[node_index] <= max_local_group_tokens
        ):
            assign_local_group(subtree_indices(node_index))
            return

        assign_local_group((node_index,))
        for child_index in children_by_node[node_index]:
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
    if cp_size == 1:
        tree_segment_buckets_by_depth = _build_chunk_aligned_cp1_tree_buckets(
            spec,
            tuple(tree_has_children),
            device=device,
            planner_config=planner_config,
        )
    else:
        tree_segment_buckets_by_depth = tuple(
            _build_tree_bucket_plans(
                tuple(segments_by_rank_depth[cp_rank][depth]),
                spec.tree_parent_indices,
                tuple(tree_has_children),
                local_token_ranges=local_token_ranges,
                sequence_length=spec.sequence_length,
                device=device,
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


def parse_gdn_prefix_tree_segments(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> GdnPackedExecutionSpec:
    """Parse ART packed prefix-tree metadata into generic GDN tree nodes."""

    groups = _rank2_long_cpu("group_ids", group_ids)
    parents = _rank2_long_cpu("parent_ids", parent_ids)
    if tuple(groups.shape) != tuple(parents.shape):
        raise ValueError(
            "group_ids and parent_ids must have the same shape, got "
            f"{tuple(groups.shape)} and {tuple(parents.shape)}"
        )

    batch_size, sequence_length = (int(groups.shape[0]), int(groups.shape[1]))
    rows = parse_prefix_tree(group_ids=groups, parent_ids=parents)
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
    if cp_size != 1:
        raise ValueError("GDN CP planning requires attention_token_layout_index")
    ranges_by_rank = (((0, int(spec.real_token_count), 0),),)
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
) -> bool:
    return segment.length >= cp_size and segment.length // FLA_CHUNK_SIZE >= cp_size


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
    return _best_search_owner(
        segment_length,
        on_rank_tokens,
        rank_loads,
        planner_config=planner_config,
    )


def _best_search_owner(
    segment_length: int,
    on_rank_tokens: tuple[int, ...],
    rank_loads: list[int],
    *,
    planner_config: GdnPlannerConfig,
) -> int:
    best: tuple[float, float, int, int, int] | None = None
    for rank, tokens in enumerate(on_rank_tokens):
        projected_loads = list(rank_loads)
        projected_loads[rank] += segment_length
        rank_runtime_ms = max(
            load / planner_config.runtime_local_recurrent_tokens_per_ms
            for load in projected_loads
        )
        cross_rank_tokens = segment_length - int(tokens)
        exchange_runtime_ms = _predict_layout_exchange_runtime_ms(
            cross_rank_tokens,
            planner_config,
        )
        candidate = (
            rank_runtime_ms + exchange_runtime_ms,
            exchange_runtime_ms,
            cross_rank_tokens,
            -int(tokens),
            rank,
        )
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return _least_loaded_rank(rank_loads)
    return best[-1]


def _select_chain_segment_keys(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    planner_config: GdnPlannerConfig,
) -> frozenset[GdnSegmentDecisionKey]:
    if cp_size <= 1:
        return frozenset()
    legal_chain_segments = tuple(
        segment
        for segment in spec.tree_segments
        if _can_chain_tree_segment(segment, cp_size=cp_size)
    )
    if not legal_chain_segments:
        return frozenset()
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]] = {}
    chain_cross_rank_tokens_by_key: dict[GdnSegmentDecisionKey, int] = {}
    for segment in legal_chain_segments:
        key = _segment_key(segment)
        (
            chain_rank_counts_by_key[key],
            chain_cross_rank_tokens_by_key[key],
        ) = _chain_segment_rank_counts_and_cross_rank_tokens(
            segment,
            spec,
            cp_size=cp_size,
            attention_layout_index=attention_layout_index,
        )

    search_metadata = _build_chain_search_metadata(
        spec,
        segment_attention_counts=segment_attention_counts,
    )
    score_cache: dict[frozenset[GdnSegmentDecisionKey], _GdnChainSearchDecision] = {}

    def decision_for(
        chain_segment_keys: frozenset[GdnSegmentDecisionKey],
    ) -> _GdnChainSearchDecision:
        cached = score_cache.get(chain_segment_keys)
        if cached is not None:
            return cached
        decision = _GdnChainSearchDecision(
            chain_segment_keys=chain_segment_keys,
            score=_score_chain_segment_keys(
                spec,
                cp_size=cp_size,
                search_metadata=search_metadata,
                chain_rank_counts_by_key=chain_rank_counts_by_key,
                chain_cross_rank_tokens_by_key=chain_cross_rank_tokens_by_key,
                chain_segment_keys=chain_segment_keys,
                planner_config=planner_config,
            ),
        )
        score_cache[chain_segment_keys] = decision
        return decision

    legal_chain_keys = frozenset(
        _segment_key(segment) for segment in legal_chain_segments
    )
    best = decision_for(frozenset())
    all_chain = decision_for(legal_chain_keys)
    if best.score - all_chain.score > planner_config.cp_chain_min_runtime_delta_ms:
        best = all_chain
    beam = _best_chain_search_decisions(
        (decision_for(frozenset()), all_chain),
        limit=planner_config.cp_chain_beam_width,
    )
    candidate_groups = _bounded_chain_candidate_groups(
        spec,
        legal_chain_segments,
        segment_attention_counts=segment_attention_counts,
        chain_rank_counts_by_key=chain_rank_counts_by_key,
        planner_config=planner_config,
    )
    stale_steps = 0
    for _ in range(max(0, planner_config.cp_chain_beam_max_steps)):
        if not candidate_groups:
            break
        expanded: dict[frozenset[GdnSegmentDecisionKey], _GdnChainSearchDecision] = {}
        for decision in beam:
            neighbors: list[_GdnChainSearchDecision] = []
            for segment_keys in _chain_beam_neighbor_groups(
                decision.chain_segment_keys,
                candidate_groups=candidate_groups,
                branch_factor=planner_config.cp_chain_beam_branch_factor,
            ):
                next_keys = (
                    decision.chain_segment_keys - segment_keys
                    if segment_keys.issubset(decision.chain_segment_keys)
                    else decision.chain_segment_keys | segment_keys
                )
                neighbors.append(decision_for(frozenset(next_keys)))
            for neighbor in _best_chain_search_decisions(
                neighbors,
                limit=planner_config.cp_chain_beam_branch_factor,
            ):
                expanded[neighbor.chain_segment_keys] = neighbor
        if not expanded:
            break
        beam = _best_chain_search_decisions(
            (*beam, *expanded.values()),
            limit=planner_config.cp_chain_beam_width,
        )
        step_best = beam[0]
        if best.score - step_best.score > planner_config.cp_chain_min_runtime_delta_ms:
            best = step_best
            stale_steps = 0
        else:
            stale_steps += 1
            if stale_steps >= 2:
                break
    return best.chain_segment_keys


def _build_chain_search_metadata(
    spec: GdnPackedExecutionSpec,
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
) -> _GdnChainSearchMetadata:
    children_by_node: list[list[int]] = [[] for _ in spec.tree_segments]
    root_indices: list[int] = []
    for node_index, parent_index in enumerate(spec.tree_parent_indices):
        if parent_index < 0:
            root_indices.append(node_index)
        else:
            children_by_node[parent_index].append(node_index)
    segment_keys = tuple(_segment_key(segment) for segment in spec.tree_segments)
    segment_lengths = tuple(segment.length for segment in spec.tree_segments)
    segment_depths = tuple(
        spec.tree_depths[segment.family_index] for segment in spec.tree_segments
    )
    segment_attention_counts_by_node = tuple(
        segment_attention_counts[key] for key in segment_keys
    )
    subtree_token_counts = list(segment_lengths)
    subtree_indices_by_node: list[tuple[int, ...]] = [
        (node_index,) for node_index in range(len(spec.tree_segments))
    ]
    subtree_attention_counts = [
        tuple(counts) for counts in segment_attention_counts_by_node
    ]
    for node_index in reversed(range(len(spec.tree_segments))):
        for child_index in children_by_node[node_index]:
            subtree_token_counts[node_index] += subtree_token_counts[child_index]
            subtree_indices_by_node[node_index] = (
                *subtree_indices_by_node[node_index],
                *subtree_indices_by_node[child_index],
            )
            subtree_attention_counts[node_index] = tuple(
                parent + child
                for parent, child in zip(
                    subtree_attention_counts[node_index],
                    subtree_attention_counts[child_index],
                    strict=True,
                )
            )
    return _GdnChainSearchMetadata(
        depth_count=max(spec.tree_depths, default=0) + 1,
        children_by_node=tuple(tuple(children) for children in children_by_node),
        root_indices=tuple(root_indices),
        subtree_token_counts=tuple(subtree_token_counts),
        subtree_indices_by_node=tuple(subtree_indices_by_node),
        subtree_attention_counts=tuple(subtree_attention_counts),
        segment_keys=segment_keys,
        segment_lengths=segment_lengths,
        segment_depths=segment_depths,
        segment_attention_counts=segment_attention_counts_by_node,
    )


def _score_chain_segment_keys(
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    search_metadata: _GdnChainSearchMetadata,
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
    chain_cross_rank_tokens_by_key: dict[GdnSegmentDecisionKey, int],
    chain_segment_keys: frozenset[GdnSegmentDecisionKey],
    planner_config: GdnPlannerConfig,
) -> float:
    chain_nodes = [key in chain_segment_keys for key in search_metadata.segment_keys]
    subtree_has_chained_node = list(chain_nodes)
    for node_index in reversed(range(len(spec.tree_segments))):
        subtree_has_chained_node[node_index] = subtree_has_chained_node[
            node_index
        ] or any(
            subtree_has_chained_node[child_index]
            for child_index in search_metadata.children_by_node[node_index]
        )

    rank_loads = [0] * cp_size
    owner_by_node = [-1] * len(spec.tree_segments)
    local_lengths_by_rank_depth: list[list[list[int]]] = [
        [[] for _ in range(search_metadata.depth_count)] for _ in range(cp_size)
    ]
    chain_rank_counts_by_depth: list[list[tuple[int, ...]]] = [
        [] for _ in range(search_metadata.depth_count)
    ]
    cross_rank_token_count = 0
    target_rank_load = spec.real_token_count / max(1, cp_size)
    max_local_group_tokens = max(1, int(target_rank_load))

    def add_local_group(
        node_indices: tuple[int, ...],
        segment_length: int,
        on_rank_tokens: tuple[int, ...],
    ) -> None:
        nonlocal cross_rank_token_count
        owner = _best_search_owner(
            segment_length,
            on_rank_tokens,
            rank_loads,
            planner_config=planner_config,
        )
        rank_loads[owner] += segment_length
        cross_rank_token_count += segment_length - on_rank_tokens[owner]
        for node_index in node_indices:
            owner_by_node[node_index] = owner
            local_lengths_by_rank_depth[owner][
                search_metadata.segment_depths[node_index]
            ].append(search_metadata.segment_lengths[node_index])

    def add_chain_segment(segment: GdnSegmentSpec) -> None:
        nonlocal cross_rank_token_count
        key = _segment_key(segment)
        chain_rank_counts_by_depth[spec.tree_depths[segment.family_index]].append(
            chain_rank_counts_by_key[key]
        )
        cross_rank_token_count += chain_cross_rank_tokens_by_key[key]
        for rank, token_count in enumerate(chain_rank_counts_by_key[key]):
            rank_loads[rank] += token_count

    def assign_tree(node_index: int) -> None:
        segment = spec.tree_segments[node_index]
        if chain_nodes[node_index]:
            add_chain_segment(segment)
            for child_index in search_metadata.children_by_node[node_index]:
                assign_tree(child_index)
            return
        if (
            not subtree_has_chained_node[node_index]
            and search_metadata.subtree_token_counts[node_index]
            <= max_local_group_tokens
        ):
            add_local_group(
                search_metadata.subtree_indices_by_node[node_index],
                search_metadata.subtree_token_counts[node_index],
                search_metadata.subtree_attention_counts[node_index],
            )
            return
        add_local_group(
            (node_index,),
            search_metadata.segment_lengths[node_index],
            search_metadata.segment_attention_counts[node_index],
        )
        for child_index in search_metadata.children_by_node[node_index]:
            assign_tree(child_index)

    for root_index in search_metadata.root_indices:
        assign_tree(root_index)

    local_runtime = _estimate_local_runtime_from_lengths(
        tuple(
            tuple(tuple(depth_segments) for depth_segments in rank_segments)
            for rank_segments in local_lengths_by_rank_depth
        )
    )
    chain_runtime = _estimate_chain_runtime_from_counts(
        tuple(tuple(depth_counts) for depth_counts in chain_rank_counts_by_depth),
        cp_size=cp_size,
    )
    parent_state_exchange_count = _count_parent_state_exchanges(
        spec,
        owner_by_node=tuple(owner_by_node),
        chain_nodes=tuple(chain_nodes),
    )
    return _predict_gdn_plan_runtime_ms(
        local_runtime,
        chain_runtime,
        cross_rank_token_count=cross_rank_token_count,
        parent_state_exchange_count=parent_state_exchange_count,
        planner_config=planner_config,
    )


def _add_local_search_load(
    rank_loads: list[int],
    owner: int,
    segment: GdnSegmentSpec,
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
) -> int:
    rank_loads[owner] += segment.length
    return segment.length - segment_attention_counts[_segment_key(segment)][owner]


def _estimate_local_rank_kernel_work(
    local_segments_by_rank_depth: tuple[tuple[tuple[GdnSegmentSpec, ...], ...], ...],
) -> tuple[tuple[int, ...], int, int]:
    estimate = _estimate_local_runtime_from_lengths(
        tuple(
            tuple(
                tuple(segment.length for segment in segments)
                for segments in rank_segments
            )
            for rank_segments in local_segments_by_rank_depth
        )
    )
    return (
        estimate.rank_work,
        max(estimate.rank_bucket_counts, default=0),
        max(estimate.rank_segment_counts, default=0),
    )


def _estimate_local_rank_kernel_work_from_lengths(
    local_lengths_by_rank_depth: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[tuple[int, ...], int, int]:
    estimate = _estimate_local_runtime_from_lengths(local_lengths_by_rank_depth)
    return (
        estimate.rank_work,
        max(estimate.rank_bucket_counts, default=0),
        max(estimate.rank_segment_counts, default=0),
    )


def _estimate_local_runtime_from_lengths(
    local_lengths_by_rank_depth: tuple[tuple[tuple[int, ...], ...], ...],
) -> _GdnLocalRuntimeEstimate:
    rank_work: list[int] = []
    rank_bucket_counts: list[int] = []
    rank_segment_counts: list[int] = []
    for lengths_by_depth in local_lengths_by_rank_depth:
        work = 0
        bucket_count = 0
        segment_count = 0
        for lengths in lengths_by_depth:
            bucket_work, depth_bucket_count = _padded_work_from_lengths(lengths)
            work += bucket_work
            bucket_count += depth_bucket_count
            segment_count += len(lengths)
        rank_work.append(work)
        rank_bucket_counts.append(bucket_count)
        rank_segment_counts.append(segment_count)
    return _GdnLocalRuntimeEstimate(
        rank_work=tuple(rank_work),
        rank_bucket_counts=tuple(rank_bucket_counts),
        rank_segment_counts=tuple(rank_segment_counts),
    )


def _estimate_chain_rank_kernel_work(
    chain_segments_by_depth: tuple[tuple[GdnSegmentSpec, ...], ...],
    *,
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
    cp_size: int,
) -> tuple[tuple[int, ...], int]:
    estimate = _estimate_chain_runtime_from_counts(
        tuple(
            tuple(
                chain_rank_counts_by_key[_segment_key(segment)] for segment in segments
            )
            for segments in chain_segments_by_depth
        ),
        cp_size=cp_size,
    )
    return estimate.rank_work, estimate.bucket_count


def _estimate_chain_rank_kernel_work_from_counts(
    chain_rank_counts_by_depth: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    cp_size: int,
) -> tuple[tuple[int, ...], int]:
    estimate = _estimate_chain_runtime_from_counts(
        chain_rank_counts_by_depth,
        cp_size=cp_size,
    )
    return estimate.rank_work, estimate.bucket_count


def _estimate_chain_runtime_from_counts(
    chain_rank_counts_by_depth: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    cp_size: int,
) -> _GdnChainRuntimeEstimate:
    rank_work = [0] * cp_size
    bucket_count = 0
    segment_count = 0
    for rank_counts_by_segment in chain_rank_counts_by_depth:
        if not rank_counts_by_segment:
            continue
        bucket_count += 1
        segment_count += len(rank_counts_by_segment)
        for rank in range(cp_size):
            lengths = tuple(counts[rank] for counts in rank_counts_by_segment)
            rank_work[rank] += max(lengths, default=0) * len(lengths)
    return _GdnChainRuntimeEstimate(
        rank_work=tuple(rank_work),
        bucket_count=bucket_count,
        segment_count=segment_count,
    )


def _predict_gdn_plan_runtime_ms(
    local_runtime: _GdnLocalRuntimeEstimate,
    chain_runtime: _GdnChainRuntimeEstimate,
    *,
    cross_rank_token_count: int,
    parent_state_exchange_count: int,
    planner_config: GdnPlannerConfig,
) -> float:
    """Predict exposed fwd+bwd runtime for a GDN CP plan in milliseconds.

    This is a runtime cost model fitted from GDN CP lab measurements. It models
    the critical per-rank recurrent path separately from exposed communication
    and chain-summary work, so chain/local decisions compare predicted wall time
    rather than abstract balance or token-count penalties.
    """

    rank_runtime_ms = 0.0
    rank_count = max(
        len(local_runtime.rank_work),
        len(chain_runtime.rank_work),
    )
    for rank in range(rank_count):
        local_work = (
            local_runtime.rank_work[rank] if rank < len(local_runtime.rank_work) else 0
        )
        local_bucket_count = (
            local_runtime.rank_bucket_counts[rank]
            if rank < len(local_runtime.rank_bucket_counts)
            else 0
        )
        local_segment_count = (
            local_runtime.rank_segment_counts[rank]
            if rank < len(local_runtime.rank_segment_counts)
            else 0
        )
        chain_work = (
            chain_runtime.rank_work[rank] if rank < len(chain_runtime.rank_work) else 0
        )
        rank_runtime_ms = max(
            rank_runtime_ms,
            _predict_local_rank_runtime_ms(
                local_work,
                bucket_count=local_bucket_count,
                segment_count=local_segment_count,
                planner_config=planner_config,
            )
            + chain_work / planner_config.runtime_chain_recurrent_tokens_per_ms,
        )
    return (
        rank_runtime_ms
        + _predict_chain_exposed_runtime_ms(
            chain_runtime,
            planner_config=planner_config,
        )
        + _predict_layout_exchange_runtime_ms(
            cross_rank_token_count,
            planner_config,
        )
        + _predict_parent_state_exchange_runtime_ms(
            parent_state_exchange_count,
            planner_config,
        )
    )


def _predict_local_rank_runtime_ms(
    recurrent_work: int,
    *,
    bucket_count: int,
    segment_count: int,
    planner_config: GdnPlannerConfig,
) -> float:
    return (
        recurrent_work / planner_config.runtime_local_recurrent_tokens_per_ms
        + bucket_count * planner_config.runtime_local_bucket_launch_ms
        + segment_count * planner_config.runtime_local_segment_launch_ms
    )


def _predict_chain_exposed_runtime_ms(
    chain_runtime: _GdnChainRuntimeEstimate,
    *,
    planner_config: GdnPlannerConfig,
) -> float:
    if chain_runtime.bucket_count == 0:
        return 0.0
    summary_bytes = (
        chain_runtime.segment_count
        * planner_config.runtime_cp_summary_bytes_per_segment
        * planner_config.runtime_cp_summary_exchange_count_per_bucket
    )
    summary_exchange_ms = (
        summary_bytes / planner_config.runtime_cp_summary_bandwidth_bytes_per_ms
        + chain_runtime.bucket_count
        * planner_config.runtime_cp_summary_exchange_count_per_bucket
        * planner_config.runtime_cp_summary_collective_latency_ms
    )
    return (
        chain_runtime.bucket_count * planner_config.runtime_chain_bucket_launch_ms
        + summary_exchange_ms
        + chain_runtime.segment_count
        / planner_config.runtime_cp_summary_compute_segments_per_ms
        + chain_runtime.bucket_count * planner_config.runtime_cp_suffix_scan_latency_ms
        + chain_runtime.segment_count
        / planner_config.runtime_cp_suffix_scan_segments_per_ms
    )


def _predict_layout_exchange_runtime_ms(
    cross_rank_token_count: int,
    planner_config: GdnPlannerConfig,
) -> float:
    if cross_rank_token_count <= 0:
        return 0.0
    bytes_moved = (
        cross_rank_token_count
        * planner_config.runtime_hidden_bytes_per_token
        * planner_config.runtime_layout_exchange_count
    )
    return (
        bytes_moved / planner_config.runtime_layout_bandwidth_bytes_per_ms
        + planner_config.runtime_layout_exchange_count
        * planner_config.runtime_layout_collective_latency_ms
    )


def _predict_parent_state_exchange_runtime_ms(
    exchange_count: int,
    planner_config: GdnPlannerConfig,
) -> float:
    if exchange_count <= 0:
        return 0.0
    return (
        exchange_count
        * planner_config.runtime_parent_state_bytes_per_exchange
        / planner_config.runtime_parent_state_bandwidth_bytes_per_ms
        + exchange_count * planner_config.runtime_parent_state_latency_ms
    )


def _padded_work_from_lengths(lengths: tuple[int, ...]) -> tuple[int, int]:
    real_lengths = tuple(length for length in lengths if length > 0)
    if not real_lengths:
        return 0, 0
    return max(real_lengths) * len(real_lengths), 1


def _count_parent_state_exchanges(
    spec: GdnPackedExecutionSpec,
    *,
    owner_by_node: tuple[int, ...],
    chain_nodes: tuple[bool, ...],
) -> int:
    exchange_count = 0
    for child_index, parent_index in enumerate(spec.tree_parent_indices):
        if parent_index < 0 or chain_nodes[parent_index]:
            continue
        parent_owner = owner_by_node[parent_index]
        if parent_owner < 0:
            continue
        if chain_nodes[child_index] or owner_by_node[child_index] != parent_owner:
            exchange_count += 1
    return exchange_count


def _chain_beam_neighbor_groups(
    chain_segment_keys: frozenset[GdnSegmentDecisionKey],
    *,
    candidate_groups: tuple[frozenset[GdnSegmentDecisionKey], ...],
    branch_factor: int,
) -> tuple[frozenset[GdnSegmentDecisionKey], ...]:
    selected: list[frozenset[GdnSegmentDecisionKey]] = []
    for group in candidate_groups:
        if group and not group.issubset(chain_segment_keys):
            selected.append(group)
            if len(selected) >= branch_factor:
                return tuple(selected)
    for group in reversed(candidate_groups):
        if group and group.intersection(chain_segment_keys) and group not in selected:
            selected.append(group)
            if len(selected) >= branch_factor:
                break
    return tuple(selected)


def _best_chain_search_decisions(
    decisions: Any,
    *,
    limit: int,
) -> tuple[_GdnChainSearchDecision, ...]:
    return tuple(
        sorted(
            decisions,
            key=lambda decision: (
                decision.score,
                len(decision.chain_segment_keys),
                tuple(sorted(decision.chain_segment_keys)),
            ),
        )[: max(1, limit)]
    )


def _bounded_chain_candidate_groups(
    spec: GdnPackedExecutionSpec,
    legal_chain_segments: tuple[GdnSegmentSpec, ...],
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
    planner_config: GdnPlannerConfig,
) -> tuple[frozenset[GdnSegmentDecisionKey], ...]:
    legal_key_set = frozenset(_segment_key(segment) for segment in legal_chain_segments)
    if not legal_key_set:
        return ()
    root_keys = frozenset(
        _segment_key(segment)
        for segment in legal_chain_segments
        if spec.tree_parent_indices[segment.family_index] < 0
    )
    nonroot_keys = legal_key_set - root_keys
    groups: list[frozenset[GdnSegmentDecisionKey]] = []
    for group in (legal_key_set, root_keys, nonroot_keys):
        if group and group not in groups:
            groups.append(group)
    for group in _ranked_chain_beam_groups(
        spec,
        legal_chain_segments,
        segment_attention_counts=segment_attention_counts,
        chain_rank_counts_by_key=chain_rank_counts_by_key,
    ):
        if group and group not in groups:
            groups.append(group)
        if len(groups) >= planner_config.cp_chain_beam_candidate_limit:
            break
    return tuple(groups[: max(1, planner_config.cp_chain_beam_candidate_limit)])


def _ranked_chain_beam_groups(
    spec: GdnPackedExecutionSpec,
    legal_chain_segments: tuple[GdnSegmentSpec, ...],
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
) -> tuple[frozenset[GdnSegmentDecisionKey], ...]:
    priority_by_key = {
        _segment_key(segment): _chain_beam_segment_priority(
            segment,
            segment_attention_counts=segment_attention_counts,
            chain_rank_counts_by_key=chain_rank_counts_by_key,
        )
        for segment in legal_chain_segments
    }
    groups: set[frozenset[GdnSegmentDecisionKey]] = {
        frozenset((key,)) for key in priority_by_key
    }
    children_by_parent: dict[int, list[GdnSegmentDecisionKey]] = {}
    for segment in legal_chain_segments:
        parent_index = spec.tree_parent_indices[segment.family_index]
        if parent_index >= 0:
            children_by_parent.setdefault(parent_index, []).append(
                _segment_key(segment)
            )
    for child_keys in children_by_parent.values():
        if len(child_keys) > 1:
            groups.add(frozenset(child_keys))
    return tuple(
        sorted(
            groups,
            key=lambda group: _chain_beam_group_priority(
                group,
                priority_by_key=priority_by_key,
            ),
            reverse=True,
        )
    )


def _chain_beam_group_priority(
    group: frozenset[GdnSegmentDecisionKey],
    *,
    priority_by_key: dict[GdnSegmentDecisionKey, tuple[int, int, int, int]],
) -> tuple[int, int, int, int, int]:
    priorities = tuple(priority_by_key[key] for key in group)
    return (
        sum(priority[0] for priority in priorities),
        sum(priority[1] for priority in priorities),
        max((priority[2] for priority in priorities), default=0),
        sum(priority[3] for priority in priorities),
        len(group),
    )


def _chain_beam_segment_priority(
    segment: GdnSegmentSpec,
    *,
    segment_attention_counts: dict[tuple[int, int, int], tuple[int, ...]],
    chain_rank_counts_by_key: dict[GdnSegmentDecisionKey, tuple[int, ...]],
) -> tuple[int, int, int, int]:
    key = _segment_key(segment)
    chain_max_load = max(chain_rank_counts_by_key[key], default=0)
    best_attention_locality = max(segment_attention_counts[key], default=0)
    chain_load_relief = segment.length - chain_max_load
    minimum_local_exchange = segment.length - best_attention_locality
    return (
        chain_load_relief,
        segment.length,
        best_attention_locality,
        -minimum_local_exchange,
    )


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
        depth = spec.tree_depths[child_index]
        if source_rank < 0:
            raise ValueError(
                "tree state exchange requires every local parent to have an owner"
            )
        if chained_nodes[child_index]:
            for dest_rank in range(cp_size):
                if source_rank == dest_rank:
                    continue
                families_by_depth_pair[depth].setdefault(
                    (source_rank, dest_rank), set()
                ).add(parent_index)
            continue
        dest_rank = owner_by_node[child_index]
        if dest_rank < 0:
            raise ValueError(
                "tree state exchange requires every local child to have an owner"
            )
        if source_rank != dest_rank:
            families_by_depth_pair[depth].setdefault(
                (source_rank, dest_rank), set()
            ).add(parent_index)

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


def _append_chain_segment(
    gdn_ranges_by_rank: list[list[tuple[int, int, int]]],
    rank_loads: list[int],
    segment: GdnSegmentSpec,
    spec: GdnPackedExecutionSpec,
    *,
    attention_layout_index: _AttentionLayoutIndex | None = None,
) -> int:
    rank_ranges, cross_rank_tokens = _chain_segment_rank_ranges_and_cross_rank_tokens(
        segment,
        spec,
        cp_size=len(gdn_ranges_by_rank),
        attention_layout_index=attention_layout_index,
    )
    for rank, (shard_start, shard_end) in enumerate(rank_ranges):
        shard_length = shard_end - shard_start
        if shard_length <= 0:
            raise ValueError(
                "CP chain planning requires non-empty shards; "
                f"segment={segment.kind}:{segment.family_index} "
                f"length={segment.length} cp_size={len(gdn_ranges_by_rank)}"
            )
        position_start = rank_loads[rank]
        gdn_ranges_by_rank[rank].append((shard_start, shard_end, position_start))
        rank_loads[rank] += shard_length
    return cross_rank_tokens


def _chain_segment_rank_counts_and_cross_rank_tokens(
    segment: GdnSegmentSpec,
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex | None,
) -> tuple[tuple[int, ...], int]:
    rank_ranges, cross_rank_tokens = _chain_segment_rank_ranges_and_cross_rank_tokens(
        segment,
        spec,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )
    return tuple(end - start for start, end in rank_ranges), cross_rank_tokens


def _chain_segment_rank_ranges_and_cross_rank_tokens(
    segment: GdnSegmentSpec,
    spec: GdnPackedExecutionSpec,
    *,
    cp_size: int,
    attention_layout_index: _AttentionLayoutIndex | None,
) -> tuple[tuple[tuple[int, int], ...], int]:
    token_start = _segment_token_start(segment, spec.sequence_length)
    attention_shards = _attention_contiguous_chain_shards(
        token_start,
        segment.length,
        cp_size=cp_size,
        attention_layout_index=attention_layout_index,
    )
    if attention_shards is not None:
        return tuple((shard.start, shard.stop) for shard in attention_shards), 0
    shard_lengths = _fla_aligned_chain_shard_lengths(segment.length, cp_size=cp_size)
    rank_ranges: list[tuple[int, int]] = []
    cross_rank_tokens = 0
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
        shard_end = shard_start + shard_length
        rank_ranges.append((shard_start, shard_end))
        if attention_layout_index is not None:
            cross_rank_tokens += shard_length - _range_overlap_count(
                shard_start,
                shard_end,
                attention_layout_index.token_ranges_by_rank[rank],
                attention_layout_index.token_range_ends_by_rank[rank],
            )
        start = end
    return tuple(rank_ranges), cross_rank_tokens


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
    token_ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...] | None = None,
    split_by_final_state: bool = True,
) -> tuple[GdnSegmentBucketPlan, ...]:
    segment_buckets = (
        _batch_tree_segments_by_padded_work(
            segments,
            tree_has_children,
        )
        if split_by_final_state
        else _batch_segments_by_padded_work(segments)
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


def _build_chunk_aligned_cp1_tree_buckets(
    spec: GdnPackedExecutionSpec,
    tree_has_children: tuple[bool, ...],
    *,
    device: torch.device | str,
    planner_config: GdnPlannerConfig,
) -> tuple[tuple[GdnSegmentBucketPlan, ...], ...]:
    depth_count = max(spec.tree_depths, default=0) + 1
    children_by_node: list[list[int]] = [[] for _ in spec.tree_segments]
    for node_index, parent_index in enumerate(spec.tree_parent_indices):
        if parent_index >= 0:
            children_by_node[parent_index].append(node_index)

    regular_by_depth: list[list[GdnSegmentSpec]] = [[] for _ in range(depth_count)]
    boundary_by_depth: list[list[GdnSegmentSpec]] = [[] for _ in range(depth_count)]
    child_columns_by_depth: list[list[_ExplicitBucketColumn]] = [
        [] for _ in range(depth_count)
    ]

    for node_index, segment in enumerate(spec.tree_segments):
        depth = spec.tree_depths[node_index]
        parent_index = spec.tree_parent_indices[node_index]
        if parent_index >= 0:
            continue
        if not tree_has_children[node_index]:
            regular_by_depth[depth].append(segment)
            continue
        boundary_end = _prefix_chunk_boundary_end(segment)
        if boundary_end > segment.start:
            boundary_by_depth[depth].append(
                replace(segment, start=segment.start, end=boundary_end)
            )

    for parent_index, child_indices in enumerate(children_by_node):
        if not child_indices:
            continue
        parent = spec.tree_segments[parent_index]
        parent_depth = spec.tree_depths[parent_index]
        child_depth = min(parent_depth + 1, depth_count - 1)
        parent_tree_parent = spec.tree_parent_indices[parent_index]
        if parent_tree_parent < 0:
            boundary_end = _prefix_chunk_boundary_end(parent)
            tail_positions = tuple(range(boundary_end, parent.end))
            explicit_parent = parent.family_index if boundary_end > parent.start else -1
        else:
            tail_positions = ()
            explicit_parent = parent.family_index
        for child_offset, child_index in enumerate(child_indices):
            child = spec.tree_segments[child_index]
            child_positions = tail_positions + tuple(range(child.start, child.end))
            child_columns_by_depth[child_depth].append(
                _ExplicitBucketColumn(
                    row_index=child.row_index,
                    family_index=child.family_index,
                    parent_index=explicit_parent,
                    positions=child_positions,
                    output_mask=(
                        ((child_offset == 0),) * len(tail_positions)
                        + (True,) * child.length
                    ),
                    needs_final_state=tree_has_children[child.family_index],
                )
            )

    return tuple(
        (
            *_build_tree_bucket_plans(
                tuple(boundary_by_depth[depth]),
                spec.tree_parent_indices,
                tree_has_children,
                local_token_ranges=None,
                sequence_length=spec.sequence_length,
                device=device,
            ),
            *_build_tree_bucket_plans(
                tuple(regular_by_depth[depth]),
                spec.tree_parent_indices,
                tree_has_children,
                local_token_ranges=None,
                sequence_length=spec.sequence_length,
                device=device,
            ),
            *_build_explicit_bucket_plans(
                tuple(child_columns_by_depth[depth]),
                device=device,
            ),
        )
        for depth in range(depth_count)
    )


def _build_explicit_bucket_plans(
    columns: tuple[_ExplicitBucketColumn, ...],
    *,
    device: torch.device | str,
) -> tuple[GdnSegmentBucketPlan, ...]:
    return tuple(
        _build_explicit_bucket_plan(
            batch,
            needs_final_state=any(column.needs_final_state for column in batch),
            device=device,
        )
        for batch in _batch_explicit_columns(
            columns,
        )
    )


def _batch_explicit_columns(
    columns: tuple[_ExplicitBucketColumn, ...],
) -> tuple[tuple[_ExplicitBucketColumn, ...], ...]:
    if not columns:
        return ()
    return (columns,)


def _build_explicit_bucket_plan(
    columns: tuple[_ExplicitBucketColumn, ...],
    *,
    needs_final_state: bool,
    device: torch.device | str,
) -> GdnSegmentBucketPlan:
    lengths_cpu = torch.tensor([column.length for column in columns], dtype=torch.long)
    max_length = int(lengths_cpu.max().item())
    row_indices_cpu = torch.zeros(max_length, len(columns), dtype=torch.long)
    position_indices_cpu = torch.zeros(max_length, len(columns), dtype=torch.long)
    output_mask_cpu = torch.zeros(max_length, len(columns), dtype=torch.bool)
    for column_index, column in enumerate(columns):
        length = column.length
        row_indices_cpu[:length, column_index] = column.row_index
        position_indices_cpu[:length, column_index] = torch.tensor(
            column.positions, dtype=torch.long
        )
        output_mask_cpu[:length, column_index] = torch.tensor(
            column.output_mask, dtype=torch.bool
        )
    plan = _build_bucket_plan(
        tuple(
            GdnSegmentSpec(
                row_index=column.row_index,
                family_index=column.family_index,
                group_id=column.family_index,
                parent_id=column.parent_index,
                start=0,
                end=column.length,
                kind="completion",
            )
            for column in columns
        ),
        lengths_cpu=lengths_cpu,
        row_indices_cpu=row_indices_cpu,
        position_indices_cpu=position_indices_cpu,
        device=device,
    )
    parent_indices_cpu = torch.tensor(
        [column.parent_index for column in columns], dtype=torch.long
    )
    return replace(
        plan,
        parent_indices=_move_planner_tensor(parent_indices_cpu, device),
        parent_indices_cpu=parent_indices_cpu,
        output_mask=_move_planner_tensor(
            _pack_bucket_column_major(output_mask_cpu, lengths_cpu),
            device,
        ),
        needs_final_state=needs_final_state,
    )


def _prefix_chunk_boundary_end(segment: GdnSegmentSpec) -> int:
    aligned_length = (segment.length // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
    return segment.start + aligned_length


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
) -> tuple[tuple[GdnSegmentSpec, ...], ...]:
    if not segments:
        return ()
    ordered = sorted(
        segments, key=lambda segment: (segment.length, segment.family_index)
    )
    return (tuple(ordered),)


def _batch_tree_segments_by_padded_work(
    segments: tuple[GdnSegmentSpec, ...],
    tree_has_children: tuple[bool, ...],
) -> tuple[tuple[GdnSegmentSpec, ...], ...]:
    del tree_has_children
    return _batch_segments_by_padded_work(segments)


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
    if (
        int(row_indices_cpu.shape[0]) < max_length
        or int(position_indices_cpu.shape[0]) < max_length
    ):
        raise ValueError("bucket index tensors are shorter than max segment length")
    row_indices_cpu = _pack_bucket_column_major(row_indices_cpu, lengths_cpu)
    position_indices_cpu = _pack_bucket_column_major(position_indices_cpu, lengths_cpu)
    real_mask_cpu = torch.ones(int(lengths_cpu.sum().item()), dtype=torch.bool)
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


def _pack_bucket_column_major(
    values_cpu: torch.Tensor,
    lengths_cpu: torch.Tensor,
) -> torch.Tensor:
    pieces = [
        values_cpu[: int(length), column]
        for column, length in enumerate(lengths_cpu.tolist())
        if int(length) > 0
    ]
    if not pieces:
        return values_cpu.new_empty((0,))
    return torch.cat(pieces, dim=0).contiguous()


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
