from __future__ import annotations

from typing import Literal

import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn.gdn_prefix_tree import (
    FLA_CHUNK_SIZE,
    GdnGlobalExecutionDecision,
    GdnPlannerConfig,
    GdnRankExecutionPlan,
    build_gdn_global_execution_decision,
    materialize_gdn_rank_execution_plan,
    move_gdn_rank_execution_plan_to_device,
)

from .contract import LinearRecurrentContract
from .prefix_tree import RecurrentPackedExecutionSpec


class GdnRecurrentFamilyAdapter:
    family_key = "gated_delta_net"
    partition_kind: Literal["token_sharded_chain"] = "token_sharded_chain"
    global_decision_type = GdnGlobalExecutionDecision
    rank_plan_type = GdnRankExecutionPlan

    def validate_planning_inputs(
        self,
        contract: LinearRecurrentContract,
        planner_config: object | None,
    ) -> None:
        self._validate(contract, planner_config)

    def build_global_decision(
        self,
        spec: RecurrentPackedExecutionSpec,
        *,
        contract: LinearRecurrentContract,
        token_layout_index: TokenLayoutIndex,
        cp_size: int,
        planner_config: object | None,
    ) -> object:
        config = self._validate(contract, planner_config)
        if len(token_layout_index.token_counts_by_rank) != cp_size:
            raise ValueError("GDN token layout and CP size must match")
        return build_gdn_global_execution_decision(
            spec,
            cp_size=cp_size,
            attention_token_layout_index=token_layout_index,
            planner_config=config,
        )

    def validate_global_decision(
        self,
        spec: RecurrentPackedExecutionSpec,
        decision: object,
        *,
        contract: LinearRecurrentContract,
        token_layout_index: TokenLayoutIndex,
        cp_size: int,
        planner_config: object | None,
    ) -> None:
        self._validate(contract, planner_config)
        if not isinstance(decision, GdnGlobalExecutionDecision):
            raise TypeError("GDN adapter received an incompatible global decision")
        family_count = spec.family_count
        depth_count = max(spec.tree_depths, default=-1) + 1
        has_children = [False] * family_count
        for parent in spec.tree_parent_indices:
            if parent >= 0:
                has_children[parent] = True
        if (
            decision.cp_size != cp_size
            or decision.source_layout != token_layout_index
            or decision.depth_count != depth_count
            or len(decision.gdn_token_counts_by_rank) != cp_size
            or len(decision.owner_by_node) != family_count
            or len(decision.chained_nodes) != family_count
            or decision.tree_has_children != tuple(has_children)
            or len(decision.gdn_ranges_by_rank_by_position) != cp_size
            or len(decision.gdn_ranges_by_rank_by_source) != cp_size
            or len(decision.segments_by_rank_depth) != cp_size
            or len(decision.chain_segments_by_depth) != depth_count
            or decision.cross_rank_token_count < 0
        ):
            raise ValueError("GDN global decision metadata is inconsistent")
        local_families: dict[int, int] = {}
        chain_families: set[int] = set()
        for rank, depths in enumerate(decision.segments_by_rank_depth):
            if len(depths) != depth_count:
                raise ValueError("GDN global decision depth metadata is inconsistent")
            for depth, segments in enumerate(depths):
                for segment in segments:
                    family = segment.family_index
                    if (
                        family in local_families
                        or family in chain_families
                        or family < 0
                        or family >= family_count
                        or segment != spec.tree_segments[family]
                        or spec.tree_depths[family] != depth
                        or decision.owner_by_node[family] != rank
                        or decision.chained_nodes[family]
                    ):
                        raise ValueError("GDN local segment assignment is inconsistent")
                    local_families[family] = rank
        for depth, segments in enumerate(decision.chain_segments_by_depth):
            for segment in segments:
                family = segment.family_index
                if (
                    family in local_families
                    or family in chain_families
                    or family < 0
                    or family >= family_count
                    or segment != spec.tree_segments[family]
                    or spec.tree_depths[family] != depth
                    or decision.owner_by_node[family] != -1
                    or not decision.chained_nodes[family]
                ):
                    raise ValueError("GDN chain segment assignment is inconsistent")
                chain_families.add(family)
        if len(local_families) + len(chain_families) != family_count:
            raise ValueError("GDN global decision does not cover every tree segment")
        for rank, ranges in enumerate(decision.gdn_ranges_by_rank_by_position):
            count = _validate_ranges(ranges)
            if (
                count != decision.gdn_token_counts_by_rank[rank]
                or tuple(sorted(ranges)) != decision.gdn_ranges_by_rank_by_source[rank]
            ):
                raise ValueError("GDN global decision ranges are inconsistent")

    def build_rank_plan(
        self,
        spec: RecurrentPackedExecutionSpec,
        decision: object,
        *,
        contract: LinearRecurrentContract,
        cp_rank: int,
        planner_config: object | None,
    ) -> object:
        config = self._validate(contract, planner_config)
        if not isinstance(decision, GdnGlobalExecutionDecision):
            raise TypeError("GDN adapter received an incompatible global decision")
        return materialize_gdn_rank_execution_plan(
            spec,
            decision,
            device="cpu",
            cp_rank=cp_rank,
            planner_config=config,
        )

    def materialize_rank_plan(
        self,
        plan: object,
        *,
        device: torch.device | str,
    ) -> object:
        if not isinstance(plan, GdnRankExecutionPlan):
            raise TypeError("GDN adapter received an incompatible rank plan")
        return move_gdn_rank_execution_plan_to_device(plan, device)

    def validate_rank_plan(
        self,
        spec: RecurrentPackedExecutionSpec,
        plan: object,
        *,
        contract: LinearRecurrentContract,
        cp_rank: int,
        planner_config: object | None,
        device: torch.device | str,
    ) -> None:
        self._validate(contract, planner_config)
        if not isinstance(plan, GdnRankExecutionPlan):
            raise TypeError("GDN adapter received an incompatible rank plan")
        target = torch.device(device)
        depth_count = max(spec.tree_depths, default=-1) + 1
        expected_shape = (
            (spec.batch_size, spec.sequence_length)
            if plan.cp_size == 1
            else (1, plan.gdn_token_count)
        )
        if (
            plan.cp_rank != cp_rank
            or cp_rank < 0
            or cp_rank >= plan.cp_size
            or plan.packed_batch_size != spec.batch_size
            or plan.packed_sequence_length != spec.sequence_length
            or tuple(plan.real_token_mask.shape) != expected_shape
            or plan.real_token_mask.dtype != torch.bool
            or not _tensor_on_device(plan.real_token_mask, target)
            or plan.attention_token_count
            != _validate_ranges(plan.attention_token_ranges)
            or plan.gdn_token_count != _validate_ranges(plan.gdn_token_ranges)
            or len(plan.tree_segment_buckets_by_depth) != depth_count
            or len(plan.tree_chain_buckets_by_depth) != depth_count
            or len(plan.tree_state_exchanges_by_depth) != depth_count
        ):
            raise ValueError("GDN rank execution plan metadata is inconsistent")
        _validate_exchange_device(plan.attention_to_gdn, target)
        _validate_exchange_device(plan.gdn_to_attention, target)
        for depths in (
            plan.tree_segment_buckets_by_depth,
            plan.tree_chain_buckets_by_depth,
        ):
            for buckets in depths:
                for bucket in buckets:
                    _validate_bucket(bucket, spec, target)
        for exchange in plan.tree_state_exchanges_by_depth:
            if exchange is None:
                continue
            if any(
                family < 0 or family >= spec.family_count
                for family in (
                    *exchange.source_family_indices,
                    *exchange.dest_family_indices,
                )
            ):
                raise ValueError("GDN state exchange family metadata is inconsistent")
            _validate_exchange_device(exchange.exchange, target)
            _validate_exchange_device(exchange.reverse_exchange, target)

    def model_token_counts(
        self,
        decision: object,
        *,
        attention_token_counts: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not isinstance(decision, GdnGlobalExecutionDecision):
            raise TypeError("GDN adapter received an incompatible global decision")
        if len(attention_token_counts) != len(decision.gdn_token_counts_by_rank):
            raise ValueError("GDN and attention token-count ranks must match")
        return tuple(
            max(attention, recurrent)
            for attention, recurrent in zip(
                attention_token_counts,
                decision.gdn_token_counts_by_rank,
                strict=True,
            )
        )

    @staticmethod
    def _validate(
        contract: LinearRecurrentContract,
        planner_config: object | None,
    ) -> GdnPlannerConfig:
        if (
            contract.family_key != "gated_delta_net"
            or contract.contract_version != "1"
            or contract.partition_kind != "token_sharded_chain"
            or contract.local_kernel_implementation_id
            != "art.gdn.fla_prefix_tree.chunk64.v1"
            or contract.layout_compatibility_key != "art.gdn.hidden_token_layout.v1"
            or contract.activation != "silu"
        ):
            raise ValueError("GDN adapter received an incompatible contract")
        chain = contract.chain
        if contract.local_chunk_size != FLA_CHUNK_SIZE or (
            chain is None
            or chain.alignment != FLA_CHUNK_SIZE
            or chain.legality_key != "art.gdn.fla_chunk64.v1"
            or chain.summary_implementation_id != "art.gdn.native_cp_summary.v1"
        ):
            raise ValueError("GDN chain metadata is incompatible")
        if not isinstance(planner_config, GdnPlannerConfig):
            raise TypeError("GDN adapter requires GdnPlannerConfig")
        streams = contract.projected_streams
        states = contract.states
        if tuple(stream.name for stream in streams) != (
            "qkv",
            "gate",
            "beta",
            "alpha",
        ) or any(
            (stream.shard_axis, stream.shard_count, stream.replication)
            != ("head", None, "none")
            for stream in streams
        ):
            raise ValueError("GDN projected-stream metadata is incompatible")
        if tuple(state.name for state in states) != ("conv", "recurrent"):
            raise ValueError("GDN contract requires ordered conv and recurrent states")
        qkv, gate, beta, alpha = streams
        conv, recurrent = states
        if len(recurrent.shape) != 3:
            raise ValueError("GDN recurrent state must be [heads, key_dim, value_dim]")
        value_heads, key_dim, value_dim = recurrent.shape
        key_projection_width = qkv.width - gate.width
        if (
            beta.width != value_heads
            or alpha.width != value_heads
            or gate.width != value_heads * value_dim
            or key_projection_width <= 0
            or key_projection_width % (2 * key_dim)
            or conv.shape != (qkv.width, contract.convolution_width - 1)
            or recurrent.dtype != "float32"
        ):
            raise ValueError("GDN projected streams and state shapes are inconsistent")
        cost = chain.cost
        if (
            cost.summary_bytes_per_segment
            != planner_config.runtime_cp_summary_bytes_per_segment
            or cost.summary_exchange_count
            != planner_config.runtime_cp_summary_exchange_count_per_bucket
            or cost.summary_bandwidth_bytes_per_ms
            != planner_config.runtime_cp_summary_bandwidth_bytes_per_ms
            or cost.summary_compute_segments_per_ms
            != planner_config.runtime_cp_summary_compute_segments_per_ms
            or cost.suffix_scan_latency_ms
            != planner_config.runtime_cp_suffix_scan_latency_ms
            or cost.suffix_scan_segments_per_ms
            != planner_config.runtime_cp_suffix_scan_segments_per_ms
        ):
            raise ValueError("GDN contract cost metadata does not match planner config")
        return planner_config


def _validate_ranges(ranges: tuple[tuple[int, int, int], ...]) -> int:
    cursor = 0
    for start, end, position in ranges:
        if start < 0 or end <= start or position != cursor:
            raise ValueError("GDN token ranges are invalid")
        cursor += end - start
    return cursor


def _validate_bucket(
    bucket: object,
    spec: RecurrentPackedExecutionSpec,
    device: torch.device,
) -> None:
    dynamic_names = (
        "lengths",
        "real_mask",
        "cu_seqlens",
        "row_indices",
        "position_indices",
        "family_indices",
        "parent_indices",
        "output_mask",
    )
    cpu_names = (
        "lengths_cpu",
        "cu_seqlens_cpu",
        "lengths_by_rank_cpu",
        "family_indices_cpu",
        "parent_indices_cpu",
    )
    if any(
        isinstance(tensor := getattr(bucket, name, None), torch.Tensor)
        and not _tensor_on_device(tensor, device)
        for name in dynamic_names
    ) or any(
        isinstance(tensor := getattr(bucket, name, None), torch.Tensor)
        and tensor.device.type != "cpu"
        for name in cpu_names
    ):
        raise ValueError("GDN bucket tensors are on the wrong device")
    lengths_cpu = getattr(bucket, "lengths_cpu", None)
    cu_seqlens_cpu = getattr(bucket, "cu_seqlens_cpu", None)
    family_indices_cpu = getattr(bucket, "family_indices_cpu", None)
    parent_indices_cpu = getattr(bucket, "parent_indices_cpu", None)
    if (
        not isinstance(lengths_cpu, torch.Tensor)
        or not isinstance(cu_seqlens_cpu, torch.Tensor)
        or not isinstance(family_indices_cpu, torch.Tensor)
        or not isinstance(parent_indices_cpu, torch.Tensor)
    ):
        raise ValueError("GDN bucket is missing immutable CPU metadata")
    segment_count = int(getattr(bucket, "segment_count"))
    real_token_count = int(getattr(bucket, "real_token_count_static"))
    family_tuple = tuple(getattr(bucket, "family_indices_cpu_tuple"))
    parent_tuple = getattr(bucket, "parent_indices_cpu_tuple")
    if (
        int(getattr(bucket, "length")) <= 0
        or int(getattr(bucket, "padded_length")) != int(getattr(bucket, "length"))
        or int(lengths_cpu.numel()) != segment_count
        or int(family_indices_cpu.numel()) != segment_count
        or int(parent_indices_cpu.numel()) != segment_count
        or int(cu_seqlens_cpu.numel()) != segment_count + 1
        or int(lengths_cpu.sum().item()) != real_token_count
        or int(cu_seqlens_cpu[-1].item()) != real_token_count
        or family_tuple != tuple(int(value) for value in family_indices_cpu.tolist())
        or parent_tuple != tuple(int(value) for value in parent_indices_cpu.tolist())
        or any(family < 0 or family >= spec.family_count for family in family_tuple)
        or any(parent < -1 or parent >= spec.family_count for parent in parent_tuple)
        or any(
            getattr(bucket, name, None) is not None
            for name in (
                "dense_real_mask",
                "dense_token_indices",
                "flat_token_indices",
                "dense_row_indices",
                "dense_position_indices",
            )
        )
    ):
        raise ValueError("GDN bucket metadata is inconsistent")
    for name in ("row_indices", "position_indices", "real_mask"):
        tensor = getattr(bucket, name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() != real_token_count:
            raise ValueError("GDN bucket compact indices are inconsistent")


def _validate_exchange_device(exchange: object, device: torch.device) -> None:
    from art.megatron.gdn.layout import GdnCpExchangePlan

    if not isinstance(exchange, GdnCpExchangePlan):
        raise TypeError("GDN adapter received an incompatible exchange plan")
    for transfer in exchange.transfers:
        for tensor in (
            transfer.source_positions_tensor,
            transfer.dest_positions_tensor,
        ):
            if isinstance(tensor, torch.Tensor) and not _tensor_on_device(
                tensor, device
            ):
                raise ValueError("GDN exchange tensors are on the wrong device")


def _tensor_on_device(tensor: torch.Tensor, device: torch.device) -> bool:
    return tensor.device.type == device.type and (
        device.index is None or tensor.device.index == device.index
    )
