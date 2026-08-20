from __future__ import annotations

from hashlib import sha256
import json
from typing import Literal

import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.context_parallel.types import (
    HeadShardedRecurrentGlobalDecision,
    HeadShardedRecurrentRankExecutionPlan,
)
from art.megatron.recurrent.buckets import (
    RecurrentSegmentBucketPlan,
    build_recurrent_tree_bucket_plans,
    move_recurrent_segment_bucket_plans,
)
from art.megatron.recurrent.contract import (
    HeadShardedFullTreePlannerConfig,
    LinearRecurrentContract,
)
from art.megatron.recurrent.prefix_tree import RecurrentPackedExecutionSpec

from .exchange import (
    MambaHeadShardDevicePlan,
    MambaHeadShardExchangePlan,
    build_mamba_head_shard_exchange_plan,
    materialize_mamba_head_shard_exchange_plan,
)
from .operator import (
    MAMBA_FAMILY_KEY,
    MAMBA_KERNEL_ID,
    MAMBA_LAYOUT_KEY,
)


class Mamba2RecurrentFamilyAdapter:
    family_key = MAMBA_FAMILY_KEY
    partition_kind: Literal["head_sharded_full_tree"] = "head_sharded_full_tree"
    global_decision_type = HeadShardedRecurrentGlobalDecision
    rank_plan_type = HeadShardedRecurrentRankExecutionPlan

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
        if (
            cp_size < 1
            or cp_size != len(token_layout_index.ownership_ranges_by_rank)
            or cp_size != len(token_layout_index.token_counts_by_rank)
        ):
            raise ValueError("Mamba token layout and CP size must match")
        if sum(token_layout_index.token_counts_by_rank) != spec.real_token_count:
            raise ValueError("Mamba token layout and recurrent token count must match")
        heads, head_dim, groups, state_dim = self._geometry(contract, cp_size)
        token_positions_by_rank = _token_positions_by_rank(token_layout_index)
        exchange_plan = build_mamba_head_shard_exchange_plan(
            token_positions_by_rank,
            canonical_flat_token_positions=_canonical_flat_positions(spec),
            heads_local_tp=heads,
            head_dim=head_dim,
            groups_local_tp=groups,
            state_dim=state_dim,
        )
        buckets_by_depth = self._build_buckets(spec, contract, config)
        token_counts = token_layout_index.token_counts_by_rank
        return HeadShardedRecurrentGlobalDecision(
            cp_size=cp_size,
            exchange_plan=exchange_plan,
            tree_segment_buckets_by_depth=buckets_by_depth,
            external_token_counts_by_rank=token_counts,
            family_artifact_identity=_family_artifact_identity(
                contract,
                spec,
                exchange_plan,
                buckets_by_depth,
                token_counts,
                config,
            ),
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
        config = self._validate(contract, planner_config)
        if not isinstance(
            decision, HeadShardedRecurrentGlobalDecision
        ) or not isinstance(decision.exchange_plan, MambaHeadShardExchangePlan):
            raise TypeError("Mamba adapter received an incompatible global decision")
        heads, head_dim, groups, state_dim = self._geometry(contract, cp_size)
        token_positions_by_rank = _token_positions_by_rank(token_layout_index)
        exchange = decision.exchange_plan
        actual_identity = _family_artifact_identity(
            contract,
            spec,
            decision.exchange_plan,
            decision.tree_segment_buckets_by_depth,
            decision.external_token_counts_by_rank,
            config,
        )
        if (
            decision.cp_size != cp_size
            or decision.external_token_counts_by_rank
            != token_layout_index.token_counts_by_rank
            or exchange.token_positions_by_rank != token_positions_by_rank
            or exchange.canonical_flat_token_positions
            != _canonical_flat_positions(spec)
            or (
                exchange.heads_local_tp,
                exchange.head_dim,
                exchange.groups_local_tp,
                exchange.state_dim,
            )
            != (heads, head_dim, groups, state_dim)
            or decision.family_artifact_identity != actual_identity
        ):
            raise ValueError("cached Mamba global decision has incompatible semantics")
        _validate_bucket_devices(
            decision.tree_segment_buckets_by_depth, torch.device("cpu")
        )

    def build_rank_plan(
        self,
        spec: RecurrentPackedExecutionSpec,
        decision: object,
        *,
        contract: LinearRecurrentContract,
        cp_rank: int,
        planner_config: object | None,
    ) -> object:
        self._validate(contract, planner_config)
        if not isinstance(
            decision, HeadShardedRecurrentGlobalDecision
        ) or not isinstance(decision.exchange_plan, MambaHeadShardExchangePlan):
            raise TypeError("Mamba adapter received an incompatible global decision")
        if cp_rank < 0 or cp_rank >= decision.cp_size:
            raise ValueError("Mamba rank must lie within its CP decision")
        if sum(decision.external_token_counts_by_rank) != spec.real_token_count:
            raise ValueError("Mamba decision and recurrent token count must match")
        return self.materialize_rank_plan(
            HeadShardedRecurrentRankExecutionPlan(
                cp_rank=cp_rank,
                cp_size=decision.cp_size,
                exchange_plan=decision.exchange_plan,
                tree_segment_buckets_by_depth=decision.tree_segment_buckets_by_depth,
                external_token_counts_by_rank=decision.external_token_counts_by_rank,
                family_artifact_identity=decision.family_artifact_identity,
            ),
            device="cpu",
        )

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
        config = self._validate(contract, planner_config)
        if not isinstance(
            plan, HeadShardedRecurrentRankExecutionPlan
        ) or not isinstance(plan.exchange_plan, MambaHeadShardDevicePlan):
            raise TypeError("Mamba adapter received an incompatible rank plan")
        target = torch.device(device)
        cpu = plan.exchange_plan.cpu
        if (
            plan.cp_rank != cp_rank
            or plan.cp_size != cpu.cp_size
            or len(plan.external_token_counts_by_rank) != plan.cp_size
            or tuple(map(len, cpu.token_positions_by_rank))
            != plan.external_token_counts_by_rank
            or sum(plan.external_token_counts_by_rank) != spec.real_token_count
        ):
            raise ValueError("cached Mamba rank-plan envelope is incompatible")
        heads, head_dim, groups, state_dim = self._geometry(contract, plan.cp_size)
        actual_identity = _family_artifact_identity(
            contract,
            spec,
            cpu,
            plan.tree_segment_buckets_by_depth,
            plan.external_token_counts_by_rank,
            config,
        )
        if (
            plan.family_artifact_identity != actual_identity
            or cpu.canonical_flat_token_positions != _canonical_flat_positions(spec)
            or (cpu.heads_local_tp, cpu.head_dim, cpu.groups_local_tp, cpu.state_dim)
            != (heads, head_dim, groups, state_dim)
        ):
            raise ValueError("cached Mamba rank plan has incompatible semantics")
        _validate_exchange_device_plan(plan.exchange_plan, target)
        _validate_bucket_devices(plan.tree_segment_buckets_by_depth, target)

    def materialize_rank_plan(
        self,
        plan: object,
        *,
        device: torch.device | str,
    ) -> object:
        if not isinstance(plan, HeadShardedRecurrentRankExecutionPlan):
            raise TypeError("Mamba adapter received an incompatible rank plan")
        if isinstance(plan.exchange_plan, MambaHeadShardDevicePlan):
            cpu_exchange = plan.exchange_plan.cpu
        elif isinstance(plan.exchange_plan, MambaHeadShardExchangePlan):
            cpu_exchange = plan.exchange_plan
        else:
            raise TypeError("Mamba adapter received an incompatible exchange plan")
        return plan.model_copy(
            update={
                "exchange_plan": materialize_mamba_head_shard_exchange_plan(
                    cpu_exchange, device
                ),
                "tree_segment_buckets_by_depth": tuple(
                    move_recurrent_segment_bucket_plans(buckets, device)
                    for buckets in plan.tree_segment_buckets_by_depth
                ),
            }
        )

    def model_token_counts(
        self,
        decision: object,
        *,
        attention_token_counts: tuple[int, ...],
    ) -> tuple[int, ...]:
        if not isinstance(
            decision, HeadShardedRecurrentGlobalDecision
        ) or not isinstance(decision.exchange_plan, MambaHeadShardExchangePlan):
            raise TypeError("Mamba adapter received an incompatible global decision")
        if attention_token_counts != decision.external_token_counts_by_rank:
            raise ValueError("Mamba and attention token ownership must match")
        return decision.external_token_counts_by_rank

    @staticmethod
    def _build_buckets(
        spec: RecurrentPackedExecutionSpec,
        contract: LinearRecurrentContract,
        config: HeadShardedFullTreePlannerConfig,
    ) -> tuple[tuple[RecurrentSegmentBucketPlan, ...], ...]:
        has_children = [False] * spec.family_count
        for parent in spec.tree_parent_indices:
            if parent >= 0:
                has_children[parent] = True
        return tuple(
            build_recurrent_tree_bucket_plans(
                tuple(
                    segment
                    for segment, segment_depth in zip(
                        spec.tree_segments, spec.tree_depths, strict=True
                    )
                    if segment_depth == depth
                ),
                spec.tree_parent_indices,
                tuple(has_children),
                sequence_length=spec.sequence_length,
                device="cpu",
                max_padded_tokens=config.max_padded_tokens,
                pad_to_multiple=contract.local_chunk_size,
                build_dense_rows=True,
            )
            for depth in range(max(spec.tree_depths, default=-1) + 1)
        )

    @staticmethod
    def _validate(
        contract: LinearRecurrentContract,
        planner_config: object | None,
    ) -> HeadShardedFullTreePlannerConfig:
        if (
            contract.family_key != MAMBA_FAMILY_KEY
            or contract.contract_version != "1"
            or contract.partition_kind != "head_sharded_full_tree"
            or contract.local_kernel_implementation_id != MAMBA_KERNEL_ID
            or contract.layout_compatibility_key != MAMBA_LAYOUT_KEY
            or contract.convolution_width != 4
            or contract.local_chunk_size != 128
            or contract.activation != "silu"
        ):
            raise ValueError("Mamba adapter received an incompatible contract")
        if tuple(stream.name for stream in contract.projected_streams) != (
            "z",
            "x",
            "B",
            "C",
            "dt",
        ) or tuple(state.name for state in contract.states) != ("conv", "ssm"):
            raise ValueError("Mamba adapter requires ordered streams and states")
        if not isinstance(planner_config, HeadShardedFullTreePlannerConfig):
            raise TypeError("Mamba adapter requires HeadShardedFullTreePlannerConfig")
        return planner_config

    @staticmethod
    def _geometry(
        contract: LinearRecurrentContract,
        cp_size: int,
    ) -> tuple[int, int, int, int]:
        streams = {stream.name: stream for stream in contract.projected_streams}
        states = {state.name: state for state in contract.states}
        heads = streams["dt"].width
        if streams["z"].width != streams["x"].width or streams["z"].width % heads:
            raise ValueError("Mamba z/x widths must define equal whole heads")
        head_dim = streams["z"].width // heads
        if heads % cp_size:
            raise ValueError("Mamba heads must divide evenly across CP ranks")
        ssm_shape = states["ssm"].shape
        if len(ssm_shape) != 3 or ssm_shape[:2] != (heads // cp_size, head_dim):
            raise ValueError("Mamba SSM state must be [heads, head_dim, state_dim]")
        state_dim = ssm_shape[2]
        if streams["B"].width != streams["C"].width or (streams["B"].width % state_dim):
            raise ValueError("Mamba B/C widths must define whole state groups")
        groups = streams["B"].width // state_dim
        if heads % groups:
            raise ValueError("Mamba heads must divide evenly across state groups")
        if any(
            (
                streams[name].shard_axis,
                streams[name].shard_count,
                streams[name].replication,
                streams[name].replication_factor,
            )
            != ("head", heads, "none", 1)
            for name in ("z", "x", "dt")
        ):
            raise ValueError("Mamba z/x/dt sharding metadata is inconsistent")
        if any(
            (
                streams[name].shard_axis,
                streams[name].shard_count,
                streams[name].replication,
                streams[name].replication_factor,
            )
            != ("group", groups, "group_to_heads", heads // groups)
            for name in ("B", "C")
        ):
            raise ValueError("Mamba B/C sharding metadata is inconsistent")
        groups_local_cp = max(groups // cp_size, 1)
        expected_conv = (
            heads // cp_size * head_dim + 2 * groups_local_cp * state_dim,
            contract.convolution_width - 1,
        )
        if states["conv"].shape != expected_conv or states["ssm"].dtype != "float32":
            raise ValueError("Mamba recurrent state metadata is inconsistent")
        return heads, head_dim, groups, state_dim


_BUCKET_DEVICE_FIELDS = (
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


def _family_artifact_identity(
    contract: LinearRecurrentContract,
    spec: RecurrentPackedExecutionSpec,
    exchange: MambaHeadShardExchangePlan,
    buckets_by_depth: tuple[tuple[object, ...], ...],
    token_counts: tuple[int, ...],
    planner_config: HeadShardedFullTreePlannerConfig,
) -> str:
    bucket_payload = []
    for buckets in buckets_by_depth:
        depth = []
        for bucket in buckets:
            if not isinstance(bucket, RecurrentSegmentBucketPlan):
                raise TypeError("Mamba adapter received an incompatible tree bucket")
            depth.append(
                {
                    "length": bucket.length,
                    "padded_length": bucket.padded_length,
                    "real_token_count_static": bucket.real_token_count_static,
                    "family_indices_cpu_tuple": bucket.family_indices_cpu_tuple,
                    "parent_indices_cpu_tuple": bucket.parent_indices_cpu_tuple,
                    "needs_final_state": bucket.needs_final_state,
                    "artifact_identity": bucket.artifact_identity,
                }
            )
        bucket_payload.append(depth)
    payload = {
        "contract": contract.planning_identity,
        "spec": spec.model_dump(mode="json"),
        "planner_config": planner_config.model_dump(mode="json"),
        "exchange": exchange.model_dump(mode="json"),
        "token_counts": token_counts,
        "buckets_by_depth": bucket_payload,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_exchange_device_plan(
    plan: MambaHeadShardDevicePlan,
    device: torch.device,
) -> None:
    fields = (
        "token_positions_by_rank",
        "projected_feature_positions_by_rank",
        "conv_feature_positions_by_rank",
        "head_positions_by_rank",
        "group_positions_by_rank",
        "head_feature_positions_by_rank",
    )
    for name in fields:
        tensors = getattr(plan, name)
        expected_rows = getattr(plan.cpu, name)
        if len(tensors) != len(expected_rows):
            raise ValueError("Mamba device exchange rank metadata is incompatible")
        for tensor, expected in zip(tensors, expected_rows, strict=True):
            _validate_index_tensor(tensor, expected, device)
    _validate_index_tensor(
        plan.canonical_to_received_positions,
        plan.cpu.canonical_to_received_positions,
        device,
    )
    _validate_index_tensor(
        plan.canonical_flat_token_positions,
        plan.cpu.canonical_flat_token_positions,
        device,
    )


def _validate_index_tensor(
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    device: torch.device,
) -> None:
    invalid = (
        tensor.device != device
        or tensor.dtype != torch.long
        or tuple(tensor.shape) != (len(expected),)
    )
    if device.type == "cpu" and not invalid:
        invalid = tuple(tensor.tolist()) != expected
    if invalid:
        raise ValueError("Mamba device exchange indices are incompatible")


def _validate_bucket_devices(
    buckets_by_depth: tuple[tuple[object, ...], ...],
    device: torch.device,
) -> None:
    for buckets in buckets_by_depth:
        for bucket in buckets:
            if not isinstance(bucket, RecurrentSegmentBucketPlan):
                raise TypeError("Mamba adapter received an incompatible tree bucket")
            _validate_bucket_device_metadata(bucket, device)


def _validate_bucket_device_metadata(
    bucket: RecurrentSegmentBucketPlan,
    device: torch.device,
) -> None:
    expected = {
        "lengths": (torch.long, (bucket.segment_count,)),
        "real_mask": (torch.bool, (bucket.real_token_count_static,)),
        "dense_real_mask": (torch.bool, (bucket.segment_count, bucket.length)),
        "dense_token_indices": (torch.long, (bucket.real_token_count_static,)),
        "cu_seqlens": (torch.long, (bucket.segment_count + 1,)),
        "row_indices": (torch.long, (bucket.real_token_count_static,)),
        "position_indices": (torch.long, (bucket.real_token_count_static,)),
        "flat_token_indices": (torch.long, (bucket.real_token_count_static,)),
        "dense_row_indices": (torch.long, (bucket.segment_count, bucket.length)),
        "dense_position_indices": (torch.long, (bucket.segment_count, bucket.length)),
        "family_indices": (torch.long, (bucket.segment_count,)),
        "parent_indices": (torch.long, (bucket.segment_count,)),
    }
    for name in _BUCKET_DEVICE_FIELDS:
        tensor = getattr(bucket, name)
        if tensor is None:
            if name in {
                "dense_real_mask",
                "dense_token_indices",
                "flat_token_indices",
                "dense_row_indices",
                "dense_position_indices",
                "parent_indices",
            }:
                raise ValueError(
                    "Mamba tree bucket is missing dense execution metadata"
                )
            continue
        dtype, shape = expected.get(name, (torch.bool, tuple(tensor.shape)))
        if (
            tensor.device != device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
        ):
            raise ValueError("Mamba tree bucket device metadata is incompatible")


def _token_positions_by_rank(
    token_layout_index: TokenLayoutIndex,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            token
            for start, end, _position in sorted(ranges, key=lambda item: item[2])
            for token in range(int(start), int(end))
        )
        for ranges in token_layout_index.ownership_ranges_by_rank
    )


def _canonical_flat_positions(
    spec: RecurrentPackedExecutionSpec,
) -> tuple[int, ...]:
    return tuple(
        row * spec.sequence_length + position
        for row, valid_length in enumerate(spec.valid_lengths)
        for position in range(valid_length)
    )
