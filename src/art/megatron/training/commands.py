from __future__ import annotations

import math
from typing import Any, cast

from art.distributed.art_runtime import DistributedPackedBatch
from art.megatron.runtime.data_plane import SFTBatchData
from art.preprocessing.tokenize import SFTBatch
from art.training.contracts import (
    CheckpointRef,
    ForwardBackwardRequest,
    ForwardRequest,
    PackingOutcome,
    PolicyTokenCount,
)

from ..runtime.specs import ExperimentalTrainConfig, RlForwardBackwardConfig


def forward_backward_config(
    request: ForwardRequest | ForwardBackwardRequest,
) -> RlForwardBackwardConfig:
    values = request.loss.values
    kl_penalty_coef = values.get("kl_penalty_coef", 0.0)
    if not isinstance(kl_penalty_coef, int | float):
        raise TypeError("kl_penalty_coef must be numeric")
    return RlForwardBackwardConfig(
        kl_penalty_coef=float(kl_penalty_coef),
        kl_penalty_source=cast(Any, values.get("kl_penalty_source", "current_learner")),
        grad_accumulation_sequences=cast(
            int | None, values.get("grad_accumulation_sequences")
        ),
    )


def experimental_train_config(
    request: ForwardRequest | ForwardBackwardRequest,
) -> ExperimentalTrainConfig:
    values = {
        name: value
        for name, value in request.loss.values.items()
        if name in ExperimentalTrainConfig.model_fields
    }
    values["ppo"] = request.loss.name == "ppo"
    values["scale_rewards"] = bool(values.get("scale_rewards", True))
    return ExperimentalTrainConfig.model_validate(values)


def packing_outcome(
    packed: DistributedPackedBatch, *, target_packed_sequences: int
) -> PackingOutcome:
    ref = packed.leases.ref
    stats = ref.prefix_tree_packing_stats
    if stats is None or stats.policy_token_counts is None:
        raise RuntimeError("RL packed batch has no exact policy-token provenance")
    return PackingOutcome(
        packed_sequence_length=ref.sequence_length,
        packed_sequences=ref.num_sequences,
        target_packed_sequences=target_packed_sequences,
        nominal_capacity_tokens=(
            math.ceil(ref.num_sequences / target_packed_sequences)
            * target_packed_sequences
            * ref.sequence_length
        ),
        physical_tokens=stats.physical_tokens,
        non_padding_tokens=packed.non_padding_tokens,
        loss_bearing_tokens=packed.loss_bearing_tokens,
        trainable_assistant_tokens=packed.trainable_assistant_tokens,
        policy_token_counts=tuple(
            PolicyTokenCount(
                policy_version=version,
                trainable_assistant_tokens=count,
            )
            for version, count in sorted(stats.policy_token_counts.items())
        ),
        group_shapes=tuple(
            shape for shape in packed.packed_group_shapes if shape is not None
        ),
    )


def packing_metrics(packed: DistributedPackedBatch) -> dict[str, float]:
    return {
        "time/step_trajectory_fetch_s": packed.trajectory_fetch_s,
        "time/step_trajectory_receive_s": packed.trajectory_receive_s,
        "time/step_trajectory_build_s": packed.trajectory_build_s,
        "time/step_packing_core_s": packed.packing_core_s,
        "time/step_packing_lock_wait_s": packed.packing_lock_wait_s,
        "time/step_packing_compute_s": packed.packing_compute_s,
        "time/step_trajectory_log_wait_s": packed.trajectory_log_wait_s,
        "time/step_packed_batch_finalize_s": packed.packed_batch_finalize_s,
        "time/step_packing_rpc_s": packed.packing_rpc_s,
        "time/step_packed_batch_fanout_s": packed.packed_batch_fanout_s,
    }


def sft_batch_data(batch: SFTBatch) -> SFTBatchData:
    return SFTBatchData(
        trajectory_tensors=tuple(batch.trajectory_tensors),
        learning_rate=batch.learning_rate,
        num_trajectories=batch.num_trajectories,
        num_tokens=batch.num_tokens,
        num_trainable_tokens=batch.num_trainable_tokens,
        num_dropped_trajectories=batch.num_dropped_trajectories,
    )


def sft_packing_outcome(batch: SFTBatchData) -> PackingOutcome:
    max_length = max(
        int(tensors["input_ids"].numel()) for tensors in batch.trajectory_tensors
    )
    return PackingOutcome(
        packed_sequence_length=max_length,
        packed_sequences=batch.num_trajectories,
        target_packed_sequences=batch.num_trajectories,
        nominal_capacity_tokens=batch.num_tokens,
        physical_tokens=batch.num_tokens,
        non_padding_tokens=batch.num_tokens,
        loss_bearing_tokens=batch.num_trainable_tokens,
        trainable_assistant_tokens=batch.num_trainable_tokens,
        policy_token_counts=None,
        group_shapes=(),
    )


def checkpoint_ref(
    run_id: str, learner_version: int, checkpoint_id: str
) -> CheckpointRef:
    return CheckpointRef(
        run_id=run_id,
        learner_version=learner_version,
        checkpoint_id=checkpoint_id,
    )
