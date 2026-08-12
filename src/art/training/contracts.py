from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.pipeline_tuner.config import PackedGroupShape
from art.trajectories import Trajectory


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunCommand(Contract):
    run_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    sequence_id: int = Field(ge=0)


class RlTrajectoryBatch(Contract):
    kind: Literal["rl"] = "rl"
    groups: tuple[TrajectoryGroupBundle, ...] = Field(min_length=1)
    min_source_version: int = Field(ge=0)
    max_source_version: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_source_versions(self) -> "RlTrajectoryBatch":
        if self.max_source_version < self.min_source_version:
            raise ValueError("max_source_version must be >= min_source_version")
        return self


class SupervisedTrajectoryBatch(Contract):
    kind: Literal["sft"] = "sft"
    trajectories: tuple[Trajectory, ...] = Field(min_length=1)
    assistant_turns: Literal["all", "last"] = "all"


TrainingBatch = Annotated[
    RlTrajectoryBatch | SupervisedTrajectoryBatch,
    Field(discriminator="kind"),
]


class LossConfig(Contract):
    name: Literal["cross_entropy", "cispo", "ppo"]
    normalize_advantages: bool = True
    values: dict[str, float | int | bool | str | None] = Field(default_factory=dict)


class ForwardRequest(RunCommand):
    batch: TrainingBatch
    loss: LossConfig
    collect_packing_shapes: bool = False

    @model_validator(mode="after")
    def _validate_loss(self) -> "ForwardRequest":
        expected = {"cross_entropy"} if self.batch.kind == "sft" else {"cispo", "ppo"}
        if self.loss.name not in expected:
            raise ValueError(
                f"{self.batch.kind} batches require one of {sorted(expected)}, "
                f"got {self.loss.name!r}"
            )
        return self


class ForwardBackwardRequest(ForwardRequest):
    pass


class AdamConfig(Contract):
    learning_rate: float = Field(ge=0)
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.95, ge=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)


class OptimStepRequest(RunCommand):
    optimizer: AdamConfig


class SamplerPublication(Contract):
    mode: Literal["none", "versioned_lora", "in_flight_lora"]
    model_alias: str | None = None

    @model_validator(mode="after")
    def _validate_alias(self) -> "SamplerPublication":
        if (self.mode == "none") != (self.model_alias is None):
            raise ValueError(
                "model_alias is required exactly when publication is enabled"
            )
        return self


class SaveWeightsForSamplerRequest(RunCommand):
    checkpoint_name: str = Field(min_length=1)
    ttl_seconds: int | None = Field(default=None, ge=1)
    publication: SamplerPublication


class SaveStateRequest(RunCommand):
    checkpoint_name: str = Field(min_length=1)
    ttl_seconds: int | None = Field(default=None, ge=1)
    overwrite: bool = False


class LoadStateRequest(RunCommand):
    checkpoint: str = Field(min_length=1)


OperationKind = Literal[
    "forward",
    "forward_backward",
    "optim_step",
    "save_sampler",
    "save_state",
    "load_state",
]


class OperationRef(Contract):
    run_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    sequence_id: int = Field(ge=0)
    learner_parent_version: int = Field(ge=0)
    reserved_output_learner_version: int | None = Field(default=None, ge=0)
    kind: OperationKind

    @model_validator(mode="after")
    def _validate_transition(self) -> "OperationRef":
        transition = self.kind in {"optim_step", "load_state"}
        if transition != (self.reserved_output_learner_version is not None):
            raise ValueError(
                "optimizer and load operations must reserve an output learner "
                "version; other operations must not"
            )
        if transition and self.reserved_output_learner_version != (
            self.learner_parent_version + 1
        ):
            raise ValueError("learner transitions must advance exactly one version")
        return self


class CheckpointRef(Contract):
    run_id: str = Field(min_length=1)
    learner_version: int = Field(ge=0)
    checkpoint_id: str = Field(min_length=1)


class PolicyTokenCount(Contract):
    policy_version: int = Field(ge=0)
    trainable_assistant_tokens: int = Field(ge=1)


class PackingOutcome(Contract):
    packed_sequence_length: int = Field(ge=1)
    packed_sequences: int = Field(ge=1)
    target_packed_sequences: int = Field(ge=1)
    nominal_capacity_tokens: int = Field(ge=1)
    physical_tokens: int = Field(ge=1)
    non_padding_tokens: int = Field(ge=1)
    loss_bearing_tokens: int = Field(ge=0)
    trainable_assistant_tokens: int = Field(ge=0)
    policy_token_counts: tuple[PolicyTokenCount, ...] | None
    group_shapes: tuple[PackedGroupShape, ...]

    @model_validator(mode="after")
    def _validate_counts(self) -> "PackingOutcome":
        if self.non_padding_tokens > self.physical_tokens:
            raise ValueError("non_padding_tokens cannot exceed physical_tokens")
        counts = self.policy_token_counts
        if counts is not None:
            versions = [count.policy_version for count in counts]
            if versions != sorted(set(versions)):
                raise ValueError("policy_token_counts must be unique and sorted")
            if sum(count.trainable_assistant_tokens for count in counts) != (
                self.trainable_assistant_tokens
            ):
                raise ValueError(
                    "policy_token_counts must sum to trainable_assistant_tokens"
                )
        return self


class LossFnOutput(Contract):
    token_logprobs: tuple[float, ...]
    metrics: dict[str, float] = Field(default_factory=dict)


class ForwardResult(Contract):
    operation_id: str = Field(min_length=1)
    packing: PackingOutcome
    loss_fn_outputs: tuple[LossFnOutput, ...]
    metrics: dict[str, float] = Field(default_factory=dict)


class ForwardBackwardResult(ForwardResult):
    pass


class OptimStepResult(Contract):
    operation_id: str = Field(min_length=1)
    contributing_forward_backward_operation_ids: tuple[str, ...] = Field(min_length=1)
    checkpoint: CheckpointRef
    metrics: dict[str, float] = Field(default_factory=dict)


class SamplerWeightsResult(Contract):
    operation_id: str = Field(min_length=1)
    checkpoint: CheckpointRef
    lora: str = Field(min_length=1)
    publication_metrics: dict[str, float] = Field(default_factory=dict)


class SaveStateResult(Contract):
    operation_id: str = Field(min_length=1)
    checkpoint: CheckpointRef
    optimizer_state: str = Field(min_length=1)


class LoadStateResult(Contract):
    operation_id: str = Field(min_length=1)
    checkpoint: CheckpointRef
    optimizer_restored: bool
