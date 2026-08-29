from __future__ import annotations

from array import array
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.pipeline_tuner.config import PackedGroupShape
from art.trajectories import Trajectory

from .tokenized import (
    MAX_TOKENIZED_PHYSICAL_VALUES,
    TokenizedDatum,
    tokenized_physical_value_count,
    validate_tokenized_loss_values,
)

MAX_CONTROL_IDENTIFIER_LENGTH = 255
MAX_CHECKPOINT_REFERENCE_LENGTH = 2048
MAX_TOKEN_LOGPROB_VALUES = 16_777_216
MAX_TARGET_MODULES = 256


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AdapterSpec(Contract):
    rank: int = Field(ge=1, le=1024)
    target_modules: tuple[str, ...] = Field(min_length=1, max_length=MAX_TARGET_MODULES)

    @model_validator(mode="after")
    def _validate_targets(self) -> AdapterSpec:
        if any(not target or len(target) > 255 for target in self.target_modules):
            raise ValueError("LoRA target modules must be nonempty and bounded")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target modules must be unique")
        return self


class TrainingRunSpec(Contract):
    base_model: str = Field(min_length=1, max_length=512)
    adapter: AdapterSpec
    seed: int | None = None
    dtype: Literal["bfloat16"] = "bfloat16"


class RunCommand(Contract):
    run_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    request_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
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


class TokenizedTrainingBatch(Contract):
    kind: Literal["tokenized"] = "tokenized"
    datums: tuple[TokenizedDatum, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_size(self) -> TokenizedTrainingBatch:
        if tokenized_physical_value_count(self.datums) > MAX_TOKENIZED_PHYSICAL_VALUES:
            raise ValueError("tokenized batch exceeds the configured value limit")
        return self


TrainingBatch = Annotated[
    RlTrajectoryBatch | SupervisedTrajectoryBatch | TokenizedTrainingBatch,
    Field(discriminator="kind"),
]


class LossConfig(Contract):
    name: Literal["cross_entropy", "importance_sampling", "cispo", "ppo"]
    normalize_advantages: bool = True
    values: dict[str, FiniteFloat | int | bool | str | None] = Field(
        default_factory=dict
    )


class ForwardRequest(RunCommand):
    batch: TrainingBatch
    loss: LossConfig
    collect_packing_shapes: bool = False
    return_token_logprobs: bool = True

    @model_validator(mode="after")
    def _validate_loss(self) -> "ForwardRequest":
        expected = {
            "sft": {"cross_entropy"},
            "rl": {"cispo", "ppo"},
            "tokenized": {"cross_entropy", "importance_sampling", "cispo"},
        }[self.batch.kind]
        if self.loss.name not in expected:
            raise ValueError(
                f"{self.batch.kind} batches require one of {sorted(expected)}, "
                f"got {self.loss.name!r}"
            )
        if isinstance(self.batch, TokenizedTrainingBatch):
            loss = self.loss.name
            assert loss != "ppo"
            validate_tokenized_loss_values(loss, self.loss.values)
            for datum in self.batch.datums:
                datum.validate_for_loss(loss)
        return self


class ForwardBackwardRequest(ForwardRequest):
    pass


class AdamConfig(Contract):
    learning_rate: FiniteFloat = Field(ge=0)
    beta1: FiniteFloat = Field(default=0.9, ge=0, lt=1)
    beta2: FiniteFloat = Field(default=0.99, ge=0, lt=1)
    eps: FiniteFloat = Field(default=1e-13, gt=0)
    weight_decay: FiniteFloat = Field(default=0.1, ge=0)
    grad_clip_norm: FiniteFloat = Field(default=0.1, ge=0)


class OptimStepRequest(RunCommand):
    optimizer: AdamConfig


class SamplerPublication(Contract):
    mode: Literal["none", "versioned_lora", "in_flight_lora", "merged_weights"]
    model_alias: str | None = Field(
        default=None, min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH
    )

    @model_validator(mode="after")
    def _validate_alias(self) -> "SamplerPublication":
        if (self.mode == "none") != (self.model_alias is None):
            raise ValueError(
                "model_alias is required exactly when publication is enabled"
            )
        return self


class SaveWeightsForSamplerRequest(RunCommand):
    checkpoint_name: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    ttl_seconds: int | None = Field(default=None, ge=1)
    publication: SamplerPublication


class SaveStateRequest(RunCommand):
    checkpoint_name: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    ttl_seconds: int | None = Field(default=None, ge=1)
    overwrite: bool = False


class LoadStateRequest(RunCommand):
    checkpoint: str = Field(min_length=1, max_length=MAX_CHECKPOINT_REFERENCE_LENGTH)
    restore_optimizer: bool = False


OperationKind = Literal[
    "forward",
    "forward_backward",
    "optim_step",
    "save_sampler",
    "save_state",
    "load_state",
]


class OperationRef(Contract):
    run_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    operation_id: str = Field(min_length=1, max_length=64)
    sequence_id: int = Field(ge=0)
    learner_parent_version: int = Field(ge=0)
    reserved_output_learner_version: int | None = Field(default=None, ge=1)
    kind: OperationKind

    @model_validator(mode="after")
    def _validate_transition(self) -> "OperationRef":
        transition = self.kind in {"optim_step", "load_state"}
        if transition != (self.reserved_output_learner_version is not None):
            raise ValueError("learner transitions must reserve an output version")
        if transition and self.reserved_output_learner_version != (
            self.learner_parent_version + 1
        ):
            raise ValueError("learner transitions must advance exactly one version")
        return self


class CheckpointRef(Contract):
    run_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    learner_version: int = Field(ge=0)
    checkpoint_id: str = Field(min_length=1, max_length=MAX_CHECKPOINT_REFERENCE_LENGTH)


class PackingOutcome(Contract):
    packed_sequence_length: int = Field(ge=1)
    packed_sequences: int = Field(ge=0)
    target_packed_sequences: int = Field(ge=1)
    physical_tokens: int = Field(ge=0)
    non_padding_tokens: int = Field(ge=0)
    loss_bearing_tokens: int = Field(ge=0)
    trainable_assistant_tokens: int = Field(ge=0)
    group_shapes: tuple[PackedGroupShape, ...] = ()

    @model_validator(mode="after")
    def _validate_counts(self) -> "PackingOutcome":
        if self.non_padding_tokens > self.physical_tokens:
            raise ValueError("non_padding_tokens cannot exceed physical_tokens")
        if (self.packed_sequences == 0) != (self.physical_tokens == 0):
            raise ValueError("zero packed sequences and physical tokens must agree")
        if self.loss_bearing_tokens > self.non_padding_tokens:
            raise ValueError("loss_bearing_tokens cannot exceed non_padding_tokens")
        return self


class TokenLogprobs(Contract):
    model_config = ConfigDict(
        extra="forbid", frozen=True, ser_json_bytes="base64", val_json_bytes="base64"
    )

    shape: tuple[int, ...] = Field(min_length=1, max_length=2)
    data: bytes

    @model_validator(mode="after")
    def _validate_buffer(self) -> "TokenLogprobs":
        if any(value < 0 for value in self.shape):
            raise ValueError("token logprob dimensions cannot be negative")
        if len(self.shape) == 2 and self.shape[1] < 1:
            raise ValueError("token logprob candidate width must be positive")
        if self.value_count > MAX_TOKEN_LOGPROB_VALUES:
            raise ValueError("token logprob buffer exceeds the configured value limit")
        if len(self.data) != self.value_count * 4:
            raise ValueError("token logprob buffer size differs from its shape")
        return self

    @property
    def value_count(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    @classmethod
    def from_values(
        cls, values: list[float], *, shape: tuple[int, ...] | None = None
    ) -> "TokenLogprobs":
        buffer = array("f", values)
        if sys.byteorder != "little":
            buffer.byteswap()
        return cls(shape=shape or (len(values),), data=buffer.tobytes())

    def to_values(self) -> list[float]:
        buffer = array("f")
        buffer.frombytes(self.data)
        if sys.byteorder != "little":
            buffer.byteswap()
        return buffer.tolist()


class OperationResult(Contract):
    operation_id: str = Field(min_length=1, max_length=64)
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)


class ForwardResult(OperationResult):
    kind: Literal["forward"] = "forward"
    packing: PackingOutcome
    token_logprobs: tuple[TokenLogprobs, ...] = ()


class ForwardBackwardResult(ForwardResult):
    kind: Literal["forward_backward"] = "forward_backward"


class OptimStepResult(OperationResult):
    kind: Literal["optim_step"] = "optim_step"
    contributing_forward_backward_operation_ids: tuple[str, ...] = Field(min_length=1)
    checkpoint: CheckpointRef


class SamplerWeightsResult(OperationResult):
    kind: Literal["save_sampler"] = "save_sampler"
    checkpoint: CheckpointRef
    lora: str = Field(min_length=1, max_length=MAX_CHECKPOINT_REFERENCE_LENGTH)


class SaveStateResult(OperationResult):
    kind: Literal["save_state"] = "save_state"
    checkpoint: CheckpointRef
    optimizer_state: str = Field(
        min_length=1, max_length=MAX_CHECKPOINT_REFERENCE_LENGTH
    )


class LoadStateResult(OperationResult):
    kind: Literal["load_state"] = "load_state"
    checkpoint: CheckpointRef
    optimizer_restored: bool


OperationResultType = Annotated[
    ForwardResult
    | ForwardBackwardResult
    | OptimStepResult
    | SamplerWeightsResult
    | SaveStateResult
    | LoadStateResult,
    Field(discriminator="kind"),
]
