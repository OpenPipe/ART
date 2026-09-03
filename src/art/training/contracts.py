from __future__ import annotations

from array import array
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from art.pipeline_tuner.config import PackedGroupShape

from .token_matrix import (
    LossContractId,
    NamedLossRequest,
    TokenMatrixBatch,
    validate_token_matrix_batch,
)

MAX_CONTROL_IDENTIFIER_LENGTH = 255
MAX_CHECKPOINT_REFERENCE_LENGTH = 2048
MAX_TOKEN_LOGPROB_VALUES = 16_777_216
MAX_TARGET_MODULES = 256
MAX_INPUT_OBJECT_REFERENCE_LENGTH = 2048


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


class ServiceCheckpointSource(Contract):
    kind: Literal["service_checkpoint"] = "service_checkpoint"
    checkpoint_id: str = Field(min_length=1, max_length=128)


class WandbArtifactCheckpointSource(Contract):
    kind: Literal["wandb_artifact"] = "wandb_artifact"
    artifact: str = Field(
        min_length=5,
        max_length=768,
        pattern=r"^[^/:\\\x00-\x1f]+/[^/:\\\x00-\x1f]+/[^/:\\\x00-\x1f]+:v(?:0|[1-9][0-9]*)$",
    )


CheckpointSource = Annotated[
    ServiceCheckpointSource | WandbArtifactCheckpointSource,
    Field(discriminator="kind"),
]


class RunInitialState(Contract):
    source: CheckpointSource
    restore_optimizer: bool = False


class RunCommand(Contract):
    run_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    request_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    sequence_id: int = Field(ge=0)


class TrainingInputObject(Contract):
    """Authenticated immutable object identity; access remains resolver-owned."""

    store: Literal["caios"] = "caios"
    locator: str = Field(min_length=1, max_length=MAX_INPUT_OBJECT_REFERENCE_LENGTH)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainingInputObjectRef(Contract):
    """Digest-bound identity for an immutable, externally stored input batch."""

    kind: Literal["input_object"] = "input_object"
    run_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    operation_id: str = Field(min_length=1, max_length=64)
    encoding: Literal["art_token_matrix_batch_json_v1"] = (
        "art_token_matrix_batch_json_v1"
    )
    object: TrainingInputObject
    lease_id: str = Field(min_length=1, max_length=512)


class PackedInputCaptureRef(Contract):
    """Disposable slot-local materialization of one immutable packed input."""

    kind: Literal["captured"] = "captured"
    run_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    capture_id: str = Field(min_length=1, max_length=64)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_object: TrainingInputObjectRef | None = None


TrainingBatch = Annotated[
    TokenMatrixBatch | TrainingInputObjectRef | PackedInputCaptureRef,
    Field(discriminator="kind"),
]


class ForwardRequest(RunCommand):
    batch: TrainingBatch
    loss: NamedLossRequest
    collect_packing_shapes: bool = False
    retain_packed_input: bool = False
    return_token_logprobs: bool = True

    @model_validator(mode="after")
    def _validate_loss(self) -> "ForwardRequest":
        if isinstance(self.batch, TokenMatrixBatch):
            validate_token_matrix_batch(
                self.batch,
                self.loss,
                output_rows=("learner_logprobs",) if self.return_token_logprobs else (),
            )
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
    mode: Literal[
        "none",
        "versioned_lora",
        "in_flight_lora",
        "external_lora",
        "merged_weights",
    ]
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


class CheckpointArchiveRef(Contract):
    checkpoint_id: str = Field(min_length=1, max_length=MAX_CHECKPOINT_REFERENCE_LENGTH)
    learner_version: int = Field(ge=0)
    components: tuple[Literal["weights", "optimizer"], ...] = Field(
        min_length=1, max_length=2
    )
    wandb_artifact: str = Field(
        min_length=5,
        max_length=768,
        pattern=r"^[^/:\\\x00-\x1f]+/[^/:\\\x00-\x1f]+/[^/:\\\x00-\x1f]+:v(?:0|[1-9][0-9]*)$",
    )


class ImmutablePublicationRef(Contract):
    locator: str = Field(min_length=1, max_length=MAX_CHECKPOINT_REFERENCE_LENGTH)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExternalLoraReceipt(Contract):
    generation_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    active_alias: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    manifest: ImmutablePublicationRef
    shards: tuple[ImmutablePublicationRef, ...] = Field(min_length=1, max_length=10_000)


class PackingOutcome(Contract):
    packed_sequence_length: int = Field(ge=1)
    packed_sequences: int = Field(ge=0)
    target_packed_sequences: int = Field(ge=1)
    logical_tokens: int = Field(ge=0)
    physical_tokens: int = Field(ge=0)
    packed_capacity_tokens: int = Field(ge=0)
    padding_tokens: int = Field(ge=0)
    group_shapes: tuple[PackedGroupShape, ...] = ()

    @model_validator(mode="after")
    def _validate_counts(self) -> "PackingOutcome":
        expected_capacity = self.packed_sequences * self.packed_sequence_length
        if self.packed_capacity_tokens != expected_capacity:
            raise ValueError("packed capacity does not match shape")
        if self.physical_tokens > self.logical_tokens:
            raise ValueError("physical tokens cannot exceed logical tokens")
        if self.physical_tokens > self.packed_capacity_tokens:
            raise ValueError("physical tokens cannot exceed packed capacity")
        if self.padding_tokens != self.packed_capacity_tokens - self.physical_tokens:
            raise ValueError("padding tokens must equal capacity minus physical tokens")
        if (self.packed_sequences == 0) != (self.logical_tokens == 0):
            raise ValueError("zero packed sequences and logical tokens must agree")
        return self


class PolicyTokenCount(Contract):
    policy_version: int = Field(ge=0)
    accepted_trainable_tokens: int = Field(ge=1)


class TrainingOutcome(Contract):
    accepted_trainable_tokens: int = Field(ge=0)
    policy_token_counts: tuple[PolicyTokenCount, ...] | None

    @model_validator(mode="after")
    def _validate_policy_counts(self) -> "TrainingOutcome":
        if self.policy_token_counts is None:
            return self
        versions = tuple(item.policy_version for item in self.policy_token_counts)
        if versions != tuple(sorted(set(versions))):
            raise ValueError("policy token counts must be sorted and unique")
        if (
            sum(item.accepted_trainable_tokens for item in self.policy_token_counts)
            != self.accepted_trainable_tokens
        ):
            raise ValueError("policy token counts must cover accepted trainable tokens")
        return self


class NamedLossOutcome(Contract):
    contract_id: LossContractId
    value: FiniteFloat
    reduction: Literal["mean_active_token"] = "mean_active_token"


class TokenLogprobs(Contract):
    model_config = ConfigDict(
        extra="forbid", frozen=True, ser_json_bytes="base64", val_json_bytes="base64"
    )

    matrix_id: str = Field(min_length=1, max_length=255)
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
        cls,
        values: list[float],
        *,
        matrix_id: str,
        shape: tuple[int, ...] | None = None,
    ) -> "TokenLogprobs":
        buffer = array("f", values)
        if sys.byteorder != "little":
            buffer.byteswap()
        return cls(
            matrix_id=matrix_id,
            shape=shape or (len(values),),
            data=buffer.tobytes(),
        )

    def to_values(self) -> list[float]:
        buffer = array("f")
        buffer.frombytes(self.data)
        if sys.byteorder != "little":
            buffer.byteswap()
        return buffer.tolist()


UsageCoverage = Literal["complete", "exact_partial", "unknown", "not_applicable"]


class UsageMeasurement(Contract):
    coverage: UsageCoverage
    value: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_coverage(self) -> "UsageMeasurement":
        if (self.coverage in {"complete", "exact_partial"}) != (self.value is not None):
            raise ValueError("known usage coverage requires a value")
        return self

    @classmethod
    def complete(cls, value: int) -> "UsageMeasurement":
        return cls(coverage="complete", value=value)

    @classmethod
    def exact_partial(cls, value: int) -> "UsageMeasurement":
        return cls(coverage="exact_partial", value=value)

    @classmethod
    def unknown(cls) -> "UsageMeasurement":
        return cls(coverage="unknown")

    @classmethod
    def not_applicable(cls) -> "UsageMeasurement":
        return cls(coverage="not_applicable")


class CommandExecutionUsage(Contract):
    """ART facts; GPU service is max-rank duration and count stays separate."""

    logical_nonpadding_tokens: UsageMeasurement
    executed_token_equivalents: UsageMeasurement
    gpu_count: UsageMeasurement
    gpu_service_ns: UsageMeasurement

    @classmethod
    def unknown(cls) -> "CommandExecutionUsage":
        return cls(
            logical_nonpadding_tokens=UsageMeasurement.unknown(),
            executed_token_equivalents=UsageMeasurement.unknown(),
            gpu_count=UsageMeasurement.unknown(),
            gpu_service_ns=UsageMeasurement.unknown(),
        )

    @classmethod
    def no_work(cls) -> "CommandExecutionUsage":
        return cls(
            logical_nonpadding_tokens=UsageMeasurement.exact_partial(0),
            executed_token_equivalents=UsageMeasurement.exact_partial(0),
            gpu_count=UsageMeasurement.exact_partial(0),
            gpu_service_ns=UsageMeasurement.exact_partial(0),
        )

    @classmethod
    def not_applicable(cls) -> "CommandExecutionUsage":
        return cls(
            logical_nonpadding_tokens=UsageMeasurement.not_applicable(),
            executed_token_equivalents=UsageMeasurement.not_applicable(),
            gpu_count=UsageMeasurement.not_applicable(),
            gpu_service_ns=UsageMeasurement.not_applicable(),
        )


class OperationResult(Contract):
    operation_id: str = Field(min_length=1, max_length=64)
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    usage: CommandExecutionUsage = Field(
        default_factory=CommandExecutionUsage.not_applicable
    )


class ForwardResult(OperationResult):
    kind: Literal["forward"] = "forward"
    packing: PackingOutcome
    packed_input_capture: PackedInputCaptureRef | None = None
    token_logprobs: tuple[TokenLogprobs, ...] = ()


class ForwardBackwardResult(ForwardResult):
    kind: Literal["forward_backward"] = "forward_backward"
    training: TrainingOutcome
    loss: NamedLossOutcome
    produced_gradient: bool = True

    @model_validator(mode="after")
    def _validate_gradient(self) -> "ForwardBackwardResult":
        expected = self.training.accepted_trainable_tokens > 0
        if self.produced_gradient != expected:
            raise ValueError("gradient production must match loss-bearing tokens")
        if not self.produced_gradient and (
            self.packed_input_capture is not None or self.token_logprobs
        ):
            raise ValueError("zero-work F/B cannot retain input or token outputs")
        return self


class OptimStepResult(OperationResult):
    kind: Literal["optim_step"] = "optim_step"
    contributing_forward_backward_operation_ids: tuple[str, ...] = Field(min_length=1)
    checkpoint: CheckpointRef


class SamplerWeightsResult(OperationResult):
    kind: Literal["save_sampler"] = "save_sampler"
    checkpoint: CheckpointRef
    lora: str = Field(min_length=1, max_length=MAX_CHECKPOINT_REFERENCE_LENGTH)
    external_lora: ExternalLoraReceipt | None = None

    @model_validator(mode="after")
    def _validate_external_lora(self) -> "SamplerWeightsResult":
        if (
            self.external_lora is not None
            and self.lora != self.external_lora.manifest.locator
        ):
            raise ValueError("external LoRA result differs from its manifest")
        return self


class SaveStateResult(OperationResult):
    kind: Literal["save_state"] = "save_state"
    checkpoint: CheckpointRef
    archive: CheckpointArchiveRef | None = None

    @model_validator(mode="after")
    def _validate_archive(self) -> "SaveStateResult":
        if self.archive is not None and (
            self.archive.checkpoint_id,
            self.archive.learner_version,
        ) != (self.checkpoint.checkpoint_id, self.checkpoint.learner_version):
            raise ValueError("durable archive differs from its checkpoint")
        return self


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
