from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
import hashlib
import sys
from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    PrivateAttr,
    field_validator,
    model_validator,
)

from art.distributed.moe_route_store import (
    MoeRouteGroupPayload,
    MoeRouteObjectBatchTransfer,
)
from art.distributed.trajectory_store import (
    TrajectoryGroupAnnotations,
    TrajectoryGroupBundle,
)
from art.pipeline_tuner.config import PackedGroupShape
from art.trajectories import Trajectory

from .tokenized import (
    MAX_TOKENIZED_LOGPROB_VALUES,
    TokenizedDatum,
    TokenizedLossName,
    tokenized_result_value_count,
    validate_tokenized_loss_values,
)

COMMAND_CONTRACT_VERSION = "art_training_commands_v1"
PACKING_CONTRACT_VERSION = "art_prefix_tree_v1"
MAX_CONTROL_IDENTIFIER_LENGTH = 255
MAX_CHECKPOINT_REFERENCE_LENGTH = 2048
MAX_MODEL_ALIAS_LENGTH = 255


def operation_generation_id(operation_id: str, learner_version: int) -> str:
    if not operation_id or learner_version < 1:
        raise ValueError("learner transition identity is invalid")
    return (
        f"step-{learner_version:08d}-"
        f"{hashlib.sha256(operation_id.encode()).hexdigest()[:32]}"
    )


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunCommand(Contract):
    run_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    request_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
    sequence_id: int = Field(ge=0)


class RlTrajectoryBatch(Contract):
    kind: Literal["rl"] = "rl"
    groups: tuple[TrajectoryGroupBundle, ...] = Field(min_length=1)
    min_source_version: int = Field(ge=0)
    max_source_version: int = Field(ge=0)
    _local_groups: tuple[Any, ...] | None = PrivateAttr(default=None)
    _local_packed_batch: Any | None = PrivateAttr(default=None)
    _local_moe_route_groups: tuple[MoeRouteGroupPayload, ...] = PrivateAttr(default=())
    _local_moe_route_object_transfer: MoeRouteObjectBatchTransfer | None = PrivateAttr(
        default=None
    )
    _local_group_annotations: tuple[TrajectoryGroupAnnotations | None, ...] = (
        PrivateAttr(default=())
    )

    @model_validator(mode="after")
    def _validate_source_versions(self) -> "RlTrajectoryBatch":
        if self.max_source_version < self.min_source_version:
            raise ValueError("max_source_version must be >= min_source_version")
        return self

    @classmethod
    def from_groups(
        cls,
        groups: Sequence[Any],
        *,
        default_source_version: int,
        local_packed_batch: Any | None = None,
    ) -> "RlTrajectoryBatch":
        local_groups = tuple(groups)
        versions = [
            version
            for group in local_groups
            for trajectory in group.trajectories
            for version in (
                trajectory.initial_policy_version,
                trajectory.final_policy_version,
            )
            if version is not None
        ]
        batch = cls(
            groups=tuple(
                TrajectoryGroupBundle.from_group(group) for group in local_groups
            ),
            min_source_version=min(versions, default=default_source_version),
            max_source_version=max(versions, default=default_source_version),
        )
        object.__setattr__(batch, "_local_groups", local_groups)
        object.__setattr__(batch, "_local_packed_batch", local_packed_batch)
        return batch

    @classmethod
    def from_group_bundles(
        cls,
        bundles: Sequence[TrajectoryGroupBundle],
        *,
        min_source_version: int,
        max_source_version: int,
        groups: Sequence[Any] | None = None,
        moe_route_groups: Sequence[MoeRouteGroupPayload] = (),
        moe_route_object_transfer: MoeRouteObjectBatchTransfer | None = None,
        group_annotations: Sequence[TrajectoryGroupAnnotations | None] = (),
    ) -> "RlTrajectoryBatch":
        if groups is not None and len(bundles) != len(groups):
            raise ValueError(
                "trajectory bundles and materialized groups are not aligned"
            )
        if moe_route_groups and len(bundles) != len(moe_route_groups):
            raise ValueError("trajectory bundles and route groups are not aligned")
        if moe_route_object_transfer is not None and len(bundles) != len(
            moe_route_object_transfer.groups
        ):
            raise ValueError("trajectory bundles and object routes are not aligned")
        if group_annotations and len(bundles) != len(group_annotations):
            raise ValueError("trajectory bundles and annotations are not aligned")
        batch = cls(
            groups=tuple(bundles),
            min_source_version=min_source_version,
            max_source_version=max_source_version,
        )
        if groups is not None:
            object.__setattr__(batch, "_local_groups", tuple(groups))
        object.__setattr__(batch, "_local_moe_route_groups", tuple(moe_route_groups))
        object.__setattr__(
            batch, "_local_moe_route_object_transfer", moe_route_object_transfer
        )
        object.__setattr__(batch, "_local_group_annotations", tuple(group_annotations))
        return batch

    def require_local_groups(self) -> tuple[Any, ...]:
        if self._local_groups is None:
            raise RuntimeError(
                "local Megatron request has no in-process trajectory groups"
            )
        return self._local_groups

    def require_local_packed_batch(self) -> Any:
        if self._local_packed_batch is None:
            raise RuntimeError("local Megatron request has no packed batch lease")
        return self._local_packed_batch

    def local_moe_route_groups(self) -> tuple[MoeRouteGroupPayload, ...]:
        return self._local_moe_route_groups

    def local_moe_route_object_transfer(self) -> MoeRouteObjectBatchTransfer | None:
        return self._local_moe_route_object_transfer

    def local_group_annotations(
        self,
    ) -> tuple[TrajectoryGroupAnnotations | None, ...]:
        return self._local_group_annotations


class SupervisedTrajectoryBatch(Contract):
    kind: Literal["sft"] = "sft"
    trajectories: tuple[Trajectory, ...] = Field(min_length=1)
    assistant_turns: Literal["all", "last"] = "all"


class TokenizedTrainingBatch(Contract):
    kind: Literal["tokenized"] = "tokenized"
    datums: tuple[TokenizedDatum, ...] = Field(min_length=1)
    _encoded_payload: bytes | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_result_size(self) -> "TokenizedTrainingBatch":
        values = tokenized_result_value_count(self.datums)
        if values > MAX_TOKENIZED_LOGPROB_VALUES:
            raise ValueError(
                "tokenized result exceeds the configured value limit: "
                f"{values} > {MAX_TOKENIZED_LOGPROB_VALUES}"
            )
        routed = [datum.moe_routes is not None for datum in self.datums]
        if any(routed) and not all(routed):
            raise ValueError("tokenized batch must provide MoE routes for every datum")
        return self

    def encoded_payload(self) -> bytes | None:
        return self._encoded_payload

    def remember_encoded_payload(self, payload: bytes) -> None:
        object.__setattr__(self, "_encoded_payload", payload)


TrainingBatch = Annotated[
    RlTrajectoryBatch | SupervisedTrajectoryBatch | TokenizedTrainingBatch,
    Field(discriminator="kind"),
]


class LossConfig(Contract):
    name: TokenizedLossName
    normalize_advantages: bool = True
    values: dict[str, FiniteFloat | int | bool | str | None] = Field(
        default_factory=dict
    )


class ForwardRequest(RunCommand):
    batch: TrainingBatch
    loss: LossConfig
    collect_packing_shapes: bool = False

    @model_validator(mode="after")
    def _validate_loss(self) -> "ForwardRequest":
        expected = {
            "sft": {"cross_entropy"},
            "rl": {"cispo", "ppo"},
            "tokenized": {
                "cross_entropy",
                "importance_sampling",
                "ppo",
                "cispo",
            },
        }[self.batch.kind]
        if self.loss.name not in expected:
            raise ValueError(
                f"{self.batch.kind} batches require one of {sorted(expected)}, "
                f"got {self.loss.name!r}"
            )
        if isinstance(self.batch, TokenizedTrainingBatch):
            validate_tokenized_loss_values(self.loss.name, self.loss.values)
            for datum in self.batch.datums:
                datum.validate_for_loss(self.loss.name)
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
        default=None, min_length=1, max_length=MAX_MODEL_ALIAS_LENGTH
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


class TokenLogprobs(Contract):
    """Rectangular selected-token logprobs without per-value Python objects."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    shape: tuple[int, ...] = Field(min_length=1, max_length=2)
    data: bytes

    @model_validator(mode="after")
    def _validate_buffer(self) -> "TokenLogprobs":
        if any(value < 0 for value in self.shape):
            raise ValueError("token logprob dimensions cannot be negative")
        if len(self.shape) == 2 and self.shape[1] < 1:
            raise ValueError("token logprob candidate width must be positive")
        if self.value_count > MAX_TOKENIZED_LOGPROB_VALUES:
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
    def from_values(cls, values: object) -> "TokenLogprobs":
        if isinstance(values, cls):
            return values
        if isinstance(values, Mapping):
            return cls.model_validate(values)
        if not isinstance(values, Sequence) or isinstance(values, str | bytes):
            raise TypeError("token logprobs must be a flat or rectangular sequence")
        rows = tuple(cast(Sequence[Any], values))
        nested = (
            bool(rows)
            and isinstance(rows[0], Sequence)
            and not isinstance(rows[0], str | bytes)
        )
        if nested:
            matrix = tuple(tuple(cast(Sequence[Any], row)) for row in rows)
            width = len(matrix[0])
            if width < 1 or any(len(row) != width for row in matrix):
                raise ValueError("token logprobs must be rectangular")
            shape = (len(matrix), width)
            flat = (item for row in matrix for item in row)
        else:
            shape = (len(rows),)
            flat = iter(rows)
        packed = array("f", (float(value) for value in flat))
        if sys.byteorder != "little":
            packed.byteswap()
        return cls(shape=shape, data=packed.tobytes())

    def to_list(self) -> list[float]:
        values = array("f")
        values.frombytes(self.data)
        if sys.byteorder != "little":
            values.byteswap()
        return values.tolist()


class LossFnOutput(Contract):
    token_logprobs: TokenLogprobs
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("token_logprobs", mode="before")
    @classmethod
    def _pack_logprobs(cls, value: object) -> TokenLogprobs:
        return TokenLogprobs.from_values(value)


class OperationResult(Contract):
    operation_id: str = Field(min_length=1)


class ForwardResult(OperationResult):
    packing: PackingOutcome
    loss_fn_outputs: tuple[LossFnOutput, ...]
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_output_size(self) -> "ForwardResult":
        values = sum(
            output.token_logprobs.value_count for output in self.loss_fn_outputs
        )
        if values > MAX_TOKENIZED_LOGPROB_VALUES:
            raise ValueError(
                "forward result exceeds the configured logprob value limit"
            )
        return self


class ForwardBackwardResult(ForwardResult):
    pass


class OptimStepResult(OperationResult):
    contributing_forward_backward_operation_ids: tuple[str, ...] = Field(min_length=1)
    metrics: dict[str, float] = Field(default_factory=dict)


class SamplerWeightsResult(OperationResult):
    checkpoint: CheckpointRef
    lora: str = Field(min_length=1)
    training_session_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    lora_bytes: int = Field(gt=0)
    publication_metrics: dict[str, float] = Field(default_factory=dict)


class SaveStateResult(OperationResult):
    checkpoint: CheckpointRef
    lora: str = Field(min_length=1)
    training_session_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    lora_bytes: int = Field(gt=0)
    optimizer_state: str = Field(min_length=1)
    optimizer_bytes: int = Field(gt=0)
    metrics: dict[str, float] = Field(default_factory=dict)


class LoadStateResult(OperationResult):
    checkpoint: CheckpointRef
    lora: str = Field(min_length=1)
    training_session_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    lora_bytes: int = Field(gt=0)
    optimizer_restored: bool
