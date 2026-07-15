from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from art.distributed.specs import TrainerMeshSpec
from art.megatron.runtime.jobs import MergedWeightTransferSpec
from art.types import TrainConfig


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainerRuntimeSpec(_Spec):
    art_revision: str = Field(min_length=1)
    model_identifier: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    handler_name: str = Field(min_length=1)
    lora_rank: int = Field(ge=1)
    lora_alpha: float = Field(default=32.0, gt=0)
    lora_target_modules: tuple[str, ...]
    dtype: Literal["bfloat16", "float16", "float32"]
    trainer_mesh: TrainerMeshSpec
    packed_sequence_length: int = Field(ge=1)
    compile_enabled: bool
    compile_fingerprint: str = Field(min_length=1)
    optimizer_layout_fingerprint: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_lora_targets(self) -> "TrainerRuntimeSpec":
        if self.lora_alpha != 32.0:
            raise ValueError("current Megatron LoRA semantics require lora_alpha=32")
        if not self.lora_target_modules:
            raise ValueError("lora_target_modules must not be empty")
        if len(set(self.lora_target_modules)) != len(self.lora_target_modules):
            raise ValueError("lora_target_modules must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


class TrainingRunSpec(_Spec):
    run_id: str = Field(min_length=1)
    runtime_fingerprint: str = Field(min_length=1)
    training_session_id: str = Field(min_length=1)
    initial_learner_version: int = Field(ge=0)
    initial_adapter_path: str = Field(min_length=1)
    optimizer_state_path: str = Field(min_length=1)


class TensorManifestEntry(_Spec):
    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    shape: tuple[int, ...]
    nbytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_shape(self) -> "TensorManifestEntry":
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be non-negative")
        return self


class PackedBatchRef(_Spec):
    batch_id: str = Field(min_length=1)
    lease_id: str = Field(min_length=1)
    format: Literal["art_packed_rl_v1"] = "art_packed_rl_v1"
    source_policy_version: int = Field(ge=0)
    num_sequences: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    tensors: tuple[TensorManifestEntry, ...]

    @model_validator(mode="after")
    def _validate_tensors(self) -> "PackedBatchRef":
        names = [tensor.name for tensor in self.tensors]
        if not names:
            raise ValueError("packed batch tensor manifest must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("packed batch tensor names must be unique")
        return self


class CurrentTrainConfig(TrainConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentalTrainConfig(_Spec):
    advantage_balance: float = 0.0
    allow_training_without_logprobs: bool | None = None
    epsilon: float | None = None
    epsilon_high: float | None = None
    importance_sampling_level: Literal[
        "token", "sequence", "average", "geometric_average"
    ] = "token"
    kimi_k2_tau: float | None = None
    kl_penalty_coef: float = Field(default=0.0, ge=0)
    kl_penalty_reference_step: int | None = Field(default=None, ge=0)
    kl_penalty_source: Literal["current_learner", "sample"] = "current_learner"
    kl_penalty_step_lag: int | None = Field(default=None, ge=0)
    kl_ref_adapter_path: str | None = None
    logprob_calculation_chunk_size: int | None = Field(default=None, ge=1)
    mask_prob_ratio: bool = False
    max_negative_advantage_importance_sampling_weight: float | None = None
    num_trajectories_learning_rate_multiplier_power: float | None = None
    packed_sequence_length: int | None = Field(default=None, ge=1)
    plot_tensors: bool | None = None
    ppo: bool = False
    precalculate_logprobs: bool = False
    scale_learning_rate_by_reward_std_dev: bool | None = None
    scale_rewards: bool = True
    truncated_importance_sampling: float | None = None
    moe_routing_replay_strict: bool = True


class DurableTrainOutput(_Spec):
    adapter_path: str = Field(min_length=1)
    optimizer_state_path: str = Field(min_length=1)


class TrainJobSpec(_Spec):
    job_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    training_session_id: str = Field(min_length=1)
    expected_learner_version: int = Field(ge=0)
    learner_version: int = Field(ge=1)
    batch: PackedBatchRef
    config: CurrentTrainConfig
    experimental_config: ExperimentalTrainConfig = ExperimentalTrainConfig()
    output: DurableTrainOutput
    merged_weight_transfer: MergedWeightTransferSpec | None = None

    @model_validator(mode="after")
    def _validate_versions(self) -> "TrainJobSpec":
        if self.learner_version != self.expected_learner_version + 1:
            raise ValueError(
                "learner_version must immediately follow expected_learner_version"
            )
        if self.batch.source_policy_version > self.expected_learner_version:
            raise ValueError(
                "batch source policy version cannot be newer than the learner"
            )
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    # These aliases keep the Megatron executor on the current train semantics.
    @property
    def step(self) -> int:
        return self.learner_version

    @property
    def source_policy_step(self) -> int:
        return self.expected_learner_version

    @property
    def lora_path(self) -> str:
        return self.output.adapter_path

    @property
    def optimizer_state_path(self) -> str:
        return self.output.optimizer_state_path


class _TrainEvent(_Spec):
    kind: str
    job_id: str
    run_id: str
    sequence: int = Field(ge=0)


class TrainAccepted(_TrainEvent):
    kind: Literal["accepted"] = "accepted"
    expected_learner_version: int = Field(ge=0)


class TrainProgress(_TrainEvent):
    kind: Literal["progress"] = "progress"
    step_index: int = Field(ge=0)
    num_steps: int = Field(ge=1)
    metrics: dict[str, float]


class AdapterReady(_TrainEvent):
    kind: Literal["adapter_ready"] = "adapter_ready"
    learner_version: int = Field(ge=1)
    adapter_path: str = Field(min_length=1)


class TrainCompleted(_TrainEvent):
    kind: Literal["completed"] = "completed"
    learner_version: int = Field(ge=1)
    metrics: dict[str, float] = Field(default_factory=dict)


class TrainFailed(_TrainEvent):
    kind: Literal["failed"] = "failed"
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    runtime_invalidated: bool


class TrainCancelled(_TrainEvent):
    kind: Literal["cancelled"] = "cancelled"
    reason: str = Field(min_length=1)
    runtime_invalidated: bool = True


TrainEvent: TypeAlias = Annotated[
    TrainAccepted
    | TrainProgress
    | AdapterReady
    | TrainCompleted
    | TrainFailed
    | TrainCancelled,
    Field(discriminator="kind"),
]
TRAIN_EVENT_ADAPTER = TypeAdapter(TrainEvent)
TERMINAL_EVENT_KINDS = frozenset({"completed", "failed", "cancelled"})


def is_terminal_event(event: TrainEvent) -> bool:
    return event.kind in TERMINAL_EVENT_KINDS


def validate_event_stream(events: Sequence[TrainEvent]) -> None:
    if not events:
        raise ValueError("train event stream must not be empty")
    if not isinstance(events[0], TrainAccepted):
        raise ValueError("train event stream must begin with accepted")
    if [event.sequence for event in events] != list(range(len(events))):
        raise ValueError("train event sequence must be contiguous from zero")
    terminals = [event for event in events if is_terminal_event(event)]
    if len(terminals) != 1 or events[-1] is not terminals[0]:
        raise ValueError("train event stream must end with exactly one terminal event")
    identity = {(event.run_id, event.job_id) for event in events}
    if len(identity) != 1:
        raise ValueError("all train events must identify the same run and job")


def _fingerprint(value: BaseModel) -> str:
    payload = json.dumps(
        value.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()
