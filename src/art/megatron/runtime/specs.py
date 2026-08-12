from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from art.distributed.adapter_transport import AdapterTransferTarget
from art.distributed.data_plane import PackedBatchRef
from art.distributed.specs import NixlTransportSpec, TrainerMeshSpec
from art.megatron.optimizer_state import OptimizerAdapter
from art.training.contracts import AdamConfig
from art.types import TrainConfig, TrainSFTConfig

from .weight_transfer import MergedWeightTransferSpec


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HybridEpRuntimeSpec(_Spec):
    ranks_per_nvlink_domain: int = Field(ge=1)
    run_id: str = Field(min_length=1)
    nixl_transport: NixlTransportSpec | None = None

    @property
    def multinode(self) -> bool:
        return self.nixl_transport is not None


class TrainerRuntimeSpec(_Spec):
    art_revision: str = Field(min_length=1)
    model_identifier: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    model_initialization: Literal["pretrained", "random"] = "pretrained"
    cache_root: str | None = Field(default=None, min_length=1)
    model_support_key: str = Field(min_length=1)
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
    allow_unvalidated_arch: bool = False
    enable_moe_routing_replay: bool = False
    streaming_weight_offload: bool = False
    offload_between_jobs: bool = False
    random_state: int | None = None
    hybrid_ep: HybridEpRuntimeSpec | None = None
    snapshot_pool_capacity: int = Field(default=2, ge=1, le=4)

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
    initial_event_timeout_s: float | None = Field(default=None, gt=0)
    event_timeout_s: float = Field(default=300.0, gt=0)
    shutdown_timeout_s: float = Field(default=240.0, gt=0)


class CurrentTrainConfig(TrainConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RlForwardBackwardConfig(_Spec):
    kl_penalty_coef: float = Field(default=0.0, ge=0)
    kl_penalty_source: Literal["current_learner", "sample"] = "current_learner"
    grad_accumulation_sequences: int | None = Field(default=None, ge=1)


class CurrentSFTConfig(TrainSFTConfig):
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


class TrainerGeneration(_Spec):
    training_session_id: str = Field(min_length=1)
    policy_step: int = Field(ge=0)
    generation_id: str = Field(pattern=r"^step-\d{8,}-[0-9a-f]{32}$")
    adapter_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_generation_step(self) -> "TrainerGeneration":
        if int(self.generation_id.split("-", 2)[1]) != self.policy_step:
            raise ValueError("generation ID and policy step must match")
        return self


class DurableTrainOutput(_Spec):
    generation: TrainerGeneration
    staging_adapter_path: str = Field(min_length=1)
    optimizer_state_path: str = Field(min_length=1)


class _TrainerJobSpec(_Spec):
    job_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    training_session_id: str = Field(min_length=1)
    expected_learner_version: int = Field(ge=0)
    learner_version: int = Field(ge=1)
    source: TrainerGeneration
    output: DurableTrainOutput
    publication_targets: tuple[AdapterTransferTarget, ...] = ()
    merged_weight_transfer: MergedWeightTransferSpec | None = None

    @model_validator(mode="after")
    def _validate_versions(self) -> "_TrainerJobSpec":
        if self.learner_version != self.expected_learner_version + 1:
            raise ValueError(
                "learner_version must immediately follow expected_learner_version"
            )
        if (
            self.source.training_session_id != self.training_session_id
            or self.source.policy_step != self.expected_learner_version
        ):
            raise ValueError("source generation does not identify the expected learner")
        if (
            self.output.generation.training_session_id != self.training_session_id
            or self.output.generation.policy_step != self.learner_version
        ):
            raise ValueError("output generation does not identify the new learner")
        if self.source.generation_id == self.output.generation.generation_id:
            raise ValueError("source and output generation IDs must differ")
        if self.source.adapter_path == self.output.staging_adapter_path:
            raise ValueError("source adapter and output staging paths must differ")
        if self.output.generation.adapter_path == self.output.staging_adapter_path:
            raise ValueError("final and staging adapter paths must differ")
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
    def source_adapter_path(self) -> str:
        return self.source.adapter_path

    @property
    def output_adapter_path(self) -> str:
        return self.output.generation.adapter_path

    @property
    def output_generation_id(self) -> str:
        return self.output.generation.generation_id

    @property
    def optimizer_state_path(self) -> str:
        return self.output.optimizer_state_path


class TrainJobSpec(_TrainerJobSpec):
    kind: Literal["rl"] = "rl"
    batch: PackedBatchRef
    config: CurrentTrainConfig
    experimental_config: ExperimentalTrainConfig = ExperimentalTrainConfig()

    @model_validator(mode="after")
    def _validate_batch_version(self) -> "TrainJobSpec":
        if self.batch.max_source_version > self.expected_learner_version:
            raise ValueError(
                "batch source policy version cannot be newer than the learner"
            )
        return self


class ForwardBackwardJobSpec(_Spec):
    """One admitted RL F/B contribution against a fixed learner parent."""

    operation_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence_id: int = Field(ge=0)
    training_session_id: str = Field(min_length=1)
    expected_learner_version: int = Field(ge=0)
    source: TrainerGeneration
    optimizer_state_path: str = Field(min_length=1)
    batch: PackedBatchRef
    config: RlForwardBackwardConfig = RlForwardBackwardConfig()
    experimental_config: ExperimentalTrainConfig = ExperimentalTrainConfig()

    @model_validator(mode="after")
    def _validate_source(self) -> "ForwardBackwardJobSpec":
        if (
            self.source.training_session_id != self.training_session_id
            or self.source.policy_step != self.expected_learner_version
        ):
            raise ValueError("source generation does not identify the F/B parent")
        if self.batch.max_source_version > self.expected_learner_version:
            raise ValueError(
                "batch source policy version cannot be newer than the learner"
            )
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)

    @property
    def source_policy_step(self) -> int:
        return self.expected_learner_version

    @property
    def source_adapter_path(self) -> str:
        return self.source.adapter_path


class OptimizerJobSpec(_Spec):
    """Seal exact F/B contributions into one learner transition."""

    operation_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence_id: int = Field(ge=0)
    training_session_id: str = Field(min_length=1)
    expected_learner_version: int = Field(ge=0)
    learner_version: int = Field(ge=1)
    contributing_forward_backward_operation_ids: tuple[str, ...] = Field(min_length=1)
    optimizer: AdamConfig

    @model_validator(mode="after")
    def _validate_transition(self) -> "OptimizerJobSpec":
        if self.learner_version != self.expected_learner_version + 1:
            raise ValueError("optimizer learner version must advance exactly one step")
        if len(set(self.contributing_forward_backward_operation_ids)) != len(
            self.contributing_forward_backward_operation_ids
        ):
            raise ValueError("optimizer contribution IDs must be unique")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


class GenerationSnapshotJobSpec(_Spec):
    """Stage one immutable learner generation, optionally with optimizer state."""

    operation_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence_id: int = Field(ge=0)
    training_session_id: str = Field(min_length=1)
    learner_version: int = Field(ge=0)
    generation: TrainerGeneration
    optimizer_state_path: str = Field(min_length=1)
    staging_adapter_path: str | None = Field(default=None, min_length=1)
    existing_adapter: OptimizerAdapter | None = None
    publication_targets: tuple[AdapterTransferTarget, ...] = ()
    save_optimizer: bool = False

    @model_validator(mode="after")
    def _validate_snapshot(self) -> "GenerationSnapshotJobSpec":
        if (
            self.generation.training_session_id != self.training_session_id
            or self.generation.policy_step != self.learner_version
        ):
            raise ValueError("snapshot generation does not identify the learner")
        creating_adapter = self.staging_adapter_path is not None
        if creating_adapter == (self.existing_adapter is not None):
            raise ValueError(
                "snapshot requires exactly one new or existing adapter source"
            )
        if creating_adapter:
            if self.staging_adapter_path == self.generation.adapter_path:
                raise ValueError("final and staging adapter paths must differ")
        else:
            adapter = self.existing_adapter
            assert adapter is not None
            if (
                adapter.training_session_id,
                adapter.step,
                adapter.generation_id,
                adapter.identity,
            ) != (
                self.generation.training_session_id,
                self.generation.policy_step,
                self.generation.generation_id,
                self.generation.adapter_path,
            ):
                raise ValueError("existing adapter does not match snapshot generation")
            if not self.save_optimizer:
                raise ValueError("existing-adapter snapshot must add optimizer state")
            if self.publication_targets:
                raise ValueError(
                    "existing adapter cannot be transferred as a new snapshot"
                )
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


class SFTJobSpec(_TrainerJobSpec):
    kind: Literal["sft"] = "sft"
    batch_id: str = Field(min_length=1)
    num_batches: int = Field(ge=1)
    config: CurrentSFTConfig
    weight_decay: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _validate_batch_size(self) -> "SFTJobSpec":
        if not isinstance(self.config.batch_size, int):
            raise ValueError("typed SFT jobs require a resolved integer batch size")
        return self


TrainerJobSpec: TypeAlias = Annotated[
    TrainJobSpec | SFTJobSpec,
    Field(discriminator="kind"),
]
TRAIN_JOB_ADAPTER = TypeAdapter(TrainerJobSpec)


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
