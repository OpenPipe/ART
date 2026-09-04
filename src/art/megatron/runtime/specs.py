from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from art.distributed.adapter_transport import AdapterTransferTarget
from art.distributed.data_plane import PackedBatchRef
from art.distributed.specs import NixlTransportSpec, TrainerMeshSpec
from art.types import TrainConfig, TrainSFTConfig


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
    compile_cache: bool = False
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
        if self.compile_cache and not self.compile_enabled:
            raise ValueError("compile_cache requires compile_enabled")
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


class CurrentSFTConfig(TrainSFTConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentalTrainConfig(_Spec):
    loss_type: Literal["grpo", "bnpo", "dr_grpo"] = "bnpo"
    max_completion_length: int | None = Field(default=None, ge=1)
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


class ResidentScoreJobSpec(_Spec):
    job_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    learner: TrainerGeneration
    batch: PackedBatchRef
    global_grad_accumulation_sequences: int = Field(ge=1)
    top_k: int = Field(default=20, ge=1, le=1024)

    @model_validator(mode="after")
    def _validate_batch_generation(self) -> "ResidentScoreJobSpec":
        if not (
            self.batch.min_source_version
            == self.batch.max_source_version
            == self.learner.policy_step
        ):
            raise ValueError(
                "resident score batch must come exclusively from the requested learner"
            )
        return self


class PackedTokenScore(_Spec):
    sample_index: int = Field(ge=0)
    logit_index: int = Field(ge=0)
    target_token_id: int = Field(ge=0)
    target_logprob: float
    top_token_ids: tuple[int, ...]
    top_logprobs: tuple[float, ...]

    @model_validator(mode="after")
    def _validate_score(self) -> "PackedTokenScore":
        if not math.isfinite(self.target_logprob) or any(
            not math.isfinite(value) for value in self.top_logprobs
        ):
            raise ValueError("resident token scores must be finite")
        if not self.top_token_ids or len(self.top_token_ids) != len(self.top_logprobs):
            raise ValueError("resident top-k token IDs and logprobs must align")
        if any(token_id < 0 for token_id in self.top_token_ids):
            raise ValueError("resident top-k token IDs must be non-negative")
        if len(set(self.top_token_ids)) != len(self.top_token_ids):
            raise ValueError("resident top-k token IDs must be unique")
        if any(
            left < right
            for left, right in zip(
                self.top_logprobs, self.top_logprobs[1:], strict=False
            )
        ):
            raise ValueError("resident top-k logprobs must be descending")
        return self


def _validate_score_records(scores: tuple[PackedTokenScore, ...], top_k: int) -> None:
    keys = [(score.sample_index, score.logit_index) for score in scores]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("resident token scores must have unique sorted coordinates")
    if any(len(score.top_token_ids) != top_k for score in scores):
        raise ValueError("resident token score width does not match top_k")


class ResidentScoreShard(_Spec):
    rank: int = Field(ge=0)
    job_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    learner: TrainerGeneration
    batch_id: str = Field(min_length=1)
    batch_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(ge=1)
    expected_score_count: int = Field(ge=1)
    routing_replay_packed_tokens: int = Field(ge=0)
    scores: tuple[PackedTokenScore, ...]

    @model_validator(mode="after")
    def _validate_scores(self) -> "ResidentScoreShard":
        _validate_score_records(self.scores, self.top_k)
        if len(self.scores) > self.expected_score_count:
            raise ValueError("resident score shard exceeds the packed target count")
        return self


class ResidentScoreResult(_Spec):
    job_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    learner: TrainerGeneration
    batch_id: str = Field(min_length=1)
    batch_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    ranks: tuple[int, ...]
    top_k: int = Field(ge=1)
    expected_score_count: int = Field(ge=1)
    routing_replay_packed_tokens: int = Field(ge=0)
    scores: tuple[PackedTokenScore, ...]

    @model_validator(mode="after")
    def _validate_result(self) -> "ResidentScoreResult":
        if not self.ranks or len(self.ranks) != len(set(self.ranks)):
            raise ValueError("resident score result ranks must be unique and nonempty")
        _validate_score_records(self.scores, self.top_k)
        if len(self.scores) != self.expected_score_count:
            raise ValueError("resident score result does not cover every packed target")
        return self


class ResidentLoraInspectionSpec(_Spec):
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    learner: TrainerGeneration
    target_modules: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_targets(self) -> "ResidentLoraInspectionSpec":
        if not self.target_modules or len(self.target_modules) != len(
            set(self.target_modules)
        ):
            raise ValueError("resident LoRA target modules must be unique and nonempty")
        return self


class ResidentLoraExport(_Spec):
    base_name: str = Field(min_length=1)
    adapter_keys: tuple[str | None, ...]

    @model_validator(mode="after")
    def _validate_keys(self) -> "ResidentLoraExport":
        if not self.adapter_keys or len(self.adapter_keys) != len(
            set(self.adapter_keys)
        ):
            raise ValueError("resident LoRA export keys must be unique and nonempty")
        return self


class ResidentLoraRankSummary(_Spec):
    rank: int = Field(ge=0)
    module_count: int = Field(ge=0)
    trainable_parameter_count: int = Field(ge=0)
    trainable_numel: int = Field(ge=0)


class ResidentLoraInspectionShard(_Spec):
    rank: int = Field(ge=0)
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    learner: TrainerGeneration
    target_modules: tuple[str, ...]
    module_count: int = Field(ge=0)
    wrapped_adapter_prefixes: tuple[str, ...]
    exports: tuple[ResidentLoraExport, ...]
    trainable_lora_parameter_names: tuple[str, ...]
    unexpected_trainable_parameter_names: tuple[str, ...]
    trainable_numel: int = Field(ge=0)


class ResidentLoraInspectionResult(_Spec):
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    learner: TrainerGeneration
    target_modules: tuple[str, ...]
    rank_summaries: tuple[ResidentLoraRankSummary, ...]
    wrapped_adapter_prefixes: tuple[str, ...]
    exports: tuple[ResidentLoraExport, ...]
    trainable_lora_parameter_names: tuple[str, ...]
    unexpected_trainable_parameter_names: tuple[str, ...]


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
