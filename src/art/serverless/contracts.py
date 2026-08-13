from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from art.training.contracts import (
    Contract,
    ForwardRequest,
    LossConfig,
    OperationRef,
    RunCommand,
)

TRAINING_DATA_FORMAT = "art_training_batch_msgpack_v1"


class TrainingDataRef(Contract):
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=1)
    format: Literal["art_training_batch_msgpack_v1"] = TRAINING_DATA_FORMAT
    batch_kind: Literal["rl", "sft"]


class RemoteForwardRequest(RunCommand):
    batch: TrainingDataRef
    loss: LossConfig
    collect_packing_shapes: bool = False

    @classmethod
    def from_command(
        cls, command: ForwardRequest, batch: TrainingDataRef
    ) -> "RemoteForwardRequest":
        if command.batch.kind != batch.batch_kind:
            raise ValueError("training data kind differs from the forward command")
        return cls(
            run_id=command.run_id,
            request_id=command.request_id,
            sequence_id=command.sequence_id,
            batch=batch,
            loss=command.loss,
            collect_packing_shapes=command.collect_packing_shapes,
        )

    @model_validator(mode="after")
    def _validate_loss(self) -> "RemoteForwardRequest":
        expected = (
            {"cross_entropy"}
            if self.batch.batch_kind == "sft"
            else {
                "cispo",
                "ppo",
            }
        )
        if self.loss.name not in expected:
            raise ValueError(
                f"{self.batch.batch_kind} batches require one of {sorted(expected)}, "
                f"got {self.loss.name!r}"
            )
        return self


class AdapterSpec(Contract):
    rank: int = Field(ge=1, le=32)
    alpha: Literal[32] = 32
    target_modules: tuple[str, ...] = Field(min_length=1)


class TrainingRunSpec(Contract):
    run_name: str = Field(min_length=1, max_length=255)
    base_model: str = Field(min_length=1)
    adapter: AdapterSpec
    seed: int = 0
    dtype: Literal["bfloat16"] = "bfloat16"
    packing_contract_version: str = Field(min_length=1)
    art_version: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class CreateTrainingRunRequest(Contract):
    spec: TrainingRunSpec
    checkpoint: str | None = None
    restore_optimizer: bool = False

    @model_validator(mode="after")
    def _validate_checkpoint(self) -> "CreateTrainingRunRequest":
        if self.restore_optimizer and self.checkpoint is None:
            raise ValueError("restore_optimizer requires checkpoint")
        return self


RunStatus = Literal["open", "closing", "closed", "failed"]
OperationStatus = Literal[
    "admitted",
    "packing",
    "ready",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]


class TrainingRunView(Contract):
    run_id: str
    spec: TrainingRunSpec
    checkpoint: str | None = None
    restore_optimizer: bool = False
    status: RunStatus
    next_sequence_id: int = Field(ge=0)
    projected_learner_version: int = Field(ge=0)
    committed_learner_version: int = Field(ge=0)
    slot_id: str | None = None
    created_at: datetime
    updated_at: datetime


class OperationView(Contract):
    ref: OperationRef
    status: OperationStatus
    contributing_forward_backward_operation_ids: tuple[str, ...] = ()
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    event_cursor: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class CancelOperationRequest(Contract):
    request_id: str = Field(min_length=1)


class RunEvent(Contract):
    cursor: int = Field(ge=1)
    run_id: str
    operation_id: str | None = None
    event: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EventPage(Contract):
    events: tuple[RunEvent, ...]
    next_cursor: int = Field(ge=0)


class CloseRunRequest(Contract):
    request_id: str = Field(min_length=1)


class TrainingCapabilities(Contract):
    command_contract_version: str
    packing_contract_versions: tuple[str, ...]
    supported_losses: tuple[str, ...]
    supported_dtypes: tuple[str, ...]
    max_lora_rank: int = Field(ge=1)


class CheckpointView(Contract):
    checkpoint_id: str
    revision: int = Field(ge=1)
    learner_version: int = Field(ge=0)
    aliases: tuple[str, ...] = ()
    has_optimizer: bool
    state: Literal["ready", "deleting"]
    adapter_bytes: int = Field(ge=1)
    optimizer_bytes: int | None = Field(default=None, ge=1)
    expires_at: datetime | None = None
    archive_ref: str | None = None
    local_available: bool = True
    storage_error: str | None = None
    created_at: datetime


class CheckpointPage(Contract):
    checkpoints: tuple[CheckpointView, ...]


class CheckpointRevision(Contract):
    checkpoint_id: str
    revision: int = Field(ge=1)


class ApplyCheckpointRetentionRequest(Contract):
    observed: tuple[CheckpointRevision, ...]
    retain_checkpoint_ids: tuple[str, ...] = ()
    archive_checkpoint_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_ids(self) -> "ApplyCheckpointRetentionRequest":
        observed = [item.checkpoint_id for item in self.observed]
        if len(observed) != len(set(observed)):
            raise ValueError("observed checkpoint ids must be unique")
        selected = self.retain_checkpoint_ids + self.archive_checkpoint_ids
        if len(self.retain_checkpoint_ids) != len(set(self.retain_checkpoint_ids)):
            raise ValueError("retained checkpoint ids must be unique")
        if len(self.archive_checkpoint_ids) != len(set(self.archive_checkpoint_ids)):
            raise ValueError("archived checkpoint ids must be unique")
        if not set(selected).issubset(observed):
            raise ValueError("retention selections must be observed checkpoints")
        return self


class SetCheckpointTtlRequest(Contract):
    ttl_seconds: int | None = Field(default=None, ge=1)


class DeleteCheckpointResult(Contract):
    checkpoint_id: str
    state: Literal["deleting"] = "deleting"
