from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from art.distributed.moe_route_store import (
    MoeRouteSlice,
    validate_moe_route_bindings,
    validate_moe_route_slices,
)
from art.distributed.object_store import MOE_ROUTE_OBJECT_FORMAT
from art.distributed.trajectory_store import TrajectoryGroupAnnotations
from art.training.contracts import (
    MAX_CHECKPOINT_REFERENCE_LENGTH,
    MAX_CONTROL_IDENTIFIER_LENGTH,
    Contract,
    ForwardRequest,
    LossConfig,
    OperationRef,
    RunCommand,
)

RL_GROUP_DATA_FORMAT = "art_trajectory_group_msgpack_v3"
SFT_DATA_FORMAT = "art_sft_batch_msgpack_v1"
TOKENIZED_DATA_FORMAT = "art_tokenized_batch_msgpack_v2"
OPERATION_RESULT_FORMAT = "art_operation_result_msgpack_v1"
MAX_OPERATION_RESULT_BYTES = 512 << 20
MAX_BASE_MODEL_LENGTH = 512
MAX_TARGET_MODULE_COUNT = 256
MAX_TARGET_MODULE_LENGTH = 255
MAX_RUN_METADATA_ITEMS = 64
MAX_RUN_METADATA_KEY_LENGTH = 128
MAX_RUN_METADATA_VALUE_LENGTH = 4096
MAX_CHECKPOINT_RETENTION_ITEMS = 512
DEFAULT_CHECKPOINT_PAGE_LIMIT = 100
MAX_CHECKPOINT_PAGE_LIMIT = 512
MAX_CHECKPOINT_CURSOR_LENGTH = 512
DEFAULT_CHECKPOINT_ALIAS_PAGE_LIMIT = 100
MAX_CHECKPOINT_ALIAS_PAGE_LIMIT = 512
MAX_CHECKPOINT_ALIASES_PER_VIEW = DEFAULT_CHECKPOINT_ALIAS_PAGE_LIMIT

TargetModule = Annotated[str, Field(min_length=1, max_length=MAX_TARGET_MODULE_LENGTH)]
RunMetadataKey = Annotated[
    str, Field(min_length=1, max_length=MAX_RUN_METADATA_KEY_LENGTH)
]
RunMetadataValue = Annotated[str, Field(max_length=MAX_RUN_METADATA_VALUE_LENGTH)]
ControlIdentifier = Annotated[
    str, Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)
]
CheckpointCursor = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_CHECKPOINT_CURSOR_LENGTH,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]


class TrainingDataRef(Contract):
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=1)
    format: Literal[
        "art_trajectory_group_msgpack_v3",
        "art_sft_batch_msgpack_v1",
        "art_tokenized_batch_msgpack_v2",
        "art_moe_route_bundle_v2",
    ]


class RemoteRouteObjectRef(Contract):
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=1)
    format: Literal["art_moe_route_bundle_v2"] = MOE_ROUTE_OBJECT_FORMAT
    transport: Literal["object_store", "command"] = "object_store"
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_transport(self) -> "RemoteRouteObjectRef":
        if (self.transport == "command") != (self.sha256 is not None):
            raise ValueError("command route objects require sha256 exclusively")
        return self

    def training_data_ref(self) -> TrainingDataRef:
        if self.transport != "command" or self.sha256 is None:
            raise ValueError("object-store routes are not training-data objects")
        return TrainingDataRef(
            object_id=self.object_id,
            sha256=self.sha256,
            byte_count=self.byte_count,
            format=MOE_ROUTE_OBJECT_FORMAT,
        )


RemoteRouteSlice = MoeRouteSlice


class RemoteRouteObject(Contract):
    ref: RemoteRouteObjectRef
    slices: tuple[RemoteRouteSlice, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_slices(self) -> "RemoteRouteObject":
        positions = [
            (
                value.trajectory_index,
                value.scope,
                value.scope_index,
                value.choice_index,
                value.segment_index,
            )
            for value in self.slices
        ]
        if len(positions) != len(set(positions)):
            raise ValueError("route object contains duplicate trajectory segments")
        if any(
            value.offset + value.byte_count > self.ref.byte_count
            for value in self.slices
        ):
            raise ValueError("route slice leaves its object bounds")
        validate_moe_route_slices(self.slices, self.ref.byte_count)
        return self


class RemoteRlGroupRef(Contract):
    data: TrainingDataRef
    routes: tuple[RemoteRouteObject, ...] = ()
    annotations: TrajectoryGroupAnnotations | None = None

    @model_validator(mode="after")
    def _validate_format(self) -> "RemoteRlGroupRef":
        if self.data.format != RL_GROUP_DATA_FORMAT:
            raise ValueError("RL group data has the wrong wire format")
        positions = [
            (
                value.trajectory_index,
                value.scope,
                value.scope_index,
                value.choice_index,
                value.segment_index,
            )
            for route in self.routes
            for value in route.slices
        ]
        if len(positions) != len(set(positions)):
            raise ValueError("RL group routes contain duplicate trajectory segments")
        validate_moe_route_bindings(
            value for route in self.routes for value in route.slices
        )
        return self


class RemoteRlBatchRef(Contract):
    kind: Literal["rl"] = "rl"
    groups: tuple[RemoteRlGroupRef, ...] = Field(min_length=1)
    min_source_version: int = Field(ge=0)
    max_source_version: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_versions(self) -> "RemoteRlBatchRef":
        if self.max_source_version < self.min_source_version:
            raise ValueError("max_source_version must be >= min_source_version")
        return self


class RemoteSftBatchRef(Contract):
    kind: Literal["sft"] = "sft"
    data: TrainingDataRef

    @model_validator(mode="after")
    def _validate_format(self) -> "RemoteSftBatchRef":
        if self.data.format != SFT_DATA_FORMAT:
            raise ValueError("SFT data has the wrong wire format")
        return self


class RemoteTokenizedBatchRef(Contract):
    kind: Literal["tokenized"] = "tokenized"
    data: TrainingDataRef

    @model_validator(mode="after")
    def _validate_format(self) -> "RemoteTokenizedBatchRef":
        if self.data.format != TOKENIZED_DATA_FORMAT:
            raise ValueError("tokenized data has the wrong wire format")
        return self


RemoteTrainingBatchRef = Annotated[
    RemoteRlBatchRef | RemoteSftBatchRef | RemoteTokenizedBatchRef,
    Field(discriminator="kind"),
]


class OperationResultRef(Contract):
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=1, le=MAX_OPERATION_RESULT_BYTES)
    format: Literal["art_operation_result_msgpack_v1"] = OPERATION_RESULT_FORMAT


class RemoteForwardRequest(RunCommand):
    batch: RemoteTrainingBatchRef
    loss: LossConfig
    collect_packing_shapes: bool = False

    @classmethod
    def from_command(
        cls, command: ForwardRequest, batch: RemoteTrainingBatchRef
    ) -> "RemoteForwardRequest":
        if command.batch.kind != batch.kind:
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
        expected = {
            "sft": {"cross_entropy"},
            "rl": {
                "cispo",
                "ppo",
            },
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
        return self


def training_data_refs(batch: RemoteTrainingBatchRef) -> tuple[TrainingDataRef, ...]:
    if isinstance(batch, RemoteRlBatchRef):
        return tuple(group.data for group in batch.groups) + tuple(
            route.ref.training_data_ref()
            for group in batch.groups
            for route in group.routes
            if route.ref.transport == "command"
        )
    return (batch.data,)


def route_object_refs(
    batch: RemoteTrainingBatchRef,
) -> tuple[RemoteRouteObjectRef, ...]:
    if not isinstance(batch, RemoteRlBatchRef):
        return ()
    refs: dict[str, RemoteRouteObjectRef] = {}
    for group in batch.groups:
        for route in group.routes:
            if route.ref.transport == "command":
                continue
            prior = refs.setdefault(route.ref.object_id, route.ref)
            if prior != route.ref:
                raise ValueError("route object identity changed within a batch")
    return tuple(refs.values())


def command_route_object_refs(
    batch: RemoteTrainingBatchRef,
) -> tuple[RemoteRouteObjectRef, ...]:
    if not isinstance(batch, RemoteRlBatchRef):
        return ()
    refs: dict[str, RemoteRouteObjectRef] = {}
    for group in batch.groups:
        for route in group.routes:
            if route.ref.transport != "command":
                continue
            prior = refs.setdefault(route.ref.object_id, route.ref)
            if prior != route.ref:
                raise ValueError("route object identity changed within a batch")
    return tuple(refs.values())


class AdapterSpec(Contract):
    rank: int = Field(ge=1, le=32)
    alpha: Literal[32] = 32
    target_modules: tuple[TargetModule, ...] = Field(
        min_length=1, max_length=MAX_TARGET_MODULE_COUNT
    )
    moe_parameterization: Literal["per_expert", "shared_outer"] = "per_expert"


class TrainingRunSpec(Contract):
    run_name: str = Field(min_length=1, max_length=255)
    base_model: str = Field(min_length=1, max_length=MAX_BASE_MODEL_LENGTH)
    adapter: AdapterSpec
    seed: int = 0
    dtype: Literal["bfloat16"] = "bfloat16"
    metadata: dict[RunMetadataKey, RunMetadataValue] = Field(
        default_factory=dict, max_length=MAX_RUN_METADATA_ITEMS
    )


class CreateTrainingRunRequest(Contract):
    spec: TrainingRunSpec
    checkpoint: str | None = Field(
        default=None, min_length=1, max_length=MAX_CHECKPOINT_REFERENCE_LENGTH
    )
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
    result: OperationResultRef | dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    event_cursor: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class CancelOperationRequest(Contract):
    request_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)


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
    request_id: str = Field(min_length=1, max_length=MAX_CONTROL_IDENTIFIER_LENGTH)


class CheckpointView(Contract):
    checkpoint_id: str
    revision: int = Field(ge=1)
    learner_version: int = Field(ge=0)
    aliases: tuple[ControlIdentifier, ...] = Field(
        default=(), max_length=MAX_CHECKPOINT_ALIASES_PER_VIEW
    )
    aliases_next_cursor: CheckpointCursor | None = None
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
    checkpoints: tuple[CheckpointView, ...] = Field(
        max_length=MAX_CHECKPOINT_PAGE_LIMIT
    )
    current_checkpoint_id: str | None
    next_cursor: CheckpointCursor | None


class CheckpointAliasPage(Contract):
    aliases: tuple[ControlIdentifier, ...] = Field(
        max_length=MAX_CHECKPOINT_ALIAS_PAGE_LIMIT
    )
    next_cursor: CheckpointCursor | None


class CheckpointRevision(Contract):
    checkpoint_id: ControlIdentifier
    revision: int = Field(ge=1)


class ApplyCheckpointRetentionRequest(Contract):
    observed: tuple[CheckpointRevision, ...] = Field(
        max_length=MAX_CHECKPOINT_RETENTION_ITEMS
    )
    retain_checkpoint_ids: tuple[ControlIdentifier, ...] = Field(
        default=(), max_length=MAX_CHECKPOINT_RETENTION_ITEMS
    )
    archive_checkpoint_ids: tuple[ControlIdentifier, ...] = Field(
        default=(), max_length=MAX_CHECKPOINT_RETENTION_ITEMS
    )

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
