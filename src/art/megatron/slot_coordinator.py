from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from art.distributed.art_runtime import ArtRuntime
from art.training import (
    CommandAdmission,
    CommandExecutionUsage,
    ForwardBackwardRequest,
    ForwardRequest,
    LoadStateRequest,
    OperationExecutionError,
    OperationExecutionOutcome,
    OperationFailureCode,
    OperationRef,
    OperationResultType,
    OperationWorker,
    OptimStepRequest,
    PackedInputCaptureRef,
    RunCommand,
    SaveStateRequest,
    SaveWeightsForSamplerRequest,
    SupervisedTrajectoryBatch,
    TrainingInputObjectRef,
    TrainingInputResolver,
)

from .operation_handler import (
    MegatronArtifactResourcePlan,
    MegatronCheckpointOperations,
    MegatronOperationConfig,
    MegatronOperationHandler,
    MegatronPairedPublisher,
    MegatronSamplerPublicationReceipt,
)
from .route_retention import (
    RouteBundleOwnershipProvider,
    RouteBundleOwnershipTransfer,
)
from .runtime.portable_snapshot import (
    PortableSnapshotArchive,
    PortableSnapshotExportReceipt,
    PortableSnapshotInstallReceipt,
    PortableSnapshotLoadReceipt,
)
from .runtime.publication import TrainerRankPublication
from .runtime.residency import (
    ResidencyCapacityUnavailable,
    ResidencyL1ReloadReceipt,
    ResidencyTier,
    ResidencyWorkingSetTooLarge,
)
from .runtime.specs import (
    CommandPublicationSpec,
    TrainerCommandRunState,
    TrainerGeneration,
    TrainingRunSpec,
)

SlotComponent = Literal["weights", "optimizer", "accumulator"]
_SLOT_COMPONENTS: tuple[SlotComponent, ...] = (
    "weights",
    "optimizer",
    "accumulator",
)
_RESIDENCY_TIERS: tuple[ResidencyTier, ...] = (
    "l1_gpu",
    "l2_cpu",
    "l3_nvme",
)


class MegatronResidencyTopologySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank_count: int = Field(ge=1)
    data_parallel_size: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)
    context_parallel_size: int = Field(ge=1)
    expert_parallel_size: int = Field(ge=1)
    pipeline_parallel_size: int = Field(ge=1)
    virtual_pipeline_parallel_size: int | None = Field(default=None, ge=1)
    expert_tensor_parallel_size: int = Field(ge=1)


class MegatronResidencyComponentSummary(BaseModel):
    """Bounded aggregate for one physical run-state component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    component: SlotComponent
    required_for_operation: bool
    observed_rank_count: int = Field(ge=0)
    l1_ready_rank_count: int = Field(ge=0)
    l1_ready_bytes: int = Field(ge=0)
    l2_present_rank_count: int = Field(ge=0)
    l3_present_rank_count: int = Field(ge=0)
    all_ranks_l1_ready: bool
    l1_reload_rank_count: int = Field(ge=0)
    l1_reload_bytes: int = Field(ge=0)
    l1_reload_sources: tuple[Literal["l2_cpu", "l3_nvme"], ...] = Field(max_length=2)
    all_ranks_reloaded_after_eviction: bool


class MegatronResidencyTierSummary(BaseModel):
    """Aggregate copy and byte facts for one bounded residency tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: ResidencyTier
    copy_count: int = Field(ge=0)
    ready_copy_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    ready_bytes: int = Field(ge=0)


class MegatronOperationResidencySummary(BaseModel):
    """Service-persistable aggregate backed by private ART rank evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["megatron-operation-residency.v2"] = (
        "megatron-operation-residency.v2"
    )
    run_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1, max_length=64)
    operation_kind: Literal[
        "forward",
        "forward_backward",
        "optim_step",
        "save_sampler",
        "save_state",
        "load_state",
    ]
    operation_sequence: int = Field(ge=0)
    learner_parent_version: int = Field(ge=0)
    reserved_output_learner_version: int | None = Field(default=None, ge=1)
    learner_version: int = Field(ge=0)
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    topology: MegatronResidencyTopologySummary
    requested_components: tuple[SlotComponent, ...] = Field(max_length=3)
    components: tuple[MegatronResidencyComponentSummary, ...] = Field(
        min_length=3, max_length=3
    )
    tiers: tuple[MegatronResidencyTierSummary, ...] = Field(min_length=3, max_length=3)
    observed_rank_count: int = Field(ge=0)
    all_ranks_reported: bool
    all_requested_components_l1_ready: bool
    l1_ready_bytes: int = Field(ge=0)
    residency_wait_s: FiniteFloat = Field(ge=0)
    rank_max_prepare_s: FiniteFloat = Field(ge=0)
    detailed_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_components(self) -> "MegatronOperationResidencySummary":
        if len(set(self.requested_components)) != len(self.requested_components):
            raise ValueError("requested residency components must be unique")
        if tuple(item.component for item in self.components) != _SLOT_COMPONENTS:
            raise ValueError("residency component aggregates must be canonical")
        if tuple(item.tier for item in self.tiers) != _RESIDENCY_TIERS:
            raise ValueError("residency tier aggregates must be canonical")
        requested = set(self.requested_components)
        if any(
            item.required_for_operation != (item.component in requested)
            for item in self.components
        ):
            raise ValueError("residency component requirement flags changed")
        return self


class _ResidencyRankCopyEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: ResidencyTier
    byte_count: int = Field(ge=1)
    ready: bool


class _ResidencyRankComponentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component: SlotComponent
    generation_id: str = Field(min_length=1)
    required_for_operation: bool
    byte_count: int = Field(ge=0)
    tiers: tuple[ResidencyTier, ...] = Field(max_length=3)
    l1_ready: bool
    copies: tuple[_ResidencyRankCopyEvidence, ...] = Field(max_length=3)
    last_l1_reload: ResidencyL1ReloadReceipt | None = None
    reloaded_for_operation: bool = False

    @model_validator(mode="after")
    def _validate_copies(self) -> "_ResidencyRankComponentEvidence":
        copy_tiers = tuple(copy.tier for copy in self.copies)
        if len(set(copy_tiers)) != len(copy_tiers) or copy_tiers != self.tiers:
            raise ValueError("residency copy tiers changed identity")
        l1 = next((copy for copy in self.copies if copy.tier == "l1_gpu"), None)
        if self.l1_ready != (l1 is not None) or self.byte_count != (
            0 if l1 is None else l1.byte_count
        ):
            raise ValueError("legacy L1 residency facts disagree with copy evidence")
        if self.last_l1_reload is not None and (
            l1 is None or self.last_l1_reload.byte_count != l1.byte_count
        ):
            raise ValueError("L1 reload evidence differs from the ready L1 copy")
        if self.reloaded_for_operation != (self.last_l1_reload is not None):
            raise ValueError("operation reload flag and exact receipt disagree")
        return self


class _ResidencyRankEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=0)
    run_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1, max_length=64)
    requested_components: tuple[SlotComponent, ...] = Field(max_length=3)
    components: tuple[_ResidencyRankComponentEvidence, ...] = Field(max_length=3)


class _DetailedResidencyEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1, max_length=64)
    run_id: str = Field(min_length=1)
    requested_components: tuple[SlotComponent, ...] = Field(max_length=3)
    learner_version: int = Field(ge=0)
    rank_evidence: tuple[_ResidencyRankEvidence, ...]
    wait_s: FiniteFloat = Field(ge=0)
    rank_max_s: FiniteFloat = Field(ge=0)


class MegatronSlotScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    deficit_quantum_tokens: int = Field(default=32_768, ge=1)
    optimizer_turn_limit: int = Field(default=2, ge=1, le=16)
    max_ready_commands: int = Field(default=256, ge=1, le=4096)
    max_pending_results: int = Field(default=8, ge=1, le=64)


class MegatronSlotResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    source: TrainerGeneration
    optimizer_state_path: str = Field(min_length=1)
    components: tuple[SlotComponent, ...]


class MegatronMigrationContribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1)
    input_object: TrainingInputObjectRef


class MegatronMigrationFence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fence_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    generation: TrainerGeneration
    optimizer_state_path: str = Field(min_length=1)
    next_operation_sequence: int = Field(ge=0)
    open_contributions: tuple[MegatronMigrationContribution, ...] = Field(max_length=64)


@dataclass(frozen=True, slots=True)
class MegatronMigrationReplay:
    request: RunCommand
    admission: CommandAdmission


class MegatronSlotResourceManager(Protocol):
    """Off-loop residency planner; rank execution remains authoritative."""

    def prefetch(self, request: MegatronSlotResourceRequest) -> None: ...

    async def ensure(self, request: MegatronSlotResourceRequest) -> dict[str, Any]: ...

    async def release(self, request: MegatronSlotResourceRequest) -> None: ...

    async def release_run(self, run_id: str) -> None: ...


class MegatronOperationEvidenceSink(Protocol):
    """Idempotently retain a bounded summary before ART releases replay state."""

    def retain_residency_evidence(
        self, evidence: MegatronOperationResidencySummary
    ) -> None: ...


class InlineMegatronSlotResources:
    """Use the rank-local command ensure path without speculative readiness."""

    def prefetch(self, request: MegatronSlotResourceRequest) -> None:
        del request

    async def ensure(self, request: MegatronSlotResourceRequest) -> dict[str, Any]:
        del request
        return {}

    async def release(self, request: MegatronSlotResourceRequest) -> None:
        del request

    async def release_run(self, run_id: str) -> None:
        del run_id


class TrainerMegatronSlotResources:
    """Drive rank-local L1-L3 preparation ahead of serialized GPU commands."""

    def __init__(self, trainer: "_SharedTrainer") -> None:
        self._trainer = trainer
        self._prefetches: dict[str, tuple[str, asyncio.Task[dict[str, Any]]]] = {}
        # One executing command plus one command preparing ahead of it.
        self._admission_slots = asyncio.Semaphore(2)
        self._prepare_lock = asyncio.Lock()
        self._capacity_changed = asyncio.Condition()
        self._capacity_epoch = 0
        self._admitted: set[str] = set()

    def prefetch(self, request: MegatronSlotResourceRequest) -> None:
        if request.operation_id in self._prefetches:
            return
        task = asyncio.create_task(
            self._admit_when_available(request),
            name=f"megatron-residency-prefetch-{request.operation_id}",
        )
        self._prefetches[request.operation_id] = (request.run_id, task)
        task.add_done_callback(_consume_task_exception)

    async def ensure(self, request: MegatronSlotResourceRequest) -> dict[str, Any]:
        pending = self._prefetches.pop(request.operation_id, None)
        if pending is None:
            return await self._admit_when_available(request)
        return await pending[1]

    async def release(self, request: MegatronSlotResourceRequest) -> None:
        pending = self._prefetches.pop(request.operation_id, None)
        if pending is not None:
            pending[1].cancel()
            await asyncio.gather(pending[1], return_exceptions=True)
        await self._trainer.release_command_run_residency_admission(
            request.operation_id
        )
        self._release_admission_slot(request.operation_id)
        await self._notify_capacity_changed()

    async def release_run(self, run_id: str) -> None:
        pending = tuple(
            (operation_id, task)
            for operation_id, (pending_run_id, task) in self._prefetches.items()
            if pending_run_id == run_id
        )
        for operation_id, task in pending:
            self._prefetches.pop(operation_id, None)
            task.cancel()
        if pending:
            await asyncio.gather(
                *(task for _operation_id, task in pending), return_exceptions=True
            )
            for operation_id, _task in pending:
                await self._trainer.release_command_run_residency_admission(
                    operation_id
                )
                self._release_admission_slot(operation_id)
            await self._notify_capacity_changed()

    async def _admit_when_available(
        self, request: MegatronSlotResourceRequest
    ) -> dict[str, Any]:
        acquired = False
        try:
            await self._trainer.prefetch_command_run_residency(
                request.run_id,
                request.components,
                request.source.policy_step,
            )
            await self._admission_slots.acquire()
            acquired = True
            async with self._prepare_lock:
                while True:
                    observed_epoch = self._capacity_epoch
                    try:
                        evidence = await self._trainer.admit_command_run_residency(
                            request.operation_id,
                            request.run_id,
                            request.components,
                            request.source.policy_step,
                        )
                    except ResidencyWorkingSetTooLarge:
                        raise
                    except ResidencyCapacityUnavailable:
                        async with self._capacity_changed:
                            await self._capacity_changed.wait_for(
                                lambda: self._capacity_epoch != observed_epoch
                            )
                    else:
                        self._admitted.add(request.operation_id)
                        return evidence
        except BaseException:
            if acquired:
                self._admission_slots.release()
            raise

    def _release_admission_slot(self, operation_id: str) -> None:
        if operation_id in self._admitted:
            self._admitted.remove(operation_id)
            self._admission_slots.release()

    async def _notify_capacity_changed(self) -> None:
        async with self._capacity_changed:
            self._capacity_epoch += 1
            self._capacity_changed.notify_all()


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


class _SharedTrainer(Protocol):
    runtime_spec: Any

    async def forward(self, job: Any, batch: Any) -> dict[str, Any]: ...

    async def start_forward(self, job: Any, batch: Any) -> Any: ...

    async def forward_backward(self, job: Any, batch: Any) -> dict[str, Any]: ...

    async def start_forward_backward(self, job: Any, batch: Any) -> Any: ...

    async def sft_forward(self, job: Any, batch: Any) -> dict[str, Any]: ...

    async def start_sft_forward(self, job: Any, batch: Any) -> Any: ...

    async def sft_forward_backward(self, job: Any, batch: Any) -> dict[str, Any]: ...

    async def start_sft_forward_backward(self, job: Any, batch: Any) -> Any: ...

    async def prefetch_command_run_residency(
        self,
        run_id: str,
        components: tuple[str, ...],
        learner_version: int,
    ) -> dict[str, Any]: ...

    async def admit_command_run_residency(
        self,
        operation_id: str,
        run_id: str,
        components: tuple[str, ...],
        learner_version: int,
    ) -> dict[str, Any]: ...

    async def release_command_run_residency_admission(
        self, operation_id: str
    ) -> None: ...

    async def optim_step(self, job: Any) -> dict[str, Any]: ...

    async def publish_command_generation(
        self, spec: CommandPublicationSpec
    ) -> tuple[tuple[TrainerRankPublication, ...], dict[str, float]]: ...

    async def publish_external_lora(
        self,
        target: Any,
        sink_spec: Any,
        *,
        source_topology: str,
    ) -> tuple[Any, Any, dict[str, float]]: ...

    async def register_command_run(
        self, run_spec: TrainingRunSpec
    ) -> PortableSnapshotInstallReceipt | None: ...

    async def command_run_state(self, run_id: str) -> TrainerCommandRunState: ...

    async def export_command_run_checkpoint(
        self,
        run_id: str,
        generation: TrainerGeneration,
        export_id: str,
    ) -> PortableSnapshotExportReceipt: ...

    async def install_command_run_checkpoint(
        self,
        operation: OperationRef,
        generation: TrainerGeneration,
        archive: PortableSnapshotArchive,
        *,
        restore_optimizer: bool,
    ) -> PortableSnapshotLoadReceipt: ...

    async def commit_prepared_command_run_checkpoint(
        self, operation: OperationRef
    ) -> PortableSnapshotLoadReceipt: ...

    async def discard_prepared_command_run_checkpoint(
        self, operation: OperationRef
    ) -> None: ...

    async def record_control_command(
        self,
        operation: OperationRef,
        learner_version: int,
    ) -> None: ...

    async def record_no_work_command(
        self,
        operation: OperationRef,
        learner_version: int,
    ) -> None: ...

    async def release_command_run_for_migration(self, run_id: str) -> None: ...

    async def drain_command_run(self, run_id: str) -> None: ...


@dataclass(slots=True)
class MegatronSlotRun:
    run_id: str
    worker: OperationWorker
    portable_install: PortableSnapshotInstallReceipt | None = None


@dataclass(slots=True)
class _RunState:
    handler: MegatronOperationHandler
    worker: _SlotOperationWorker
    draining: bool = False
    maintenance: bool = False
    preparing: int = 0
    l4_hydration_operation_id: str | None = None
    worker_calls: int = 0
    settling: int = 0
    migration_fence_id: str | None = None
    migration_fence: MegatronMigrationFence | None = None
    migration_restore_id: str | None = None
    activated_restore_id: str | None = None
    activated_migration_fence: MegatronMigrationFence | None = None
    migration_replay: tuple[MegatronMigrationReplay, ...] | None = None
    migration_replay_outcomes: tuple[OperationExecutionOutcome, ...] | None = None
    migration_replay_error: str | None = None
    migration_replaying: bool = False
    portable_archive_sha256: str | None = None
    portable_install: PortableSnapshotInstallReceipt | None = None


@dataclass(slots=True)
class _ReadyCommand:
    run_id: str
    request: RunCommand
    operation: OperationRef
    contributions: tuple[str, ...]
    cost: int
    future: asyncio.Future[OperationResultType]
    resources: MegatronSlotResourceRequest

    @property
    def optimizer(self) -> bool:
        return isinstance(self.request, OptimStepRequest)


class _ScheduledRunHandler:
    def __init__(self, slot: "MegatronSlotCoordinator", run_id: str) -> None:
        self._slot = slot
        self._run_id = run_id

    async def __call__(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributions: tuple[str, ...],
    ) -> OperationResultType:
        if operation.run_id != self._run_id:
            raise ValueError("operation belongs to another slot run")
        return await self._slot._execute(request, operation, contributions)


class _SlotOperationWorker(OperationWorker):
    def __init__(
        self,
        slot: "MegatronSlotCoordinator",
        run_id: str,
        handler: _ScheduledRunHandler,
        *,
        max_retained_operations: int,
    ) -> None:
        super().__init__(handler, max_retained_operations=max_retained_operations)
        self._slot = slot
        self._run_id = run_id

    async def execute(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributing_forward_backward_operation_ids: tuple[str, ...] = (),
    ) -> OperationExecutionOutcome:
        await self._slot._enter_worker_call(self._run_id, self)
        try:
            return await super().execute(
                request,
                operation,
                contributing_forward_backward_operation_ids,
            )
        finally:
            await self._slot._leave_worker_call(self._run_id, self)

    async def execute_replay(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributing_forward_backward_operation_ids: tuple[str, ...],
    ) -> OperationExecutionOutcome:
        return await super().execute(
            request,
            operation,
            contributing_forward_backward_operation_ids,
        )

    def retire(self, operation_id: str) -> None:
        self._slot._retire_operation(self._run_id, self, operation_id)
        super().retire(operation_id)


class MegatronSlotCoordinator:
    """Fairly multiplex logical runs over one persistent trainer allocation."""

    def __init__(
        self,
        runtime: ArtRuntime,
        trainer: _SharedTrainer,
        *,
        resources: MegatronSlotResourceManager | None = None,
        schedule: MegatronSlotScheduleConfig | None = None,
        publisher: MegatronPairedPublisher | None = None,
        route_ownership: RouteBundleOwnershipProvider | None = None,
        input_resolver: TrainingInputResolver | None = None,
        operation_evidence_sink: MegatronOperationEvidenceSink | None = None,
        command_timeout_s: float = 300.0,
    ) -> None:
        if command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be positive")
        self.runtime = runtime
        self.trainer = trainer
        self.resources = resources or TrainerMegatronSlotResources(trainer)
        self.schedule = schedule or MegatronSlotScheduleConfig()
        self.publisher = publisher
        self.route_ownership = route_ownership
        self.input_resolver = input_resolver
        self.operation_evidence_sink = operation_evidence_sink
        self.command_timeout_s = command_timeout_s
        self._runs: dict[str, _RunState] = {}
        self._ready: list[_ReadyCommand] = []
        self._deficit: dict[str, int] = {}
        self._order: list[str] = []
        self._cursor = 0
        self._optimizer_turn_remaining = 0
        self._condition = asyncio.Condition()
        self._active_run_id: str | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._settlement_slots = asyncio.BoundedSemaphore(
            self.schedule.max_pending_results
        )
        self._settlement_tasks: set[asyncio.Task[None]] = set()
        self._released_migration_fences: dict[str, MegatronMigrationFence] = {}
        self._resumed_migration_fences: dict[tuple[str, str], None] = {}
        self._aborted_migration_restores: dict[
            tuple[str, str], tuple[MegatronOperationConfig, str | None]
        ] = {}
        self._residency_evidence: dict[tuple[str, str], dict[str, Any]] = {}
        self._residency_summaries: dict[
            tuple[str, str], MegatronOperationResidencySummary
        ] = {}
        self._portable_exports: dict[
            tuple[str, str], PortableSnapshotExportReceipt
        ] = {}
        self._registering_runs: dict[
            str, tuple[MegatronOperationConfig, str | None, str | None, bool]
        ] = {}
        self._closed = False

    async def register_run(
        self,
        config: MegatronOperationConfig,
        *,
        checkpoints: MegatronCheckpointOperations | None = None,
        max_retained_operations: int = 128,
        portable_archive: PortableSnapshotArchive | None = None,
        restore_optimizer: bool = True,
    ) -> MegatronSlotRun:
        return await self._register_run(
            config,
            checkpoints=checkpoints,
            max_retained_operations=max_retained_operations,
            restore_id=None,
            portable_archive=portable_archive,
            restore_optimizer=restore_optimizer,
        )

    async def _register_run(
        self,
        config: MegatronOperationConfig,
        *,
        checkpoints: MegatronCheckpointOperations | None,
        max_retained_operations: int,
        restore_id: str | None,
        portable_archive: PortableSnapshotArchive | None,
        restore_optimizer: bool = True,
    ) -> MegatronSlotRun:
        if config.source.training_session_id != config.training_session_id:
            raise ValueError("source generation belongs to another training session")
        archive_sha256 = (
            None if portable_archive is None else portable_archive.archive_sha256
        )
        runtime_spec = self.trainer.runtime_spec
        if config.adapter.rank > runtime_spec.lora_rank:
            raise ValueError("run LoRA rank exceeds the slot capability")
        if not set(config.adapter.target_modules).issubset(
            runtime_spec.lora_target_modules
        ):
            raise ValueError("run LoRA targets exceed the slot capability")
        run_spec = TrainingRunSpec(
            run_id=config.run_id,
            runtime_fingerprint=runtime_spec.fingerprint,
            training_session_id=config.training_session_id,
            initial_learner_version=config.source.policy_step,
            initial_generation_id=config.source.generation_id,
            initial_operation_sequence=config.initial_operation_sequence,
            lora_rank=config.adapter.rank,
            lora_target_modules=config.adapter.target_modules,
            initial_adapter_path=config.source.adapter_path,
            optimizer_state_path=config.optimizer_state_path,
            initial_portable_snapshot=portable_archive,
            initial_restore_optimizer=restore_optimizer,
            event_timeout_s=self.command_timeout_s,
        )
        registration_identity = (
            config,
            archive_sha256,
            restore_id,
            restore_optimizer,
        )
        aborted_key = None if restore_id is None else (config.run_id, restore_id)
        while True:
            async with self._condition:
                if self._closed:
                    raise RuntimeError("Megatron slot coordinator is closed")
                pending = self._registering_runs.get(config.run_id)
                if pending is not None:
                    if pending != registration_identity:
                        raise RuntimeError(
                            "run_id is registering with different trainer state"
                        )
                    await self._condition.wait()
                    continue
                aborted = (
                    None
                    if aborted_key is None
                    else self._aborted_migration_restores.get(aborted_key)
                )
                identity = (config, archive_sha256)
                if aborted is not None and aborted != identity:
                    raise RuntimeError(
                        "aborted migration restore configuration changed"
                    )
                prior = self._runs.get(config.run_id)
                if prior is not None:
                    if prior.draining or prior.migration_fence_id is not None:
                        raise RuntimeError("Megatron slot run is draining")
                    if prior.migration_restore_id != restore_id and not (
                        restore_id is not None
                        and prior.activated_restore_id == restore_id
                    ):
                        raise RuntimeError("run_id is already in another lifecycle")
                    if prior.handler.config != config:
                        raise RuntimeError(
                            "run_id was reused with different configuration"
                        )
                    if prior.portable_archive_sha256 != archive_sha256:
                        raise RuntimeError(
                            "run_id was reused with another portable archive"
                        )
                    if (
                        prior.portable_install is not None
                        and prior.portable_install.restore_optimizer
                        is not restore_optimizer
                    ):
                        raise RuntimeError(
                            "run_id was reused with another restore selection"
                        )
                    return MegatronSlotRun(
                        config.run_id, prior.worker, prior.portable_install
                    )
                self._registering_runs[config.run_id] = registration_identity
                break

        trainer_registered = False
        try:
            portable_install = await self.trainer.register_command_run(run_spec)
            trainer_registered = True
            if (portable_archive is None) != (portable_install is None):
                raise RuntimeError(
                    "trainer returned inconsistent portable install evidence"
                )
            if portable_install is not None:
                if portable_install.runtime_fingerprint != runtime_spec.fingerprint:
                    raise RuntimeError("trainer installed another runtime")
                assert portable_archive is not None
                portable_install.validate_archive(portable_archive)
            handler = MegatronOperationHandler(
                self.runtime,
                self.trainer,
                config,
                checkpoints=checkpoints,
                publisher=self.publisher,
                route_ownership=self.route_ownership,
                input_resolver=self.input_resolver,
            )
            scheduled = _ScheduledRunHandler(self, config.run_id)
            worker = _SlotOperationWorker(
                self,
                config.run_id,
                scheduled,
                max_retained_operations=max_retained_operations,
            )
            state = _RunState(
                handler=handler,
                worker=worker,
                migration_restore_id=restore_id,
                portable_archive_sha256=archive_sha256,
                portable_install=portable_install,
            )
            async with self._condition:
                if self._closed:
                    raise RuntimeError("Megatron slot coordinator closed during setup")
                self._runs[config.run_id] = state
                if aborted_key is not None:
                    self._aborted_migration_restores.pop(aborted_key, None)
                self._released_migration_fences.pop(config.run_id, None)
                self._deficit[config.run_id] = 0
                self._order.append(config.run_id)
                self._registering_runs.pop(config.run_id, None)
                if self._pump_task is None:
                    self._pump_task = asyncio.create_task(self._pump())
                self._condition.notify_all()
            return MegatronSlotRun(config.run_id, worker, portable_install)
        except BaseException as error:
            if trainer_registered:
                try:
                    await self.trainer.drain_command_run(config.run_id)
                except BaseException as cleanup_error:
                    error.add_note(
                        "run registration rollback also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            async with self._condition:
                self._registering_runs.pop(config.run_id, None)
                self._condition.notify_all()
            raise

    def resolve_run(self, run_id: str) -> MegatronSlotRun:
        state = self._runs.get(run_id)
        if (
            state is None
            or state.draining
            or state.migration_fence_id is not None
            or state.migration_restore_id is not None
        ):
            raise KeyError(f"Megatron slot run {run_id!r} is unavailable")
        return MegatronSlotRun(run_id, state.worker, state.portable_install)

    async def plan_artifacts(
        self, run_id: str, request: RunCommand
    ) -> MegatronArtifactResourcePlan:
        state = self._runs.get(run_id)
        if (
            state is None
            or state.draining
            or state.migration_fence_id is not None
            or state.migration_restore_id is not None
        ):
            raise KeyError(f"Megatron slot run {run_id!r} is unavailable")
        return await state.handler.plan_artifacts(request)

    async def export_run_checkpoint(
        self,
        operation: OperationRef,
    ) -> PortableSnapshotExportReceipt:
        """Commit exact canonical rank files for the active save-state command."""

        async with self._condition:
            state = self._runs.get(operation.run_id)
            if (
                operation.kind != "save_state"
                or state is None
                or state.draining
                or state.migration_fence_id is not None
                or state.migration_restore_id is not None
                or self._active_run_id != operation.run_id
            ):
                raise RuntimeError(
                    "checkpoint export is not the active save-state command"
                )
            generation = state.handler.generation
        receipt = await self.trainer.export_command_run_checkpoint(
            operation.run_id,
            generation,
            operation.operation_id,
        )
        exported = receipt.generation
        if receipt.export_id != operation.operation_id or (
            exported.training_session_id,
            exported.policy_step,
            exported.generation_id,
        ) != (
            generation.training_session_id,
            generation.policy_step,
            generation.generation_id,
        ):
            raise RuntimeError("portable export changed the active save-state identity")
        key = (operation.run_id, operation.operation_id)
        prior = self._portable_exports.setdefault(key, receipt)
        if prior != receipt:
            raise RuntimeError("portable export changed operation-keyed evidence")
        return receipt

    async def install_run_checkpoint(
        self,
        operation: OperationRef,
        generation: TrainerGeneration,
        archive: PortableSnapshotArchive,
        *,
        restore_optimizer: bool,
    ) -> PortableSnapshotLoadReceipt:
        """Prepare one authenticated checkpoint for an admitted load command."""

        async with self._condition:
            state = self._runs.get(operation.run_id)
            preparing = (
                state is not None
                and state.l4_hydration_operation_id == operation.operation_id
            )
            if (
                operation.kind != "load_state"
                or state is None
                or state.draining
                or state.migration_fence_id is not None
                or state.migration_restore_id is not None
                or not (preparing or self._active_run_id == operation.run_id)
                or state.handler.retained_contribution_inputs()
            ):
                raise RuntimeError(
                    "checkpoint load is not the admitted quiescent command"
                )
        return await self.trainer.install_command_run_checkpoint(
            operation,
            generation,
            archive,
            restore_optimizer=restore_optimizer,
        )

    def sampler_publication_receipt(
        self,
        run_id: str,
        operation_id: str,
    ) -> MegatronSamplerPublicationReceipt | None:
        """Return exact private publication evidence until operation retirement."""

        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"Megatron slot run {run_id!r} is unavailable")
        return state.handler.sampler_publication_receipt(operation_id)

    def portable_snapshot_receipt(
        self,
        run_id: str,
        operation_id: str,
    ) -> PortableSnapshotExportReceipt | None:
        """Return exact private save-state evidence until operation retirement."""

        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"Megatron slot run {run_id!r} is unavailable")
        return self._portable_exports.get((run_id, operation_id))

    async def acknowledge_operation(self, run_id: str, operation_id: str) -> None:
        """Release physical command inputs after an external durable acknowledgement."""

        state = self._runs.get(run_id)
        if state is None or state.draining:
            raise KeyError(f"Megatron slot run {run_id!r} is unavailable")
        await state.handler.acknowledge_operation(operation_id)

    def _retire_operation(
        self,
        run_id: str,
        worker: _SlotOperationWorker,
        operation_id: str,
    ) -> None:
        state = self._runs.get(run_id)
        if state is not None and state.worker is worker:
            summary = self._residency_summaries.get((run_id, operation_id))
            if summary is not None and self.operation_evidence_sink is not None:
                self.operation_evidence_sink.retain_residency_evidence(summary)
            state.handler.retire_operation(operation_id)
            self._residency_evidence.pop((run_id, operation_id), None)
            self._residency_summaries.pop((run_id, operation_id), None)
            self._portable_exports.pop((run_id, operation_id), None)

    def residency_evidence(
        self, run_id: str, operation_id: str
    ) -> dict[str, Any] | None:
        """Return exact rank readiness evidence until operation retirement."""

        return self._residency_evidence.get((run_id, operation_id))

    def residency_summary(
        self, run_id: str, operation_id: str
    ) -> MegatronOperationResidencySummary | None:
        """Return the bounded service-facing aggregate until operation retirement."""

        return self._residency_summaries.get((run_id, operation_id))

    async def _enter_worker_call(
        self,
        run_id: str,
        worker: _SlotOperationWorker,
    ) -> None:
        async with self._condition:
            while True:
                state = self._runs.get(run_id)
                if (
                    state is None
                    or state.worker is not worker
                    or state.draining
                    or state.migration_fence_id is not None
                    or state.migration_restore_id is not None
                    or self._closed
                ):
                    raise KeyError(f"Megatron slot run {run_id!r} is unavailable")
                if not state.maintenance:
                    break
                await self._condition.wait()
            state.worker_calls += 1

    async def _leave_worker_call(
        self,
        run_id: str,
        worker: _SlotOperationWorker,
    ) -> None:
        async with self._condition:
            state = self._runs.get(run_id)
            if state is None or state.worker is not worker:
                return
            state.worker_calls -= 1
            self._condition.notify_all()

    async def fence_and_quiesce_run(
        self,
        run_id: str,
        fence_id: str,
    ) -> MegatronMigrationFence:
        """Fence new work and resolve every command already entering the slot."""

        if not fence_id:
            raise ValueError("fence_id must not be empty")
        async with self._condition:
            if (run_id, fence_id) in self._resumed_migration_fences:
                raise RuntimeError("migration fence was already resumed")
            state = self._runs.get(run_id)
            if state is None or state.migration_restore_id is not None:
                raise KeyError(f"Megatron slot run {run_id!r} is unavailable")
            if state.draining:
                raise RuntimeError("Megatron slot run is draining")
            while state.maintenance:
                await self._condition.wait()
                if self._runs.get(run_id) is not state or state.draining:
                    raise RuntimeError("Megatron slot run changed while waiting")
            if state.migration_fence_id not in {None, fence_id}:
                raise RuntimeError("another migration fence owns this run")
            if state.migration_fence is not None:
                return state.migration_fence
            state.migration_fence_id = fence_id
            while (
                state.preparing
                or state.worker_calls
                or self._active_run_id == run_id
                or any(item.run_id == run_id for item in self._ready)
            ):
                await self._condition.wait()

        trainer_state = await self.trainer.command_run_state(run_id)
        retained = state.handler.durable_contribution_inputs()
        retained_ids = tuple(operation_id for operation_id, _ in retained)
        if (
            trainer_state.run_id != run_id
            or trainer_state.training_session_id
            != state.handler.config.training_session_id
            or trainer_state.learner_version != state.handler.generation.policy_step
            or trainer_state.open_forward_backward_operation_ids != retained_ids
        ):
            raise RuntimeError("trainer and command handler migration state differ")
        fence = MegatronMigrationFence(
            fence_id=fence_id,
            run_id=run_id,
            generation=state.handler.generation,
            optimizer_state_path=state.handler.optimizer_state_path,
            next_operation_sequence=trainer_state.next_operation_sequence,
            open_contributions=tuple(
                MegatronMigrationContribution(
                    operation_id=operation_id,
                    input_object=input_object,
                )
                for operation_id, input_object in retained
            ),
        )
        async with self._condition:
            if (
                self._runs.get(run_id) is not state
                or state.migration_fence_id != fence_id
                or state.draining
            ):
                raise RuntimeError("Megatron slot run changed while fencing")
            state.migration_fence = fence
            self._condition.notify_all()
            return fence

    async def release_retained_input(self, ref: TrainingInputObjectRef) -> None:
        """Release one local materialization after durable recovery coverage."""

        async with self._condition:
            state = self._runs.get(ref.run_id)
            if (
                state is None
                or state.draining
                or state.maintenance
                or state.migration_fence_id is not None
                or state.migration_restore_id is not None
            ):
                raise RuntimeError("Megatron slot run is unavailable for input release")
            state.maintenance = True
            try:
                while (
                    state.preparing
                    or state.worker_calls
                    or self._active_run_id == ref.run_id
                    or any(item.run_id == ref.run_id for item in self._ready)
                ):
                    await self._condition.wait()
            except BaseException:
                state.maintenance = False
                self._condition.notify_all()
                raise
        try:
            await state.handler.discard_input_object(ref)
        finally:
            async with self._condition:
                if self._runs.get(ref.run_id) is state:
                    state.maintenance = False
                self._condition.notify_all()

    async def resume_migration_source(self, run_id: str, fence_id: str) -> None:
        """Abort a migration before activation and reopen the unchanged source run."""

        async with self._condition:
            state = self._runs.get(run_id)
            if (
                state is None
                or state.migration_fence_id != fence_id
                or state.draining
                or state.preparing
            ):
                raise RuntimeError("migration fence is absent or changed")
            state.migration_fence = None
            state.migration_fence_id = None
            self._resumed_migration_fences[(run_id, fence_id)] = None
            self._bound_migration_tombstones(self._resumed_migration_fences)
            self._condition.notify_all()

    async def transfer_migration_route_ownership(
        self,
        fence: MegatronMigrationFence,
        *,
        transfer_id: str,
        target_owner_id: str,
    ) -> tuple[RouteBundleOwnershipTransfer, ...]:
        """Retain exact open-suffix routes for a migration target owner."""

        async with self._condition:
            state = self._runs.get(fence.run_id)
            if (
                state is None
                or state.draining
                or state.migration_fence != fence
                or state.migration_fence_id != fence.fence_id
                or state.preparing
            ):
                raise RuntimeError("migration fence is absent, changed, or active")
            state.preparing += 1
        transfers: list[RouteBundleOwnershipTransfer] = []
        try:
            for contribution in fence.open_contributions:
                handle = await state.handler.transfer_route_ownership(
                    contribution.input_object,
                    transfer_id=transfer_id,
                    target_owner_id=target_owner_id,
                )
                if handle is not None:
                    transfers.append(
                        RouteBundleOwnershipTransfer(
                            operation_id=contribution.operation_id,
                            input_object=contribution.input_object,
                            handle=handle,
                        )
                    )
            async with self._condition:
                if (
                    self._runs.get(fence.run_id) is not state
                    or state.migration_fence != fence
                ):
                    raise RuntimeError("migration source changed during route transfer")
            return tuple(transfers)
        except BaseException as error:
            provider = self.route_ownership
            if provider is not None:
                for transfer in reversed(transfers):
                    try:
                        await provider.release(transfer.handle)
                    except BaseException as cleanup_error:
                        error.add_note(
                            "route transfer rollback also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
            raise
        finally:
            async with self._condition:
                if self._runs.get(fence.run_id) is state:
                    state.preparing -= 1
                self._condition.notify_all()

    async def install_migration_run(
        self,
        config: MegatronOperationConfig,
        *,
        restore_id: str,
        checkpoints: MegatronCheckpointOperations | None = None,
        max_retained_operations: int = 128,
        portable_archive: PortableSnapshotArchive | None = None,
    ) -> MegatronSlotRun:
        """Install a hidden recovery head for exact replay before activation."""

        if not restore_id:
            raise ValueError("restore_id must not be empty")
        return await self._register_run(
            config,
            checkpoints=checkpoints,
            max_retained_operations=max_retained_operations,
            restore_id=restore_id,
            portable_archive=portable_archive,
            restore_optimizer=True,
        )

    async def replay_migration_operations(
        self,
        run_id: str,
        restore_id: str,
        operations: tuple[MegatronMigrationReplay, ...],
    ) -> tuple[OperationExecutionOutcome, ...]:
        """Replay one exact ordered suffix while the target remains hidden."""

        async with self._condition:
            state = self._runs.get(run_id)
            if (
                state is None
                or state.draining
                or state.migration_restore_id != restore_id
                or state.migration_replaying
            ):
                raise RuntimeError("migration restore is absent, changed, or active")
            if state.migration_replay is not None:
                if state.migration_replay != operations:
                    raise RuntimeError("migration replay identity changed")
                if state.migration_replaying:
                    raise RuntimeError("migration replay is already active")
                if state.migration_replay_outcomes is not None:
                    return state.migration_replay_outcomes
                raise RuntimeError(
                    "prior migration replay did not reach a terminal result: "
                    f"{state.migration_replay_error or 'unknown error'}"
                )
            state.migration_replay = operations
            state.migration_replaying = True
        outcomes: list[OperationExecutionOutcome] = []
        try:
            initial = await self.trainer.command_run_state(run_id)
            expected = _validate_migration_replay(initial, operations)
            for replay in operations:
                outcome = await state.worker.execute_replay(
                    replay.request,
                    replay.admission.ref,
                    replay.admission.contributing_forward_backward_operation_ids,
                )
                outcomes.append(outcome)
                if outcome.status == "failed":
                    break
            if len(outcomes) == len(operations) and all(
                outcome.status == "succeeded" for outcome in outcomes
            ):
                actual = await self.trainer.command_run_state(run_id)
                if actual != expected or (
                    state.handler.generation.policy_step != actual.learner_version
                ):
                    raise RuntimeError("migration replay produced the wrong run state")
            terminal = tuple(outcomes)
            async with self._condition:
                if self._runs.get(run_id) is not state:
                    raise RuntimeError("migration restore changed while replaying")
                state.migration_replay_outcomes = terminal
            return terminal
        except BaseException as error:
            async with self._condition:
                if self._runs.get(run_id) is state:
                    state.migration_replay_error = (
                        f"{type(error).__name__}: {str(error).strip()}"
                    )
            raise
        finally:
            async with self._condition:
                if self._runs.get(run_id) is state:
                    state.migration_replaying = False
                self._condition.notify_all()

    async def activate_migration_run(
        self,
        run_id: str,
        restore_id: str,
        expected: MegatronMigrationFence,
    ) -> MegatronSlotRun:
        """Expose a restored target only after durable service activation."""

        async with self._condition:
            state = self._runs.get(run_id)
            if state is None or state.draining:
                raise RuntimeError("migration restore is absent or changed")
            if state.activated_restore_id == restore_id:
                if state.activated_migration_fence != expected:
                    raise RuntimeError("activated migration source fence changed")
                return MegatronSlotRun(run_id, state.worker, state.portable_install)
            if (
                state.migration_restore_id != restore_id
                or state.migration_replaying
                or expected.run_id != run_id
                or state.migration_replay_outcomes is None
                or any(
                    outcome.status != "succeeded"
                    for outcome in state.migration_replay_outcomes
                )
            ):
                raise RuntimeError("migration restore is absent, changed, or active")
        actual = await self.trainer.command_run_state(run_id)
        retained = tuple(
            MegatronMigrationContribution(
                operation_id=operation_id,
                input_object=input_object,
            )
            for operation_id, input_object in state.handler.durable_contribution_inputs()
        )
        if (
            not _same_generation_identity(
                state.handler.generation,
                expected.generation,
            )
            or actual.learner_version != expected.generation.policy_step
            or actual.next_operation_sequence != expected.next_operation_sequence
            or actual.open_forward_backward_operation_ids
            != tuple(item.operation_id for item in retained)
            or retained != expected.open_contributions
        ):
            raise RuntimeError("restored target does not match the source fence")
        async with self._condition:
            if (
                self._runs.get(run_id) is not state
                or state.migration_restore_id != restore_id
                or state.migration_replaying
            ):
                raise RuntimeError("migration restore changed while activating")
            state.migration_restore_id = None
            state.activated_restore_id = restore_id
            state.activated_migration_fence = expected
            self._condition.notify_all()
            return MegatronSlotRun(run_id, state.worker, state.portable_install)

    async def abort_migration_run(self, run_id: str, restore_id: str) -> None:
        """Discard a hidden target after a failed or abandoned restore."""

        async with self._condition:
            state = self._runs.get(run_id)
            if state is None:
                if (run_id, restore_id) in self._aborted_migration_restores:
                    return
                raise RuntimeError("migration restore is absent or changed")
            if (
                state.migration_restore_id != restore_id
                or state.migration_replaying
                or state.draining
            ):
                raise RuntimeError("migration restore is absent, changed, or active")
            state.draining = True
        await self._release_migration_run(
            state,
            run_id,
            aborted_restore_id=restore_id,
        )

    async def release_migration_source(self, fence: MegatronMigrationFence) -> None:
        """Discard source-local state only after destination activation."""

        async with self._condition:
            state = self._runs.get(fence.run_id)
            if state is None:
                if self._released_migration_fences.get(fence.run_id) == fence:
                    return
                raise RuntimeError("migration fence is absent or changed")
            if (
                state.migration_fence_id != fence.fence_id
                or state.migration_fence != fence
                or state.draining
                or state.preparing
            ):
                raise RuntimeError("migration fence is absent, changed, or releasing")
            state.draining = True
        await self._release_migration_run(
            state,
            fence.run_id,
            released_fence=fence,
        )

    async def _release_migration_run(
        self,
        state: _RunState,
        run_id: str,
        *,
        released_fence: MegatronMigrationFence | None = None,
        aborted_restore_id: str | None = None,
    ) -> None:
        try:
            await self.trainer.release_command_run_for_migration(run_id)
            await state.handler.release_after_migration()
            await self.resources.release_run(run_id)
            async with self._condition:
                if self._runs.get(run_id) is not state:
                    raise RuntimeError("Megatron slot run changed while releasing")
                if released_fence is not None:
                    self._released_migration_fences[run_id] = released_fence
                    self._bound_migration_tombstones(self._released_migration_fences)
                if aborted_restore_id is not None:
                    self._aborted_migration_restores[(run_id, aborted_restore_id)] = (
                        state.handler.config,
                        state.portable_archive_sha256,
                    )
                    self._bound_migration_tombstones(self._aborted_migration_restores)
                self._runs.pop(run_id)
                self._deficit.pop(run_id, None)
                self._purge_run_evidence(run_id)
                if run_id in self._order:
                    self._order.remove(run_id)
                self._condition.notify_all()
        except BaseException:
            async with self._condition:
                if self._runs.get(run_id) is state:
                    state.draining = False
                    self._condition.notify_all()
            raise

    @staticmethod
    def _bound_migration_tombstones(values: dict[Any, Any]) -> None:
        while len(values) > 128:
            values.pop(next(iter(values)))

    def _purge_run_evidence(self, run_id: str) -> None:
        for values in (
            self._residency_evidence,
            self._residency_summaries,
            self._portable_exports,
        ):
            for key in tuple(values):
                if key[0] == run_id:
                    values.pop(key)

    async def drain_run(self, run_id: str) -> None:
        async with self._condition:
            state = self._runs.get(run_id)
            if state is None:
                return
            if state.migration_fence_id is not None:
                raise RuntimeError("cannot drain a migration-fenced run")
            if state.migration_restore_id is not None:
                raise RuntimeError("abort a restoring migration before draining")
            state.draining = True
            while (
                state.maintenance
                or state.preparing
                or state.worker_calls
                or self._active_run_id == run_id
                or any(item.run_id == run_id for item in self._ready)
            ):
                await self._condition.wait()
            if state.handler.retained_contribution_inputs():
                state.draining = False
                self._condition.notify_all()
                raise RuntimeError("cannot drain a run with open F/B contributions")
        await state.handler.aclose()
        await self.trainer.drain_command_run(run_id)
        await self.resources.release_run(run_id)
        async with self._condition:
            self._runs.pop(run_id, None)
            self._deficit.pop(run_id, None)
            self._purge_run_evidence(run_id)
            if run_id in self._order:
                self._order.remove(run_id)
            self._condition.notify_all()

    async def aclose(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._registering_runs:
                await self._condition.wait()
        for run_id in tuple(self._runs):
            await self.drain_run(run_id)
        if self._pump_task is not None:
            await self._pump_task
        if self._settlement_tasks:
            await asyncio.gather(*tuple(self._settlement_tasks))
        if self.publisher is not None:
            await self.publisher.aclose()

    async def _execute(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributions: tuple[str, ...],
    ) -> OperationResultType:
        async with self._condition:
            state = self._runs.get(operation.run_id)
            if (
                state is None
                or state.draining
                or state.migration_fence_id is not None
                or (
                    state.migration_restore_id is not None
                    and not state.migration_replaying
                )
                or self._closed
            ):
                raise _not_executed("cancelled", "Megatron slot run is draining")
            if isinstance(request, LoadStateRequest):
                if state.l4_hydration_operation_id is not None:
                    raise RuntimeError("another L4 hydration is preparing for this run")
                state.l4_hydration_operation_id = operation.operation_id
            state.preparing += 1
        capture: PackedInputCaptureRef | None = None
        resource_request: MegatronSlotResourceRequest | None = None
        try:
            if isinstance(request, LoadStateRequest):
                await state.handler.prefetch_load_state(request, operation)
            components = _components(request)
            raw_sft = isinstance(
                request, (ForwardRequest, ForwardBackwardRequest)
            ) and (isinstance(request.batch, SupervisedTrajectoryBatch))
            if not raw_sft:
                resource_request = MegatronSlotResourceRequest(
                    run_id=operation.run_id,
                    operation_id=operation.operation_id,
                    source=state.handler.generation,
                    optimizer_state_path=state.handler.optimizer_state_path,
                    components=components,
                )
                self.resources.prefetch(resource_request)
            cost = 1
            if isinstance(request, (ForwardRequest, ForwardBackwardRequest)):
                capture = await state.handler.prepare_input(request, operation)
                packing = await state.handler.packing_for(capture)
                await state.handler.prepare_cp_lookahead(request, operation, capture)
                request = request.model_copy(update={"batch": capture})
                cost = max(1, packing.physical_tokens)
                if raw_sft:
                    resource_request = MegatronSlotResourceRequest(
                        run_id=operation.run_id,
                        operation_id=operation.operation_id,
                        source=state.handler.generation,
                        optimizer_state_path=state.handler.optimizer_state_path,
                        components=(components if packing.loss_bearing_tokens else ()),
                    )
                    self.resources.prefetch(resource_request)
            assert resource_request is not None
            evidence = await self.resources.ensure(resource_request)
            evidence_key = (operation.run_id, operation.operation_id)
            self._residency_evidence[evidence_key] = evidence
            self._residency_summaries[evidence_key] = _summarize_residency_evidence(
                operation=operation,
                expected_generation=resource_request.source,
                runtime_spec=self.trainer.runtime_spec,
                evidence=evidence,
            )
            loop = asyncio.get_running_loop()
            future: asyncio.Future[OperationResultType] = loop.create_future()
            ready = _ReadyCommand(
                run_id=operation.run_id,
                request=request,
                operation=operation,
                contributions=contributions,
                cost=cost,
                future=future,
                resources=resource_request,
            )
            async with self._condition:
                if (
                    state.draining
                    or state.migration_fence_id is not None
                    or (
                        state.migration_restore_id is not None
                        and not state.migration_replaying
                    )
                    or self._closed
                ):
                    raise _not_executed("cancelled", "Megatron slot run is draining")
                if len(self._ready) >= self.schedule.max_ready_commands:
                    raise _not_executed(
                        "capacity_exhausted", "Megatron slot ready queue is full"
                    )
                self._ready.append(ready)
                self._condition.notify_all()
        except BaseException as error:
            if isinstance(request, LoadStateRequest):
                try:
                    await state.handler.discard_load_state(operation)
                except BaseException as cleanup_error:
                    error.add_note(
                        "L4 hydration cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            if resource_request is not None:
                try:
                    await self.resources.release(resource_request)
                except BaseException as cleanup_error:
                    error.add_note(
                        "residency cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            if capture is not None:
                try:
                    await state.handler.discard_prepared_input(capture)
                except BaseException as cleanup_error:
                    error.add_note(
                        "packed-input cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise
        finally:
            async with self._condition:
                state.preparing -= 1
                if state.l4_hydration_operation_id == operation.operation_id:
                    state.l4_hydration_operation_id = None
                self._condition.notify_all()
        return await _await_terminal(future)

    async def _pump(self) -> None:
        while True:
            async with self._condition:
                while not any(
                    not self._runs[item.run_id].settling for item in self._ready
                ):
                    if self._closed and not self._ready:
                        return
                    await self._condition.wait()
                command = self._select_ready()
                self._ready.remove(command)
                self._active_run_id = command.run_id
            settlement_slot = False
            try:
                state = self._runs[command.run_id]
                if isinstance(
                    command.request, (ForwardRequest, ForwardBackwardRequest)
                ):
                    await self._settlement_slots.acquire()
                    settlement_slot = True
                    launch = await state.handler.launch(
                        command.request,
                        command.operation,
                        command.contributions,
                    )
                    if launch is None:
                        raise RuntimeError("forward command returned no trainer launch")
                    async with self._condition:
                        state.settling += 1
                        self._condition.notify_all()
                    task = asyncio.create_task(
                        self._settle_command(command, state, launch.completion),
                        name=f"megatron-slot-result-{command.operation.operation_id}",
                    )
                    self._settlement_tasks.add(task)
                    task.add_done_callback(self._settlement_tasks.discard)
                    settlement_slot = False
                else:
                    result = await state.handler(
                        command.request,
                        command.operation,
                        command.contributions,
                    )
            except BaseException as error:
                if not command.future.done():
                    command.future.set_exception(error)
            else:
                if (
                    not isinstance(
                        command.request, (ForwardRequest, ForwardBackwardRequest)
                    )
                    and not command.future.done()
                ):
                    command.future.set_result(result)
            finally:
                try:
                    await self.resources.release(command.resources)
                except BaseException as cleanup_error:
                    if not command.future.done():
                        command.future.set_exception(cleanup_error)
                if settlement_slot:
                    self._settlement_slots.release()
                async with self._condition:
                    self._active_run_id = None
                    self._condition.notify_all()

    async def _settle_command(
        self,
        command: _ReadyCommand,
        state: _RunState,
        completion: asyncio.Future[OperationResultType],
    ) -> None:
        try:
            result = await asyncio.shield(completion)
        except BaseException as error:
            if not command.future.done():
                command.future.set_exception(error)
        else:
            if not command.future.done():
                command.future.set_result(result)
        finally:
            async with self._condition:
                state.settling -= 1
                self._settlement_slots.release()
                self._condition.notify_all()

    def _select_ready(self) -> _ReadyCommand:
        available = [
            item for item in self._ready if not self._runs[item.run_id].settling
        ]
        optimizers = [item for item in available if item.optimizer]
        forwards = [item for item in available if not item.optimizer]
        if optimizers and (self._optimizer_turn_remaining > 0 or not forwards):
            candidates = optimizers
            if self._optimizer_turn_remaining > 0:
                self._optimizer_turn_remaining -= 1
        else:
            candidates = forwards or optimizers
            if forwards:
                self._optimizer_turn_remaining = self.schedule.optimizer_turn_limit
        return self._deficit_select(candidates)

    def _deficit_select(self, candidates: list[_ReadyCommand]) -> _ReadyCommand:
        by_run = {item.run_id: item for item in candidates}
        ordered = self._order[self._cursor :] + self._order[: self._cursor]
        eligible = [run_id for run_id in ordered if run_id in by_run]
        if not eligible:
            raise RuntimeError("ready command has no registered run")
        cycles = min(
            max(
                1,
                math.ceil(
                    (by_run[run_id].cost - self._deficit[run_id])
                    / self.schedule.deficit_quantum_tokens
                ),
            )
            for run_id in eligible
        )
        for run_id in eligible:
            self._deficit[run_id] += cycles * self.schedule.deficit_quantum_tokens
        selected = next(
            run_id
            for run_id in eligible
            if self._deficit[run_id] >= by_run[run_id].cost
        )
        self._deficit[selected] -= by_run[selected].cost
        self._cursor = (self._order.index(selected) + 1) % len(self._order)
        return by_run[selected]


def _components(request: RunCommand) -> tuple[SlotComponent, ...]:
    if isinstance(request, ForwardBackwardRequest):
        return ("weights", "accumulator")
    if isinstance(request, ForwardRequest):
        return ("weights",)
    if isinstance(request, OptimStepRequest):
        return ("weights", "optimizer", "accumulator")
    return ()


def _summarize_residency_evidence(
    *,
    operation: OperationRef,
    expected_generation: TrainerGeneration,
    runtime_spec: Any,
    evidence: dict[str, Any],
) -> MegatronOperationResidencySummary:
    detail = _DetailedResidencyEvidence.model_validate(evidence)
    if (
        detail.run_id != operation.run_id
        or detail.operation_id != operation.operation_id
        or detail.learner_version != operation.learner_parent_version
        or detail.learner_version != expected_generation.policy_step
    ):
        raise RuntimeError("residency evidence changed operation or learner identity")
    if len(set(detail.requested_components)) != len(detail.requested_components):
        raise RuntimeError("residency evidence repeated a requested component")

    rank_count = len(runtime_spec.trainer_mesh.ranks)
    ranks = tuple(item.rank for item in detail.rank_evidence)
    if len(set(ranks)) != len(ranks) or any(rank >= rank_count for rank in ranks):
        raise RuntimeError("residency evidence returned invalid trainer ranks")
    requested = set(detail.requested_components)
    aggregates: dict[SlotComponent, dict[str, Any]] = {
        component: {
            "observed": 0,
            "l1_ready": 0,
            "l1_bytes": 0,
            "l2_present": 0,
            "l3_present": 0,
            "l1_reload": 0,
            "l1_reload_bytes": 0,
            "l1_reload_sources": set(),
        }
        for component in _SLOT_COMPONENTS
    }
    tier_aggregates = {
        tier: {"copies": 0, "ready_copies": 0, "bytes": 0, "ready_bytes": 0}
        for tier in _RESIDENCY_TIERS
    }
    for rank in detail.rank_evidence:
        if (
            rank.run_id != operation.run_id
            or rank.operation_id != operation.operation_id
            or rank.requested_components != detail.requested_components
        ):
            raise RuntimeError("rank residency evidence changed operation identity")
        names = tuple(item.component for item in rank.components)
        if len(set(names)) != len(names):
            raise RuntimeError("rank residency evidence repeated a component")
        for component in rank.components:
            tiers = set(component.tiers)
            if len(tiers) != len(
                component.tiers
            ) or component.required_for_operation != (component.component in requested):
                raise RuntimeError("rank residency component evidence is inconsistent")
            if (
                component.required_for_operation
                and component.generation_id != expected_generation.generation_id
            ):
                raise RuntimeError("required residency component changed generation")
            aggregate = aggregates[component.component]
            aggregate["observed"] += 1
            l1 = next(
                (copy for copy in component.copies if copy.tier == "l1_gpu"), None
            )
            aggregate["l1_ready"] += int(l1 is not None and l1.ready)
            aggregate["l1_bytes"] += 0 if l1 is None or not l1.ready else l1.byte_count
            aggregate["l2_present"] += int("l2_cpu" in tiers)
            aggregate["l3_present"] += int("l3_nvme" in tiers)
            reload = component.last_l1_reload
            if component.reloaded_for_operation:
                assert reload is not None
                aggregate["l1_reload"] += 1
                aggregate["l1_reload_bytes"] += reload.byte_count
                aggregate["l1_reload_sources"].add(reload.source)
            for copy in component.copies:
                tier = tier_aggregates[copy.tier]
                tier["copies"] += 1
                tier["ready_copies"] += int(copy.ready)
                tier["bytes"] += copy.byte_count
                tier["ready_bytes"] += copy.byte_count if copy.ready else 0

    components = tuple(
        MegatronResidencyComponentSummary(
            component=component,
            required_for_operation=component in requested,
            observed_rank_count=values["observed"],
            l1_ready_rank_count=values["l1_ready"],
            l1_ready_bytes=values["l1_bytes"],
            l2_present_rank_count=values["l2_present"],
            l3_present_rank_count=values["l3_present"],
            all_ranks_l1_ready=(
                values["observed"] == rank_count and values["l1_ready"] == rank_count
            ),
            l1_reload_rank_count=values["l1_reload"],
            l1_reload_bytes=values["l1_reload_bytes"],
            l1_reload_sources=tuple(sorted(values["l1_reload_sources"])),
            all_ranks_reloaded_after_eviction=(
                values["observed"] == rank_count and values["l1_reload"] == rank_count
            ),
        )
        for component, values in aggregates.items()
    )
    tiers = tuple(
        MegatronResidencyTierSummary(
            tier=tier,
            copy_count=values["copies"],
            ready_copy_count=values["ready_copies"],
            byte_count=values["bytes"],
            ready_bytes=values["ready_bytes"],
        )
        for tier, values in tier_aggregates.items()
    )
    all_ranks_reported = len(detail.rank_evidence) == rank_count
    topology = runtime_spec.trainer_mesh.topology
    topology_width = topology.tp * topology.cp * topology.pp
    if rank_count % topology_width:
        raise RuntimeError("trainer topology does not divide its rank count")
    canonical_detail = json.dumps(
        detail.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return MegatronOperationResidencySummary(
        run_id=operation.run_id,
        operation_id=operation.operation_id,
        operation_kind=operation.kind,
        operation_sequence=operation.sequence_id,
        learner_parent_version=operation.learner_parent_version,
        reserved_output_learner_version=operation.reserved_output_learner_version,
        learner_version=detail.learner_version,
        runtime_fingerprint=runtime_spec.fingerprint,
        topology=MegatronResidencyTopologySummary(
            rank_count=rank_count,
            data_parallel_size=rank_count // topology_width,
            tensor_parallel_size=topology.tp,
            context_parallel_size=topology.cp,
            expert_parallel_size=getattr(topology, "ep", 1),
            pipeline_parallel_size=topology.pp,
            virtual_pipeline_parallel_size=getattr(topology, "vpp", None),
            expert_tensor_parallel_size=getattr(topology, "etp", 1),
        ),
        requested_components=detail.requested_components,
        components=components,
        tiers=tiers,
        observed_rank_count=len(detail.rank_evidence),
        all_ranks_reported=all_ranks_reported,
        all_requested_components_l1_ready=(
            all_ranks_reported
            and all(
                component.all_ranks_l1_ready
                for component in components
                if component.required_for_operation
            )
        ),
        l1_ready_bytes=tiers[0].ready_bytes,
        residency_wait_s=detail.wait_s,
        rank_max_prepare_s=detail.rank_max_s,
        detailed_evidence_sha256=hashlib.sha256(canonical_detail).hexdigest(),
    )


def _same_generation_identity(
    target: TrainerGeneration,
    source: TrainerGeneration,
) -> bool:
    return (
        target.training_session_id,
        target.policy_step,
        target.generation_id,
    ) == (
        source.training_session_id,
        source.policy_step,
        source.generation_id,
    )


def _validate_migration_replay(
    initial: TrainerCommandRunState,
    operations: tuple[MegatronMigrationReplay, ...],
) -> TrainerCommandRunState:
    next_sequence = initial.next_operation_sequence
    learner_version = initial.learner_version
    open_ids = list(initial.open_forward_backward_operation_ids)
    seen: set[str] = set()
    expected_types = {
        "forward": ForwardRequest,
        "forward_backward": ForwardBackwardRequest,
        "optim_step": OptimStepRequest,
        "save_sampler": SaveWeightsForSamplerRequest,
        "save_state": SaveStateRequest,
        "load_state": LoadStateRequest,
    }
    for replay in operations:
        request = replay.request
        admission = replay.admission
        operation = admission.ref
        if (
            request.run_id != initial.run_id
            or operation.run_id != initial.run_id
            or request.sequence_id != next_sequence
            or operation.sequence_id != next_sequence
        ):
            raise ValueError("migration replay sequence or run identity changed")
        if operation.operation_id in seen:
            raise ValueError("migration replay operation IDs must be unique")
        seen.add(operation.operation_id)
        if type(request) is not expected_types[operation.kind]:
            raise TypeError("migration replay request kind changed")
        if operation.learner_parent_version != learner_version:
            raise ValueError("migration replay learner lineage changed")
        contributions = admission.contributing_forward_backward_operation_ids
        if operation.kind == "forward_backward":
            if contributions or len(open_ids) >= 64:
                raise ValueError("migration replay F/B contribution is invalid")
            open_ids.append(operation.operation_id)
        elif operation.kind == "optim_step":
            if not open_ids or contributions != tuple(open_ids):
                raise ValueError("migration replay optimizer contribution set changed")
            open_ids.clear()
            assert operation.reserved_output_learner_version is not None
            learner_version = operation.reserved_output_learner_version
        elif operation.kind == "load_state":
            if open_ids or contributions:
                raise ValueError("migration replay load cannot discard gradients")
            assert operation.reserved_output_learner_version is not None
            learner_version = operation.reserved_output_learner_version
        elif contributions:
            raise ValueError("migration replay attached unexpected contributions")
        next_sequence += 1
    return initial.model_copy(
        update={
            "learner_version": learner_version,
            "next_operation_sequence": next_sequence,
            "open_forward_backward_operation_ids": tuple(open_ids),
        }
    )


def _not_executed(code: OperationFailureCode, message: str) -> OperationExecutionError:
    return OperationExecutionError(
        code,
        message,
        usage=CommandExecutionUsage.no_work(),
    )


async def _await_terminal(
    future: asyncio.Future[OperationResultType],
) -> OperationResultType:
    # Task cancellation cannot decide an already-dispatched GPU command's outcome.
    while True:
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.done():
                return future.result()
