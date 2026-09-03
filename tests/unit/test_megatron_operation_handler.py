import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest
import torch

from art import TrainableModel
from art.distributed.art_runtime import DistributedPackedBatch
from art.distributed.data_plane import (
    PackedBatchLeaseSet,
    PackedBatchRef,
    PrefixTreePackingStatsSpec,
    TensorSpec,
    TokenizedOutputMapSpec,
)
from art.distributed.packing import TrajectoryGroupPayload
from art.distributed.rollout import DistributedTrajectorySelection, RolloutModelSpec
from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.megatron import MegatronBackend, MegatronOperationResidencySummary
from art.megatron.operation_handler import (
    POLICY_ACTIVATION_LAG_METRIC,
    MegatronInferenceUpdateUsage,
    MegatronOperationConfig,
    MegatronOperationHandler,
    MegatronPolicyActivationTiming,
    MegatronRetainedState,
    MegatronSamplerPublicationReceipt,
)
from art.megatron.runtime.monarch import MonarchTrainerRun
from art.megatron.runtime.portable_snapshot import (
    PortableSnapshotArchive,
    PortableSnapshotFile,
    PortableSnapshotGeneration,
    PortableSnapshotInstallReceipt,
    PortableSnapshotRankReceipt,
    PortableSnapshotReadFile,
    PortableSnapshotReadReceipt,
    build_portable_snapshot_archive,
)
from art.megatron.runtime.residency import (
    ResidencyCapacityUnavailable,
    ResidencyKey,
    ResidencyL1ReloadReceipt,
    ResidencyLedger,
    ResidencyLimits,
    TierCapacity,
)
from art.megatron.runtime.run_residency import RunResidencyManager
from art.megatron.runtime.specs import TrainerCommandRunState, TrainerGeneration
from art.megatron.slot_coordinator import (
    MegatronMigrationContribution,
    MegatronMigrationFence,
    MegatronMigrationReplay,
    MegatronSlotCoordinator,
    MegatronSlotResourceRequest,
    MegatronSlotRun,
    TrainerMegatronSlotResources,
)
from art.megatron.slot_runtime import MegatronRunBinding
from art.megatron.training import LocalMegatronTrainingClient
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from art.preprocessing.tokenize import SFTBatch
from art.training import (
    AdamConfig,
    AdapterSpec,
    CheckpointRef,
    CommandAdmission,
    ExternalLoraReceipt,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    ImmutablePublicationRef,
    LossConfig,
    OperationExecutionError,
    OperationRef,
    OperationSucceeded,
    OptimStepRequest,
    OptimStepResult,
    PackingOutcome,
    RlTrajectoryBatch,
    SamplerPublication,
    SamplerWeightsResult,
    SaveWeightsForSamplerRequest,
    TrainingInputObject,
    TrainingInputObjectRef,
    bootstrap_operation_worker,
)
from art.trajectories import Trajectory, TrajectoryGroup
from art.types import TrainSFTConfig
from art.vllm_route_transport import (
    RetainedRouteBundleRef,
    RouteBundleChoiceLayout,
    RouteBundleLayout,
    RouteBundleObjectRef,
    route_bundle_id,
)


def _packed_batch(
    *,
    content_sha256: str | None = None,
    batch_id: str = "batch",
    non_padding_tokens: int = 7,
    loss_bearing_tokens: int = 4,
    trainable_assistant_tokens: int = 4,
    tokenized: bool = False,
) -> DistributedPackedBatch:
    item_sizes = {
        "tokens": ("int64", 8),
        "group_ids": ("int64", 8),
        "parent_ids": ("int64", 8),
        "input_pos": ("int64", 8),
        "assistant_mask": ("bool", 1),
        "logprobs": ("float32", 4),
        "advantages": ("float32", 4),
        "weights": ("float32", 4),
    }
    if tokenized:
        item_sizes.update(
            {
                "target_tokens": ("int64", 8),
                "loss_weights": ("float32", 4),
                "behavior_logprobs": ("float32", 4),
                "token_advantages": ("float32", 4),
            }
        )
    offset = 0
    tensors = []
    for name, (dtype, item_size) in item_sizes.items():
        byte_count = 8 * item_size
        tensors.append(
            TensorSpec(
                name=name,
                dtype=dtype,
                shape=(1, 8, 1)
                if name
                in {
                    "target_tokens",
                    "loss_weights",
                    "behavior_logprobs",
                    "token_advantages",
                }
                else (1, 8),
                offset=offset,
                byte_count=byte_count,
            )
        )
        offset += byte_count
    ref = PackedBatchRef(
        batch_id=batch_id,
        owner_actor_id="owner",
        lease_id="lease",
        shared_memory_name="shm",
        owner_process_id=1,
        tensors=tuple(tensors),
        num_sequences=1,
        sequence_length=8,
        byte_count=offset,
        storage_byte_count=offset,
        content_sha256=content_sha256,
        pixel_values_present=(False,),
        image_grid_thw_present=(False,),
        prefix_tree_packing_stats=PrefixTreePackingStatsSpec(
            logical_tokens=7, physical_tokens=8
        ),
        training_kind="tokenized" if tokenized else "rl",
        tokenized_output_map=(
            TokenizedOutputMapSpec(packed_positions=((0, 1),), candidate_counts=(1,))
            if tokenized
            else None
        ),
    )
    return DistributedPackedBatch(
        leases=PackedBatchLeaseSet(ref=ref, host_refs={"host": ref}),
        packed_group_shapes=(),
        trainable_assistant_tokens=trainable_assistant_tokens,
        loss_bearing_tokens=loss_bearing_tokens,
        non_padding_tokens=non_padding_tokens,
        packing_generation_id="packing",
    )


class _Runtime:
    def __init__(self) -> None:
        self.packed = _packed_batch()
        self.pack_requests = []
        self.released: list[DistributedPackedBatch] = []

    async def pack(self, request):
        self.pack_requests.append(request)
        if request.tokenized_batch is not None:
            self.packed = _packed_batch(
                batch_id="tokenized",
                content_sha256=("b" * 64 if request.compute_content_sha256 else None),
                non_padding_tokens=sum(
                    len(datum.input_tokens) for datum in request.tokenized_batch.datums
                ),
                loss_bearing_tokens=sum(
                    weight != 0.0
                    for datum in request.tokenized_batch.datums
                    for row in datum.weights or ()
                    for weight in row
                ),
                trainable_assistant_tokens=sum(
                    weight != 0.0
                    for datum in request.tokenized_batch.datums
                    for row in datum.weights or ()
                    for weight in row
                ),
                tokenized=True,
            )
        if (
            request.compute_content_sha256
            and self.packed.leases.ref.content_sha256 is None
        ):
            self.packed = _packed_batch(content_sha256="b" * 64)
        return self.packed

    async def release_batch(self, batch):
        self.released.append(batch)


class _InputResolver:
    def __init__(self, batch: RlTrajectoryBatch) -> None:
        self.batch = batch
        self.calls: list[tuple[TrainingInputObjectRef, OperationRef]] = []

    async def resolve(
        self,
        input_object: TrainingInputObjectRef,
        *,
        operation: OperationRef,
    ) -> RlTrajectoryBatch:
        self.calls.append((input_object, operation))
        return self.batch


class _RouteOwnership:
    def __init__(self) -> None:
        self.acquired = []
        self.transferred = []
        self.released = []

    async def acquire(self, *, operation, bundles):
        handle = ("source", operation.operation_id)
        self.acquired.append((operation, bundles, handle))
        return handle

    async def transfer(self, handle, *, transfer_id, target_owner_id):
        target = ("target", transfer_id, target_owner_id)
        self.transferred.append((handle, target))
        return target

    async def release(self, handle):
        self.released.append(handle)


class _Trainer:
    def __init__(self) -> None:
        topology = SimpleNamespace(tp=1, cp=1, pp=1)
        self.runtime_spec = SimpleNamespace(
            fingerprint="e" * 64,
            lora_rank=32,
            lora_target_modules=("q_proj", "v_proj"),
            packed_sequence_length=8,
            enable_moe_routing_replay=False,
            trainer_mesh=SimpleNamespace(ranks=(0,), topology=topology),
        )
        self.fail_optimizer = True
        self.active = 0
        self.max_active = 0
        self.executed_runs: list[str] = []
        self.result_gates: dict[str, asyncio.Event] = {}
        self.registered_runs: set[str] = set()
        self.registered_adapters: dict[str, tuple[int, tuple[str, ...]]] = {}
        self.registered_timeouts: dict[str, float] = {}
        self.run_generation_ids: dict[str, str] = {}
        self.run_states: dict[str, TrainerCommandRunState] = {}
        self.migration_releases: list[str] = []
        self.optimizer_jobs = []
        self.forward_backward_jobs = []
        self.cp_lookaheads = []

    async def prepare_cp_lookahead(self, batch, *, global_grad_accumulation_sequences):
        self.cp_lookaheads.append((batch, global_grad_accumulation_sequences))
        return {"time/step_cp_lookahead_wait_s": 0.25}

    async def prefetch_command_run_residency(self, run_id, components, learner_version):
        return {
            "run_id": run_id,
            "requested_components": components,
            "learner_version": learner_version,
        }

    async def admit_command_run_residency(
        self, operation_id, run_id, components, learner_version
    ):
        return {
            "operation_id": operation_id,
            "run_id": run_id,
            "requested_components": components,
            "learner_version": learner_version,
            "rank_evidence": (
                {
                    "rank": 0,
                    "run_id": run_id,
                    "operation_id": operation_id,
                    "requested_components": components,
                    "components": tuple(
                        {
                            "component": component,
                            "generation_id": self.run_generation_ids[run_id],
                            "required_for_operation": True,
                            "byte_count": 1024,
                            "tiers": ("l1_gpu", "l2_cpu"),
                            "l1_ready": True,
                            "copies": (
                                {
                                    "tier": "l1_gpu",
                                    "byte_count": 1024,
                                    "ready": True,
                                },
                                {
                                    "tier": "l2_cpu",
                                    "byte_count": 1024,
                                    "ready": True,
                                },
                            ),
                            "last_l1_reload": {
                                "source": "l2_cpu",
                                "byte_count": 1024,
                                "eviction_sequence": 1,
                                "reload_sequence": 2,
                                "source_immutable_ref": None,
                                "source_digest": None,
                            },
                            "reloaded_for_operation": True,
                        }
                        for component in components
                    ),
                },
            ),
            "wait_s": 0.25,
            "rank_max_s": 0.2,
        }

    async def release_command_run_residency_admission(self, operation_id):
        del operation_id

    async def register_command_run(self, run_spec):
        self.registered_runs.add(run_spec.run_id)
        self.registered_adapters[run_spec.run_id] = (
            run_spec.lora_rank,
            run_spec.lora_target_modules,
        )
        self.registered_timeouts[run_spec.run_id] = run_spec.event_timeout_s
        self.registered_restore_optimizer = getattr(
            self, "registered_restore_optimizer", {}
        )
        self.registered_restore_optimizer[run_spec.run_id] = (
            run_spec.initial_restore_optimizer
        )
        assert run_spec.initial_generation_id is not None
        self.run_generation_ids[run_spec.run_id] = run_spec.initial_generation_id
        self.run_states[run_spec.run_id] = TrainerCommandRunState(
            run_id=run_spec.run_id,
            training_session_id=run_spec.training_session_id,
            learner_version=run_spec.initial_learner_version,
            next_operation_sequence=run_spec.initial_operation_sequence,
            open_forward_backward_operation_ids=(),
        )
        archive = run_spec.initial_portable_snapshot
        if archive is None:
            return None
        return PortableSnapshotInstallReceipt(
            archive_sha256=archive.archive_sha256,
            runtime_fingerprint=self.runtime_spec.fingerprint,
            restore_optimizer=run_spec.initial_restore_optimizer,
            ranks=(
                PortableSnapshotReadReceipt(
                    archive_sha256=archive.archive_sha256,
                    destination_rank=0,
                    files=tuple(
                        PortableSnapshotReadFile(
                            source_rank=receipt.rank,
                            relative_path=file.relative_path,
                            byte_count=file.byte_count,
                            sha256=file.sha256,
                        )
                        for receipt in archive.ranks
                        for file in receipt.files
                    ),
                ),
            ),
        )

    async def command_run_state(self, run_id: str) -> TrainerCommandRunState:
        return self.run_states[run_id]

    async def record_control_command(
        self, operation: OperationRef, learner_version: int
    ) -> None:
        state = self.run_states[operation.run_id]
        self.run_states[operation.run_id] = state.model_copy(
            update={
                "learner_version": learner_version,
                "next_operation_sequence": state.next_operation_sequence + 1,
            }
        )

    async def record_no_work_command(
        self, operation: OperationRef, learner_version: int
    ) -> None:
        await self.record_control_command(operation, learner_version)

    async def release_command_run_for_migration(self, run_id: str) -> None:
        self.migration_releases.append(run_id)
        self.run_states.pop(run_id, None)
        self.registered_runs.discard(run_id)

    async def drain_command_run(self, run_id: str) -> None:
        self.run_states.pop(run_id, None)
        self.registered_runs.discard(run_id)

    def _advance(self, job, *, backward: bool = False, optimizer: bool = False) -> None:
        state = self.run_states.get(job.run_id)
        if state is None:
            return
        open_ids = state.open_forward_backward_operation_ids
        learner_version = state.learner_version
        if backward:
            open_ids = (*open_ids, job.operation_id)
        if optimizer:
            open_ids = ()
            learner_version = job.learner_version
            self.run_generation_ids[job.run_id] = job.generation.generation_id
        self.run_states[job.run_id] = state.model_copy(
            update={
                "learner_version": learner_version,
                "next_operation_sequence": state.next_operation_sequence + 1,
                "open_forward_backward_operation_ids": open_ids,
            }
        )

    async def _enter(self, run_id: str) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.executed_runs.append(run_id)
        await asyncio.sleep(0)
        self.active -= 1

    async def start_forward(self, job, _batch):
        await self._enter(job.run_id)
        self._advance(job)
        result = {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "logical_nonpadding_tokens": 7,
            "executed_token_equivalents": 8,
            "gpu_count": 4,
            "gpu_service_ns": 12_000_000,
            "metrics": {
                "data/gradient_step_nonpadding_logical_tokens": 7.0,
                "data/gradient_step_loss_bearing_tokens": 4.0,
                "data/gradient_step_executed_token_equivalents": 8.0,
                "data/gradient_step_nominal_schedule_capacity_tokens": 8.0,
                "data/gradient_step_dummy_executed_token_equivalents": 0.0,
                "data/gradient_step_dummy_schedule_capacity_tokens": 0.0,
                "pipeline/gradient_step_real_microbatches": 1.0,
                "pipeline/gradient_step_dummy_microbatches": 0.0,
                "time/forward_backward_s": 0.01,
            },
        }

        async def settle():
            if gate := self.result_gates.get(job.run_id):
                await gate.wait()
            return result

        return SimpleNamespace(completion=asyncio.create_task(settle()))

    async def forward(self, job, batch):
        return await (await self.start_forward(job, batch)).completion

    async def start_forward_backward(self, job, _batch):
        self.forward_backward_jobs.append(job)
        await self._enter(job.run_id)
        self._advance(job, backward=True)
        result = {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "logical_nonpadding_tokens": 7,
            "executed_token_equivalents": 8,
            "gpu_count": 4,
            "gpu_service_ns": 12_000_000,
            "metrics": {
                "data/gradient_step_nonpadding_logical_tokens": 7.0,
                "data/gradient_step_loss_bearing_tokens": 4.0,
                "data/gradient_step_executed_token_equivalents": 8.0,
                "data/gradient_step_nominal_schedule_capacity_tokens": 8.0,
                "data/gradient_step_dummy_executed_token_equivalents": 0.0,
                "data/gradient_step_dummy_schedule_capacity_tokens": 0.0,
                "pipeline/gradient_step_real_microbatches": 1.0,
                "pipeline/gradient_step_dummy_microbatches": 0.0,
                "time/forward_backward_s": 0.01,
            },
        }

        async def settle():
            if gate := self.result_gates.get(job.run_id):
                await gate.wait()
            return result

        return SimpleNamespace(completion=asyncio.create_task(settle()))

    async def forward_backward(self, job, batch):
        return await (await self.start_forward_backward(job, batch)).completion

    async def optim_step(self, job):
        if self.fail_optimizer:
            raise RuntimeError("optimizer failed")
        self.optimizer_jobs.append(job)
        self._advance(job, optimizer=True)
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": (
                job.contributing_forward_backward_operation_ids
            ),
            "gpu_count": 4,
            "gpu_service_ns": 3_000_000,
            "metrics": {"time/optimizer_step_s": 0.005},
        }


@pytest.mark.asyncio
async def test_slot_resources_keep_only_one_command_ahead_under_pressure() -> None:
    class _CapacityTrainer:
        def __init__(self) -> None:
            self.attempts: list[str] = []
            self.admitted: set[str] = set()

        async def prefetch_command_run_residency(
            self, run_id, components, learner_version
        ):
            del run_id, components, learner_version
            return {}

        async def admit_command_run_residency(
            self, operation_id, run_id, components, learner_version
        ):
            del run_id, components, learner_version
            self.attempts.append(operation_id)
            if self.admitted:
                raise ResidencyCapacityUnavailable("current command pins L1")
            self.admitted.add(operation_id)
            return {"operation_id": operation_id}

        async def release_command_run_residency_admission(self, operation_id):
            self.admitted.discard(operation_id)

    generation = TrainerGeneration(
        training_session_id="session",
        policy_step=0,
        generation_id=f"step-00000000-{'a' * 32}",
        adapter_path="/adapter/0",
    )

    def request(index: int) -> MegatronSlotResourceRequest:
        return MegatronSlotResourceRequest(
            run_id=f"run-{index}",
            operation_id=f"operation-{index}",
            source=generation,
            optimizer_state_path=f"/optimizer/{index}",
            components=("weights", "accumulator"),
        )

    trainer = _CapacityTrainer()
    resources = TrainerMegatronSlotResources(trainer)  # type: ignore[arg-type]
    requests = tuple(request(index) for index in range(3))
    for item in requests:
        resources.prefetch(item)

    assert await resources.ensure(requests[0]) == {"operation_id": "operation-0"}
    second = asyncio.create_task(resources.ensure(requests[1]))
    await asyncio.sleep(0)
    assert trainer.attempts == ["operation-0", "operation-1"]
    assert "operation-2" not in trainer.attempts

    await resources.release(requests[0])
    assert await asyncio.wait_for(second, timeout=1.0) == {
        "operation_id": "operation-1"
    }
    await asyncio.sleep(0)
    assert trainer.attempts == [
        "operation-0",
        "operation-1",
        "operation-1",
        "operation-2",
    ]

    await resources.release(requests[1])
    assert await resources.ensure(requests[2]) == {"operation_id": "operation-2"}
    await resources.release(requests[2])


def test_run_residency_pins_under_l1_admission_lock() -> None:
    class _TrackedRLock:
        def __init__(self) -> None:
            self._lock = RLock()
            self.depth = 0

        def __enter__(self):
            self._lock.acquire()
            self.depth += 1
            return self

        def __exit__(self, *_args) -> None:
            self.depth -= 1
            self._lock.release()

    class _Ledger:
        present = False
        pinned: tuple[tuple[ResidencyKey, str], ...] = ()

        def has_copy(self, _key, tier) -> bool:
            return tier == "l1_gpu" and self.present

        def pin_many(self, copies) -> None:
            assert admission.depth > 0
            self.pinned = tuple(copies)

        def claim_l1_reloads(self, _keys):
            return {}

    key = ResidencyKey(
        training_session_id="session",
        run_id="run",
        generation_id="generation",
        topology_fingerprint="topology",
        adapter_layout_fingerprint="adapter",
    )
    admission = _TrackedRLock()
    ledger = _Ledger()
    manager = object.__new__(RunResidencyManager)
    manager._admission_locks = {"l1_gpu": admission}  # type: ignore[assignment]
    manager._lock = RLock()
    manager._states = {key: SimpleNamespace(l1_transition=None)}  # type: ignore[assignment]
    manager._retirements = {}
    manager._failures = []
    manager._closing = False
    manager._closed = False

    def prepare(_keys) -> None:
        assert admission.depth > 0
        ledger.present = True

    manager.ledger = ledger  # type: ignore[assignment]
    manager.prepare_l1_working_set = prepare  # type: ignore[method-assign]
    manager.acquire_l1_working_set((key,))

    assert ledger.pinned == ((key, "l1_gpu"),)


def test_residency_ledger_records_pressure_eviction_and_exact_reload() -> None:
    l1_capacity = TierCapacity(max_bytes=1024)
    lower_capacity = TierCapacity(max_bytes=4096)
    ledger = ResidencyLedger(
        ResidencyLimits(
            l1_gpu=l1_capacity,
            l2_cpu=lower_capacity,
            l3_nvme=lower_capacity,
        )
    )
    keys = tuple(
        ResidencyKey(
            training_session_id="session",
            run_id=f"run-{index}",
            generation_id="generation",
            topology_fingerprint="topology",
            adapter_layout_fingerprint="adapter",
        )
        for index in range(2)
    )
    for index, key in enumerate(keys):
        l2 = ledger.reserve(key, source=None, target="l2_cpu", byte_count=1024)
        ledger.commit(
            l2, immutable_ref=f"host-image-{index}", digest=f"{index + 1}" * 64
        )

    first_l1 = ledger.reserve(
        keys[0], source="l2_cpu", target="l1_gpu", byte_count=1024
    )
    ledger.commit(first_l1)
    assert ledger.entry(keys[0]).last_l1_reload is None

    assert ledger.admission_evictions("l1_gpu", 1024, 1024, protected={keys[1]}) == (
        keys[0],
    )
    ledger.drop(keys[0], "l1_gpu")
    second_l1 = ledger.reserve(
        keys[1], source="l2_cpu", target="l1_gpu", byte_count=1024
    )
    ledger.commit(second_l1)

    assert ledger.admission_evictions("l1_gpu", 1024, 1024, protected={keys[0]}) == (
        keys[1],
    )
    ledger.drop(keys[1], "l1_gpu")
    reloaded_l1 = ledger.reserve(
        keys[0], source="l2_cpu", target="l1_gpu", byte_count=1024
    )
    ledger.commit(reloaded_l1)

    assert ledger.entry(keys[0]).last_l1_reload == ResidencyL1ReloadReceipt(
        source="l2_cpu",
        byte_count=1024,
        eviction_sequence=1,
        reload_sequence=3,
        source_immutable_ref="host-image-0",
        source_digest="1" * 64,
    )


def _operation(
    operation_id: str,
    kind: str,
    sequence_id: int,
    *,
    parent: int = 0,
    output: int | None = None,
) -> OperationRef:
    return OperationRef(
        run_id="run",
        operation_id=operation_id,
        sequence_id=sequence_id,
        learner_parent_version=parent,
        reserved_output_learner_version=output,
        kind=kind,
    )


def _batch() -> RlTrajectoryBatch:
    return RlTrajectoryBatch(
        groups=(TrajectoryGroupBundle.from_group(TrajectoryGroup()),),
        min_source_version=0,
        max_source_version=0,
    )


@pytest.mark.asyncio
async def test_local_megatron_client_uses_slot_worker_and_exact_contributions() -> None:
    packing = PackingOutcome(
        packed_sequence_length=8,
        packed_sequences=1,
        target_packed_sequences=1,
        physical_tokens=8,
        non_padding_tokens=8,
        loss_bearing_tokens=8,
        trainable_assistant_tokens=8,
    )

    class _Worker:
        def __init__(self) -> None:
            self.calls = []
            self.retired = []

        async def execute(self, request, operation, contributions=()):
            self.calls.append((request, operation, contributions))
            if isinstance(request, ForwardBackwardRequest):
                result = ForwardBackwardResult(
                    operation_id=operation.operation_id,
                    packing=packing,
                    produced_gradient=True,
                )
            else:
                assert isinstance(request, OptimStepRequest)
                result = OptimStepResult(
                    operation_id=operation.operation_id,
                    contributing_forward_backward_operation_ids=contributions,
                    checkpoint=CheckpointRef(
                        run_id="run", learner_version=1, checkpoint_id="generation-1"
                    ),
                )
            return OperationSucceeded(operation=operation, result=result)

        def retire(self, operation_id):
            self.retired.append(operation_id)

    worker = _Worker()
    client = LocalMegatronTrainingClient(
        MegatronSlotRun("run", worker),  # type: ignore[arg-type]
        learner_version=0,
    )
    forward = await client.forward_backward(
        ForwardBackwardRequest(
            run_id="run",
            request_id="forward",
            sequence_id=0,
            batch=_batch(),
            loss=LossConfig(name="cispo"),
        )
    )
    assert (await forward.result()).produced_gradient
    optimizer = await client.optim_step(
        OptimStepRequest(
            run_id="run",
            request_id="optimizer",
            sequence_id=1,
            optimizer=AdamConfig(learning_rate=1e-5),
        )
    )
    result = await optimizer.result()

    assert worker.calls[1][2] == (forward.ref.operation_id,)
    assert result.contributing_forward_backward_operation_ids == (
        forward.ref.operation_id,
    )
    assert client.projected_learner_version == 1
    assert (await client.operation_evidence(optimizer.ref.operation_id)).status == (
        "succeeded"
    )
    assert not client._ledger._records
    assert worker.retired == []
    assert client.retire_operation(forward.ref.operation_id)
    assert client.retire_operation(optimizer.ref.operation_id)
    assert worker.retired == [forward.ref.operation_id, optimizer.ref.operation_id]
    await client.close()


@pytest.mark.asyncio
async def test_local_client_resumes_sequence_and_retries_evidence_retirement() -> None:
    class _Worker:
        def __init__(self) -> None:
            self.retire_attempts = 0

        async def execute(self, request, operation, contributions=()):
            assert request.sequence_id == 17
            assert not contributions
            return OperationSucceeded(
                operation=operation,
                result=ForwardResult(
                    operation_id=operation.operation_id,
                    packing=PackingOutcome(
                        packed_sequence_length=1,
                        packed_sequences=1,
                        target_packed_sequences=1,
                        physical_tokens=1,
                        non_padding_tokens=1,
                        loss_bearing_tokens=1,
                        trainable_assistant_tokens=1,
                    ),
                ),
            )

        def retire(self, operation_id):
            del operation_id
            self.retire_attempts += 1
            if self.retire_attempts == 1:
                raise RuntimeError("evidence store unavailable")

    worker = _Worker()
    client = LocalMegatronTrainingClient(
        MegatronSlotRun("run", worker),  # type: ignore[arg-type]
        learner_version=3,
        initial_operation_sequence=17,
    )
    operation = await client.forward(
        ForwardRequest(
            run_id="run",
            request_id="forward-resumed",
            sequence_id=17,
            batch=_batch(),
            loss=LossConfig(name="cispo"),
        )
    )
    await operation.result()

    with pytest.raises(RuntimeError, match="unconsumed local operation evidence"):
        await client.close()
    with pytest.raises(RuntimeError, match="evidence store unavailable"):
        client.retire_operation(operation.ref.operation_id)
    assert client.operation_ids == (operation.ref.operation_id,)
    assert (await client.operation_evidence(operation.ref.operation_id)).status == (
        "succeeded"
    )
    assert client.retire_operation(operation.ref.operation_id)
    assert client.operation_ids == ()
    await client.close()


@pytest.mark.asyncio
async def test_local_megatron_client_retires_real_worker_records_across_window() -> (
    None
):
    packing = PackingOutcome(
        packed_sequence_length=8,
        packed_sequences=1,
        target_packed_sequences=1,
        physical_tokens=8,
        non_padding_tokens=8,
        loss_bearing_tokens=8,
        trainable_assistant_tokens=8,
    )

    async def handler(request, operation, contributions):
        assert isinstance(request, ForwardRequest)
        assert not contributions
        return ForwardResult(operation_id=operation.operation_id, packing=packing)

    worker = bootstrap_operation_worker(handler, max_retained_operations=2)
    client = LocalMegatronTrainingClient(
        MegatronSlotRun("run", worker),
        learner_version=0,
        max_retained_operations=2,
    )
    retained = []
    for sequence_id in range(2):
        request = ForwardRequest(
            run_id="run",
            request_id=f"forward-{sequence_id}",
            sequence_id=sequence_id,
            batch=_batch(),
            loss=LossConfig(name="cispo"),
        )
        operation = await client.forward(request)
        assert (await operation.result()).operation_id == operation.ref.operation_id
        assert await client.forward(request) is operation
        assert (await client.operation_evidence(operation.ref.operation_id)).status == (
            "succeeded"
        )
        retained.append(operation)

    with pytest.raises(RuntimeError, match="replay window is full"):
        await client.forward(
            ForwardRequest(
                run_id="run",
                request_id="forward-2",
                sequence_id=2,
                batch=_batch(),
                loss=LossConfig(name="cispo"),
            )
        )

    client.retire_operation(retained[0].ref.operation_id)
    for sequence_id in range(2, 5):
        operation = await client.forward(
            ForwardRequest(
                run_id="run",
                request_id=f"forward-{sequence_id}",
                sequence_id=sequence_id,
                batch=_batch(),
                loss=LossConfig(name="cispo"),
            )
        )
        await operation.result()
        client.retire_operation(operation.ref.operation_id)
    client.retire_operation(retained[1].ref.operation_id)

    assert client.operation_ids == ()
    await client.close()


def _input_object(operation_id: str) -> TrainingInputObjectRef:
    return TrainingInputObjectRef(
        run_id="run",
        operation_id=operation_id,
        input_kind="rl",
        object=TrainingInputObject(
            locator=f"caios://training-input/{operation_id}",
            size_bytes=128,
            sha256="c" * 64,
        ),
        lease_id=f"training-input:{operation_id}",
    )


def _batch_with_retained_route() -> tuple[RlTrajectoryBatch, RetainedRouteBundleRef]:
    choice = RouteBundleChoiceLayout(
        choice_index=0,
        dtype="uint8",
        shape=(1, 1, 1),
        offset=0,
        byte_count=1,
        token_ids_sha256="a" * 64,
    )
    identity = {
        "protocol_version": 1,
        "format": "art_inference_route_bundle_v1",
        "request_id": "route-request",
        "owner_id": "route-owner",
        "model_identity": "model",
        "response_id": "response",
        "num_experts": 1,
        "choices": [choice.model_dump(mode="json")],
        "byte_count": 1,
        "sha256": "b" * 64,
    }
    layout = RouteBundleLayout(bundle_id=route_bundle_id(identity), **identity)
    ref = RetainedRouteBundleRef(
        object=RouteBundleObjectRef(
            store="caios",
            locator="caios://route",
            size_bytes=1,
            sha256=layout.sha256,
        ),
        layout=layout,
        lease_id="producer-lease",
    )
    payload = TrajectoryGroupPayload.from_group(TrajectoryGroup()).model_copy(
        update={"retained_route_bundles": (ref,)}
    )
    return (
        RlTrajectoryBatch(
            groups=(TrajectoryGroupBundle.from_payload(payload),),
            min_source_version=0,
            max_source_version=0,
        ),
        ref,
    )


def _portable_archive(
    generation: TrainerGeneration,
    *,
    source_ref: str,
    payload_sha256: str = "a" * 64,
) -> PortableSnapshotArchive:
    return build_portable_snapshot_archive(
        generation=PortableSnapshotGeneration(
            training_session_id=generation.training_session_id,
            policy_step=generation.policy_step,
            generation_id=generation.generation_id,
        ),
        checkpoint_digest="d" * 64,
        ranks=(
            PortableSnapshotRankReceipt(
                rank=0,
                checkpoint_digest="d" * 64,
                files=tuple(
                    PortableSnapshotFile(
                        object_id=f"object/{path}",
                        relative_path=path,
                        component=(
                            "metadata"
                            if path in {"adapter_config.json", "checkpoint.json"}
                            else "adapter"
                        ),
                        byte_count=1,
                        sha256=payload_sha256,
                        source_ref=f"{source_ref}/{path}",
                    )
                    for path in (
                        "adapter_config.json",
                        "adapter_model.safetensors",
                        "checkpoint.json",
                    )
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_handler_retains_f_b_input_until_optimizer_commit() -> None:
    runtime = _Runtime()
    trainer = _Trainer()
    handler = MegatronOperationHandler(
        runtime,  # type: ignore[arg-type]
        trainer,
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
            adapter=AdapterSpec(rank=8, target_modules=("q_proj",)),
            source=TrainerGeneration(
                training_session_id="session",
                policy_step=0,
                generation_id=f"step-00000000-{'a' * 32}",
                adapter_path="/adapter/0",
            ),
            optimizer_state_path="/optimizer",
            rollout_model=RolloutModelSpec(payload={}),
            output_adapter_root="/adapter",
        ),
    )
    fb_request = ForwardBackwardRequest(
        run_id="run",
        request_id="fb",
        sequence_id=0,
        batch=_batch(),
        loss=LossConfig(name="cispo"),
    )
    fb = await handler(fb_request, _operation("fb", "forward_backward", 0), ())

    assert fb.packed_input_capture is not None
    assert fb.usage.logical_nonpadding_tokens.value == 7
    assert fb.usage.executed_token_equivalents.value == 8
    assert fb.usage.gpu_count.value == 4
    assert fb.usage.gpu_service_ns.value == 12_000_000
    assert handler.retained_contribution_inputs() == (("fb", fb.packed_input_capture),)
    assert runtime.released == []

    optim_request = OptimStepRequest(
        run_id="run",
        request_id="optim",
        sequence_id=1,
        optimizer=AdamConfig(learning_rate=1e-5),
    )
    optim = _operation("optim", "optim_step", 1, output=1)
    with pytest.raises(OperationExecutionError, match="optimizer failed") as failure:
        await handler(optim_request, optim, ("fb",))
    assert failure.value.usage.gpu_count.value == 1
    assert failure.value.usage.gpu_service_ns.coverage == "unknown"
    assert handler.retained_contribution_inputs()
    assert runtime.released == []

    trainer.fail_optimizer = False
    result = await handler(optim_request, optim, ("fb",))
    assert result.checkpoint.learner_version == 1
    assert result.usage.gpu_count.value == 4
    assert result.usage.gpu_service_ns.value == 3_000_000
    assert handler.retained_contribution_inputs() == (("fb", fb.packed_input_capture),)
    assert runtime.released == []
    with pytest.raises(RuntimeError, match="unacknowledged optimizer"):
        handler.retire_operation("optim")

    await handler.acknowledge_operation("optim")
    assert handler.retained_contribution_inputs() == ()
    assert runtime.released == [runtime.packed]
    handler.retire_operation("optim")

    runtime.packed = _packed_batch()
    forward_request = ForwardRequest(
        run_id="run",
        request_id="forward",
        sequence_id=2,
        batch=_batch(),
        loss=LossConfig(name="cispo"),
    )
    forward = _operation("forward", "forward", 2, parent=1)
    await handler(forward_request, forward, ())
    assert handler.retained_contribution_inputs() == ()
    await handler.release_operation_input("forward")
    assert runtime.released[-1] == runtime.packed


@pytest.mark.asyncio
async def test_handler_connects_optimizer_snapshot_to_paired_publication() -> None:
    class _Publisher:
        def __init__(self) -> None:
            self.calls = []

        async def aclose(self):
            pass

        async def plan_artifacts(self, request, generation, *, template_adapter_path):
            del request, generation, template_adapter_path
            return SimpleNamespace()

        async def save_weights_for_sampler(
            self,
            request,
            operation,
            generation,
            *,
            template_adapter_path,
            optimizer_state_path,
            staging_adapter_path,
        ):
            self.calls.append(
                (
                    generation,
                    template_adapter_path,
                    optimizer_state_path,
                    staging_adapter_path,
                )
            )
            timing = MegatronPolicyActivationTiming(
                trainer_completed_monotonic_s=1,
                serving_activated_monotonic_s=2,
            )
            result = SamplerWeightsResult(
                operation_id=operation.operation_id,
                metrics={POLICY_ACTIVATION_LAG_METRIC: 1.0},
                checkpoint=CheckpointRef(
                    run_id=operation.run_id,
                    learner_version=generation.policy_step,
                    checkpoint_id=request.checkpoint_name,
                ),
                lora="run:active",
            )
            return MegatronSamplerPublicationReceipt(
                operation_id=operation.operation_id,
                request_id=request.request_id,
                publication_mode="in_flight_lora",
                requested_public_alias="run",
                runtime_model_name="model",
                runtime_lora_name="run:active",
                serving_generation_id=generation.generation_id,
                learner_version=generation.policy_step,
                policy_activation_timing=timing,
                inference_update_usage=MegatronInferenceUpdateUsage(
                    staging_s=0.25,
                    apply_s=0.75,
                ),
                holder_update_sequence=1,
                holder_update_id="update-1",
                retained=(
                    MegatronRetainedState(
                        owner_id="owner",
                        resource="lora",
                        bytes=4096,
                        work_fingerprint="f" * 64,
                    ),
                ),
                result=result,
            )

    runtime = _Runtime()
    trainer = _Trainer()
    trainer.fail_optimizer = False
    trainer.run_states["run"] = TrainerCommandRunState(
        run_id="run",
        training_session_id="session",
        learner_version=0,
        next_operation_sequence=0,
        open_forward_backward_operation_ids=(),
    )
    publisher = _Publisher()
    handler = MegatronOperationHandler(
        runtime,  # type: ignore[arg-type]
        trainer,
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
            adapter=AdapterSpec(rank=8, target_modules=("q_proj",)),
            source=TrainerGeneration(
                training_session_id="session",
                policy_step=0,
                generation_id=f"step-00000000-{'a' * 32}",
                adapter_path="/adapter/0",
            ),
            optimizer_state_path="/optimizer",
            rollout_model=RolloutModelSpec(payload={}),
            output_adapter_root="/adapter",
        ),
        publisher=publisher,
    )
    await handler(
        ForwardBackwardRequest(
            run_id="run",
            request_id="fb",
            sequence_id=0,
            batch=_batch(),
            loss=LossConfig(name="cispo"),
        ),
        _operation("fb", "forward_backward", 0),
        (),
    )
    await handler(
        OptimStepRequest(
            run_id="run",
            request_id="optim",
            sequence_id=1,
            optimizer=AdamConfig(learning_rate=1e-5),
        ),
        _operation("optim", "optim_step", 1, output=1),
        ("fb",),
    )
    saved = await handler(
        SaveWeightsForSamplerRequest(
            run_id="run",
            request_id="save",
            sequence_id=2,
            checkpoint_name="step-1",
            publication=SamplerPublication(mode="in_flight_lora", model_alias="run"),
        ),
        _operation("save", "save_sampler", 2, parent=1),
        (),
    )

    generation = trainer.optimizer_jobs[-1].generation
    assert publisher.calls == [
        (
            generation,
            "/adapter/0",
            "/optimizer",
            f"/megatron_runtime/staging/{generation.generation_id}",
        )
    ]
    assert saved.lora == "run:active"


@pytest.mark.asyncio
async def test_handler_retains_route_ownership_through_optimizer() -> None:
    runtime = _Runtime()
    trainer = _Trainer()
    trainer.fail_optimizer = False
    trainer.runtime_spec.enable_moe_routing_replay = True
    ownership = _RouteOwnership()
    batch, route = _batch_with_retained_route()
    input_object = _input_object("fb")
    handler = MegatronOperationHandler(
        runtime,  # type: ignore[arg-type]
        trainer,
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
            adapter=AdapterSpec(rank=8, target_modules=("q_proj",)),
            source=TrainerGeneration(
                training_session_id="session",
                policy_step=0,
                generation_id=f"step-00000000-{'a' * 32}",
                adapter_path="/adapter/0",
            ),
            optimizer_state_path="/optimizer",
            rollout_model=RolloutModelSpec(payload={}),
            output_adapter_root="/adapter",
        ),
        route_ownership=ownership,
        input_resolver=_InputResolver(batch),
    )
    result = await handler(
        ForwardBackwardRequest(
            run_id="run",
            request_id="fb",
            sequence_id=0,
            batch=input_object,
            loss=LossConfig(name="cispo"),
        ),
        _operation("fb", "forward_backward", 0),
        (),
    )
    capture = result.packed_input_capture
    assert capture is not None
    assert ownership.acquired[0][1] == (route,)
    target = await handler.transfer_route_ownership(
        input_object,
        transfer_id="migration:routes",
        target_owner_id="target-runtime",
    )
    assert target is not None and ownership.released == []

    await handler(
        OptimStepRequest(
            run_id="run",
            request_id="optim",
            sequence_id=1,
            optimizer=AdamConfig(learning_rate=1e-5),
        ),
        _operation("optim", "optim_step", 1, output=1),
        ("fb",),
    )
    assert ownership.released == []
    await handler.acknowledge_operation("optim")
    assert ownership.released == [("source", "fb")]
    await ownership.release(target)


@pytest.mark.asyncio
async def test_input_object_releases_image_and_rematerializes_for_replay() -> None:
    runtime = _Runtime()
    runtime.packed = _packed_batch(content_sha256="b" * 64)
    trainer = _Trainer()
    trainer.fail_optimizer = False
    resolver = _InputResolver(_batch())
    input_object = _input_object("fb")
    handler = MegatronOperationHandler(
        runtime,  # type: ignore[arg-type]
        trainer,
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
            adapter=AdapterSpec(rank=8, target_modules=("q_proj",)),
            source=TrainerGeneration(
                training_session_id="session",
                policy_step=0,
                generation_id=f"step-00000000-{'a' * 32}",
                adapter_path="/adapter/0",
            ),
            optimizer_state_path="/optimizer",
            rollout_model=RolloutModelSpec(payload={}),
            output_adapter_root="/adapter",
        ),
        input_resolver=resolver,
    )
    operation = _operation("fb", "forward_backward", 0)
    first = await handler(
        ForwardBackwardRequest(
            run_id="run",
            request_id="fb",
            sequence_id=0,
            batch=input_object,
            loss=LossConfig(name="cispo"),
            retain_packed_input=True,
        ),
        operation,
        (),
    )
    capture = first.packed_input_capture
    assert capture is not None and capture.content_sha256 == "b" * 64
    assert capture.input_object == input_object
    assert handler.durable_contribution_inputs() == (("fb", input_object),)
    first_image = runtime.packed
    assert runtime.released == [first_image]
    await handler(
        OptimStepRequest(
            run_id="run",
            request_id="optim",
            sequence_id=1,
            optimizer=AdamConfig(learning_rate=1e-5),
        ),
        _operation("optim", "optim_step", 1, output=1),
        ("fb",),
    )
    assert runtime.released == [first_image]

    replay_image = _packed_batch(content_sha256="b" * 64, batch_id="replacement-image")
    runtime.packed = replay_image
    replay_operation = _operation("replay", "forward_backward", 2, parent=1)
    await handler(
        ForwardBackwardRequest(
            run_id="run",
            request_id="replay",
            sequence_id=2,
            batch=capture,
            loss=LossConfig(name="cispo"),
        ),
        replay_operation,
        (),
    )
    await handler(
        OptimStepRequest(
            run_id="run",
            request_id="replay-optim",
            sequence_id=3,
            optimizer=AdamConfig(learning_rate=1e-5),
        ),
        _operation("replay-optim", "optim_step", 3, parent=1, output=2),
        ("replay",),
    )
    assert runtime.released == [first_image, replay_image]
    await handler.discard_input_object(input_object)
    await handler.discard_input_object(input_object)
    assert runtime.released == [first_image, replay_image]
    assert resolver.calls == [
        (input_object, operation),
        (input_object, replay_operation),
    ]


@pytest.mark.asyncio
async def test_slot_coordinator_serializes_gpu_work_before_result_settlement() -> None:
    runtime = _Runtime()
    trainer = _Trainer()
    trainer.fail_optimizer = False
    slot = MegatronSlotCoordinator(  # type: ignore[arg-type]
        runtime, trainer, command_timeout_s=1_200
    )
    runs = []
    for index in range(4):
        run_id = f"run-{index}"
        session_id = f"session-{index}"
        generation = TrainerGeneration(
            training_session_id=session_id,
            policy_step=0,
            generation_id=f"step-00000000-{index:032x}",
            adapter_path=f"/adapter/{index}",
        )
        runs.append(
            await slot.register_run(
                MegatronOperationConfig(
                    run_id=run_id,
                    training_session_id=session_id,
                    adapter=AdapterSpec(
                        rank=index + 1,
                        target_modules=("q_proj",) if index % 2 else ("v_proj",),
                    ),
                    source=generation,
                    optimizer_state_path=f"/optimizer/{index}",
                    rollout_model=RolloutModelSpec(payload={}),
                    output_adapter_root=f"/adapter/{index}",
                ),
                portable_archive=(
                    _portable_archive(generation, source_ref="local://portable")
                    if index == 0
                    else None
                ),
                restore_optimizer=index != 0,
            )
        )

    result_gate = asyncio.Event()
    trainer.result_gates = {run.run_id: result_gate for run in runs}
    tasks = [
        asyncio.create_task(
            run.worker.execute(
                ForwardRequest(
                    run_id=run.run_id,
                    request_id="forward",
                    sequence_id=0,
                    batch=_batch(),
                    loss=LossConfig(name="cispo"),
                ),
                OperationRef(
                    run_id=run.run_id,
                    operation_id=f"{run.run_id}-forward",
                    sequence_id=0,
                    learner_parent_version=0,
                    kind="forward",
                ),
            )
        )
        for run in runs
    ]
    while len(trainer.executed_runs) < len(runs):
        await asyncio.sleep(0)
    assert all(not task.done() for task in tasks)
    result_gate.set()
    outcomes = await asyncio.gather(*tasks)

    assert {outcome.status for outcome in outcomes} == {"succeeded"}
    assert trainer.max_active == 1
    assert runs[0].portable_install is not None
    assert runs[0].portable_install.restore_optimizer is False
    assert all(run.portable_install is None for run in runs[1:])
    assert set(trainer.executed_runs) == {run.run_id for run in runs}
    assert trainer.registered_adapters == {
        f"run-{index}": (
            index + 1,
            ("q_proj",) if index % 2 else ("v_proj",),
        )
        for index in range(4)
    }
    assert set(trainer.registered_timeouts.values()) == {1_200}
    await slot.drain_run("run-0")
    with pytest.raises(KeyError):
        slot.resolve_run("run-0")
    await slot.aclose()


@pytest.mark.asyncio
async def test_megatron_backend_uses_distributed_service_without_training_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Service:
        def __init__(self) -> None:
            self.prefetches = 0
            self.start_configs = []

        def prefetch_trainer(self) -> None:
            self.prefetches += 1

        async def start_openai_server(self, *, config):
            self.start_configs.append(config)
            return "127.0.0.1", 4321

    backend = MegatronBackend(path=str(tmp_path), enable_expert_replay=False)
    model = TrainableModel(
        run_name="local-run",
        name="local-run",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    service = _Service()
    service_requests = []
    binding_requests = []

    async def get_service(requested_model):
        service_requests.append(requested_model)
        return service

    async def bind_owned_training_run(bound_model, openai_config):
        binding_requests.append((bound_model, openai_config))

    monkeypatch.setattr(backend, "_get_service", get_service)
    monkeypatch.setattr(backend, "_bind_owned_training_run", bind_owned_training_run)

    base_url, api_key = await backend._prepare_backend_for_training(model)

    assert binding_requests == []
    assert backend._training_binding is None
    assert service_requests == [model, model]
    assert service.prefetches == 1
    assert len(service.start_configs) == 1
    assert service.start_configs[0]["server_args"]["api_key"] == api_key
    assert base_url == "http://127.0.0.1:4321/v1"


@pytest.mark.asyncio
async def test_megatron_backend_lowers_rl_and_sft_through_one_bound_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Publisher:
        def __init__(self) -> None:
            self.receipts = {}
            self.exact_leases = []

        async def aclose(self) -> None:
            pass

        def activated_publication(self, alias, learner_version=None):
            if learner_version is None:
                matches = [
                    receipt
                    for (candidate, _step), receipt in self.receipts.items()
                    if candidate == alias
                ]
                return max(matches, key=lambda item: item.learner_version, default=None)
            return self.receipts.get((alias, learner_version))

        @asynccontextmanager
        async def exact_publication_lease(self, alias, learner_version):
            self.exact_leases.append((alias, learner_version))
            receipt = self.activated_publication(alias, learner_version)
            if receipt is None:
                raise RuntimeError("publication is not activated")
            yield receipt

        async def prune_versioned_adapters(self, alias, *, retain_steps):
            self.receipts = {
                key: receipt
                for key, receipt in self.receipts.items()
                if key[0] != alias or key[1] in retain_steps
            }

        async def plan_artifacts(self, *_args, **_kwargs):
            return SimpleNamespace()

        async def save_weights_for_sampler(
            self,
            request,
            operation,
            generation,
            **_kwargs,
        ):
            alias = request.publication.model_alias
            assert alias is not None
            result = SamplerWeightsResult(
                operation_id=operation.operation_id,
                checkpoint=CheckpointRef(
                    run_id=operation.run_id,
                    learner_version=generation.policy_step,
                    checkpoint_id=generation.generation_id,
                ),
                lora=f"{alias}@{generation.policy_step}",
                metrics={POLICY_ACTIVATION_LAG_METRIC: 0.25},
            )
            receipt = MegatronSamplerPublicationReceipt(
                operation_id=operation.operation_id,
                request_id=request.request_id,
                publication_mode=request.publication.mode,
                requested_public_alias=alias,
                runtime_model_name=alias,
                runtime_lora_name=result.lora,
                serving_generation_id=generation.generation_id,
                learner_version=generation.policy_step,
                policy_activation_timing=MegatronPolicyActivationTiming(
                    trainer_completed_monotonic_s=1.0,
                    serving_activated_monotonic_s=1.25,
                ),
                inference_update_usage=MegatronInferenceUpdateUsage(
                    staging_s=0.1,
                    apply_s=0.15,
                ),
                holder_update_sequence=generation.policy_step,
                holder_update_id=f"update-{generation.policy_step}",
                retained=(
                    MegatronRetainedState(
                        owner_id=f"lora/{operation.operation_id}",
                        resource="lora",
                        bytes=1024,
                        work_fingerprint="f" * 64,
                    ),
                ),
                result=result,
            )
            self.receipts[(alias, generation.policy_step)] = receipt
            return receipt

    class _OutcomeSink:
        def __init__(self) -> None:
            self.admissions = []
            self.kinds = []

        async def retain_admission(self, request, admission):
            assert request.sequence_id == admission.ref.sequence_id
            self.admissions.append(admission.ref.kind)

        async def retain_outcome(self, request, outcome):
            assert request.sequence_id == outcome.operation.sequence_id
            self.kinds.append(outcome.operation.kind)

    class _Queue:
        def __init__(self, materialized: TrajectoryGroup) -> None:
            self.materialized = materialized
            self.materializations = 0
            self.marked = []
            self.released = []

        async def materialize_selection(self, selection):
            assert selection.queue is self
            self.materializations += 1
            return self.materialized

        async def mark_packed(self, selections, generation_id):
            self.marked.append((tuple(selections), generation_id))

        async def release_selections(
            self, selections, *, disposition, generation_id=None
        ):
            assert trainer.optimizer_jobs
            self.released.append((tuple(selections), disposition, generation_id))

    class _SftTokenizer:
        def tokenize(self, model, batch):
            assert model.base_model == "test-model"
            assert len(batch.trajectories) == 1
            tensors = {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "labels": torch.tensor([[-100, 2]], dtype=torch.long),
            }
            return SFTBatch(
                trajectory_tensors=[tensors],
                learning_rate=0.0,
                num_trajectories=1,
                num_tokens=2,
                num_trainable_tokens=1,
            )

    runtime = _Runtime()
    trainer = _Trainer()
    trainer.fail_optimizer = False
    publisher = _Publisher()
    sink = _OutcomeSink()
    slot = MegatronSlotCoordinator(  # type: ignore[arg-type]
        runtime,
        trainer,
        publisher=publisher,  # type: ignore[arg-type]
    )
    generation = TrainerGeneration(
        training_session_id="session",
        policy_step=0,
        generation_id=f"step-00000000-{'a' * 32}",
        adapter_path="/adapter/0",
    )
    model = TrainableModel(
        run_name="registered-slot-run",
        name="registered-slot-run",
        run_id="run",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    model.inference_base_url = "http://inference.test/v1"
    model.inference_api_key = "test-key"
    model.inference_model_name = "registered-slot-run"
    assert model.run_id == "run"
    operation_config = MegatronOperationConfig(
        run_id="run",
        training_session_id="session",
        adapter=AdapterSpec(rank=8, target_modules=("q_proj",)),
        source=generation,
        initial_operation_sequence=11,
        optimizer_state_path="/optimizer",
        rollout_model=RolloutModelSpec.from_model(model),
        output_adapter_root="/adapter",
    )
    run = await slot.register_run(operation_config)
    slot._runs["run"].handler._sft_tokenizer = _SftTokenizer()  # type: ignore[assignment]
    binding = MegatronRunBinding(
        run=run,
        config=operation_config,
        coordinator=slot,
        publisher=publisher,  # type: ignore[arg-type]
        outcome_sink=sink,
    )
    backend = MegatronBackend(path=str(tmp_path), training_binding=binding)
    monkeypatch.setattr(TrainableModel, "_get_wandb_run", lambda _self: None)
    monkeypatch.setattr(
        "art.megatron.backend.get_megatron_runtime_config",
        lambda: SimpleNamespace(packed_sequence_length=8),
    )
    await model.register(backend)
    assert model.run_id == "run"
    assert model.get_inference_name() == "registered-slot-run"
    assert backend.supports_async_pipeline_packing(model)

    materialized = TrajectoryGroup(
        [
            Trajectory(reward=1.0, initial_policy_version=0),
            Trajectory(reward=0.0, initial_policy_version=0),
        ]
    )
    queue = _Queue(materialized)
    selection = DistributedTrajectorySelection(queue, SimpleNamespace())  # type: ignore[arg-type]
    group = TrajectoryGroup()
    group._distributed_lease = selection

    result = await backend.train(
        model,
        [group],
        learning_rate=1e-5,
        save_checkpoint=False,
    )

    assert result.step == 1
    assert model.get_inference_name() == "registered-slot-run@1"
    assert queue.materializations == 1
    assert len(runtime.pack_requests) == 1
    assert runtime.pack_requests[0].tokenized_batch is None
    assert len(queue.marked) == 1
    assert queue.released[0][1] == "consumed"
    assert trainer.optimizer_jobs[-1].contributing_forward_backward_operation_ids
    async with backend.adapter_lease(model, 1):
        assert model.get_inference_name() == "registered-slot-run@1"
    assert publisher.exact_leases == []
    async with backend.exact_adapter_lease(model, 1):
        assert model.get_inference_name() == "registered-slot-run@1"
    assert publisher.exact_leases == [("registered-slot-run", 1)]
    client = await backend.training_client(model)
    assert client.operation_ids == ()

    sft_metrics = [
        metrics
        async for metrics in backend._train_sft(
            model,
            [Trajectory()],
            TrainSFTConfig(learning_rate=2e-5, batch_size=1),
            {},
        )
    ]
    assert sft_metrics[0][TRAIN_GRADIENT_STEPS_KEY] == 1.0
    assert len(runtime.pack_requests) == 2
    sft_pack = runtime.pack_requests[-1]
    assert sft_pack.trajectory_groups == ()
    assert sft_pack.tokenized_loss == "cross_entropy"
    assert sft_pack.tokenized_batch is not None
    assert len(sft_pack.tokenized_batch.datums) == 1
    assert sft_pack.tokenized_batch.datums[0].input_tokens == (1, 2)
    assert sft_pack.tokenized_batch.datums[0].target_tokens == ((2,), (0,))
    assert sft_pack.tokenized_batch.datums[0].weights == ((1.0,), (0.0,))
    assert trainer.forward_backward_jobs[-1].loss.name == "cross_entropy"
    assert trainer.optimizer_jobs[-1].operation.kind == "optim_step"
    assert client.projected_learner_version == 2
    assert client.operation_ids == ()
    assert sink.kinds == [
        "forward_backward",
        "optim_step",
        "save_sampler",
        "forward_backward",
        "optim_step",
        "save_sampler",
    ]
    assert sink.admissions == sink.kinds

    optimizer_count = len(trainer.optimizer_jobs)
    runtime.packed = _packed_batch(
        batch_id="zero-work",
        loss_bearing_tokens=0,
        trainable_assistant_tokens=0,
    )
    zero_rl = await backend.train(
        model,
        [TrajectoryGroup([Trajectory(initial_policy_version=2)])],
        learning_rate=1e-5,
        save_checkpoint=False,
    )
    assert zero_rl.step == 2
    assert zero_rl.metrics[TRAIN_GRADIENT_STEPS_KEY] == 0.0
    assert len(trainer.optimizer_jobs) == optimizer_count

    class _ZeroSftTokenizer:
        def tokenize(self, _model, _batch):
            return SFTBatch(
                trajectory_tensors=[],
                learning_rate=0.0,
                num_trajectories=0,
                num_tokens=0,
                num_trainable_tokens=0,
            )

    slot._runs["run"].handler._sft_tokenizer = _ZeroSftTokenizer()  # type: ignore[assignment]
    pack_count = len(runtime.pack_requests)
    zero_sft = [
        metrics
        async for metrics in backend._train_sft(
            model,
            [Trajectory()],
            TrainSFTConfig(learning_rate=2e-5, batch_size=1),
            {},
        )
    ]
    assert len(zero_sft) == 1
    assert zero_sft[0]["data/sft_zero_work"] == 1.0
    assert zero_sft[0]["data/step_num_trajectories"] == 1.0
    assert zero_sft[0]["data/step_trainable_assistant_tokens"] == 0.0
    assert zero_sft[0][TRAIN_GRADIENT_STEPS_KEY] == 0.0
    assert len(trainer.optimizer_jobs) == optimizer_count
    assert len(runtime.pack_requests) == pack_count

    runtime.packed = _packed_batch(batch_id="pipeline")
    pipeline_materialized = TrajectoryGroup(
        [Trajectory(reward=1.0, initial_policy_version=2)]
    )
    pipeline_queue = _Queue(pipeline_materialized)
    pipeline_group = TrajectoryGroup()
    pipeline_group._distributed_lease = DistributedTrajectorySelection(
        pipeline_queue, SimpleNamespace()
    )  # type: ignore[arg-type]
    pack_count = len(runtime.pack_requests)
    lookahead_count = len(trainer.cp_lookaheads)
    context = await backend.prepare_pipeline_commands(
        model,
        [pipeline_group],
        learner_parent_version=2,
        train_kwargs={"learning_rate": 1e-5, "save_checkpoint": False},
    )
    assert context is not None
    assert pipeline_queue.materializations == 1
    assert len(runtime.pack_requests) == pack_count
    assert len(trainer.cp_lookaheads) == lookahead_count
    pipeline_result = await context.complete(None, None)
    assert pipeline_result.step == 3
    assert pipeline_queue.materializations == 1
    assert len(runtime.pack_requests) == pack_count + 1
    assert len(trainer.cp_lookaheads) == lookahead_count + 1
    assert trainer.cp_lookaheads[-1] == (runtime.packed.leases, None)
    assert client.operation_ids == ()
    assert not backend._services

    await backend.close()
    await slot.aclose()


@pytest.mark.asyncio
async def test_slot_coordinator_summarizes_private_residency_evidence() -> None:
    runtime = _Runtime()
    trainer = _Trainer()
    slot = MegatronSlotCoordinator(runtime, trainer)  # type: ignore[arg-type]
    run = await slot.register_run(
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
            adapter=AdapterSpec(rank=8, target_modules=("q_proj",)),
            source=TrainerGeneration(
                training_session_id="session",
                policy_step=0,
                generation_id=f"step-00000000-{'a' * 32}",
                adapter_path="/adapter/0",
            ),
            optimizer_state_path="/optimizer",
            rollout_model=RolloutModelSpec(payload={}),
            output_adapter_root="/adapter",
        )
    )
    operation = _operation("forward", "forward", 0)
    outcome = await run.worker.execute(
        ForwardRequest(
            run_id="run",
            request_id="forward",
            sequence_id=0,
            batch=_batch(),
            loss=LossConfig(name="cispo"),
        ),
        operation,
    )

    assert outcome.status == "succeeded"
    detail = slot.residency_evidence("run", "forward")
    summary = slot.residency_summary("run", "forward")
    assert detail is not None
    assert isinstance(summary, MegatronOperationResidencySummary)
    assert summary.requested_components == ("weights",)
    assert summary.topology.rank_count == 1
    assert summary.all_ranks_reported
    assert summary.all_requested_components_l1_ready
    assert summary.l1_ready_bytes == 1024
    assert summary.components[0].component == "weights"
    assert summary.components[0].l1_ready_rank_count == 1
    assert summary.components[0].l1_reload_rank_count == 1
    assert summary.components[0].l1_reload_bytes == 1024
    assert summary.components[0].l1_reload_sources == ("l2_cpu",)
    assert summary.components[0].all_ranks_reloaded_after_eviction
    assert summary.components[1].component == "optimizer"
    assert summary.components[1].observed_rank_count == 0
    assert summary.tiers[0].tier == "l1_gpu"
    assert summary.tiers[0].ready_bytes == 1024
    assert summary.tiers[1].tier == "l2_cpu"
    assert summary.tiers[1].ready_bytes == 1024
    canonical = json.dumps(
        detail, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    assert summary.detailed_evidence_sha256 == hashlib.sha256(canonical).hexdigest()
    assert "rank_evidence" not in summary.model_dump(mode="json")

    run.worker.retire("forward")
    assert slot.residency_summary("run", "forward") is None
    await slot.aclose()


@pytest.mark.asyncio
async def test_sampler_publication_receipt_lives_until_operation_retirement() -> None:
    class _EvidenceSink:
        def __init__(self) -> None:
            self.attempts = 0
            self.evidence = []

        def retain_residency_evidence(self, evidence):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("evidence store unavailable")
            self.evidence.append(evidence)

    class _Checkpoints:
        async def save_weights_for_sampler(self, request, operation, generation):
            result = SamplerWeightsResult(
                operation_id=operation.operation_id,
                metrics={POLICY_ACTIVATION_LAG_METRIC: 1.25},
                checkpoint=CheckpointRef(
                    run_id=operation.run_id,
                    learner_version=generation.policy_step,
                    checkpoint_id="public-checkpoint",
                ),
                lora="public-lora",
            )
            return MegatronSamplerPublicationReceipt(
                operation_id=operation.operation_id,
                request_id=request.request_id,
                publication_mode=request.publication.mode,
                requested_public_alias=request.publication.model_alias,
                runtime_model_name="paired-model",
                runtime_lora_name="paired-model@step-0",
                serving_generation_id=generation.generation_id,
                learner_version=generation.policy_step,
                policy_activation_timing=MegatronPolicyActivationTiming(
                    trainer_completed_monotonic_s=10.0,
                    serving_activated_monotonic_s=11.25,
                ),
                inference_update_usage=MegatronInferenceUpdateUsage(
                    staging_s=0.25,
                    apply_s=1.0,
                ),
                holder_update_sequence=3,
                holder_update_id="holder-update-3",
                retained=(
                    MegatronRetainedState(
                        owner_id="lora-owner/run/publish",
                        resource="lora",
                        bytes=4096,
                        work_fingerprint="f" * 64,
                        expires_at=datetime(2026, 8, 30, tzinfo=UTC),
                    ),
                ),
                result=result,
            )

    runtime = _Runtime()
    trainer = _Trainer()
    evidence_sink = _EvidenceSink()
    slot = MegatronSlotCoordinator(  # type: ignore[arg-type]
        runtime, trainer, operation_evidence_sink=evidence_sink
    )
    run = await slot.register_run(
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
            adapter=AdapterSpec(rank=8, target_modules=("q_proj",)),
            source=TrainerGeneration(
                training_session_id="session",
                policy_step=0,
                generation_id=f"step-00000000-{'a' * 32}",
                adapter_path="/adapter/0",
            ),
            optimizer_state_path="/optimizer",
            rollout_model=RolloutModelSpec(payload={}),
            output_adapter_root="/adapter",
        ),
        checkpoints=_Checkpoints(),  # type: ignore[arg-type]
    )
    outcome = await run.worker.execute(
        SaveWeightsForSamplerRequest(
            run_id="run",
            request_id="publish-request",
            sequence_id=0,
            checkpoint_name="step-0",
            ttl_seconds=60,
            publication=SamplerPublication(
                mode="in_flight_lora",
                model_alias="public-policy",
            ),
        ),
        _operation("publish", "save_sampler", 0),
    )

    assert outcome.status == "succeeded"
    receipt = slot.sampler_publication_receipt("run", "publish")
    assert receipt is not None
    assert receipt.requested_public_alias == "public-policy"
    assert receipt.retained[0].owner_id == "lora-owner/run/publish"
    with pytest.raises(RuntimeError, match="evidence store unavailable"):
        run.worker.retire("publish")
    assert slot.sampler_publication_receipt("run", "publish") is receipt
    assert (
        await run.worker.execute(
            SaveWeightsForSamplerRequest(
                run_id="run",
                request_id="publish-request",
                sequence_id=0,
                checkpoint_name="step-0",
                ttl_seconds=60,
                publication=SamplerPublication(
                    mode="in_flight_lora",
                    model_alias="public-policy",
                ),
            ),
            _operation("publish", "save_sampler", 0),
        )
        == outcome
    )
    run.worker.retire("publish")
    assert evidence_sink.evidence[0].operation_id == "publish"
    assert evidence_sink.evidence[0].requested_components == ()
    assert slot.sampler_publication_receipt("run", "publish") is None
    await slot.aclose()


def test_external_sampler_receipt_requires_manifest_without_holder_update() -> None:
    generation = TrainerGeneration(
        training_session_id="session",
        policy_step=0,
        generation_id=f"step-00000000-{'a' * 32}",
        adapter_path="/adapter/0",
    )
    operation = _operation("publish", "save_sampler", 0)
    request = SaveWeightsForSamplerRequest(
        run_id="run",
        request_id="publish-request",
        sequence_id=0,
        checkpoint_name="step-0",
        publication=SamplerPublication(
            mode="external_lora",
            model_alias="public-policy",
        ),
    )
    manifest = ImmutablePublicationRef(
        locator="caios://manifest",
        size_bytes=1024,
        sha256="a" * 64,
    )
    result = SamplerWeightsResult(
        operation_id="publish",
        checkpoint=CheckpointRef(
            run_id="run",
            learner_version=0,
            checkpoint_id="step-0",
        ),
        lora=manifest.locator,
        external_lora=ExternalLoraReceipt(
            generation_id=generation.generation_id,
            active_alias="public-policy",
            manifest=manifest,
            shards=(
                ImmutablePublicationRef(
                    locator="caios://shard-0",
                    size_bytes=4096,
                    sha256="b" * 64,
                ),
            ),
        ),
    )
    receipt = MegatronSamplerPublicationReceipt(
        operation_id="publish",
        request_id="publish-request",
        publication_mode="external_lora",
        requested_public_alias="public-policy",
        runtime_model_name="model@revision",
        serving_generation_id=generation.generation_id,
        learner_version=0,
        retained=(
            MegatronRetainedState(
                owner_id="external-lora:manifest",
                resource="lora",
                bytes=4096,
                work_fingerprint="f" * 64,
            ),
        ),
        result=result,
    )

    receipt.validate_command(request, operation, generation)
    with pytest.raises(RuntimeError, match="wrong resource kind"):
        receipt.model_copy(
            update={"result": result.model_copy(update={"external_lora": None})}
        ).validate_command(request, operation, generation)
    with pytest.raises(RuntimeError, match="holder update evidence"):
        receipt.model_copy(update={"holder_update_sequence": 1}).validate_command(
            request, operation, generation
        )


@pytest.mark.asyncio
async def test_slot_migration_fences_replays_and_releases_one_run() -> None:
    runtime = _Runtime()
    trainer = _Trainer()
    trainer.fail_optimizer = False
    input_object = _input_object("fb-replay")
    slot = MegatronSlotCoordinator(  # type: ignore[arg-type]
        runtime, trainer, input_resolver=_InputResolver(_batch())
    )
    run = await slot.install_migration_run(
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
            adapter=AdapterSpec(rank=8, target_modules=("q_proj",)),
            source=TrainerGeneration(
                training_session_id="session",
                policy_step=2,
                generation_id=f"step-00000002-{'a' * 32}",
                adapter_path="/adapter/2",
            ),
            initial_operation_sequence=4,
            optimizer_state_path="/optimizer/2",
            rollout_model=RolloutModelSpec(payload={}),
            output_adapter_root="/adapter",
        ),
        restore_id="restore-1",
    )
    with pytest.raises(KeyError):
        slot.resolve_run("run")
    request = ForwardBackwardRequest(
        run_id="run",
        request_id="fb-replay",
        sequence_id=4,
        batch=input_object,
        loss=LossConfig(name="cispo"),
    )
    operation = _operation("fb-replay", "forward_backward", 4, parent=2)

    replays = (
        MegatronMigrationReplay(
            request=request,
            admission=CommandAdmission(ref=operation),
        ),
    )
    outcomes = await slot.replay_migration_operations(
        "run",
        "restore-1",
        replays,
    )
    assert [outcome.status for outcome in outcomes] == ["succeeded"]
    assert (
        await slot.replay_migration_operations("run", "restore-1", replays) == outcomes
    )
    result = outcomes[0]
    assert result.status == "succeeded"
    assert isinstance(result.result, ForwardBackwardResult)
    assert result.result.packed_input_capture is not None
    source_fence = MegatronMigrationFence(
        fence_id="source-fence",
        run_id="run",
        generation=TrainerGeneration(
            training_session_id="session",
            policy_step=2,
            generation_id=f"step-00000002-{'a' * 32}",
            adapter_path="/source-adapter/2",
        ),
        optimizer_state_path="/optimizer/2",
        next_operation_sequence=5,
        open_contributions=(
            MegatronMigrationContribution(
                operation_id="fb-replay",
                input_object=input_object,
            ),
        ),
    )
    await slot.activate_migration_run("run", "restore-1", source_fence)
    assert await slot.activate_migration_run("run", "restore-1", source_fence) == run
    with pytest.raises(RuntimeError, match="source fence changed"):
        await slot.activate_migration_run(
            "run",
            "restore-1",
            source_fence.model_copy(update={"fence_id": "other-source"}),
        )
    assert slot.resolve_run("run") == run

    fence = await slot.fence_and_quiesce_run("run", "fence-2")
    assert fence.generation.policy_step == 2
    assert fence.next_operation_sequence == 5
    assert [item.operation_id for item in fence.open_contributions] == ["fb-replay"]
    assert await slot.fence_and_quiesce_run("run", "fence-2") == fence
    with pytest.raises(KeyError):
        slot.resolve_run("run")

    await slot.resume_migration_source("run", "fence-2")
    assert slot.resolve_run("run") == run
    with pytest.raises(RuntimeError, match="already resumed"):
        await slot.fence_and_quiesce_run("run", "fence-2")
    fence = await slot.fence_and_quiesce_run("run", "fence-3")
    await slot.release_migration_source(fence)
    await slot.release_migration_source(fence)
    assert trainer.migration_releases == ["run"]
    assert runtime.released == [runtime.packed]
    with pytest.raises(KeyError):
        slot.resolve_run("run")

    abort_config = MegatronOperationConfig(
        run_id="abort-run",
        training_session_id="abort-session",
        adapter=AdapterSpec(rank=8, target_modules=("q_proj",)),
        source=TrainerGeneration(
            training_session_id="abort-session",
            policy_step=0,
            generation_id=f"step-00000000-{'b' * 32}",
            adapter_path="/adapter/abort",
        ),
        optimizer_state_path="/optimizer/abort",
        rollout_model=RolloutModelSpec(payload={}),
        output_adapter_root="/adapter",
    )
    abort_archive = _portable_archive(
        abort_config.source,
        source_ref="caios://source-a",
    )
    abort = await slot.install_migration_run(
        abort_config,
        restore_id="abort-restore",
        portable_archive=abort_archive,
    )
    assert abort.portable_install is not None
    assert abort.run_id == "abort-run"
    await slot.abort_migration_run("abort-run", "abort-restore")
    await slot.abort_migration_run("abort-run", "abort-restore")
    retry_archive = _portable_archive(
        abort_config.source,
        source_ref="wandb://source-b",
    )
    assert retry_archive.archive_sha256 == abort_archive.archive_sha256
    assert retry_archive.receipt_sha256 != abort_archive.receipt_sha256
    retry = await slot.install_migration_run(
        abort_config,
        restore_id="abort-restore",
        portable_archive=retry_archive,
    )
    assert retry.run_id == "abort-run"
    await slot.abort_migration_run("abort-run", "abort-restore")
    with pytest.raises(RuntimeError, match="configuration changed"):
        await slot.install_migration_run(
            abort_config.model_copy(
                update={"optimizer_state_path": "/optimizer/changed"}
            ),
            restore_id="abort-restore",
            portable_archive=retry_archive,
        )
    await slot.aclose()


@pytest.mark.asyncio
async def test_control_commands_advance_shared_run_sequence() -> None:
    run = MonarchTrainerRun.__new__(MonarchTrainerRun)
    run._lock = asyncio.Lock()
    state = SimpleNamespace(
        learner_version=2,
        next_operation_sequence=4,
        open_forward_backward_ids=[],
    )
    run._command_runs = {"run": state}

    await run.record_control_command(_operation("save", "save_state", 4, parent=2), 2)
    await run.record_control_command(
        _operation("load", "load_state", 5, parent=2, output=3), 3
    )

    assert state.next_operation_sequence == 6
    assert state.learner_version == 3


def test_rank_gpu_service_uses_exclusive_duration_and_exact_count() -> None:
    run = MonarchTrainerRun.__new__(MonarchTrainerRun)
    run.runtime_spec = SimpleNamespace(trainer_mesh=SimpleNamespace(ranks=(0, 1, 2, 3)))
    results = [
        {
            "command_status": "succeeded",
            "rank": rank,
            "gpu_service_ns": value,
        }
        for rank, value in enumerate((10, 12, 11, 9))
    ]

    assert run._aggregate_gpu_service(results) == (12, 4)
