import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from art.distributed.art_runtime import DistributedPackedBatch
from art.distributed.data_plane import (
    PackedBatchLeaseSet,
    PackedBatchRef,
    PrefixTreePackingStatsSpec,
    TensorSpec,
)
from art.distributed.rollout import RolloutModelSpec
from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.megatron.operation_handler import (
    MegatronOperationConfig,
    MegatronOperationHandler,
    MegatronRetainedState,
    MegatronSamplerPublicationReceipt,
)
from art.megatron.runtime.monarch import MonarchTrainerRun
from art.megatron.runtime.specs import TrainerCommandRunState, TrainerGeneration
from art.megatron.slot_coordinator import (
    MegatronMigrationContribution,
    MegatronMigrationFence,
    MegatronMigrationReplay,
    MegatronSlotCoordinator,
)
from art.training import (
    AdamConfig,
    CheckpointRef,
    CommandAdmission,
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    LossConfig,
    OperationExecutionError,
    OperationRef,
    OptimStepRequest,
    RlTrajectoryBatch,
    SamplerPublication,
    SamplerWeightsResult,
    SaveWeightsForSamplerRequest,
)
from art.trajectories import TrajectoryGroup


def _packed_batch() -> DistributedPackedBatch:
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
    offset = 0
    tensors = []
    for name, (dtype, item_size) in item_sizes.items():
        byte_count = 8 * item_size
        tensors.append(
            TensorSpec(
                name=name,
                dtype=dtype,
                shape=(1, 8),
                offset=offset,
                byte_count=byte_count,
            )
        )
        offset += byte_count
    ref = PackedBatchRef(
        batch_id="batch",
        owner_actor_id="owner",
        lease_id="lease",
        shared_memory_name="shm",
        owner_process_id=1,
        tensors=tuple(tensors),
        num_sequences=1,
        sequence_length=8,
        byte_count=offset,
        storage_byte_count=offset,
        pixel_values_present=(False,),
        image_grid_thw_present=(False,),
        prefix_tree_packing_stats=PrefixTreePackingStatsSpec(
            logical_tokens=7, physical_tokens=8
        ),
    )
    return DistributedPackedBatch(
        leases=PackedBatchLeaseSet(ref=ref, host_refs={"host": ref}),
        packed_group_shapes=(),
        trainable_assistant_tokens=4,
        loss_bearing_tokens=4,
        non_padding_tokens=7,
        packing_generation_id="packing",
    )


class _Runtime:
    def __init__(self) -> None:
        self.packed = _packed_batch()
        self.released: list[DistributedPackedBatch] = []

    async def pack(self, _request):
        return self.packed

    async def release_batch(self, batch):
        self.released.append(batch)


class _Trainer:
    def __init__(self) -> None:
        topology = SimpleNamespace(tp=1, cp=1, pp=1)
        self.runtime_spec = SimpleNamespace(
            fingerprint="runtime",
            packed_sequence_length=8,
            enable_moe_routing_replay=False,
            trainer_mesh=SimpleNamespace(ranks=(0,), topology=topology),
        )
        self.fail_optimizer = True
        self.active = 0
        self.max_active = 0
        self.executed_runs: list[str] = []
        self.registered_runs: set[str] = set()
        self.run_states: dict[str, TrainerCommandRunState] = {}
        self.migration_releases: list[str] = []

    def register_command_run(self, run_spec) -> None:
        self.registered_runs.add(run_spec.run_id)
        self.run_states[run_spec.run_id] = TrainerCommandRunState(
            run_id=run_spec.run_id,
            training_session_id=run_spec.training_session_id,
            learner_version=run_spec.initial_learner_version,
            next_operation_sequence=run_spec.initial_operation_sequence,
            open_forward_backward_operation_ids=(),
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

    async def forward(self, job, _batch):
        await self._enter(job.run_id)
        self._advance(job)
        return {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "logical_nonpadding_tokens": 7,
            "executed_token_equivalents": 8,
            "gpu_count": 4,
            "gpu_service_ns": 12_000_000,
        }

    async def forward_backward(self, job, _batch):
        await self._enter(job.run_id)
        self._advance(job, backward=True)
        return {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "logical_nonpadding_tokens": 7,
            "executed_token_equivalents": 8,
            "gpu_count": 4,
            "gpu_service_ns": 12_000_000,
        }

    async def optim_step(self, job):
        if self.fail_optimizer:
            raise RuntimeError("optimizer failed")
        self._advance(job, optimizer=True)
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": (
                job.contributing_forward_backward_operation_ids
            ),
            "gpu_count": 4,
            "gpu_service_ns": 3_000_000,
        }


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
async def test_handler_retains_f_b_input_until_optimizer_commit() -> None:
    runtime = _Runtime()
    trainer = _Trainer()
    handler = MegatronOperationHandler(
        runtime,  # type: ignore[arg-type]
        trainer,
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
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
    assert handler.retained_contribution_inputs() == ()
    assert runtime.released == [runtime.packed]

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
async def test_slot_coordinator_serializes_four_logical_runs() -> None:
    runtime = _Runtime()
    trainer = _Trainer()
    trainer.fail_optimizer = False
    slot = MegatronSlotCoordinator(runtime, trainer)  # type: ignore[arg-type]
    runs = []
    for index in range(4):
        run_id = f"run-{index}"
        session_id = f"session-{index}"
        runs.append(
            await slot.register_run(
                MegatronOperationConfig(
                    run_id=run_id,
                    training_session_id=session_id,
                    source=TrainerGeneration(
                        training_session_id=session_id,
                        policy_step=0,
                        generation_id=f"step-00000000-{index:032x}",
                        adapter_path=f"/adapter/{index}",
                    ),
                    optimizer_state_path=f"/optimizer/{index}",
                    rollout_model=RolloutModelSpec(payload={}),
                    output_adapter_root=f"/adapter/{index}",
                )
            )
        )

    outcomes = await asyncio.gather(
        *(
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
            for run in runs
        )
    )

    assert {outcome.status for outcome in outcomes} == {"succeeded"}
    assert trainer.max_active == 1
    assert set(trainer.executed_runs) == {run.run_id for run in runs}
    await slot.drain_run("run-0")
    with pytest.raises(KeyError):
        slot.resolve_run("run-0")
    await slot.aclose()


@pytest.mark.asyncio
async def test_sampler_publication_receipt_lives_until_operation_retirement() -> None:
    class _Checkpoints:
        async def save_weights_for_sampler(self, request, operation, generation):
            result = SamplerWeightsResult(
                operation_id=operation.operation_id,
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
    slot = MegatronSlotCoordinator(runtime, trainer)  # type: ignore[arg-type]
    run = await slot.register_run(
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
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
    run.worker.retire("publish")
    assert slot.sampler_publication_receipt("run", "publish") is None
    await slot.aclose()


@pytest.mark.asyncio
async def test_slot_migration_fences_replays_and_releases_one_run() -> None:
    runtime = _Runtime()
    trainer = _Trainer()
    trainer.fail_optimizer = False
    slot = MegatronSlotCoordinator(runtime, trainer)  # type: ignore[arg-type]
    run = await slot.install_migration_run(
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
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
        batch=_batch(),
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
                packed_input=result.result.packed_input_capture,
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
    abort = await slot.install_migration_run(
        abort_config,
        restore_id="abort-restore",
    )
    assert abort.run_id == "abort-run"
    await slot.abort_migration_run("abort-run", "abort-restore")
    await slot.abort_migration_run("abort-run", "abort-restore")
    with pytest.raises(RuntimeError, match="already aborted"):
        await slot.install_migration_run(
            abort_config,
            restore_id="abort-restore",
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
