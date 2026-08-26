import asyncio
from contextlib import contextmanager
from threading import Event, Lock, local
from types import MethodType, SimpleNamespace

import pytest
import torch

from art.megatron.optimizer_state import CheckpointFile, OptimizerAdapter
from art.megatron.runtime.monarch import (
    MonarchTrainerActor,
    MonarchTrainerRun,
    _CommandReady,
    _snapshot_readiness_timeout,
)
from art.megatron.runtime.publication import SnapshotRankWritePlan
from art.megatron.runtime.specs import (
    GenerationSnapshotJobSpec,
    TrainerGeneration,
)


def snapshot_job() -> GenerationSnapshotJobSpec:
    generation = TrainerGeneration(
        training_session_id="session",
        policy_step=1,
        generation_id="step-00000001-" + "a" * 32,
        adapter_path="/final/adapter",
    )
    return GenerationSnapshotJobSpec(
        operation_id="snapshot-1",
        run_id="run",
        sequence_id=0,
        training_session_id="session",
        learner_version=1,
        generation=generation,
        optimizer_state_path="/optimizer",
        staging_adapter_path="/staging/adapter",
    )


def bare_trainer(snapshot_call, *, ranks: int) -> MonarchTrainerRun:
    trainer = MonarchTrainerRun.__new__(MonarchTrainerRun)
    trainer._actors = SimpleNamespace(execute_snapshot=snapshot_call)
    trainer._rank_processes = (object(),) * ranks
    trainer._operations = {}
    trainer._snapshot_launches = {}
    trainer._publications = {}
    trainer._operation_sequence_ids = {}
    trainer._next_operation_sequence = 0
    trainer._lock = asyncio.Lock()
    trainer._active_job_id = None
    trainer._active_collective = None
    trainer._validate_operation = lambda _job: None
    trainer._expire_prior_publications = lambda: None
    trainer._retire_publication = lambda _state: None
    trainer._command_timeout_s = lambda: 2.0

    async def complete(self, completed_job, rank_call, _receiver, state, _deadline):
        await rank_call
        state.train_done = True
        state.drain_done = True
        state.future.set_result(())
        self._snapshot_launches.pop(completed_job.operation_id, None)
        return {"operation_id": completed_job.operation_id}

    async def invalidate(self, error, context):
        self.invalidated = (error, context)

    trainer._complete_snapshot_prepare = MethodType(complete, trainer)
    trainer._invalidate_command = MethodType(invalidate, trainer)
    return trainer


def test_deferred_response_inherits_actor_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = local()
    state.device = 1

    def current_device() -> int:
        return getattr(state, "device", 0)

    @contextmanager
    def device(index: int):
        previous = current_device()
        state.device = index
        try:
            yield
        finally:
            state.device = previous

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setenv("LOCAL_RANK", "1")
    delivered = Event()
    started = Event()
    release = Event()
    result = {}

    class Port:
        def send(self, value):
            result.update(value)
            delivered.set()

        def exception(self, error):
            result["error"] = error
            delivered.set()

    actor = MonarchTrainerActor.__new__(MonarchTrainerActor)
    actor._deferred_response_lock = Lock()
    actor._deferred_response_stopping = False
    actor._deferred_response_threads = set()
    actor._snapshot_phase_lock = Lock()
    actor._snapshot_phases = {}
    actor._runtime = SimpleNamespace(rank=1)
    actor._valid = True

    def materialize():
        actor._record_snapshot_phase("snapshot-device", "executor_enter")
        started.set()
        release.wait()
        return {"device": current_device()}

    actor._defer_response(
        Port(),
        materialize,
        name="art-snapshot-prepare-snapshot-device",
        invalidate_on_error=True,
    )
    assert started.wait(1)
    report = actor._snapshot_phase_report("snapshot-device")
    assert report["expected_cuda_device"] == 1
    assert report["phase"]["phase"] == "executor_enter"
    assert report["phase"]["cuda_device"] == 1
    assert report["thread_alive"] is True
    assert "release.wait" in report["thread_stack"]
    release.set()
    assert delivered.wait(1)
    assert result == {"device": 1}


def test_snapshot_waits_for_every_rank_before_advancing_parent() -> None:
    async def run_test() -> None:
        rank_zero_ready = asyncio.Event()
        rank_one_release = asyncio.Event()
        plan_gate = asyncio.Event()
        job = snapshot_job()

        class SnapshotCall:
            async def call(self, job_json, _event_port, ready_port):
                parsed = GenerationSnapshotJobSpec.model_validate_json(job_json)
                for rank, gate in ((0, rank_zero_ready), (1, rank_one_release)):
                    if rank:
                        await gate.wait()
                    ready_port.send(
                        _CommandReady(
                            rank=rank,
                            operation_id=parsed.operation_id,
                            learner_version=parsed.learner_version,
                        ).model_dump(mode="json")
                    )
                    gate.set()
                await plan_gate.wait()
                return {0: {"rank": 0}, 1: {"rank": 1}}

        trainer = bare_trainer(SnapshotCall(), ranks=2)
        start = asyncio.create_task(trainer.start_prepare_snapshot(job))
        await asyncio.wait_for(rank_zero_ready.wait(), 1)
        await asyncio.sleep(0)
        assert not start.done()
        assert trainer._next_operation_sequence == 0

        rank_one_release.set()
        launch = await asyncio.wait_for(start, 1)
        assert trainer._next_operation_sequence == 1
        assert not launch.completion.done()
        plan_gate.set()
        await asyncio.wait_for(launch.completion, 1)

    asyncio.run(run_test())


def test_snapshot_rank_failure_does_not_wait_for_missing_readiness() -> None:
    async def run_test() -> None:
        job = snapshot_job()

        class SnapshotCall:
            async def call(self, job_json, _event_port, ready_port):
                parsed = GenerationSnapshotJobSpec.model_validate_json(job_json)
                ready_port.send(
                    _CommandReady(
                        rank=0,
                        operation_id=parsed.operation_id,
                        learner_version=parsed.learner_version,
                        error_type="RuntimeError",
                        message="rank preparation failed",
                    ).model_dump(mode="json")
                )
                await asyncio.Event().wait()

        trainer = bare_trainer(SnapshotCall(), ranks=2)
        with pytest.raises(RuntimeError, match="rank preparation failed"):
            await asyncio.wait_for(trainer.start_prepare_snapshot(job), 1)
        assert trainer._next_operation_sequence == 0
        assert trainer._active_job_id is None
        assert trainer._active_collective is None

    asyncio.run(run_test())


def test_snapshot_timeout_reports_received_ranks_and_rank_phase() -> None:
    async def run_test() -> None:
        class Values:
            def values(self):
                return (
                    {
                        "rank": 0,
                        "operation_id": "snapshot-1",
                        "phase": {"phase": "readiness_sent"},
                    },
                    {
                        "rank": 1,
                        "operation_id": "snapshot-1",
                        "phase": {"phase": "executor_enter"},
                    },
                )

        class Inspect:
            async def call(self, operation_id):
                assert operation_id == "snapshot-1"
                return Values()

        error = await _snapshot_readiness_timeout(
            SimpleNamespace(inspect_snapshot_phase=Inspect()), snapshot_job(), {0}
        )
        message = str(error)
        assert "received_ranks=[0]" in message
        assert '"rank": 1' in message
        assert '"phase": "executor_enter"' in message

    asyncio.run(run_test())


def test_snapshot_sequence_releases_at_staged_ready() -> None:
    async def run_test() -> None:
        plan_gate = asyncio.Event()
        job = snapshot_job()

        class SnapshotCall:
            async def call(self, job_json, _event_port, ready_port):
                parsed = GenerationSnapshotJobSpec.model_validate_json(job_json)
                ready_port.send(
                    _CommandReady(
                        rank=0,
                        operation_id=parsed.operation_id,
                        learner_version=parsed.learner_version,
                    ).model_dump(mode="json")
                )
                await plan_gate.wait()
                return {0: {"rank": 0}}

        trainer = bare_trainer(SnapshotCall(), ranks=1)
        launch = await asyncio.wait_for(trainer.start_prepare_snapshot(job), 1.0)

        assert not plan_gate.is_set()
        assert trainer._next_operation_sequence == 1
        assert not launch.completion.done()

        plan_gate.set()
        assert (await asyncio.wait_for(launch.completion, 1.0))["operation_id"] == (
            job.operation_id
        )

    asyncio.run(run_test())


def test_snapshot_fences_mutation_before_plan_construction() -> None:
    pytest.importorskip("megatron")
    from art.megatron.runtime.executor import MegatronTrainJobExecutor

    job = snapshot_job()
    order = []
    adapter = OptimizerAdapter(
        identity=job.generation.adapter_path,
        training_session_id=job.training_session_id,
        step=job.learner_version,
        generation_id=job.generation.generation_id,
        files=(
            CheckpointFile(
                name="adapter_config.json", size_bytes=1, sha256="b" * 64
            ),
            CheckpointFile(
                name="adapter_model.safetensors", size_bytes=1, sha256="a" * 64
            ),
        ),
    )
    rank_plan = SnapshotRankWritePlan(
        rank=0,
        generation=job.generation,
        adapter=adapter,
        saves_optimizer=False,
    )

    class Publisher:
        def raise_if_failed(self):
            return None

        def ensure_generation(self, **_kwargs):
            order.append("stage")
            return {}

        def prepare(self, **_kwargs):
            order.append("plan")
            return rank_plan, {}

    executor = MegatronTrainJobExecutor.__new__(MegatronTrainJobExecutor)
    executor._closing = False
    executor._closed = False
    executor._publisher = Publisher()
    executor.runtime = SimpleNamespace(
        resident_training_session_id=job.training_session_id,
        resident_policy_step=job.learner_version,
        adapter_export_dtypes={},
        adapter_export_config={},
        optimizer_state_loaded=False,
        optimizer=None,
    )
    executor.execute_snapshot(
        job,
        SimpleNamespace(),
        lambda: order.append("ready"),
    )
    assert order == ["stage", "ready", "plan"]
