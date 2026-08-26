from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
import pytest
import torch

from art.distributed.data_plane import PackedBatchRef
from art.loss import LossOffPolicyDiagnosticsAccumulator
from art.megatron.context_parallel.types import TrainingStepWorkload
from art.megatron.optimizer_state import CheckpointFile, OptimizerAdapter
from art.megatron.runtime.data_plane import InMemoryPackedBatch
from art.megatron.runtime.executor import (
    _ForwardBackwardResultStagerPool,
    _stage_forward_backward_rank_result,
    _stage_forward_rank_result,
    _stage_sft_rank_result,
)
import art.megatron.runtime.monarch as monarch_module
from art.megatron.runtime.monarch import MonarchTrainerRun, MonarchTrainerSlot
from art.megatron.runtime.specs import (
    RankLocalOptimizerWorkSummary,
    RunOptimizerWorkSummary,
    RunSlotRegistration,
)
from art.megatron.training.command_telemetry import (
    PendingRankCommandTelemetry,
    RankTelemetryTopology,
    aggregate_rank_command_telemetry,
    materialize_rank_telemetry,
    rank_telemetry_statistics,
)
from art.megatron.training.pipeline_schedule import PipelineScheduleTelemetry


def _adapter(session: str) -> OptimizerAdapter:
    return OptimizerAdapter(
        identity="/adapter",
        training_session_id=session,
        step=0,
        generation_id="step-00000000-0123456789abcdef0123456789abcdef",
        files=(
            CheckpointFile(name="adapter_config.json", size_bytes=1),
            CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
        ),
    )


class _JobResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    telemetry: PendingRankCommandTelemetry
    new_logprobs: list[torch.Tensor]


def _workload(tokens: int) -> TrainingStepWorkload:
    return TrainingStepWorkload(
        logical_nonpadding_tokens=tokens,
        loss_bearing_tokens=tokens,
        executed_token_equivalents=tokens,
        nominal_schedule_capacity_tokens=tokens,
        dummy_executed_token_equivalents=0,
        dummy_schedule_capacity_tokens=0,
        real_microbatches=1,
        dummy_microbatches=0,
    )


def _pending_rank_telemetry(
    *,
    rank: int = 0,
    ranks: int = 1,
    total_tokens: int = 3,
    loss: float = 2.0,
    program: Literal["rl", "sft"] = "rl",
) -> PendingRankCommandTelemetry:
    local_tokens = total_tokens // ranks + (rank < total_tokens % ranks)
    zero = torch.zeros(())
    return PendingRankCommandTelemetry(
        program=program,
        backward=True,
        topology=RankTelemetryTopology(
            global_rank=rank,
            dp_cp_ranks=tuple(range(ranks)),
            pp_ranks=(rank,),
        ),
        statistics=rank_telemetry_statistics(
            loss_sum=torch.tensor(loss * local_tokens),
            token_count=torch.tensor(local_tokens),
            correlation=torch.zeros(6),
            kl_sum=zero,
            kl_count=zero,
            diagnostics=LossOffPolicyDiagnosticsAccumulator(),
        ),
        workload=_workload(local_tokens),
        schedules=(
            PipelineScheduleTelemetry(
                pp_rank=0,
                pp_size=1,
                vp_size=1,
                num_microbatches=1,
                real_microbatches=1,
                dummy_microbatches=0,
                micro_batch_size=1,
                seq_length=max(local_tokens, 1),
                microbatch_group_size=1,
            ),
        ),
    )


def _rank_telemetry_payload(
    *,
    rank: int,
    ranks: int,
    total_tokens: int,
    program: Literal["rl", "sft"] = "rl",
) -> dict[str, Any]:
    pending = _pending_rank_telemetry(
        rank=rank,
        ranks=ranks,
        total_tokens=total_tokens,
        program=program,
    )
    return materialize_rank_telemetry(pending, pending.statistics.clone())


class _Port:
    def __init__(self, queue: asyncio.Queue[Any]) -> None:
        self.queue = queue

    def send(self, value: Any) -> None:
        self.queue.put_nowait(value)


class _Receiver:
    def __init__(self, queue: asyncio.Queue[Any]) -> None:
        self.queue = queue

    async def recv(self) -> Any:
        return await self.queue.get()


class _Call:
    def __init__(self, method: Any) -> None:
        self.call = method


class _Job:
    def __init__(
        self, operation_id: str, *, contributions: tuple[str, ...] = ()
    ) -> None:
        self.operation_id = operation_id
        self.expected_learner_version = 3
        self.learner_version = 4
        self.fingerprint = f"fingerprint:{operation_id}"
        self.trainable_token_count = 7
        self.contributing_forward_backward_operation_ids = contributions
        self.batch = "batch-ref"
        self.batch_fingerprint = "sft-batch"
        self.loss = None

    def model_dump_json(self) -> str:
        return self.operation_id


class _Batch:
    def __init__(self) -> None:
        self.ref = "batch-ref"

    def model_dump_json(self) -> str:
        return "batch"


class _SftBatch:
    def __init__(self) -> None:
        self.manifest = SimpleNamespace(fingerprint="sft-batch", num_trainable_tokens=7)

    def model_dump_json(self) -> str:
        return "sft-batch"


class _Supervision:
    def close(self, **_: Any) -> None:
        return None


class _ProcMesh:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class _Actors:
    def __init__(self, ranks: int = 2) -> None:
        self.ranks = ranks
        self.allow_host_materialization = asyncio.Event()
        self.host_materialization_started = asyncio.Event()
        self.ready = asyncio.Event()
        self.events: list[str] = []
        self.calls = 0
        self.fail_before_ready = False
        self.fail_one_before_ready = False
        self.fail_late = False
        self.registration_started = asyncio.Event()
        self.registration_ready = asyncio.Event()
        self.registration_notifications: set[asyncio.Task[None]] = set()
        self.load_started = asyncio.Event()
        self.load_ready = asyncio.Event()
        self.load_notifications: set[asyncio.Task[None]] = set()
        self.cleanup_started = asyncio.Event()
        self.cleanup_gate = asyncio.Event()
        self.start_run_slot_forward_backward = _Call(self._forward_backward)
        self.start_run_slot_forward = _Call(self._forward)
        self.start_run_slot_sft_forward_backward = _Call(self._sft_forward_backward)
        self.start_run_slot_sft_forward = _Call(self._forward)
        self.execute_run_slot_optimizer = _Call(self._optimizer)
        self.start_prepare_run_slot_registration = _Call(self._start_registration)
        self.start_prepare_run_slot_load_state = _Call(self._start_load)
        self.finish_prepare_run_slot_registration = _Call(self._finish_registration)
        self.discard_run_slot_registration = _Call(self._discard_registration)
        self.start_unregister_run_slot = _Call(self._start_unregister)
        self.finish_unregister_run_slot = _Call(self._finish_unregister)

    async def _forward(
        self, operation_id: str, _batch: str, ready_port: _Port
    ) -> dict[int, dict[str, Any]]:
        for rank in range(self.ranks):
            ready_port.send(
                {
                    "rank": rank,
                    "operation_id": operation_id,
                    "learner_version": 3,
                }
            )
        self.events.append(f"ready:{operation_id}")
        self.host_materialization_started.set()
        await self.allow_host_materialization.wait()
        self.events.append(f"result:{operation_id}")
        return {
            rank: {
                "rank": rank,
                "operation_id": operation_id,
                "learner_version": 3,
                "metrics": {"rank": float(rank)} if rank == 0 else {},
                "_rank_telemetry": None,
                "token_logprobs": ({"shape": [1], "data": b"data"},)
                if rank == 0
                else (),
            }
            for rank in range(self.ranks)
        }

    async def _forward_backward(
        self,
        operation_id: str,
        _batch: str,
        ready_port: _Port,
        *,
        program: Literal["rl", "sft"] = "rl",
    ) -> dict[int, dict[str, Any]]:
        self.calls += 1
        if self.fail_one_before_ready:
            ready_port.send(
                {
                    "rank": 1,
                    "operation_id": operation_id,
                    "learner_version": 3,
                    "error_type": "ValueError",
                    "message": "rank-local preparation failed",
                }
            )
            await asyncio.Event().wait()
        if self.fail_before_ready:
            for rank in range(self.ranks):
                ready_port.send(
                    {
                        "rank": rank,
                        "operation_id": operation_id,
                        "learner_version": 3,
                        "error_type": "RuntimeError",
                        "message": "F/B failed",
                    }
                )
            raise RuntimeError("F/B failed")
        for rank in range(self.ranks):
            ready_port.send(
                {
                    "rank": rank,
                    "operation_id": operation_id,
                    "learner_version": 3,
                }
            )
        self.events.append(f"ready:{operation_id}")
        self.ready.set()
        self.host_materialization_started.set()
        await self.allow_host_materialization.wait()
        if self.fail_late:
            raise RuntimeError("late serialization failed")
        self.events.append(f"result:{operation_id}")
        return {
            rank: {
                "rank": rank,
                "operation_id": operation_id,
                "learner_version": 3,
                "token_count": 7,
                "metrics": {},
                "_rank_telemetry": _rank_telemetry_payload(
                    rank=rank,
                    ranks=self.ranks,
                    total_tokens=7,
                    program=program,
                ),
                "token_logprobs": ({"shape": [1], "data": b"data"},)
                if rank == 0
                else (),
            }
            for rank in range(self.ranks)
        }

    async def _sft_forward_backward(
        self, operation_id: str, batch: str, ready_port: _Port
    ) -> dict[int, dict[str, Any]]:
        return await self._forward_backward(
            operation_id, batch, ready_port, program="sft"
        )

    async def _optimizer(self, operation_id: str) -> dict[int, dict[str, Any]]:
        self.events.append(f"optimizer:{operation_id}")
        return {
            rank: {
                "rank": rank,
                "operation_id": operation_id,
                "learner_version": 4,
                "contributing_forward_backward_operation_ids": ("fb-0", "fb-1"),
                "metrics": {},
            }
            for rank in range(self.ranks)
        }

    def _registration_results(
        self, *, ready: bool | None = None
    ) -> dict[int, dict[str, Any]]:
        return {
            rank: {
                "rank": rank,
                "run_id": "run",
                **({} if ready is None else {"ready": ready}),
            }
            for rank in range(self.ranks)
        }

    @staticmethod
    def _optimizer_work(rank: int) -> RankLocalOptimizerWorkSummary:
        return RankLocalOptimizerWorkSummary(
            rank=rank,
            adapter_rank=2,
            target_modules=("q_proj",),
            trainable_lora_numel=32 * (rank + 1),
            optimizer_passes=3,
            parameter_count=2,
            layout_fingerprint=f"{rank + 1:064x}",
        )

    async def _start_registration(
        self, _payload: str, ready_port: _Port
    ) -> dict[int, dict[str, Any]]:
        self.registration_started.set()

        async def ready() -> None:
            await self.registration_ready.wait()
            for rank in range(self.ranks):
                ready_port.send(
                    {
                        "rank": rank,
                        "run_id": "run",
                        "optimizer_work": self._optimizer_work(rank).model_dump(
                            mode="json"
                        ),
                    }
                )

        task = asyncio.create_task(ready())
        self.registration_notifications.add(task)
        task.add_done_callback(self.registration_notifications.discard)
        return self._registration_results()

    async def _finish_registration(self, _payload: str) -> dict[int, dict[str, Any]]:
        return self._registration_results()

    async def _start_load(
        self, operation_id: str, ready_port: _Port
    ) -> dict[int, dict[str, Any]]:
        self.load_started.set()

        async def ready() -> None:
            await self.load_ready.wait()
            for rank in range(self.ranks):
                ready_port.send(
                    {
                        "rank": rank,
                        "operation_id": operation_id,
                        "learner_version": 4,
                    }
                )

        task = asyncio.create_task(ready())
        self.load_notifications.add(task)
        task.add_done_callback(self.load_notifications.discard)
        return {
            rank: {
                "rank": rank,
                "operation_id": operation_id,
                "learner_version": 4,
            }
            for rank in range(self.ranks)
        }

    async def _discard_registration(self, _run_id: str) -> dict[int, dict[str, Any]]:
        return self._registration_results()

    async def _start_unregister(self, _run_id: str) -> dict[int, dict[str, Any]]:
        self.events.append("detach:run")
        self.cleanup_started.set()
        return self._registration_results()

    async def _finish_unregister(self, _run_id: str) -> dict[int, dict[str, Any]]:
        await self.cleanup_gate.wait()
        self.events.append("cleanup:run")
        return self._registration_results()


@pytest.fixture(autouse=True)
def fake_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    def open_channel() -> tuple[_Port, _Receiver]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        return _Port(queue), _Receiver(queue)

    monkeypatch.setattr(monarch_module.Channel, "open", staticmethod(open_channel))


def _slot(actors: _Actors) -> MonarchTrainerSlot:
    return MonarchTrainerSlot(
        SimpleNamespace(),
        actors,
        _ProcMesh(),
        _Supervision(),
        tuple(object() for _ in range(actors.ranks)),
        (),
        (),
        command_timeout_s=2.0,
        shutdown_timeout_s=1.0,
    )


def _legacy_run(actors: _Actors) -> MonarchTrainerRun:
    actors.start_forward_backward = actors.start_run_slot_forward_backward
    run = object.__new__(MonarchTrainerRun)
    run.runtime_spec = SimpleNamespace(
        packed_sequence_length=8,
        trainer_mesh=SimpleNamespace(ranks=tuple(range(actors.ranks))),
    )
    run.run_spec = SimpleNamespace(
        run_id="run",
        training_session_id="session",
        event_timeout_s=2.0,
        initial_event_timeout_s=None,
    )
    run._actors = actors
    run._rank_processes = tuple(object() for _ in range(actors.ranks))
    run._learner_version = 3
    run._jobs = {}
    run._operations = {}
    run._forward_backward_launches = {}
    run._operation_sequence_ids = {}
    run._cancelled_operations = {}
    run._next_operation_sequence = 0
    run._open_forward_backward_ids = []
    run._lock = asyncio.Lock()
    run._active_job_id = None
    run._active_collective = None
    run._closed = False
    run._valid = True
    return run


@pytest.mark.asyncio
async def test_legacy_run_releases_forward_backward_at_gradient_ready() -> None:
    actors = _Actors()
    run = _legacy_run(actors)
    job = _Job("fb-0")
    job.run_id = "run"
    job.training_session_id = "session"
    job.sequence_id = 0
    job.batch = SimpleNamespace(sequence_length=8)
    batch = _Batch()
    batch.ref = job.batch

    launch = await run.start_forward_backward(job, batch)

    assert run._next_operation_sequence == 1
    assert run._open_forward_backward_ids == ["fb-0"]
    assert not launch.completion.done()
    actors.allow_host_materialization.set()
    assert (await launch.completion)["token_count"] == 7
    assert run._operations["fb-0"][0] == job.fingerprint


@pytest.mark.asyncio
async def test_optimizer_launches_after_ready_before_late_result() -> None:
    actors = _Actors()
    slot = _slot(actors)
    batch = _Batch()
    first = await slot.start_forward_backward(_Job("fb-0"), batch)
    second = await slot.start_forward_backward(_Job("fb-1"), batch)

    assert not first.completion.done()
    assert not second.completion.done()
    await actors.host_materialization_started.wait()
    optimizer = await slot.optim_step(_Job("optimizer", contributions=("fb-0", "fb-1")))
    assert optimizer["contributing_forward_backward_operation_ids"] == (
        "fb-0",
        "fb-1",
    )
    assert actors.events == ["ready:fb-0", "ready:fb-1", "optimizer:optimizer"]

    actors.allow_host_materialization.set()
    assert (await first.completion)["operation_id"] == "fb-0"
    assert (await second.completion)["operation_id"] == "fb-1"
    assert actors.events[-2:] == ["result:fb-0", "result:fb-1"]


@pytest.mark.asyncio
async def test_sft_optimizer_launches_after_ready_before_late_result() -> None:
    actors = _Actors()
    slot = _slot(actors)
    launch = await slot.start_sft_forward_backward(_Job("sft-fb"), _SftBatch())

    assert actors.events == ["ready:sft-fb"]
    optimizer = await slot.optim_step(_Job("optimizer", contributions=("fb-0", "fb-1")))
    assert optimizer["operation_id"] == "optimizer"
    assert actors.events == ["ready:sft-fb", "optimizer:optimizer"]
    assert not launch.completion.done()

    actors.allow_host_materialization.set()
    assert (await launch.completion)["operation_id"] == "sft-fb"
    assert actors.events[-1] == "result:sft-fb"


@pytest.mark.asyncio
async def test_forward_releases_gpu_turn_before_late_result() -> None:
    actors = _Actors()
    slot = _slot(actors)

    launch = await slot.start_forward(_Job("forward"), _Batch())
    assert not launch.completion.done()
    optimizer = await slot.optim_step(_Job("optimizer", contributions=("fb-0", "fb-1")))

    assert optimizer["operation_id"] == "optimizer"
    assert actors.events == ["ready:forward", "optimizer:optimizer"]
    actors.allow_host_materialization.set()
    assert (await launch.completion)["operation_id"] == "forward"
    assert actors.events[-1] == "result:forward"


@pytest.mark.asyncio
async def test_run_cleanup_does_not_hold_gpu_turn_or_tombstone_identity() -> None:
    actors = _Actors()
    slot = _slot(actors)
    old_work = RunOptimizerWorkSummary(
        run_id="run", ranks=tuple(actors._optimizer_work(rank) for rank in range(2))
    )
    slot._registrations["run"] = ("old", old_work)

    cleanup = asyncio.create_task(slot.unregister_run("run"))
    await actors.cleanup_started.wait()
    launch = await slot.start_forward_backward(_Job("fb-0"), _Batch())

    assert not cleanup.done()
    assert actors.events == ["detach:run", "ready:fb-0"]
    actors.cleanup_gate.set()
    await cleanup
    actors.allow_host_materialization.set()
    await launch.completion

    actors.registration_ready.set()
    registration = RunSlotRegistration(
        tenant_id="tenant",
        run_id="run",
        training_session_id="new-session",
        learner_version=0,
        generation_id="new-generation",
        adapter=_adapter("new-session"),
        optimizer_state_path="/optimizer",
    )
    work = await slot.register_run(registration)
    assert slot._registrations["run"] == (registration.model_dump_json(), work)
    assert work.critical_rank.rank == 1


@pytest.mark.asyncio
async def test_load_preparation_uses_one_rank_completion_notification() -> None:
    actors = _Actors()
    slot = _slot(actors)

    preparation = asyncio.create_task(slot.prepare_load_state(_Job("load")))
    await actors.load_started.wait()
    assert not preparation.done()
    actors.load_ready.set()
    await preparation

    assert not actors.load_notifications


@pytest.mark.asyncio
async def test_registration_preparation_does_not_hold_gpu_command_lock() -> None:
    actors = _Actors()
    slot = _slot(actors)
    registration = RunSlotRegistration(
        tenant_id="tenant",
        run_id="run",
        training_session_id="session",
        learner_version=0,
        generation_id="generation",
        adapter=_adapter("session"),
        optimizer_state_path="/optimizer",
    )
    pending_registration = asyncio.create_task(slot.register_run(registration))
    await actors.registration_started.wait()
    assert not pending_registration.done()

    launch = await slot.start_forward_backward(_Job("fb-0"), _Batch())
    await actors.ready.wait()
    assert not pending_registration.done()
    actors.allow_host_materialization.set()
    assert (await launch.completion)["operation_id"] == "fb-0"

    actors.registration_ready.set()
    work = await pending_registration
    assert slot._registrations["run"] == (registration.model_dump_json(), work)
    assert work.critical_rank.trainable_lora_numel == 64


@pytest.mark.asyncio
async def test_inflight_retry_reuses_one_rank_launch() -> None:
    actors = _Actors()
    slot = _slot(actors)
    job = _Job("fb-0")
    batch = _Batch()
    first = await slot.start_forward_backward(job, batch)
    second = await slot.start_forward_backward(job, batch)
    assert first.completion is second.completion
    assert actors.calls == 1
    actors.allow_host_materialization.set()
    await first.completion


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_late_result() -> None:
    actors = _Actors()
    slot = _slot(actors)
    waiting = asyncio.create_task(slot.forward_backward(_Job("fb-0"), _Batch()))
    await actors.ready.wait()
    while "fb-0" not in slot._forward_backward_launches:
        await asyncio.sleep(0)
    launch = slot._forward_backward_launches["fb-0"][1]
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not launch.completion.cancelled()
    actors.allow_host_materialization.set()
    assert (await launch.completion)["operation_id"] == "fb-0"
    assert slot.valid


@pytest.mark.asyncio
@pytest.mark.parametrize("late", [False, True])
async def test_rank_failure_invalidates_the_slot(late: bool) -> None:
    actors = _Actors()
    actors.fail_before_ready = not late
    actors.fail_late = late
    slot = _slot(actors)
    if not late:
        with pytest.raises(RuntimeError, match="failed before gradient-ready"):
            await slot.start_forward_backward(_Job("fb-0"), _Batch())
    else:
        launch = await slot.start_forward_backward(_Job("fb-0"), _Batch())
        actors.allow_host_materialization.set()
        with pytest.raises(RuntimeError, match="late serialization failed"):
            await launch.completion
    assert not slot.valid
    assert slot._proc_mesh.stopped


@pytest.mark.asyncio
async def test_one_rank_preparation_failure_does_not_wait_for_other_ranks() -> None:
    actors = _Actors()
    actors.fail_one_before_ready = True
    slot = _slot(actors)
    with pytest.raises(RuntimeError, match="rank-local preparation failed"):
        async with asyncio.timeout(1):
            await slot.start_forward_backward(_Job("fb-0"), _Batch())
    assert not slot.valid
    assert slot._proc_mesh.stopped


def test_result_snapshot_is_not_aliased_with_live_tensors() -> None:
    logprobs = [torch.tensor([1.0]), torch.tensor([4.0])]
    telemetry = _pending_rank_telemetry(total_tokens=3)
    result = _JobResult(
        telemetry=telemetry,
        new_logprobs=logprobs,
    )
    ref = PackedBatchRef.model_construct(
        training_kind="rl",
        tokenized_output_map=None,
        num_sequences=1,
        sequence_length=2,
    )
    launch = _stage_forward_backward_rank_result(
        _ForwardBackwardResultStagerPool(2),
        SimpleNamespace(
            operation_id="fb",
            expected_learner_version=3,
            trainable_token_count=3,
            return_token_logprobs=True,
        ),
        InMemoryPackedBatch.model_construct(ref=ref, tensors={}),
        result,
        coordinator=True,
    )
    staged = launch.snapshot.payload["token_logprobs"]
    assert (
        staged[0].untyped_storage().data_ptr() == staged[1].untyped_storage().data_ptr()
    )
    telemetry.statistics.fill_(9)
    for value in logprobs:
        value.fill_(9)

    materialized = launch.materialize()
    metrics = aggregate_rank_command_telemetry(
        [materialized["_rank_telemetry"]], expected_token_count=3
    )
    assert metrics["loss/train"] == 2.0
    assert materialized["token_count"] == 3
    values = torch.cat(
        [
            torch.frombuffer(bytearray(value["data"]), dtype=torch.float32)
            for value in materialized["token_logprobs"]
        ]
    )
    assert torch.equal(values, torch.tensor([1.0, 4.0]))


def test_forward_snapshot_is_not_aliased_with_live_tensors() -> None:
    logprobs = [torch.tensor([1.0]), torch.tensor([4.0])]
    ref = PackedBatchRef.model_construct(
        training_kind="rl",
        tokenized_output_map=None,
        num_sequences=1,
        sequence_length=2,
    )
    launch = _stage_forward_rank_result(
        _ForwardBackwardResultStagerPool(2),
        SimpleNamespace(
            operation_id="forward",
            expected_learner_version=3,
            return_token_logprobs=True,
        ),
        InMemoryPackedBatch.model_construct(ref=ref, tensors={}),
        {"metrics": {"time/forward_s": 1.0}, "token_logprobs": logprobs},
        coordinator=True,
    )
    for value in logprobs:
        value.fill_(9)

    materialized = launch.materialize()
    values = torch.cat(
        [
            torch.frombuffer(bytearray(value["data"]), dtype=torch.float32)
            for value in materialized["token_logprobs"]
        ]
    )
    assert torch.equal(values, torch.tensor([1.0, 4.0]))


def test_sft_snapshot_is_not_aliased_with_live_tensors() -> None:
    logprobs = torch.tensor([[1.0, 4.0]])
    present = torch.tensor([1], dtype=torch.int32)
    telemetry = _pending_rank_telemetry(total_tokens=3, program="sft")
    launch = _stage_sft_rank_result(
        _ForwardBackwardResultStagerPool(2),
        SimpleNamespace(
            operation_id="sft",
            expected_learner_version=3,
            trainable_token_count=3,
            return_token_logprobs=True,
        ),
        {
            "operation_id": "sft",
            "telemetry": telemetry,
            "logprob_values": logprobs,
            "logprob_present": present,
            "logprob_lengths": (2,),
        },
        coordinator=True,
    )
    telemetry.statistics.fill_(9)
    logprobs.fill_(9)
    present.zero_()

    result = launch.materialize()
    assert result["token_count"] == 3
    metrics = aggregate_rank_command_telemetry(
        [result["_rank_telemetry"]], expected_token_count=3
    )
    assert metrics["loss/train"] == 2.0
    assert result["token_logprobs"] == ((1.0, 4.0),)


def test_non_coordinator_stages_rank_telemetry_without_logprobs() -> None:
    class NonCoordinatorResult:
        telemetry = _pending_rank_telemetry(total_tokens=5)

        @property
        def new_logprobs(self) -> Any:
            raise AssertionError("non-coordinator touched coordinator logprobs")

    ref = PackedBatchRef.model_construct(
        training_kind="rl",
        tokenized_output_map=None,
        num_sequences=1,
        sequence_length=1,
    )
    launch = _stage_forward_backward_rank_result(
        _ForwardBackwardResultStagerPool(2),
        SimpleNamespace(
            operation_id="fb",
            expected_learner_version=3,
            trainable_token_count=5,
            return_token_logprobs=True,
        ),
        InMemoryPackedBatch.model_construct(ref=ref, tensors={}),
        NonCoordinatorResult(),
        coordinator=False,
    )
    result = launch.materialize()
    assert result["operation_id"] == "fb"
    assert result["learner_version"] == 3
    assert result["token_count"] == 5
    assert result["metrics"] == {}
    assert result["token_logprobs"] == ()
    assert (
        aggregate_rank_command_telemetry(
            [result["_rank_telemetry"]], expected_token_count=5
        )["loss/train"]
        == 2.0
    )


def test_outstanding_results_hold_exclusive_staging_leases() -> None:
    pool = _ForwardBackwardResultStagerPool(2)
    ref = PackedBatchRef.model_construct(
        training_kind="rl",
        tokenized_output_map=None,
        num_sequences=1,
        sequence_length=1,
    )
    batch = InMemoryPackedBatch.model_construct(ref=ref, tensors={})

    def stage(operation_id: str, value: float) -> Any:
        return _stage_forward_backward_rank_result(
            pool,
            SimpleNamespace(
                operation_id=operation_id,
                expected_learner_version=3,
                trainable_token_count=1,
                return_token_logprobs=True,
            ),
            batch,
            _JobResult(
                telemetry=_pending_rank_telemetry(
                    total_tokens=1,
                    loss=value,
                ),
                new_logprobs=[torch.tensor([value])],
            ),
            coordinator=True,
        )

    first = stage("fb-1", 1.0)
    second = stage("fb-2", 2.0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        third_future = executor.submit(stage, "fb-3", 3.0)
        with pytest.raises(TimeoutError):
            third_future.result(timeout=0.05)
        first_result = first.materialize()
        third = third_future.result(timeout=1.0)
        second_result = second.materialize()
        third_result = third.materialize()

    for result, expected in (
        (first_result, 1.0),
        (second_result, 2.0),
        (third_result, 3.0),
    ):
        metrics = aggregate_rank_command_telemetry(
            [result["_rank_telemetry"]], expected_token_count=1
        )
        assert metrics["loss/train"] == expected
    assert first.materialize() is first_result


def test_result_projection_omits_logprob_staging() -> None:
    ref = PackedBatchRef.model_construct(
        training_kind="rl",
        tokenized_output_map=None,
        num_sequences=1,
        sequence_length=1,
    )
    launch = _stage_forward_backward_rank_result(
        _ForwardBackwardResultStagerPool(2),
        SimpleNamespace(
            operation_id="fb",
            expected_learner_version=3,
            trainable_token_count=1,
            return_token_logprobs=False,
        ),
        InMemoryPackedBatch.model_construct(ref=ref, tensors={}),
        _JobResult(
            telemetry=_pending_rank_telemetry(total_tokens=1),
            new_logprobs=[torch.tensor([7.0])],
        ),
        coordinator=True,
    )

    assert launch.snapshot.payload["token_logprobs"] == ()
    assert launch.materialize()["token_logprobs"] == ()
