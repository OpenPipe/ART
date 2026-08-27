from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.distributed.rollout import (
    DistributedTrajectoryQueue,
    DistributedTrajectorySelection,
)
from art.pipeline_trainer import PipelineRuntimeConfig, PipelineTrainer
from art.pipeline_trainer import trainer as trainer_module
from art.training.contracts import OperationRef
from art.types import TrainResult


class _Operation:
    def __init__(self, ref: OperationRef) -> None:
        self.ref = ref
        self.cancelled = False

    async def cancel(self) -> None:
        self.cancelled = True


class _ImmediateClient:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.next_sequence_id = 0
        self.projected_learner_version = 0
        self.operations: list[_Operation] = []

    def operation(
        self,
        kind: str,
        *,
        learner_parent_version: int | None = None,
        transition: bool = False,
    ) -> _Operation:
        parent = (
            self.projected_learner_version
            if learner_parent_version is None
            else learner_parent_version
        )
        reserved = parent + 1 if transition else None
        operation = _Operation(
            OperationRef(
                run_id=self.run_id,
                operation_id=f"{kind}-{self.next_sequence_id}",
                sequence_id=self.next_sequence_id,
                learner_parent_version=parent,
                reserved_output_learner_version=reserved,
                kind=kind,
            )
        )
        self.next_sequence_id += 1
        if reserved is not None:
            self.projected_learner_version = reserved
        self.operations.append(operation)
        return operation

    async def optim_step(self, request: SimpleNamespace) -> _Operation:
        assert request.sequence_id == self.next_sequence_id
        await asyncio.sleep(0)
        return self.operation("optim_step", transition=True)

    async def save_weights_for_sampler(
        self, request: SimpleNamespace
    ) -> _Operation:
        assert request.sequence_id == self.next_sequence_id
        await asyncio.sleep(0)
        return self.operation("save_sampler")


class _ImmediateContext:
    def __init__(
        self,
        client: _ImmediateClient,
        groups: list[TrajectoryGroup],
        *,
        learner_parent_version: int,
        generation_id: str,
    ) -> None:
        self.client = client
        self.groups = groups
        self.learner_parent_version = learner_parent_version
        self.generation_id = generation_id
        self.preparation_metrics: dict[str, float] = {}
        self._released = False

    async def mark_packed(self) -> None:
        selections = tuple(
            group._distributed_lease
            for group in self.groups
            if isinstance(group._distributed_lease, DistributedTrajectorySelection)
        )
        assert len(selections) == len(self.groups)
        queues = {id(selection.queue): selection.queue for selection in selections}
        assert len(queues) == 1
        await next(iter(queues.values())).mark_packed(selections, self.generation_id)

    async def forward_backward(self, sequence_id: int) -> _Operation:
        assert sequence_id == self.client.next_sequence_id
        await asyncio.sleep(0)
        return self.client.operation(
            "forward_backward",
            learner_parent_version=self.learner_parent_version,
        )

    @staticmethod
    def optimizer_request(sequence_id: int) -> SimpleNamespace:
        return SimpleNamespace(sequence_id=sequence_id)

    @staticmethod
    async def sampler_request(step: int, sequence_id: int) -> SimpleNamespace:
        return SimpleNamespace(step=step, sequence_id=sequence_id)

    @staticmethod
    def state_request(step: int, sequence_id: int) -> None:
        del step, sequence_id
        return None

    async def commands_admitted(self, **_operations: object) -> None:
        await asyncio.sleep(0)

    async def _release(self, disposition: str) -> None:
        if self._released:
            return
        self._released = True
        for group in self.groups:
            selection = group._distributed_lease
            if isinstance(selection, DistributedTrajectorySelection):
                await selection.queue.release_selection(
                    selection,
                    disposition=disposition,
                    generation_id=self.generation_id,
                )
                group._distributed_lease = None

    async def complete(
        self,
        *,
        step: int,
        forward: _Operation,
        optimizer: _Operation,
        forward_submit_s: float,
    ) -> TrainResult:
        del forward, optimizer, forward_submit_s
        await asyncio.sleep(0)
        await self._release("consumed")
        return TrainResult(
            step=step,
            metrics={"data/step_trainable_assistant_tokens": 2.0},
            packed_policy_token_counts=((self.learner_parent_version, 2),),
        )

    async def abort(
        self,
        forward: _Operation | None,
        optimizer: _Operation | None,
        sampler: _Operation | None,
        *,
        optimizer_admitted: bool,
    ) -> None:
        del optimizer_admitted
        for operation in (forward, optimizer, sampler):
            if operation is not None:
                await operation.cancel()
        await self._release("discarded")


class _ImmediateBackend:
    def __init__(self, run_id: str) -> None:
        self.client = _ImmediateClient(run_id)
        self.prepared = 0
        self.finalized = 0
        self.prepare_gate: asyncio.Event | None = None
        self.prepare_started = asyncio.Event()
        self.prepare_cancelled = asyncio.Event()

    async def _get_step(self, _model: TrainableModel) -> int:
        return self.client.projected_learner_version

    @staticmethod
    def supports_async_pipeline_packing(_model: TrainableModel) -> bool:
        return True

    async def prepare_pipeline_commands(
        self,
        _model: TrainableModel,
        groups: list[TrajectoryGroup],
        *,
        normalize_advantages: bool,
        learner_parent_version: int,
        train_kwargs: dict[str, object],
    ) -> _ImmediateContext:
        del normalize_advantages, train_kwargs
        assert self.client.projected_learner_version == learner_parent_version
        if self.prepare_gate is not None:
            self.prepare_started.set()
            try:
                await self.prepare_gate.wait()
            except asyncio.CancelledError:
                self.prepare_cancelled.set()
                raise
        self.prepared += 1
        context = _ImmediateContext(
            self.client,
            groups,
            learner_parent_version=learner_parent_version,
            generation_id=f"packed-{self.client.run_id}-{self.prepared}",
        )
        await context.mark_packed()
        return context

    async def finalize_training_session(
        self, _model: TrainableModel
    ) -> dict[str, float]:
        self.finalized += 1
        return {}


def _group() -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(reward=0.0, metrics={"completion_tokens": 1}),
            Trajectory(reward=1.0, metrics={"completion_tokens": 1}),
        ]
    )


def _trainer(tmp_path: Path, name: str) -> tuple[PipelineTrainer, _ImmediateBackend]:
    backend = _ImmediateBackend(name)
    model = TrainableModel(
        name=name,
        run_name=name,
        project="pipeline-shutdown",
        base_model="test-model",
        base_path=str(tmp_path / name),
    )
    model._backend = backend  # type: ignore[assignment]

    async def rollout_fn(
        _model: TrainableModel, _scenario: int, _config: None
    ) -> TrajectoryGroup:
        await asyncio.sleep(0)
        return _group()

    trainer = PipelineTrainer(
        model=model,
        backend=backend,  # type: ignore[arg-type]
        rollout_fn=rollout_fn,
        scenarios=range(100),
        config=None,
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=2,
            min_batch_size=1,
            max_batch_size=1,
            queue_maxsize=4,
        ),
        max_steps=3,
        eval_fn=None,
        save_checkpoint=False,
        resume=False,
    )
    trainer._status = MagicMock()
    return trainer, backend


def _live_pipeline_tasks() -> set[asyncio.Task[object]]:
    names = {
        "rollout_stage",
        "packing_stage",
        "training_stage",
        "eval_stage",
        "status_loop",
    }
    return {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and (
            task.get_name() in names
            or task.get_name().startswith(("pipeline_", "post_train_"))
        )
    }


@pytest.mark.asyncio
async def test_sibling_clients_stop_cleanly_at_max_steps_with_async_packing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_log(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(TrainableModel, "log", no_log)
    for repetition in range(8):
        trainers = tuple(
            _trainer(tmp_path / str(repetition), f"run-{repetition}-{index}")
            for index in range(2)
        )

        async with asyncio.timeout(3):
            await asyncio.gather(
                *(trainer.train(handle_signals=False) for trainer, _backend in trainers)
            )

        for trainer, backend in trainers:
            assert trainer.state.next_training_step == 3
            assert trainer.state.done
            assert isinstance(trainer._output_queue, DistributedTrajectoryQueue)
            assert trainer._packed_queue is not None
            assert trainer._output_queue._closed
            assert backend.finalized == 1
            assert [operation.ref.kind for operation in backend.client.operations] == [
                kind
                for _step in range(3)
                for kind in ("forward_backward", "optim_step", "save_sampler")
            ]
    assert not _live_pipeline_tasks()


@pytest.mark.asyncio
async def test_stop_cancels_blocked_pack_ahead_preparation_before_stage_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_log(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(TrainableModel, "log", no_log)
    # Keep the real 25% cooperative-stage fraction while leaving enough event-
    # loop time for both rollout workers to release their queue references.
    monkeypatch.setattr(trainer_module, "_PIPELINE_SHUTDOWN_TIMEOUT_SECONDS", 0.4)
    trainer, backend = _trainer(tmp_path, "blocked-prepare")
    backend.prepare_gate = asyncio.Event()
    operation = asyncio.create_task(trainer.train(handle_signals=False))
    await asyncio.wait_for(backend.prepare_started.wait(), timeout=1)

    trainer.request_stop()
    await asyncio.wait_for(operation, timeout=1)

    assert backend.prepare_cancelled.is_set()
    assert isinstance(trainer._output_queue, DistributedTrajectoryQueue)
    assert trainer._output_queue._closed
    assert not _live_pipeline_tasks()
