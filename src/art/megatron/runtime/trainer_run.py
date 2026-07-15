from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from threading import Event
from typing import Protocol

from .data_plane import InMemoryPackedBatch, PackedBatch, validate_packed_batch
from .specs import (
    AdapterReady,
    TrainAccepted,
    TrainCancelled,
    TrainCompleted,
    TrainerRuntimeSpec,
    TrainEvent,
    TrainFailed,
    TrainingRunSpec,
    TrainJobSpec,
    TrainProgress,
    validate_event_stream,
)


class TrainingCancelledError(RuntimeError):
    pass


class EventSink(Protocol):
    def progress(
        self, *, step_index: int, num_steps: int, metrics: dict[str, float]
    ) -> None: ...

    def adapter_ready(self, *, learner_version: int, adapter_path: str) -> None: ...


class TrainJobExecutor(Protocol):
    def execute(
        self,
        job: TrainJobSpec,
        batch: InMemoryPackedBatch,
        sink: EventSink,
        cancelled: Event,
    ) -> dict[str, float]: ...

    def close(self) -> None: ...


class TrainerRun(Protocol):
    @property
    def valid(self) -> bool: ...

    def train(
        self, job: TrainJobSpec, batch: PackedBatch
    ) -> AsyncIterator[TrainEvent]: ...

    async def close(self) -> None: ...


class _EventEmitter:
    def __init__(
        self,
        job: TrainJobSpec,
        publish: Callable[[TrainEvent], None],
        events: list[TrainEvent],
    ) -> None:
        self._job = job
        self._publish = publish
        self._events = events

    def accepted(self) -> None:
        self._emit(
            TrainAccepted(
                job_id=self._job.job_id,
                run_id=self._job.run_id,
                sequence=0,
                expected_learner_version=self._job.expected_learner_version,
            )
        )

    def progress(
        self, *, step_index: int, num_steps: int, metrics: dict[str, float]
    ) -> None:
        self._emit(
            TrainProgress(
                job_id=self._job.job_id,
                run_id=self._job.run_id,
                sequence=len(self._events),
                step_index=step_index,
                num_steps=num_steps,
                metrics=metrics,
            )
        )

    def adapter_ready(self, *, learner_version: int, adapter_path: str) -> None:
        self._emit(
            AdapterReady(
                job_id=self._job.job_id,
                run_id=self._job.run_id,
                sequence=len(self._events),
                learner_version=learner_version,
                adapter_path=adapter_path,
            )
        )

    def completed(self, metrics: dict[str, float]) -> None:
        self._emit(
            TrainCompleted(
                job_id=self._job.job_id,
                run_id=self._job.run_id,
                sequence=len(self._events),
                learner_version=self._job.learner_version,
                metrics=metrics,
            )
        )

    def failed(self, exc: BaseException, *, runtime_invalidated: bool) -> None:
        self._emit(
            TrainFailed(
                job_id=self._job.job_id,
                run_id=self._job.run_id,
                sequence=len(self._events),
                error_type=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
                runtime_invalidated=runtime_invalidated,
            )
        )

    def cancelled(self, reason: str) -> None:
        self._emit(
            TrainCancelled(
                job_id=self._job.job_id,
                run_id=self._job.run_id,
                sequence=len(self._events),
                reason=reason,
            )
        )

    def _emit(self, event: TrainEvent) -> None:
        if self._events and self._events[-1].kind in {
            "completed",
            "failed",
            "cancelled",
        }:
            raise RuntimeError("cannot emit an event after a terminal train event")
        self._events.append(event)
        self._publish(event)


class LocalTrainerRun:
    """Run-scoped invariant enforcement around an in-process Megatron executor."""

    def __init__(
        self,
        runtime_spec: TrainerRuntimeSpec,
        run_spec: TrainingRunSpec,
        executor: TrainJobExecutor,
    ) -> None:
        if run_spec.runtime_fingerprint != runtime_spec.fingerprint:
            raise ValueError(
                "training run does not match the trainer runtime fingerprint"
            )
        self.runtime_spec = runtime_spec
        self.run_spec = run_spec
        self._executor = executor
        self._learner_version = run_spec.initial_learner_version
        self._jobs: dict[str, tuple[str, tuple[TrainEvent, ...]]] = {}
        self._job_lock = asyncio.Lock()
        self._active_cancel: Event | None = None
        self._closed = False
        self._valid = True

    @property
    def learner_version(self) -> int:
        return self._learner_version

    @property
    def valid(self) -> bool:
        return self._valid

    async def train(
        self, job: TrainJobSpec, batch: PackedBatch
    ) -> AsyncIterator[TrainEvent]:
        cached = self._jobs.get(job.job_id)
        if cached is not None and cached[0] == job.fingerprint:
            for event in cached[1]:
                yield event
            return

        async with self._job_lock:
            cached = self._jobs.get(job.job_id)
            if cached is not None:
                if cached[0] == job.fingerprint:
                    for event in cached[1]:
                        yield event
                    return
                async for event in self._rejected(
                    job, RuntimeError("job_id was already used with a different job")
                ):
                    yield event
                return

            error = self._validate_job(job, batch)
            if error is not None:
                events = [
                    TrainAccepted(
                        job_id=job.job_id,
                        run_id=job.run_id,
                        sequence=0,
                        expected_learner_version=job.expected_learner_version,
                    ),
                    TrainFailed(
                        job_id=job.job_id,
                        run_id=job.run_id,
                        sequence=1,
                        error_type=type(error).__name__,
                        message=str(error),
                        runtime_invalidated=not self._valid,
                    ),
                ]
                validate_event_stream(events)
                self._jobs[job.job_id] = (job.fingerprint, tuple(events))
                for event in events:
                    yield event
                return

            assert isinstance(batch, InMemoryPackedBatch)
            validate_packed_batch(batch)
            queue: asyncio.Queue[TrainEvent | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            events: list[TrainEvent] = []

            def publish(event: TrainEvent) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, event)

            emitter = _EventEmitter(
                job,
                publish,
                events,
            )
            cancelled = Event()
            self._active_cancel = cancelled
            emitter.accepted()

            def execute() -> None:
                try:
                    metrics = self._executor.execute(job, batch, emitter, cancelled)
                    if cancelled.is_set():
                        raise TrainingCancelledError("train job was cancelled")
                    self._learner_version = job.learner_version
                    emitter.completed(metrics)
                except TrainingCancelledError as exc:
                    self._valid = False
                    emitter.cancelled(str(exc))
                except BaseException as exc:
                    self._valid = False
                    emitter.failed(exc, runtime_invalidated=True)
                finally:
                    validate_event_stream(events)
                    self._jobs[job.job_id] = (job.fingerprint, tuple(events))
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            task = asyncio.create_task(asyncio.to_thread(execute))
            try:
                while (event := await queue.get()) is not None:
                    yield event
                await task
            finally:
                if not task.done():
                    cancelled.set()
                    self._valid = False
                    await asyncio.shield(task)
                self._active_cancel = None

    async def _rejected(
        self, job: TrainJobSpec, error: BaseException
    ) -> AsyncIterator[TrainEvent]:
        yield TrainAccepted(
            job_id=job.job_id,
            run_id=job.run_id,
            sequence=0,
            expected_learner_version=job.expected_learner_version,
        )
        yield TrainFailed(
            job_id=job.job_id,
            run_id=job.run_id,
            sequence=1,
            error_type=type(error).__name__,
            message=str(error),
            runtime_invalidated=False,
        )

    def _validate_job(
        self, job: TrainJobSpec, batch: PackedBatch
    ) -> BaseException | None:
        if self._closed:
            return RuntimeError("trainer run is closed")
        if not self._valid:
            return RuntimeError("trainer runtime is invalid")
        if job.run_id != self.run_spec.run_id:
            return ValueError("job run_id does not match this training run")
        if job.training_session_id != self.run_spec.training_session_id:
            return ValueError(
                "job training_session_id does not match this training run"
            )
        if job.output.optimizer_state_path != self.run_spec.optimizer_state_path:
            return ValueError(
                "job optimizer state path does not match this training run"
            )
        if job.expected_learner_version != self._learner_version:
            return ValueError(
                "expected learner version mismatch: "
                f"job={job.expected_learner_version}, runtime={self._learner_version}"
            )
        if batch.ref != job.batch:
            return ValueError("job batch ref does not match the supplied packed batch")
        if job.batch.sequence_length != self.runtime_spec.packed_sequence_length:
            return ValueError(
                "packed batch sequence length does not match the trainer runtime"
            )
        if not isinstance(batch, InMemoryPackedBatch):
            return TypeError("local trainer runs require an InMemoryPackedBatch")
        try:
            validate_packed_batch(batch)
        except ValueError as exc:
            return exc
        return None

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._valid = False
        if self._active_cancel is not None:
            self._active_cancel.set()
        async with self._job_lock:
            await asyncio.to_thread(self._executor.close)
