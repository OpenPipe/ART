from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from art.distributed.art_runtime import ArtRuntime
from art.training import (
    CommandExecutionUsage,
    ForwardBackwardRequest,
    ForwardRequest,
    OperationExecutionError,
    OperationFailureCode,
    OperationRef,
    OperationResultType,
    OperationWorker,
    OptimStepRequest,
    PackedInputCaptureRef,
    RunCommand,
    bootstrap_operation_worker,
)

from .operation_handler import (
    MegatronArtifactResourcePlan,
    MegatronCheckpointOperations,
    MegatronOperationConfig,
    MegatronOperationHandler,
)
from .runtime.specs import TrainerGeneration, TrainingRunSpec

SlotComponent = Literal["weights", "optimizer", "accumulator"]


class MegatronSlotScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    deficit_quantum_tokens: int = Field(default=32_768, ge=1)
    optimizer_turn_limit: int = Field(default=2, ge=1, le=16)
    max_ready_commands: int = Field(default=256, ge=1, le=4096)


class MegatronSlotResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    source: TrainerGeneration
    optimizer_state_path: str = Field(min_length=1)
    components: tuple[SlotComponent, ...]


class MegatronSlotResourceManager(Protocol):
    """Optional off-loop residency planner; rank execution remains authoritative."""

    def prefetch(self, request: MegatronSlotResourceRequest) -> None: ...

    async def ensure(self, request: MegatronSlotResourceRequest) -> None: ...

    async def release_run(self, run_id: str) -> None: ...


class InlineMegatronSlotResources:
    """Use the rank-local command ensure path without speculative readiness."""

    def prefetch(self, request: MegatronSlotResourceRequest) -> None:
        del request

    async def ensure(self, request: MegatronSlotResourceRequest) -> None:
        del request

    async def release_run(self, run_id: str) -> None:
        del run_id


class _SharedTrainer(Protocol):
    runtime_spec: Any

    async def forward(self, job: Any, batch: Any) -> dict[str, Any]: ...

    async def forward_backward(self, job: Any, batch: Any) -> dict[str, Any]: ...

    async def optim_step(self, job: Any) -> dict[str, Any]: ...

    def register_command_run(self, run_spec: TrainingRunSpec) -> None: ...

    async def drain_command_run(self, run_id: str) -> None: ...


@dataclass(slots=True)
class MegatronSlotRun:
    run_id: str
    worker: OperationWorker


@dataclass(slots=True)
class _RunState:
    handler: MegatronOperationHandler
    worker: OperationWorker
    draining: bool = False
    preparing: int = 0


@dataclass(slots=True)
class _ReadyCommand:
    run_id: str
    request: RunCommand
    operation: OperationRef
    contributions: tuple[str, ...]
    cost: int
    future: asyncio.Future[OperationResultType]

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


class MegatronSlotCoordinator:
    """Fairly multiplex logical runs over one persistent trainer allocation."""

    def __init__(
        self,
        runtime: ArtRuntime,
        trainer: _SharedTrainer,
        *,
        resources: MegatronSlotResourceManager | None = None,
        schedule: MegatronSlotScheduleConfig | None = None,
    ) -> None:
        self.runtime = runtime
        self.trainer = trainer
        self.resources = resources or InlineMegatronSlotResources()
        self.schedule = schedule or MegatronSlotScheduleConfig()
        self._runs: dict[str, _RunState] = {}
        self._ready: list[_ReadyCommand] = []
        self._deficit: dict[str, int] = {}
        self._order: list[str] = []
        self._cursor = 0
        self._optimizer_turn_remaining = 0
        self._condition = asyncio.Condition()
        self._active_run_id: str | None = None
        self._pump_task: asyncio.Task[None] | None = None
        self._closed = False

    async def register_run(
        self,
        config: MegatronOperationConfig,
        *,
        checkpoints: MegatronCheckpointOperations | None = None,
        max_retained_operations: int = 128,
    ) -> MegatronSlotRun:
        async with self._condition:
            if self._closed:
                raise RuntimeError("Megatron slot coordinator is closed")
            prior = self._runs.get(config.run_id)
            if prior is not None:
                if prior.draining:
                    raise RuntimeError("Megatron slot run is draining")
                if prior.handler.config != config:
                    raise RuntimeError("run_id was reused with different configuration")
                return MegatronSlotRun(config.run_id, prior.worker)
            run_spec = TrainingRunSpec(
                run_id=config.run_id,
                runtime_fingerprint=self.trainer.runtime_spec.fingerprint,
                training_session_id=config.training_session_id,
                initial_learner_version=config.source.policy_step,
                initial_adapter_path=config.source.adapter_path,
                optimizer_state_path=config.optimizer_state_path,
            )
            self.trainer.register_command_run(run_spec)
            handler = MegatronOperationHandler(
                self.runtime,
                self.trainer,
                config,
                checkpoints=checkpoints,
            )
            scheduled = _ScheduledRunHandler(self, config.run_id)
            worker = bootstrap_operation_worker(
                scheduled,
                max_retained_operations=max_retained_operations,
            )
            self._runs[config.run_id] = _RunState(handler=handler, worker=worker)
            self._deficit[config.run_id] = 0
            self._order.append(config.run_id)
            if self._pump_task is None:
                self._pump_task = asyncio.create_task(self._pump())
            return MegatronSlotRun(config.run_id, worker)

    def resolve_run(self, run_id: str) -> MegatronSlotRun:
        state = self._runs.get(run_id)
        if state is None or state.draining:
            raise KeyError(f"Megatron slot run {run_id!r} is unavailable")
        return MegatronSlotRun(run_id, state.worker)

    async def plan_artifacts(
        self, run_id: str, request: RunCommand
    ) -> MegatronArtifactResourcePlan:
        state = self._runs.get(run_id)
        if state is None or state.draining:
            raise KeyError(f"Megatron slot run {run_id!r} is unavailable")
        return await state.handler.plan_artifacts(request)

    async def drain_run(self, run_id: str) -> None:
        async with self._condition:
            state = self._runs.get(run_id)
            if state is None:
                return
            state.draining = True
            while (
                state.preparing
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
            if run_id in self._order:
                self._order.remove(run_id)
            self._condition.notify_all()

    async def aclose(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        for run_id in tuple(self._runs):
            await self.drain_run(run_id)
        if self._pump_task is not None:
            await self._pump_task

    async def _execute(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributions: tuple[str, ...],
    ) -> OperationResultType:
        async with self._condition:
            state = self._runs.get(operation.run_id)
            if state is None or state.draining or self._closed:
                raise _not_executed("cancelled", "Megatron slot run is draining")
            state.preparing += 1
        capture: PackedInputCaptureRef | None = None
        try:
            components = _components(request)
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
                request = request.model_copy(update={"batch": capture})
                cost = max(1, packing.physical_tokens)
            await self.resources.ensure(resource_request)
            loop = asyncio.get_running_loop()
            future: asyncio.Future[OperationResultType] = loop.create_future()
            ready = _ReadyCommand(
                run_id=operation.run_id,
                request=request,
                operation=operation,
                contributions=contributions,
                cost=cost,
                future=future,
            )
            async with self._condition:
                if state.draining or self._closed:
                    raise _not_executed("cancelled", "Megatron slot run is draining")
                if len(self._ready) >= self.schedule.max_ready_commands:
                    raise _not_executed(
                        "capacity_exhausted", "Megatron slot ready queue is full"
                    )
                self._ready.append(ready)
                self._condition.notify_all()
        except BaseException as error:
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
                self._condition.notify_all()
        return await _await_terminal(future)

    async def _pump(self) -> None:
        while True:
            async with self._condition:
                while not self._ready and not self._closed:
                    await self._condition.wait()
                if self._closed and not self._ready:
                    return
                command = self._select_ready()
                self._ready.remove(command)
                self._active_run_id = command.run_id
            try:
                state = self._runs[command.run_id]
                result = await state.handler(
                    command.request,
                    command.operation,
                    command.contributions,
                )
            except BaseException as error:
                if not command.future.done():
                    command.future.set_exception(error)
            else:
                if not command.future.done():
                    command.future.set_result(result)
            finally:
                async with self._condition:
                    self._active_run_id = None
                    self._condition.notify_all()

    def _select_ready(self) -> _ReadyCommand:
        optimizers = [item for item in self._ready if item.optimizer]
        forwards = [item for item in self._ready if not item.optimizer]
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
