from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Generic, TypeVar, cast

from art.training import (
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    OperationRef,
    OperationResult,
    OptimStepRequest,
    OptimStepResult,
    RunCommand,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
    TrainingRunSpec,
)

from .client import (
    NativeTrainingOperation,
    NativeTrainingRun,
    TrainingRuns,
    request_kind,
)

ResultT = TypeVar("ResultT", bound=OperationResult)


class RemoteTrainingError(RuntimeError):
    pass


class RemoteTrainingOperationError(RemoteTrainingError):
    def __init__(self, operation_id: str, error: Mapping[str, Any]) -> None:
        super().__init__(
            f"remote training operation {operation_id} failed: {dict(error)}"
        )
        self.operation_id = operation_id
        self.error = dict(error)


class RemoteTrainingOperationCancelled(RemoteTrainingError):
    pass


class RemoteTrainingOperation(Generic[ResultT]):
    """One retained native operation identity and its terminal evidence."""

    def __init__(
        self,
        ref: OperationRef,
        service: TrainingRuns,
        result_type: type[ResultT],
        *,
        poll_interval_s: float,
    ) -> None:
        self._ref = ref
        self._service = service
        self._result_type = result_type
        self._poll_interval_s = poll_interval_s
        self._terminal: asyncio.Task[NativeTrainingOperation] | None = None

    @property
    def ref(self) -> OperationRef:
        return self._ref

    async def terminal_evidence(self) -> NativeTrainingOperation:
        if self._terminal is None:
            self._terminal = asyncio.create_task(
                self._wait_terminal(),
                name=f"remote-training-{self._ref.operation_id}",
            )
        return await asyncio.shield(self._terminal)

    async def result(self) -> ResultT:
        operation = await self.terminal_evidence()
        if operation.status == "failed":
            raise RemoteTrainingOperationError(
                self._ref.operation_id, operation.error or {}
            )
        if operation.status == "cancelled":
            raise RemoteTrainingOperationCancelled(
                f"remote training operation {self._ref.operation_id} was cancelled"
            )
        if operation.status != "succeeded" or operation.result is None:
            raise RemoteTrainingError("terminal operation has no successful result")
        result = self._result_type.model_validate(operation.result)
        if result.operation_id != self._ref.operation_id:
            raise RemoteTrainingError("operation result identity changed")
        return result

    async def cancel(self) -> NativeTrainingOperation:
        return await self._service.cancel(self._ref.run_id, self._ref.operation_id)

    async def _wait_terminal(self) -> NativeTrainingOperation:
        while True:
            operation = await self._service.operation(
                self._ref.run_id, self._ref.operation_id
            )
            if _operation_ref(operation) != self._ref:
                raise RemoteTrainingError("remote operation identity changed")
            if operation.status in {"succeeded", "failed", "cancelled"}:
                return operation
            await asyncio.sleep(self._poll_interval_s)


class RemoteTrainingClient:
    """Run-scoped native client retaining exact run and operation IDs."""

    def __init__(
        self,
        service: TrainingRuns,
        run: NativeTrainingRun,
        *,
        poll_interval_s: float = 0.1,
    ) -> None:
        if run.status != "open":
            raise ValueError(f"remote training run is {run.status}, not open")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self._service = service
        self._run = run
        self._next_sequence_id = run.next_sequence_id
        self._projected_learner_version = run.projected_learner_version
        self._poll_interval_s = poll_interval_s
        self._operations: dict[str, tuple[str, RemoteTrainingOperation[Any]]] = {}
        self._operations_by_id: dict[str, RemoteTrainingOperation[Any]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def resolve(
        cls,
        service: TrainingRuns,
        *,
        request_id: str,
        run_name: str,
        spec: TrainingRunSpec,
        poll_interval_s: float = 0.1,
    ) -> RemoteTrainingClient:
        run = await service.resolve(request_id=request_id, run_name=run_name, spec=spec)
        return cls(service, run, poll_interval_s=poll_interval_s)

    @property
    def run_id(self) -> str:
        return self._run.run_id

    @property
    def next_sequence_id(self) -> int:
        return self._next_sequence_id

    @property
    def projected_learner_version(self) -> int:
        return self._projected_learner_version

    @property
    def operation_ids(self) -> tuple[str, ...]:
        return tuple(self._operations_by_id)

    async def operation_evidence(self, operation_id: str) -> NativeTrainingOperation:
        operation = await self._service.operation(self.run_id, operation_id)
        retained = self._operations_by_id.get(operation_id)
        if retained is not None and _operation_ref(operation) != retained.ref:
            raise RemoteTrainingError("remote operation identity changed")
        return operation

    async def forward(
        self, request: ForwardRequest
    ) -> RemoteTrainingOperation[ForwardResult]:
        return await self._submit(request, ForwardResult)

    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> RemoteTrainingOperation[ForwardBackwardResult]:
        return await self._submit(request, ForwardBackwardResult)

    async def optim_step(
        self, request: OptimStepRequest
    ) -> RemoteTrainingOperation[OptimStepResult]:
        return await self._submit(request, OptimStepResult)

    async def save_weights_for_sampler(
        self, request: SaveWeightsForSamplerRequest
    ) -> RemoteTrainingOperation[SamplerWeightsResult]:
        return await self._submit(request, SamplerWeightsResult)

    async def save_state(
        self, request: SaveStateRequest
    ) -> RemoteTrainingOperation[SaveStateResult]:
        return await self._submit(request, SaveStateResult)

    async def load_state(
        self, request: LoadStateRequest
    ) -> RemoteTrainingOperation[LoadStateResult]:
        return await self._submit(
            request.model_copy(update={"restore_optimizer": False}), LoadStateResult
        )

    async def load_state_with_optimizer(
        self, request: LoadStateRequest
    ) -> RemoteTrainingOperation[LoadStateResult]:
        return await self._submit(
            request.model_copy(update={"restore_optimizer": True}), LoadStateResult
        )

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
        self._run = await self._service.close(self.run_id)

    async def _submit(
        self, request: RunCommand, result_type: type[ResultT]
    ) -> RemoteTrainingOperation[ResultT]:
        fingerprint = request.model_dump_json()
        async with self._lock:
            if self._closed:
                raise RuntimeError("remote training client is closed")
            if request.run_id != self.run_id:
                raise ValueError("command run_id does not match the remote client")
            prior = self._operations.get(request.request_id)
            if prior is not None:
                if prior[0] != fingerprint:
                    raise ValueError("request_id was reused with different content")
                return cast(RemoteTrainingOperation[ResultT], prior[1])
            if request.sequence_id != self._next_sequence_id:
                raise ValueError(
                    f"expected sequence {self._next_sequence_id}, "
                    f"got {request.sequence_id}"
                )
            admitted = await self._service.submit(request)
            ref = _operation_ref(admitted)
            if (
                ref.run_id != self.run_id
                or ref.sequence_id != request.sequence_id
                or ref.kind != request_kind(request)
                or ref.learner_parent_version != self._projected_learner_version
            ):
                raise RemoteTrainingError("server admitted a divergent command")
            operation = RemoteTrainingOperation(
                ref,
                self._service,
                result_type,
                poll_interval_s=self._poll_interval_s,
            )
            self._operations[request.request_id] = (fingerprint, operation)
            self._operations_by_id[ref.operation_id] = operation
            self._next_sequence_id += 1
            if ref.reserved_output_learner_version is not None:
                self._projected_learner_version = ref.reserved_output_learner_version
            return operation


def _operation_ref(operation: NativeTrainingOperation) -> OperationRef:
    kind = "save_sampler" if operation.kind == "save_weights" else operation.kind
    return OperationRef(
        run_id=operation.run_id,
        operation_id=operation.operation_id,
        sequence_id=operation.sequence_id,
        learner_parent_version=operation.learner_parent_version,
        reserved_output_learner_version=(operation.reserved_output_learner_version),
        kind=kind,
    )
