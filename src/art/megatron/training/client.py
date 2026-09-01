from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from art.training import (
    ForwardBackwardRequest,
    ForwardBackwardResult,
    ForwardRequest,
    ForwardResult,
    LoadStateRequest,
    LoadStateResult,
    OperationExecutionOutcome,
    OperationFailed,
    OperationRef,
    OptimStepRequest,
    OptimStepResult,
    RunCommand,
    RunCommandLedger,
    SamplerWeightsResult,
    SaveStateRequest,
    SaveStateResult,
    SaveWeightsForSamplerRequest,
)
from art.training.contracts import OperationKind, OperationResult

if TYPE_CHECKING:
    from art.megatron.slot_coordinator import MegatronSlotRun

ResultT = TypeVar("ResultT", bound=OperationResult)


class LocalMegatronTrainingError(RuntimeError):
    """Terminal error from the production-compatible local command path."""


class LocalMegatronTrainingOperation(Generic[ResultT]):
    """One local operation retaining the same identity as a slot execution."""

    def __init__(
        self,
        ref: OperationRef,
        completion: asyncio.Task[OperationExecutionOutcome],
        result_type: type[ResultT],
    ) -> None:
        self._ref = ref
        self._completion = completion
        self._result_type = result_type

    @property
    def ref(self) -> OperationRef:
        return self._ref

    async def outcome(self) -> OperationExecutionOutcome:
        return await asyncio.shield(self._completion)

    async def result(self) -> ResultT:
        outcome = await self.outcome()
        if isinstance(outcome, OperationFailed):
            failure = outcome.failure
            raise LocalMegatronTrainingError(
                f"{failure.code}: {failure.error_type}: {failure.message}"
            )
        result = outcome.result
        if not isinstance(result, self._result_type):
            raise TypeError("local Megatron operation returned the wrong result type")
        return result

    async def cancel(self) -> None:
        self._completion.cancel()
        await asyncio.gather(self._completion, return_exceptions=True)


class LocalMegatronTrainingClient:
    """Serial oracle facade over the same slot worker used in production."""

    def __init__(self, run: MegatronSlotRun, *, learner_version: int) -> None:
        self._run = run
        self._ledger = RunCommandLedger(run.run_id, learner_version=learner_version)
        self._operations: dict[
            str, LocalMegatronTrainingOperation[OperationResult]
        ] = {}
        self._closed = False

    @property
    def run_id(self) -> str:
        return self._run.run_id

    @property
    def next_sequence_id(self) -> int:
        return self._ledger.next_sequence_id

    @property
    def projected_learner_version(self) -> int:
        return self._ledger.projected_learner_version

    @property
    def operation_ids(self) -> tuple[str, ...]:
        return tuple(self._operations)

    async def forward(
        self, request: ForwardRequest
    ) -> LocalMegatronTrainingOperation[ForwardResult]:
        return await self._submit(request, kind="forward", result_type=ForwardResult)

    async def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> LocalMegatronTrainingOperation[ForwardBackwardResult]:
        return await self._submit(
            request, kind="forward_backward", result_type=ForwardBackwardResult
        )

    async def optim_step(
        self, request: OptimStepRequest
    ) -> LocalMegatronTrainingOperation[OptimStepResult]:
        return await self._submit(
            request, kind="optim_step", result_type=OptimStepResult
        )

    async def save_weights_for_sampler(
        self, request: SaveWeightsForSamplerRequest
    ) -> LocalMegatronTrainingOperation[SamplerWeightsResult]:
        return await self._submit(
            request, kind="save_sampler", result_type=SamplerWeightsResult
        )

    async def save_state(
        self, request: SaveStateRequest
    ) -> LocalMegatronTrainingOperation[SaveStateResult]:
        return await self._submit(
            request, kind="save_state", result_type=SaveStateResult
        )

    async def load_state(
        self, request: LoadStateRequest
    ) -> LocalMegatronTrainingOperation[LoadStateResult]:
        return await self._submit(
            request.model_copy(update={"restore_optimizer": False}),
            kind="load_state",
            result_type=LoadStateResult,
        )

    async def load_state_with_optimizer(
        self, request: LoadStateRequest
    ) -> LocalMegatronTrainingOperation[LoadStateResult]:
        return await self._submit(
            request.model_copy(update={"restore_optimizer": True}),
            kind="load_state",
            result_type=LoadStateResult,
        )

    async def operation_evidence(self, operation_id: str) -> OperationExecutionOutcome:
        try:
            operation = self._operations[operation_id]
        except KeyError:
            raise KeyError(
                f"local Megatron operation {operation_id!r} is unknown"
            ) from None
        return await operation.outcome()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        pending = tuple(
            operation._completion
            for operation in self._operations.values()
            if not operation._completion.done()
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def _submit(
        self,
        request: RunCommand,
        *,
        kind: OperationKind,
        result_type: type[ResultT],
    ) -> LocalMegatronTrainingOperation[ResultT]:
        if self._closed:
            raise RuntimeError("local Megatron training client is closed")
        admission = await self._ledger.admit(request, kind=kind)
        prior = self._operations.get(admission.ref.operation_id)
        if prior is not None:
            return cast(LocalMegatronTrainingOperation[ResultT], prior)

        async def execute() -> OperationExecutionOutcome:
            outcome = await self._run.worker.execute(
                request,
                admission.ref,
                admission.contributing_forward_backward_operation_ids,
            )
            error: BaseException | None = None
            if isinstance(outcome, OperationFailed):
                error = LocalMegatronTrainingError(outcome.failure.message)
            elif isinstance(request, ForwardBackwardRequest):
                result = outcome.result
                if (
                    isinstance(result, ForwardBackwardResult)
                    and not result.produced_gradient
                ):
                    self._ledger.cancel_pending_forward_backward(
                        request.request_id, admission
                    )
            self._ledger.mark_terminal(request.request_id, admission, error=error)
            return outcome

        completion = asyncio.create_task(
            execute(), name=f"local-megatron-{kind}-{request.sequence_id}"
        )
        operation = LocalMegatronTrainingOperation(
            admission.ref, completion, result_type
        )
        self._operations[admission.ref.operation_id] = cast(
            LocalMegatronTrainingOperation[OperationResult], operation
        )
        return operation
