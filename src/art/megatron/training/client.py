from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from art.training import (
    CommandAdmission,
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
    from art.megatron.slot_runtime import MegatronRunBinding

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
    """In-process command facade over the production slot worker."""

    def __init__(
        self,
        run: MegatronSlotRun,
        *,
        learner_version: int,
        initial_operation_sequence: int = 0,
        max_retained_operations: int = 128,
        coordinator: object | None = None,
        outcome_sink: object | None = None,
    ) -> None:
        if not 1 <= max_retained_operations <= 1024:
            raise ValueError("max_retained_operations must be between 1 and 1024")
        self._run = run
        self._ledger = RunCommandLedger(
            run.run_id,
            learner_version=learner_version,
            initial_operation_sequence=initial_operation_sequence,
        )
        self._max_retained_operations = max_retained_operations
        self._coordinator = coordinator
        self._outcome_sink = outcome_sink
        self._operations: dict[
            str, LocalMegatronTrainingOperation[OperationResult]
        ] = {}
        self._requests: dict[
            str,
            tuple[
                RunCommand,
                OperationKind,
                LocalMegatronTrainingOperation[OperationResult],
            ],
        ] = {}
        self._retained_outcomes: dict[str, OperationExecutionOutcome] = {}
        self._request_ids_by_operation: dict[str, str] = {}
        self._ledger_records: dict[str, tuple[str, CommandAdmission]] = {}
        self._terminal_operation_ids: set[str] = set()
        self._awaiting_acknowledgement: set[str] = set()
        self._acknowledged_operation_ids: set[str] = set()
        self._closed = False

    @classmethod
    def from_binding(
        cls,
        binding: MegatronRunBinding,
        *,
        max_retained_operations: int = 128,
    ) -> LocalMegatronTrainingClient:
        config = binding.config
        if binding.run.run_id != config.run_id:
            raise ValueError("Megatron run binding changed run identity")
        return cls(
            binding.run,
            learner_version=config.source.policy_step,
            initial_operation_sequence=config.initial_operation_sequence,
            max_retained_operations=max_retained_operations,
            coordinator=binding.coordinator,
            outcome_sink=binding.outcome_sink,
        )

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

    @property
    def open_forward_backward_operation_ids(self) -> tuple[str, ...]:
        return self._ledger.open_forward_backward_operation_ids

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
        retained = self._retained_outcomes.get(operation_id)
        if retained is not None:
            return retained
        try:
            operation = self._operations[operation_id]
        except KeyError:
            raise KeyError(
                f"local Megatron operation {operation_id!r} is unknown"
            ) from None
        return await operation.outcome()

    def retire_operation(self, operation_id: str) -> bool:
        """Release one completed local replay record after evidence is consumed."""

        operation = self._operations.get(operation_id)
        if operation is None:
            return True
        if not operation._completion.done():
            raise RuntimeError("cannot retire a nonterminal local operation")
        if operation_id in self._ledger_records:
            return False
        if (
            self._coordinator is not None or self._outcome_sink is not None
        ) and operation_id not in self._acknowledged_operation_ids:
            return False
        self._run.worker.retire(operation_id)
        request_id = self._request_ids_by_operation.pop(operation_id)
        self._requests.pop(request_id)
        self._retained_outcomes.pop(operation_id, None)
        self._terminal_operation_ids.discard(operation_id)
        self._awaiting_acknowledgement.discard(operation_id)
        self._acknowledged_operation_ids.discard(operation_id)
        self._operations.pop(operation_id)
        return True

    async def acknowledge_operation(self, operation_id: str) -> None:
        """Acknowledge externally retained evidence and release physical inputs."""

        operation = self._operations.get(operation_id)
        if operation is None:
            if operation_id in self._acknowledged_operation_ids:
                return
            raise KeyError(f"local Megatron operation {operation_id!r} is unknown")
        if not operation._completion.done():
            raise RuntimeError("cannot acknowledge a nonterminal local operation")
        if self._coordinator is not None:
            acknowledge = getattr(self._coordinator, "acknowledge_operation", None)
            if not callable(acknowledge):
                raise RuntimeError("Megatron coordinator cannot acknowledge operations")
            await acknowledge(self.run_id, operation_id)
        self._awaiting_acknowledgement.discard(operation_id)
        self._acknowledged_operation_ids.add(operation_id)

    def retire_acknowledged_operations(self) -> None:
        for operation_id in tuple(self._acknowledged_operation_ids):
            self.retire_operation(operation_id)

    async def close(self) -> None:
        if self._closed and not self._operations:
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
        if self._operations:
            raise RuntimeError(
                "cannot close with unconsumed local operation evidence; "
                "persist and retire every terminal outcome first"
            )

    async def _submit(
        self,
        request: RunCommand,
        *,
        kind: OperationKind,
        result_type: type[ResultT],
    ) -> LocalMegatronTrainingOperation[ResultT]:
        if self._closed:
            raise RuntimeError("local Megatron training client is closed")
        replay = self._requests.get(request.request_id)
        if replay is not None:
            prior_request, prior_kind, prior_operation = replay
            if prior_request != request or prior_kind != kind:
                raise RuntimeError("request_id was reused for a different command")
            return cast(LocalMegatronTrainingOperation[ResultT], prior_operation)
        if self._awaiting_acknowledgement:
            raise RuntimeError(
                "prior terminal operation requires durable acknowledgement"
            )
        if len(self._operations) >= self._max_retained_operations:
            raise RuntimeError(
                "local operation replay window is full; retire consumed evidence"
            )
        admission = await self._ledger.admit(request, kind=kind)
        if self._outcome_sink is not None:
            retain_admission = getattr(self._outcome_sink, "retain_admission", None)
            if not callable(retain_admission):
                raise RuntimeError("operation outcome sink cannot retain admission")
            await retain_admission(request, admission)
        prior = self._operations.get(admission.ref.operation_id)
        if prior is not None:
            return cast(LocalMegatronTrainingOperation[ResultT], prior)

        async def execute() -> OperationExecutionOutcome:
            try:
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
                self._retained_outcomes[admission.ref.operation_id] = outcome
                return outcome
            except BaseException as error:
                self._ledger.mark_terminal(request.request_id, admission, error=error)
                raise
            finally:
                self._terminal_operation_ids.add(admission.ref.operation_id)
                if self._coordinator is not None or self._outcome_sink is not None:
                    self._awaiting_acknowledgement.add(admission.ref.operation_id)
                self._retire_terminal_ledger_records()

        completion = asyncio.create_task(
            execute(), name=f"local-megatron-{kind}-{request.sequence_id}"
        )
        operation = LocalMegatronTrainingOperation(
            admission.ref, completion, result_type
        )
        self._operations[admission.ref.operation_id] = cast(
            LocalMegatronTrainingOperation[OperationResult], operation
        )
        self._requests[request.request_id] = (
            request,
            kind,
            cast(LocalMegatronTrainingOperation[OperationResult], operation),
        )
        self._request_ids_by_operation[admission.ref.operation_id] = request.request_id
        self._ledger_records[admission.ref.operation_id] = (
            request.request_id,
            admission,
        )
        return operation

    def _retire_terminal_ledger_records(self) -> None:
        open_forward_backward = set(self._ledger.open_forward_backward_operation_ids)
        for operation_id in tuple(self._terminal_operation_ids - open_forward_backward):
            retained = self._ledger_records.pop(operation_id, None)
            if retained is None:
                continue
            request_id, admission = retained
            self._ledger.retire(request_id, admission)
            self._terminal_operation_ids.remove(operation_id)
