from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .contracts import (
    CommandExecutionUsage,
    Contract,
    ForwardBackwardRequest,
    ForwardRequest,
    LoadStateRequest,
    OperationRef,
    OperationResultType,
    OptimStepRequest,
    RunCommand,
    SaveStateRequest,
    SaveWeightsForSamplerRequest,
)

OperationHandler = Callable[
    [RunCommand, OperationRef, tuple[str, ...]], Awaitable[OperationResultType]
]
OperationFailureCode = Literal[
    "invalid_request",
    "operation_conflict",
    "capacity_exhausted",
    "cancelled",
    "execution_failed",
]


class TerminalOperationFailure(Contract):
    code: OperationFailureCode
    error_type: str = Field(min_length=1, max_length=255)
    message: str = Field(min_length=1, max_length=2048)
    usage: CommandExecutionUsage


class OperationExecutionError(RuntimeError):
    """Typed handler failure carrying exact partial or unknown producer usage."""

    def __init__(
        self,
        code: OperationFailureCode,
        message: str,
        *,
        usage: CommandExecutionUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.usage = usage or CommandExecutionUsage.unknown()


class OperationSucceeded(Contract):
    status: Literal["succeeded"] = "succeeded"
    operation: OperationRef
    result: OperationResultType

    @model_validator(mode="after")
    def _validate_result(self) -> "OperationSucceeded":
        if (
            self.result.operation_id != self.operation.operation_id
            or self.result.kind != self.operation.kind
        ):
            raise ValueError("operation result identity or kind changed")
        return self


class OperationFailed(Contract):
    status: Literal["failed"] = "failed"
    operation: OperationRef
    failure: TerminalOperationFailure


OperationExecutionOutcome = Annotated[
    OperationSucceeded | OperationFailed, Field(discriminator="status")
]


class OperationWorker:
    """Serialize exact run operations and replay retained terminal outcomes."""

    def __init__(
        self,
        handler: OperationHandler,
        *,
        max_retained_operations: int = 128,
    ) -> None:
        if not 1 <= max_retained_operations <= 1024:
            raise ValueError("max_retained_operations must be between 1 and 1024")
        self._handler = handler
        self._max_retained_operations = max_retained_operations
        self._outcomes: dict[str, tuple[str, OperationExecutionOutcome]] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        request: RunCommand,
        operation: OperationRef,
        contributing_forward_backward_operation_ids: tuple[str, ...] = (),
    ) -> OperationExecutionOutcome:
        fingerprint = _execution_fingerprint(
            request, operation, contributing_forward_backward_operation_ids
        )
        async with self._lock:
            cached = self._outcomes.get(operation.operation_id)
            if cached is not None:
                if cached[0] == fingerprint:
                    return cached[1]
                return _failed(
                    operation,
                    "operation_conflict",
                    RuntimeError("operation_id was reused for different execution"),
                    usage=CommandExecutionUsage.no_work(),
                )
            try:
                _validate_execution(
                    request, operation, contributing_forward_backward_operation_ids
                )
                if len(self._outcomes) >= self._max_retained_operations:
                    raise _TerminalExecutionError(
                        "capacity_exhausted",
                        "retire a durable terminal operation before admitting another",
                    )
                result = await self._handler(
                    request, operation, contributing_forward_backward_operation_ids
                )
                outcome: OperationExecutionOutcome = OperationSucceeded(
                    operation=operation, result=result
                )
            except OperationExecutionError as error:
                outcome = _failed(operation, error.code, error, usage=error.usage)
            except asyncio.CancelledError as error:
                outcome = _failed(
                    operation,
                    "cancelled",
                    error,
                    usage=CommandExecutionUsage.unknown(),
                )
            except (TypeError, ValueError) as error:
                outcome = _failed(
                    operation,
                    "invalid_request",
                    error,
                    usage=CommandExecutionUsage.no_work(),
                )
            except Exception as error:
                outcome = _failed(
                    operation,
                    "execution_failed",
                    error,
                    usage=CommandExecutionUsage.unknown(),
                )
            self._outcomes[operation.operation_id] = (fingerprint, outcome)
            return outcome

    def retire(self, operation_id: str) -> None:
        self._outcomes.pop(operation_id, None)


def bootstrap_operation_worker(
    handler: OperationHandler,
    *,
    max_retained_operations: int = 128,
) -> OperationWorker:
    return OperationWorker(handler, max_retained_operations=max_retained_operations)


class _TerminalExecutionError(OperationExecutionError):
    def __init__(self, code: OperationFailureCode, message: str) -> None:
        super().__init__(code, message, usage=CommandExecutionUsage.no_work())


def _validate_execution(
    request: RunCommand,
    operation: OperationRef,
    contributions: tuple[str, ...],
) -> None:
    expected = {
        "forward": ForwardRequest,
        "forward_backward": ForwardBackwardRequest,
        "optim_step": OptimStepRequest,
        "save_sampler": SaveWeightsForSamplerRequest,
        "save_state": SaveStateRequest,
        "load_state": LoadStateRequest,
    }[operation.kind]
    if type(request) is not expected:
        raise TypeError(
            f"{operation.kind} requires {expected.__name__}, got {type(request).__name__}"
        )
    if (
        request.run_id != operation.run_id
        or request.sequence_id != operation.sequence_id
    ):
        raise ValueError("request and operation identity differ")
    if operation.kind == "optim_step":
        if not contributions or len(set(contributions)) != len(contributions):
            raise ValueError("optimizer requires unique F/B contribution IDs")
    elif contributions:
        raise ValueError("only optimizer execution accepts F/B contribution IDs")


def _execution_fingerprint(
    request: RunCommand,
    operation: OperationRef,
    contributions: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "request": request.model_dump(mode="json"),
            "operation": operation.model_dump(mode="json"),
            "contributions": contributions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _failed(
    operation: OperationRef,
    code: OperationFailureCode,
    error: BaseException,
    *,
    usage: CommandExecutionUsage,
) -> OperationFailed:
    message = str(error).strip() or type(error).__name__
    return OperationFailed(
        operation=operation,
        failure=TerminalOperationFailure(
            code=code,
            error_type=type(error).__name__,
            message=message[:2048],
            usage=usage,
        ),
    )
