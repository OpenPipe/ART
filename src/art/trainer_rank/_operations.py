"""Identified operations shared by dedicated clients and native rank-zero views."""

from __future__ import annotations

import asyncio
from copy import copy
from dataclasses import dataclass
import hashlib
import inspect
from typing import Any

import cloudpickle


@dataclass(frozen=True)
class TrainerOperation:
    id: str
    kind: str
    payload: bytes

    @classmethod
    def capture(cls, id: str, kind: str, payload: Any) -> TrainerOperation:
        """Freeze arguments before asynchronous submission can observe mutations."""
        return cls(id, kind, cloudpickle.dumps(payload))


@dataclass
class _Outcome:
    fingerprint: tuple[str, bytes]
    completion: asyncio.Future[Any] | None


@dataclass(frozen=True)
class _Failure:
    payload: bytes

    @classmethod
    def capture(cls, error: BaseException) -> _Failure:
        # Retaining the exception itself retains its traceback and activation
        # frames. Serialization also gives each retry its own exception object.
        try:
            payload = cloudpickle.dumps(error)
            if len(payload) > 65536 or not isinstance(
                cloudpickle.loads(payload), BaseException
            ):
                raise ValueError("exception cannot be retained as a small outcome")
        except BaseException:
            try:
                message = str(error)[:8192]
            except BaseException:
                message = "exception message unavailable"
            payload = cloudpickle.dumps(
                RuntimeError(f"{type(error).__qualname__}: {message}")
            )
        return cls(payload)


class OperationResultReleasedError(RuntimeError):
    """The operation completed, but its acknowledged result is no longer retained."""


def _ledger(rank_zero: Any) -> dict[str, _Outcome]:
    owner = rank_zero._rank
    if not hasattr(owner, "_operation_outcomes"):
        owner._operation_outcomes = {}
    return owner._operation_outcomes


async def execute_operation(rank_zero: Any, operation: TrainerOperation) -> Any:
    """Apply at most once, including concurrent retries and failed mutations.

    This ledger is scoped to the actor process lifetime. Worker loss fails the
    session; callers must not transparently replay updates on a replacement.
    """
    ledger = _ledger(rank_zero)
    payload = cloudpickle.loads(operation.payload)
    if operation.kind == "acknowledge":
        for operation_id in payload:
            if (outcome := ledger.get(operation_id)) is not None:
                if outcome.completion is not None and not outcome.completion.done():
                    raise RuntimeError(
                        "cannot acknowledge an unfinished trainer operation"
                    )
                outcome.completion = None
        return None
    fingerprint = (operation.kind, hashlib.sha256(operation.payload).digest())
    if (outcome := ledger.get(operation.id)) is not None:
        if outcome.fingerprint != fingerprint:
            raise ValueError(
                "trainer operation identity was reused with different arguments"
            )
        if outcome.completion is None:
            raise OperationResultReleasedError(operation.id)
        result = await asyncio.shield(outcome.completion)
        if isinstance(result, _Failure):
            raise cloudpickle.loads(result.payload) from None
        return result
    completion = asyncio.get_running_loop().create_future()
    ledger[operation.id] = _Outcome(fingerprint, completion)
    try:
        if operation.kind in ("forward", "batches_next"):
            # Only driver replies use CPU transport views. The physical command
            # receives the original policy, and native callback views are intact.
            transport = copy(rank_zero)
            transport._transport_handles = []
            try:
                tree = (
                    transport.forward(**payload)
                    if operation.kind == "forward"
                    else transport.next_forward_batch(**payload)
                )
                result = None if tree is None else transport.export_forward(tree)
            except BaseException:
                if transport._transport_handles:
                    transport._invoke("release", tuple(transport._transport_handles))
                raise
        elif operation.kind == "backward":
            result = rank_zero.backward_packets(**payload)
        elif operation.kind == "optim_step":
            result = rank_zero.optim_step(**payload)
        elif operation.kind == "release":
            result = rank_zero.release_forward(**payload)
        elif operation.kind == "batches_open":
            result = rank_zero.open_forward_batches(**payload)
        elif operation.kind == "batches_close":
            result = rank_zero.close_forward_batches(payload["handle"])
            pending = ledger.get(payload.get("pending_operation"))
            if (
                pending is not None
                and pending.completion is not None
                and pending.completion.done()
            ):
                if (
                    packet := pending.completion.result()
                ) is not None and not isinstance(packet, _Failure):
                    rank_zero.release_forward((packet.handle,))
        elif operation.kind.startswith("head_"):
            from ._heads import execute_head_operation

            result = execute_head_operation(rank_zero, operation.kind, payload)
        else:
            raise ValueError(f"unknown trainer operation: {operation.kind!r}")
        if inspect.isawaitable(result):
            result = await result
    except BaseException as error:
        completion.set_result(_Failure.capture(error))
        raise
    else:
        completion.set_result(result)
        return result
