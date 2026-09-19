"""Identified operations shared by dedicated clients and native rank-zero views."""

from __future__ import annotations

import asyncio
from copy import copy
from dataclasses import dataclass, field
import hashlib
import inspect
import secrets
from typing import Any

import cloudpickle

from . import _transport

OperationId = tuple[str, int]


@dataclass(frozen=True)
class TrainerOperation:
    id: OperationId
    kind: str
    payload: bytes

    @classmethod
    def capture(cls, id: OperationId, kind: str, payload: Any) -> TrainerOperation:
        """Freeze arguments before asynchronous submission can observe mutations."""
        return cls(id, kind, _transport.encode(payload))


@dataclass
class OperationSequence:
    """Client IDs and cumulative settlement, including work never admitted."""

    session: str = field(default_factory=lambda: secrets.token_hex(16))
    issued: int = 0
    pending: set[int] = field(default_factory=set)
    abandoned: set[int] = field(default_factory=set)

    def next(self) -> OperationId:
        self.issued += 1
        self.pending.add(self.issued)
        return self.session, self.issued

    def acknowledge(self, *ids: OperationId, abandon: bool = False) -> TrainerOperation:
        for session, sequence in ids:
            if session != self.session:
                raise ValueError("trainer operation belongs to another client session")
            self.pending.discard(sequence)
            if abandon:
                self.abandoned.add(sequence)
        return TrainerOperation.capture(
            (self.session, self.issued), "acknowledge", (self.pending, self.abandoned)
        )


@dataclass
class _Outcome:
    fingerprint: tuple[str, bytes]
    completion: asyncio.Future[Any]


@dataclass
class _Acknowledged:
    through: int = 0
    pending: set[int] = field(default_factory=set)


@dataclass
class _Ledger:
    outcomes: dict[OperationId, _Outcome] = field(default_factory=dict)
    acknowledged: dict[str, _Acknowledged] = field(default_factory=dict)

    def retired(self, id: OperationId) -> bool:
        session, sequence = id
        state = self.acknowledged.get(session)
        return (
            state is not None
            and sequence <= state.through
            and sequence not in state.pending
        )

    def acknowledge(self, id: OperationId, pending: set[int]) -> None:
        session, through = id
        if through < 1 or any(
            sequence < 1 or sequence > through for sequence in pending
        ):
            raise ValueError("invalid trainer acknowledgement sequence")
        state = self.acknowledged.setdefault(session, _Acknowledged())
        # Snapshots can arrive out of order. Old snapshots may retire more IDs,
        # but must never resurrect an already retired operation.
        state.pending = {
            sequence
            for sequence in state.pending
            if sequence > through or sequence in pending
        } | {sequence for sequence in pending if sequence > state.through}
        state.through = max(state.through, through)
        for operation_id, outcome in tuple(self.outcomes.items()):
            if self.retired(operation_id) and outcome.completion.done():
                del self.outcomes[operation_id]


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
    """The client retired this operation; its result is no longer available."""


def _ledger(rank_zero: Any) -> _Ledger:
    owner = rank_zero._rank
    if not hasattr(owner, "_operation_outcomes"):
        owner._operation_outcomes = _Ledger()
    return owner._operation_outcomes


async def _abandon(rank_zero: Any, outcome: _Outcome) -> None:
    result = await asyncio.shield(outcome.completion)
    if result is None or isinstance(result, _Failure):
        return
    if outcome.fingerprint[0] == "batches_open":
        cleanup = rank_zero.close_forward_batches(result)
    else:
        cleanup = rank_zero.release_forward((result.handle,))
    if inspect.isawaitable(cleanup):
        await cleanup


async def execute_operation(rank_zero: Any, operation: TrainerOperation) -> Any:
    """Apply at most once, including concurrent retries and failed mutations.

    This ledger is scoped to the actor process lifetime. Worker loss fails the
    session; callers must not transparently replay updates on a replacement.
    """
    ledger = _ledger(rank_zero)
    if operation.kind == "acknowledge":
        pending, abandoned = _transport.decode(operation.payload)
        for sequence in abandoned:
            outcome = ledger.outcomes.get((operation.id[0], sequence))
            if outcome is not None:
                await _abandon(rank_zero, outcome)
        ledger.acknowledge(operation.id, set(pending))
        return None
    if ledger.retired(operation.id):
        raise OperationResultReleasedError(operation.id)
    fingerprint = (operation.kind, hashlib.sha256(operation.payload).digest())
    if (outcome := ledger.outcomes.get(operation.id)) is not None:
        if outcome.fingerprint != fingerprint:
            raise ValueError(
                "trainer operation identity was reused with different arguments"
            )
        result = await asyncio.shield(outcome.completion)
        if isinstance(result, _Failure):
            raise cloudpickle.loads(result.payload) from None
        return result
    completion = asyncio.get_running_loop().create_future()
    ledger.outcomes[operation.id] = _Outcome(fingerprint, completion)
    try:
        payload = _transport.decode(operation.payload)
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
            pending = ledger.outcomes.get(payload.get("pending_operation"))
            if pending is not None:
                await _abandon(rank_zero, pending)
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
    finally:
        # A cancellation can retire an operation before its remote completion.
        # Fence retries immediately, then drop the result once execution settles.
        if ledger.retired(operation.id):
            ledger.outcomes.pop(operation.id, None)
