"""Deployment-neutral fanout semantics for public mutable-LoRA updates.

The inference frontend owns holder discovery and routing eligibility.  This
module defines the ordering boundary without importing ART training APIs or a
particular deployment control plane.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar
from uuid import uuid4

SourceT = TypeVar("SourceT", covariant=True)


@dataclass(frozen=True)
class LoraMutation(Generic[SourceT]):
    owner_id: str
    request_id: str
    lora_slot: str
    expected_update_seq: int
    policy_version: int
    generation_id: str
    source: SourceT


@dataclass(frozen=True)
class HolderMutation:
    lora_slot: str
    update_seq: int
    policy_version: int
    generation_id: str
    source: object


@dataclass(frozen=True)
class HolderAck:
    holder_id: str
    lora_slot: str
    update_seq: int
    generation_id: str


@dataclass(frozen=True)
class MutationAdmission:
    operation_id: str
    update_seq: int
    targeted_holders: int


@dataclass(frozen=True)
class MutationResult:
    operation_id: str
    update_seq: int
    succeeded_holders: tuple[str, ...]
    failed_holders: tuple[str, ...]


class HolderDirectory(Protocol):
    async def current_holders(
        self, owner_id: str, lora_slot: str
    ) -> tuple[str, ...]: ...

    async def set_eligible(
        self,
        owner_id: str,
        lora_slot: str,
        holder_id: str,
        *,
        eligible: bool,
        update_seq: int,
    ) -> None: ...


HolderApply = Callable[[str, HolderMutation], Awaitable[HolderAck]]


@dataclass
class _SlotState:
    update_seq: int = 0
    pending: asyncio.Task[MutationResult] | None = None


class HolderUpdateCoordinator:
    """Admit one ordered fanout per owned slot and quarantine failed holders."""

    def __init__(
        self,
        directory: HolderDirectory,
        apply: HolderApply,
        *,
        holder_timeout_s: float = 300.0,
        max_holders_per_update: int = 1024,
        max_operations: int = 65_536,
    ) -> None:
        if min(holder_timeout_s, max_holders_per_update, max_operations) <= 0:
            raise ValueError("holder update limits must be positive")
        self._directory = directory
        self._apply = apply
        self._holder_timeout_s = holder_timeout_s
        self._max_holders_per_update = max_holders_per_update
        self._max_operations = max_operations
        self._states: dict[tuple[str, str], _SlotState] = {}
        self._operations: dict[str, asyncio.Task[MutationResult]] = {}
        self._request_ids: dict[
            tuple[str, str], tuple[LoraMutation[object], MutationAdmission]
        ] = {}
        self._lock = asyncio.Lock()

    async def admit(self, mutation: LoraMutation[SourceT]) -> MutationAdmission:
        self._validate_mutation(mutation)
        key = (mutation.owner_id, mutation.lora_slot)
        async with self._lock:
            request_key = (mutation.owner_id, mutation.request_id)
            prior = self._request_ids.get(request_key)
            if prior is not None:
                if prior[0] != mutation:
                    raise RuntimeError(
                        "LoRA mutation request id was reused with different content"
                    )
                return prior[1]
            if len(self._operations) >= self._max_operations:
                raise RuntimeError("LoRA mutation operation capacity is exhausted")
            state = self._states.setdefault(key, _SlotState())
            if state.pending is not None and not state.pending.done():
                raise RuntimeError("a LoRA update is already active for this slot")
            if mutation.expected_update_seq != state.update_seq:
                raise RuntimeError(
                    "LoRA update sequence precondition failed: "
                    f"expected {mutation.expected_update_seq}, current {state.update_seq}"
                )
            holders = tuple(
                sorted(
                    set(
                        await self._directory.current_holders(
                            mutation.owner_id, mutation.lora_slot
                        )
                    )
                )
            )
            if not holders:
                raise RuntimeError("mutable LoRA slot has no current holders")
            if len(holders) > self._max_holders_per_update:
                raise RuntimeError(
                    "mutable LoRA update exceeds its holder fanout bound"
                )
            update_seq = state.update_seq + 1
            operation_id = uuid4().hex
            command = HolderMutation(
                lora_slot=mutation.lora_slot,
                update_seq=update_seq,
                policy_version=mutation.policy_version,
                generation_id=mutation.generation_id,
                source=mutation.source,
            )
            task = asyncio.create_task(
                self._fanout(operation_id, mutation.owner_id, holders, command),
                name=f"lora-holder-fanout-{update_seq}",
            )
            task.add_done_callback(_consume_task)
            state.update_seq = update_seq
            state.pending = task
            self._operations[operation_id] = task
            admission = MutationAdmission(
                operation_id=operation_id,
                update_seq=update_seq,
                targeted_holders=len(holders),
            )
            self._request_ids[request_key] = (mutation, admission)
            return admission

    async def wait(self, operation_id: str) -> MutationResult:
        async with self._lock:
            task = self._operations.get(operation_id)
            if task is None:
                raise KeyError("LoRA update operation was not found")
        return await asyncio.shield(task)

    async def _fanout(
        self,
        operation_id: str,
        owner_id: str,
        holders: tuple[str, ...],
        command: HolderMutation,
    ) -> MutationResult:
        async def apply_one(holder_id: str) -> tuple[str, bool]:
            success = False
            try:
                ack = await asyncio.wait_for(
                    self._apply(holder_id, command), timeout=self._holder_timeout_s
                )
                success = ack == HolderAck(
                    holder_id=holder_id,
                    lora_slot=command.lora_slot,
                    update_seq=command.update_seq,
                    generation_id=command.generation_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                success = False
            try:
                await self._directory.set_eligible(
                    owner_id,
                    command.lora_slot,
                    holder_id,
                    eligible=success,
                    update_seq=command.update_seq,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # If routing state cannot be committed, fail closed. A later
                # successful update may make this holder eligible again.
                success = False
                await self._directory.set_eligible(
                    owner_id,
                    command.lora_slot,
                    holder_id,
                    eligible=False,
                    update_seq=command.update_seq,
                )
            return holder_id, success

        outcomes = await asyncio.gather(*(apply_one(holder) for holder in holders))
        succeeded = tuple(holder for holder, success in outcomes if success)
        failed = tuple(holder for holder, success in outcomes if not success)
        return MutationResult(
            operation_id=operation_id,
            update_seq=command.update_seq,
            succeeded_holders=succeeded,
            failed_holders=failed,
        )

    @staticmethod
    def _validate_mutation(mutation: LoraMutation[object]) -> None:
        if not all(
            (
                mutation.owner_id,
                mutation.request_id,
                mutation.lora_slot,
                mutation.generation_id,
            )
        ):
            raise ValueError("LoRA mutation identity fields must be non-empty")
        if mutation.expected_update_seq < 0 or mutation.policy_version < 0:
            raise ValueError("LoRA mutation sequence and policy must be non-negative")


def _consume_task(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()
