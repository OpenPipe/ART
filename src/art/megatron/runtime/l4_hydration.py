from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import traceback
from typing import Any, Generic, Protocol, TypeVar

from .portable_snapshot import (
    PortableSnapshotArchive,
    PortableSnapshotReadReceipt,
)
from .specs import TrainerGeneration

_T = TypeVar("_T")


class _BlockingReceive(Protocol):
    def get(self) -> Any: ...


class _RequestReceiver(Protocol):
    def recv(self) -> _BlockingReceive: ...


class _RankHydrationExecutor(Protocol):
    def prepare_run_checkpoint(
        self,
        operation_id: str,
        run_id: str,
        generation: TrainerGeneration,
        archive: PortableSnapshotArchive,
        *,
        restore_optimizer: bool,
    ) -> PortableSnapshotReadReceipt: ...

    def commit_prepared_run_checkpoint(
        self, operation_id: str, run_id: str
    ) -> PortableSnapshotReadReceipt: ...

    def discard_prepared_run_checkpoint(
        self, operation_id: str, run_id: str | None = None
    ) -> bool: ...


@dataclass(slots=True)
class _PendingHydration(Generic[_T]):
    operation_id: str
    fingerprint: str
    task: asyncio.Task[_T]
    discard: Callable[[], Awaitable[None]]


class L4HydrationSidecar(Generic[_T]):
    """Bound one authenticated L4 restore to its load-state operation."""

    def __init__(self) -> None:
        self._pending: _PendingHydration[_T] | None = None
        self._closed = False

    async def prefetch(
        self,
        *,
        operation_id: str,
        fingerprint: str,
        hydrate: Callable[[], Awaitable[_T]],
        discard: Callable[[], Awaitable[None]],
    ) -> _T:
        if self._closed:
            raise RuntimeError("L4 hydration sidecar is closed")
        pending = self._pending
        if pending is None:

            async def run_hydration() -> _T:
                return await hydrate()

            task = asyncio.create_task(
                run_hydration(), name=f"megatron-l4-hydration-{operation_id}"
            )
            pending = _PendingHydration(
                operation_id=operation_id,
                fingerprint=fingerprint,
                task=task,
                discard=discard,
            )
            self._pending = pending
        elif pending.operation_id != operation_id or pending.fingerprint != fingerprint:
            raise RuntimeError("another L4 hydration is pending for this run")
        return await pending.task

    async def acknowledge(self, *, operation_id: str, fingerprint: str) -> None:
        pending = self._require(operation_id, fingerprint)
        if not pending.task.done() or pending.task.cancelled():
            raise RuntimeError("L4 hydration is not ready to commit")
        pending.task.result()
        self._pending = None

    async def discard(self, operation_id: str) -> None:
        pending = self._pending
        if pending is None:
            return
        if pending.operation_id != operation_id:
            raise RuntimeError("L4 hydration discard changed operation identity")
        if not pending.task.done():
            pending.task.cancel()
        await asyncio.gather(pending.task, return_exceptions=True)
        await pending.discard()
        if self._pending is pending:
            self._pending = None

    async def aclose(self) -> None:
        pending = self._pending
        if pending is not None:
            await self.discard(pending.operation_id)
        self._closed = True

    def _require(self, operation_id: str, fingerprint: str) -> _PendingHydration[_T]:
        pending = self._pending
        if pending is None:
            raise RuntimeError("L4 hydration is absent")
        if pending.operation_id != operation_id or pending.fingerprint != fingerprint:
            raise RuntimeError("L4 hydration changed operation identity")
        return pending


class RankL4HydrationService:
    """Run one rank's portable hydration work outside the actor request thread."""

    def __init__(self, executor: _RankHydrationExecutor, *, rank: int) -> None:
        self._executor = executor
        self._rank = rank

    def run(self, receiver: _RequestReceiver) -> None:
        while (request := receiver.recv().get()) is not None:
            (
                action,
                operation_id,
                run_id,
                generation_json,
                archive_json,
                restore_optimizer,
                reply,
            ) = request
            result = {
                "rank": self._rank,
                "run_id": run_id,
                "operation_id": operation_id,
            }
            try:
                if action == "prepare":
                    generation = TrainerGeneration.model_validate_json(generation_json)
                    archive = PortableSnapshotArchive.model_validate_json(archive_json)
                    receipt = self._executor.prepare_run_checkpoint(
                        operation_id,
                        run_id,
                        generation,
                        archive,
                        restore_optimizer=restore_optimizer,
                    )
                    result["receipt"] = receipt.model_dump(mode="json")
                elif action == "commit":
                    receipt = self._executor.commit_prepared_run_checkpoint(
                        operation_id, run_id
                    )
                    result["receipt"] = receipt.model_dump(mode="json")
                elif action == "discard":
                    result["discarded"] = (
                        self._executor.discard_prepared_run_checkpoint(
                            operation_id, run_id
                        )
                    )
                else:
                    raise ValueError(f"unknown L4 hydration action {action!r}")
            except BaseException as error:
                result.update(
                    error_type=type(error).__name__,
                    message=str(error),
                    traceback_text=traceback.format_exc(),
                )
            reply.send(result)


def raise_l4_hydration_failures(
    results: tuple[dict[str, object], ...], *, label: str
) -> None:
    failures = [result for result in results if result.get("error_type") is not None]
    if not failures:
        return
    details = "\n".join(
        f"rank {result['rank']}: {result['error_type']}: {result.get('message', '')}\n"
        f"{result.get('traceback_text', '')}"
        for result in failures
    )
    raise RuntimeError(f"L4 checkpoint {label} failed:\n{details}")
