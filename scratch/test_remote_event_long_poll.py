from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from art.serverless.client import _RunEventObserver
from art.serverless.contracts import EventPage, RunEvent


class _BlockingEvents:
    def __init__(self, event: RunEvent) -> None:
        self.event = event
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def get_events(self, run_id: str, *, after: int) -> EventPage:
        self.calls += 1
        assert run_id == self.event.run_id
        self.started.set()
        await self.release.wait()
        if self.event.cursor <= after:
            return EventPage(events=(), next_cursor=after)
        return EventPage(events=(self.event,), next_cursor=self.event.cursor)


@pytest.mark.asyncio
async def test_observer_keeps_one_long_poll_outstanding() -> None:
    service = _BlockingEvents(
        RunEvent(
            cursor=1,
            run_id="run",
            operation_id="operation",
            event="operation_succeeded",
            created_at=datetime.now(UTC),
        )
    )
    observer = _RunEventObserver(service, "run")
    terminal = observer.reserve("operation")
    observer.claim("operation", terminal)
    await service.started.wait()
    await asyncio.sleep(0.15)
    assert service.calls == 1
    service.release.set()
    assert (await terminal).operation_id == "operation"
    await asyncio.sleep(0)
    assert service.calls == 1
    await observer.close()


@pytest.mark.asyncio
async def test_run_terminal_event_is_retained_without_polling_status() -> None:
    service = _BlockingEvents(
        RunEvent(
            cursor=3,
            run_id="run",
            event="run_closed",
            created_at=datetime.now(UTC),
        )
    )
    observer = _RunEventObserver(service, "run")
    terminal = observer.reserve_run_terminal()
    await service.started.wait()
    service.release.set()
    assert (await terminal).event == "run_closed"
    assert (await observer.reserve_run_terminal()).cursor == 3
    assert service.calls == 1
    await observer.close()
