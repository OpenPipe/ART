from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import Any


def unique_exception_leaves(
    errors: Iterable[BaseException | None],
) -> list[BaseException]:
    leaves: list[BaseException] = []
    seen: set[int] = set()

    def visit(error: BaseException | None) -> None:
        if error is None:
            return
        if isinstance(error, BaseExceptionGroup):
            for child in error.exceptions:
                visit(child)
            return
        if id(error) not in seen:
            seen.add(id(error))
            leaves.append(error)

    for error in errors:
        visit(error)
    return leaves


async def settle_tasks(
    tasks: Iterable[asyncio.Task[Any]],
    *,
    deadline: float,
    description: str,
    cancel: bool = False,
    interrupt_events: Sequence[asyncio.Event] = (),
) -> list[BaseException]:
    tracked = tuple(dict.fromkeys(tasks))
    if not tracked:
        return []

    loop = asyncio.get_running_loop()
    interrupted = any(event.is_set() for event in interrupt_events)
    if cancel or interrupted:
        for task in tracked:
            task.cancel()

    completion = asyncio.gather(*tracked, return_exceptions=True)
    watchers = [
        asyncio.create_task(event.wait())
        for event in interrupt_events
        if not event.is_set()
    ]
    timed_out = False
    pending_count = 0
    try:
        if not (cancel or interrupted):
            done, _ = await asyncio.wait(
                (completion, *watchers),
                timeout=max(0.0, deadline - loop.time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            interrupted = any(watcher in done for watcher in watchers)
            timed_out = completion not in done and not interrupted
            if interrupted or timed_out:
                pending_count = sum(not task.done() for task in tracked)
                for task in tracked:
                    if not task.done():
                        task.cancel()
        results = await completion
    finally:
        for watcher in watchers:
            watcher.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)

    failures = unique_exception_leaves(
        result
        for result in results
        if isinstance(result, BaseException)
        and not isinstance(result, asyncio.CancelledError)
    )
    if timed_out:
        failures.append(
            TimeoutError(
                f"{pending_count} {description} did not "
                "stop before the pipeline shutdown deadline."
            )
        )
    return failures
