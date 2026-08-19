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


def retain_task(task: asyncio.Task[Any], ownership: set[asyncio.Task[Any]]) -> None:
    if task.done():
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        return
    if task in ownership:
        return
    ownership.add(task)

    def completed(done: asyncio.Task[Any]) -> None:
        ownership.discard(done)
        try:
            done.exception()
        except asyncio.CancelledError:
            pass

    task.add_done_callback(completed)


async def _collect_results(
    tasks: tuple[asyncio.Task[Any], ...],
) -> list[Any]:
    # This join may outlive shutdown, but retain_task keeps it owned and observed.
    return await asyncio.gather(*tasks, return_exceptions=True)


async def settle_tasks(
    tasks: Iterable[asyncio.Task[Any]],
    *,
    deadline: float,
    description: str,
    ownership: set[asyncio.Task[Any]],
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

    completion = asyncio.create_task(
        _collect_results(tracked), name="pipeline_task_settlement"
    )
    watchers = [
        asyncio.create_task(event.wait(), name="pipeline_shutdown_interrupt")
        for event in interrupt_events
        if not event.is_set()
    ]
    try:
        if not (cancel or interrupted):
            done, _ = await asyncio.wait(
                (completion, *watchers),
                timeout=max(0.0, deadline - loop.time()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            interrupted = any(watcher in done for watcher in watchers)
            if interrupted:
                for task in tracked:
                    if not task.done():
                        task.cancel()
        if not completion.done():
            await asyncio.wait((completion,), timeout=max(0.0, deadline - loop.time()))
    except BaseException:
        for task in tracked:
            if not task.done():
                task.cancel()
                retain_task(task, ownership)
        retain_task(completion, ownership)
        raise
    finally:
        for watcher in watchers:
            watcher.cancel()
            retain_task(watcher, ownership)

    if completion.done() and not completion.cancelled():
        results = completion.result()
        failures = unique_exception_leaves(
            result
            for result in results
            if isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        )
    else:
        failures = unique_exception_leaves(
            task.exception() for task in tracked if task.done() and not task.cancelled()
        )

    live = tuple(task for task in tracked if not task.done())
    if not completion.done():
        retain_task(completion, ownership)
    if live:
        for task in live:
            task.cancel()
            retain_task(task, ownership)
        failures.append(
            TimeoutError(
                f"{len(live)} live task(s) remain in {description} after the "
                "pipeline shutdown deadline."
            )
        )
    return failures
