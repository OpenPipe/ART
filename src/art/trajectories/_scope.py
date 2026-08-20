from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Coroutine, Iterable, Iterator
from contextlib import contextmanager
import contextvars
from types import TracebackType
from typing import Any

from . import MetadataValue, PydanticException, Trajectory, TrajectoryGroup

_scopes: contextvars.ContextVar[tuple[Trajectory, ...]] = contextvars.ContextVar(
    "art_trajectory_scopes", default=()
)


def get_current_trajectory(*, required: bool) -> Trajectory | None:
    current = _scopes.get()
    if current:
        return current[-1]
    if required:
        raise RuntimeError("No trajectory is active in this context")
    return None


def enter_trajectory(trajectory: Trajectory) -> Trajectory:
    from ._capture import install

    install()
    _scopes.set((*_scopes.get(), trajectory))
    return trajectory


def exit_trajectory(
    trajectory: Trajectory,
    _exc_type: type[BaseException] | None,
    _exc_value: BaseException | None,
    _traceback: TracebackType | None,
) -> None:
    current = _scopes.get()
    if not current or current[-1] is not trajectory:
        raise RuntimeError("Trajectory contexts must exit in stack order")
    _scopes.set(current[:-1])
    trajectory.finish()


@contextmanager
def no_capture() -> Iterator[None]:
    """Hide enclosing trajectory capture while allowing new nested scopes."""

    token = _scopes.set(())
    try:
        yield
    finally:
        _scopes.reset(token)


def _require_raw_coroutine(value: object) -> None:
    if isinstance(value, (asyncio.Task, asyncio.Future)) or not isinstance(
        value, Coroutine
    ):
        raise TypeError("Expected a raw coroutine, not a Task, Future, or awaitable")


async def capture_trajectory(coroutine: Coroutine[Any, Any, object]) -> Trajectory:
    _require_raw_coroutine(coroutine)
    with Trajectory() as captured:
        await coroutine
    return captured


async def capture_trajectory_group(
    trajectories: Iterable[Trajectory | BaseException | Awaitable[Trajectory]],
    *,
    exceptions: Iterable[BaseException | PydanticException],
    metadata: dict[str, MetadataValue] | None,
    metrics: dict[str, float | int | bool] | None,
    logs: list[str] | None,
    return_exceptions: bool,
) -> TrajectoryGroup:
    with no_capture():
        provided_exceptions = list(exceptions)
        for error in provided_exceptions:
            if isinstance(error, BaseException) and not isinstance(error, Exception):
                raise error
        results: list[Trajectory | BaseException | None] = []
        pending: dict[asyncio.Future[Trajectory], list[int]] = {}
        by_identity: dict[int, asyncio.Future[Trajectory]] = {}
        try:
            for item in trajectories:
                index = len(results)
                if isinstance(item, (Trajectory, BaseException)):
                    if isinstance(item, BaseException) and not isinstance(
                        item, Exception
                    ):
                        raise item
                    results.append(item)
                    continue
                results.append(None)
                task = by_identity.get(id(item))
                if task is None:
                    task = asyncio.ensure_future(item)
                    by_identity[id(item)] = task
                    pending[task] = []
                pending[task].append(index)
            while pending:
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    indexes = pending.pop(task)
                    try:
                        result: Trajectory | BaseException = task.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        if not return_exceptions:
                            raise
                        result = error
                    if not isinstance(result, (Trajectory, BaseException)):
                        raise TypeError(
                            "trajectory_group awaitables must resolve to trajectories"
                        )
                    if isinstance(result, BaseException) and not isinstance(
                        result, Exception
                    ):
                        raise result
                    for index in indexes:
                        results[index] = result
        except BaseException:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise
    return TrajectoryGroup(
        (result for result in results if result is not None),
        exceptions=provided_exceptions,
        metadata=metadata,
        metrics=metrics,
        logs=logs,
    )
