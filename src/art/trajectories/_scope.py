from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterable, Iterator
from contextlib import contextmanager
import contextvars
from types import TracebackType
from typing import Any

from . import PydanticException, Trajectory, TrajectoryGroup
from ._compat import exception_model

_trajectories: contextvars.ContextVar[tuple[Trajectory, ...]] = contextvars.ContextVar(
    "art_trajectories", default=()
)


def get_current_trajectory(*, required: bool) -> Trajectory | None:
    current = _trajectories.get()
    if current:
        return current[-1]
    if required:
        raise RuntimeError("No trajectory is active in this context")
    return None


def enter_trajectory(trajectory: Trajectory) -> Trajectory:
    from ._capture import install

    install()
    _trajectories.set((*_trajectories.get(), trajectory))
    return trajectory


def exit_trajectory(
    trajectory: Trajectory,
    _exc_type: type[BaseException] | None,
    _exc_value: BaseException | None,
    _traceback: TracebackType | None,
) -> None:
    current = _trajectories.get()
    if not current or current[-1] is not trajectory:
        raise RuntimeError("Trajectory contexts must exit in stack order")
    _trajectories.set(current[:-1])
    trajectory.finish()


@contextmanager
def no_capture() -> Iterator[None]:
    """Hide enclosing trajectory capture while allowing new nested scopes."""

    token = _trajectories.set(())
    try:
        yield
    finally:
        _trajectories.reset(token)


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
    trajectories: Iterable[Coroutine[Any, Any, Trajectory]],
    *,
    return_exceptions: bool,
) -> TrajectoryGroup:
    coroutines = list(trajectories)
    for coroutine in coroutines:
        _require_raw_coroutine(coroutine)
    with no_capture():
        results = await asyncio.gather(
            *coroutines,
            return_exceptions=return_exceptions,
        )
    if not return_exceptions:
        return TrajectoryGroup(results)
    completed: list[Trajectory] = []
    exceptions: list[PydanticException] = []
    for result in results:
        if isinstance(result, BaseException):
            exceptions.append(exception_model(result))
        else:
            completed.append(result)
    return TrajectoryGroup(completed, exceptions=exceptions)
