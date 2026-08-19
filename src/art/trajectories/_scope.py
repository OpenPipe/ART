from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterable, Iterator
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from . import PydanticException, Trajectory, TrajectoryGroup
from ._compat import exception_model


@dataclass(frozen=True, slots=True)
class _TrajectoryScope:
    trajectory: Trajectory


_scopes: contextvars.ContextVar[tuple[_TrajectoryScope, ...]] = contextvars.ContextVar(
    "art_trajectory_scopes", default=()
)


def get_current_trajectory(*, required: bool) -> Trajectory | None:
    current = _scopes.get()
    if current:
        return current[-1].trajectory
    if required:
        raise RuntimeError("No trajectory is active in this context")
    return None


def _get_current_scope() -> _TrajectoryScope | None:
    current = _scopes.get()
    return current[-1] if current else None


def enter_trajectory(trajectory: Trajectory) -> Trajectory:
    from ._capture import install

    install()
    _scopes.set((*_scopes.get(), _TrajectoryScope(trajectory)))
    return trajectory


def exit_trajectory(
    trajectory: Trajectory,
    _exc_type: type[BaseException] | None,
    _exc_value: BaseException | None,
    _traceback: TracebackType | None,
) -> None:
    current = _scopes.get()
    if not current or current[-1].trajectory is not trajectory:
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
    trajectories: Iterable[Coroutine[Any, Any, Trajectory]],
    *,
    return_exceptions: bool,
) -> TrajectoryGroup:
    with no_capture():
        coroutines = list(trajectories)
        for coroutine in coroutines:
            _require_raw_coroutine(coroutine)
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
