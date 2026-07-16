from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Iterable
import contextvars
from types import TracebackType
from typing import Any

from . import PydanticException, Trajectory, TrajectoryGroup
from ._compat import exception_model

_trajectories: contextvars.ContextVar[tuple[Trajectory, ...]] = contextvars.ContextVar(
    "art_trajectories", default=()
)
_groups: contextvars.ContextVar[tuple[TrajectoryGroup, ...]] = contextvars.ContextVar(
    "art_trajectory_groups", default=()
)


def get_current_trajectory(*, required: bool) -> Trajectory | None:
    current = _trajectories.get()
    if current:
        return current[-1]
    if required:
        raise RuntimeError("No trajectory is active in this context")
    return None


def get_current_trajectory_group(*, required: bool) -> TrajectoryGroup | None:
    current = _groups.get()
    if current:
        return current[-1]
    if required:
        raise RuntimeError("No trajectory group is active in this context")
    return None


def enter_trajectory(trajectory: Trajectory) -> Trajectory:
    from ._capture import install

    install()
    _trajectories.set((*_trajectories.get(), trajectory))
    return trajectory


def exit_trajectory(
    trajectory: Trajectory,
    _exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    _traceback: TracebackType | None,
) -> None:
    current = _trajectories.get()
    if not current or current[-1] is not trajectory:
        raise RuntimeError("Trajectory contexts must exit in stack order")
    _trajectories.set(current[:-1])
    trajectory.finish()
    group = get_current_trajectory_group(required=False)
    if group is not None:
        if exc_value is not None:
            group.exceptions.append(exception_model(exc_value))
        elif all(item is not trajectory for item in group.trajectories):
            group.trajectories.append(trajectory)


def enter_trajectory_group(group: TrajectoryGroup) -> TrajectoryGroup:
    _groups.set((*_groups.get(), group))
    return group


def exit_trajectory_group(
    group: TrajectoryGroup,
    _exc_type: type[BaseException] | None,
    _exc_value: BaseException | None,
    _traceback: TracebackType | None,
) -> None:
    current = _groups.get()
    if not current or current[-1] is not group:
        raise RuntimeError("TrajectoryGroup contexts must exit in stack order")
    _groups.set(current[:-1])


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
    if not return_exceptions:
        return TrajectoryGroup(await asyncio.gather(*coroutines))
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    completed: list[Trajectory] = []
    exceptions: list[PydanticException] = []
    for result in results:
        if isinstance(result, BaseException):
            exceptions.append(exception_model(result))
        else:
            completed.append(result)
    return TrajectoryGroup(completed, exceptions=exceptions)
