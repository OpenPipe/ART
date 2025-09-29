import contextvars
from typing import Any, Coroutine

from .trajectories import Trajectory


def yield_trajectory(trajectory: Trajectory) -> None:
    yield_trajectory_context_var.get().trajectory = trajectory


async def capture_yielded_trajectory(coroutine: Coroutine[Any, Any, Any]) -> Trajectory:
    with YieldTrajectoryContext():
        await coroutine
        trajectory = yield_trajectory_context_var.get().trajectory
        if trajectory is None:
            raise RuntimeError("No trajectory yielded")
        return trajectory


class YieldTrajectoryContext:
    def __init__(self) -> None:
        self.trajectory: Trajectory | None = None

    def __enter__(self) -> None:
        self.token = yield_trajectory_context_var.set(self)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        yield_trajectory_context_var.reset(self.token)


yield_trajectory_context_var: contextvars.ContextVar[YieldTrajectoryContext] = (
    contextvars.ContextVar("yield_trajectory_context", default=YieldTrajectoryContext())
)
