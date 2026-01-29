from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import art
from art import Trajectory, TrajectoryGroup

ScenarioT = TypeVar("ScenarioT", bound=dict)
ConfigT = TypeVar("ConfigT")
ScalarMetadataValue = float | int | str | bool | None


RolloutFn = Callable[
    [art.TrainableModel, ScenarioT, ConfigT], Awaitable[TrajectoryGroup]
]

SingleRolloutFn = Callable[
    [art.TrainableModel, ScenarioT, ConfigT], Awaitable[Trajectory]
]

EvalFn = Callable[
    [art.TrainableModel, int, ConfigT],
    Awaitable[list[Trajectory] | dict[str, list[Trajectory]]],
]
