from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
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
    Awaitable[
        Sequence[Trajectory | TrajectoryGroup]
        | Mapping[str, Sequence[Trajectory | TrajectoryGroup]]
    ],
]
