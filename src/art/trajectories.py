import asyncio
import pydantic
from typing import Awaitable, cast, Iterable, Iterator, overload

from .types import MessagesAndChoices


MetadataValue = float | int | str | bool | None


class Trajectory(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    reward: float
    metrics: dict[str, float] = {}
    metadata: dict[str, MetadataValue] = {}


class TrajectoryGroup(pydantic.BaseModel):
    trajectories: list[Trajectory]
    metadata: dict[str, MetadataValue] = {}
    exceptions: list[BaseException] = []

    def __init__(
        self,
        trajectories: Iterable[Trajectory] | Iterable[Awaitable[Trajectory]],
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> None:
        super().__init__(
            trajectories=list(trajectories),
            metadata=metadata,
            exceptions=exceptions,
        )

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Trajectory],
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> "TrajectoryGroup": ...

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Awaitable[Trajectory]],
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> Awaitable["TrajectoryGroup"]: ...

    def __new__(
        cls,
        trajectories: Iterable[Trajectory] | Iterable[Awaitable[Trajectory]],
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> "TrajectoryGroup | Awaitable[TrajectoryGroup]":
        ts = list(trajectories)
        if all(isinstance(t, Trajectory) for t in ts):
            group = super().__new__(cls)
            group.__init__(
                trajectories=cast(list[Trajectory], ts),
                exceptions=exceptions,
                metadata=metadata,
            )
            return group
        else:

            async def _(exceptions: list[BaseException]):
                from .gather import get_gather_context, record_metrics

                context = get_gather_context()
                trajectories = []
                for future in asyncio.as_completed(
                    cast(list[Awaitable[Trajectory]], ts)
                ):
                    try:
                        trajectory = await future
                        trajectories.append(trajectory)
                        record_metrics(context, trajectory)
                        context.update_pbar(n=1)
                    except BaseException as e:
                        exceptions.append(e)
                        context.metric_sums["exceptions"] += 1
                        context.update_pbar(n=0)
                        if context.too_many_exceptions():
                            raise
                return TrajectoryGroup(
                    trajectories=trajectories,
                    exceptions=exceptions,
                    metadata=metadata,
                )

            coro = _(exceptions.copy())
            setattr(coro, "_num_trajectories", len(ts))
            return coro
