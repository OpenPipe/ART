import asyncio
import pydantic
import traceback
from typing import Awaitable, Any, cast, Iterable, Iterator, overload
from openai.types.chat.chat_completion import Choice
from .types import MessagesAndChoices


MetadataValue = float | int | str | bool | None


class PydanticException(pydantic.BaseModel):
    type: str
    message: str
    traceback: str


class Trajectory(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    reward: float
    metrics: dict[str, float] = {}
    metadata: dict[str, MetadataValue] = {}
    logs: list[str] = []

    def __str__(self) -> str:
        return f"Trajectory(reward={self.reward}, metrics={self.metrics}, metadata={self.metadata})"

    def for_logging(self) -> dict[str, Any]:
        loggable_dict = {
            "reward": self.reward,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "messages": [],
            "logs": self.logs,
        }
        for message_or_choice in self.messages_and_choices:
            trainable = isinstance(message_or_choice, Choice)
            message = (
                message_or_choice.message.to_dict() if trainable else message_or_choice
            )
            loggable_dict["messages"].append({**message, "trainable": trainable})
        return loggable_dict


class TrajectoryGroup(pydantic.BaseModel):
    trajectories: list[Trajectory]
    metadata: dict[str, MetadataValue] = {}
    exceptions: list[PydanticException] = []

    def __init__(
        self,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ),
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> None:
        super().__init__(
            trajectories=[
                trajectory
                for trajectory in trajectories
                if isinstance(trajectory, Trajectory)
            ]
            or getattr(self, "trajectories", []),
            metadata=metadata,
            exceptions=[
                PydanticException(
                    type=str(type(exception)),
                    message=str(exception),
                    traceback="\n".join(
                        traceback.format_exception(
                            type(exception), exception, exception.__traceback__
                        )
                    ),
                )
                for exception in (
                    [
                        exception
                        for exception in trajectories
                        if isinstance(exception, BaseException)
                    ]
                    + exceptions
                )
            ],
        )

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Trajectory | BaseException],
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
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ),
        *,
        metadata: dict[str, MetadataValue] = {},
        exceptions: list[BaseException] = [],
    ) -> "TrajectoryGroup | Awaitable[TrajectoryGroup]":
        ts = list(trajectories)
        if all(isinstance(t, Trajectory) or isinstance(t, BaseException) for t in ts):
            group = super().__new__(cls)
            group.__init__(
                trajectories=cast(list[Trajectory | BaseException], ts),
                metadata=metadata,
                exceptions=exceptions,
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

            class CoroutineWithMetadata:
                def __init__(self, coro, num_trajectories):
                    self.coro = coro
                    self._num_trajectories = num_trajectories

                def __await__(self):
                    return self.coro.__await__()

            coro = _(exceptions.copy())
            return CoroutineWithMetadata(coro, len(ts))
