import asyncio
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Awaitable,
    Iterable,
    Iterator,
    cast,
    overload,
)

import pydantic
from openai.types.chat.chat_completion import Choice
from pydantic import SkipValidation

from .types import Messages, MessagesAndChoices, Tools

MetadataValue = float | int | str | bool | None


class PydanticException(pydantic.BaseModel):
    type: str
    message: str
    traceback: str


class History(pydantic.BaseModel):
    messages_and_choices: Annotated[MessagesAndChoices, SkipValidation]
    tools: Tools | None = None

    def messages(self) -> Messages:
        return get_messages(self.messages_and_choices)


class Trajectory(pydantic.BaseModel):
    messages_and_choices: Annotated[MessagesAndChoices, SkipValidation]
    tools: Tools | None = None
    additional_histories: list[History] = []
    reward: float
    metrics: dict[str, float | int | bool] = {}
    auto_metrics: dict[str, float | int | bool] = {}
    metadata: dict[str, MetadataValue] = {}
    logs: list[str] = []
    start_time: datetime = pydantic.Field(default_factory=datetime.now, exclude=True)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.start_time = datetime.now()

    def log(self, message: str) -> None:
        self.logs.append(message)

    def finish(self) -> "Trajectory":
        duration = (datetime.now() - self.start_time).total_seconds()
        self.metrics["duration"] = duration
        return self

    @asynccontextmanager
    async def track_duration(self, metric_name: str) -> AsyncGenerator[None, None]:
        start_time = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            metric_key = f"{metric_name}_duration"
            self.metrics[metric_key] = self.metrics.get(metric_key, 0.0) + duration

    def __str__(self) -> str:
        return f"Trajectory(reward={self.reward}, metrics={self.metrics}, metadata={self.metadata})"

    def messages(self) -> Messages:
        return get_messages(self.messages_and_choices)

    # Used for logging to console
    def for_logging(self) -> dict[str, Any]:
        loggable_dict = {
            "reward": self.reward,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "messages": [],
            "tools": self.tools,
            "logs": self.logs,
        }
        for message_or_choice in self.messages_and_choices:
            trainable = isinstance(message_or_choice, Choice)
            message = (
                message_or_choice.message.to_dict() if trainable else message_or_choice
            )
            loggable_dict["messages"].append({**message, "trainable": trainable})
        return loggable_dict


def get_messages(messages_and_choices: MessagesAndChoices) -> Messages:
    messages: Messages = []
    for message_or_choice in messages_and_choices:
        if isinstance(message_or_choice, Choice):
            content = message_or_choice.message.content or ""
            tool_calls = message_or_choice.message.tool_calls or []
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    **(
                        {
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": tool_call.type,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                                for tool_call in tool_calls
                            ]
                        }
                        if tool_calls
                        else {}
                    ),  # type: ignore
                }
            )
        else:
            # Ensure content is always a string for tokenizer chat templates
            msg = dict(message_or_choice)
            if msg.get("content") is None:
                msg["content"] = ""
            messages.append(msg)  # type: ignore[arg-type]
    return messages


class TrajectoryGroup(pydantic.BaseModel):
    trajectories: list[Trajectory]
    exceptions: list[PydanticException] = []

    def __init__(
        self,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ),
        *,
        exceptions: list[BaseException] = [],
    ) -> None:
        super().__init__(
            trajectories=[
                trajectory
                for trajectory in trajectories
                if isinstance(trajectory, Trajectory)
            ]
            or getattr(self, "trajectories", []),
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

    def __copy__(self):
        """Support for copy.copy()"""
        import copy

        # Create a new instance using the constructor
        # Pass shallow copies of the lists to avoid shared mutation
        new_instance = self.__class__(
            trajectories=self.trajectories[:],  # Shallow copy of list
            exceptions=[],  # Will be set below
        )
        # Manually copy exceptions since they're PydanticException objects
        new_instance.exceptions = self.exceptions[:]
        return new_instance

    def __deepcopy__(self, memo: dict[int, Any] | None = None):
        """Support for copy.deepcopy()"""
        import copy

        # Initialize memo if not provided
        if memo is None:
            memo = {}

        # Check memo to handle circular references
        if id(self) in memo:
            return memo[id(self)]

        # Create a new instance with deep copies
        new_instance = self.__class__(
            trajectories=copy.deepcopy(self.trajectories, memo),
            exceptions=[],  # Will be set below
        )
        # Register in memo before deep copying attributes to handle circular refs
        memo[id(self)] = new_instance
        # Deep copy exceptions
        new_instance.exceptions = copy.deepcopy(self.exceptions, memo)
        return new_instance

    def __iter__(self) -> Iterator[Trajectory]:  # type: ignore[override]
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Trajectory | BaseException],
        *,
        exceptions: list[BaseException] = [],
    ) -> "TrajectoryGroup": ...

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Awaitable[Trajectory]],
        *,
        exceptions: list[BaseException] = [],
    ) -> Awaitable["TrajectoryGroup"]: ...

    def __new__(
        cls,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ),
        *,
        exceptions: list[BaseException] = [],
    ) -> "TrajectoryGroup | Awaitable[TrajectoryGroup]":
        ts = list(trajectories)
        if any(hasattr(t, "__await__") for t in ts):

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
                )

            class CoroutineWithMetadata:
                def __init__(self, coro, num_trajectories):
                    self.coro = coro
                    self._num_trajectories = num_trajectories

                def __await__(self):
                    return self.coro.__await__()

            coro = _(exceptions.copy())
            return CoroutineWithMetadata(coro, len(ts))
        else:
            group = super().__new__(cls)
            group.__init__(
                trajectories=cast(list[Trajectory | BaseException], ts),
                exceptions=exceptions,
            )
            return group
