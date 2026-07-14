from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Coroutine, Iterable, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
import time
from typing import Any, Literal, cast, overload

from anthropic.types import Message as AnthropicMessage
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.responses import Response
import pydantic
from typing_extensions import deprecated

from ..types import Messages, MessagesAndChoices, Tools

MetadataValue = Any


class ChatCompletionsRequest(pydantic.RootModel[dict[str, Any]]):
    """The JSON body sent to an OpenAI-compatible Chat Completions endpoint."""


class CompletionsRequest(pydantic.RootModel[dict[str, Any]]):
    """The JSON body sent to an OpenAI-compatible Completions endpoint."""


class ResponsesRequest(pydantic.RootModel[dict[str, Any]]):
    """The JSON body sent to an OpenAI-compatible Responses endpoint."""


class MessagesRequest(pydantic.RootModel[dict[str, Any]]):
    """The JSON body sent to an Anthropic-compatible Messages endpoint."""


class ChatCompletionsExchange(pydantic.BaseModel):
    request: ChatCompletionsRequest
    response: ChatCompletion
    model: str | None
    start_time: datetime
    end_time: datetime


class CompletionsExchange(pydantic.BaseModel):
    request: CompletionsRequest
    response: Completion
    model: str | None
    start_time: datetime
    end_time: datetime


class ResponsesExchange(pydantic.BaseModel):
    request: ResponsesRequest
    response: Response
    model: str | None
    start_time: datetime
    end_time: datetime


class MessagesExchange(pydantic.BaseModel):
    request: MessagesRequest
    response: AnthropicMessage
    model: str | None
    start_time: datetime
    end_time: datetime


class TrajectoryExchanges(pydantic.BaseModel):
    chat_completions: list[ChatCompletionsExchange] = pydantic.Field(
        default_factory=list
    )
    completions: list[CompletionsExchange] = pydantic.Field(default_factory=list)
    responses: list[ResponsesExchange] = pydantic.Field(default_factory=list)
    messages: list[MessagesExchange] = pydantic.Field(default_factory=list)

    def __bool__(self) -> bool:
        return any(
            (self.chat_completions, self.completions, self.responses, self.messages)
        )


class PydanticException(pydantic.BaseModel):
    type: str
    message: str
    traceback: str


class History(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    tools: Tools | None = None

    def messages(self) -> Messages:
        return get_messages(self.messages_and_choices)


class Trajectory(pydantic.BaseModel):
    exchanges: TrajectoryExchanges = pydantic.Field(default_factory=TrajectoryExchanges)
    messages_and_choices: MessagesAndChoices = pydantic.Field(
        default_factory=list,
    )
    tools: Tools | None = None
    additional_histories: list[History] = pydantic.Field(
        default_factory=list,
    )
    reward: float = 0.0
    initial_policy_version: int | None = None
    final_policy_version: int | None = None
    metrics: dict[str, float | int | bool] = pydantic.Field(default_factory=dict)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    logs: list[str] = pydantic.Field(default_factory=list)
    start_time: datetime = pydantic.Field(default_factory=datetime.now, exclude=True)

    @pydantic.model_validator(mode="after")
    def validate_representation(self) -> Trajectory:
        if self.exchanges and (
            self.messages_and_choices
            or self.tools is not None
            or self.additional_histories
        ):
            raise ValueError(
                "A trajectory cannot contain both exchanges and legacy histories"
            )
        return self

    def __enter__(self) -> Trajectory:
        from ._scope import enter_trajectory

        return enter_trajectory(self)

    def __exit__(self, *exc_info: Any) -> None:
        from ._scope import exit_trajectory

        exit_trajectory(self, *exc_info)

    def log(self, message: str) -> None:
        self.logs.append(message)

    def finish(self) -> Trajectory:
        self.metrics["duration"] = (datetime.now() - self.start_time).total_seconds()
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

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_defaults", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_defaults", True)
        return super().model_dump_json(**kwargs)

    def messages(self) -> Messages:
        return get_messages(self.messages_and_choices)

    def for_logging(self) -> dict[str, Any]:
        from ._compat import trajectory_for_logging

        return trajectory_for_logging(self)


class TrajectoryGroup(pydantic.BaseModel):
    trajectories: list[Trajectory] = pydantic.Field(default_factory=list)
    exceptions: list[PydanticException] = pydantic.Field(default_factory=list)
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    metrics: dict[str, float | int | bool] = pydantic.Field(default_factory=dict)
    logs: list[str] = pydantic.Field(default_factory=list)

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Trajectory | BaseException] = (),
        **kwargs: Any,
    ) -> TrajectoryGroup: ...

    @overload
    @deprecated("Use await art.trajectory_group(...) instead.")
    def __new__(
        cls,
        trajectories: Iterable[Awaitable[Trajectory]],
        **kwargs: Any,
    ) -> Awaitable[TrajectoryGroup]: ...

    def __new__(cls, trajectories: Iterable[Any] = (), **kwargs: Any) -> Any:
        from ._compat import new_trajectory_group

        return new_trajectory_group(cls, trajectories, kwargs)

    def __init__(
        self,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ) = (),
        *,
        exceptions: Iterable[BaseException | PydanticException] = (),
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, float | int | bool] | None = None,
        logs: list[str] | None = None,
    ) -> None:
        from ._compat import init_trajectory_group

        init_trajectory_group(
            self,
            cast(Iterable[Trajectory | BaseException], trajectories),
            exceptions=exceptions,
            metadata=metadata,
            metrics=metrics,
            logs=logs,
        )

    def __enter__(self) -> TrajectoryGroup:
        from ._scope import enter_trajectory_group

        return enter_trajectory_group(self)

    def __exit__(self, *exc_info: Any) -> None:
        from ._scope import exit_trajectory_group

        exit_trajectory_group(self, *exc_info)

    def __copy__(self) -> TrajectoryGroup:
        from ._compat import copy_trajectory_group

        return copy_trajectory_group(self)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> TrajectoryGroup:
        from ._compat import deepcopy_trajectory_group

        return deepcopy_trajectory_group(self, memo)

    def log(self, message: str) -> None:
        self.logs.append(message)

    def __iter__(self) -> Iterator[Trajectory]:  # type: ignore[override]
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_defaults", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("exclude_defaults", True)
        return super().model_dump_json(**kwargs)


class TokenizedTrajectory(pydantic.BaseModel):
    token_ids: list[int]
    logprobs: list[float]
    assistant_mask: list[bool]
    underlying: Trajectory


class TokenizedTrajectoryGroup(pydantic.BaseModel):
    trajectories: list[TokenizedTrajectory]
    underlying: TrajectoryGroup


@overload
def current_trajectory(*, required: Literal[True]) -> Trajectory: ...


@overload
def current_trajectory(*, required: Literal[False] = False) -> Trajectory | None: ...


def current_trajectory(*, required: bool = False) -> Trajectory | None:
    from ._scope import get_current_trajectory

    return get_current_trajectory(required=required)


@overload
def current_trajectory_group(*, required: Literal[True]) -> TrajectoryGroup: ...


@overload
def current_trajectory_group(
    *, required: Literal[False] = False
) -> TrajectoryGroup | None: ...


def current_trajectory_group(*, required: bool = False) -> TrajectoryGroup | None:
    from ._scope import get_current_trajectory_group

    return get_current_trajectory_group(required=required)


async def trajectory(coroutine: Coroutine[Any, Any, Any]) -> Trajectory:
    from ._scope import capture_trajectory

    return await capture_trajectory(coroutine)


async def trajectory_group(
    trajectories: Iterable[Coroutine[Any, Any, Trajectory]],
    *,
    return_exceptions: bool = False,
) -> TrajectoryGroup:
    from ._scope import capture_trajectory_group

    return await capture_trajectory_group(
        trajectories,
        return_exceptions=return_exceptions,
    )


def tokenize_trajectory(
    trajectory: Trajectory,
    base_model: str | None = None,
    *,
    model: str | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> TokenizedTrajectory:
    from ._tokenize import tokenize_one

    return tokenize_one(
        trajectory,
        base_model,
        model=model,
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
    )


def tokenize_trajectories(
    trajectories: Iterable[Trajectory],
    base_model: str | None = None,
    **kwargs: Any,
) -> list[TokenizedTrajectory]:
    return [tokenize_trajectory(item, base_model, **kwargs) for item in trajectories]


def tokenize_trajectory_group(
    group: TrajectoryGroup,
    base_model: str | None = None,
    **kwargs: Any,
) -> TokenizedTrajectoryGroup:
    return TokenizedTrajectoryGroup(
        trajectories=tokenize_trajectories(group, base_model, **kwargs),
        underlying=group,
    )


def tokenize_trajectory_groups(
    groups: Iterable[TrajectoryGroup],
    base_model: str | None = None,
    **kwargs: Any,
) -> list[TokenizedTrajectoryGroup]:
    return [tokenize_trajectory_group(group, base_model, **kwargs) for group in groups]


@deprecated("Use current_trajectory() instead.")
def auto_trajectory(*, required: bool = False) -> Trajectory | None:
    return current_trajectory(required=required)


@deprecated("Use trajectory() instead.")
async def capture_auto_trajectory(
    coroutine: Coroutine[Any, Any, Any],
) -> Trajectory:
    return await trajectory(coroutine)


def get_messages(messages_and_choices: MessagesAndChoices) -> Messages:
    from ._compat import messages_from_legacy_history

    return messages_from_legacy_history(messages_and_choices)


__all__ = [
    "ChatCompletionsRequest",
    "CompletionsRequest",
    "ResponsesRequest",
    "MessagesRequest",
    "ChatCompletionsExchange",
    "CompletionsExchange",
    "ResponsesExchange",
    "MessagesExchange",
    "TrajectoryExchanges",
    "PydanticException",
    "History",
    "Trajectory",
    "TrajectoryGroup",
    "TokenizedTrajectory",
    "TokenizedTrajectoryGroup",
    "MetadataValue",
    "current_trajectory",
    "current_trajectory_group",
    "trajectory",
    "trajectory_group",
    "tokenize_trajectory",
    "tokenize_trajectories",
    "tokenize_trajectory_group",
    "tokenize_trajectory_groups",
    "auto_trajectory",
    "capture_auto_trajectory",
    "get_messages",
]
