from __future__ import annotations

from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Coroutine,
    Iterable,
    Iterator,
    Mapping,
)
from contextlib import asynccontextmanager
from datetime import datetime
from enum import IntFlag
import time
from types import TracebackType
from typing import Annotated, Any, Literal, TypeAlias, overload

from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    MessageParam as AnthropicMessageParam,
)
from anthropic.types import (
    TextBlockParam as AnthropicTextBlockParam,
)
from anthropic.types import (
    ThinkingConfigParam as AnthropicThinkingConfigParam,
)
from anthropic.types import (
    ToolUnionParam as AnthropicToolParam,
)
from openai.types import Completion
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.responses import (
    Response,
    ResponseInputParam,
)
from openai.types.responses import (
    ToolParam as ResponsesToolParam,
)
import pydantic
from typing_extensions import TypedDict, deprecated

from ..types import Messages, MessagesAndChoices, Tools
from ._serialization import (
    _CompactModel,
    serialize_chat_completion,
    serialize_messages_and_choices,
)

# Deliberately open: Pydantic enforces serializability when callers dump in JSON mode.
MetadataValue = Any


class TokenFlag(IntFlag):
    """Independent facts about a token; members may be combined."""

    # The ID came from inference metadata rather than client-side tokenization.
    EXACT = 1 << 0
    # The token belongs to a model-sampled response rather than its prompt.
    SAMPLED = 1 << 1


class ChatCompletionsRequest(TypedDict, total=False, extra_items=Any):
    """The JSON body sent to an OpenAI-compatible Chat Completions endpoint."""

    model: str
    messages: list[ChatCompletionMessageParam]
    stream: bool
    tools: list[ChatCompletionToolParam]
    max_completion_tokens: int
    max_tokens: int
    temperature: float
    top_p: float
    logprobs: bool
    top_logprobs: int
    chat_template: str
    chat_template_kwargs: dict[str, Any]


class CompletionsRequest(TypedDict, total=False, extra_items=Any):
    """The JSON body sent to an OpenAI-compatible Completions endpoint."""

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    stream: bool
    max_tokens: int
    temperature: float
    top_p: float
    logprobs: int
    echo: bool
    stop: str | list[str]
    seed: int


class ResponsesRequest(TypedDict, total=False, extra_items=Any):
    """The JSON body sent to an OpenAI-compatible Responses endpoint."""

    model: str
    input: str | ResponseInputParam
    instructions: str
    previous_response_id: str
    stream: bool
    tools: list[ResponsesToolParam]
    max_output_tokens: int
    temperature: float
    top_p: float
    chat_template: str
    chat_template_kwargs: dict[str, Any]


class MessagesRequest(TypedDict, total=False, extra_items=Any):
    """The JSON body sent to an Anthropic-compatible Messages endpoint."""

    model: str
    messages: list[AnthropicMessageParam]
    max_tokens: int
    stream: bool
    system: str | list[AnthropicTextBlockParam]
    tools: list[AnthropicToolParam]
    thinking: AnthropicThinkingConfigParam
    temperature: float
    top_p: float
    top_k: int
    stop_sequences: list[str]
    chat_template: str
    chat_template_kwargs: dict[str, Any]


class ChatCompletionsExchange(pydantic.BaseModel):
    request: Annotated[
        pydantic.SerializeAsAny[ChatCompletionsRequest], pydantic.SkipValidation
    ]
    response: ChatCompletion
    start_time: datetime
    end_time: datetime

    @pydantic.field_serializer("response", when_used="json")
    def serialize_response(self, response: ChatCompletion) -> dict[str, Any]:
        return serialize_chat_completion(response)

    @pydantic.computed_field
    @property
    def model(self) -> str | None:
        requested = self.request.get("model")
        return requested if isinstance(requested, str) else self.response.model


class CompletionsExchange(pydantic.BaseModel):
    request: Annotated[
        pydantic.SerializeAsAny[CompletionsRequest], pydantic.SkipValidation
    ]
    response: Completion
    start_time: datetime
    end_time: datetime

    @pydantic.computed_field
    @property
    def model(self) -> str | None:
        requested = self.request.get("model")
        return requested if isinstance(requested, str) else self.response.model


class ResponsesExchange(pydantic.BaseModel):
    request: Annotated[
        pydantic.SerializeAsAny[ResponsesRequest], pydantic.SkipValidation
    ]
    response: Response
    start_time: datetime
    end_time: datetime

    @pydantic.computed_field
    @property
    def model(self) -> str | None:
        requested = self.request.get("model")
        return requested if isinstance(requested, str) else self.response.model


class MessagesExchange(pydantic.BaseModel):
    request: Annotated[
        pydantic.SerializeAsAny[MessagesRequest], pydantic.SkipValidation
    ]
    response: AnthropicMessage
    start_time: datetime
    end_time: datetime

    @pydantic.computed_field
    @property
    def model(self) -> str | None:
        requested = self.request.get("model")
        return requested if isinstance(requested, str) else self.response.model


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

    @pydantic.field_serializer("messages_and_choices", when_used="json")
    def serialize_messages_and_choices(self, value: MessagesAndChoices) -> list[Any]:
        return serialize_messages_and_choices(value)

    def messages(self) -> Messages:
        return get_messages(self.messages_and_choices)

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        from ._history import legacy_as_chat_completions_history

        return legacy_as_chat_completions_history(self)


class ChatCompletionsHistory(pydantic.BaseModel):
    model: str | None
    messages: Annotated[pydantic.SerializeAsAny[Messages], pydantic.SkipValidation]
    tools: Annotated[pydantic.SerializeAsAny[Tools | None], pydantic.SkipValidation] = (
        None
    )
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        return self


class AnthropicMessagesHistory(pydantic.BaseModel):
    model: str
    system: Annotated[
        pydantic.SerializeAsAny[str | list[AnthropicTextBlockParam] | None],
        pydantic.SkipValidation,
    ] = None
    messages: Annotated[
        pydantic.SerializeAsAny[list[AnthropicMessageParam]], pydantic.SkipValidation
    ]
    tools: Annotated[
        pydantic.SerializeAsAny[list[AnthropicToolParam] | None],
        pydantic.SkipValidation,
    ] = None
    thinking: Annotated[
        pydantic.SerializeAsAny[AnthropicThinkingConfigParam | None],
        pydantic.SkipValidation,
    ] = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        from ._history import anthropic_as_chat_completions_history

        return anthropic_as_chat_completions_history(self)


class ResponsesHistory(pydantic.BaseModel):
    model: str
    input: Annotated[
        pydantic.SerializeAsAny[ResponseInputParam], pydantic.SkipValidation
    ]
    instructions: str | None = None
    tools: Annotated[
        pydantic.SerializeAsAny[list[ResponsesToolParam] | None],
        pydantic.SkipValidation,
    ] = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        from ._history import responses_as_chat_completions_history

        return responses_as_chat_completions_history(self)


class CompletionsHistory(pydantic.BaseModel):
    model: str
    token_ids: list[int]
    sampled_spans: list[tuple[int, int]]

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        raise ValueError("Raw Completions history has no chat-message structure")


TrajectoryHistory: TypeAlias = (
    History
    | ChatCompletionsHistory
    | AnthropicMessagesHistory
    | ResponsesHistory
    | CompletionsHistory
)


class Trajectory(_CompactModel):
    exchanges: TrajectoryExchanges = pydantic.Field(default_factory=TrajectoryExchanges)
    messages_and_choices: MessagesAndChoices = pydantic.Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )
    tools: Tools | None = None
    additional_histories: list[History] = pydantic.Field(
        default_factory=list,
    )
    reward: float = 0.0
    initial_policy_version: int | None = None
    final_policy_version: int | None = None
    metrics: dict[str, float | int | bool] = pydantic.Field(default_factory=dict)
    metadata: dict[str, MetadataValue] = pydantic.Field(default_factory=dict)
    logs: list[str] = pydantic.Field(default_factory=list)
    start_time: datetime = pydantic.Field(default_factory=datetime.now, exclude=True)

    @pydantic.field_serializer("messages_and_choices", when_used="json")
    def serialize_messages_and_choices(self, value: MessagesAndChoices) -> list[Any]:
        return serialize_messages_and_choices(value)

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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        from ._scope import exit_trajectory

        exit_trajectory(self, exc_type, exc_value, traceback)

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

    def chat_completions_history(
        self, *, model: str | None = None
    ) -> ChatCompletionsHistory:
        from ._history import chat_completions_history

        return chat_completions_history(self, model=model)

    def anthropic_messages_history(
        self, *, model: str | None = None
    ) -> AnthropicMessagesHistory:
        from ._history import anthropic_messages_history

        return anthropic_messages_history(self, model=model)

    def responses_history(self, *, model: str | None = None) -> ResponsesHistory:
        from ._history import responses_history

        return responses_history(self, model=model)

    def completions_history(self, *, model: str | None = None) -> CompletionsHistory:
        from ._history import completions_history

        return completions_history(self, model=model)

    def history(self, *, model: str | None = None) -> TrajectoryHistory:
        from ._history import trajectory_history

        return trajectory_history(self, model=model)

    def messages(self) -> Messages:
        from ._history import trajectory_messages

        return trajectory_messages(self)

    def for_logging(self) -> dict[str, object]:
        from ._compat import trajectory_for_logging

        return trajectory_for_logging(self)


class TrajectoryGroup(_CompactModel):
    trajectories: list[Trajectory] = pydantic.Field(default_factory=list)
    exceptions: list[PydanticException] = pydantic.Field(default_factory=list)
    metadata: dict[str, MetadataValue] = pydantic.Field(default_factory=dict)
    metrics: dict[str, float | int | bool] = pydantic.Field(default_factory=dict)
    logs: list[str] = pydantic.Field(default_factory=list)
    _collect_packing_shape: bool = pydantic.PrivateAttr(default=False)
    _packed_group_shape: Any = pydantic.PrivateAttr(default=None)

    @overload
    def __new__(
        cls,
        trajectories: Iterable[Trajectory | BaseException] = (),
        *,
        exceptions: Iterable[BaseException | PydanticException] = (),
        metadata: dict[str, MetadataValue] | None = None,
        metrics: dict[str, float | int | bool] | None = None,
        logs: list[str] | None = None,
    ) -> TrajectoryGroup: ...

    @overload
    @deprecated("Use await art.trajectory_group(...) instead.")
    def __new__(
        cls,
        trajectories: Iterable[Awaitable[Trajectory]],
        *,
        exceptions: Iterable[BaseException | PydanticException] = (),
        metadata: dict[str, MetadataValue] | None = None,
        metrics: dict[str, float | int | bool] | None = None,
        logs: list[str] | None = None,
    ) -> Awaitable[TrajectoryGroup]: ...

    def __new__(
        cls,
        trajectories: Iterable[Trajectory | BaseException | Awaitable[Trajectory]] = (),
        *,
        exceptions: Iterable[BaseException | PydanticException] = (),
        metadata: dict[str, MetadataValue] | None = None,
        metrics: dict[str, float | int | bool] | None = None,
        logs: list[str] | None = None,
    ) -> TrajectoryGroup | Awaitable[TrajectoryGroup]:
        from ._compat import new_trajectory_group

        return new_trajectory_group(
            cls,
            trajectories,
            exceptions=exceptions,
            metadata=metadata,
            metrics=metrics,
            logs=logs,
        )

    def __init__(
        self,
        trajectories: (
            Iterable[Trajectory | BaseException] | Iterable[Awaitable[Trajectory]]
        ) = (),
        *,
        exceptions: Iterable[BaseException | PydanticException] = (),
        metadata: dict[str, MetadataValue] | None = None,
        metrics: dict[str, float | int | bool] | None = None,
        logs: list[str] | None = None,
    ) -> None:
        from ._compat import init_trajectory_group

        init_trajectory_group(
            self,
            trajectories,
            exceptions=exceptions,
            metadata=metadata,
            metrics=metrics,
            logs=logs,
        )

    def __enter__(self) -> TrajectoryGroup:
        from ._scope import enter_trajectory_group

        return enter_trajectory_group(self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        from ._scope import exit_trajectory_group

        exit_trajectory_group(self, exc_type, exc_value, traceback)

    def __copy__(self) -> TrajectoryGroup:
        from ._compat import copy_trajectory_group

        return copy_trajectory_group(self)

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> TrajectoryGroup:
        from ._compat import deepcopy_trajectory_group

        return deepcopy_trajectory_group(self, memo)

    def log(self, message: str) -> None:
        self.logs.append(message)

    # Legacy groups iterate over trajectories rather than Pydantic field pairs.
    def __iter__(self) -> Iterator[Trajectory]:  # ty: ignore[invalid-method-override]
        return iter(self.trajectories)

    def __len__(self) -> int:
        return len(self.trajectories)


class TokenizedTrajectory(pydantic.BaseModel):
    token_ids: list[int]
    logprobs: list[float]
    flags: list[TokenFlag]
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


async def trajectory(coroutine: Coroutine[Any, Any, object]) -> Trajectory:
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
    *,
    base_model: str | None = None,
    model: str | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: Mapping[str, object] | None = None,
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
    *,
    base_model: str | None = None,
    model: str | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: Mapping[str, object] | None = None,
) -> list[TokenizedTrajectory]:
    return [
        tokenize_trajectory(
            item,
            base_model=base_model,
            model=model,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        for item in trajectories
    ]


def tokenize_trajectory_group(
    group: TrajectoryGroup,
    *,
    base_model: str | None = None,
    model: str | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: Mapping[str, object] | None = None,
) -> TokenizedTrajectoryGroup:
    return TokenizedTrajectoryGroup(
        trajectories=tokenize_trajectories(
            group,
            base_model=base_model,
            model=model,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        ),
        underlying=group,
    )


def tokenize_trajectory_groups(
    groups: Iterable[TrajectoryGroup],
    *,
    base_model: str | None = None,
    model: str | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: Mapping[str, object] | None = None,
) -> list[TokenizedTrajectoryGroup]:
    return [
        tokenize_trajectory_group(
            group,
            base_model=base_model,
            model=model,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        for group in groups
    ]


@overload
@deprecated("Use current_trajectory() instead.")
def auto_trajectory(*, required: Literal[True]) -> Trajectory: ...


@overload
@deprecated("Use current_trajectory() instead.")
def auto_trajectory(*, required: Literal[False] = False) -> Trajectory | None: ...


@deprecated("Use current_trajectory() instead.")
def auto_trajectory(*, required: bool = False) -> Trajectory | None:
    return current_trajectory(required=required)


@deprecated("Use trajectory() instead.")
async def capture_auto_trajectory(
    coroutine: Coroutine[Any, Any, object],
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
    "ChatCompletionsHistory",
    "AnthropicMessagesHistory",
    "ResponsesHistory",
    "CompletionsHistory",
    "TrajectoryHistory",
    "Trajectory",
    "TrajectoryGroup",
    "TokenizedTrajectory",
    "TokenizedTrajectoryGroup",
    "TokenFlag",
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
