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
from dataclasses import dataclass
from datetime import datetime
from enum import IntFlag
import time
from types import TracebackType
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)

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
from openai.types.responses.response_create_params import (
    Conversation as ResponsesConversation,
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


class Tokenizer(Protocol):
    """Minimal tokenizer surface used by trajectory tokenization."""

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> object: ...

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: object,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
        **kwargs: object,
    ) -> object: ...


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
    suffix: str


class ResponsesRequest(TypedDict, total=False, extra_items=Any):
    """The JSON body sent to an OpenAI-compatible Responses endpoint."""

    model: str
    input: str | ResponseInputParam
    instructions: str
    previous_response_id: str
    conversation: ResponsesConversation
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


class LegacyHistory(pydantic.BaseModel):
    messages_and_choices: MessagesAndChoices
    tools: Tools | None = None

    @pydantic.field_serializer("messages_and_choices", when_used="json")
    def serialize_messages_and_choices(self, value: MessagesAndChoices) -> list[Any]:
        return serialize_messages_and_choices(value)

    def messages(self) -> Messages:
        return get_messages(self.messages_and_choices)

    def as_chat_completions_history(
        self, *, model: str | None = None
    ) -> ChatCompletionsHistory:
        from ._history import legacy_as_chat_completions_history

        return legacy_as_chat_completions_history(self, model=model)

    def tokenize(
        self,
        *,
        model: str,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> TokenizedHistory:
        from ._tokenize import tokenize_history

        return tokenize_history(
            self,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )


class History:
    """Mutable, protocol-native view of one tokenizable sequence."""

    model: str | None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        raise ValueError(
            f"{type(self).__name__} cannot be represented as Chat Completions"
        )

    def tokenize(
        self,
        *,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> TokenizedHistory:
        from ._tokenize import tokenize_history

        return tokenize_history(
            self,
            model=self.model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )


@dataclass(frozen=True, slots=True)
class ChatCompletionsMessageSource:
    exchange: ChatCompletionsExchange | MessagesExchange | ResponsesExchange
    request_index: int | None = None
    choice_index: int | None = None
    output_index: int | None = None
    generation_index: int | None = None


@dataclass(frozen=True, slots=True)
class AnthropicMessageSource:
    exchange: MessagesExchange
    request_index: int | None = None


@dataclass(frozen=True, slots=True)
class ResponsesItemSource:
    exchange: ResponsesExchange
    request_index: int | None = None
    output_index: int | None = None
    generation_index: int | None = None


@dataclass(frozen=True, slots=True)
class CompletionsSource:
    exchange: CompletionsExchange
    prompt_index: int
    choice_index: int | None = None


@dataclass(frozen=True, slots=True)
class CompletionsTokenSourceSpan:
    start: int
    end: int
    source: CompletionsSource | None


@dataclass(frozen=True, slots=True)
class CompletionsStringSourceSpan:
    start: int
    end: int
    source: CompletionsSource | None


@dataclass
class ChatCompletionsHistory(History):
    model: str | None
    messages: Messages
    message_sources: list[ChatCompletionsMessageSource | None]
    tools: Tools | None = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        return self


@dataclass
class AnthropicMessagesHistory(History):
    model: str
    messages: list[AnthropicMessageParam]
    message_sources: list[AnthropicMessageSource | None]
    system: str | list[AnthropicTextBlockParam] | None = None
    system_source: MessagesExchange | None = None
    tools: list[AnthropicToolParam] | None = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        from ._history import anthropic_as_chat_completions_history

        return anthropic_as_chat_completions_history(self)


@dataclass
class ResponsesHistory(History):
    model: str
    input: ResponseInputParam
    input_sources: list[ResponsesItemSource | None]
    instructions: str | None = None
    instructions_source: ResponsesExchange | None = None
    tools: list[ResponsesToolParam] | None = None
    conversation: ResponsesConversation | None = None
    previous_response_id: str | None = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        from ._history import responses_as_chat_completions_history

        return responses_as_chat_completions_history(self)


@dataclass
class CompletionsTokenHistory(History):
    model: str
    prompt: list[int]
    prompt_sources: list[CompletionsTokenSourceSpan]
    sampled_spans: list[tuple[int, int]]

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        raise ValueError("Raw Completions history has no chat-message structure")


@dataclass
class CompletionsStringHistory(History):
    model: str
    prompt: str
    prompt_sources: list[CompletionsStringSourceSpan]
    sampled_spans: list[tuple[int, int]]

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        raise ValueError("Raw Completions history has no chat-message structure")


TrajectoryHistory: TypeAlias = (
    LegacyHistory
    | ChatCompletionsHistory
    | AnthropicMessagesHistory
    | ResponsesHistory
    | CompletionsTokenHistory
    | CompletionsStringHistory
)


class Trajectory(_CompactModel):
    exchanges: TrajectoryExchanges = pydantic.Field(default_factory=TrajectoryExchanges)
    messages_and_choices: MessagesAndChoices = pydantic.Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )
    tools: Tools | None = None
    additional_histories: list[LegacyHistory] = pydantic.Field(
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

    def chat_completions_histories(
        self, *, model: str | None = None
    ) -> list[ChatCompletionsHistory]:
        from ._history import chat_completions_histories

        return chat_completions_histories(self, model=model)

    def anthropic_messages_history(
        self, *, model: str | None = None
    ) -> AnthropicMessagesHistory:
        from ._history import anthropic_messages_history

        return anthropic_messages_history(self, model=model)

    def anthropic_messages_histories(
        self, *, model: str | None = None
    ) -> list[AnthropicMessagesHistory]:
        from ._history import anthropic_messages_histories

        return anthropic_messages_histories(self, model=model)

    def responses_history(self, *, model: str | None = None) -> ResponsesHistory:
        from ._history import responses_history

        return responses_history(self, model=model)

    def responses_histories(
        self, *, model: str | None = None
    ) -> list[ResponsesHistory]:
        from ._history import responses_histories

        return responses_histories(self, model=model)

    def completions_token_history(
        self, *, model: str | None = None
    ) -> CompletionsTokenHistory:
        from ._history import completions_token_history

        return completions_token_history(self, model=model)

    def completions_token_histories(
        self, *, model: str | None = None
    ) -> list[CompletionsTokenHistory]:
        from ._history import completions_token_histories

        return completions_token_histories(self, model=model)

    def completions_string_history(
        self, *, model: str | None = None
    ) -> CompletionsStringHistory:
        from ._history import completions_string_history

        return completions_string_history(self, model=model)

    def completions_string_histories(
        self, *, model: str | None = None
    ) -> list[CompletionsStringHistory]:
        from ._history import completions_string_histories

        return completions_string_histories(self, model=model)

    def history(self, *, model: str | None = None) -> TrajectoryHistory:
        from ._history import trajectory_history

        return trajectory_history(self, model=model)

    def histories(self, *, model: str | None = None) -> list[TrajectoryHistory]:
        from ._history import trajectory_histories

        return trajectory_histories(self, model=model)

    @overload
    def tokenize(
        self,
        *,
        multi_history: Literal[False] = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> TokenizedTrajectory: ...

    @overload
    def tokenize(
        self,
        *,
        multi_history: Literal[True],
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> TokenizedMultiHistoryTrajectory: ...

    def tokenize(
        self,
        *,
        multi_history: bool = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> TokenizedTrajectory | TokenizedMultiHistoryTrajectory:
        from ._tokenize import tokenize_trajectory

        return tokenize_trajectory(
            self,
            multi_history=multi_history,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )

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

    @overload
    def tokenize(
        self,
        *,
        multi_history: Literal[False] = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> TokenizedTrajectoryGroup[TokenizedTrajectory]: ...

    @overload
    def tokenize(
        self,
        *,
        multi_history: Literal[True],
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> TokenizedTrajectoryGroup[TokenizedMultiHistoryTrajectory]: ...

    def tokenize(
        self,
        *,
        multi_history: bool = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
    ) -> (
        TokenizedTrajectoryGroup[TokenizedTrajectory]
        | TokenizedTrajectoryGroup[TokenizedMultiHistoryTrajectory]
    ):
        from ._tokenize import tokenize_group

        return tokenize_group(
            self,
            multi_history=multi_history,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )


class TokenizedHistory(pydantic.BaseModel):
    model: str
    token_ids: list[int]
    logprobs: list[float]
    flags: list[TokenFlag]

    @pydantic.model_validator(mode="after")
    def validate_tokenwise_lengths(self) -> TokenizedHistory:
        if not (len(self.token_ids) == len(self.logprobs) == len(self.flags)):
            raise ValueError("Tokenized history fields differ in length")
        return self


class TokenizedTrajectory(TokenizedHistory):
    reward: float
    metrics: dict[str, float | int | bool]
    metadata: dict[str, MetadataValue]


class TokenizedMultiHistoryTrajectory(pydantic.BaseModel):
    histories: list[TokenizedHistory]
    reward: float
    metrics: dict[str, float | int | bool]
    metadata: dict[str, MetadataValue]


TokenizedTrajectoryT = TypeVar(
    "TokenizedTrajectoryT", TokenizedTrajectory, TokenizedMultiHistoryTrajectory
)


class TokenizedTrajectoryGroup(pydantic.BaseModel, Generic[TokenizedTrajectoryT]):
    trajectories: list[TokenizedTrajectoryT]
    metrics: dict[str, float | int | bool]
    metadata: dict[str, MetadataValue]


@overload
def current_trajectory(*, require: Literal[True]) -> Trajectory: ...


@overload
def current_trajectory(*, require: Literal[False] = False) -> Trajectory | None: ...


def current_trajectory(*, require: bool = False) -> Trajectory | None:
    from ._scope import get_current_trajectory

    return get_current_trajectory(required=require)


@overload
def current_trajectory_group(*, require: Literal[True]) -> TrajectoryGroup: ...


@overload
def current_trajectory_group(
    *, require: Literal[False] = False
) -> TrajectoryGroup | None: ...


def current_trajectory_group(*, require: bool = False) -> TrajectoryGroup | None:
    from ._scope import get_current_trajectory_group

    return get_current_trajectory_group(required=require)


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


@overload
@deprecated("Use current_trajectory() instead.")
def auto_trajectory(*, require: Literal[True]) -> Trajectory: ...


@overload
@deprecated("Use current_trajectory() instead.")
def auto_trajectory(*, require: Literal[False] = False) -> Trajectory | None: ...


@deprecated("Use current_trajectory() instead.")
def auto_trajectory(*, require: bool = False) -> Trajectory | None:
    return current_trajectory(require=require)


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
    "LegacyHistory",
    "ChatCompletionsMessageSource",
    "AnthropicMessageSource",
    "ResponsesItemSource",
    "CompletionsSource",
    "CompletionsTokenSourceSpan",
    "CompletionsStringSourceSpan",
    "ChatCompletionsHistory",
    "AnthropicMessagesHistory",
    "ResponsesHistory",
    "CompletionsTokenHistory",
    "CompletionsStringHistory",
    "TrajectoryHistory",
    "Trajectory",
    "TrajectoryGroup",
    "TokenizedTrajectory",
    "TokenizedHistory",
    "TokenizedMultiHistoryTrajectory",
    "TokenizedTrajectoryGroup",
    "Tokenizer",
    "TokenFlag",
    "MetadataValue",
    "current_trajectory",
    "current_trajectory_group",
    "trajectory",
    "trajectory_group",
    "auto_trajectory",
    "capture_auto_trajectory",
    "get_messages",
]
