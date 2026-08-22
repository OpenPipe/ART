from __future__ import annotations

from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Coroutine,
    Iterable,
    Iterator,
    Mapping,
)
from contextlib import AbstractContextManager, asynccontextmanager
from datetime import datetime
from enum import IntFlag
import time
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
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
from typing_extensions import TypedDict, TypeForm, deprecated

if TYPE_CHECKING:
    import torch

    from .tensors import (
        TensorizedHistory,
        TensorizedMultiHistoryTrajectory,
        TensorizedTrajectory,
        TensorizedTrajectoryGroup,
    )

from ..types import Messages, MessagesAndChoices, Tools
from ._serialization import (
    _CompactModel,
    _rebind_history_sources,
    _StringInterningModel,
    _StringPool,
    serialize_chat_completion,
    serialize_history,
    serialize_messages_and_choices,
    validate_history,
)
from ._serialization import (
    _intern_strings as _intern_string_graph,
)

# Deliberately open: Pydantic enforces serializability when callers dump in JSON mode.
MetadataValue = Any
type _Preserved[T] = Annotated[
    pydantic.SerializeAsAny[T],
    pydantic.SkipValidation,
]

type CompactTrajectoryKind = Literal[
    "trajectory",
    "trajectories",
    "trajectory_group",
    "trajectory_groups",
    "tokenized_history",
    "tokenized_histories",
    "tokenized_trajectory",
    "tokenized_trajectories",
    "tokenized_multi_history_trajectory",
    "tokenized_multi_history_trajectories",
    "tokenized_trajectory_group",
    "tokenized_trajectory_groups",
    "tensorized_history",
    "tensorized_histories",
    "tensorized_trajectory",
    "tensorized_trajectories",
    "tensorized_multi_history_trajectory",
    "tensorized_multi_history_trajectories",
    "tensorized_trajectory_group",
    "tensorized_trajectory_groups",
]


class CompactTrajectoryPayload(TypedDict):
    """Versioned, JSON-compatible compact trajectory envelope."""

    format: Literal["art.trajectories"]
    version: Literal[1]
    kind: CompactTrajectoryKind
    # Maps readable references such as "$0" to literal string contents.
    strings: dict[str, str]
    # Ordinary Pydantic JSON data, with profitable strings replaced by references.
    data: pydantic.JsonValue


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
    request: _Preserved[ChatCompletionsRequest]
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
    request: _Preserved[CompletionsRequest]
    response: Completion
    start_time: datetime
    end_time: datetime

    @pydantic.computed_field
    @property
    def model(self) -> str | None:
        requested = self.request.get("model")
        return requested if isinstance(requested, str) else self.response.model


class ResponsesExchange(pydantic.BaseModel):
    request: _Preserved[ResponsesRequest]
    response: Response
    start_time: datetime
    end_time: datetime

    @pydantic.computed_field
    @property
    def model(self) -> str | None:
        requested = self.request.get("model")
        return requested if isinstance(requested, str) else self.response.model


class MessagesExchange(pydantic.BaseModel):
    request: _Preserved[MessagesRequest]
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


class LegacyHistory(_StringInterningModel):
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

    def tensorize(
        self,
        *,
        model: str,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ) -> TensorizedHistory:
        return self.tokenize(
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        ).tensorize(device=device)


class History(_StringInterningModel):
    """Mutable, protocol-native view of one tokenizable sequence."""

    model_config = pydantic.ConfigDict(extra="forbid")

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

    def tensorize(
        self,
        *,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ) -> TensorizedHistory:
        return self.tokenize(
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        ).tensorize(device=device)


class _HistorySource(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class ChatCompletionsMessageSource(_HistorySource):
    exchange: ChatCompletionsExchange | MessagesExchange | ResponsesExchange
    request_index: pydantic.StrictInt | None = None
    choice_index: pydantic.StrictInt | None = None
    output_indices: tuple[pydantic.StrictInt, ...] | None = None
    generation_index: pydantic.StrictInt | None = None


class AnthropicMessageSource(_HistorySource):
    exchange: MessagesExchange
    request_index: pydantic.StrictInt | None = None


class ResponsesItemSource(_HistorySource):
    exchange: ResponsesExchange
    request_index: pydantic.StrictInt | None = None
    output_index: pydantic.StrictInt | None = None
    generation_index: pydantic.StrictInt | None = None


class CompletionsSource(_HistorySource):
    exchange: CompletionsExchange
    prompt_index: pydantic.StrictInt
    choice_index: pydantic.StrictInt | None = None


class CompletionsTokenSourceSpan(_HistorySource):
    start: pydantic.StrictInt
    end: pydantic.StrictInt
    source: CompletionsSource | None


class CompletionsStringSourceSpan(_HistorySource):
    start: pydantic.StrictInt
    end: pydantic.StrictInt
    source: CompletionsSource | None


class ChatCompletionsHistory(History):
    model: str | None
    messages: _Preserved[Messages]
    message_sources: list[ChatCompletionsMessageSource | None]
    tools: _Preserved[Tools | None] = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        return self


class AnthropicMessagesHistory(History):
    model: str
    messages: _Preserved[list[AnthropicMessageParam]]
    message_sources: list[AnthropicMessageSource | None]
    system: _Preserved[str | list[AnthropicTextBlockParam] | None] = None
    system_source: MessagesExchange | None = None
    tools: _Preserved[list[AnthropicToolParam] | None] = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        from ._history import anthropic_as_chat_completions_history

        return anthropic_as_chat_completions_history(self)


class ResponsesHistory(History):
    model: str
    input: _Preserved[ResponseInputParam]
    input_sources: list[ResponsesItemSource | None]
    instructions: str | None = None
    instructions_source: ResponsesExchange | None = None
    tools: _Preserved[list[ResponsesToolParam] | None] = None
    conversation: _Preserved[ResponsesConversation | None] = None
    previous_response_id: str | None = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        from ._history import responses_as_chat_completions_history

        return responses_as_chat_completions_history(self)


class CompletionsTokenHistory(History):
    model: str
    prompt: list[int]
    prompt_sources: list[CompletionsTokenSourceSpan]
    sampled_spans: list[tuple[int, int]]

    def as_chat_completions_history(self) -> ChatCompletionsHistory:
        raise ValueError("Raw Completions history has no chat-message structure")


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

    def _intern_strings(self, pool: _StringPool | None = None) -> None:
        _intern_string_graph(self, pool)

    def compact_dump(self) -> CompactTrajectoryPayload:
        """Return the explicit string-table representation of this trajectory."""

        from ._compact import dump_trajectory

        return dump_trajectory(self)

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

    # Model selectors accept exact identities or shell patterns. Reconciliation
    # opts into merging text-equivalent histories with different served token IDs.
    def chat_completions_history(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> ChatCompletionsHistory:
        from ._history import chat_completions_history

        return chat_completions_history(
            self, model, reconcile_text_equivalent_tokenizations
        )

    def chat_completions_histories(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> list[ChatCompletionsHistory]:
        from ._history import chat_completions_histories

        return chat_completions_histories(
            self, model, reconcile_text_equivalent_tokenizations
        )

    def anthropic_messages_history(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> AnthropicMessagesHistory:
        from ._history import anthropic_messages_history

        return anthropic_messages_history(
            self, model, reconcile_text_equivalent_tokenizations
        )

    def anthropic_messages_histories(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> list[AnthropicMessagesHistory]:
        from ._history import anthropic_messages_histories

        return anthropic_messages_histories(
            self, model, reconcile_text_equivalent_tokenizations
        )

    def responses_history(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> ResponsesHistory:
        from ._history import responses_history

        return responses_history(self, model, reconcile_text_equivalent_tokenizations)

    def responses_histories(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> list[ResponsesHistory]:
        from ._history import responses_histories

        return responses_histories(self, model, reconcile_text_equivalent_tokenizations)

    def completions_token_history(
        self,
        *,
        model: str | None = None,
    ) -> CompletionsTokenHistory:
        from ._history import completions_token_history

        return completions_token_history(self, model)

    def completions_token_histories(
        self,
        *,
        model: str | None = None,
    ) -> list[CompletionsTokenHistory]:
        from ._history import completions_token_histories

        return completions_token_histories(self, model)

    def completions_string_history(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> CompletionsStringHistory:
        from ._history import completions_string_history

        return completions_string_history(
            self, model, reconcile_text_equivalent_tokenizations
        )

    def completions_string_histories(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> list[CompletionsStringHistory]:
        from ._history import completions_string_histories

        return completions_string_histories(
            self, model, reconcile_text_equivalent_tokenizations
        )

    def history(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> TrajectoryHistory:
        from ._history import trajectory_history

        return trajectory_history(self, model, reconcile_text_equivalent_tokenizations)

    def histories(
        self,
        *,
        model: str | None = None,
        reconcile_text_equivalent_tokenizations: bool = False,
    ) -> list[TrajectoryHistory]:
        from ._history import trajectory_histories

        return trajectory_histories(
            self, model, reconcile_text_equivalent_tokenizations
        )

    @overload
    def tokenize(
        self,
        *,
        multi_history: Literal[False] = False,
        reconcile_text_equivalent_tokenizations: bool = False,
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
        reconcile_text_equivalent_tokenizations: bool = False,
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
        reconcile_text_equivalent_tokenizations: bool = False,
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
            reconcile_text_equivalent_tokenizations=reconcile_text_equivalent_tokenizations,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )

    @overload
    def tensorize(
        self,
        *,
        multi_history: Literal[False] = False,
        reconcile_text_equivalent_tokenizations: bool = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ) -> TensorizedTrajectory: ...

    @overload
    def tensorize(
        self,
        *,
        multi_history: Literal[True],
        reconcile_text_equivalent_tokenizations: bool = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ) -> TensorizedMultiHistoryTrajectory: ...

    def tensorize(
        self,
        *,
        multi_history: bool = False,
        reconcile_text_equivalent_tokenizations: bool = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ) -> TensorizedTrajectory | TensorizedMultiHistoryTrajectory:
        return self.tokenize(
            multi_history=multi_history,
            reconcile_text_equivalent_tokenizations=reconcile_text_equivalent_tokenizations,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        ).tensorize(device=device)

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

    def _intern_strings(self, pool: _StringPool | None = None) -> None:
        _intern_string_graph(self, pool)

    def compact_dump(self) -> CompactTrajectoryPayload:
        """Return the explicit string-table representation of this group."""

        from ._compact import dump_trajectory_group

        return dump_trajectory_group(self)

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
        reconcile_text_equivalent_tokenizations: bool = False,
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
        reconcile_text_equivalent_tokenizations: bool = False,
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
        reconcile_text_equivalent_tokenizations: bool = False,
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
            reconcile_text_equivalent_tokenizations=reconcile_text_equivalent_tokenizations,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )

    @overload
    def tensorize(
        self,
        *,
        multi_history: Literal[False] = False,
        reconcile_text_equivalent_tokenizations: bool = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ) -> TensorizedTrajectoryGroup[TensorizedTrajectory]: ...

    @overload
    def tensorize(
        self,
        *,
        multi_history: Literal[True],
        reconcile_text_equivalent_tokenizations: bool = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ) -> TensorizedTrajectoryGroup[TensorizedMultiHistoryTrajectory]: ...

    def tensorize(
        self,
        *,
        multi_history: bool = False,
        reconcile_text_equivalent_tokenizations: bool = False,
        model: str | None = None,
        base_model: str | None = None,
        tokenizer: Tokenizer | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        device: torch.device | str | None = None,
    ) -> (
        TensorizedTrajectoryGroup[TensorizedTrajectory]
        | TensorizedTrajectoryGroup[TensorizedMultiHistoryTrajectory]
    ):
        return self.tokenize(
            multi_history=multi_history,
            reconcile_text_equivalent_tokenizations=reconcile_text_equivalent_tokenizations,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        ).tensorize(device=device)


class TokenizedHistory(_StringInterningModel):
    model_config = pydantic.ConfigDict(ser_json_inf_nan="strings")

    history: TrajectoryHistory
    model: str
    tokens: list[int]
    logprobs: list[float]
    flags: list[TokenFlag]

    @pydantic.field_serializer("history")
    def serialize_source_history(
        self, value: TrajectoryHistory
    ) -> dict[str, pydantic.JsonValue]:
        return serialize_history(value)

    @pydantic.field_validator("history", mode="before")
    @classmethod
    def validate_source_history(cls, value: object) -> object:
        return validate_history(value)

    @pydantic.model_validator(mode="after")
    def validate_tokenwise_lengths(self) -> TokenizedHistory:
        if not (len(self.tokens) == len(self.logprobs) == len(self.flags)):
            raise ValueError("Tokenized history fields differ in length")
        return self

    def tensorize(
        self, *, device: torch.device | str | None = None
    ) -> TensorizedHistory:
        return _load_tensors().tensorize_history(self, device=device)

    def compact_dump(self) -> CompactTrajectoryPayload:
        """Return a compact representation retaining the source history."""

        from ._compact import dump_tokenized_history

        return dump_tokenized_history(self)


class TokenizedTrajectory(TokenizedHistory):
    trajectory: Trajectory

    @property
    def reward(self) -> float:
        return self.trajectory.reward

    @reward.setter
    def reward(self, value: float) -> None:
        self.trajectory.reward = value

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory.metrics

    @metrics.setter
    def metrics(self, value: dict[str, float | int | bool]) -> None:
        self.trajectory.metrics = value

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        return self.trajectory.metadata

    @metadata.setter
    def metadata(self, value: dict[str, MetadataValue]) -> None:
        self.trajectory.metadata = value

    @pydantic.model_validator(mode="after")
    def _bind_source_trajectory(self) -> TokenizedTrajectory:
        _rebind_history_sources(self.history, self.trajectory)
        return self

    def compact_dump(self) -> CompactTrajectoryPayload:
        """Return a compact representation retaining the source trajectory."""

        from ._compact import dump_tokenized_trajectory

        return dump_tokenized_trajectory(self)

    def tensorize(
        self, *, device: torch.device | str | None = None
    ) -> TensorizedTrajectory:
        return _load_tensors().tensorize_trajectory(self, device=device)


class TokenizedMultiHistoryTrajectory(_StringInterningModel):
    model_config = pydantic.ConfigDict(ser_json_inf_nan="strings")

    trajectory: Trajectory
    histories: list[TokenizedHistory]

    @property
    def reward(self) -> float:
        return self.trajectory.reward

    @reward.setter
    def reward(self, value: float) -> None:
        self.trajectory.reward = value

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory.metrics

    @metrics.setter
    def metrics(self, value: dict[str, float | int | bool]) -> None:
        self.trajectory.metrics = value

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        return self.trajectory.metadata

    @metadata.setter
    def metadata(self, value: dict[str, MetadataValue]) -> None:
        self.trajectory.metadata = value

    @pydantic.model_validator(mode="after")
    def _intern_source_graph(self) -> TokenizedMultiHistoryTrajectory:
        for history in self.histories:
            _rebind_history_sources(history.history, self.trajectory)
        return self

    def compact_dump(self) -> CompactTrajectoryPayload:
        """Return a compact representation retaining the source trajectory."""

        from ._compact import dump_tokenized_multi_history_trajectory

        return dump_tokenized_multi_history_trajectory(self)

    def tensorize(
        self, *, device: torch.device | str | None = None
    ) -> TensorizedMultiHistoryTrajectory:
        return _load_tensors().tensorize_multi_history_trajectory(self, device=device)


TokenizedTrajectoryT = TypeVar(
    "TokenizedTrajectoryT", TokenizedTrajectory, TokenizedMultiHistoryTrajectory
)


class TokenizedTrajectoryGroup(_StringInterningModel, Generic[TokenizedTrajectoryT]):
    model_config = pydantic.ConfigDict(ser_json_inf_nan="strings")

    trajectory_group: TrajectoryGroup
    trajectories: list[TokenizedTrajectoryT]

    @property
    def metrics(self) -> dict[str, float | int | bool]:
        return self.trajectory_group.metrics

    @metrics.setter
    def metrics(self, value: dict[str, float | int | bool]) -> None:
        self.trajectory_group.metrics = value

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        return self.trajectory_group.metadata

    @metadata.setter
    def metadata(self, value: dict[str, MetadataValue]) -> None:
        self.trajectory_group.metadata = value

    @pydantic.model_validator(mode="after")
    def _intern_source_graph(self) -> TokenizedTrajectoryGroup[TokenizedTrajectoryT]:
        if len(self.trajectories) != len(self.trajectory_group.trajectories):
            raise ValueError("Tokenized group differs in length from its source group")
        for tokenized, trajectory in zip(
            self.trajectories, self.trajectory_group.trajectories, strict=True
        ):
            if (
                tokenized.trajectory is not trajectory
                and tokenized.trajectory.model_dump() != trajectory.model_dump()
            ):
                raise ValueError("Tokenized trajectory does not match its source group")
            tokenized.trajectory = trajectory
            if isinstance(tokenized, TokenizedTrajectory):
                _rebind_history_sources(tokenized.history, trajectory)
            else:
                for history in tokenized.histories:
                    _rebind_history_sources(history.history, trajectory)
        return self

    def compact_dump(self) -> CompactTrajectoryPayload:
        """Return a compact representation retaining the source group."""

        from ._compact import dump_tokenized_trajectory_group

        return dump_tokenized_trajectory_group(self)

    def tensorize(
        self, *, device: torch.device | str | None = None
    ) -> (
        TensorizedTrajectoryGroup[TensorizedTrajectory]
        | TensorizedTrajectoryGroup[TensorizedMultiHistoryTrajectory]
    ):
        return _load_tensors().tensorize_group(self, device=device)


CompactDumpable: TypeAlias = Union[
    Trajectory,
    TrajectoryGroup,
    TokenizedHistory,
    TokenizedTrajectory,
    TokenizedMultiHistoryTrajectory,
    TokenizedTrajectoryGroup[TokenizedTrajectory],
    TokenizedTrajectoryGroup[TokenizedMultiHistoryTrajectory],
    "TensorizedHistory",
    "TensorizedTrajectory",
    "TensorizedMultiHistoryTrajectory",
    "TensorizedTrajectoryGroup[TensorizedTrajectory]",
    "TensorizedTrajectoryGroup[TensorizedMultiHistoryTrajectory]",
]
_CompactValidated: TypeAlias = Union[CompactDumpable, list[CompactDumpable]]


def compact_dump(
    value: CompactDumpable | Iterable[CompactDumpable],
) -> CompactTrajectoryPayload:
    """Compact one supported value or a homogeneous iterable of values."""

    from ._compact import dump

    return dump(value)


@overload
def compact_validate[T](
    payload: Mapping[str, object],
    *,
    type: TypeForm[T],
    device: torch.device | str | None = None,
) -> T: ...


@overload
def compact_validate(
    payload: Mapping[str, object],
    *,
    type: None = None,
    device: torch.device | str | None = None,
) -> _CompactValidated: ...


def compact_validate[T](
    payload: Mapping[str, object],
    *,
    type: TypeForm[T] | None = None,
    device: torch.device | str | None = None,
) -> T | _CompactValidated:
    """Validate a compact value, inferring or checking its requested type."""

    from ._compact import validate

    return validate(payload, type=type, device=device)


@overload
def current_trajectory(*, require: Literal[True]) -> Trajectory: ...


@overload
def current_trajectory(*, require: Literal[False] = False) -> Trajectory | None: ...


def current_trajectory(*, require: bool = False) -> Trajectory | None:
    from ._scope import get_current_trajectory

    return get_current_trajectory(required=require)


def no_capture() -> AbstractContextManager[None]:
    """Hide enclosing trajectory capture while allowing new nested scopes."""

    from ._scope import no_capture as capture_barrier

    return capture_barrier()


async def trajectory(coroutine: Coroutine[Any, Any, object]) -> Trajectory:
    from ._scope import capture_trajectory

    return await capture_trajectory(coroutine)


async def trajectory_group(
    trajectories: Iterable[Trajectory | BaseException | Awaitable[Trajectory]],
    *,
    exceptions: Iterable[BaseException | PydanticException] = (),
    metadata: dict[str, MetadataValue] | None = None,
    metrics: dict[str, float | int | bool] | None = None,
    logs: list[str] | None = None,
    return_exceptions: bool = False,
) -> TrajectoryGroup:
    from ._scope import capture_trajectory_group

    return await capture_trajectory_group(
        trajectories,
        exceptions=exceptions,
        metadata=metadata,
        metrics=metrics,
        logs=logs,
        return_exceptions=return_exceptions,
    )


def get_messages(messages_and_choices: MessagesAndChoices) -> Messages:
    from ._compat import messages_from_legacy_history

    return messages_from_legacy_history(messages_and_choices)


_TENSOR_EXPORTS = frozenset(
    {
        "TensorizedHistory",
        "TensorizedMultiHistoryTrajectory",
        "TensorizedTrajectory",
        "TensorizedTrajectoryGroup",
    }
)


def _load_tensors():
    from importlib import import_module

    try:
        return import_module(f"{__name__}.tensors")
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise ModuleNotFoundError(
                "Tensorized trajectories require openpipe-art[tensors]"
            ) from error
        raise


def __getattr__(name: str) -> object:
    if name in _TENSOR_EXPORTS:
        return getattr(_load_tensors(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _TENSOR_EXPORTS)


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
    "TensorizedHistory",
    "TensorizedMultiHistoryTrajectory",
    "TensorizedTrajectory",
    "TensorizedTrajectoryGroup",
    "Tokenizer",
    "TokenFlag",
    "MetadataValue",
    "CompactDumpable",
    "CompactTrajectoryKind",
    "CompactTrajectoryPayload",
    "compact_dump",
    "compact_validate",
    "current_trajectory",
    "no_capture",
    "trajectory",
    "trajectory_group",
    "get_messages",
]
