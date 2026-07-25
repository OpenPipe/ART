from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import copy
from dataclasses import dataclass, replace
from datetime import datetime
from fnmatch import fnmatchcase
import json
from typing import Generic, Protocol, TypeVar, cast

from anthropic.types import (
    MessageParam as AnthropicMessageParam,
)
from anthropic.types import (
    TextBlockParam as AnthropicTextBlockParam,
)
from anthropic.types import (
    ToolUnionParam as AnthropicToolParam,
)
from openai.types import CompletionChoice
from openai.types.responses import ResponseInputParam
from openai.types.responses import ToolParam as ResponsesToolParam
from openai.types.responses.response_create_params import (
    Conversation as ResponsesConversation,
)
from openai.types.responses.response_input_param import ResponseInputItemParam
import pydantic

from ..types import Message, Messages, Tools
from . import (
    AnthropicMessagesHistory,
    AnthropicMessageSource,
    ChatCompletionsExchange,
    ChatCompletionsHistory,
    ChatCompletionsMessageSource,
    CompletionsExchange,
    CompletionsSource,
    CompletionsStringHistory,
    CompletionsStringSourceSpan,
    CompletionsTokenHistory,
    CompletionsTokenSourceSpan,
    LegacyHistory,
    MessagesExchange,
    ResponsesExchange,
    ResponsesHistory,
    ResponsesItemSource,
    Trajectory,
    TrajectoryHistory,
)


class _ModelledExchange(Protocol):
    start_time: datetime
    end_time: datetime

    @property
    def model(self) -> str | None: ...


class _Indexed(Protocol):
    index: int


_ExchangeT = TypeVar("_ExchangeT", bound=_ModelledExchange)
_IndexedT = TypeVar("_IndexedT", bound=_Indexed)
_ItemT = TypeVar("_ItemT")
_SourceT = TypeVar("_SourceT")
_ContextT = TypeVar("_ContextT")
_MESSAGES = pydantic.TypeAdapter(Messages)
_MESSAGE = pydantic.TypeAdapter(Message)
_TOOLS = pydantic.TypeAdapter(Tools | None)
_CHAT_KWARGS = pydantic.TypeAdapter(dict[str, object] | None)
_ANTHROPIC_SYSTEM = pydantic.TypeAdapter(str | list[AnthropicTextBlockParam] | None)
_ANTHROPIC_TOOLS = pydantic.TypeAdapter(list[AnthropicToolParam] | None)
_RESPONSE_TOOLS = pydantic.TypeAdapter(list[ResponsesToolParam] | None)
_RESPONSE_CONVERSATION = pydantic.TypeAdapter(ResponsesConversation | None)


@dataclass(frozen=True)
class _ChatContext:
    tools: Tools | None
    template: str | None
    kwargs: dict[str, object] | None


@dataclass(frozen=True)
class _AnthropicContext:
    system: str | list[AnthropicTextBlockParam] | None
    tools: list[AnthropicToolParam] | None
    template: str | None
    kwargs: dict[str, object] | None


@dataclass(frozen=True)
class _ResponsesContext:
    instructions: str | None
    tools: list[ResponsesToolParam] | None
    conversation: ResponsesConversation | None
    previous_response_id: str | None
    template: str | None
    kwargs: dict[str, object] | None


type _ResponsesRecord = tuple[
    ResponseInputParam,
    list[ResponsesItemSource | None],
    ResponsesConversation | None,
    str | None,
]


@dataclass(frozen=True)
class _ResponsesGeneration:
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    output_indices: list[int]


@dataclass
class _Branch(Generic[_ItemT, _SourceT, _ContextT]):
    items: list[_ItemT]
    sources: list[_SourceT | None]
    context: _ContextT
    order: tuple[int, ...]
    first_time: datetime
    context_source: _ModelledExchange | None
    lineage_keys: list[object] | None = None


def _selected_models(
    exchanges: Sequence[_ExchangeT], model: str | None, protocol: str
) -> list[tuple[str, list[_ExchangeT]]]:
    has_exact_match = model is not None and any(
        exchange.model == model for exchange in exchanges
    )
    selected = [
        exchange
        for exchange in exchanges
        if model is None
        or (
            exchange.model == model
            if has_exact_match
            else _model_matches(exchange.model, model)
        )
    ]
    if not selected:
        suffix = f" for model {model!r}" if model is not None else ""
        raise ValueError(f"Trajectory contains no {protocol} exchanges{suffix}")
    if any(exchange.model is None for exchange in selected):
        raise ValueError(f"Every {protocol} exchange must identify its model")
    grouped: dict[str, list[_ExchangeT]] = {}
    for exchange in sorted(selected, key=lambda item: (item.start_time, item.end_time)):
        if exchange.model is None:
            raise AssertionError("model identity was checked above")
        grouped.setdefault(exchange.model, []).append(exchange)
    return list(grouped.items())


def _model_matches(candidate: str | None, pattern: str) -> bool:
    return candidate is not None and (
        candidate == pattern or fnmatchcase(candidate, pattern)
    )


def _one(history: Sequence[_ItemT], protocol: str) -> _ItemT:
    if len(history) != 1:
        raise ValueError(
            f"{protocol} requires exactly one history; found {len(history)}"
        )
    return history[0]


def _ordered_choices(choices: Sequence[_IndexedT], *, protocol: str) -> list[_IndexedT]:
    if not choices:
        raise ValueError(f"{protocol} response contains no choices")
    indices = [choice.index for choice in choices]
    if any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in indices
    ):
        raise ValueError(f"{protocol} choice indices must be non-negative integers")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{protocol} response contains duplicate choice indices")
    return sorted(choices, key=lambda choice: choice.index)


def _is_prefix(prefix: Sequence[object], value: Sequence[object]) -> bool:
    return len(prefix) <= len(value) and all(
        left == right for left, right in zip(prefix, value[: len(prefix)], strict=True)
    )


def _lineage_prompt_sources(
    branches: Sequence[_Branch[_ItemT, _SourceT, _ContextT]],
    *,
    prompt: Sequence[_ItemT],
    defaults: Sequence[_SourceT | None],
    equivalent: Callable[[_ItemT, _ItemT], bool],
    prompt_keys: Sequence[object] | None = None,
) -> list[_SourceT | None]:
    candidates = [
        branch
        for branch in branches
        if len(branch.items) < len(prompt)
        and (
            list(prompt_keys[: len(branch.items)]) == branch.lineage_keys
            if prompt_keys is not None and branch.lineage_keys is not None
            else all(
                equivalent(existing, current)
                for existing, current in zip(branch.items, prompt, strict=False)
            )
        )
    ]
    if not candidates:
        return list(defaults)
    parent = min(candidates, key=lambda branch: (-len(branch.items), branch.order))
    return [*parent.sources, *defaults[len(parent.items) :]]


def _extend_branches(
    branches: list[_Branch[_ItemT, _SourceT, _ContextT]],
    *,
    prompt: Sequence[_ItemT],
    prompt_sources: Sequence[_SourceT | None],
    outputs: Sequence[tuple[int, Sequence[_ItemT], Sequence[_SourceT | None]]],
    context: _ContextT,
    sequence: int,
    start_time: datetime,
    context_source: _ModelledExchange | None = None,
    continuation: Callable[[_Branch[_ItemT, _SourceT, _ContextT]], bool] | None = None,
    prompt_lineage_keys: Sequence[object] | None = None,
    output_lineage_keys: Sequence[Sequence[object]] | None = None,
) -> None:
    if len(prompt) != len(prompt_sources):
        raise AssertionError("prompt sources must parallel prompt items")
    if (prompt_lineage_keys is None) != (output_lineage_keys is None):
        raise AssertionError("prompt and output lineage keys must be provided together")
    if prompt_lineage_keys is not None and len(prompt_lineage_keys) != len(prompt):
        raise AssertionError("prompt lineage keys must parallel prompt items")
    if output_lineage_keys is not None and len(output_lineage_keys) != len(outputs):
        raise AssertionError("output lineage keys must parallel outputs")
    if output_lineage_keys is not None and any(
        len(keys) != len(output)
        for keys, (_, output, _) in zip(output_lineage_keys, outputs, strict=True)
    ):
        raise AssertionError("output lineage keys must parallel output items")
    candidates = [
        (index, branch)
        for index, branch in enumerate(branches)
        if branch.context == context
        and _is_prefix(branch.items, prompt)
        and (continuation is None or continuation(branch))
    ]
    parent: _Branch[_ItemT, _SourceT, _ContextT] | None = None
    remove_parent = False
    if candidates:
        index, parent = min(
            candidates,
            key=lambda item: (-len(item[1].items), item[1].order),
        )
        remove_parent = True
        branches.pop(index)
        sources = [
            *parent.sources,
            *prompt_sources[len(parent.items) :],
        ]
    else:
        regenerations = [
            branch
            for branch in branches
            if branch.context == context
            and _is_prefix(prompt, branch.items)
            and (continuation is None or continuation(branch))
        ]
        if regenerations:
            parent = min(regenerations, key=lambda branch: branch.order)
            sources = copy.copy(parent.sources[: len(prompt)])
        else:
            sources = copy.copy(prompt_sources)

    base_order = parent.order if parent is not None else (sequence,)
    first_time = parent.first_time if parent is not None else start_time
    created = [
        _Branch(
            items=(
                [*prompt, *output]
                if prompt_lineage_keys is not None
                else [*copy.deepcopy(prompt), *copy.deepcopy(output)]
            ),
            sources=[*sources, *output_sources],
            context=copy.deepcopy(context),
            order=(*base_order, choice_index),
            first_time=first_time,
            context_source=context_source,
            lineage_keys=(
                [*prompt_lineage_keys, *output_lineage_keys[position]]
                if prompt_lineage_keys is not None and output_lineage_keys is not None
                else None
            ),
        )
        for position, (choice_index, output, output_sources) in enumerate(outputs)
    ]
    if not created and remove_parent and parent is not None:
        branches.append(parent)
    branches.extend(created)


def _contains_tokens(tokens: Sequence[int], sampled: Sequence[int]) -> bool:
    return any(
        list(tokens[start : start + len(sampled)]) == list(sampled)
        for start in range(len(tokens) - len(sampled) + 1)
    )


def _chat_retains_sampled_reasoning(
    branch: _Branch[Message, ChatCompletionsMessageSource, _ChatContext],
    exchange: ChatCompletionsExchange,
    prompt_length: int,
) -> bool:
    """Split histories when a template drops a prior sampled reasoning span."""

    if not exchange.response.choices:
        return True
    from ._tokenize import _chat_choice_tokens

    response_data = exchange.response.model_dump(mode="python")
    prompt_ids, _, _ = _chat_choice_tokens(exchange.response.choices[0], response_data)
    if prompt_ids is None:
        return True
    for message, source in zip(
        branch.items[:prompt_length], branch.sources[:prompt_length], strict=True
    ):
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if (
            not reasoning
            or source is None
            or source.choice_index is None
            or not isinstance(source.exchange, ChatCompletionsExchange)
        ):
            continue
        source_response = source.exchange.response
        choice = next(
            (
                item
                for item in source_response.choices
                if item.index == source.choice_index
            ),
            None,
        )
        if choice is None:
            continue
        _, sampled_ids, _ = _chat_choice_tokens(
            choice, source_response.model_dump(mode="python")
        )
        if sampled_ids and not _contains_tokens(prompt_ids, sampled_ids):
            return False
    return True


def _chat_message_key(message: Message, *, visible_only: bool = False) -> str:
    data = normalize_chat_message(message)
    if visible_only:
        data.pop("reasoning", None)
        data.pop("reasoning_content", None)
    return json.dumps(data, sort_keys=True, default=str)


def normalize_chat_message(message: Mapping[str, object]) -> dict[str, object]:
    """Return the canonical history form of an OpenAI chat message."""

    data = copy.deepcopy(dict(message))
    data.pop("annotations", None)
    if data.get("role") == "assistant" and data.get("content") is None:
        data["content"] = ""
    elif data.get("content") is None:
        data.pop("content", None)
    if data.get("tool_calls") == []:
        data.pop("tool_calls")
    return data


def _require_unmixed(trajectory: Trajectory) -> None:
    if trajectory.exchanges and (
        trajectory.messages_and_choices
        or trajectory.tools is not None
        or trajectory.additional_histories
    ):
        raise ValueError(
            "A trajectory cannot contain both exchanges and legacy histories"
        )


def legacy_as_chat_completions_history(
    history: LegacyHistory, *, model: str | None
) -> ChatCompletionsHistory:
    messages = history.messages()
    return ChatCompletionsHistory(
        model=model,
        messages=messages,
        message_sources=[None] * len(messages),
        tools=copy.deepcopy(history.tools),
    )


def chat_completions_histories(
    trajectory: Trajectory, *, model: str | None
) -> list[ChatCompletionsHistory]:
    _require_unmixed(trajectory)
    if not trajectory.exchanges:
        return [
            history.as_chat_completions_history(model=model)
            for history in [
                LegacyHistory(
                    messages_and_choices=trajectory.messages_and_choices,
                    tools=trajectory.tools,
                ),
                *trajectory.additional_histories,
            ]
        ]

    histories: list[ChatCompletionsHistory] = []
    for selected_model, exchanges in _selected_models(
        trajectory.exchanges.chat_completions, model, "Chat Completions"
    ):
        branches: list[
            _Branch[Message, ChatCompletionsMessageSource, _ChatContext]
        ] = []
        for sequence, exchange in enumerate(exchanges):
            prompt = [
                cast(Message, normalize_chat_message(message))
                for message in _MESSAGES.validate_python(
                    exchange.request.get("messages", [])
                )
            ]
            prompt_lineage_keys = [
                _chat_message_key(message, visible_only=True) for message in prompt
            ]
            prompt_sources = _lineage_prompt_sources(
                branches,
                prompt=prompt,
                defaults=[
                    ChatCompletionsMessageSource(exchange=exchange, request_index=index)
                    for index in range(len(prompt))
                ],
                equivalent=lambda left, right: (
                    _chat_message_key(left, visible_only=True)
                    == _chat_message_key(right, visible_only=True)
                ),
                prompt_keys=prompt_lineage_keys,
            )
            outputs: list[
                tuple[
                    int,
                    list[Message],
                    list[ChatCompletionsMessageSource | None],
                ]
            ] = []
            for choice in _ordered_choices(
                exchange.response.choices, protocol="Chat Completions"
            ):
                response = _MESSAGE.validate_python(
                    normalize_chat_message(
                        choice.message.model_dump(mode="python", exclude_none=True)
                    )
                )
                response_source = ChatCompletionsMessageSource(
                    exchange=exchange, choice_index=choice.index
                )
                outputs.append(
                    (
                        choice.index,
                        [response],
                        [response_source],
                    )
                )
            output_lineage_keys = [
                [_chat_message_key(output[0], visible_only=True)]
                for _, output, _ in outputs
            ]
            _extend_branches(
                branches,
                prompt=prompt,
                prompt_sources=prompt_sources,
                outputs=outputs,
                context=_ChatContext(
                    tools=_TOOLS.validate_python(exchange.request.get("tools")),
                    template=exchange.request.get("chat_template"),
                    kwargs=_CHAT_KWARGS.validate_python(
                        exchange.request.get("chat_template_kwargs")
                    ),
                ),
                sequence=sequence,
                start_time=exchange.start_time,
                continuation=lambda branch: _chat_retains_sampled_reasoning(
                    branch, exchange, len(prompt)
                ),
                prompt_lineage_keys=prompt_lineage_keys,
                output_lineage_keys=output_lineage_keys,
            )
        for branch in sorted(branches, key=lambda item: (item.first_time, item.order)):
            histories.append(
                ChatCompletionsHistory(
                    model=selected_model,
                    messages=copy.deepcopy(branch.items),
                    message_sources=copy.copy(branch.sources),
                    tools=copy.deepcopy(branch.context.tools),
                    chat_template=branch.context.template,
                    chat_template_kwargs=copy.deepcopy(branch.context.kwargs),
                )
            )
    return histories


def chat_completions_history(
    trajectory: Trajectory, *, model: str | None
) -> ChatCompletionsHistory:
    histories = chat_completions_histories(trajectory, model=model)
    if model is None and len({history.model for history in histories}) > 1:
        raise ValueError(
            "Chat Completions history requires exactly one model; pass model= to select one"
        )
    return _one(histories, "Chat Completions history")


def anthropic_messages_histories(
    trajectory: Trajectory, *, model: str | None
) -> list[AnthropicMessagesHistory]:
    _require_unmixed(trajectory)
    histories: list[AnthropicMessagesHistory] = []
    for selected_model, exchanges in _selected_models(
        trajectory.exchanges.messages, model, "Anthropic Messages"
    ):
        branches: list[
            _Branch[AnthropicMessageParam, AnthropicMessageSource, _AnthropicContext]
        ] = []
        for sequence, exchange in enumerate(exchanges):
            prompt = _anthropic_prompt(exchange.request.get("messages", []))
            prompt_sources = _lineage_prompt_sources(
                branches,
                prompt=prompt,
                defaults=[
                    AnthropicMessageSource(exchange=exchange, request_index=index)
                    for index in range(len(prompt))
                ],
                equivalent=lambda left, right: (
                    _anthropic_message_key(left, visible_only=True)
                    == _anthropic_message_key(right, visible_only=True)
                ),
            )
            response = cast(
                AnthropicMessageParam,
                {
                    "role": "assistant",
                    "content": [
                        block.model_dump(mode="json", exclude_none=True)
                        for block in exchange.response.content
                    ],
                },
            )
            response_source = AnthropicMessageSource(exchange=exchange)
            _extend_branches(
                branches,
                prompt=prompt,
                prompt_sources=prompt_sources,
                outputs=[
                    (
                        0,
                        [response],
                        [response_source],
                    )
                ],
                context=_AnthropicContext(
                    system=_ANTHROPIC_SYSTEM.validate_python(
                        exchange.request.get("system")
                    ),
                    tools=_ANTHROPIC_TOOLS.validate_python(
                        exchange.request.get("tools")
                    ),
                    template=exchange.request.get("chat_template"),
                    kwargs=_CHAT_KWARGS.validate_python(
                        exchange.request.get("chat_template_kwargs")
                    ),
                ),
                sequence=sequence,
                start_time=exchange.start_time,
                context_source=(
                    exchange if exchange.request.get("system") is not None else None
                ),
            )
        for branch in sorted(branches, key=lambda item: (item.first_time, item.order)):
            histories.append(
                AnthropicMessagesHistory(
                    model=selected_model,
                    messages=copy.deepcopy(branch.items),
                    message_sources=copy.copy(branch.sources),
                    system=copy.deepcopy(branch.context.system),
                    system_source=(
                        cast(MessagesExchange, branch.context_source)
                        if branch.context_source is not None
                        else None
                    ),
                    tools=copy.deepcopy(branch.context.tools),
                    chat_template=branch.context.template,
                    chat_template_kwargs=copy.deepcopy(branch.context.kwargs),
                )
            )
    return histories


def _anthropic_prompt(value: object) -> list[AnthropicMessageParam]:
    if not isinstance(value, list):
        raise ValueError("Anthropic messages must be a list")
    messages: list[AnthropicMessageParam] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError("Anthropic messages must be JSON objects")
        content = message.get("content")
        if not isinstance(content, (str, list)):
            raise ValueError("Anthropic message content must be text or a list")
        if isinstance(content, list) and any(
            not isinstance(block, (Mapping, pydantic.BaseModel)) for block in content
        ):
            raise ValueError("Anthropic message content blocks must be JSON objects")
        messages.append(cast(AnthropicMessageParam, copy.deepcopy(dict(message))))
    return messages


def _anthropic_message_key(
    message: AnthropicMessageParam, *, visible_only: bool = False
) -> str:
    normalized = copy.deepcopy(dict(message))
    content = message.get("content")
    if isinstance(content, str):
        blocks: list[dict[str, object]] = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, pydantic.BaseModel):
                data = block.model_dump(mode="json", exclude_none=True)
            elif isinstance(block, Mapping):
                data = copy.deepcopy(dict(block))
            else:
                raise ValueError(
                    "Anthropic message content blocks must be JSON objects"
                )
            kind = data.get("type")
            if visible_only and kind in {"thinking", "redacted_thinking"}:
                continue
            for field in ("token_ids", "logprobs"):
                data.pop(field, None)
            blocks.append(data)
    else:
        raise ValueError("Anthropic message content must be text or a list")
    normalized["content"] = blocks
    return json.dumps(normalized, sort_keys=True, default=str)


def anthropic_messages_history(
    trajectory: Trajectory, *, model: str | None
) -> AnthropicMessagesHistory:
    histories = anthropic_messages_histories(trajectory, model=model)
    if model is None and len({history.model for history in histories}) > 1:
        raise ValueError(
            "Anthropic Messages history requires exactly one model; pass model= to select one"
        )
    return _one(histories, "Anthropic Messages history")


def _responses_input(value: object) -> ResponseInputParam:
    if isinstance(value, str):
        return [cast(ResponseInputItemParam, {"role": "user", "content": value})]
    if isinstance(value, list):
        return [_copy_response_item(item) for item in value]
    if value is None:
        return []
    raise ValueError("Responses input must be text or a list of input items")


def _copy_response_item(value: object) -> ResponseInputItemParam:
    if not isinstance(value, dict):
        raise ValueError("Responses input items must be JSON objects")
    # OpenAI models this as a large TypedDict union. Capture already validated
    # the wire shape; this boundary makes the detached protocol-native copy.
    return cast(ResponseInputItemParam, copy.deepcopy(value))


def responses_histories(
    trajectory: Trajectory, *, model: str | None
) -> list[ResponsesHistory]:
    _require_unmixed(trajectory)
    histories: list[ResponsesHistory] = []
    for selected_model, exchanges in _selected_models(
        trajectory.exchanges.responses, model, "Responses"
    ):
        branches: list[
            _Branch[ResponseInputItemParam, ResponsesItemSource, _ResponsesContext]
        ] = []
        responses: dict[str, _ResponsesRecord] = {}
        conversations: dict[str, _ResponsesRecord] = {}
        for sequence, exchange in enumerate(exchanges):
            request = exchange.request
            prompt = _responses_input(request.get("input"))
            prompt_sources: list[ResponsesItemSource | None] = [
                ResponsesItemSource(exchange=exchange, request_index=index)
                for index in range(len(prompt))
            ]
            requested_conversation = _RESPONSE_CONVERSATION.validate_python(
                request.get("conversation")
            )
            previous = request.get("previous_response_id")
            inherited_conversation: ResponsesConversation | None = None
            inherited_previous: str | None = None
            prior: _ResponsesRecord | None = None
            if isinstance(previous, str) and previous in responses:
                prior = responses[previous]
            elif requested_conversation is not None:
                prior = conversations.get(
                    json.dumps(requested_conversation, sort_keys=True, default=str)
                )
            if prior is not None:
                (
                    prior_items,
                    prior_sources,
                    inherited_conversation,
                    inherited_previous,
                ) = prior
                prompt = [*copy.deepcopy(prior_items), *prompt]
                prompt_sources = [*copy.copy(prior_sources), *prompt_sources]
            prompt_sources = _lineage_prompt_sources(
                branches,
                prompt=prompt,
                defaults=prompt_sources,
                equivalent=lambda left, right: (
                    _response_item_key(left) == _response_item_key(right)
                ),
            )
            output = [
                cast(
                    ResponseInputItemParam,
                    item.model_dump(mode="json", exclude_none=True),
                )
                for item in exchange.response.output
            ]
            generations, generation_indices = _responses_generations(
                exchange, output=output
            )
            output_sources: list[ResponsesItemSource | None] = [
                ResponsesItemSource(
                    exchange=exchange,
                    output_index=index,
                    generation_index=generation_indices[index],
                )
                for index in range(len(output))
            ]
            instructions = request.get("instructions")
            if instructions is not None and not isinstance(instructions, str):
                raise ValueError("Responses instructions must be text")
            external_previous = (
                previous
                if isinstance(previous, str) and previous not in responses
                else inherited_previous
            )
            conversation = (
                requested_conversation
                if requested_conversation is not None
                else inherited_conversation
            )
            context = _ResponsesContext(
                instructions=instructions,
                tools=_RESPONSE_TOOLS.validate_python(request.get("tools")),
                conversation=conversation,
                previous_response_id=external_previous,
                template=request.get("chat_template"),
                kwargs=_CHAT_KWARGS.validate_python(
                    request.get("chat_template_kwargs")
                ),
            )
            final_items = [*copy.deepcopy(prompt), *copy.deepcopy(output)]
            final_sources = [*copy.copy(prompt_sources), *copy.copy(output_sources)]
            if generations:
                for generation_index, generation in enumerate(generations):
                    outputless = not generation.output_indices
                    if outputless and generation_index != len(generations) - 1:
                        raise ValueError(
                            "A nonterminal Responses token generation without "
                            "native output items cannot be projected as a history"
                        )
                    if outputless:
                        output_start = output_end = len(output)
                    else:
                        output_start = generation.output_indices[0]
                        output_end = generation.output_indices[-1] + 1
                    if not outputless and generation_index == len(generations) - 1:
                        output_end = len(output)
                    generation_prompt = [
                        *copy.deepcopy(prompt),
                        *copy.deepcopy(output[:output_start]),
                    ]
                    generation_prompt_sources = [
                        *copy.copy(prompt_sources),
                        *copy.copy(output_sources[:output_start]),
                    ]
                    continuation = lambda branch, generation=generation: (
                        _responses_generation_extends(
                            branch,
                            generation=generation,
                        )
                    )
                    extends = any(
                        branch.context == context
                        and _is_prefix(branch.items, generation_prompt)
                        and continuation(branch)
                        for branch in branches
                    )
                    if not extends:
                        retained = [
                            (item, source)
                            for item, source in zip(
                                generation_prompt,
                                generation_prompt_sources,
                                strict=True,
                            )
                            if not (
                                item.get("type") == "reasoning"
                                and source is not None
                                and source.output_index is not None
                                and source.generation_index is not None
                            )
                        ]
                        generation_prompt = [item for item, _ in retained]
                        generation_prompt_sources = [
                            (
                                replace(source, generation_index=None)
                                if source is not None
                                and source.exchange is exchange
                                and source.generation_index is not None
                                else source
                            )
                            for _, source in retained
                        ]
                    if outputless:
                        generation_output = [
                            cast(
                                ResponseInputItemParam,
                                {"role": "assistant", "content": ""},
                            )
                        ]
                        generation_output_sources = [
                            ResponsesItemSource(
                                exchange=exchange,
                                generation_index=generation_index,
                            )
                        ]
                    else:
                        generation_output = output[output_start:output_end]
                        generation_output_sources = output_sources[
                            output_start:output_end
                        ]
                    _extend_branches(
                        branches,
                        prompt=generation_prompt,
                        prompt_sources=generation_prompt_sources,
                        outputs=[
                            (
                                generation_index,
                                generation_output,
                                generation_output_sources,
                            )
                        ],
                        context=context,
                        sequence=sequence,
                        start_time=exchange.start_time,
                        context_source=(exchange if instructions is not None else None),
                        continuation=continuation,
                    )
                    final_items = [
                        *copy.deepcopy(generation_prompt),
                        *copy.deepcopy(generation_output),
                    ]
                    final_sources = [
                        *copy.copy(generation_prompt_sources),
                        *copy.copy(generation_output_sources),
                    ]
            else:
                _extend_branches(
                    branches,
                    prompt=prompt,
                    prompt_sources=prompt_sources,
                    outputs=[(0, output, output_sources)],
                    context=context,
                    sequence=sequence,
                    start_time=exchange.start_time,
                    context_source=exchange if instructions is not None else None,
                )
            record = (
                final_items,
                final_sources,
                copy.deepcopy(conversation),
                external_previous,
            )
            responses[exchange.response.id] = record
            if conversation is not None:
                conversations[json.dumps(conversation, sort_keys=True, default=str)] = (
                    record
                )
        for branch in sorted(branches, key=lambda item: (item.first_time, item.order)):
            histories.append(
                ResponsesHistory(
                    model=selected_model,
                    input=copy.deepcopy(branch.items),
                    input_sources=copy.copy(branch.sources),
                    instructions=branch.context.instructions,
                    instructions_source=(
                        cast(ResponsesExchange, branch.context_source)
                        if branch.context_source is not None
                        else None
                    ),
                    tools=copy.deepcopy(branch.context.tools),
                    conversation=copy.deepcopy(branch.context.conversation),
                    previous_response_id=branch.context.previous_response_id,
                    chat_template=branch.context.template,
                    chat_template_kwargs=copy.deepcopy(branch.context.kwargs),
                )
            )
    return histories


def _response_item_key(item: ResponseInputItemParam) -> str:
    return json.dumps(item, sort_keys=True, default=str)


def _responses_generations(
    exchange: ResponsesExchange, *, output: Sequence[ResponseInputItemParam]
) -> tuple[list[_ResponsesGeneration], list[int | None]]:
    from ._tokenize import _response_generations

    generations = _response_generations(exchange.response)
    result: list[int | None] = [None] * len(output)
    parsed: list[_ResponsesGeneration] = []
    for generation_index, generation in enumerate(generations):
        if generation.prompt_token_ids is None:
            raise ValueError(
                "Responses generation without an exact prompt cannot be projected"
            )
        if generation.output_token_ids is None:
            raise ValueError(
                "Responses generation without exact output tokens cannot yet be "
                "projected as a history"
            )
        for output_index in generation.output_indices:
            result[output_index] = generation_index
        parsed.append(
            _ResponsesGeneration(
                prompt_token_ids=generation.prompt_token_ids,
                output_token_ids=generation.output_token_ids,
                output_indices=generation.output_indices,
            )
        )
    return parsed, result


def _responses_generation_extends(
    branch: _Branch[ResponseInputItemParam, ResponsesItemSource, _ResponsesContext],
    *,
    generation: _ResponsesGeneration,
) -> bool:
    prior_source = next(
        (
            source
            for source in reversed(branch.sources)
            if source is not None and source.generation_index is not None
        ),
        None,
    )
    if prior_source is None or prior_source.generation_index is None:
        return True
    prior_exchange = prior_source.exchange
    prior_output = [
        cast(
            ResponseInputItemParam,
            item.model_dump(mode="json", exclude_none=True),
        )
        for item in prior_exchange.response.output
    ]
    prior_generations, _ = _responses_generations(prior_exchange, output=prior_output)
    if not 0 <= prior_source.generation_index < len(prior_generations):
        raise ValueError("Responses generation source index is out of bounds")
    prior = prior_generations[prior_source.generation_index]
    if _is_prefix(
        [*prior.prompt_token_ids, *prior.output_token_ids],
        generation.prompt_token_ids,
    ):
        return True
    return not any(
        getattr(prior_exchange.response.output[index], "type", None) == "reasoning"
        for index in prior.output_indices
    )


def responses_history(trajectory: Trajectory, *, model: str | None) -> ResponsesHistory:
    histories = responses_histories(trajectory, model=model)
    if model is None and len({history.model for history in histories}) > 1:
        raise ValueError(
            "Responses history requires exactly one model; pass model= to select one"
        )
    return _one(histories, "Responses history")


def _completion_exact_tokens(
    exchange: CompletionsExchange, choice_index: int
) -> tuple[list[int] | None, list[int] | None]:
    from ._tokenize import _completion_tokens, _exact_token_ids

    choice = next(
        choice for choice in exchange.response.choices if choice.index == choice_index
    )
    prompt, completion, _ = _completion_tokens(
        exchange.response.model_copy(update={"choices": [choice]}),
        echo=exchange.request.get("echo") is True,
    )
    request_prompt = exchange.request.get("prompt")
    if prompt is None and isinstance(request_prompt, list):
        try:
            prompt = _exact_token_ids(
                request_prompt, field="Completions request prompt"
            )
        except ValueError:
            pass
    return prompt, completion


def completions_token_histories(
    trajectory: Trajectory, *, model: str | None
) -> list[CompletionsTokenHistory]:
    _require_unmixed(trajectory)
    histories: list[CompletionsTokenHistory] = []
    for selected_model, exchanges in _selected_models(
        trajectory.exchanges.completions, model, "Completions"
    ):
        branches: list[_Branch[int, CompletionsSource, tuple[()]]] = []
        complete = True
        for sequence, exchange in enumerate(exchanges):
            if exchange.request.get("suffix") is not None:
                raise ValueError("Completions suffix is not supported")
            choice_groups = _completion_choice_groups(exchange)
            for prompt_index, request_prompt in enumerate(
                _completion_prompts(exchange.request.get("prompt"))
            ):
                choices = choice_groups[prompt_index]
                for choice in choices:
                    prompt, completion = _completion_exact_tokens(
                        exchange, choice.index
                    )
                    if prompt is None:
                        if not isinstance(request_prompt, list):
                            complete = False
                            continue
                        prompt = copy.deepcopy(request_prompt)
                    if completion is None:
                        complete = False
                        continue
                    prompt_source = CompletionsSource(
                        exchange=exchange, prompt_index=prompt_index
                    )
                    output_source = CompletionsSource(
                        exchange=exchange,
                        prompt_index=prompt_index,
                        choice_index=choice.index,
                    )
                    _extend_branches(
                        branches,
                        prompt=prompt,
                        prompt_sources=[prompt_source] * len(prompt),
                        outputs=[
                            (
                                choice.index,
                                completion,
                                [output_source] * len(completion),
                            )
                        ],
                        context=(),
                        sequence=sequence,
                        start_time=exchange.start_time,
                    )
        if not complete:
            raise ValueError(
                "Completions token history requires exact token IDs for every choice"
            )
        for branch in branches:
            histories.append(
                CompletionsTokenHistory(
                    model=selected_model,
                    prompt=copy.deepcopy(branch.items),
                    prompt_sources=_token_source_spans(branch.sources),
                    sampled_spans=_sampled_spans(branch.sources),
                )
            )
    if not histories:
        raise ValueError("Completions token history requires exact token IDs")
    return histories


def completions_token_history(
    trajectory: Trajectory, *, model: str | None
) -> CompletionsTokenHistory:
    return _one(
        completions_token_histories(trajectory, model=model),
        "Completions token history",
    )


def _completion_prompts(value: object) -> list[str | list[int]]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
            return [_checked_token_ids(value)]
        prompts: list[str | list[int]] = []
        for item in value:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, list) and all(
                isinstance(token, int) and not isinstance(token, bool) for token in item
            ):
                prompts.append(_checked_token_ids(item))
            else:
                raise ValueError("Invalid batched Completions prompt")
        return prompts
    raise ValueError("Completions prompt must be text or token IDs")


def _checked_token_ids(value: Sequence[object]) -> list[int]:
    result: list[int] = []
    for token in value:
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise ValueError(
                "Completions token prompts must contain non-negative integers"
            )
        result.append(token)
    return result


def _completion_choice_groups(
    exchange: CompletionsExchange,
) -> list[list[CompletionChoice]]:
    prompts = _completion_prompts(exchange.request.get("prompt"))
    count = len(prompts)
    choices = _ordered_choices(exchange.response.choices, protocol="Completions")
    missing_prompt_index = object()
    explicit_matches: dict[int, int] = {}
    for choice in choices:
        raw_prompt_index = (choice.model_extra or {}).get(
            "prompt_index", missing_prompt_index
        )
        if raw_prompt_index is missing_prompt_index:
            continue
        if (
            not isinstance(raw_prompt_index, int)
            or isinstance(raw_prompt_index, bool)
            or not 0 <= raw_prompt_index < count
        ):
            raise ValueError(
                "Completions choice prompt_index must identify a batched prompt"
            )
        explicit_matches[choice.index] = raw_prompt_index
    if count == 1:
        return [choices]
    requested_n = exchange.request.get("n")
    if requested_n is not None and (
        not isinstance(requested_n, int)
        or isinstance(requested_n, bool)
        or requested_n < 1
    ):
        raise ValueError("Completions n must be a positive integer")

    exact_matches: dict[int, int] = {}
    for choice in choices:
        prompt_ids, _ = _completion_exact_tokens(exchange, choice.index)
        if prompt_ids is None:
            continue
        matches = [
            prompt_index
            for prompt_index, prompt in enumerate(prompts)
            if isinstance(prompt, list) and prompt == prompt_ids
        ]
        explicit = explicit_matches.get(choice.index)
        if (
            explicit is not None
            and isinstance(prompts[explicit], list)
            and prompts[explicit] != prompt_ids
        ):
            raise ValueError(
                "Completions prompt_index contradicts exact prompt evidence"
            )
        if len(matches) == 1:
            exact_matches[choice.index] = matches[0]

    evidence_matches = {**exact_matches, **explicit_matches}
    if len(evidence_matches) == len(choices):
        groups = [[] for _ in prompts]
        for choice in choices:
            groups[evidence_matches[choice.index]].append(choice)
        if all(groups) and (
            requested_n is None or all(len(group) == requested_n for group in groups)
        ):
            return groups
        raise ValueError("Cannot associate Completions choices with batched prompts")

    if len(choices) % count:
        raise ValueError("Cannot associate Completions choices with batched prompts")
    per_prompt = len(choices) // count
    if requested_n is not None and requested_n != per_prompt:
        raise ValueError("Cannot associate Completions choices with batched prompts")
    if [choice.index for choice in choices] != list(range(len(choices))):
        raise ValueError("Ambiguous Completions choice-to-prompt association")
    groups = [
        choices[prompt_index * per_prompt : (prompt_index + 1) * per_prompt]
        for prompt_index in range(count)
    ]
    for prompt_index, group in enumerate(groups):
        if any(
            choice.index in evidence_matches
            and evidence_matches[choice.index] != prompt_index
            for choice in group
        ):
            raise ValueError("Completions prompt evidence contradicts choice indices")
    return groups


def completions_string_histories(
    trajectory: Trajectory, *, model: str | None
) -> list[CompletionsStringHistory]:
    _require_unmixed(trajectory)
    histories: list[CompletionsStringHistory] = []
    for selected_model, exchanges in _selected_models(
        trajectory.exchanges.completions, model, "Completions"
    ):
        branches: list[_Branch[str, CompletionsSource, tuple[()]]] = []
        complete = True
        for sequence, exchange in enumerate(exchanges):
            if exchange.request.get("suffix") is not None:
                raise ValueError("Completions suffix is not supported")
            choice_groups = _completion_choice_groups(exchange)
            for prompt_index, prompt in enumerate(
                _completion_prompts(exchange.request.get("prompt"))
            ):
                if not isinstance(prompt, str):
                    complete = False
                    continue
                for choice in choice_groups[prompt_index]:
                    text = choice.text
                    if exchange.request.get("echo") is True:
                        if not text.startswith(prompt):
                            raise ValueError(
                                "Cannot locate echoed Completions prompt boundary"
                            )
                        text = text[len(prompt) :]
                    prompt_source = CompletionsSource(
                        exchange=exchange, prompt_index=prompt_index
                    )
                    output_source = CompletionsSource(
                        exchange=exchange,
                        prompt_index=prompt_index,
                        choice_index=choice.index,
                    )
                    _extend_branches(
                        branches,
                        prompt=list(prompt),
                        prompt_sources=[prompt_source] * len(prompt),
                        outputs=[
                            (
                                choice.index,
                                list(text),
                                [output_source] * len(text),
                            )
                        ],
                        context=(),
                        sequence=sequence,
                        start_time=exchange.start_time,
                    )
        if not complete:
            raise ValueError(
                "Completions string history requires text prompts for every choice"
            )
        histories.extend(
            CompletionsStringHistory(
                model=selected_model,
                prompt="".join(branch.items),
                prompt_sources=_string_source_spans(branch.sources),
                sampled_spans=_sampled_spans(branch.sources),
            )
            for branch in branches
        )
    if not histories:
        raise ValueError("Completions string history requires text prompts")
    return histories


def completions_string_history(
    trajectory: Trajectory, *, model: str | None
) -> CompletionsStringHistory:
    return _one(
        completions_string_histories(trajectory, model=model),
        "Completions string history",
    )


def _source_runs(
    sources: Sequence[CompletionsSource | None],
) -> Iterable[tuple[int, int, CompletionsSource | None]]:
    if not sources:
        return
    start = 0
    source = sources[0]
    for index, item in enumerate(sources[1:], 1):
        if item != source:
            yield start, index, source
            start, source = index, item
    yield start, len(sources), source


def _token_source_spans(
    sources: Sequence[CompletionsSource | None],
) -> list[CompletionsTokenSourceSpan]:
    return [
        CompletionsTokenSourceSpan(start=start, end=end, source=source)
        for start, end, source in _source_runs(sources)
    ]


def _string_source_spans(
    sources: Sequence[CompletionsSource | None],
) -> list[CompletionsStringSourceSpan]:
    return [
        CompletionsStringSourceSpan(start=start, end=end, source=source)
        for start, end, source in _source_runs(sources)
    ]


def _sampled_spans(
    sources: Sequence[CompletionsSource | None],
) -> list[tuple[int, int]]:
    return [
        (start, end)
        for start, end, source in _source_runs(sources)
        if source is not None and source.choice_index is not None
    ]


def anthropic_as_chat_completions_history(
    history: AnthropicMessagesHistory,
) -> ChatCompletionsHistory:
    from ._tokenize import _anthropic_messages, _openai_tools

    messages: list[dict[str, object]] = []
    sources: list[ChatCompletionsMessageSource | None] = []
    if history.system:
        messages.extend(_anthropic_messages({"system": history.system, "messages": []}))
        sources.append(
            ChatCompletionsMessageSource(exchange=history.system_source)
            if history.system_source is not None
            else None
        )
    for message, source in zip(history.messages, history.message_sources, strict=True):
        converted_messages = _anthropic_messages({"messages": [message]})
        messages.extend(converted_messages)
        converted = (
            ChatCompletionsMessageSource(
                exchange=source.exchange,
                request_index=source.request_index,
                output_indices=(0,) if source.request_index is None else None,
            )
            if source is not None
            else None
        )
        sources.extend([converted] * len(converted_messages))
    tools = _TOOLS.validate_python(_openai_tools(history.tools, dialect="messages"))
    return ChatCompletionsHistory(
        model=history.model,
        messages=_MESSAGES.validate_python(messages),
        message_sources=sources,
        tools=tools,
        chat_template=history.chat_template,
        chat_template_kwargs=copy.deepcopy(history.chat_template_kwargs),
    )


def responses_as_chat_completions_history(
    history: ResponsesHistory,
) -> ChatCompletionsHistory:
    if history.previous_response_id is not None or history.conversation is not None:
        raise ValueError(
            "Opaque Responses context cannot be represented as Chat Completions"
        )
    from ._tokenize import _openai_tools, _responses_messages

    messages = _responses_messages({"instructions": history.instructions, "input": []})
    sources: list[ChatCompletionsMessageSource | None] = []
    if history.instructions:
        sources.append(
            ChatCompletionsMessageSource(exchange=history.instructions_source)
            if history.instructions_source is not None
            else None
        )

    def converted(
        contributors: Sequence[ResponsesItemSource | None],
    ) -> ChatCompletionsMessageSource | None:
        if not contributors or any(source is None for source in contributors):
            return None
        present = [source for source in contributors if source is not None]
        exchanges = {id(source.exchange): source.exchange for source in present}
        if len(exchanges) != 1:
            raise ValueError(
                "One projected Chat message cannot span multiple Responses exchanges"
            )
        generation_indices = {
            source.generation_index
            for source in present
            if source.generation_index is not None
        }
        if len(generation_indices) > 1:
            raise ValueError(
                "One projected Chat message cannot span multiple Responses generations"
            )
        output_indices = tuple(
            dict.fromkeys(
                source.output_index
                for source in present
                if source.output_index is not None
            )
        )
        generation_index = next(iter(generation_indices), None)
        first = present[0]
        request_index = next(
            (
                source.request_index
                for source in present
                if source.request_index is not None
            ),
            None,
        )
        if output_indices or generation_index is not None:
            return ChatCompletionsMessageSource(
                exchange=first.exchange,
                request_index=request_index,
                output_indices=output_indices,
                generation_index=generation_index,
            )
        return ChatCompletionsMessageSource(
            exchange=first.exchange,
            request_index=request_index,
        )

    no_output = object()

    def compatible(
        contributors: Sequence[ResponsesItemSource | None],
        source: ResponsesItemSource | None,
    ) -> bool:
        if source is None:
            return True
        present = [item for item in contributors if item is not None]
        if present and source.exchange is not present[0].exchange:
            return False

        def output_generation(item: ResponsesItemSource) -> object:
            if item.output_index is None and item.generation_index is None:
                return no_output
            return item.generation_index

        generations = {
            value
            for item in present
            if (value := output_generation(item)) is not no_output
        }
        candidate = output_generation(source)
        return candidate is no_output or not generations or candidate in generations

    item_groups: list[list[ResponseInputItemParam]] = []
    source_groups: list[list[ResponsesItemSource | None]] = []
    pending_reasoning_items: list[ResponseInputItemParam] = []
    pending_reasoning: list[ResponsesItemSource | None] = []
    tool_message_items: list[ResponseInputItemParam] | None = None
    tool_message_sources: list[ResponsesItemSource | None] | None = None
    for item, source in zip(history.input, history.input_sources, strict=True):
        kind = item.get("type")
        if kind == "reasoning":
            if pending_reasoning and not compatible(pending_reasoning, source):
                item_groups.append([*pending_reasoning_items])
                source_groups.append([*pending_reasoning])
                pending_reasoning_items.clear()
                pending_reasoning.clear()
            tool_message_items = None
            tool_message_sources = None
            pending_reasoning_items.append(item)
            pending_reasoning.append(source)
            continue
        if kind == "function_call":
            if tool_message_sources is None:
                if pending_reasoning and not compatible(pending_reasoning, source):
                    item_groups.append([*pending_reasoning_items])
                    source_groups.append([*pending_reasoning])
                    pending_reasoning_items.clear()
                    pending_reasoning.clear()
                tool_message_items = [*pending_reasoning_items, item]
                tool_message_sources = [*pending_reasoning, source]
                item_groups.append(tool_message_items)
                source_groups.append(tool_message_sources)
                pending_reasoning_items.clear()
                pending_reasoning.clear()
            elif compatible(tool_message_sources, source):
                assert tool_message_items is not None
                tool_message_items.append(item)
                tool_message_sources.append(source)
            else:
                tool_message_items = [item]
                tool_message_sources = [source]
                item_groups.append(tool_message_items)
                source_groups.append(tool_message_sources)
            continue
        tool_message_items = None
        tool_message_sources = None
        if pending_reasoning:
            if (
                kind in {None, "message"}
                and item.get("role") == "assistant"
                and compatible(pending_reasoning, source)
            ):
                item_groups.append([*pending_reasoning_items, item])
                source_groups.append([*pending_reasoning, source])
            else:
                item_groups.append([*pending_reasoning_items])
                source_groups.append([*pending_reasoning])
                item_groups.append([item])
                source_groups.append([source])
            pending_reasoning_items.clear()
            pending_reasoning.clear()
        else:
            item_groups.append([item])
            source_groups.append([source])
    if pending_reasoning:
        item_groups.append(pending_reasoning_items)
        source_groups.append(pending_reasoning)
    for items, group in zip(item_groups, source_groups, strict=True):
        projected = _responses_messages({"input": items})
        if len(projected) != 1:
            raise AssertionError(
                "One Responses projection group must produce one Chat message"
            )
        messages.extend(projected)
        sources.append(converted(group))
    if len(sources) != len(messages):
        raise AssertionError("Responses conversion sources must parallel messages")
    tools = _TOOLS.validate_python(_openai_tools(history.tools, dialect="responses"))
    return ChatCompletionsHistory(
        model=history.model,
        messages=_MESSAGES.validate_python(messages),
        message_sources=sources,
        tools=tools,
        chat_template=history.chat_template,
        chat_template_kwargs=copy.deepcopy(history.chat_template_kwargs),
    )


def trajectory_histories(
    trajectory: Trajectory, *, model: str | None
) -> list[TrajectoryHistory]:
    _require_unmixed(trajectory)
    if not trajectory.exchanges:
        return [
            LegacyHistory(
                messages_and_choices=copy.deepcopy(trajectory.messages_and_choices),
                tools=copy.deepcopy(trajectory.tools),
            ),
            *copy.deepcopy(trajectory.additional_histories),
        ]

    all_exchanges = [
        *trajectory.exchanges.chat_completions,
        *trajectory.exchanges.completions,
        *trajectory.exchanges.responses,
        *trajectory.exchanges.messages,
    ]
    has_exact_match = model is not None and any(
        exchange.model == model for exchange in all_exchanges
    )

    def is_selected(exchange: _ModelledExchange) -> bool:
        if model is None:
            return True
        if has_exact_match:
            return exchange.model == model
        return _model_matches(exchange.model, model)

    candidates: list[TrajectoryHistory] = []
    if any(is_selected(exchange) for exchange in trajectory.exchanges.chat_completions):
        candidates.extend(chat_completions_histories(trajectory, model=model))
    if any(is_selected(exchange) for exchange in trajectory.exchanges.completions):
        try:
            candidates.extend(completions_token_histories(trajectory, model=model))
        except ValueError as error:
            if "requires exact token IDs" not in str(error):
                raise
            candidates.extend(completions_string_histories(trajectory, model=model))
    if any(is_selected(exchange) for exchange in trajectory.exchanges.responses):
        candidates.extend(responses_histories(trajectory, model=model))
    if any(is_selected(exchange) for exchange in trajectory.exchanges.messages):
        candidates.extend(anthropic_messages_histories(trajectory, model=model))
    if not candidates:
        suffix = f" for model {model!r}" if model is not None else ""
        raise ValueError(f"Trajectory contains no exchanges{suffix}")
    protocols = {type(history) for history in candidates}
    if len(protocols) != 1:
        raise ValueError(
            "Trajectory resolves to multiple protocol histories; use a protocol-specific method"
        )
    return candidates


def trajectory_history(
    trajectory: Trajectory, *, model: str | None
) -> TrajectoryHistory:
    histories = trajectory_histories(trajectory, model=model)
    if (
        model is None
        and len(
            {
                history.model
                for history in histories
                if not isinstance(history, LegacyHistory)
            }
        )
        > 1
    ):
        raise ValueError(
            "Trajectory history requires exactly one model; pass model= to select one"
        )
    return _one(histories, "Trajectory history")


def trajectory_messages(trajectory: Trajectory) -> Messages:
    if not trajectory.exchanges:
        return LegacyHistory(
            messages_and_choices=trajectory.messages_and_choices,
            tools=trajectory.tools,
        ).messages()
    return (
        trajectory_history(trajectory, model=None)
        .as_chat_completions_history()
        .messages
    )
