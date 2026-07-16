from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import copy
from datetime import datetime
from typing import Protocol, TypeVar

import pydantic

from ..types import Message, Messages, Tools
from . import (
    AnthropicMessagesHistory,
    ChatCompletionsExchange,
    ChatCompletionsHistory,
    CompletionsExchange,
    CompletionsHistory,
    History,
    MessagesExchange,
    ResponsesExchange,
    ResponsesHistory,
    Trajectory,
    TrajectoryHistory,
)


class _ModelledExchange(Protocol):
    start_time: datetime
    end_time: datetime

    @property
    def model(self) -> str | None: ...


_ExchangeT = TypeVar("_ExchangeT", bound=_ModelledExchange)
_ItemT = TypeVar("_ItemT")
_MESSAGES = pydantic.TypeAdapter(Messages)
_MESSAGE = pydantic.TypeAdapter(Message)
_TOOLS = pydantic.TypeAdapter(Tools | None)


def _select(
    exchanges: Sequence[_ExchangeT], model: str | None, protocol: str
) -> list[_ExchangeT]:
    selected = [
        exchange for exchange in exchanges if model is None or exchange.model == model
    ]
    if not selected:
        suffix = f" for model {model!r}" if model is not None else ""
        raise ValueError(f"Trajectory contains no {protocol} exchanges{suffix}")
    models = {exchange.model for exchange in selected}
    if None in models:
        raise ValueError(f"Every {protocol} exchange must identify its model")
    if len(models) != 1:
        raise ValueError(
            f"{protocol} history requires exactly one model; pass model= to select one"
        )
    return sorted(
        selected, key=lambda exchange: (exchange.start_time, exchange.end_time)
    )


def _require_constant(values: Iterable[object], field: str) -> None:
    iterator = iter(values)
    first = next(iterator)
    if any(value != first for value in iterator):
        raise ValueError(f"Exchanges with different {field} form different histories")


def _require_context(
    requests: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    for field in fields:
        _require_constant((request.get(field) for request in requests), field)


def _extend(
    history: list[_ItemT] | None,
    prompt: Sequence[_ItemT],
    completion: Sequence[_ItemT],
    protocol: str,
) -> list[_ItemT]:
    if history is None:
        history = copy.deepcopy(list(prompt))
    elif len(prompt) < len(history) or list(prompt[: len(history)]) != history:
        raise ValueError(f"{protocol} exchanges do not form one append-only history")
    else:
        history.extend(copy.deepcopy(list(prompt[len(history) :])))
    history.extend(copy.deepcopy(list(completion)))
    return history


def _only_choice(exchange: ChatCompletionsExchange | CompletionsExchange) -> None:
    if len(exchange.response.choices) != 1:
        raise ValueError("Multiple response choices form multiple histories")


def _model(exchange: _ModelledExchange) -> str:
    if exchange.model is None:
        raise AssertionError("_select returned an exchange without a model")
    return exchange.model


def _require_unmixed(trajectory: Trajectory) -> None:
    if trajectory.exchanges and (
        trajectory.messages_and_choices
        or trajectory.tools is not None
        or trajectory.additional_histories
    ):
        raise ValueError(
            "A trajectory cannot contain both exchanges and legacy histories"
        )


def legacy_as_chat_completions_history(history: History) -> ChatCompletionsHistory:
    return ChatCompletionsHistory(
        model=None,
        messages=history.messages(),
        tools=copy.deepcopy(history.tools),
    )


def chat_completions_history(
    trajectory: Trajectory, *, model: str | None
) -> ChatCompletionsHistory:
    _require_unmixed(trajectory)
    if not trajectory.exchanges:
        if trajectory.additional_histories:
            raise ValueError("Trajectory contains multiple legacy histories")
        return ChatCompletionsHistory(
            model=model,
            messages=History(
                messages_and_choices=trajectory.messages_and_choices,
                tools=trajectory.tools,
            ).messages(),
            tools=copy.deepcopy(trajectory.tools),
        )
    exchanges = _select(
        trajectory.exchanges.chat_completions, model, "Chat Completions"
    )
    _require_context(
        [exchange.request for exchange in exchanges],
        ("tools", "chat_template", "chat_template_kwargs", "cache_salt"),
    )
    messages: Messages | None = None
    for exchange in exchanges:
        _only_choice(exchange)
        prompt = _MESSAGES.validate_python(exchange.request.get("messages", []))
        response = _MESSAGE.validate_python(
            exchange.response.choices[0].message.model_dump(
                mode="python", exclude_none=True
            )
        )
        messages = _extend(messages, prompt, [response], "Chat Completions")
    first = exchanges[0].request
    return ChatCompletionsHistory(
        model=_model(exchanges[0]),
        messages=messages or [],
        tools=copy.deepcopy(first.get("tools")),
        chat_template=first.get("chat_template"),
        chat_template_kwargs=copy.deepcopy(first.get("chat_template_kwargs")),
    )


def anthropic_messages_history(
    trajectory: Trajectory, *, model: str | None
) -> AnthropicMessagesHistory:
    _require_unmixed(trajectory)
    exchanges = _select(trajectory.exchanges.messages, model, "Anthropic Messages")
    _require_context(
        [exchange.request for exchange in exchanges],
        (
            "system",
            "tools",
            "thinking",
            "chat_template",
            "chat_template_kwargs",
            "cache_salt",
        ),
    )
    messages: list[object] | None = None
    for exchange in exchanges:
        prompt = copy.deepcopy(exchange.request.get("messages", []))
        response = {
            "role": "assistant",
            "content": [
                block.model_dump(mode="python", exclude_none=True)
                for block in exchange.response.content
            ],
        }
        messages = _extend(messages, prompt, [response], "Anthropic Messages")
    first = exchanges[0].request
    return AnthropicMessagesHistory(
        model=_model(exchanges[0]),
        system=copy.deepcopy(first.get("system")),
        messages=messages or [],
        tools=copy.deepcopy(first.get("tools")),
        thinking=copy.deepcopy(first.get("thinking")),
        chat_template=first.get("chat_template"),
        chat_template_kwargs=copy.deepcopy(first.get("chat_template_kwargs")),
    )


def _responses_input(value: object) -> list[object]:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, list):
        return [copy.deepcopy(item) for item in value]
    if value is None:
        return []
    raise ValueError("Responses input must be text or a list of input items")


def responses_history(trajectory: Trajectory, *, model: str | None) -> ResponsesHistory:
    _require_unmixed(trajectory)
    exchanges = _select(trajectory.exchanges.responses, model, "Responses")
    _require_context(
        [exchange.request for exchange in exchanges],
        (
            "instructions",
            "tools",
            "chat_template",
            "chat_template_kwargs",
            "cache_salt",
        ),
    )
    items: list[object] | None = None
    previous_response_id: str | None = None
    for exchange in exchanges:
        request = exchange.request
        prompt = _responses_input(request.get("input"))
        previous = request.get("previous_response_id")
        if previous is not None:
            if previous != previous_response_id or items is None:
                raise ValueError(
                    "Responses exchange refers to a response outside this history"
                )
            prompt = [*items, *prompt]
        output = [
            item.model_dump(mode="python", exclude_none=True)
            for item in exchange.response.output
        ]
        items = _extend(items, prompt, output, "Responses")
        previous_response_id = exchange.response.id
    first = exchanges[0].request
    return ResponsesHistory(
        model=_model(exchanges[0]),
        input=items or [],
        instructions=first.get("instructions"),
        tools=copy.deepcopy(first.get("tools")),
        chat_template=first.get("chat_template"),
        chat_template_kwargs=copy.deepcopy(first.get("chat_template_kwargs")),
    )


def completions_history(
    trajectory: Trajectory, *, model: str | None
) -> CompletionsHistory:
    from ._tokenize import _completion_tokens, _exact_token_ids

    _require_unmixed(trajectory)
    exchanges = _select(trajectory.exchanges.completions, model, "Completions")
    _require_context([exchange.request for exchange in exchanges], ("cache_salt",))
    token_ids: list[int] = []
    sampled_spans: list[tuple[int, int]] = []
    for index, exchange in enumerate(exchanges):
        _only_choice(exchange)
        if exchange.request.get("echo") is True:
            raise ValueError("Completions history does not support echo=True")
        prompt, completion, _ = _completion_tokens(exchange.response)
        request_prompt = exchange.request.get("prompt")
        if prompt is None and isinstance(request_prompt, list):
            prompt = _exact_token_ids(
                request_prompt, field="Completions request prompt"
            )
        if prompt is None or completion is None:
            raise ValueError(
                "Completions history requires exact prompt and output token IDs"
            )
        if index == 0:
            token_ids.extend(prompt)
        elif len(prompt) < len(token_ids) or prompt[: len(token_ids)] != token_ids:
            raise ValueError(
                "Completions exchanges do not form one append-only token history"
            )
        else:
            token_ids.extend(prompt[len(token_ids) :])
        start = len(token_ids)
        token_ids.extend(completion)
        sampled_spans.append((start, len(token_ids)))
    return CompletionsHistory(
        model=_model(exchanges[0]),
        token_ids=token_ids,
        sampled_spans=sampled_spans,
    )


def anthropic_as_chat_completions_history(
    history: AnthropicMessagesHistory,
) -> ChatCompletionsHistory:
    from ._tokenize import _anthropic_messages, _openai_tools

    messages = _anthropic_messages(
        {"system": history.system, "messages": history.messages}
    )
    tools = _TOOLS.validate_python(_openai_tools(history.tools, dialect="messages"))
    return ChatCompletionsHistory(
        model=history.model,
        messages=_MESSAGES.validate_python(messages),
        tools=tools,
        chat_template=history.chat_template,
        chat_template_kwargs=copy.deepcopy(history.chat_template_kwargs),
    )


def responses_as_chat_completions_history(
    history: ResponsesHistory,
) -> ChatCompletionsHistory:
    from ._tokenize import _openai_tools, _responses_messages

    messages = _responses_messages(
        {"instructions": history.instructions, "input": history.input}
    )
    tools = _TOOLS.validate_python(_openai_tools(history.tools, dialect="responses"))
    return ChatCompletionsHistory(
        model=history.model,
        messages=_MESSAGES.validate_python(messages),
        tools=tools,
        chat_template=history.chat_template,
        chat_template_kwargs=copy.deepcopy(history.chat_template_kwargs),
    )


def trajectory_history(
    trajectory: Trajectory, *, model: str | None
) -> TrajectoryHistory:
    _require_unmixed(trajectory)
    if not trajectory.exchanges:
        if trajectory.additional_histories:
            raise ValueError("Trajectory contains multiple legacy histories")
        if model is not None:
            raise ValueError("Legacy trajectory histories do not identify a model")
        return History(
            messages_and_choices=copy.deepcopy(trajectory.messages_and_choices),
            tools=copy.deepcopy(trajectory.tools),
        )

    candidates = [
        name
        for name, exchanges in (
            ("chat_completions", trajectory.exchanges.chat_completions),
            ("completions", trajectory.exchanges.completions),
            ("responses", trajectory.exchanges.responses),
            ("messages", trajectory.exchanges.messages),
        )
        if any(model is None or exchange.model == model for exchange in exchanges)
    ]
    if not candidates:
        suffix = f" for model {model!r}" if model is not None else ""
        raise ValueError(f"Trajectory contains no exchanges{suffix}")
    if len(candidates) != 1:
        raise ValueError(
            "Trajectory resolves to multiple protocol histories; use a protocol-specific method"
        )
    protocol = candidates[0]
    if protocol == "chat_completions":
        return chat_completions_history(trajectory, model=model)
    if protocol == "completions":
        return completions_history(trajectory, model=model)
    if protocol == "responses":
        return responses_history(trajectory, model=model)
    return anthropic_messages_history(trajectory, model=model)


def trajectory_messages(trajectory: Trajectory) -> Messages:
    if not trajectory.exchanges:
        return History(
            messages_and_choices=trajectory.messages_and_choices,
            tools=trajectory.tools,
        ).messages()
    return (
        trajectory_history(trajectory, model=None)
        .as_chat_completions_history()
        .messages
    )
