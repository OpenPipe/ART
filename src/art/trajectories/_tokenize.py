from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any, Protocol, cast

from anthropic.types import Message
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.responses import Response
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from . import (
    ChatCompletionsExchange,
    CompletionsExchange,
    MessagesExchange,
    ResponsesExchange,
    TokenizedTrajectory,
    Trajectory,
)
from ._protocols import Exchange

_TOKEN_ID = re.compile(r"token_id:(\d+)$")


@dataclass
class _TokenizerConfig:
    base_model: str
    revision: str | None = None
    chat_template: str | None = None
    chat_template_kwargs: Mapping[str, object] | None = None


class _Tokenizer(Protocol):
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


def _as_tokenizer(tokenizer: PreTrainedTokenizerBase) -> _Tokenizer:
    # Transformers' annotation permits only string-valued message dictionaries,
    # although its runtime API supports the structured content ART must tokenize.
    return cast(_Tokenizer, tokenizer)


def _string_dict(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _dict_list(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("Expected a list of JSON objects")
    result: list[dict[str, Any]] = []
    for item in value:
        if (mapping := _string_dict(item)) is None:
            raise TypeError("Expected a list of JSON objects")
        result.append(mapping)
    return result


def _dump(value: object) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        result = value.model_dump(mode="python")
        return result if isinstance(result, dict) else {}
    return _string_dict(value) or {}


def _token_id(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and (match := _TOKEN_ID.fullmatch(value)):
        return int(match.group(1))
    return None


def _pairs(values: object) -> tuple[list[int], list[float]]:
    if not isinstance(values, list):
        return [], []
    token_ids: list[int] = []
    logprobs: list[float] = []
    for value in values:
        data = _dump(value)
        token_id = _token_id(data.get("token_id"))
        if token_id is None:
            token_id = _token_id(data.get("token"))
        if token_id is None:
            return [], []
        logprob = data.get("logprob")
        token_ids.append(token_id)
        logprobs.append(
            float(logprob) if isinstance(logprob, (int, float)) else math.nan
        )
    return token_ids, logprobs


def _logprob_values(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for value in values:
        logprob = _dump(value).get("logprob")
        if not isinstance(logprob, (int, float)):
            return []
        result.append(float(logprob))
    return result


def _chat_choice_tokens(
    choice: Choice, response_data: dict[str, Any]
) -> tuple[list[int] | None, list[int], list[float]]:
    choice_data = _dump(choice)
    prompt = choice_data.get("prompt_token_ids") or response_data.get(
        "prompt_token_ids"
    )
    prompt_ids = (
        [token for value in prompt if (token := _token_id(value)) is not None]
        if isinstance(prompt, list)
        else None
    )
    token_ids = [
        token
        for value in choice_data.get("token_ids") or []
        if (token := _token_id(value)) is not None
    ]
    logprob_values = None
    if choice.logprobs is not None:
        logprob_values = choice.logprobs.content or choice.logprobs.refusal
    values = list(logprob_values or [])
    pair_ids, logprobs = _pairs(values)
    if token_ids and pair_ids and token_ids != pair_ids:
        raise ValueError("Response token IDs disagree with choice logprobs")
    return (
        prompt_ids,
        token_ids or pair_ids,
        logprobs or _logprob_values(values) or [math.nan] * len(token_ids),
    )


def _chat_tokens(
    response: ChatCompletion,
) -> tuple[list[int] | None, list[int], list[float]]:
    if len(response.choices) != 1:
        raise ValueError("Trajectory tokenization requires exactly one response choice")
    return _chat_choice_tokens(response.choices[0], _dump(response))


def _completion_tokens(
    response: Completion,
) -> tuple[list[int] | None, list[int], list[float]]:
    if len(response.choices) != 1:
        raise ValueError("Trajectory tokenization requires exactly one response choice")
    choice = response.choices[0]
    response_data = _dump(response)
    choice_data = _dump(choice)
    prompt = choice_data.get("prompt_token_ids") or response_data.get(
        "prompt_token_ids"
    )
    prompt_ids = (
        [token for value in prompt if (token := _token_id(value)) is not None]
        if isinstance(prompt, list)
        else None
    )
    token_ids = [
        token
        for value in choice_data.get("token_ids") or []
        if (token := _token_id(value)) is not None
    ]
    logprobs = _dump(choice.logprobs)
    tokens = logprobs.get("tokens") or []
    pair_ids = [token for value in tokens if (token := _token_id(value)) is not None]
    pair_logprobs = [
        float(value) if isinstance(value, (int, float)) else math.nan
        for value in logprobs.get("token_logprobs") or []
    ]
    if token_ids and pair_ids and token_ids != pair_ids:
        raise ValueError("Response token IDs disagree with completion logprobs")
    selected = token_ids or pair_ids
    if selected and len(pair_logprobs) != len(selected):
        pair_logprobs = [math.nan] * len(selected)
    return prompt_ids, selected, pair_logprobs


def _responses_tokens(response: Response) -> tuple[None, list[int], list[float]]:
    data = _dump(response)
    token_ids, logprobs = _pairs(data.get("raw_output_tokens"))
    if token_ids:
        return None, token_ids, logprobs
    for output in data.get("output") or []:
        for content in _dump(output).get("content") or []:
            values = _dump(content).get("logprobs")
            token_ids, logprobs = _pairs(values)
            if token_ids:
                return None, token_ids, logprobs
            if logprobs := _logprob_values(values):
                return None, [], logprobs
    return None, [], []


def _messages_tokens(response: Message) -> tuple[None, list[int], list[float]]:
    data = _dump(response)
    token_ids = [
        token
        for value in data.get("token_ids") or []
        if (token := _token_id(value)) is not None
    ]
    logprobs = [
        float(value) if isinstance(value, (int, float)) else math.nan
        for value in data.get("logprobs") or []
    ]
    if len(logprobs) != len(token_ids):
        logprobs = [math.nan] * len(token_ids)
    return None, token_ids, logprobs


def _exchange_list(trajectory: Trajectory, model: str | None) -> list[Exchange]:
    exchanges = [
        *trajectory.exchanges.chat_completions,
        *trajectory.exchanges.completions,
        *trajectory.exchanges.responses,
        *trajectory.exchanges.messages,
    ]
    if model is not None:
        exchanges = [exchange for exchange in exchanges if exchange.model == model]
        if not exchanges:
            raise ValueError(f"Trajectory contains no exchanges for model {model!r}")
    models = {exchange.model for exchange in exchanges}
    if None in models:
        raise ValueError("Every tokenized exchange must identify its model")
    if len(models) != 1:
        raise ValueError(
            "Trajectory tokenization requires exactly one model; pass model= to select one"
        )
    return sorted(
        exchanges, key=lambda exchange: (exchange.start_time, exchange.end_time)
    )


def _artifact_config(model: str) -> _TokenizerConfig:
    import wandb

    artifact_path = model.removeprefix("wandb-artifact:///")
    artifact = wandb.Api().artifact(f"{artifact_path}:latest")
    metadata = artifact.metadata
    base_model = metadata.get("base_model") or metadata.get("wandb.base_model")
    if not isinstance(base_model, str):
        raise ValueError(f"Checkpoint {model!r} does not identify its base model")
    renderer = metadata.get("renderer")
    renderer = renderer if isinstance(renderer, dict) else {}
    kwargs = renderer.get("chat_template_kwargs")
    return _TokenizerConfig(
        base_model=base_model,
        revision=(
            renderer.get("tokenizer_revision")
            if isinstance(renderer.get("tokenizer_revision"), str)
            else None
        ),
        chat_template=(
            renderer.get("chat_template")
            if isinstance(renderer.get("chat_template"), str)
            else None
        ),
        chat_template_kwargs=kwargs if isinstance(kwargs, dict) else None,
    )


def _tokenizer_config(model: str, base_model: str | None) -> _TokenizerConfig:
    if model.startswith("wandb-artifact:///"):
        config = _artifact_config(model)
        if base_model is not None:
            if base_model != config.base_model:
                config.revision = None
            config.base_model = base_model
        return config
    if base_model is not None:
        return _TokenizerConfig(base_model)
    return _TokenizerConfig(model)


def _load_tokenizer(config: _TokenizerConfig) -> _Tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer fallback requires ART's backend or tinker dependencies"
        ) from exc
    try:
        return _as_tokenizer(
            AutoTokenizer.from_pretrained(
                config.base_model,
                revision=config.revision,
            )
        )
    except Exception as exc:
        raise ValueError(
            f"Could not load tokenizer for {config.base_model!r}; pass base_model explicitly"
        ) from exc


def _ids(value: object) -> list[int]:
    if (input_ids := getattr(value, "input_ids", None)) is not None:
        value = input_ids
    if callable(to_list := getattr(value, "tolist", None)):
        value = to_list()
    if mapping := _string_dict(value):
        value = mapping.get("input_ids")
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError("Tokenizer did not return one token ID sequence")
    token_ids = [
        item for item in value if isinstance(item, int) and not isinstance(item, bool)
    ]
    if len(token_ids) != len(value):
        raise TypeError("Tokenizer did not return one token ID sequence")
    return token_ids


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text = ""
    for block in content:
        data = _string_dict(block)
        if data is not None and data.get("type") in {
            "input_text",
            "output_text",
            "text",
        }:
            value = data.get("text")
            if isinstance(value, str):
                text += value
    return text


def _anthropic_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = request.get("system")
    if system:
        messages.append({"role": "system", "content": _content_text(system)})
    for raw in request.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role", "user")
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text = ""
        reasoning = ""
        tool_calls: list[dict[str, Any]] = []
        for block in content if isinstance(content, list) else ():
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text += str(block.get("text") or "")
            elif kind == "thinking":
                reasoning += str(block.get("thinking") or "")
            elif kind == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": __import__("json").dumps(
                                block.get("input") or {}
                            ),
                        },
                    }
                )
            elif kind == "tool_result":
                if text:
                    messages.append({"role": role, "content": text})
                    text = ""
                result = block.get("content", "")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", block.get("id")),
                        "content": (
                            result if isinstance(result, str) else _content_text(result)
                        ),
                    }
                )
        message: dict[str, Any] = {"role": role, "content": text}
        if reasoning:
            message["reasoning"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        if text or reasoning or tool_calls or role == "assistant":
            messages.append(message)
    return messages


def _responses_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions := request.get("instructions"):
        messages.append({"role": "system", "content": instructions})
    value = request.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id"),
                        "content": item.get("output", ""),
                    }
                )
            elif item.get("type") == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.get("call_id"),
                                "type": "function",
                                "function": {
                                    "name": item.get("name"),
                                    "arguments": item.get("arguments", "{}"),
                                },
                            }
                        ],
                    }
                )
            elif item.get("role"):
                messages.append(
                    {
                        "role": item["role"],
                        "content": _content_text(item.get("content")),
                    }
                )
    return messages


def _openai_tools(tools: object, *, dialect: str) -> object:
    if not isinstance(tools, list) or dialect == "chat":
        return tools
    normalized = []
    for tool in tools:
        data = _string_dict(tool)
        if data is None or data.get("type", "function") != "function":
            normalized.append(tool)
            continue
        if dialect == "messages":
            function = {
                "name": data.get("name"),
                "description": data.get("description"),
                "parameters": data.get("input_schema", {}),
            }
        else:
            function = {
                "name": data.get("name"),
                "description": data.get("description"),
                "parameters": data.get("parameters", {}),
            }
        normalized.append(
            {
                "type": "function",
                "function": {
                    key: value for key, value in function.items() if value is not None
                },
            }
        )
    return normalized


def _request_messages(
    exchange: ChatCompletionsExchange | MessagesExchange | ResponsesExchange,
    messages_override: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], object]:
    request = exchange.request.root
    if isinstance(exchange, ChatCompletionsExchange):
        return _dict_list(request.get("messages")), request.get("tools")
    if isinstance(exchange, MessagesExchange):
        return _anthropic_messages(request), _openai_tools(
            request.get("tools"), dialect="messages"
        )
    if isinstance(exchange, ResponsesExchange):
        return (
            messages_override
            if messages_override is not None
            else _responses_messages(request),
            _openai_tools(request.get("tools"), dialect="responses"),
        )
    raise TypeError("Completions requests do not use chat templates")


def _response_message(
    exchange: ChatCompletionsExchange | MessagesExchange | ResponsesExchange,
) -> dict[str, Any]:
    if isinstance(exchange, ChatCompletionsExchange):
        return exchange.response.choices[0].message.model_dump(
            mode="python", exclude_none=True
        )
    if isinstance(exchange, MessagesExchange):
        data = exchange.response.model_dump(mode="python")
        request = {"messages": [{"role": "assistant", "content": data["content"]}]}
        return _anthropic_messages(request)[0]
    if isinstance(exchange, ResponsesExchange):
        data = exchange.response.model_dump(mode="python")
        content = []
        tool_calls = []
        for item in data.get("output") or []:
            if item.get("type") == "message":
                content.extend(item.get("content") or [])
            elif item.get("type") == "function_call":
                tool_calls.append(
                    {
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": _content_text(content),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message
    raise TypeError("Completions responses do not use chat templates")


def _template_ids(
    tokenizer: _Tokenizer,
    exchange: Exchange,
    *,
    completed: bool,
    config: _TokenizerConfig,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
    messages_override: list[dict[str, Any]] | None = None,
) -> list[int]:
    request = exchange.request.root
    if isinstance(exchange, CompletionsExchange):
        prompt = request.get("prompt", "")
        if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
            prompt_ids = prompt
        else:
            prompt_ids = _ids(tokenizer(str(prompt), add_special_tokens=False))
        if not completed:
            return prompt_ids
        return [
            *prompt_ids,
            *_ids(
                tokenizer(exchange.response.choices[0].text, add_special_tokens=False)
            ),
        ]

    messages, tools = _request_messages(exchange, messages_override)
    if completed:
        messages.append(_response_message(exchange))
    request_kwargs = request.get("chat_template_kwargs")
    kwargs = {
        **(config.chat_template_kwargs or {}),
        **(request_kwargs if isinstance(request_kwargs, dict) else {}),
        **(chat_template_kwargs or {}),
    }
    if isinstance(exchange, MessagesExchange) and isinstance(
        thinking := request.get("thinking"), dict
    ):
        kwargs.setdefault("enable_thinking", thinking.get("type") == "enabled")
        if budget := thinking.get("budget_tokens"):
            kwargs.setdefault("thinking_budget", budget)
    template = chat_template or request.get("chat_template") or config.chat_template
    result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=not completed,
        **({"chat_template": template} if isinstance(template, str) else {}),
        **kwargs,
    )
    return _ids(result)


def _exchange_tokens(
    exchange: Exchange,
) -> tuple[list[int] | None, list[int], list[float]]:
    if isinstance(exchange, ChatCompletionsExchange):
        return _chat_tokens(exchange.response)
    if isinstance(exchange, CompletionsExchange):
        return _completion_tokens(exchange.response)
    if isinstance(exchange, ResponsesExchange):
        return _responses_tokens(exchange.response)
    if isinstance(exchange, MessagesExchange):
        return _messages_tokens(exchange.response)
    raise TypeError(f"Unknown exchange type: {type(exchange)!r}")


def _visible_logprobs(exchange: Exchange) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    if isinstance(exchange, ChatCompletionsExchange):
        logprobs = exchange.response.choices[0].logprobs
        entries = (logprobs.content or logprobs.refusal or []) if logprobs else []
        for entry in entries:
            data = _dump(entry)
            raw_bytes = data.get("bytes")
            text = (
                bytes(raw_bytes).decode("utf-8")
                if isinstance(raw_bytes, list)
                else data.get("token")
            )
            logprob = data.get("logprob")
            if isinstance(text, str) and isinstance(logprob, (int, float)):
                values.append((text, float(logprob)))
    elif isinstance(exchange, CompletionsExchange):
        logprobs = exchange.response.choices[0].logprobs
        if logprobs is not None:
            for text, logprob in zip(
                logprobs.tokens or [], logprobs.token_logprobs or [], strict=False
            ):
                if logprob is not None:
                    values.append((text, float(logprob)))
    elif isinstance(exchange, ResponsesExchange):
        for output in _dump(exchange.response).get("output") or []:
            for content in _dump(output).get("content") or []:
                for entry in _dump(content).get("logprobs") or []:
                    data = _dump(entry)
                    text = data.get("token")
                    logprob = data.get("logprob")
                    if isinstance(text, str) and isinstance(logprob, (int, float)):
                        values.append((text, float(logprob)))
    return values


def _align_visible_logprobs(
    tokenizer: _Tokenizer | None, completion: list[int], exchange: Exchange
) -> list[float] | None:
    values = _visible_logprobs(exchange)
    if not values or tokenizer is None:
        return None
    aligned = [math.nan] * len(completion)
    cursor = 0
    for text, logprob in values:
        encoded = _ids(tokenizer(text, add_special_tokens=False))
        if len(encoded) != 1:
            return None
        try:
            index = completion.index(encoded[0], cursor)
        except ValueError:
            return None
        aligned[index] = logprob
        cursor = index + 1
    return aligned


def _legacy_tokenize(
    trajectory: Trajectory,
    base_model: str | None,
    *,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
) -> TokenizedTrajectory:
    if trajectory.additional_histories:
        raise ValueError("Tokenization requires one history")
    token_ids: list[int] = []
    logprobs: list[float] = []
    assistant_mask: list[bool] = []
    for item in trajectory.messages_and_choices:
        if not isinstance(item, Choice):
            continue
        prompt, completion, completion_logprobs = _chat_choice_tokens(item, {})
        if prompt is None or not completion:
            raise ValueError(
                "Legacy fallback tokenization is unavailable without exact choice token metadata"
            )
        if not token_ids:
            token_ids.extend(prompt)
            logprobs.extend([math.nan] * len(prompt))
            assistant_mask.extend([False] * len(prompt))
        elif prompt[: len(token_ids)] != token_ids:
            raise ValueError("Legacy trajectory does not form one append-only history")
        else:
            suffix = prompt[len(token_ids) :]
            token_ids.extend(suffix)
            logprobs.extend([math.nan] * len(suffix))
            assistant_mask.extend([False] * len(suffix))
        token_ids.extend(completion)
        logprobs.extend(completion_logprobs)
        assistant_mask.extend([True] * len(completion))
    if not token_ids:
        raise ValueError("Trajectory contains no trainable choices")
    return TokenizedTrajectory(
        token_ids=token_ids,
        logprobs=logprobs,
        assistant_mask=assistant_mask,
        underlying=trajectory,
    )


def tokenize_one(
    trajectory: Trajectory,
    base_model: str | None,
    *,
    model: str | None,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
    tokenizer_instance: _Tokenizer | None = None,
) -> TokenizedTrajectory:
    if not trajectory.exchanges:
        return _legacy_tokenize(
            trajectory,
            base_model,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
    exchanges = _exchange_list(trajectory, model)
    selected_model = exchanges[0].model
    if selected_model is None:
        raise AssertionError("_exchange_list returned an exchange without a model")
    config = _tokenizer_config(selected_model, base_model)
    tokenizer = tokenizer_instance
    token_ids: list[int] = []
    logprobs: list[float] = []
    assistant_mask: list[bool] = []
    response_histories: dict[str, list[dict[str, Any]]] = {}

    for exchange in exchanges:
        messages_override = None
        if isinstance(exchange, ResponsesExchange):
            request = exchange.request.root
            messages_override = _responses_messages(request)
            previous = request.get("previous_response_id")
            if previous is not None:
                if not isinstance(previous, str) or previous not in response_histories:
                    raise ValueError(
                        "Responses exchange refers to a previous response outside this trajectory"
                    )
                messages_override = [
                    *response_histories[previous],
                    *messages_override,
                ]
            response_histories[exchange.response.id] = [
                *messages_override,
                _response_message(exchange),
            ]
        prompt, completion, completion_logprobs = _exchange_tokens(exchange)
        if prompt is None:
            if tokenizer is None:
                tokenizer = _load_tokenizer(config)
            prompt = _template_ids(
                tokenizer,
                exchange,
                completed=False,
                config=config,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                messages_override=messages_override,
            )
        if not completion:
            if tokenizer is None:
                tokenizer = _load_tokenizer(config)
            completed = _template_ids(
                tokenizer,
                exchange,
                completed=True,
                config=config,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                messages_override=messages_override,
            )
            if completed[: len(prompt)] != prompt:
                raise ValueError(
                    "Completed response does not extend its generation prompt"
                )
            completion = completed[len(prompt) :]
            completion_logprobs = _align_visible_logprobs(
                tokenizer, completion, exchange
            ) or [math.nan] * len(completion)
        if not token_ids:
            token_ids.extend(prompt)
            logprobs.extend([math.nan] * len(prompt))
            assistant_mask.extend([False] * len(prompt))
        elif len(prompt) < len(token_ids) or prompt[: len(token_ids)] != token_ids:
            raise ValueError(
                "Exchanges do not resolve to one append-only token history"
            )
        else:
            suffix = prompt[len(token_ids) :]
            token_ids.extend(suffix)
            logprobs.extend([math.nan] * len(suffix))
            assistant_mask.extend([False] * len(suffix))
        if len(completion_logprobs) != len(completion):
            completion_logprobs = _align_visible_logprobs(
                tokenizer, completion, exchange
            ) or [math.nan] * len(completion)
        token_ids.extend(completion)
        logprobs.extend(completion_logprobs)
        assistant_mask.extend([True] * len(completion))

    return TokenizedTrajectory(
        token_ids=token_ids,
        logprobs=logprobs,
        assistant_mask=assistant_mask,
        underlying=trajectory,
    )
