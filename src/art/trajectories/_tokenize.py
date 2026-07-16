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


def _as_tokenizer(tokenizer: object) -> _Tokenizer:
    # Transformers' annotation permits only string-valued message dictionaries,
    # although its runtime API supports the structured content ART must tokenize.
    # Exact-token paths may only need decode(); fallback paths exercise these
    # capabilities directly and report the missing method at that point.
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


def _exact_token_ids(values: object, *, field: str) -> list[int] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"{field} exact token metadata must be a list")
    token_ids: list[int] = []
    for value in values:
        token_id = _token_id(value)
        if token_id is None:
            raise ValueError(f"{field} contains an invalid exact token ID")
        token_ids.append(token_id)
    return token_ids


def _pair_token_id(data: dict[str, Any], *, required: bool, field: str) -> int | None:
    if "token_id" in data:
        token_id = _token_id(data["token_id"])
        if token_id is None:
            raise ValueError(f"{field} contains an invalid exact token ID")
        return token_id
    raw_token = data.get("token")
    token_id = _token_id(raw_token)
    if token_id is not None:
        return token_id
    if isinstance(raw_token, str) and raw_token.startswith("token_id:"):
        raise ValueError(f"{field} contains an invalid exact token ID")
    if required:
        raise ValueError(f"{field} is missing an exact token ID")
    return None


def _pairs(
    values: object, *, require_token_ids: bool = False, field: str = "token pairs"
) -> tuple[list[int], list[float]]:
    if not isinstance(values, list):
        if require_token_ids:
            raise ValueError(f"{field} exact token metadata must be a list")
        return [], []
    token_ids: list[int] = []
    logprobs: list[float] = []
    complete = True
    for value in values:
        data = _dump(value)
        token_id = _pair_token_id(data, required=require_token_ids, field=field)
        if token_id is None:
            complete = False
            continue
        logprob = data.get("logprob")
        token_ids.append(token_id)
        logprobs.append(
            float(logprob)
            if isinstance(logprob, (int, float)) and not isinstance(logprob, bool)
            else math.nan
        )
    return (token_ids, logprobs) if complete else ([], [])


def _logprob_values(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for value in values:
        logprob = _dump(value).get("logprob")
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            return []
        result.append(float(logprob))
    return result


def _chat_choice_tokens(
    choice: Choice, response_data: dict[str, Any]
) -> tuple[list[int] | None, list[int], list[float]]:
    choice_data = _dump(choice)
    prompt = choice_data.get("prompt_token_ids")
    if prompt is None:
        prompt = response_data.get("prompt_token_ids")
    prompt_ids = _exact_token_ids(prompt, field="Chat Completions prompt_token_ids")
    token_ids = _exact_token_ids(
        choice_data.get("token_ids"),
        field="Chat Completions token_ids",
    )
    logprob_values = None
    if choice.logprobs is not None:
        logprob_values = choice.logprobs.content or choice.logprobs.refusal
    values = list(logprob_values or [])
    pair_ids, logprobs = _pairs(values, field="Chat Completions logprobs")
    token_ids = token_ids or []
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
    prompt = choice_data.get("prompt_token_ids")
    if prompt is None:
        prompt = response_data.get("prompt_token_ids")
    prompt_ids = _exact_token_ids(prompt, field="Completions prompt_token_ids")
    token_ids = _exact_token_ids(
        choice_data.get("token_ids"), field="Completions token_ids"
    )
    token_ids = token_ids or []
    logprobs = _dump(choice.logprobs)
    tokens = logprobs.get("tokens") or []
    pair_ids: list[int] = []
    complete_pairs = True
    for value in tokens:
        token = _token_id(value)
        if token is None:
            if isinstance(value, str) and value.startswith("token_id:"):
                raise ValueError(
                    "Completions logprobs contain an invalid exact token ID"
                )
            complete_pairs = False
        else:
            pair_ids.append(token)
    if not complete_pairs:
        pair_ids = []
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
    if "raw_output_tokens" in data:
        token_ids, logprobs = _pairs(
            data["raw_output_tokens"],
            require_token_ids=True,
            field="Responses raw_output_tokens",
        )
        return None, token_ids, logprobs
    token_ids: list[int] = []
    logprobs: list[float] = []
    saw_rendered_output = False
    complete = True
    for output in data.get("output") or []:
        output_data = _dump(output)
        if output_data.get("type") != "message":
            complete = False
            continue
        for content in output_data.get("content") or []:
            content_data = _dump(content)
            text = content_data.get("text") or content_data.get("refusal")
            if not isinstance(text, str) or not text:
                continue
            saw_rendered_output = True
            pair_ids, pair_logprobs = _pairs(
                content_data.get("logprobs"), field="Responses content logprobs"
            )
            if not pair_ids:
                complete = False
                continue
            token_ids.extend(pair_ids)
            logprobs.extend(pair_logprobs)
    if saw_rendered_output and complete:
        return None, token_ids, logprobs
    return None, [], []


def _messages_tokens(response: Message) -> tuple[None, list[int], list[float]]:
    data = _dump(response)
    token_ids = (
        _exact_token_ids(data.get("token_ids"), field="Messages token_ids") or []
    )
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
    from wandb.apis.public import Api

    artifact_path = model.removeprefix("wandb-artifact:///")
    artifact = Api().artifact(f"{artifact_path}:latest")
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
            else:
                raise ValueError(f"Unsupported Anthropic content block type: {kind!r}")
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
    instructions = request.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValueError("Responses instructions must be text")
    if instructions:
        messages.append({"role": "system", "content": instructions})
    value = request.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
    elif isinstance(value, list):
        pending_reasoning = ""
        pending_tool_calls: list[dict[str, Any]] | None = None
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Responses input items must be JSON objects")
            kind = item.get("type")
            if kind == "reasoning":
                pending_tool_calls = None
                reasoning = _responses_reasoning_text(item)
                if not reasoning:
                    raise ValueError("Responses reasoning item has no renderable text")
                pending_reasoning += reasoning
                continue
            if kind == "function_call":
                if pending_tool_calls is None:
                    pending_tool_calls = []
                    message: dict[str, Any] = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": pending_tool_calls,
                    }
                    if pending_reasoning:
                        message["reasoning"] = pending_reasoning
                        pending_reasoning = ""
                    messages.append(message)
                pending_tool_calls.append(
                    {
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                )
                continue
            pending_tool_calls = None
            if kind == "function_call_output":
                message: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": _responses_input_text(
                        item.get("output", ""), field="function_call_output"
                    ),
                }
            elif kind in {None, "message"} and item.get("role"):
                if item.get("phase") is not None:
                    raise ValueError("Unsupported Responses message phase")
                message = {
                    "role": item["role"],
                    "content": _responses_input_text(
                        item.get("content"), field="message content"
                    ),
                }
            else:
                raise ValueError(f"Unsupported Responses input item type: {kind!r}")
            if pending_reasoning:
                if message["role"] == "assistant":
                    message["reasoning"] = pending_reasoning
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning": pending_reasoning,
                        }
                    )
                pending_reasoning = ""
            messages.append(message)
        if pending_reasoning:
            messages.append(
                {"role": "assistant", "content": "", "reasoning": pending_reasoning}
            )
    elif value is not None:
        raise ValueError("Responses input must be text or a list of input items")
    return messages


def _responses_input_text(content: object, *, field: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError(f"Responses {field} must contain text")
    text = ""
    for block in content:
        data = _string_dict(block)
        if data is None:
            raise ValueError(f"Responses {field} blocks must be JSON objects")
        kind = data.get("type")
        if kind not in {"input_text", "output_text", "refusal", "text"}:
            raise ValueError(f"Unsupported Responses content block type: {kind!r}")
        value = data.get("refusal" if kind == "refusal" else "text")
        if not isinstance(value, str):
            raise ValueError(f"Responses {field} blocks must contain text")
        text += value
    return text


def _responses_output_text(content: object) -> str:
    if not isinstance(content, list):
        raise ValueError("Responses message output content must be a list")
    text = ""
    for block in content:
        data = _string_dict(block)
        if data is None:
            raise ValueError("Responses output content blocks must be JSON objects")
        kind = data.get("type")
        key = (
            "text"
            if kind == "output_text"
            else "refusal"
            if kind == "refusal"
            else None
        )
        if key is None:
            raise ValueError(f"Unsupported Responses output content type: {kind!r}")
        value = data.get(key)
        if not isinstance(value, str):
            raise ValueError(f"Responses {kind} content must be text")
        text += value
    return text


def _responses_reasoning_text(item: Mapping[str, object]) -> str:
    text = ""
    for field in ("content", "summary"):
        blocks = item.get(field)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            data = _string_dict(block)
            if data is not None and isinstance(data.get("text"), str):
                text += data["text"]
    return text


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
    request = exchange.request
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
        content = ""
        reasoning = ""
        tool_calls = []
        for raw_item in data.get("output") or []:
            item = _string_dict(raw_item)
            if item is None:
                raise ValueError("Responses output items must be JSON objects")
            kind = item.get("type")
            if kind == "message":
                if item.get("phase") is not None:
                    raise ValueError("Unsupported Responses message phase")
                content += _responses_output_text(item.get("content"))
            elif kind == "reasoning":
                rendered = _responses_reasoning_text(item)
                if not rendered:
                    raise ValueError("Responses reasoning item has no renderable text")
                reasoning += rendered
            elif kind == "function_call":
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
            else:
                raise ValueError(f"Unsupported Responses output item type: {kind!r}")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if reasoning:
            message["reasoning"] = reasoning
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
    request = exchange.request
    if isinstance(exchange, CompletionsExchange):
        prompt = request.get("prompt", "")
        if isinstance(prompt, list) and all(isinstance(item, int) for item in prompt):
            prompt_ids = _ids(prompt)
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
        messages = [*messages, _response_message(exchange)]
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
            if isinstance(raw_bytes, list):
                try:
                    text = bytes(raw_bytes).decode("utf-8")
                except (TypeError, ValueError, UnicodeDecodeError):
                    return []
            else:
                text = data.get("token")
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
    token_ids: list[int] = []
    logprobs: list[float] = []
    for text, logprob in values:
        encoded = _ids(tokenizer(text, add_special_tokens=False))
        if len(encoded) != 1:
            return None
        token_ids.append(encoded[0])
        logprobs.append(logprob)

    left: list[int] = []
    cursor = 0
    for token_id in token_ids:
        try:
            index = completion.index(token_id, cursor)
        except ValueError:
            return None
        left.append(index)
        cursor = index + 1

    right: list[int] = []
    cursor = len(completion)
    for token_id in reversed(token_ids):
        while cursor:
            cursor -= 1
            if completion[cursor] == token_id:
                right.append(cursor)
                break
        else:
            return None
    right.reverse()
    if left != right:
        return None

    aligned = [math.nan] * len(completion)
    for index, logprob in zip(left, logprobs, strict=True):
        aligned[index] = logprob
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
    sampled_spans: list[tuple[int, int]] = []
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
        start = len(token_ids)
        token_ids.extend(completion)
        sampled_spans.append((start, len(token_ids)))
        if len(completion_logprobs) != len(completion):
            completion_logprobs = [math.nan] * len(completion)
        logprobs.extend(completion_logprobs)
        assistant_mask.extend([True] * len(completion))
    if not token_ids:
        raise ValueError("Trajectory contains no trainable choices")
    return TokenizedTrajectory(
        token_ids=token_ids,
        logprobs=logprobs,
        assistant_mask=assistant_mask,
        sampled_spans=sampled_spans,
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
    if trajectory.exchanges and (
        trajectory.messages_and_choices
        or trajectory.tools is not None
        or trajectory.additional_histories
    ):
        raise ValueError(
            "A trajectory cannot contain both exchanges and legacy histories"
        )
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
    sampled_spans: list[tuple[int, int]] = []
    response_histories: dict[
        str, tuple[list[dict[str, Any]] | None, ResponsesExchange]
    ] = {}

    for exchange in exchanges:
        if isinstance(exchange, CompletionsExchange):
            prompt = exchange.request.get("prompt")
            if isinstance(prompt, list) and not all(
                isinstance(item, int) and not isinstance(item, bool) for item in prompt
            ):
                raise ValueError(
                    "Trajectory tokenization does not support batched Completions prompts"
                )
            if not isinstance(prompt, (str, list)):
                raise ValueError("Completions prompt must be text or one token ID list")
            if exchange.request.get("echo") is True:
                raise ValueError(
                    "Trajectory tokenization does not support Completions echo=True"
                )
        prompt, completion, completion_logprobs = _exchange_tokens(exchange)
        messages_override: list[dict[str, Any]] | None = None
        if isinstance(exchange, ResponsesExchange):
            request = exchange.request
            try:
                messages_override = _responses_messages(request)
            except ValueError:
                if prompt is None:
                    raise
            previous = request.get("previous_response_id")
            if previous is not None:
                if not isinstance(previous, str) or previous not in response_histories:
                    raise ValueError(
                        "Responses exchange refers to a previous response outside this trajectory"
                    )
                previous_messages, previous_exchange = response_histories[previous]
                if prompt is None:
                    if previous_messages is None or messages_override is None:
                        raise ValueError(
                            "Responses history cannot be rendered without exact prompt tokens"
                        )
                    messages_override = [
                        *previous_messages,
                        _response_message(previous_exchange),
                        *messages_override,
                    ]
            response_histories[exchange.response.id] = (messages_override, exchange)
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
        start = len(token_ids)
        token_ids.extend(completion)
        sampled_spans.append((start, len(token_ids)))
        logprobs.extend(completion_logprobs)
        assistant_mask.extend([True] * len(completion))

    return TokenizedTrajectory(
        token_ids=token_ids,
        logprobs=logprobs,
        assistant_mask=assistant_mask,
        sampled_spans=sampled_spans,
        underlying=trajectory,
    )
