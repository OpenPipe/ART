from __future__ import annotations

from bisect import bisect_left
import codecs
from collections.abc import Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from hashlib import sha256
import json
import math
import re
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast
import warnings

from anthropic.types import Message, MessageParam, TextBlock
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.responses import Response
from pydantic import BaseModel

from ..utils.chat_template import (
    chat_template_with_preserved_thinking,
    configure_preserved_thinking_chat_template,
    normalize_tool_call_arguments_for_chat_template,
)
from . import (
    AnthropicMessagesHistory,
    ChatCompletionsExchange,
    ChatCompletionsHistory,
    CompletionsExchange,
    CompletionsSource,
    CompletionsStringHistory,
    CompletionsStringSourceSpan,
    CompletionsTokenHistory,
    CompletionsTokenSourceSpan,
    History,
    LegacyHistory,
    MessagesExchange,
    ResponsesExchange,
    ResponsesHistory,
    TokenFlag,
    TokenizedHistory,
    TokenizedMultiHistoryTrajectory,
    TokenizedTrajectory,
    TokenizedTrajectoryGroup,
    Tokenizer,
    Trajectory,
    TrajectoryGroup,
    TrajectoryHistory,
)
from ._history import _model_matches
from ._protocols import Exchange

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_TOKEN_ID = re.compile(r"token_id:(\d+)$")
_WARNED_PREFIX_RETOKENIZATION = False


@dataclass
class _TokenizerConfig:
    base_model: str
    revision: str | None = None
    chat_template: str | None = None
    chat_template_kwargs: Mapping[str, object] | None = None


class _OffsetTokenizer(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: Literal[True],
    ) -> object: ...


@dataclass(frozen=True)
class _SampledOutput:
    text: str | None
    token_ids: list[int]
    start: int


@dataclass(frozen=True)
class _SampledSourceKey:
    protocol: Literal["chat_completions", "responses", "messages", "completions"]
    response_id: str
    start_time: datetime
    end_time: datetime
    index: int
    prompt_index: int | None
    evidence_fingerprint: str


@dataclass
class _HistoryTokenizationTrace:
    source_keys: list[_SampledSourceKey | None]
    sources: dict[_SampledSourceKey, object]

    def validate(self, tokenized: TokenizedHistory) -> None:
        if len(self.source_keys) != len(tokenized.tokens):
            raise AssertionError(
                "Tokenization trace differs in length from tokenized data"
            )
        for flag, source_key in zip(tokenized.flags, self.source_keys, strict=True):
            if bool(flag & TokenFlag.SAMPLED) != (source_key is not None):
                raise AssertionError(
                    "Tokenization trace must identify every sampled token exactly once"
                )
            if source_key is not None and source_key not in self.sources:
                raise AssertionError("Tokenization trace source key is unresolved")


_SourceKeyT = TypeVar("_SourceKeyT", bound=Hashable)


def _first_introduction_mask(
    source_keys: Sequence[_SourceKeyT | None],
    seen: set[_SourceKeyT],
) -> list[bool]:
    keys = {key for key in source_keys if key is not None}
    new_keys = keys - seen
    seen.update(keys)
    return [key in new_keys if key is not None else False for key in source_keys]


def _require_causal_predecessor(trainable: Sequence[bool]) -> None:
    if trainable and trainable[0]:
        raise ValueError("A trainable trajectory cannot start with a sampled token")


@dataclass
class _TraceBuilder:
    trace: _HistoryTokenizationTrace | None = None

    def set(
        self,
        tokenized: TokenizedHistory,
        source_keys: list[_SampledSourceKey | None],
        sources: dict[_SampledSourceKey, object],
    ) -> None:
        trace = _HistoryTokenizationTrace(source_keys=source_keys, sources=sources)
        trace.validate(tokenized)
        self.trace = trace


def _fingerprint(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(serialized.encode()).hexdigest()


def _chat_logprob_fingerprint_evidence(choice: Choice) -> dict[str, object] | None:
    if choice.logprobs is None:
        return None

    def values(items: Sequence[object] | None) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for item in items or []:
            data = _dump(item)
            result.append(
                {
                    key: data[key]
                    for key in ("token", "token_id", "logprob", "bytes")
                    if key in data
                }
            )
        return result

    return {
        "content": values(choice.logprobs.content),
        "refusal": values(choice.logprobs.refusal),
    }


def _sampled_evidence_fingerprint(
    exchange: Exchange,
    *,
    protocol: Literal["chat_completions", "responses", "messages", "completions"],
    index: int,
) -> str:
    if protocol == "chat_completions":
        if not isinstance(exchange, ChatCompletionsExchange):
            raise TypeError("Chat source has the wrong exchange type")
        choice = next(
            choice for choice in exchange.response.choices if choice.index == index
        )
        choice_extra = choice.model_extra or {}
        evidence = {
            "message": choice.message.model_dump(
                mode="json",
                include={
                    "role",
                    "content",
                    "refusal",
                    "reasoning",
                    "reasoning_content",
                    "tool_calls",
                    "function_call",
                    "audio",
                },
                exclude_none=True,
            ),
            "token_ids": choice_extra.get("token_ids"),
            "logprobs": _chat_logprob_fingerprint_evidence(choice),
            "finish_reason": choice.finish_reason,
        }
    elif protocol == "completions":
        if not isinstance(exchange, CompletionsExchange):
            raise TypeError("Completions source has the wrong exchange type")
        choice = next(
            choice for choice in exchange.response.choices if choice.index == index
        )
        choice_extra = choice.model_extra or {}
        logprobs = _dump(choice.logprobs)
        evidence = {
            "text": choice.text,
            "token_ids": choice_extra.get("token_ids"),
            "logprobs": {
                key: logprobs[key]
                for key in ("tokens", "token_logprobs")
                if key in logprobs
            },
            "finish_reason": choice.finish_reason,
        }
    elif protocol == "responses":
        if not isinstance(exchange, ResponsesExchange):
            raise TypeError("Responses source has the wrong exchange type")
        generations = (exchange.response.model_extra or {}).get("token_generations")
        if isinstance(generations, list) and 0 <= index < len(generations):
            generation = _string_dict(generations[index]) or {}
            evidence = {
                key: generation[key]
                for key in ("output_tokens", "output_indices")
                if key in generation
            }
        else:
            evidence = [
                output.model_dump(mode="json", exclude_none=True)
                for output in exchange.response.output
            ]
    else:
        if not isinstance(exchange, MessagesExchange):
            raise TypeError("Messages source has the wrong exchange type")
        response_extra = exchange.response.model_extra or {}
        evidence = {
            "content": [
                block.model_dump(mode="json", exclude_none=True)
                for block in exchange.response.content
            ],
            "token_ids": response_extra.get("token_ids"),
            "logprobs": response_extra.get("logprobs"),
            "stop_reason": exchange.response.stop_reason,
        }
    return _fingerprint(evidence)


def _source_key(
    exchange: Exchange,
    *,
    protocol: Literal["chat_completions", "responses", "messages", "completions"],
    index: int,
    prompt_index: int | None = None,
) -> _SampledSourceKey:
    return _SampledSourceKey(
        protocol=protocol,
        response_id=str(getattr(exchange.response, "id", "")),
        start_time=exchange.start_time,
        end_time=exchange.end_time,
        index=index,
        prompt_index=prompt_index,
        # Internal projection may copy an exchange to isolate one choice. A
        # source-specific identity remains stable across those copies without
        # hashing a growing request or unrelated choices.
        evidence_fingerprint=_sampled_evidence_fingerprint(
            exchange, protocol=protocol, index=index
        ),
    )


def _sampled_source_key(source: object) -> _SampledSourceKey:
    exchange = getattr(source, "exchange", None)
    if isinstance(exchange, ChatCompletionsExchange):
        index = getattr(source, "choice_index", None)
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Sampled Chat source has no choice index")
        return _source_key(exchange, protocol="chat_completions", index=index)
    if isinstance(exchange, ResponsesExchange):
        index = getattr(source, "generation_index", None)
        if index is None and not _response_generations(exchange.response):
            index = 0
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Sampled Responses source has no generation identity")
        return _source_key(exchange, protocol="responses", index=index)
    if isinstance(exchange, MessagesExchange):
        return _source_key(exchange, protocol="messages", index=0)
    if isinstance(exchange, CompletionsExchange):
        index = getattr(source, "choice_index", None)
        prompt_index = getattr(source, "prompt_index", None)
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Sampled Completions source has no choice index")
        if not isinstance(prompt_index, int) or isinstance(prompt_index, bool):
            raise ValueError("Sampled Completions source has no prompt index")
        return _source_key(
            exchange,
            protocol="completions",
            index=index,
            prompt_index=prompt_index,
        )
    raise ValueError("Sampled token source has an unsupported exchange")


def _exchange_sampled_source_key(exchange: Exchange) -> _SampledSourceKey:
    if isinstance(exchange, ChatCompletionsExchange):
        return _source_key(
            exchange,
            protocol="chat_completions",
            index=exchange.response.choices[0].index,
        )
    if isinstance(exchange, CompletionsExchange):
        return _source_key(
            exchange,
            protocol="completions",
            index=exchange.response.choices[0].index,
            prompt_index=0,
        )
    if isinstance(exchange, ResponsesExchange):
        return _source_key(exchange, protocol="responses", index=0)
    if isinstance(exchange, MessagesExchange):
        return _source_key(exchange, protocol="messages", index=0)
    raise TypeError(f"Unsupported sampled exchange: {type(exchange).__name__}")


def _as_tokenizer(tokenizer: object) -> Tokenizer:
    # Transformers' annotation permits only string-valued message dictionaries,
    # although its runtime API supports the structured content ART must tokenize.
    # Exact-token paths may only need decode(); fallback paths exercise these
    # capabilities directly and report the missing method at that point.
    return cast(Tokenizer, tokenizer)


def _string_dict(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
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


def _field(value: object, name: str, default: object = None) -> object:
    return (
        value.get(name, default)
        if isinstance(value, Mapping)
        else getattr(value, name, default)
    )


def _token_id(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and (match := _TOKEN_ID.fullmatch(value)):
        return int(match.group(1))
    return None


def _exact_token_ids(
    values: object, *, field: str, empty_is_missing: bool = False
) -> list[int] | None:
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
    return None if empty_is_missing and not token_ids else token_ids


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


def _chat_logprob_entries(choice: Choice) -> list[object]:
    if choice.logprobs is None:
        return []
    return [
        *(choice.logprobs.content or []),
        *(choice.logprobs.refusal or []),
    ]


def _chat_choice_output_tokens(
    choice: Choice,
) -> tuple[list[int] | None, list[float]]:
    choice_data = _dump(choice)
    token_ids = _exact_token_ids(
        choice_data.get("token_ids"),
        field="Chat Completions token_ids",
    )
    values = _chat_logprob_entries(choice)
    message = _dump(choice.message)
    if token_ids == [] and (
        values
        or any(
            message.get(key)
            for key in (
                "content",
                "refusal",
                "reasoning",
                "reasoning_content",
                "tool_calls",
                "function_call",
            )
        )
    ):
        token_ids = None
    pair_ids, pair_logprobs = _pairs(values, field="Chat Completions logprobs")
    positional_logprobs = _logprob_values(values)
    if token_ids is not None and pair_ids and token_ids != pair_ids:
        raise ValueError("Response token IDs disagree with choice logprobs")
    selected = token_ids if token_ids is not None else pair_ids or None
    logprobs = pair_logprobs or positional_logprobs
    if selected is not None and values and len(logprobs) != len(selected):
        raise ValueError("Chat Completions token IDs and logprobs differ in length")
    return selected, logprobs or [math.nan] * len(selected or [])


def _chat_choice_tokens(
    choice: Choice, response_data: dict[str, Any]
) -> tuple[list[int] | None, list[int] | None, list[float]]:
    choice_data = _dump(choice)
    prompt = choice_data.get("prompt_token_ids")
    if prompt is None:
        prompt = response_data.get("prompt_token_ids")
    prompt_ids = _exact_token_ids(
        prompt,
        field="Chat Completions prompt_token_ids",
        empty_is_missing=True,
    )
    selected, logprobs = _chat_choice_output_tokens(choice)
    return (
        prompt_ids,
        selected,
        logprobs,
    )


def _chat_tokens(
    response: ChatCompletion,
) -> tuple[list[int] | None, list[int] | None, list[float]]:
    if len(response.choices) != 1:
        raise ValueError("Trajectory tokenization requires exactly one response choice")
    return _chat_choice_tokens(response.choices[0], _dump(response))


def _completion_evidence(
    response: Completion,
    *,
    echo: bool = False,
    empty_prompt_is_exact: bool = False,
) -> tuple[list[int] | None, list[int] | None, list[float], list[float]]:
    if len(response.choices) != 1:
        raise ValueError("Trajectory tokenization requires exactly one response choice")
    choice = response.choices[0]
    response_data = _dump(response)
    choice_data = _dump(choice)
    prompt = choice_data.get("prompt_token_ids")
    if prompt is None:
        prompt = response_data.get("prompt_token_ids")
    prompt_ids = _exact_token_ids(
        prompt,
        field="Completions prompt_token_ids",
        empty_is_missing=not empty_prompt_is_exact,
    )
    token_ids = _exact_token_ids(
        choice_data.get("token_ids"), field="Completions token_ids"
    )
    logprobs = _dump(choice.logprobs)
    tokens = logprobs.get("tokens") or []
    if token_ids == [] and (choice.text or tokens):
        token_ids = None
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
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else math.nan
        for value in logprobs.get("token_logprobs") or []
    ]
    pair_includes_prompt = (
        echo
        and prompt_ids is not None
        and token_ids is not None
        and pair_ids == [*prompt_ids, *token_ids]
    )
    token_ids_include_prompt = (
        echo
        and prompt_ids is not None
        and token_ids is not None
        and bool(pair_ids)
        and token_ids == [*prompt_ids, *pair_ids]
    )
    if token_ids is not None and pair_ids and token_ids != pair_ids:
        if not pair_includes_prompt and not token_ids_include_prompt:
            raise ValueError("Response token IDs disagree with completion logprobs")
    selected = token_ids if token_ids is not None else pair_ids or None
    prompt_logprobs: list[float] = []
    completion_logprobs = pair_logprobs
    if echo and prompt_ids is None and token_ids is None:
        # A combined logprobs.tokens carrier does not reveal where an echoed
        # prompt ends. Let the string history tokenize prompt and completion
        # independently rather than mislabeling the prompt as sampled output.
        selected = None
        completion_logprobs = []
    elif echo and prompt_ids is not None and selected is not None:
        if pair_includes_prompt:
            prompt_logprobs = pair_logprobs[: len(prompt_ids)]
            completion_logprobs = pair_logprobs[len(prompt_ids) :]
        elif len(pair_logprobs) == len(prompt_ids) + len(selected):
            prompt_logprobs = pair_logprobs[: len(prompt_ids)]
            completion_logprobs = pair_logprobs[len(prompt_ids) :]
        elif token_ids_include_prompt:
            selected = selected[len(prompt_ids) :]
            completion_logprobs = pair_logprobs
        elif selected[: len(prompt_ids)] == prompt_ids and (
            len(tokens) == len(selected)
            and all(isinstance(value, str) for value in tokens)
            and "".join(cast(str, value) for value in tokens) == choice.text
        ):
            if pair_logprobs and len(pair_logprobs) != len(selected):
                raise ValueError("Completions token IDs and logprobs differ in length")
            prompt_logprobs = pair_logprobs[: len(prompt_ids)]
            selected = selected[len(prompt_ids) :]
            completion_logprobs = pair_logprobs[len(prompt_ids) :]
        elif selected[: len(prompt_ids)] == prompt_ids:
            selected = None
            completion_logprobs = []
        elif pair_logprobs and len(pair_logprobs) != len(selected):
            raise ValueError("Completions token IDs and logprobs differ in length")
    elif selected is not None and pair_logprobs and len(pair_logprobs) != len(selected):
        raise ValueError("Completions token IDs and logprobs differ in length")
    if selected is not None and not completion_logprobs:
        completion_logprobs = [math.nan] * len(selected)
    return prompt_ids, selected, prompt_logprobs, completion_logprobs


def _completion_tokens(
    response: Completion,
    *,
    echo: bool = False,
    empty_prompt_is_exact: bool = False,
) -> tuple[list[int] | None, list[int] | None, list[float]]:
    prompt, completion, _, completion_logprobs = _completion_evidence(
        response,
        echo=echo,
        empty_prompt_is_exact=empty_prompt_is_exact,
    )
    return prompt, completion, completion_logprobs


@dataclass(frozen=True)
class _ResponseGeneration:
    prompt_token_ids: list[int] | None
    output_token_ids: list[int] | None
    output_logprobs: list[float]
    output_indices: list[int]
    output_text: str | None


def _responses_output_is_sampled(item: object) -> bool:
    return not str(_field(item, "type", "")).endswith("_output")


def _response_generations(response: Response) -> list[_ResponseGeneration]:
    data = _dump(response)
    raw_generations = data.get("token_generations")
    if raw_generations is None:
        return []
    if not isinstance(raw_generations, list):
        raise ValueError("Responses token_generations exact metadata must be a list")
    if not raw_generations:
        raise ValueError(
            "Responses token_generations must be omitted when exact evidence "
            "is unavailable"
        )
    generations: list[_ResponseGeneration] = []
    assigned_outputs: set[int] = set()
    last_output_index = -1
    for index, raw_generation in enumerate(raw_generations):
        generation = _string_dict(raw_generation)
        if generation is None:
            raise ValueError(
                f"Responses token_generations[{index}] must be a JSON object"
            )
        prompt = _exact_token_ids(
            generation.get("prompt_token_ids"),
            field=f"Responses token_generations[{index}].prompt_token_ids",
            empty_is_missing=True,
        )
        if prompt is None:
            raise ValueError(
                f"Responses token_generations[{index}].prompt_token_ids must "
                "contain the full non-empty generation prompt"
            )
        raw_indices = generation.get("output_indices")
        if not isinstance(raw_indices, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in raw_indices
        ):
            raise ValueError(
                f"Responses token_generations[{index}].output_indices must be "
                "a list of non-negative integers"
            )
        output_indices = [
            value
            for value in raw_indices
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        if len(set(output_indices)) != len(output_indices):
            raise ValueError(
                f"Responses token_generations[{index}].output_indices contains duplicates"
            )
        if output_indices != sorted(output_indices):
            raise ValueError(
                f"Responses token_generations[{index}].output_indices must be ordered"
            )
        if output_indices and output_indices[0] <= last_output_index:
            raise ValueError(
                "Responses token_generations output_indices must be ordered "
                "and nonoverlapping"
            )
        if output_indices:
            last_output_index = output_indices[-1]
        if any(value >= len(response.output) for value in output_indices):
            raise ValueError(
                f"Responses token_generations[{index}].output_indices is out of bounds"
            )
        if assigned_outputs.intersection(output_indices):
            raise ValueError(
                "Responses token_generations output_indices overlap between generations"
            )
        assigned_outputs.update(output_indices)
        output = generation.get("output_tokens")
        output_ids: list[int] | None
        output_logprobs: list[float]
        if output is None:
            output_ids, output_logprobs = None, []
            output_text = None
        else:
            output_ids, output_logprobs = _pairs(
                output,
                require_token_ids=True,
                field=f"Responses token_generations[{index}].output_tokens",
            )
            if not output_ids:
                output_ids = None
                output_logprobs = []
            output_texts = [
                item.get("text")
                for item in (_string_dict(value) for value in output)
                if item is not None
            ]
            output_text = (
                "".join(cast(str, text) for text in output_texts)
                if len(output_texts) == len(output)
                and all(isinstance(text, str) for text in output_texts)
                else None
            )
        if output_ids is None and any(
            _responses_output_is_sampled(response.output[value])
            for value in output_indices
        ):
            raise ValueError(
                f"Responses token_generations[{index}].output_tokens must contain "
                "exact tokens for its sampled output items"
            )
        generations.append(
            _ResponseGeneration(
                prompt_token_ids=prompt,
                output_token_ids=output_ids,
                output_logprobs=output_logprobs,
                output_indices=output_indices,
                output_text=output_text,
            )
        )
    required_outputs = {
        index
        for index, item in enumerate(response.output)
        if _responses_output_is_sampled(item)
    }
    if not required_outputs.issubset(assigned_outputs):
        raise ValueError(
            "Responses token_generations output_indices must cover every sampled "
            "output item"
        )
    return generations


def _responses_tokens(
    response: Response,
) -> tuple[list[int] | None, list[int] | None, list[float]]:
    data = _dump(response)
    generations = _response_generations(response)
    if generations:
        if len(generations) != 1:
            raise ValueError(
                "A multi-generation Responses exchange must be tokenized through "
                "its protocol history"
            )
        generation = generations[0]
        return (
            generation.prompt_token_ids,
            generation.output_token_ids,
            generation.output_logprobs,
        )
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
    return None, None, []


def _messages_tokens(
    response: Message,
) -> tuple[list[int] | None, list[int] | None, list[float]]:
    data = _dump(response)
    prompt_ids = _exact_token_ids(
        data.get("prompt_token_ids"),
        field="Messages prompt_token_ids",
        empty_is_missing=True,
    )
    token_ids = _exact_token_ids(data.get("token_ids"), field="Messages token_ids")
    if token_ids == [] and data.get("content"):
        token_ids = None
    logprobs = [
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else math.nan
        for value in data.get("logprobs") or []
    ]
    if token_ids is not None and logprobs and len(logprobs) != len(token_ids):
        raise ValueError("Messages token IDs and logprobs differ in length")
    if token_ids is None or not logprobs:
        logprobs = [math.nan] * len(token_ids or [])
    return prompt_ids, token_ids, logprobs


def _exchange_list(trajectory: Trajectory, model: str | None) -> list[Exchange]:
    exchanges = [
        *trajectory.exchanges.chat_completions,
        *trajectory.exchanges.completions,
        *trajectory.exchanges.responses,
        *trajectory.exchanges.messages,
    ]
    if model is not None:
        has_exact_match = any(exchange.model == model for exchange in exchanges)
        exchanges = [
            exchange
            for exchange in exchanges
            if (
                exchange.model == model
                if has_exact_match
                else _model_matches(exchange.model, model)
            )
        ]
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


def _artifact_name(model: str) -> str:
    return model.removeprefix("wandb-artifact:///")


def _artifact_identity(model: str) -> str:
    """Return an alias/version-independent checkpoint identity for base-model cache."""

    path = _artifact_name(model)
    name = path.rsplit("/", 1)[-1]
    return path[: -len(name)] + name.split(":", 1)[0]


_ARTIFACT_BASE_MODELS: dict[str, str] = {}


def _artifact_base_model(identity: str) -> str:
    if cached := _ARTIFACT_BASE_MODELS.get(identity):
        return cached
    from wandb.apis.public import Api

    artifact_path = identity
    if ":" not in artifact_path.rsplit("/", 1)[-1]:
        artifact_path = f"{artifact_path}:latest"
    artifact = Api().artifact(artifact_path)
    metadata = artifact.metadata
    base_model = metadata.get("base_model") or metadata.get("wandb.base_model")
    if not isinstance(base_model, str):
        raise ValueError(f"Checkpoint {identity!r} does not identify its base model")
    if len(_ARTIFACT_BASE_MODELS) >= 1024:
        _ARTIFACT_BASE_MODELS.pop(next(iter(_ARTIFACT_BASE_MODELS)))
    _ARTIFACT_BASE_MODELS[identity] = base_model
    return base_model


def _artifact_config(model: str) -> _TokenizerConfig:
    from wandb.apis.public import Api

    artifact_path = _artifact_name(model)
    if ":" not in artifact_path.rsplit("/", 1)[-1]:
        artifact_path = f"{artifact_path}:latest"
    artifact = Api().artifact(artifact_path)
    metadata = artifact.metadata
    base_model = metadata.get("base_model") or metadata.get("wandb.base_model")
    identity = _artifact_identity(model)
    if isinstance(base_model, str):
        if len(_ARTIFACT_BASE_MODELS) >= 1024 and identity not in _ARTIFACT_BASE_MODELS:
            _ARTIFACT_BASE_MODELS.pop(next(iter(_ARTIFACT_BASE_MODELS)))
        _ARTIFACT_BASE_MODELS[identity] = base_model
    renderer = metadata.get("renderer")
    renderer = renderer if isinstance(renderer, dict) else {}
    kwargs = renderer.get("chat_template_kwargs")
    return _TokenizerConfig(
        base_model=_artifact_base_model(identity),
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


@lru_cache(maxsize=8)
def _cached_tokenizer(base_model: str, revision: str | None) -> Tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer fallback requires ART's backend or tinker dependencies"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            revision=revision,
        )
        if base_model.startswith("deepseek-ai/DeepSeek-V4-"):
            from ..megatron.dsv4.tokenizer import get_dsv4_tokenizer

            tokenizer = get_dsv4_tokenizer(cast("PreTrainedTokenizerBase", tokenizer))
        return _as_tokenizer(configure_preserved_thinking_chat_template(tokenizer))
    except Exception as exc:
        raise ValueError(
            f"Could not load tokenizer for {base_model!r}; pass base_model explicitly"
        ) from exc


def _load_tokenizer(config: _TokenizerConfig) -> Tokenizer:
    return _cached_tokenizer(config.base_model, config.revision)


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
    tokenizer: Tokenizer,
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
    template = (
        chat_template
        or request.get("chat_template")
        or config.chat_template
        or getattr(tokenizer, "chat_template", None)
    )
    if kwargs.get("preserve_thinking") is True:
        template = chat_template_with_preserved_thinking(template)
    result = tokenizer.apply_chat_template(
        normalize_tool_call_arguments_for_chat_template(messages, template),
        tools=tools,
        tokenize=True,
        add_generation_prompt=not completed,
        **({"chat_template": template} if isinstance(template, str) else {}),
        **kwargs,
    )
    return _ids(result)


def _exchange_tokens(
    exchange: Exchange,
) -> tuple[list[int] | None, list[int] | None, list[float]]:
    if isinstance(exchange, ChatCompletionsExchange):
        return _chat_tokens(exchange.response)
    if isinstance(exchange, CompletionsExchange):
        return _completion_tokens(
            exchange.response,
            echo=exchange.request.get("echo") is True,
            empty_prompt_is_exact=exchange.request.get("prompt") in ("", []),
        )
    if isinstance(exchange, ResponsesExchange):
        return _responses_tokens(exchange.response)
    if isinstance(exchange, MessagesExchange):
        return _messages_tokens(exchange.response)
    raise TypeError(f"Unknown exchange type: {type(exchange)!r}")


def _visible_logprobs(
    exchange: Exchange, *, source: object | None = None
) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    if isinstance(exchange, ChatCompletionsExchange):
        choice = (
            _chat_choice(source) if source is not None else exchange.response.choices[0]
        )
        entries = _chat_logprob_entries(choice)
        decoder = codecs.getincrementaldecoder("utf-8")()
        for index, entry in enumerate(entries):
            data = _dump(entry)
            raw_bytes = data.get("bytes")
            if isinstance(raw_bytes, list):
                try:
                    next_data = (
                        _dump(entries[index + 1]) if index + 1 < len(entries) else {}
                    )
                    text = decoder.decode(
                        bytes(raw_bytes),
                        final=not isinstance(next_data.get("bytes"), list),
                    )
                except (TypeError, ValueError, UnicodeDecodeError):
                    return []
            else:
                decoder = codecs.getincrementaldecoder("utf-8")()
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
        outputs = exchange.response.output
        if source is not None:
            selected = _responses_source_outputs(source)
            if selected is None:
                return []
            selected_exchange, output_indices = selected
            if selected_exchange is not exchange:
                raise ValueError("Responses source belongs to a different exchange")
            outputs = [exchange.response.output[index] for index in output_indices]
        for output in outputs:
            for content in _dump(output).get("content") or []:
                for entry in _dump(content).get("logprobs") or []:
                    data = _dump(entry)
                    text = data.get("token")
                    logprob = data.get("logprob")
                    if isinstance(text, str) and isinstance(logprob, (int, float)):
                        values.append((text, float(logprob)))
    return values


def _sampled_text(exchange: Exchange, *, source: object | None = None) -> str | None:
    visible = "".join(text for text, _ in _visible_logprobs(exchange, source=source))
    if visible:
        return visible
    if isinstance(exchange, ChatCompletionsExchange):
        choice = (
            _chat_choice(source) if source is not None else exchange.response.choices[0]
        )
        content = choice.message.content
        return content if isinstance(content, str) else None
    if isinstance(exchange, CompletionsExchange):
        return exchange.response.choices[0].text
    if isinstance(exchange, MessagesExchange):
        parts = [
            block.text
            for block in exchange.response.content
            if isinstance(block, TextBlock)
        ]
        return "".join(parts) if parts else None
    if isinstance(exchange, ResponsesExchange):
        outputs = exchange.response.output
        if source is not None:
            selected = _responses_source_outputs(source)
            if selected is None:
                return None
            selected_exchange, output_indices = selected
            if selected_exchange is not exchange:
                raise ValueError("Responses source belongs to a different exchange")
            outputs = [exchange.response.output[index] for index in output_indices]
        parts: list[str] = []
        for output in outputs:
            data = _dump(output)
            if data.get("type") == "message":
                parts.append(_responses_output_text(data.get("content")))
        return "".join(parts) if parts else None
    return None


def _preserve_sampled_prefix(
    prompt: list[int],
    canonical_prefix: list[int],
    sampled_outputs: list[_SampledOutput],
    tokenizer: Tokenizer,
) -> list[int] | None:
    repaired = prompt
    for sampled in sampled_outputs:
        if sampled.text is None:
            return None
        rendered_ids = _ids(tokenizer(sampled.text, add_special_tokens=False))
        if rendered_ids == sampled.token_ids:
            continue
        start = sampled.start
        if (
            repaired[:start] != canonical_prefix[:start]
            or repaired[start : start + len(rendered_ids)] != rendered_ids
        ):
            return None
        repaired = [
            *repaired[:start],
            *sampled.token_ids,
            *repaired[start + len(rendered_ids) :],
        ]
    return repaired if repaired[: len(canonical_prefix)] == canonical_prefix else None


def _warn_prefix_retokenization() -> None:
    global _WARNED_PREFIX_RETOKENIZATION
    if _WARNED_PREFIX_RETOKENIZATION:
        return
    _WARNED_PREFIX_RETOKENIZATION = True
    warnings.warn(
        "Inference prompt token IDs retokenized an earlier sampled response; ART "
        "preserved the original sampled token IDs and logprobs. Prefer a service "
        "with prefix token-ID preservation, such as Caladan.",
        stacklevel=3,
    )


def _retained_output_suffix(
    *,
    prompt: Sequence[int],
    output: Sequence[int],
    logprobs: Sequence[float],
    later_prompt: Sequence[int],
) -> tuple[list[int], list[float]] | None:
    if list(later_prompt[: len(prompt)]) != list(prompt):
        return None
    continuation = later_prompt[len(prompt) :]
    for start in range(len(output)):
        suffix = list(output[start:])
        if list(continuation[: len(suffix)]) == suffix:
            return (
                suffix,
                list(logprobs[start:])
                if len(logprobs) == len(output)
                else [math.nan] * len(suffix),
            )
    return None


def _visible_token_evidence(
    tokenizer: Tokenizer | None,
    exchange: Exchange,
    *,
    source: object | None = None,
    sampled_text: str | None = None,
) -> tuple[list[int], list[float]] | None:
    values = _visible_logprobs(exchange, source=source)
    if not values or tokenizer is None:
        return None
    if sampled_text is not None:
        if "".join(text for text, _ in values) != sampled_text:
            matches: list[list[tuple[str, float]]] = []
            for start in range(len(values)):
                combined = ""
                for end in range(start, len(values)):
                    combined += values[end][0]
                    if combined == sampled_text:
                        matches.append(values[start : end + 1])
                        break
                    if len(combined) >= len(sampled_text):
                        break
            if len(matches) != 1:
                return None
            values = matches[0]
    logprobs = [logprob for _, logprob in values]
    text = "".join(text for text, _ in values)
    if any(not value for value, _ in values):
        token_ids = _ids(tokenizer(text, add_special_tokens=False))
        return (token_ids, logprobs) if len(token_ids) == len(values) else None
    try:
        contextual = cast(_OffsetTokenizer, tokenizer)(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, ValueError, NotImplementedError):
        contextual = None
    contextual_data = _string_dict(contextual)
    offsets = (
        contextual_data.get("offset_mapping") if contextual_data is not None else None
    )
    if isinstance(offsets, list) and len(offsets) == len(values):
        boundaries: list[tuple[int, int]] = []
        cursor = 0
        for value, _ in values:
            boundaries.append((cursor, cursor + len(value)))
            cursor += len(value)
        if offsets == boundaries:
            token_ids = _ids(contextual)
            if len(token_ids) == len(values):
                return token_ids, logprobs
    token_ids: list[int] = []
    for text, _ in values:
        encoded = _ids(tokenizer(text, add_special_tokens=False))
        if len(encoded) != 1:
            return None
        token_ids.append(encoded[0])
    return token_ids, logprobs


def _align_visible_logprobs(
    tokenizer: Tokenizer | None,
    completion: list[int],
    exchange: Exchange,
    *,
    source: object | None = None,
    sampled_text: str | None = None,
) -> list[float] | None:
    evidence = _visible_token_evidence(
        tokenizer,
        exchange,
        source=source,
        sampled_text=sampled_text,
    )
    if evidence is None:
        return None
    token_ids, logprobs = evidence

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
    history: LegacyHistory,
    *,
    model: str,
) -> TokenizedHistory:
    token_ids: list[int] = []
    logprobs: list[float] = []
    flags: list[TokenFlag] = []
    for item in history.messages_and_choices:
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
            flags.extend([TokenFlag.EXACT] * len(prompt))
        elif prompt[: len(token_ids)] != token_ids:
            raise ValueError("Legacy trajectory does not form one append-only history")
        else:
            suffix = prompt[len(token_ids) :]
            token_ids.extend(suffix)
            logprobs.extend([math.nan] * len(suffix))
            flags.extend([TokenFlag.EXACT] * len(suffix))
        token_ids.extend(completion)
        if len(completion_logprobs) != len(completion):
            completion_logprobs = [math.nan] * len(completion)
        logprobs.extend(completion_logprobs)
        flags.extend([TokenFlag.EXACT | TokenFlag.SAMPLED] * len(completion))
    if not token_ids:
        raise ValueError("Trajectory contains no trainable choices")
    return TokenizedHistory(
        history=history,
        model=model,
        tokens=token_ids,
        logprobs=logprobs,
        flags=flags,
    )


def _tokenize_exchange_trajectory(
    trajectory: Trajectory,
    history: History,
    base_model: str | None,
    *,
    model: str | None,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
    tokenizer_instance: Tokenizer | None = None,
    _trace: _TraceBuilder | None = None,
) -> TokenizedHistory:
    if trajectory.exchanges and (
        trajectory.messages_and_choices
        or trajectory.tools is not None
        or trajectory.additional_histories
    ):
        raise ValueError(
            "A trajectory cannot contain both exchanges and legacy histories"
        )
    if not trajectory.exchanges:
        if trajectory.additional_histories:
            raise ValueError("Tokenization requires one history")
        if model is None:
            raise ValueError("Legacy trajectory tokenization requires model=")
        return _legacy_tokenize(
            LegacyHistory(
                messages_and_choices=trajectory.messages_and_choices,
                tools=trajectory.tools,
            ),
            model=model,
        )
    exchanges = _exchange_list(trajectory, model)
    selected_model = exchanges[0].model
    if selected_model is None:
        raise AssertionError("_exchange_list returned an exchange without a model")
    exact_tokens = [_exchange_tokens(exchange) for exchange in exchanges]
    config = (
        _TokenizerConfig(base_model if base_model is not None else selected_model)
        if tokenizer_instance is not None
        or (
            base_model is not None
            and chat_template is not None
            and chat_template_kwargs is not None
        )
        else None
    )
    tokenizer = tokenizer_instance
    token_ids: list[int] = []
    logprobs: list[float] = []
    flags: list[TokenFlag] = []
    source_keys: list[_SampledSourceKey | None] = []
    sources: dict[_SampledSourceKey, object] = {}
    response_histories: dict[
        str, tuple[list[dict[str, Any]] | None, ResponsesExchange]
    ] = {}
    sampled_outputs: list[_SampledOutput] = []
    previous_render_state: tuple[Exchange, list[dict[str, Any]] | None] | None = None

    def fallback_config() -> _TokenizerConfig:
        nonlocal config
        if config is None:
            config = _tokenizer_config(selected_model, base_model)
        return config

    for exchange, (prompt, completion, completion_logprobs) in zip(
        exchanges, exact_tokens, strict=True
    ):
        prompt_is_exact = prompt is not None
        completion_is_exact = completion is not None
        if isinstance(exchange, CompletionsExchange):
            request_prompt = exchange.request.get("prompt")
            if isinstance(request_prompt, list) and not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in request_prompt
            ):
                raise ValueError(
                    "Trajectory tokenization does not support batched Completions prompts"
                )
            if not isinstance(request_prompt, (str, list)):
                raise ValueError("Completions prompt must be text or one token ID list")
            if exchange.request.get("echo") is True:
                raise ValueError(
                    "Trajectory tokenization does not support Completions echo=True"
                )
        messages_override: list[dict[str, Any]] | None = None
        if isinstance(exchange, ResponsesExchange):
            request = exchange.request
            if request.get("conversation") is not None and prompt is None:
                raise ValueError(
                    "Responses conversation history requires exact prompt tokens"
                )
            try:
                messages_override = _responses_messages(request)
            except ValueError:
                if prompt is None:
                    raise
            previous = request.get("previous_response_id")
            if previous is not None:
                if not isinstance(previous, str):
                    raise ValueError("Responses previous_response_id must be text")
                if previous not in response_histories and prompt is None:
                    raise ValueError(
                        "Responses exchange refers to a previous response outside this "
                        "trajectory without exact prompt tokens"
                    )
                if previous in response_histories:
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
            resolved_config = fallback_config()
            if tokenizer is None:
                tokenizer = _load_tokenizer(resolved_config)
            prompt = _template_ids(
                tokenizer,
                exchange,
                completed=False,
                config=resolved_config,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                messages_override=messages_override,
            )
        if completion is None:
            resolved_config = fallback_config()
            if tokenizer is None:
                tokenizer = _load_tokenizer(resolved_config)
            rendered_prompt = (
                _template_ids(
                    tokenizer,
                    exchange,
                    completed=False,
                    config=resolved_config,
                    chat_template=chat_template,
                    chat_template_kwargs=chat_template_kwargs,
                    messages_override=messages_override,
                )
                if prompt_is_exact
                else prompt
            )
            completed = _template_ids(
                tokenizer,
                exchange,
                completed=True,
                config=resolved_config,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                messages_override=messages_override,
            )
            if completed[: len(rendered_prompt)] != rendered_prompt:
                raise ValueError(
                    "Completed response does not extend its generation prompt"
                )
            completion = completed[len(rendered_prompt) :]
            completion_logprobs = _align_visible_logprobs(
                tokenizer, completion, exchange
            ) or [math.nan] * len(completion)
        if not token_ids:
            token_ids.extend(prompt)
            logprobs.extend([math.nan] * len(prompt))
            flags.extend(
                [TokenFlag.EXACT if prompt_is_exact else TokenFlag(0)] * len(prompt)
            )
            source_keys.extend([None] * len(prompt))
        elif len(prompt) < len(token_ids) or prompt[: len(token_ids)] != token_ids:
            resolved_config = fallback_config()
            if tokenizer is None:
                tokenizer = _load_tokenizer(resolved_config)
            repaired = _preserve_sampled_prefix(
                prompt,
                token_ids,
                sampled_outputs,
                tokenizer,
            )
            if repaired is None:
                if previous_render_state is None:
                    raise ValueError(
                        "Inference prompts do not form one append-only history"
                    )
                current_render = _template_ids(
                    tokenizer,
                    exchange,
                    completed=False,
                    config=resolved_config,
                    chat_template=chat_template,
                    chat_template_kwargs=chat_template_kwargs,
                    messages_override=messages_override,
                )
                previous_exchange, previous_messages = previous_render_state
                previous_render = _template_ids(
                    tokenizer,
                    previous_exchange,
                    completed=True,
                    config=resolved_config,
                    chat_template=chat_template,
                    chat_template_kwargs=chat_template_kwargs,
                    messages_override=previous_messages,
                )
                previous_canonical = _preserve_sampled_prefix(
                    previous_render,
                    token_ids,
                    sampled_outputs,
                    tokenizer,
                )
                if (
                    previous_canonical is None
                    or current_render[: len(previous_render)] != previous_render
                ):
                    raise ValueError(
                        "Rendered inference prompts do not form one append-only history"
                    )
                repaired = [
                    *previous_canonical,
                    *current_render[len(previous_render) :],
                ]
            prompt = repaired
            prompt_is_exact = False
            _warn_prefix_retokenization()
            suffix = prompt[len(token_ids) :]
            token_ids.extend(suffix)
            logprobs.extend([math.nan] * len(suffix))
            flags.extend([TokenFlag(0)] * len(suffix))
            source_keys.extend([None] * len(suffix))
        else:
            suffix = prompt[len(token_ids) :]
            token_ids.extend(suffix)
            logprobs.extend([math.nan] * len(suffix))
            if prompt_is_exact:
                flags = [flag | TokenFlag.EXACT for flag in flags]
            flags.extend(
                [TokenFlag.EXACT if prompt_is_exact else TokenFlag(0)] * len(suffix)
            )
            source_keys.extend([None] * len(suffix))
        if len(completion_logprobs) != len(completion):
            completion_logprobs = _align_visible_logprobs(
                tokenizer, completion, exchange
            ) or [math.nan] * len(completion)
        token_ids.extend(completion)
        logprobs.extend(completion_logprobs)
        completion_flag = TokenFlag.SAMPLED
        if completion_is_exact:
            completion_flag |= TokenFlag.EXACT
        flags.extend([completion_flag] * len(completion))
        source_key = _exchange_sampled_source_key(exchange)
        source_keys.extend([source_key] * len(completion))
        sources[source_key] = exchange
        if completion_is_exact:
            sampled_outputs.append(
                _SampledOutput(
                    text=_sampled_text(exchange),
                    token_ids=list(completion),
                    start=len(token_ids) - len(completion),
                )
            )
        previous_render_state = (exchange, messages_override)

    tokenized = TokenizedHistory(
        history=history,
        model=selected_model,
        tokens=token_ids,
        logprobs=logprobs,
        flags=flags,
    )
    if _trace is not None:
        _trace.set(tokenized, source_keys, sources)
    return tokenized


def _unique_exchanges(history: History) -> list[Exchange]:
    sources: Sequence[object]
    if isinstance(history, ChatCompletionsHistory):
        if len(history.messages) != len(history.message_sources):
            raise ValueError("messages and message_sources differ in length")
        sources = history.message_sources
    elif isinstance(history, AnthropicMessagesHistory):
        if len(history.messages) != len(history.message_sources):
            raise ValueError("messages and message_sources differ in length")
        sources = history.message_sources
    elif isinstance(history, ResponsesHistory):
        if len(history.input) != len(history.input_sources):
            raise ValueError("input and input_sources differ in length")
        sources = history.input_sources
    else:
        raise TypeError(f"Unsupported history type: {type(history).__name__}")

    exchanges: list[Exchange] = []
    seen: set[int] = set()
    for source in sources:
        exchange = getattr(source, "exchange", None)
        if not isinstance(
            exchange,
            (
                ChatCompletionsExchange,
                CompletionsExchange,
                ResponsesExchange,
                MessagesExchange,
            ),
        ):
            continue
        identity = id(exchange)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(exchange, ChatCompletionsExchange):
            choice_indexes = {
                getattr(item, "choice_index", None)
                for item in sources
                if getattr(item, "exchange", None) is exchange
                and getattr(item, "choice_index", None) is not None
            }
            if choice_indexes:
                exchange = exchange.model_copy(
                    update={
                        "response": exchange.response.model_copy(
                            update={
                                "choices": [
                                    choice
                                    for choice in exchange.response.choices
                                    if choice.index in choice_indexes
                                ]
                            }
                        )
                    }
                )
        exchanges.append(exchange)
    return sorted(exchanges, key=lambda item: (item.start_time, item.end_time))


def _without_reasoning(message: Mapping[str, object]) -> dict[str, object]:
    visible = dict(message)
    visible.pop("reasoning", None)
    visible.pop("reasoning_content", None)
    return visible


def _chat_choice_message(source: object) -> dict[str, Any] | None:
    exchange = getattr(source, "exchange", None)
    choice_index = getattr(source, "choice_index", None)
    if not isinstance(exchange, ChatCompletionsExchange) or not isinstance(
        choice_index, int
    ):
        return None
    choice = next(
        (item for item in exchange.response.choices if item.index == choice_index),
        None,
    )
    return (
        choice.message.model_dump(mode="python", exclude_none=True)
        if choice is not None
        else None
    )


def _source_index(value: object, *, length: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < length:
        raise ValueError(f"{field} is out of bounds")
    return value


def _chat_output_indices(source: object) -> tuple[int, ...] | None:
    value = getattr(source, "output_indices", None)
    if value is None:
        return None
    if not isinstance(value, tuple) or any(
        not isinstance(index, int) or isinstance(index, bool) for index in value
    ):
        raise ValueError("Chat output source indices are invalid")
    if tuple(sorted(set(value))) != value:
        raise ValueError("Chat output source indices are not strictly ordered")
    return value


def _chat_choice(source: object) -> Choice:
    exchange = getattr(source, "exchange", None)
    choice_index = getattr(source, "choice_index", None)
    if not isinstance(exchange, ChatCompletionsExchange):
        raise ValueError("Chat choice source has the wrong exchange type")
    if (
        not isinstance(choice_index, int)
        or isinstance(choice_index, bool)
        or choice_index < 0
    ):
        raise ValueError("Chat choice source index is invalid")
    choice = next(
        (item for item in exchange.response.choices if item.index == choice_index),
        None,
    )
    if choice is None:
        raise ValueError("Chat choice source index is out of bounds")
    return choice


def _validate_history_sources(history: History) -> None:
    if isinstance(history, ChatCompletionsHistory):
        from ._history import normalize_chat_message

        if len(history.messages) != len(history.message_sources):
            raise ValueError("messages and message_sources differ in length")
        for message, source in zip(
            history.messages, history.message_sources, strict=True
        ):
            if source is None:
                continue
            exchange = source.exchange
            if exchange.model != history.model:
                raise ValueError(
                    "Chat Completions history model no longer matches its source "
                    "exchange"
                )
            output_indices = _chat_output_indices(source)
            expected: list[dict[str, Any]] = []
            if isinstance(exchange, ChatCompletionsExchange):
                if source.choice_index is not None:
                    if (
                        source.request_index is not None
                        or output_indices is not None
                        or source.generation_index is not None
                    ):
                        raise ValueError("Chat source has conflicting indices")
                    choice_message = _chat_choice(source).message.model_dump(
                        mode="python", exclude_none=True
                    )
                    expected.append(choice_message)
                    visible = _without_reasoning(choice_message)
                    if visible != choice_message:
                        expected.append(visible)
                elif source.request_index is not None:
                    if (
                        output_indices is not None
                        or source.generation_index is not None
                    ):
                        raise ValueError("Chat source has conflicting indices")
                    request_messages = exchange.request.get("messages", [])
                    request_index = _source_index(
                        source.request_index,
                        length=len(request_messages),
                        field="Chat request source index",
                    )
                    expected.append(dict(request_messages[request_index]))
                else:
                    raise ValueError("Chat source has no request or choice index")
            elif isinstance(exchange, MessagesExchange):
                if (
                    source.choice_index is not None
                    or source.generation_index is not None
                ):
                    raise ValueError("Anthropic-to-Chat source has invalid indices")
                if source.request_index is not None:
                    if output_indices is not None:
                        raise ValueError("Anthropic-to-Chat source has invalid indices")
                    request_messages = exchange.request.get("messages", [])
                    request_index = _source_index(
                        source.request_index,
                        length=len(request_messages),
                        field="Anthropic request source index",
                    )
                    expected.extend(
                        _anthropic_messages(
                            {"messages": [request_messages[request_index]]}
                        )
                    )
                elif output_indices is None:
                    expected.extend(
                        _anthropic_messages(
                            {
                                "system": exchange.request.get("system"),
                                "messages": [],
                            }
                        )
                    )
                elif output_indices == (0,):
                    expected.append(_response_message(exchange))
                    visible_blocks = [
                        block.model_dump(mode="python", exclude_none=True)
                        for block in exchange.response.content
                        if getattr(block, "type", None)
                        not in {"thinking", "redacted_thinking"}
                    ]
                    expected.extend(
                        _anthropic_messages(
                            {
                                "messages": [
                                    {"role": "assistant", "content": visible_blocks}
                                ]
                            }
                        )
                    )
                else:
                    raise ValueError("Anthropic response source index is out of bounds")
            elif isinstance(exchange, ResponsesExchange):
                if source.request_index is not None:
                    if (
                        source.choice_index is not None
                        or output_indices is not None
                        or source.generation_index is not None
                    ):
                        raise ValueError("Responses-to-Chat source has invalid indices")
                    request_input = exchange.request.get("input")
                    if isinstance(request_input, list):
                        request_index = _source_index(
                            source.request_index,
                            length=len(request_input),
                            field="Responses request source index",
                        )
                        for end in range(request_index + 1, len(request_input) + 1):
                            projected = _responses_messages(
                                {"input": request_input[request_index:end]}
                            )
                            if len(projected) == 1:
                                expected.extend(projected)
                    elif isinstance(request_input, str):
                        _source_index(
                            source.request_index,
                            length=1,
                            field="Responses request source index",
                        )
                        expected.extend(_responses_messages({"input": request_input}))
                    else:
                        raise ValueError(
                            "Responses request source index is out of bounds"
                        )
                else:
                    if source.choice_index is not None:
                        raise ValueError("Responses-to-Chat source has invalid indices")
                    if output_indices is not None:
                        resolved_indices = tuple(
                            _source_index(
                                output_index,
                                length=len(exchange.response.output),
                                field="Responses output source index",
                            )
                            for output_index in output_indices
                        )
                        if source.generation_index is not None:
                            generations = _response_generations(exchange.response)
                            generation_index = _source_index(
                                source.generation_index,
                                length=len(generations),
                                field="Responses generation source index",
                            )
                            generation_outputs = generations[
                                generation_index
                            ].output_indices
                            if not resolved_indices and generation_outputs:
                                raise ValueError(
                                    "Responses empty output source references a "
                                    "generation with output items"
                                )
                            if any(
                                output_index not in generation_outputs
                                for output_index in resolved_indices
                            ):
                                raise ValueError(
                                    "Responses output source does not belong to "
                                    "its generation"
                                )
                        if not resolved_indices and source.generation_index is not None:
                            expected.append({"role": "assistant", "content": ""})
                        expected.extend(
                            _responses_messages(
                                {
                                    "input": [
                                        exchange.response.output[
                                            output_index
                                        ].model_dump(mode="python", exclude_none=True)
                                        for output_index in resolved_indices
                                    ]
                                }
                            )
                        )
                    elif message.get("role") == "system":
                        expected.extend(
                            _responses_messages(
                                {"instructions": exchange.request.get("instructions")}
                            )
                        )
                    elif source.generation_index is not None:
                        expected.append({"role": "assistant", "content": ""})
            actual = normalize_chat_message(message)
            if not any(
                actual == normalize_chat_message(candidate) for candidate in expected
            ):
                raise ValueError(
                    "Chat Completions history no longer matches its source exchange"
                )
        return
    if isinstance(history, AnthropicMessagesHistory):
        from ._history import _anthropic_message_key

        if len(history.messages) != len(history.message_sources):
            raise ValueError("messages and message_sources differ in length")
        for message, source in zip(
            history.messages, history.message_sources, strict=True
        ):
            if source is None:
                continue
            if source.exchange.model != history.model:
                raise ValueError(
                    "Anthropic Messages history model no longer matches its source "
                    "exchange"
                )
            if source.request_index is None:
                expected = {
                    "role": "assistant",
                    "content": [
                        block.model_dump(mode="json", exclude_none=True)
                        for block in source.exchange.response.content
                    ],
                }
            else:
                request_messages = source.exchange.request.get("messages", [])
                request_index = _source_index(
                    source.request_index,
                    length=len(request_messages),
                    field="Anthropic request source index",
                )
                expected = request_messages[request_index]
            matches = expected is not None and message == expected
            if not matches and source.request_index is None and expected is not None:
                matches = _anthropic_message_key(message) == _anthropic_message_key(
                    cast(MessageParam, expected), visible_only=True
                )
            if not matches:
                raise ValueError(
                    "Anthropic Messages history no longer matches its source exchange"
                )
        return
    if isinstance(history, ResponsesHistory):
        from ._history import _responses_input

        if len(history.input) != len(history.input_sources):
            raise ValueError("input and input_sources differ in length")

        for item, source in zip(history.input, history.input_sources, strict=True):
            if source is None:
                continue
            if source.exchange.model != history.model:
                raise ValueError(
                    "Responses history model no longer matches its source exchange"
                )
            if source.request_index is not None and source.output_index is not None:
                raise ValueError("Responses source has conflicting indices")
            if source.output_index is not None:
                output = source.exchange.response.output
                output_index = _source_index(
                    source.output_index,
                    length=len(output),
                    field="Responses output source index",
                )
                expected = output[output_index].model_dump(
                    mode="json", exclude_none=True
                )
                if source.generation_index is not None:
                    generations = _response_generations(source.exchange.response)
                    generation_index = _source_index(
                        source.generation_index,
                        length=len(generations),
                        field="Responses generation source index",
                    )
                    if output_index not in generations[generation_index].output_indices:
                        raise ValueError(
                            "Responses output source does not belong to its generation"
                        )
            elif source.request_index is not None:
                request_input = _responses_input(source.exchange.request.get("input"))
                request_index = _source_index(
                    source.request_index,
                    length=len(request_input),
                    field="Responses request source index",
                )
                if source.generation_index is not None:
                    raise ValueError("Responses request source has a generation index")
                expected = request_input[request_index]
            else:
                generations = _response_generations(source.exchange.response)
                if source.generation_index is None:
                    raise ValueError(
                        "Responses source has no request, output, or generation index"
                    )
                generation_index = _source_index(
                    source.generation_index,
                    length=len(generations),
                    field="Responses generation source index",
                )
                if (
                    generation_index != len(generations) - 1
                    or generations[generation_index].output_indices
                ):
                    raise ValueError(
                        "Responses generation-only source must refer to a terminal "
                        "generation without native output items"
                    )
                expected = {"role": "assistant", "content": ""}
            if expected is None or item != expected:
                raise ValueError(
                    "Responses history no longer matches its source exchange"
                )


def _trajectory_from_history(history: History) -> Trajectory:
    _validate_history_sources(history)
    exchanges = _unique_exchanges(history)
    if not exchanges:
        raise ValueError(
            "History has no source exchanges; local history-only rendering is not yet possible"
        )
    from . import TrajectoryExchanges

    return Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[
                item for item in exchanges if isinstance(item, ChatCompletionsExchange)
            ],
            completions=[
                item for item in exchanges if isinstance(item, CompletionsExchange)
            ],
            responses=[
                item for item in exchanges if isinstance(item, ResponsesExchange)
            ],
            messages=[item for item in exchanges if isinstance(item, MessagesExchange)],
        )
    )


def _last_source_exchange(sources: Sequence[object]) -> Exchange | None:
    for source in reversed(sources):
        exchange = getattr(source, "exchange", None)
        if isinstance(
            exchange,
            (
                ChatCompletionsExchange,
                CompletionsExchange,
                ResponsesExchange,
                MessagesExchange,
            ),
        ):
            return exchange
    return None


@dataclass(frozen=True)
class _HistoryRenderState:
    needs_render: bool
    context_changed: bool = False
    projection_matches: bool | None = None


def _matches_final_chat_exchange(history: ChatCompletionsHistory) -> bool | None:
    from ._history import normalize_chat_message

    for source in reversed(history.message_sources):
        if (
            source is None
            or not isinstance(source.exchange, ChatCompletionsExchange)
            or source.choice_index is None
        ):
            continue
        choice = _chat_choice(source)
        expected = [
            *source.exchange.request.get("messages", []),
            choice.message.model_dump(mode="python", exclude_none=True),
        ]
        return [normalize_chat_message(message) for message in history.messages] == [
            normalize_chat_message(message) for message in expected
        ]
    return None


def _history_render_state(history: History) -> _HistoryRenderState:
    if isinstance(history, ChatCompletionsHistory):
        if any(source is None for source in history.message_sources):
            return _HistoryRenderState(needs_render=True, projection_matches=False)
        for message, source in zip(
            history.messages, history.message_sources, strict=True
        ):
            if source is None or (original := _chat_choice_message(source)) is None:
                continue
            if dict(message) != original and dict(message) == _without_reasoning(
                original
            ):
                return _HistoryRenderState(needs_render=True)
        exchange = _last_source_exchange(history.message_sources)
        if not isinstance(exchange, ChatCompletionsExchange):
            return _HistoryRenderState(needs_render=True)
        context_changed = (
            history.tools != exchange.request.get("tools")
            or history.chat_template != exchange.request.get("chat_template")
            or history.chat_template_kwargs
            != exchange.request.get("chat_template_kwargs")
        )
        if context_changed:
            return _HistoryRenderState(needs_render=True, context_changed=True)
        projection_matches = _matches_final_chat_exchange(history)
        if projection_matches is None:
            projection_matches = _history_matches_projection(history)
        return _HistoryRenderState(
            needs_render=not projection_matches,
            projection_matches=projection_matches,
        )
    if isinstance(history, AnthropicMessagesHistory):
        if any(source is None for source in history.message_sources):
            return _HistoryRenderState(needs_render=True, projection_matches=False)
        for message, source in zip(
            history.messages, history.message_sources, strict=True
        ):
            if source is None or source.request_index is not None:
                continue
            expected = {
                "role": "assistant",
                "content": [
                    block.model_dump(mode="json", exclude_none=True)
                    for block in source.exchange.response.content
                ],
            }
            if message != expected:
                return _HistoryRenderState(needs_render=True)
        exchange = _last_source_exchange(history.message_sources)
        if not isinstance(exchange, MessagesExchange):
            return _HistoryRenderState(needs_render=False)
        context_changed = (
            history.system != exchange.request.get("system")
            or history.tools != exchange.request.get("tools")
            or history.chat_template != exchange.request.get("chat_template")
            or history.chat_template_kwargs
            != exchange.request.get("chat_template_kwargs")
        )
        if context_changed:
            return _HistoryRenderState(needs_render=True, context_changed=True)
        projection_matches = _history_matches_projection(history)
        return _HistoryRenderState(
            needs_render=not projection_matches,
            projection_matches=projection_matches,
        )
    if isinstance(history, ResponsesHistory):
        if any(source is None for source in history.input_sources):
            return _HistoryRenderState(needs_render=True, projection_matches=False)
        exchange = _last_source_exchange(history.input_sources)
        if not isinstance(exchange, ResponsesExchange):
            return _HistoryRenderState(needs_render=False)
        context_changed = (
            history.instructions != exchange.request.get("instructions")
            or history.tools != exchange.request.get("tools")
            or history.chat_template != exchange.request.get("chat_template")
            or history.chat_template_kwargs
            != exchange.request.get("chat_template_kwargs")
        )
        if context_changed:
            return _HistoryRenderState(needs_render=True, context_changed=True)
        projection_matches = _history_matches_projection(history)
        return _HistoryRenderState(
            needs_render=not projection_matches,
            projection_matches=projection_matches,
        )
    return _HistoryRenderState(needs_render=False)


def _source_signature(source: object) -> tuple[object, ...] | None:
    if source is None:
        return None
    exchange = getattr(source, "exchange", None)
    if not isinstance(
        exchange,
        (
            ChatCompletionsExchange,
            CompletionsExchange,
            ResponsesExchange,
            MessagesExchange,
        ),
    ):
        return None
    response_id = getattr(exchange.response, "id", None)
    choice_index = getattr(source, "choice_index", None)
    generation_index = getattr(source, "generation_index", None)
    evidence_fingerprint: str | None = None
    if isinstance(exchange, ChatCompletionsExchange) and isinstance(choice_index, int):
        evidence_fingerprint = _sampled_evidence_fingerprint(
            exchange, protocol="chat_completions", index=choice_index
        )
    elif isinstance(exchange, ResponsesExchange) and isinstance(generation_index, int):
        evidence_fingerprint = _sampled_evidence_fingerprint(
            exchange, protocol="responses", index=generation_index
        )
    elif isinstance(exchange, MessagesExchange) and (
        getattr(source, "output_index", None) == 0
        or _chat_output_indices(source) == (0,)
    ):
        evidence_fingerprint = _sampled_evidence_fingerprint(
            exchange, protocol="messages", index=0
        )
    return (
        type(source),
        type(exchange),
        exchange.start_time,
        exchange.end_time,
        response_id,
        evidence_fingerprint,
        getattr(source, "request_index", None),
        choice_index,
        getattr(source, "output_index", None),
        _chat_output_indices(source),
        generation_index,
        getattr(source, "prompt_index", None),
    )


def _sources_match(left: Sequence[object], right: Sequence[object]) -> bool:
    return [_source_signature(item) for item in left] == [
        _source_signature(item) for item in right
    ]


def _history_matches_projection(history: History) -> bool:
    exchanges = _unique_exchanges(history)
    if not exchanges:
        return False
    from . import TrajectoryExchanges
    from ._history import (
        anthropic_messages_histories,
        chat_completions_histories,
        responses_histories,
    )

    trajectory = Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[
                item for item in exchanges if isinstance(item, ChatCompletionsExchange)
            ],
            completions=[
                item for item in exchanges if isinstance(item, CompletionsExchange)
            ],
            responses=[
                item for item in exchanges if isinstance(item, ResponsesExchange)
            ],
            messages=[item for item in exchanges if isinstance(item, MessagesExchange)],
        )
    )
    if isinstance(history, ChatCompletionsHistory):
        for reconcile in (False, True):
            try:
                candidates: Sequence[History] = chat_completions_histories(
                    trajectory,
                    model=history.model,
                    reconcile=reconcile,
                )
            except ValueError as error:
                if "no Chat Completions exchanges" not in str(error):
                    raise
                if trajectory.exchanges.messages and not trajectory.exchanges.responses:
                    candidates = [
                        candidate.as_chat_completions_history()
                        for candidate in anthropic_messages_histories(
                            trajectory,
                            model=history.model,
                            reconcile=reconcile,
                        )
                    ]
                elif (
                    trajectory.exchanges.responses and not trajectory.exchanges.messages
                ):
                    candidates = [
                        candidate.as_chat_completions_history()
                        for candidate in responses_histories(
                            trajectory,
                            model=history.model,
                            reconcile=reconcile,
                        )
                    ]
                else:
                    return False
            if any(
                isinstance(candidate, ChatCompletionsHistory)
                and candidate.messages == history.messages
                and candidate.tools == history.tools
                and candidate.chat_template == history.chat_template
                and candidate.chat_template_kwargs == history.chat_template_kwargs
                and _sources_match(candidate.message_sources, history.message_sources)
                for candidate in candidates
            ):
                return True
        return False
    if isinstance(history, AnthropicMessagesHistory):
        for reconcile in (False, True):
            candidates = anthropic_messages_histories(
                trajectory,
                model=history.model,
                reconcile=reconcile,
            )
            if any(
                isinstance(candidate, AnthropicMessagesHistory)
                and candidate.messages == history.messages
                and candidate.system == history.system
                and candidate.tools == history.tools
                and candidate.chat_template == history.chat_template
                and candidate.chat_template_kwargs == history.chat_template_kwargs
                and _sources_match(candidate.message_sources, history.message_sources)
                for candidate in candidates
            ):
                return True
        return False
    if isinstance(history, ResponsesHistory):
        for reconcile in (False, True):
            candidates = responses_histories(
                trajectory,
                model=history.model,
                reconcile=reconcile,
            )
            if any(
                isinstance(candidate, ResponsesHistory)
                and candidate.input == history.input
                and candidate.instructions == history.instructions
                and candidate.tools == history.tools
                and candidate.conversation == history.conversation
                and candidate.previous_response_id == history.previous_response_id
                and candidate.chat_template == history.chat_template
                and candidate.chat_template_kwargs == history.chat_template_kwargs
                and _sources_match(candidate.input_sources, history.input_sources)
                for candidate in candidates
            ):
                return True
        return False
    return False


def _response_generation_text(
    response: Response, generation: _ResponseGeneration
) -> str | None:
    parts: list[str] = []
    for output_index in generation.output_indices:
        item = _dump(response.output[output_index])
        kind = item.get("type")
        if kind == "message":
            parts.append(_responses_output_text(item.get("content")))
        elif kind == "reasoning":
            parts.append(_responses_reasoning_text(item))
        else:
            return None
    return "".join(parts) or None


def _tokenize_exact_responses_history(
    history: ResponsesHistory,
    *,
    base_model: str | None,
    tokenizer: Tokenizer | None,
    _trace: _TraceBuilder | None = None,
) -> TokenizedHistory | None:
    generation_keys: list[tuple[ResponsesExchange, int]] = []
    retained_output_indices: dict[tuple[int, int], set[int]] = {}
    seen: set[tuple[int, int]] = set()
    for source in history.input_sources:
        if source is None or source.generation_index is None:
            continue
        key = (id(source.exchange), source.generation_index)
        if source.output_index is not None:
            retained_output_indices.setdefault(key, set()).add(source.output_index)
        if key not in seen:
            seen.add(key)
            generation_keys.append((source.exchange, source.generation_index))
    if not generation_keys:
        return None

    token_ids: list[int] = []
    logprobs: list[float] = []
    flags: list[TokenFlag] = []
    source_keys: list[_SampledSourceKey | None] = []
    sources: dict[_SampledSourceKey, object] = {}
    sampled_outputs: list[_SampledOutput] = []
    for position, (exchange, generation_index) in enumerate(generation_keys):
        generations = _response_generations(exchange.response)
        if not 0 <= generation_index < len(generations):
            raise ValueError("Responses source generation index is out of bounds")
        generation = generations[generation_index]
        prompt = generation.prompt_token_ids
        output = generation.output_token_ids
        if prompt is None or output is None:
            return None
        retained = retained_output_indices.get((id(exchange), generation_index), set())
        if retained != set(generation.output_indices):
            if position + 1 >= len(generation_keys):
                return None
            next_exchange, next_generation_index = generation_keys[position + 1]
            next_generations = _response_generations(next_exchange.response)
            if not 0 <= next_generation_index < len(next_generations):
                raise ValueError("Responses source generation index is out of bounds")
            next_prompt = next_generations[next_generation_index].prompt_token_ids
            if next_prompt is None:
                return None
            retained_suffix = _retained_output_suffix(
                prompt=prompt,
                output=output,
                logprobs=generation.output_logprobs,
                later_prompt=next_prompt,
            )
            if retained_suffix is None:
                return None
            output, output_logprobs = retained_suffix
            output_text = None
        else:
            output_logprobs = generation.output_logprobs
            output_text = generation.output_text or _response_generation_text(
                exchange.response, generation
            )
        if not token_ids:
            token_ids.extend(prompt)
            logprobs.extend([math.nan] * len(prompt))
            flags.extend([TokenFlag.EXACT] * len(prompt))
            source_keys.extend([None] * len(prompt))
        elif prompt[: len(token_ids)] == token_ids:
            suffix = prompt[len(token_ids) :]
            token_ids.extend(suffix)
            logprobs.extend([math.nan] * len(suffix))
            flags = [flag | TokenFlag.EXACT for flag in flags]
            flags.extend([TokenFlag.EXACT] * len(suffix))
            source_keys.extend([None] * len(suffix))
        else:
            if tokenizer is None and sampled_outputs:
                tokenizer = _load_tokenizer(
                    _tokenizer_config(history.model, base_model)
                )
            repaired = (
                _preserve_sampled_prefix(prompt, token_ids, sampled_outputs, tokenizer)
                if tokenizer is not None
                else None
            )
            if repaired is None:
                raise ValueError(
                    "Responses token generations do not form one append-only history"
                )
            _warn_prefix_retokenization()
            suffix = repaired[len(token_ids) :]
            token_ids.extend(suffix)
            logprobs.extend([math.nan] * len(suffix))
            flags.extend([TokenFlag.EXACT] * len(suffix))
            source_keys.extend([None] * len(suffix))
        token_ids.extend(output)
        logprobs.extend(output_logprobs)
        flags.extend([TokenFlag.EXACT | TokenFlag.SAMPLED] * len(output))
        source = next(
            (
                item
                for item in history.input_sources
                if item is not None
                and item.exchange is exchange
                and item.generation_index == generation_index
            ),
            None,
        )
        if source is None:
            raise AssertionError("Responses generation has no history source")
        source_key = _sampled_source_key(source)
        source_keys.extend([source_key] * len(output))
        sources[source_key] = source
        sampled_outputs.append(
            _SampledOutput(
                text=output_text,
                token_ids=list(output),
                start=len(token_ids) - len(output),
            )
        )
    tokenized = TokenizedHistory(
        history=history,
        model=history.model,
        tokens=token_ids,
        logprobs=logprobs,
        flags=flags,
    )
    if _trace is not None:
        _trace.set(tokenized, source_keys, sources)
    return tokenized


def _responses_source_outputs(
    source: object,
) -> tuple[ResponsesExchange, tuple[int, ...]] | None:
    exchange = getattr(source, "exchange", None)
    if not isinstance(exchange, ResponsesExchange):
        return None
    output_indices = _chat_output_indices(source)
    if output_indices is None:
        return None
    return exchange, tuple(
        _source_index(
            index,
            length=len(exchange.response.output),
            field="Responses source output index",
        )
        for index in output_indices
    )


def _responses_source_generation(
    source: object,
) -> tuple[ResponsesExchange, _ResponseGeneration, tuple[int, ...]] | None:
    selected = _responses_source_outputs(source)
    generation_index = getattr(source, "generation_index", None)
    if selected is None or generation_index is None:
        return None
    if not isinstance(generation_index, int) or isinstance(generation_index, bool):
        raise ValueError("Responses source generation index is invalid")
    exchange, output_indices = selected
    generations = _response_generations(exchange.response)
    if not 0 <= generation_index < len(generations):
        raise ValueError("Responses source generation index is out of bounds")
    generation = generations[generation_index]
    if bool(output_indices) != bool(generation.output_indices):
        raise ValueError("Responses empty output source does not match its generation")
    if any(index not in generation.output_indices for index in output_indices):
        raise ValueError(
            "Responses source output index does not belong to its generation"
        )
    return exchange, generation, output_indices


def _responses_generation_messages(source: object) -> list[dict[str, Any]] | None:
    selected = _responses_source_generation(source)
    if selected is None:
        return None
    exchange, _, output_indices = selected
    return _responses_messages(
        {
            "input": [
                exchange.response.output[index].model_dump(
                    mode="python", exclude_none=True
                )
                for index in output_indices
            ]
        }
    )


def _responses_generation_full_tokens(
    source: object,
) -> tuple[list[int] | None, list[float]]:
    selected = _responses_source_generation(source)
    if selected is None:
        return None, []
    _, generation, output_indices = selected
    if output_indices != tuple(generation.output_indices):
        return None, []
    messages = _responses_generation_messages(source)
    if messages is None or (messages and len(messages) != 1):
        return None, []
    return generation.output_token_ids, generation.output_logprobs


def _chat_source_full_tokens(
    source: object,
) -> tuple[list[int] | None, list[float]]:
    exchange = getattr(source, "exchange", None)
    if isinstance(exchange, ChatCompletionsExchange):
        choice_index = getattr(source, "choice_index", None)
        if choice_index is None:
            return None, []
        choice = next(
            item for item in exchange.response.choices if item.index == choice_index
        )
        return _chat_choice_output_tokens(choice)
    if isinstance(exchange, ResponsesExchange):
        return _responses_generation_full_tokens(source)
    if isinstance(exchange, MessagesExchange):
        if _chat_output_indices(source) != (0,):
            return None, []
        _, tokens, logprobs = _exchange_tokens(exchange)
        return tokens, logprobs
    return None, []


def _chat_source_prompt_tokens(source: object) -> list[int] | None:
    exchange = getattr(source, "exchange", None)
    if isinstance(exchange, ChatCompletionsExchange):
        choice_index = getattr(source, "choice_index", None)
        if choice_index is None:
            return None
        choice = next(
            item for item in exchange.response.choices if item.index == choice_index
        )
        prompt, _, _ = _chat_choice_tokens(choice, _dump(exchange.response))
        return prompt
    if isinstance(exchange, ResponsesExchange):
        generation_index = getattr(source, "generation_index", None)
        generations = _response_generations(exchange.response)
        if isinstance(generation_index, int):
            if not 0 <= generation_index < len(generations):
                raise ValueError("Responses source generation index is out of bounds")
            return generations[generation_index].prompt_token_ids
        return _responses_tokens(exchange.response)[0]
    if isinstance(exchange, MessagesExchange):
        if _chat_output_indices(source) != (0,):
            return None
        return _messages_tokens(exchange.response)[0]
    return None


def _source_is_sampled(source: object) -> bool:
    exchange = getattr(source, "exchange", None)
    if isinstance(exchange, ChatCompletionsExchange):
        return getattr(source, "choice_index", None) is not None
    if isinstance(exchange, MessagesExchange):
        return _chat_output_indices(source) == (0,)
    if isinstance(exchange, ResponsesExchange):
        if getattr(source, "generation_index", None) is not None:
            return True
        selected = _responses_source_outputs(source)
        return selected is not None and any(
            _responses_output_is_sampled(selected[0].response.output[index])
            for index in selected[1]
        )
    return False


def _tokenize_exact_projected_chat_history(
    history: ChatCompletionsHistory,
    *,
    projection_validated: bool = False,
    _trace: _TraceBuilder | None = None,
) -> TokenizedHistory | None:
    if not projection_validated and not _history_matches_projection(history):
        return None
    sampled_sources: list[object] = []
    seen: set[tuple[object, ...]] = set()
    for message, source in zip(history.messages, history.message_sources, strict=True):
        signature = _source_signature(source)
        if (
            message.get("role") == "assistant"
            and source is not None
            and signature is not None
            and signature not in seen
            and _source_is_sampled(source)
        ):
            seen.add(signature)
            sampled_sources.append(source)
    if not sampled_sources:
        return None

    final_source = sampled_sources[-1]
    final_prompt = _chat_source_prompt_tokens(final_source)
    final_output, final_logprobs = _chat_source_full_tokens(final_source)
    if final_prompt is None or final_output is None:
        return None

    token_ids = [*final_prompt, *final_output]
    logprobs = [
        *([math.nan] * len(final_prompt)),
        *(
            final_logprobs
            if len(final_logprobs) == len(final_output)
            else [math.nan] * len(final_output)
        ),
    ]
    flags = [
        *([TokenFlag.EXACT] * len(final_prompt)),
        *([TokenFlag.EXACT | TokenFlag.SAMPLED] * len(final_output)),
    ]
    final_key = _sampled_source_key(final_source)
    source_keys: list[_SampledSourceKey | None] = [
        *([None] * len(final_prompt)),
        *([final_key] * len(final_output)),
    ]
    sources: dict[_SampledSourceKey, object] = {final_key: final_source}
    for index, source in enumerate(sampled_sources[:-1]):
        prompt = _chat_source_prompt_tokens(source)
        output, output_logprobs = _chat_source_full_tokens(source)
        if (
            prompt is None
            or output is None
            or list(final_prompt[: len(prompt)]) != prompt
        ):
            return None
        retained = next(
            (
                evidence
                for later_source in sampled_sources[index + 1 :]
                if (later_prompt := _chat_source_prompt_tokens(later_source))
                is not None
                and (
                    evidence := _retained_output_suffix(
                        prompt=prompt,
                        output=output,
                        logprobs=output_logprobs,
                        later_prompt=later_prompt,
                    )
                )
                is not None
            ),
            None,
        )
        if retained is None:
            return None
        retained_ids, retained_logprobs = retained
        start = len(prompt)
        end = start + len(retained_ids)
        if final_prompt[start:end] != retained_ids:
            return None
        flags[start:end] = [TokenFlag.EXACT | TokenFlag.SAMPLED] * len(retained_ids)
        logprobs[start:end] = retained_logprobs
        source_key = _sampled_source_key(source)
        source_keys[start:end] = [source_key] * len(retained_ids)
        sources[source_key] = source
    if history.model is None:
        raise ValueError("History tokenization requires a model")
    tokenized = TokenizedHistory(
        history=history,
        model=history.model,
        tokens=token_ids,
        logprobs=logprobs,
        flags=flags,
    )
    if _trace is not None:
        _trace.set(tokenized, source_keys, sources)
    return tokenized


def _chat_message_parts(message: Mapping[str, object]) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    reasoning = message.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning:
        reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append(("reasoning", reasoning))
    if content := _content_text(message.get("content")):
        parts.append(("content", content))
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal:
        parts.append(("content", refusal))
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            call = _string_dict(tool_call)
            function = _string_dict(call.get("function")) if call is not None else None
            if function is None:
                continue
            for value in (function.get("name"), function.get("arguments")):
                if isinstance(value, str) and value:
                    parts.append(("tool_call", value))
    return parts


def _chat_message_text_slot_groups(
    message: dict[str, Any],
) -> list[list[tuple[dict[str, Any], str]]]:
    groups: list[list[tuple[dict[str, Any], str]]] = []
    reasoning_key = (
        "reasoning"
        if isinstance(message.get("reasoning"), str) and message["reasoning"]
        else "reasoning_content"
    )
    if isinstance(message.get(reasoning_key), str) and message[reasoning_key]:
        groups.append([(message, reasoning_key)])
    content = message.get("content")
    if isinstance(content, str) and content:
        groups.append([(message, "content")])
    elif isinstance(content, list):
        content_slots = [
            (block, "text")
            for block in content
            if (
                isinstance(block, dict)
                and block.get("type") in {"input_text", "output_text", "text"}
                and isinstance(block.get("text"), str)
                and block["text"]
            )
        ]
        if content_slots:
            groups.append(content_slots)
    if isinstance(message.get("refusal"), str) and message["refusal"]:
        groups.append([(message, "refusal")])
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            for key in ("name", "arguments"):
                if isinstance(function.get(key), str) and function[key]:
                    groups.append([(function, key)])
    return groups


def _chat_source_tokens(
    source: object,
    text: str,
    *,
    part: str,
    full_tokens: tuple[list[int] | None, list[float]] | None = None,
) -> tuple[list[int] | None, list[float]]:
    exchange = getattr(source, "exchange", None)
    if isinstance(exchange, ChatCompletionsExchange):
        tokens, logprobs = (
            full_tokens if full_tokens is not None else _chat_source_full_tokens(source)
        )
        if tokens is None:
            return None, []
        sampled_text = _sampled_text(exchange, source=source)
        message = _dump(
            next(
                item
                for item in exchange.response.choices
                if item.index == getattr(source, "choice_index", None)
            ).message
        )
        if (
            part == "content"
            and sampled_text == text
            and not any(message.get(key) for key in ("reasoning", "tool_calls"))
        ):
            return tokens, logprobs
        return None, []
    if isinstance(exchange, MessagesExchange) and _chat_output_indices(source) == (0,):
        block_type = "thinking" if part == "reasoning" else "text"
        blocks = [
            block
            for block in exchange.response.content
            if getattr(block, "type", None) == block_type
        ]
        block_text = "".join(
            str(getattr(block, "thinking" if part == "reasoning" else "text", ""))
            for block in blocks
        )
        if block_text == text:
            token_ids: list[int] = []
            logprobs: list[float] = []
            for block in blocks:
                extra = block.model_extra or {}
                block_ids = _exact_token_ids(
                    extra.get("token_ids"), field="Messages content token_ids"
                )
                block_logprobs = extra.get("logprobs")
                if block_ids is None or not isinstance(block_logprobs, list):
                    break
                if len(block_ids) != len(block_logprobs):
                    raise ValueError(
                        "Messages content token IDs and logprobs differ in length"
                    )
                token_ids.extend(block_ids)
                logprobs.extend(float(value) for value in block_logprobs)
            else:
                return token_ids, logprobs
        if part != "content" or any(
            getattr(block, "type", None) in {"thinking", "redacted_thinking"}
            for block in exchange.response.content
        ):
            return None, []
        _, tokens, logprobs = _messages_tokens(exchange.response)
        return tokens, logprobs
    if isinstance(exchange, ResponsesExchange) and _source_is_sampled(source):
        selected = _responses_source_outputs(source)
        output_indices = selected[1] if selected is not None else ()
        if len(output_indices) == 1:
            output_index = output_indices[0]
            item = _dump(exchange.response.output[output_index])
            if item.get("type") == "message" and part == "content":
                item_text = _responses_output_text(item.get("content"))
                if item_text == text:
                    token_ids: list[int] = []
                    logprobs: list[float] = []
                    for content in item.get("content") or []:
                        pairs, pair_logprobs = _pairs(
                            _dump(content).get("logprobs"),
                            field="Responses content logprobs",
                        )
                        if not pairs:
                            break
                        token_ids.extend(pairs)
                        logprobs.extend(pair_logprobs)
                    else:
                        return token_ids, logprobs
        # Aggregate generation evidence cannot be partitioned safely across
        # multiple projected messages. Item-local pairs above remain usable;
        # otherwise render this message without claiming exact token identity.
        return None, []
    return None, []


def _source_covers_complete_sampled_message(
    message: Mapping[str, object], source: object
) -> bool:
    from ._history import normalize_chat_message

    exchange = getattr(source, "exchange", None)
    if isinstance(exchange, ChatCompletionsExchange):
        expected = _chat_choice_message(source)
        return expected is not None and normalize_chat_message(
            message
        ) == normalize_chat_message(expected)
    if isinstance(exchange, MessagesExchange):
        if _chat_output_indices(source) != (0,):
            return False
        return normalize_chat_message(message) == normalize_chat_message(
            _response_message(exchange)
        )
    if not isinstance(exchange, ResponsesExchange):
        return False
    generation_index = getattr(source, "generation_index", None)
    if not isinstance(generation_index, int):
        return False
    generations = _response_generations(exchange.response)
    if not 0 <= generation_index < len(generations):
        raise ValueError("Responses source generation index is out of bounds")
    output_indices = _chat_output_indices(source)
    if output_indices is None:
        return False
    if not output_indices:
        return message.get("role") == "assistant" and not _chat_message_parts(message)
    generation = generations[generation_index]
    if any(index not in generation.output_indices for index in output_indices):
        raise ValueError(
            "Responses source output index does not belong to its generation"
        )
    projected = _responses_generation_messages(source)
    if projected is None:
        return False
    return len(projected) == 1 and normalize_chat_message(
        message
    ) == normalize_chat_message(projected[0])


def _tokenize_chat_view(
    history: ChatCompletionsHistory,
    *,
    base_model: str | None,
    tokenizer: Tokenizer | None,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
    _projection_matches: bool | None = None,
    _trace: _TraceBuilder | None = None,
) -> TokenizedHistory:
    _validate_history_sources(history)
    config = (
        _TokenizerConfig(base_model or history.model or "")
        if tokenizer is not None
        or (base_model is not None and chat_template is not None)
        else _tokenizer_config(history.model or "", base_model)
    )
    if tokenizer is None:
        if not history.model and base_model is None:
            raise ValueError("History tokenization requires a model or base_model")
        tokenizer = _load_tokenizer(config)
    assert tokenizer is not None
    resolved_tokenizer = tokenizer
    messages = [dict(message) for message in history.messages]
    kwargs = {
        **(config.chat_template_kwargs or {}),
        **(history.chat_template_kwargs or {}),
        **(chat_template_kwargs or {}),
    }
    last_exchange = _last_source_exchange(history.message_sources)
    if isinstance(last_exchange, MessagesExchange) and isinstance(
        thinking := last_exchange.request.get("thinking"), dict
    ):
        kwargs.setdefault("enable_thinking", thinking.get("type") == "enabled")
        if budget := thinking.get("budget_tokens"):
            kwargs.setdefault("thinking_budget", budget)
    template = chat_template or history.chat_template or config.chat_template
    if template is None:
        tokenizer_template = getattr(resolved_tokenizer, "chat_template", None)
        if isinstance(tokenizer_template, str):
            template = tokenizer_template
    if kwargs.get("preserve_thinking") is True:
        template = chat_template_with_preserved_thinking(template)
    ends_with_assistant = bool(messages) and messages[-1].get("role") == "assistant"

    def render(
        selected_messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> list[int]:
        render_messages = normalize_tool_call_arguments_for_chat_template(
            selected_messages, template
        )
        return _ids(
            resolved_tokenizer.apply_chat_template(
                render_messages,
                tools=history.tools,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **({"chat_template": template} if template is not None else {}),
                **kwargs,
            )
        )

    def probe_render(
        selected_messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> list[int] | None:
        try:
            return render(
                selected_messages, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            return None

    rendered = render(messages, add_generation_prompt=not ends_with_assistant)
    if any(
        message.get("role") == "assistant"
        and isinstance(message.get("reasoning"), str)
        and message["reasoning"]
        and not message.get("reasoning_content")
        for message in messages
    ):
        without_reasoning = deepcopy(messages)
        for message in without_reasoning:
            message.pop("reasoning", None)
        if (
            probe_render(
                without_reasoning,
                add_generation_prompt=not ends_with_assistant,
            )
            == rendered
        ):
            aliased_messages = deepcopy(messages)
            for message in aliased_messages:
                reasoning = message.pop("reasoning", None)
                if isinstance(reasoning, str) and reasoning:
                    message.setdefault("reasoning_content", reasoning)
            aliased_render = probe_render(
                aliased_messages,
                add_generation_prompt=not ends_with_assistant,
            )
            if aliased_render is not None:
                messages = aliased_messages
                rendered = aliased_render
    if any(
        message.get("role") == "assistant"
        and isinstance(message.get("refusal"), str)
        and message["refusal"]
        for message in messages
    ):
        without_refusals = deepcopy(messages)
        for message in without_refusals:
            message.pop("refusal", None)
        if (
            probe_render(
                without_refusals,
                add_generation_prompt=not ends_with_assistant,
            )
            == rendered
        ):
            merged_messages = deepcopy(messages)
            for message in merged_messages:
                refusal = message.pop("refusal", None)
                if not isinstance(refusal, str) or not refusal:
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = content + refusal
                elif isinstance(content, list):
                    message["content"] = [
                        *content,
                        {"type": "text", "text": refusal},
                    ]
                elif content is None:
                    message["content"] = refusal
                else:
                    raise ValueError(
                        "Cannot render an assistant refusal with this content shape"
                    )
            merged_render = probe_render(
                merged_messages,
                add_generation_prompt=not ends_with_assistant,
            )
            if merged_render is not None:
                messages = merged_messages
                rendered = merged_render

    prompt_cache: dict[int, list[int] | None] = {}
    output_cache: dict[int, tuple[list[int] | None, list[float]]] = {}

    def source_prompt_tokens(source: object) -> list[int] | None:
        key = id(source)
        if key not in prompt_cache:
            prompt_cache[key] = _chat_source_prompt_tokens(source)
        return prompt_cache[key]

    def source_output_tokens(
        source: object,
    ) -> tuple[list[int] | None, list[float]]:
        key = id(source)
        if key not in output_cache:
            output_cache[key] = _chat_source_full_tokens(source)
        return output_cache[key]

    def source_matches_context(source: object) -> bool:
        exchange = getattr(source, "exchange", None)
        return (
            isinstance(exchange, ChatCompletionsExchange)
            and history.tools == exchange.request.get("tools")
            and history.chat_template == exchange.request.get("chat_template")
            and history.chat_template_kwargs
            == exchange.request.get("chat_template_kwargs")
        )

    canonical_rendered = rendered
    exact_prefix_length = 0
    canonical_prefix_length = 0
    if (
        chat_template is None
        and chat_template_kwargs is None
        and _projection_matches is True
    ):
        for message_index, (message, source) in enumerate(
            zip(history.messages, history.message_sources, strict=True)
        ):
            if message.get("role") != "assistant" or source is None:
                continue
            source_prompt = source_prompt_tokens(source)
            if source_prompt and source_matches_context(source):
                rendered_prompt = probe_render(
                    messages[:message_index], add_generation_prompt=True
                )
                if (
                    rendered_prompt is not None
                    and rendered[: len(rendered_prompt)] == rendered_prompt
                ):
                    rendered = [
                        *source_prompt,
                        *rendered[len(rendered_prompt) :],
                    ]
                    exact_prefix_length = len(source_prompt)
                    canonical_prefix_length = len(rendered_prompt)
                    break

    positions_by_first_token: dict[int, list[int]] = {}
    for index, token_id in enumerate(rendered):
        positions_by_first_token.setdefault(token_id, []).append(index)
    locations_by_needle: dict[tuple[int, ...], list[tuple[int, int]]] = {}

    def locations(needle: Sequence[int], start: int) -> list[tuple[int, int]]:
        if not needle:
            return []
        key = tuple(needle)
        if key not in locations_by_needle:
            locations_by_needle[key] = [
                (index, index + len(key))
                for index in positions_by_first_token.get(key[0], [])
                if rendered[index : index + len(key)] == list(key)
            ]
        spans = locations_by_needle[key]
        return spans[bisect_left(spans, (start, -1)) :]

    replacements: list[
        tuple[
            int,
            int,
            list[int],
            list[float],
            bool,
            _SampledSourceKey,
            object,
        ]
    ] = []
    search_cursor = 0
    sampled_texts = {
        text
        for message, source in zip(messages, history.message_sources, strict=True)
        if message.get("role") == "assistant"
        and source is not None
        and _source_is_sampled(source)
        for _, text in _chat_message_parts(message)
    }
    part_ids_cache: dict[str, list[int]] = {}

    def part_ids(text: str) -> list[int]:
        if text not in part_ids_cache:
            part_ids_cache[text] = _ids(
                resolved_tokenizer(text, add_special_tokens=False)
            )
        return part_ids_cache[text]

    direct_render: list[int] = []
    direct_bounds: list[tuple[int, int]] = []
    for message in messages:
        start = len(direct_render)
        for _, text in _chat_message_parts(message):
            direct_render.extend(part_ids(text))
        direct_bounds.append((start, len(direct_render)))
    if direct_render != rendered and not (
        exact_prefix_length
        and len(direct_render) == len(rendered)
        and direct_render[exact_prefix_length:] == rendered[exact_prefix_length:]
    ):
        direct_bounds = []

    def canonical_render_to_rendered(probe: Sequence[int]) -> list[int] | None:
        if not exact_prefix_length:
            return list(probe)
        if (
            len(probe) < canonical_prefix_length
            or list(probe[:canonical_prefix_length])
            != canonical_rendered[:canonical_prefix_length]
        ):
            return None
        return [
            *rendered[:exact_prefix_length],
            *probe[canonical_prefix_length:],
        ]

    def canonical_span_to_rendered(start: int, end: int) -> tuple[int, int] | None:
        if not exact_prefix_length:
            return start, end
        if start < canonical_prefix_length:
            return None
        delta = exact_prefix_length - canonical_prefix_length
        return start + delta, end + delta

    def differing_span(probe: Sequence[int]) -> tuple[int, int] | None:
        baseline = canonical_rendered if exact_prefix_length else rendered
        prefix = 0
        while (
            prefix < len(baseline)
            and prefix < len(probe)
            and baseline[prefix] == probe[prefix]
        ):
            prefix += 1
        suffix = 0
        while (
            suffix < len(baseline) - prefix
            and suffix < len(probe) - prefix
            and baseline[-suffix - 1] == probe[-suffix - 1]
        ):
            suffix += 1
        end = len(baseline) - suffix
        span = canonical_span_to_rendered(prefix, end)
        return span if span is not None and span[0] < span[1] else None

    marked_bounds: dict[int, tuple[int, int]] = {}
    marked_part_bounds: dict[int, list[tuple[int, int]]] = {}
    if not direct_bounds:
        marked_messages = deepcopy(messages)
        marker_prefix = f"ART_TRAJECTORY_{id(marked_messages):x}_"
        markers: dict[str, tuple[int, int, Literal["start", "end"]]] = {}
        part_counts: dict[int, int] = {}
        part_whitespace: dict[tuple[int, int], tuple[str, str]] = {}
        for message_index, (message, source) in enumerate(
            zip(marked_messages, history.message_sources, strict=True)
        ):
            if (
                message.get("role") != "assistant"
                or source is None
                or not _source_is_sampled(source)
            ):
                continue
            slot_groups = _chat_message_text_slot_groups(message)
            if not slot_groups:
                continue
            part_counts[message_index] = len(slot_groups)
            for part_index, slots in enumerate(slot_groups):
                start = f"{marker_prefix}{message_index}_{part_index}_START"
                end = f"{marker_prefix}{message_index}_{part_index}_END"
                first, first_key = slots[0]
                last, last_key = slots[-1]
                first_text = str(first[first_key])
                last_text = str(last[last_key])
                leading = first_text[: len(first_text) - len(first_text.lstrip())]
                trailing = last_text[len(last_text.rstrip()) :]
                whitespace_only = (
                    first is last and first_key == last_key and not first_text.strip()
                )
                if whitespace_only:
                    trailing = ""
                if first is last and first_key == last_key:
                    core = first_text[
                        len(leading) : len(first_text) - len(trailing)
                        if trailing
                        else len(first_text)
                    ]
                    first[first_key] = leading + start + core + end + trailing
                else:
                    first[first_key] = leading + start + first_text[len(leading) :]
                    last[last_key] = (
                        last_text[: len(last_text) - len(trailing)] + end + trailing
                    )
                part_whitespace[(message_index, part_index)] = (
                    ("", "") if whitespace_only else (leading, trailing)
                )
                markers[start] = (message_index, part_index, "start")
                markers[end] = (message_index, part_index, "end")
        if markers:
            try:
                marked_text = resolved_tokenizer.apply_chat_template(
                    normalize_tool_call_arguments_for_chat_template(
                        marked_messages, template
                    ),
                    tools=history.tools,
                    tokenize=False,
                    add_generation_prompt=not ends_with_assistant,
                    **({"chat_template": template} if template is not None else {}),
                    **kwargs,
                )
            except Exception:
                marked_text = None
            if isinstance(marked_text, str):
                marker_pattern = re.compile(
                    rf"{re.escape(marker_prefix)}\d+_\d+_(?:START|END)"
                )
                matches = list(marker_pattern.finditer(marked_text))
                found_markers = [match.group(0) for match in matches]
            else:
                matches = []
                found_markers = []
            if (
                isinstance(marked_text, str)
                and len(found_markers) == len(markers)
                and set(found_markers) == set(markers)
            ):
                unmarked_parts: list[str] = []
                char_bounds: dict[tuple[int, int], list[int]] = {}
                source_cursor = 0
                target_cursor = 0
                for match in matches:
                    position = match.start()
                    marker = match.group(0)
                    message_index, part_index, boundary = markers[marker]
                    chunk = marked_text[source_cursor:position]
                    unmarked_parts.append(chunk)
                    target_cursor += len(chunk)
                    char_bounds.setdefault((message_index, part_index), [0, 0])[
                        0 if boundary == "start" else 1
                    ] = target_cursor
                    source_cursor = match.end()
                unmarked_parts.append(marked_text[source_cursor:])
                unmarked_text = "".join(unmarked_parts)
                for key, bounds in char_bounds.items():
                    leading, trailing = part_whitespace[key]
                    if (
                        leading
                        and unmarked_text[max(0, bounds[0] - len(leading)) : bounds[0]]
                        == leading
                    ):
                        bounds[0] -= len(leading)
                    if (
                        trailing
                        and unmarked_text[bounds[1] : bounds[1] + len(trailing)]
                        == trailing
                    ):
                        bounds[1] += len(trailing)
                try:
                    encoded = cast(_OffsetTokenizer, resolved_tokenizer)(
                        unmarked_text,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                except Exception:
                    encoded = None
                encoded_data = _string_dict(encoded)
                raw_offsets = (
                    encoded_data.get("offset_mapping")
                    if encoded_data is not None
                    else None
                )
                encoded_ids = _ids(encoded) if encoded is not None else []
                if (
                    encoded is not None
                    and (
                        encoded_ids == canonical_rendered
                        or (
                            exact_prefix_length
                            and len(encoded_ids) == len(canonical_rendered)
                            and encoded_ids[canonical_prefix_length:]
                            == canonical_rendered[canonical_prefix_length:]
                        )
                    )
                    and isinstance(raw_offsets, list)
                    and len(raw_offsets) == len(encoded_ids)
                ):
                    offsets: list[tuple[int, int]] = []
                    for value in raw_offsets:
                        if (
                            not isinstance(value, (list, tuple))
                            or len(value) != 2
                            or not all(isinstance(item, int) for item in value)
                        ):
                            break
                        offsets.append((value[0], value[1]))
                    if len(offsets) == len(encoded_ids):
                        token_bounds: dict[tuple[int, int], tuple[int, int]] = {}
                        token_cursor = 0
                        for key, (char_start, char_end) in sorted(
                            char_bounds.items(), key=lambda item: item[1]
                        ):
                            while (
                                token_cursor < len(offsets)
                                and offsets[token_cursor][1] <= char_start
                            ):
                                token_cursor += 1
                            token_end = token_cursor
                            while (
                                token_end < len(offsets)
                                and offsets[token_end][0] < char_end
                            ):
                                token_end += 1
                            if char_start == char_end:
                                token_bounds[key] = (token_cursor, token_cursor)
                            elif (
                                token_end > token_cursor
                                and offsets[token_cursor][0] >= char_start
                                and offsets[token_end - 1][1] <= char_end
                            ):
                                token_bounds[key] = (token_cursor, token_end)
                            token_cursor = token_end
                        for message_index, part_count in part_counts.items():
                            bounds = [
                                token_bounds[(message_index, part_index)]
                                for part_index in range(part_count)
                                if (message_index, part_index) in token_bounds
                            ]
                            if len(bounds) == part_count:
                                rendered_bounds = [
                                    canonical_span_to_rendered(*bound)
                                    for bound in bounds
                                ]
                                if any(bound is None for bound in rendered_bounds):
                                    continue
                                translated = cast(
                                    list[tuple[int, int]], rendered_bounds
                                )
                                marked_part_bounds[message_index] = translated
                                marked_bounds[message_index] = (
                                    translated[0][0],
                                    translated[-1][1],
                                )
        for message_index, bounds in list(marked_part_bounds.items()):
            whitespace_parts = [
                (part_index, text)
                for part_index, (_, text) in enumerate(
                    _chat_message_parts(messages[message_index])
                )
                if text and not text.strip()
            ]
            for part_index, _ in whitespace_parts:
                empty_messages = deepcopy(messages)
                groups = _chat_message_text_slot_groups(empty_messages[message_index])
                if part_index >= len(groups):
                    marked_bounds.pop(message_index, None)
                    marked_part_bounds.pop(message_index, None)
                    break
                for container, key in groups[part_index]:
                    container[key] = ""
                try:
                    empty_render = render(
                        empty_messages,
                        add_generation_prompt=not ends_with_assistant,
                    )
                except Exception:
                    marked_bounds.pop(message_index, None)
                    marked_part_bounds.pop(message_index, None)
                    break
                empty_render = canonical_render_to_rendered(empty_render)
                if empty_render is None:
                    marked_bounds.pop(message_index, None)
                    marked_part_bounds.pop(message_index, None)
                    break
                if empty_render == rendered:
                    continue
                anchor = bounds[part_index][0]
                start = anchor - (len(rendered) - len(empty_render))
                span = (start, anchor)
                if (
                    start < 0
                    or rendered[: span[0]] + rendered[span[1] :] != empty_render
                ):
                    marked_bounds.pop(message_index, None)
                    marked_part_bounds.pop(message_index, None)
                    break
                bounds[part_index] = span
                marked_bounds[message_index] = (bounds[0][0], bounds[-1][1])

    probed_bounds: dict[int, tuple[int, int]] = {}
    if not direct_bounds:
        for message_index, (message, source) in enumerate(
            zip(messages, history.message_sources, strict=True)
        ):
            if (
                message_index in marked_bounds
                or message.get("role") != "assistant"
                or source is None
                or not _source_is_sampled(source)
            ):
                continue
            parts = _chat_message_parts(message)
            if not parts:
                exact_output, _ = source_output_tokens(source)
                if exact_output:
                    try:
                        prefix = render(
                            messages[:message_index],
                            add_generation_prompt=True,
                        )
                        completed = render(
                            messages[: message_index + 1],
                            add_generation_prompt=False,
                        )
                    except Exception:
                        continue
                    rendered_prefix = canonical_render_to_rendered(prefix)
                    rendered_completed = canonical_render_to_rendered(completed)
                    if (
                        completed[: len(prefix)] == prefix
                        and rendered_prefix is not None
                        and rendered_completed is not None
                        and rendered[: len(rendered_completed)] == rendered_completed
                    ):
                        probed_bounds[message_index] = (
                            len(rendered_prefix),
                            len(rendered_prefix),
                        )
                continue
            if any(part == "tool_call" for part, _ in parts):
                probe_messages = deepcopy(messages)
                for part_index, slots in enumerate(
                    _chat_message_text_slot_groups(probe_messages[message_index])
                ):
                    for slot_index, (container, key) in enumerate(slots):
                        original = str(container[key])
                        leading = original[: len(original) - len(original.lstrip())]
                        trailing = original[len(original.rstrip()) :]
                        marker = (
                            f"art_trajectory_probe_{id(probe_messages):x}_"
                            f"{part_index}_{slot_index}"
                        )
                        replacement = (
                            json.dumps({marker: True}) if key == "arguments" else marker
                        )
                        container[key] = leading + replacement + trailing
                probe = probe_render(
                    probe_messages,
                    add_generation_prompt=not ends_with_assistant,
                )
                if probe is not None and (span := differing_span(probe)):
                    probed_bounds[message_index] = span
                    continue
            if len(parts) == 1:
                try:
                    prefix = render(
                        messages[:message_index], add_generation_prompt=True
                    )
                except Exception:
                    prefix = []
                local = part_ids(parts[0][1])
                rendered_prefix = canonical_render_to_rendered(prefix)
                if rendered_prefix is not None and rendered == [
                    *rendered_prefix,
                    *local,
                ]:
                    probed_bounds[message_index] = (
                        len(rendered_prefix),
                        len(rendered),
                    )
                    continue
                try:
                    completed = render(
                        messages[: message_index + 1],
                        add_generation_prompt=False,
                    )
                except Exception:
                    completed = []
                rendered_completed = canonical_render_to_rendered(completed)
                if (
                    completed == [*prefix, *local]
                    and rendered_prefix is not None
                    and rendered_completed is not None
                    and rendered[: len(rendered_completed)] == rendered_completed
                ):
                    probed_bounds[message_index] = (
                        len(rendered_prefix),
                        len(rendered_completed),
                    )
                    continue
            probe_messages = deepcopy(messages)
            slot_groups = _chat_message_text_slot_groups(probe_messages[message_index])
            if len(slot_groups) != 1 or len(slot_groups[0]) != 1:
                continue
            container, key = slot_groups[0][0]
            original = str(container[key])
            leading = original[: len(original) - len(original.lstrip())]
            trailing = original[len(original.rstrip()) :]
            container[key] = (
                leading + f"ART_TRAJECTORY_{id(probe_messages):x}_PROBE" + trailing
            )
            try:
                probe = render(
                    probe_messages,
                    add_generation_prompt=not ends_with_assistant,
                )
            except Exception:
                continue
            if span := differing_span(probe):
                probed_bounds[message_index] = span

    sampled_message_count = sum(
        message.get("role") == "assistant"
        and source is not None
        and _source_is_sampled(source)
        for message, source in zip(messages, history.message_sources, strict=True)
    )
    for message_index, (message, source) in enumerate(
        zip(messages, history.message_sources, strict=True)
    ):
        parts = _chat_message_parts(message)
        sampled = (
            message.get("role") == "assistant"
            and source is not None
            and _source_is_sampled(source)
        )
        full_exact, full_logprobs = (
            source_output_tokens(source) if sampled else (None, [])
        )
        if sampled and not parts and not full_exact:
            continue
        complete_sampled_message = (
            sampled
            and source is not None
            and _source_covers_complete_sampled_message(
                history.messages[message_index], source
            )
        )
        if (
            complete_sampled_message
            and full_exact is not None
            and message_index in marked_bounds
            and isinstance(getattr(source, "exchange", None), ChatCompletionsExchange)
            and marked_bounds[message_index][1] - marked_bounds[message_index][0]
            != len(full_exact)
        ):
            try:
                prefix = render(messages[:message_index], add_generation_prompt=True)
                completed = render(
                    messages[: message_index + 1], add_generation_prompt=False
                )
            except Exception:
                marked_bounds.pop(message_index, None)
                marked_part_bounds.pop(message_index, None)
                prefix = completed = None

            rendered_prefix = (
                canonical_render_to_rendered(prefix) if prefix is not None else None
            )
            rendered_completed = (
                canonical_render_to_rendered(completed)
                if completed is not None
                else None
            )
            corrected_bounds = (
                canonical_span_to_rendered(len(prefix), len(completed))
                if prefix is not None and completed is not None
                else None
            )
            if (
                prefix is not None
                and completed is not None
                and rendered_prefix is not None
                and rendered_completed is not None
                and corrected_bounds is not None
                and rendered[: len(rendered_prefix)] == rendered_prefix
                and rendered[: len(rendered_completed)] == rendered_completed
            ):
                marked_bounds[message_index] = corrected_bounds
                # The marker-derived per-part offsets describe the old render.
                marked_part_bounds.pop(message_index, None)
            else:
                marked_bounds.pop(message_index, None)
                marked_part_bounds.pop(message_index, None)
        source_boundary = False
        generation_start: int | None = None
        sampled_bounds: tuple[int, int] | None = None
        content_bounds_proven = False
        if sampled:
            if direct_bounds:
                sampled_bounds = direct_bounds[message_index]
                content_bounds_proven = True
            elif message_index in marked_bounds:
                sampled_bounds = marked_bounds[message_index]
                content_bounds_proven = True
            elif message_index in probed_bounds:
                sampled_bounds = probed_bounds[message_index]
                content_bounds_proven = True
            else:
                assert source is not None
                source_prompt = source_prompt_tokens(source)
                source_context_matches = source_matches_context(source)
                if (
                    source_context_matches
                    and source_prompt
                    and rendered[: len(source_prompt)] == source_prompt
                ):
                    search_cursor = max(search_cursor, len(source_prompt))
                    source_boundary = True
                    generation_start = len(source_prompt)
                    sampled_bounds = (generation_start, len(rendered))
                elif sampled_message_count == 1:
                    prompt_render = probe_render(
                        messages[:message_index], add_generation_prompt=True
                    )
                    rendered_prompt = (
                        canonical_render_to_rendered(prompt_render)
                        if prompt_render is not None
                        else None
                    )
                    if (
                        rendered_prompt is None
                        or rendered[: len(rendered_prompt)] != rendered_prompt
                    ):
                        raise ValueError(
                            "Could not locate a sampled history message in the "
                            "rendered history"
                        )
                    generation_start = len(rendered_prompt)
                    sampled_bounds = (generation_start, len(rendered))
                else:
                    raise ValueError(
                        "Could not prove a sampled history message boundary with this "
                        "tokenizer"
                    )
        full_matches = (
            locations(full_exact, search_cursor) if sampled and full_exact else []
        )
        first_part_matches = (
            locations(part_ids(parts[0][1]), search_cursor) if parts else []
        )
        if sampled_bounds is not None:
            lower, upper = sampled_bounds
            full_matches = [
                match
                for match in full_matches
                if match[0] >= lower and match[1] <= upper
            ]
            first_part_matches = [
                match
                for match in first_part_matches
                if match[0] >= lower and match[1] <= upper
            ]
        if (
            full_exact is not None
            and full_matches
            and (
                (source_boundary and full_matches[0][0] == search_cursor)
                or (
                    complete_sampled_message
                    and len(full_matches) == 1
                    and first_part_matches
                    and full_matches[0][0] == first_part_matches[0][0]
                )
            )
        ):
            span = full_matches[0]
            start, end = span
            replacements.append(
                (
                    start,
                    end,
                    full_exact,
                    full_logprobs
                    if len(full_logprobs) == len(full_exact)
                    else [math.nan] * len(full_exact),
                    True,
                    _sampled_source_key(source),
                    source,
                )
            )
            search_cursor = end
            continue

        source_exchange = getattr(source, "exchange", None)
        multi_generation_response = (
            isinstance(source_exchange, ResponsesExchange)
            and len(_response_generations(source_exchange.response)) > 1
        )
        if sampled and not content_bounds_proven:
            raise ValueError(
                "Could not uniquely locate or prove the sampled content boundary "
                f"for history message {message_index} with this tokenizer"
            )
        if (
            sampled
            and full_exact is None
            and message_index in probed_bounds
            and parts
            and any(part == "tool_call" for part, _ in parts)
        ):
            assert source is not None and sampled_bounds is not None
            start, end = sampled_bounds
            replacements.append(
                (
                    start,
                    end,
                    rendered[start:end],
                    [math.nan] * (end - start),
                    False,
                    _sampled_source_key(source),
                    source,
                )
            )
            search_cursor = end
            continue
        if (
            complete_sampled_message
            and full_exact is not None
            and (
                multi_generation_response
                or len(parts) != 1
                or len(full_exact) != len(part_ids(parts[0][1]))
            )
        ):
            if not parts and sampled_bounds is not None:
                start = end = sampled_bounds[0]
            elif sampled_bounds is None:
                raise ValueError(
                    "Could not locate a complete sampled message in the rendered history"
                )
            elif sampled_bounds[0] == sampled_bounds[1]:
                if not content_bounds_proven:
                    raise ValueError(
                        "Could not locate a complete sampled message in the rendered "
                        "history"
                    )
                start = end = sampled_bounds[0]
            else:
                start, end = sampled_bounds
            if parts and len(parts) == 1 and parts[0][0] == "content":
                visible_matches = [
                    match
                    for match in locations(part_ids(parts[0][1]), start)
                    if match[1] <= end
                ]
                if len(visible_matches) != 1 or (
                    not content_bounds_proven
                    and generation_start is not None
                    and visible_matches[0][0] != generation_start
                ):
                    raise ValueError(
                        "Could not prove the sampled content boundary in the "
                        "rendered history"
                    )
                start, end = visible_matches[0]
            elif generation_start is not None and (
                multi_generation_response or len(parts) != 1 or parts[0][0] != "content"
            ):
                start = generation_start
            replacements.append(
                (
                    start,
                    end,
                    full_exact,
                    full_logprobs
                    if len(full_logprobs) == len(full_exact)
                    else [math.nan] * len(full_exact),
                    True,
                    _sampled_source_key(source),
                    source,
                )
            )
            if (
                message_index < len(messages) - 1
                and isinstance(source_exchange, ChatCompletionsExchange)
                and rendered[start:end] != full_exact
            ):
                _warn_prefix_retokenization()
            search_cursor = end
            continue

        replacement_start = len(replacements)
        for part_index, (part, text) in enumerate(parts):
            if not sampled and text not in sampled_texts:
                continue
            local = part_ids(text)
            if not local:
                continue
            proven_part_bounds = marked_part_bounds.get(message_index)
            span = (
                proven_part_bounds[part_index]
                if proven_part_bounds is not None
                else next(iter(locations(local, search_cursor)), None)
            )
            if span is None:
                if not sampled:
                    continue
                raise ValueError(
                    "Could not locate a history message in the rendered history"
                )
            if sampled_bounds is not None:
                lower, upper = sampled_bounds
                if proven_part_bounds is None:
                    bounded_matches = [
                        match
                        for match in locations(local, max(lower, search_cursor))
                        if match[1] <= upper
                    ]
                    if len(bounded_matches) != 1:
                        raise ValueError(
                            "Could not uniquely locate a sampled history message in "
                            "the rendered history"
                        )
                    span = bounded_matches[0]
            start, end = span
            search_cursor = end
            if not sampled:
                continue
            assert source is not None
            exact, logprobs = _chat_source_tokens(
                source,
                text,
                part=part,
                full_tokens=(full_exact, full_logprobs),
            )
            if exact is not None and rendered[start : start + len(exact)] == exact:
                end = start + len(exact)
                search_cursor = end
            replacement = exact if exact is not None else rendered[start:end]
            if exact is None and not logprobs:
                exchange = getattr(source, "exchange", None)
                if isinstance(
                    exchange,
                    (ChatCompletionsExchange, ResponsesExchange, MessagesExchange),
                ):
                    evidence = _visible_token_evidence(
                        tokenizer,
                        exchange,
                        source=source,
                        sampled_text=text,
                    )
                    if evidence is not None:
                        replacement, logprobs = evidence
                    else:
                        logprobs = (
                            _align_visible_logprobs(
                                tokenizer,
                                replacement,
                                exchange,
                                source=source,
                                sampled_text=text,
                            )
                            or []
                        )
            replacements.append(
                (
                    start,
                    end,
                    replacement,
                    logprobs
                    if len(logprobs) == len(replacement)
                    else [math.nan] * len(replacement),
                    exact is not None,
                    _sampled_source_key(source),
                    source,
                )
            )
        message_replacements = replacements[replacement_start:]
        if (
            sampled
            and message_replacements
            and all(part == "tool_call" for part, _ in parts)
            and not (
                all(replacement[4] for replacement in message_replacements)
                and all(
                    left[1] == right[0]
                    for left, right in zip(
                        message_replacements,
                        message_replacements[1:],
                        strict=False,
                    )
                )
            )
        ):
            start = message_replacements[0][0]
            end = message_replacements[-1][1]
            del replacements[replacement_start:]
            replacements.append(
                (
                    start,
                    end,
                    rendered[start:end],
                    [math.nan] * (end - start),
                    False,
                    message_replacements[0][5],
                    message_replacements[0][6],
                )
            )
        if sampled and not parts and full_exact is not None:
            raise ValueError(
                "Could not locate exact sampled output in the rendered history"
            )

    token_ids: list[int] = []
    logprobs: list[float] = []
    flags: list[TokenFlag] = []
    source_keys: list[_SampledSourceKey | None] = []
    sources: dict[_SampledSourceKey, object] = {}
    cursor = 0
    for (
        start,
        end,
        replacement,
        replacement_logprobs,
        exact,
        source_key,
        source,
    ) in sorted(replacements, key=lambda item: (item[0], item[1])):
        if start < cursor:
            raise ValueError("Rendered assistant source spans overlap")
        token_ids.extend(rendered[cursor:start])
        logprobs.extend([math.nan] * (start - cursor))
        flags.extend([TokenFlag(0)] * (start - cursor))
        source_keys.extend([None] * (start - cursor))
        token_ids.extend(replacement)
        logprobs.extend(replacement_logprobs)
        flag = TokenFlag.SAMPLED | (TokenFlag.EXACT if exact else TokenFlag(0))
        flags.extend([flag] * len(replacement))
        source_keys.extend([source_key] * len(replacement))
        sources[source_key] = source
        cursor = end
    token_ids.extend(rendered[cursor:])
    logprobs.extend([math.nan] * (len(rendered) - cursor))
    flags.extend([TokenFlag(0)] * (len(rendered) - cursor))
    source_keys.extend([None] * (len(rendered) - cursor))
    for index in range(min(exact_prefix_length, len(flags))):
        flags[index] |= TokenFlag.EXACT
    if history.model is None:
        raise ValueError("History tokenization requires a model")
    tokenized = TokenizedHistory(
        history=history,
        model=history.model,
        tokens=token_ids,
        logprobs=logprobs,
        flags=flags,
    )
    if _trace is not None:
        _trace.set(tokenized, source_keys, sources)
    return tokenized


def _tokenize_completions_token_history(
    history: CompletionsTokenHistory,
    *,
    _trace: _TraceBuilder | None = None,
) -> TokenizedHistory:
    if any(
        span.start < 0 or span.end <= span.start or span.end > len(history.prompt)
        for span in history.prompt_sources
    ):
        raise ValueError("Completions token source spans are out of bounds")
    if not _spans_are_exhaustive(len(history.prompt), history.prompt_sources):
        raise ValueError(
            "Completions token source spans must exhaustively cover prompt"
        )
    _validate_completions_sources(
        model=history.model,
        source_spans=history.prompt_sources,
        sampled_spans=history.sampled_spans,
    )

    flags = [TokenFlag(0)] * len(history.prompt)
    logprobs = [math.nan] * len(history.prompt)
    source_keys: list[_SampledSourceKey | None] = [None] * len(history.prompt)
    sources: dict[_SampledSourceKey, object] = {}
    for start, end in history.sampled_spans:
        if start < 0 or end <= start or end > len(history.prompt):
            raise ValueError("Completions sampled spans are out of bounds")
        flags[start:end] = [TokenFlag.SAMPLED] * (end - start)
    for span in history.prompt_sources:
        if span.source is None:
            continue
        if span.source.choice_index is not None:
            source_key = _sampled_source_key(span.source)
            source_keys[span.start : span.end] = [source_key] * (span.end - span.start)
            sources[source_key] = span.source
        flags[span.start : span.end] = [
            flag | TokenFlag.EXACT for flag in flags[span.start : span.end]
        ]
        prompt, completion, prompt_logprobs, completion_logprobs = (
            _completion_source_evidence(span.source)
        )
        selected = prompt if span.source.choice_index is None else completion
        selected_logprobs = (
            prompt_logprobs if span.source.choice_index is None else completion_logprobs
        )
        if selected != history.prompt[span.start : span.end]:
            if span.source.choice_index is not None:
                raise ValueError(
                    "Completions sampled output no longer matches its source exchange"
                )
            flags[span.start : span.end] = [
                flag & ~TokenFlag.EXACT for flag in flags[span.start : span.end]
            ]
            continue
        if len(selected_logprobs) == span.end - span.start:
            logprobs[span.start : span.end] = selected_logprobs
    tokenized = TokenizedHistory(
        history=history,
        model=history.model,
        tokens=list(history.prompt),
        logprobs=logprobs,
        flags=flags,
    )
    if _trace is not None:
        _trace.set(tokenized, source_keys, sources)
    return tokenized


def _completion_visible_logprobs(
    source: CompletionsSource,
    text: str,
    tokenizer: Tokenizer,
    token_ids: list[int],
) -> list[float] | None:
    choice = next(
        item
        for item in source.exchange.response.choices
        if item.index == source.choice_index
    )
    data = _dump(choice.logprobs)
    raw_tokens = data.get("tokens")
    raw_logprobs = data.get("token_logprobs")
    if not isinstance(raw_tokens, list) or not isinstance(raw_logprobs, list):
        return None
    values = list(zip(raw_tokens, raw_logprobs, strict=False))
    from ._history import _completion_prompts

    request_prompts = _completion_prompts(source.exchange.request.get("prompt"))
    request_prompt = request_prompts[source.prompt_index]
    if source.exchange.request.get("echo") is True and isinstance(request_prompt, str):
        consumed = ""
        cursor = 0
        while cursor < len(values) and len(consumed) < len(request_prompt):
            token_text = values[cursor][0]
            if not isinstance(token_text, str):
                return None
            consumed += token_text
            cursor += 1
        if consumed != request_prompt:
            return None
        values = values[cursor:]
    if (
        "".join(token_text for token_text, _ in values if isinstance(token_text, str))
        != text
    ):
        return None
    aligned_ids: list[int] = []
    aligned_logprobs: list[float] = []
    for token_text, logprob in values:
        if (
            not isinstance(token_text, str)
            or not isinstance(logprob, (int, float))
            or isinstance(logprob, bool)
        ):
            return None
        encoded = _ids(tokenizer(token_text, add_special_tokens=False))
        if len(encoded) != 1:
            return None
        aligned_ids.append(encoded[0])
        aligned_logprobs.append(float(logprob))
    return aligned_logprobs if aligned_ids == token_ids else None


def _tokenize_completions_string_history(
    history: CompletionsStringHistory,
    *,
    base_model: str | None,
    tokenizer: Tokenizer | None,
    _trace: _TraceBuilder | None = None,
) -> TokenizedHistory:
    if not _spans_are_exhaustive(len(history.prompt), history.prompt_sources):
        raise ValueError(
            "Completions string source spans must exhaustively cover prompt"
        )
    _validate_completions_sources(
        model=history.model,
        source_spans=history.prompt_sources,
        sampled_spans=history.sampled_spans,
    )
    sampled = [False] * len(history.prompt)
    for start, end in history.sampled_spans:
        if start < 0 or end <= start or end > len(history.prompt):
            raise ValueError("Completions sampled spans are out of bounds")
        sampled[start:end] = [True] * (end - start)

    config: _TokenizerConfig | None = None

    def resolved_tokenizer() -> Tokenizer:
        nonlocal config, tokenizer
        if tokenizer is None:
            config = config or _tokenizer_config(history.model, base_model)
            tokenizer = _load_tokenizer(config)
        return tokenizer

    token_ids: list[int] = []
    logprobs: list[float] = []
    flags: list[TokenFlag] = []
    source_keys: list[_SampledSourceKey | None] = []
    sources: dict[_SampledSourceKey, object] = {}
    for span in history.prompt_sources:
        text = history.prompt[span.start : span.end]
        source = span.source
        exact: list[int] | None = None
        source_logprobs: list[float] = []
        is_sampled = any(sampled[span.start : span.end])
        if source is not None:
            prompt, completion, prompt_logprobs, completion_logprobs = (
                _completion_source_evidence(source)
            )
            if source.choice_index is None:
                from ._history import _completion_prompts

                request_prompts = _completion_prompts(
                    source.exchange.request.get("prompt")
                )
                original = request_prompts[source.prompt_index]
                if not isinstance(original, str):
                    raise ValueError(
                        "A string Completions history cannot reference a token prompt"
                    )
                if text == original:
                    exact = prompt
                    source_logprobs = prompt_logprobs
            else:
                choice = next(
                    item
                    for item in source.exchange.response.choices
                    if item.index == source.choice_index
                )
                expected = choice.text
                from ._history import _completion_prompts

                request_prompts = _completion_prompts(
                    source.exchange.request.get("prompt")
                )
                request_prompt = request_prompts[source.prompt_index]
                if source.exchange.request.get("echo") is True:
                    if not isinstance(request_prompt, str) or not expected.startswith(
                        request_prompt
                    ):
                        raise ValueError(
                            "Cannot locate echoed Completions prompt boundary"
                        )
                    expected = expected[len(request_prompt) :]
                if text != expected:
                    raise ValueError(
                        "Completions history text no longer matches its source exchange"
                    )
                exact = completion
                source_logprobs = completion_logprobs
        ids = (
            exact
            if exact is not None
            else _ids(resolved_tokenizer()(text, add_special_tokens=False))
        )
        token_ids.extend(ids)
        if exact is not None and len(source_logprobs) == len(ids):
            logprobs.extend(source_logprobs)
        else:
            visible = (
                _completion_visible_logprobs(source, text, resolved_tokenizer(), ids)
                if source is not None and source.choice_index is not None
                else None
            )
            logprobs.extend(visible or [math.nan] * len(ids))
        flag = TokenFlag.SAMPLED if is_sampled else TokenFlag(0)
        if exact is not None:
            flag |= TokenFlag.EXACT
        flags.extend([flag] * len(ids))
        if is_sampled:
            if source is None:
                raise AssertionError("Sampled Completions span has no source")
            source_key = _sampled_source_key(source)
            source_keys.extend([source_key] * len(ids))
            sources[source_key] = source
        else:
            source_keys.extend([None] * len(ids))
    tokenized = TokenizedHistory(
        history=history,
        model=history.model,
        tokens=token_ids,
        logprobs=logprobs,
        flags=flags,
    )
    if _trace is not None:
        _trace.set(tokenized, source_keys, sources)
    return tokenized


def _completion_source_evidence(
    source: CompletionsSource,
) -> tuple[list[int] | None, list[int] | None, list[float], list[float]]:
    from ._history import _completion_choice_groups

    prompt_groups = _completion_choice_groups(source.exchange)
    prompt_index = _source_index(
        source.prompt_index,
        length=len(prompt_groups),
        field="Completions prompt source index",
    )
    if source.choice_index is None:
        selected = prompt_groups[prompt_index][0]
    else:
        if (
            not isinstance(source.choice_index, int)
            or isinstance(source.choice_index, bool)
            or source.choice_index < 0
        ):
            raise ValueError("Completions choice source index is invalid")
        selected = next(
            (
                choice
                for choice in prompt_groups[prompt_index]
                if choice.index == source.choice_index
            ),
            None,
        )
        if selected is None:
            raise ValueError("Completions choice source does not belong to its prompt")
    return _completion_evidence(
        source.exchange.response.model_copy(update={"choices": [selected]}),
        echo=source.exchange.request.get("echo") is True,
        empty_prompt_is_exact=source.exchange.request.get("prompt") in ("", []),
    )


def _validate_completions_sources(
    *,
    model: str,
    source_spans: Sequence[CompletionsTokenSourceSpan | CompletionsStringSourceSpan],
    sampled_spans: Sequence[tuple[int, int]],
) -> None:
    expected_sampled: list[tuple[int, int]] = []
    for span in source_spans:
        source = getattr(span, "source", None)
        if source is None:
            continue
        if not isinstance(source, CompletionsSource):
            raise ValueError("Completions history has an invalid source")
        if source.exchange.model != model:
            raise ValueError(
                "Completions history model no longer matches its source exchange"
            )
        if source.choice_index is not None:
            expected_sampled.append((span.start, span.end))
    if list(sampled_spans) != expected_sampled:
        raise ValueError(
            "Completions sampled spans must exactly match choice-backed source spans"
        )


def _spans_are_exhaustive(length: int, spans: Sequence[object]) -> bool:
    cursor = 0
    for span in spans:
        start = getattr(span, "start", None)
        end = getattr(span, "end", None)
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start != cursor
            or end <= start
            or end > length
        ):
            return False
        cursor = end
    return cursor == length


def _tokenize_history(
    history: History | LegacyHistory,
    *,
    model: str | None,
    base_model: str | None,
    tokenizer: Tokenizer | None,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
    _trace: _TraceBuilder | None = None,
    _projection_validated: bool = False,
) -> TokenizedHistory:
    if isinstance(history, LegacyHistory):
        if model is None:
            raise ValueError("Legacy history tokenization requires model=")
        return _legacy_tokenize(history, model=model)
    if model is None:
        raise ValueError("History tokenization requires a model")
    if isinstance(history, CompletionsTokenHistory):
        return _tokenize_completions_token_history(history, _trace=_trace)
    if isinstance(history, CompletionsStringHistory):
        return _tokenize_completions_string_history(
            history,
            base_model=base_model,
            tokenizer=tokenizer,
            _trace=_trace,
        )
    _validate_history_sources(history)
    override_requires_render = (
        chat_template is not None
        and chat_template != getattr(history, "chat_template", None)
    ) or (
        chat_template_kwargs is not None
        and dict(chat_template_kwargs)
        != (getattr(history, "chat_template_kwargs", None) or {})
    )
    render_state = (
        _HistoryRenderState(needs_render=False, projection_matches=True)
        if _projection_validated
        else _history_render_state(history)
    )
    needs_render = render_state.needs_render or override_requires_render
    if isinstance(history, ResponsesHistory) and not needs_render:
        if exact := _tokenize_exact_responses_history(
            history, base_model=base_model, tokenizer=tokenizer, _trace=_trace
        ):
            return exact
    if isinstance(history, ChatCompletionsHistory):
        if (
            not override_requires_render
            and not render_state.context_changed
            and render_state.projection_matches is not False
            and (
                exact := _tokenize_exact_projected_chat_history(
                    history,
                    projection_validated=(
                        _projection_validated or render_state.projection_matches is True
                    ),
                    _trace=_trace,
                )
            )
        ):
            return exact
        return _tokenize_chat_view(
            history,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            _projection_matches=(
                True if _projection_validated else render_state.projection_matches
            ),
            _trace=_trace,
        )
    if isinstance(history, AnthropicMessagesHistory) and needs_render:
        converted = history.as_chat_completions_history()
        if (
            not override_requires_render
            and not render_state.context_changed
            and (
                render_state.projection_matches
                if render_state.projection_matches is not None
                else _history_matches_projection(history)
            )
            and (
                exact := _tokenize_exact_projected_chat_history(
                    converted,
                    projection_validated=True,
                    _trace=_trace,
                )
            )
        ):
            return exact
        return _tokenize_chat_view(
            converted,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            _trace=_trace,
        )
    if isinstance(history, ResponsesHistory) and needs_render:
        return _tokenize_chat_view(
            history.as_chat_completions_history(),
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            _trace=_trace,
        )
    trajectory = _trajectory_from_history(history)
    history_template = getattr(history, "chat_template", None)
    history_kwargs = getattr(history, "chat_template_kwargs", None)
    return _tokenize_exchange_trajectory(
        trajectory,
        history,
        base_model,
        model=model,
        chat_template=(
            chat_template if chat_template is not None else history_template
        ),
        chat_template_kwargs={
            **(history_kwargs or {}),
            **(chat_template_kwargs or {}),
        }
        or None,
        tokenizer_instance=tokenizer,
        _trace=_trace,
    )


def tokenize_history(
    history: History | LegacyHistory,
    *,
    model: str | None,
    base_model: str | None,
    tokenizer: Tokenizer | None,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
    _trace: _TraceBuilder | None = None,
    _projection_validated: bool = False,
) -> TokenizedHistory:
    tokenized = _tokenize_history(
        history,
        model=model,
        base_model=base_model,
        tokenizer=tokenizer,
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
        _trace=_trace,
        _projection_validated=_projection_validated,
    )
    # Internal protocol conversion is an implementation detail. The source is
    # always the public history view the caller asked to tokenize.
    if not isinstance(
        history,
        (
            LegacyHistory,
            ChatCompletionsHistory,
            AnthropicMessagesHistory,
            ResponsesHistory,
            CompletionsTokenHistory,
            CompletionsStringHistory,
        ),
    ):
        raise TypeError(f"Unsupported history type: {type(history).__name__}")
    tokenized.history = history
    return tokenized


def _materialize_trajectory(
    tokenized: TokenizedHistory, trajectory: Trajectory
) -> TokenizedTrajectory:
    return TokenizedTrajectory(
        history=tokenized.history,
        model=tokenized.model,
        tokens=tokenized.tokens,
        logprobs=tokenized.logprobs,
        flags=tokenized.flags,
        trajectory=trajectory,
    )


def tokenize_trajectory(
    trajectory: Trajectory,
    *,
    multi_history: bool,
    reconcile_text_equivalent_tokenizations: bool,
    model: str | None,
    base_model: str | None,
    tokenizer: Tokenizer | None,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
) -> TokenizedTrajectory | TokenizedMultiHistoryTrajectory:
    histories = trajectory.histories(
        model=model,
        reconcile_text_equivalent_tokenizations=reconcile_text_equivalent_tokenizations,
    )
    if not multi_history:
        if len(histories) != 1:
            selected_models = {
                history.model
                for history in histories
                if not isinstance(history, LegacyHistory)
            }
            if model is None and len(selected_models) > 1:
                raise ValueError(
                    "Trajectory tokenization requires exactly one model; pass model= to select one"
                )
            raise ValueError(
                f"Trajectory tokenization requires exactly one history; found {len(histories)}"
            )
    tokenized = [
        tokenize_history(
            history,
            model=model if isinstance(history, LegacyHistory) else history.model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            _projection_validated=not isinstance(history, LegacyHistory),
        )
        for history in histories
    ]
    if not multi_history:
        return _materialize_trajectory(tokenized[0], trajectory)
    return TokenizedMultiHistoryTrajectory(
        trajectory=trajectory,
        histories=tokenized,
    )


def _tokenize_trajectory_with_trace(
    trajectory: Trajectory,
    *,
    model: str | None = None,
    base_model: str | None = None,
    tokenizer: Tokenizer | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: Mapping[str, object] | None = None,
) -> tuple[
    TokenizedMultiHistoryTrajectory,
    list[_HistoryTokenizationTrace],
]:
    if not trajectory.exchanges:
        raise ValueError("Private exchange tokenization trace requires exchanges")
    histories = trajectory.histories(model=model)
    tokenized_histories: list[TokenizedHistory] = []
    traces: list[_HistoryTokenizationTrace] = []
    for history in histories:
        if isinstance(history, LegacyHistory):
            raise AssertionError(
                "Exchange trajectories cannot produce legacy histories"
            )
        trace_builder = _TraceBuilder()
        tokenized = tokenize_history(
            history,
            model=history.model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            _trace=trace_builder,
            _projection_validated=True,
        )
        if trace_builder.trace is None:
            raise AssertionError("Exchange tokenization did not produce a source trace")
        tokenized_histories.append(tokenized)
        traces.append(trace_builder.trace)
    return (
        TokenizedMultiHistoryTrajectory(
            trajectory=trajectory,
            histories=tokenized_histories,
        ),
        traces,
    )


def tokenize_group(
    group: TrajectoryGroup,
    *,
    multi_history: bool,
    reconcile_text_equivalent_tokenizations: bool,
    model: str | None,
    base_model: str | None,
    tokenizer: Tokenizer | None,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
) -> (
    TokenizedTrajectoryGroup[TokenizedTrajectory]
    | TokenizedTrajectoryGroup[TokenizedMultiHistoryTrajectory]
):
    trajectories = [
        tokenize_trajectory(
            trajectory,
            multi_history=multi_history,
            reconcile_text_equivalent_tokenizations=reconcile_text_equivalent_tokenizations,
            model=model,
            base_model=base_model,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        for trajectory in group.trajectories
    ]
    if multi_history:
        return TokenizedTrajectoryGroup[TokenizedMultiHistoryTrajectory](
            trajectory_group=group,
            trajectories=trajectories,
        )
    return TokenizedTrajectoryGroup[TokenizedTrajectory](
        trajectory_group=group,
        trajectories=trajectories,
    )
