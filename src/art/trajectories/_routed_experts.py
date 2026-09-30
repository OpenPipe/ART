"""Align captured expert IDs only along their exact causal token prefix."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import TokenFlag


def validate_routes(routes: list, length: int) -> tuple[int, int] | None:
    if len(routes) != length:
        raise ValueError("routed_experts differs in length from token IDs")
    shape = None
    for token in routes:
        if not isinstance(token, list) or not token:
            raise ValueError("routed_experts must have nonempty layer and top-k axes")
        for layer in token:
            if not isinstance(layer, list) or not layer:
                raise ValueError(
                    "routed_experts must have nonempty layer and top-k axes"
                )
            if any(
                type(expert) is not int or not -1 <= expert <= 65535 for expert in layer
            ):
                raise ValueError("routed_experts IDs must be integers in [-1, 65535]")
            candidate = (len(token), len(layer))
            if shape is not None and shape != candidate:
                raise ValueError("routed_experts must have one rectangular shape")
            shape = candidate
    return shape


def _list(value: Any) -> list:
    # ART's legacy binary attachment is a NumPy array; Caladan uses JSON arrays.
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("routed_experts must be a token-major array")
    return value


def choice_routes(choice: Any, response: Any) -> tuple[list[int], list] | None:
    extra = getattr(choice, "model_extra", None) or {}
    response_extra = getattr(response, "model_extra", None) or {}
    legacy = extra.get("art_moe_routing")
    prompt_routes = extra.get(
        "prompt_routed_experts", response_extra.get("prompt_routed_experts")
    )
    completion_routes = extra.get("routed_experts")
    if isinstance(legacy, dict) and legacy.get("routed_experts") is not None:
        prompt = legacy.get("prompt_token_ids")
        completion = legacy.get("completion_token_ids")
        routes = _list(legacy["routed_experts"])
        if not isinstance(prompt, list) or not isinstance(completion, list):
            raise ValueError("Captured routes require prompt and completion token IDs")
        if len(routes) not in (
            len(prompt) + len(completion),
            len(prompt) + max(0, len(completion) - 1),
        ):
            raise ValueError("Captured routes differ in length from token IDs")
        for actual, expected in (
            (
                extra.get("prompt_token_ids", response_extra.get("prompt_token_ids")),
                prompt,
            ),
            (extra.get("token_ids"), completion),
        ):
            if actual is not None and actual != expected:
                raise ValueError("Captured routes disagree with response token IDs")
        shape = validate_routes(routes, len(routes))
        if shape is not None and len(routes) < len(prompt) + len(completion):
            routes = [*routes, [[-1] * shape[1] for _ in range(shape[0])]]
        if prompt_routes is not None or completion_routes is not None:
            native = choice.model_copy(update={"art_moe_routing": None})
            other = choice_routes(native, response)
            if other is not None and other != (prompt + completion, routes):
                raise ValueError("ART and Caladan routed experts disagree")
        return prompt + completion, routes
    if prompt_routes is None and completion_routes is None:
        return None
    prompt = extra.get("prompt_token_ids", response_extra.get("prompt_token_ids"))
    completion = extra.get("token_ids")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        raise ValueError("Captured routes require prompt and completion token IDs")
    prompt_rows = _list(prompt_routes) if prompt_routes is not None else None
    completion_rows = (
        _list(completion_routes) if completion_routes is not None else None
    )
    shape = None
    for rows, length in (
        (prompt_rows, len(prompt)),
        (completion_rows, len(completion)),
    ):
        if rows is None:
            continue
        current = validate_routes(rows, length)
        if shape is not None and current is not None and current != shape:
            raise ValueError("Prompt and completion routed_experts shapes disagree")
        shape = current or shape
    if shape is None:
        return None
    layers, width = shape

    def missing(count):
        return [[[-1] * width for _ in range(layers)] for _ in range(count)]

    return prompt + completion, (
        (prompt_rows if prompt_rows is not None else missing(len(prompt)))
        + (completion_rows if completion_rows is not None else missing(len(completion)))
    )


def history_choices(history: Any) -> list[tuple[Any, Any]]:
    from openai.types.chat.chat_completion import Choice

    from . import ChatCompletionsExchange, CompletionsExchange, LegacyHistory

    records = []
    if isinstance(history, LegacyHistory):
        records = [
            (choice, None)
            for choice in history.messages_and_choices
            if isinstance(choice, Choice)
        ]
    else:
        sources = getattr(history, "message_sources", None)
        if sources is None:
            sources = [span.source for span in getattr(history, "prompt_sources", ())]
        seen = set()
        for source in sources:
            exchange = getattr(source, "exchange", None)
            index = getattr(source, "choice_index", None)
            if (
                not isinstance(exchange, (ChatCompletionsExchange, CompletionsExchange))
                or index is None
            ):
                continue
            key = (id(exchange), index)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                (
                    next(
                        choice
                        for choice in exchange.response.choices
                        if choice.index == index
                    ),
                    exchange.response,
                )
            )
    return records


def history_routes(history: Any, tokens: list[int]) -> list[list[list[int]]] | None:
    aligned = None
    shape = None
    for choice, response in history_choices(history):
        capture = choice_routes(choice, response)
        if capture is None:
            continue
        captured_tokens, routes = capture
        current = validate_routes(routes, len(routes))
        if current is None:
            continue
        if shape is not None and shape != current:
            raise ValueError(
                "Captured routed_experts shapes disagree across generations"
            )
        shape = current
        if aligned is None:
            aligned = [[[-1] * shape[1] for _ in range(shape[0])] for _ in tokens]
        # Preserve first-captured routes: later extended prompts can reroute the
        # uncached tail. Equal suffixes alone do not prove equal conditioning.
        for index, (actual, captured, row) in enumerate(
            zip(tokens, captured_tokens, routes)
        ):
            if actual != captured:
                break
            for layer_index, layer in enumerate(row):
                for slot, expert in enumerate(layer):
                    if aligned[index][layer_index][slot] == -1:
                        aligned[index][layer_index][slot] = expert
    return aligned


def history_logprob_flags(
    history: Any, tokens: list[int], flags: list[TokenFlag]
) -> None:
    from . import TokenFlag

    modes = {
        "raw_logprobs": TokenFlag.RAW_LOGPROBS,
        "processed_logprobs": TokenFlag.PROCESSED_LOGPROBS,
    }
    mask = TokenFlag.RAW_LOGPROBS | TokenFlag.PROCESSED_LOGPROBS
    for choice, response in history_choices(history):
        extra = choice.model_extra or {}
        response_extra = getattr(response, "model_extra", None) or {}
        mode = extra.get("logprobs_mode", response_extra.get("logprobs_mode"))
        if mode is None:
            continue
        if mode not in modes:
            raise ValueError(f"Unsupported logprobs_mode: {mode!r}")
        prompt = extra.get("prompt_token_ids", response_extra.get("prompt_token_ids"))
        completion = extra.get("token_ids")
        if not isinstance(prompt, list) or not isinstance(completion, list):
            continue
        for index, (actual, captured) in enumerate(zip(tokens, prompt + completion)):
            if actual != captured:
                break
            if index < len(prompt) or not flags[index] & TokenFlag.SAMPLED:
                continue
            if flags[index] & mask and flags[index] & mask != modes[mode]:
                raise ValueError("Captured logprobs modes disagree for sampled tokens")
            flags[index] |= modes[mode]


def history_top_k(history: Any, tokens: list[int]) -> Any:
    from . import TokenizedTopK

    captured = []
    width = 0
    for choice, response in history_choices(history):
        extra = choice.model_extra or {}
        top = extra.get("compact_top_logprobs")
        if top is None:
            continue
        prompt = extra.get(
            "prompt_token_ids",
            (getattr(response, "model_extra", None) or {}).get("prompt_token_ids"),
        )
        completion = extra.get("token_ids")
        if (
            not isinstance(top, dict)
            or not isinstance(prompt, list)
            or not isinstance(completion, list)
        ):
            raise ValueError(
                "compact_top_logprobs requires exact prompt and completion IDs"
            )
        ids, values = top.get("token_ids"), top.get("logprobs")
        if (
            not isinstance(ids, list)
            or not isinstance(values, list)
            or len(ids) != len(values)
            or len(ids) != len(completion)
        ):
            raise ValueError(
                "compact_top_logprobs rows must match completion token IDs"
            )
        for row_ids, row_values in zip(ids, values, strict=True):
            if (
                not isinstance(row_ids, list)
                or not isinstance(row_values, list)
                or len(row_ids) != len(row_values)
            ):
                raise ValueError(
                    "compact_top_logprobs token IDs and logprobs differ in shape"
                )
            if any(type(token) is not int or token < 0 for token in row_ids) or len(
                set(row_ids)
            ) != len(row_ids):
                raise ValueError(
                    "compact_top_logprobs requires distinct nonnegative token IDs"
                )
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in row_values
            ):
                raise ValueError("compact_top_logprobs requires finite logprobs")
            width = max(width, len(row_ids))
        captured.append((prompt, completion, ids, values))
    if not captured or not width:
        return None
    ids = [[-1] * width for _ in tokens]
    values = [[math.nan] * width for _ in tokens]
    for prompt, completion, row_ids, row_values in captured:
        if tokens[: len(prompt)] != prompt:
            continue
        for j, (actual, token) in enumerate(zip(tokens[len(prompt) :], completion)):
            if actual != token:
                break
            index = len(prompt) + j
            if all(token_id == -1 for token_id in ids[index]):
                ids[index][: len(row_ids[j])] = row_ids[j]
                values[index][: len(row_values[j])] = row_values[j]
    return TokenizedTopK(tokens=ids, logprobs=values)
