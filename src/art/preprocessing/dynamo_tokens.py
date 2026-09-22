"""Read Dynamo's complete sampled sequence without changing text logprobs."""

import math
from typing import Any, cast

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice

COMPLETION_LOGPROBS_KEY = "art_completion_logprobs"


def _ids(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or any(type(x) is not int or x < 0 for x in value):
        raise ValueError(f"Dynamo {field} must contain nonnegative integer token IDs")
    return cast(list[int], list(value))


def _logprobs(values: Any, tokens: Any) -> list[float] | None:
    if values is None:
        return None
    if (
        not isinstance(values, list)
        or not isinstance(tokens, list)
        or len(values) != len(tokens)
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
    ):
        raise ValueError("Exact completion token IDs require matching finite logprobs")
    return [float(x) for x in values]


def choice_completion_logprobs(choice: Choice) -> list[float] | None:
    """Return ART's exact sampled logprobs, when the provider supplied them."""
    extra = choice.model_extra or {}
    return _logprobs(extra.get(COMPLETION_LOGPROBS_KEY), extra.get("token_ids"))


def attach_dynamo_token_metadata(response: ChatCompletion) -> None:
    """Attach single-choice engine metadata to ART's per-choice training fields.

    Native vLLM fields remain authoritative. Multi-choice Dynamo responses are
    left untouched because response-level arrays cannot identify a choice.
    """
    extra = response.model_extra or {}
    choices = response.choices
    if choices and all(
        (choice.model_extra or {}).get("token_ids") is not None
        and (choice.model_extra or {}).get(
            "prompt_token_ids", extra.get("prompt_token_ids")
        )
        is not None
        for choice in choices
    ):
        return
    nvext = extra.get("nvext")
    engine = nvext.get("engine_data") if isinstance(nvext, dict) else None
    if not isinstance(engine, dict) or not any(
        field in engine
        for field in ("prompt_token_ids", "completion_token_ids", "completion_logprobs")
    ):
        return
    if len(choices) > 1:
        return
    if not choices or engine.get("finished") is not True:
        raise ValueError("Dynamo training metadata requires a finished single choice")
    prompt = _ids(engine.get("prompt_token_ids"), "prompt_token_ids")
    tokens = _ids(engine.get("completion_token_ids"), "completion_token_ids")
    if not prompt:
        raise ValueError("Dynamo prompt_token_ids must not be empty")
    choice = choices[0]
    target = cast(dict[str, Any], choice.model_extra)
    for existing, expected in (
        (target.get("prompt_token_ids", extra.get("prompt_token_ids")), prompt),
        (target.get("token_ids"), tokens),
    ):
        if existing is not None and existing != expected:
            raise ValueError("Dynamo metadata conflicts with native token IDs")
    values = engine.get("completion_logprobs")
    # Validate before mutating the response. Missing logprobs remain missing;
    # the training path enforces allow_training_without_logprobs as usual.
    logprobs = _logprobs(values, tokens)
    target["prompt_token_ids"] = prompt
    target["token_ids"] = tokens
    target[COMPLETION_LOGPROBS_KEY] = logprobs
