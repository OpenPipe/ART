from __future__ import annotations

from typing import Any, cast

from openai.types.chat.chat_completion import Choice

COMPLETION_TOKENS_KEY = "art_completion_tokens"


def _normalize_token_ids(raw: Any, *, field_name: str) -> list[int]:
    if raw is None:
        raise RuntimeError(f"Missing {field_name}")
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected {field_name} list, got {type(raw)}")
    return [int(token_id) for token_id in raw]


def attach_vllm_token_metadata_to_choice(
    *,
    choice: Choice,
    response_payload: dict[str, Any],
    choice_index: int = 0,
) -> None:
    prompt_token_ids = response_payload.get("prompt_token_ids")
    raw_choices = response_payload.get("choices")
    if not isinstance(raw_choices, list) or choice_index >= len(raw_choices):
        return
    raw_choice = raw_choices[choice_index]
    if not isinstance(raw_choice, dict):
        return
    completion_token_ids = raw_choice.get("token_ids")
    if prompt_token_ids is None or completion_token_ids is None:
        return
    extra = cast(dict[str, Any], choice.model_extra)
    extra["prompt_token_ids"] = _normalize_token_ids(
        prompt_token_ids,
        field_name="prompt_token_ids",
    )
    extra["token_ids"] = _normalize_token_ids(
        completion_token_ids,
        field_name="token_ids",
    )


def choice_vllm_token_metadata(choice: Choice) -> tuple[list[int], list[int]] | None:
    extra = choice.model_extra or {}
    if "prompt_token_ids" not in extra or "token_ids" not in extra:
        return None
    return (
        _normalize_token_ids(
            extra.get("prompt_token_ids"),
            field_name="prompt_token_ids",
        ),
        _normalize_token_ids(
            extra.get("token_ids"),
            field_name="token_ids",
        ),
    )


def _choice_generated_token_count(choice: Choice) -> int | None:
    token_metadata = choice_vllm_token_metadata(choice)
    if token_metadata is not None:
        return len(token_metadata[1])
    logprobs = choice.logprobs
    if logprobs is None or (logprobs.content is None and logprobs.refusal is None):
        return None
    return len(logprobs.content or []) + len(logprobs.refusal or [])


def attach_completion_token_metadata(response: Any) -> None:
    choices = getattr(response, "choices", None)
    usage = getattr(response, "usage", None)
    total = getattr(usage, "completion_tokens", None)
    if not choices or total is None:
        return
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise RuntimeError(f"Invalid response usage.completion_tokens: {total!r}")

    counts = [_choice_generated_token_count(choice) for choice in choices]
    if len(choices) == 1:
        if counts[0] is not None and counts[0] != total:
            raise RuntimeError(
                "Choice completion token count does not match response usage: "
                f"count={counts[0]}, usage.completion_tokens={total}"
            )
        counts = [total]
    elif any(count is None for count in counts):
        return
    if sum(cast(int, count) for count in counts) != total:
        raise RuntimeError(
            "Per-choice completion token counts do not match response usage: "
            f"counts={counts}, usage.completion_tokens={total}"
        )
    for choice, count in zip(choices, counts, strict=True):
        cast(dict[str, Any], choice.model_extra)[COMPLETION_TOKENS_KEY] = count


def choice_completion_tokens(choice: Choice) -> int | None:
    value = (choice.model_extra or {}).get(COMPLETION_TOKENS_KEY)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
