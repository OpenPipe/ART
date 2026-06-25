from __future__ import annotations

from typing import Any, cast

from openai.types.chat.chat_completion import Choice


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
