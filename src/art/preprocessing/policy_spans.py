from __future__ import annotations

import re
from typing import Any, cast

from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, Field, model_validator

POLICY_TOKEN_SPANS_KEY = "policy_token_spans"


class PolicyTokenSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_token: int = Field(ge=0)
    end_token: int = Field(gt=0)
    policy_version: int = Field(ge=0)
    lora_slot: str
    update_seq: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_order(self) -> "PolicyTokenSpan":
        if self.end_token <= self.start_token:
            raise RuntimeError(
                "policy token span end_token must be greater than start_token"
            )
        return self


def _normalize_policy_token_spans(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected {POLICY_TOKEN_SPANS_KEY} list, got {type(raw)}")
    return [
        PolicyTokenSpan.model_validate(span).model_dump(mode="python") for span in raw
    ]


def attach_policy_token_metadata_to_choice(
    *,
    choice: Choice,
    response_payload: dict[str, Any],
    choice_index: int = 0,
) -> None:
    raw_choices = response_payload.get("choices")
    if not isinstance(raw_choices, list) or choice_index >= len(raw_choices):
        return
    raw_choice = raw_choices[choice_index]
    if not isinstance(raw_choice, dict) or POLICY_TOKEN_SPANS_KEY not in raw_choice:
        return
    extra = cast(dict[str, Any], choice.model_extra)
    extra[POLICY_TOKEN_SPANS_KEY] = _normalize_policy_token_spans(
        raw_choice.get(POLICY_TOKEN_SPANS_KEY)
    )


def choice_policy_token_spans(choice: Choice) -> list[PolicyTokenSpan]:
    extra = choice.model_extra or {}
    return [
        PolicyTokenSpan.model_validate(span)
        for span in extra.get(POLICY_TOKEN_SPANS_KEY, [])
    ]


def validate_complete_policy_token_spans(
    choice: Choice, *, completion_tokens: int
) -> None:
    spans = choice_policy_token_spans(choice)
    cursor = 0
    for span in spans:
        if span.start_token != cursor:
            raise RuntimeError(
                "Policy token spans must form a contiguous completion partition; "
                f"expected start_token={cursor}, got {span.start_token}."
            )
        cursor = span.end_token
    if cursor != completion_tokens:
        raise RuntimeError(
            "Policy token spans must cover every completion token; "
            f"covered={cursor}, completion_tokens={completion_tokens}."
        )


def attach_static_policy_token_span_to_choice(
    *, choice: Choice, model_name: str, completion_tokens: int
) -> None:
    if completion_tokens <= 0:
        return
    match = re.search(r"@(\d+)$", model_name)
    if match is None:
        raise RuntimeError(
            "Immutable step-LoRA policy tracking requires a model name ending in @<step>."
        )
    step = int(match.group(1))
    extra = cast(dict[str, Any], choice.model_extra)
    extra[POLICY_TOKEN_SPANS_KEY] = [
        PolicyTokenSpan(
            start_token=0,
            end_token=completion_tokens,
            policy_version=step,
            lora_slot=model_name,
            update_seq=step,
        ).model_dump(mode="python")
    ]
