from __future__ import annotations

import re
from typing import Any, cast

from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, Field, model_validator

POLICY_TOKEN_SPANS_KEY = "policy_token_spans"
PROMPT_POLICY_TOKEN_SPANS_KEY = "prompt_policy_token_spans"


class PolicyTokenSpan(BaseModel):
    """Half-open token interval scored by one executing policy state.

    The version identifies the adapter used by the target model execution that
    produced the returned token and logprob, not request admission or response
    delivery. Adjacent intervals may merge only when all policy identity fields
    match.
    """

    model_config = ConfigDict(extra="forbid")

    start_token: int = Field(ge=0)
    end_token: int = Field(gt=0)
    generation_id: str = Field(min_length=1)
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
    if not isinstance(raw_choice, dict):
        return
    extra = cast(dict[str, Any], choice.model_extra)
    for key in (POLICY_TOKEN_SPANS_KEY, PROMPT_POLICY_TOKEN_SPANS_KEY):
        if key in raw_choice:
            extra[key] = _normalize_policy_token_spans(raw_choice.get(key))


def choice_policy_token_spans(choice: Choice) -> list[PolicyTokenSpan]:
    extra = choice.model_extra or {}
    return [
        PolicyTokenSpan.model_validate(span)
        for span in extra.get(POLICY_TOKEN_SPANS_KEY, [])
    ]


def choice_prompt_policy_token_spans(choice: Choice) -> list[PolicyTokenSpan]:
    extra = choice.model_extra or {}
    return [
        PolicyTokenSpan.model_validate(span)
        for span in extra.get(PROMPT_POLICY_TOKEN_SPANS_KEY, [])
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


def validate_complete_prompt_policy_token_spans(
    choice: Choice, *, prompt_tokens: int
) -> None:
    spans = choice_prompt_policy_token_spans(choice)
    cursor = min(prompt_tokens, 1)
    for span in spans:
        if span.start_token != cursor:
            raise RuntimeError(
                "Prompt policy token spans must form a contiguous causal-logprob "
                f"partition; expected start_token={cursor}, got {span.start_token}."
            )
        cursor = span.end_token
    if cursor != prompt_tokens:
        raise RuntimeError(
            "Prompt policy token spans must cover every prompt token with a causal "
            f"logprob; covered={cursor}, prompt_tokens={prompt_tokens}."
        )


def attach_static_policy_token_span_to_choice(
    *,
    choice: Choice,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
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
    if prompt_tokens > 1:
        extra[PROMPT_POLICY_TOKEN_SPANS_KEY] = [
            PolicyTokenSpan(
                start_token=1,
                end_token=prompt_tokens,
                generation_id=model_name,
                policy_version=step,
                lora_slot=model_name,
                update_seq=step,
            ).model_dump(mode="python")
        ]
    extra[POLICY_TOKEN_SPANS_KEY] = [
        PolicyTokenSpan(
            start_token=0,
            end_token=completion_tokens,
            generation_id=model_name,
            policy_version=step,
            lora_slot=model_name,
            update_seq=step,
        ).model_dump(mode="python")
    ]
