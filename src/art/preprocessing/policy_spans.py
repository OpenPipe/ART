from __future__ import annotations

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
