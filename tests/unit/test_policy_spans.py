from types import SimpleNamespace

import pytest

from art.preprocessing.policy_spans import (
    PROMPT_POLICY_TOKEN_SPANS_KEY,
    attach_policy_token_metadata_to_choice,
    attach_static_policy_token_span_to_choice,
    validate_complete_prompt_policy_token_spans,
)


def test_prompt_policy_spans_cross_response_and_validate_causal_coverage() -> None:
    choice = SimpleNamespace(model_extra={})
    spans = [
        {
            "start_token": 1,
            "end_token": 4,
            "generation_id": "generation-2",
            "policy_version": 2,
            "lora_slot": "run:active",
            "update_seq": 2,
        },
        {
            "start_token": 4,
            "end_token": 7,
            "generation_id": "generation-3",
            "policy_version": 3,
            "lora_slot": "run:active",
            "update_seq": 3,
        },
    ]

    attach_policy_token_metadata_to_choice(
        choice=choice,
        response_payload={"choices": [{PROMPT_POLICY_TOKEN_SPANS_KEY: spans}]},
    )

    validate_complete_prompt_policy_token_spans(choice, prompt_tokens=7)
    with pytest.raises(RuntimeError, match="covered=7, prompt_tokens=8"):
        validate_complete_prompt_policy_token_spans(choice, prompt_tokens=8)


def test_static_policy_span_covers_prompt_and_completion() -> None:
    choice = SimpleNamespace(model_extra={})

    attach_static_policy_token_span_to_choice(
        choice=choice,
        model_name="model@12",
        prompt_tokens=5,
        completion_tokens=2,
    )

    assert choice.model_extra[PROMPT_POLICY_TOKEN_SPANS_KEY] == [
        {
            "start_token": 1,
            "end_token": 5,
            "generation_id": "model@12",
            "policy_version": 12,
            "lora_slot": "model@12",
            "update_seq": 12,
        }
    ]
