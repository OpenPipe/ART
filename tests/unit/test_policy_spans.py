from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest

from art.model import _attach_response_art_metadata
from art.preprocessing.policy_spans import (
    POLICY_TOKEN_SPANS_KEY,
    PROMPT_POLICY_TOKEN_SPANS_KEY,
    attach_policy_token_metadata_to_choice,
    attach_static_policy_token_span_to_choice,
    validate_complete_policy_token_spans,
    validate_complete_prompt_policy_token_spans,
)


def _span(start: int, end: int, generation: str = "generation-2") -> dict[str, object]:
    return {
        "start_token": start,
        "end_token": end,
        "generation_id": generation,
        "policy_version": 2,
        "lora_slot": "run:active",
        "update_seq": 2,
    }


def _choice() -> Choice:
    return Choice(
        index=0,
        finish_reason="stop",
        message=ChatCompletionMessage(role="assistant", content="answer"),
    )


def _response(**choice_extra: object) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "completion",
            "object": "chat.completion",
            "created": 1,
            "model": "run:active",
            "prompt_token_ids": [1, 2, 3],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                    "token_ids": [4],
                    **choice_extra,
                }
            ],
        }
    )


def test_prompt_and_completion_spans_cross_response_and_validate() -> None:
    choice = _choice()
    prompt_spans = [_span(1, 4), _span(4, 7, "generation-3")]
    completion_spans = [_span(0, 2)]

    attach_policy_token_metadata_to_choice(
        choice=choice,
        response_payload={
            "choices": [
                {
                    PROMPT_POLICY_TOKEN_SPANS_KEY: prompt_spans,
                    POLICY_TOKEN_SPANS_KEY: completion_spans,
                }
            ]
        },
    )

    validate_complete_prompt_policy_token_spans(choice, prompt_tokens=7)
    validate_complete_policy_token_spans(choice, completion_tokens=2)
    with pytest.raises(RuntimeError, match="covered=7, prompt_tokens=8"):
        validate_complete_prompt_policy_token_spans(choice, prompt_tokens=8)
    with pytest.raises(RuntimeError, match="covered=2, completion_tokens=3"):
        validate_complete_policy_token_spans(choice, completion_tokens=3)


def test_static_policy_spans_cover_prompt_and_completion() -> None:
    choice = _choice()

    attach_static_policy_token_span_to_choice(
        choice=choice,
        model_name="model@12",
        prompt_tokens=5,
        completion_tokens=2,
    )

    assert choice.model_extra is not None
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
    assert choice.model_extra[POLICY_TOKEN_SPANS_KEY][0]["generation_id"] == (
        "model@12"
    )


def test_policy_span_metadata_fails_closed_on_invalid_generation() -> None:
    choice = _choice()

    with pytest.raises(ValueError, match="generation_id"):
        attach_policy_token_metadata_to_choice(
            choice=choice,
            response_payload={
                "choices": [
                    {POLICY_TOKEN_SPANS_KEY: [{**_span(0, 1), "generation_id": ""}]}
                ]
            },
        )


@pytest.mark.parametrize(
    ("choice_extra", "error"),
    [
        ({POLICY_TOKEN_SPANS_KEY: [_span(0, 1)]}, "prompt_policy_token_spans"),
        (
            {PROMPT_POLICY_TOKEN_SPANS_KEY: [_span(1, 3)]},
            "policy_token_spans",
        ),
        (
            {
                PROMPT_POLICY_TOKEN_SPANS_KEY: [_span(1, 2)],
                POLICY_TOKEN_SPANS_KEY: [_span(0, 1)],
            },
            "covered=2, prompt_tokens=3",
        ),
        (
            {
                PROMPT_POLICY_TOKEN_SPANS_KEY: [_span(1, 3)],
                POLICY_TOKEN_SPANS_KEY: [_span(0, 2)],
            },
            "covered=2, completion_tokens=1",
        ),
    ],
)
def test_required_response_policy_spans_fail_closed(
    choice_extra: dict[str, object], error: str
) -> None:
    with pytest.raises(RuntimeError, match=error):
        _attach_response_art_metadata(
            _response(**choice_extra),
            policy_span_mode="require",
            request_model="run:active",
        )
