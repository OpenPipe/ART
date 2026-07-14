from __future__ import annotations

from datetime import datetime, timedelta
import math
from types import SimpleNamespace
from typing import Any

from anthropic.types import Message
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.responses import Response
import pytest

import art
from art.trajectories import (
    ChatCompletionsExchange,
    ChatCompletionsRequest,
    MessagesExchange,
    MessagesRequest,
    ResponsesExchange,
    ResponsesRequest,
    TrajectoryExchanges,
)


def _chat_exchange(
    prompt: list[int],
    output: list[int],
    *,
    model: str = "test/model",
    offset: int = 0,
) -> ChatCompletionsExchange:
    response = ChatCompletion.model_validate(
        {
            "id": f"chat-{offset}",
            "object": "chat.completion",
            "created": offset,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "answer"},
                    "prompt_token_ids": prompt,
                    "token_ids": output,
                    "logprobs": {
                        "content": [
                            {
                                "token": f"token_id:{token}",
                                "logprob": -token / 10,
                                "bytes": [],
                                "top_logprobs": [],
                            }
                            for token in output
                        ]
                    },
                }
            ],
        }
    )
    start = datetime(2026, 1, 1) + timedelta(seconds=offset)
    return ChatCompletionsExchange(
        request=ChatCompletionsRequest(
            {
                "model": model,
                "messages": [{"role": "user", "content": f"turn {offset}"}],
            }
        ),
        response=response,
        model=model,
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )


def test_exact_tokens_form_one_append_only_history_without_tokenizer() -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[
                _chat_exchange([1], [2], offset=0),
                _chat_exchange([1, 2, 3], [4], offset=1),
            ]
        )
    )

    tokenized = art.tokenize_trajectory(trajectory)

    assert tokenized.token_ids == [1, 2, 3, 4]
    assert tokenized.assistant_mask == [False, True, False, True]
    assert math.isnan(tokenized.logprobs[0])
    assert tokenized.logprobs[1] == -0.2
    assert math.isnan(tokenized.logprobs[2])
    assert tokenized.logprobs[3] == -0.4


def test_branching_and_multiple_models_require_explicit_resolution() -> None:
    branching = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[
                _chat_exchange([1], [2], offset=0),
                _chat_exchange([9], [3], offset=1),
            ]
        )
    )
    with pytest.raises(ValueError, match="append-only"):
        art.tokenize_trajectory(branching)

    mixed = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[
                _chat_exchange([1], [2], model="one", offset=0),
                _chat_exchange([3], [4], model="two", offset=1),
            ]
        )
    )
    with pytest.raises(ValueError, match="exactly one model"):
        art.tokenize_trajectory(mixed)
    assert art.tokenize_trajectory(mixed, model="two").token_ids == [3, 4]


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: object
    ) -> list[int]:
        self.calls.append(kwargs)
        return [10, 11] if messages[-1]["role"] == "assistant" else [10]


def test_fallback_uses_template_overrides_and_nan_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test/model",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    start = datetime(2026, 1, 1)
    exchange = MessagesExchange(
        request=MessagesRequest(
            {
                "model": "test/model",
                "messages": [{"role": "user", "content": "question"}],
                "chat_template": "request-template",
                "chat_template_kwargs": {"request": True},
                "thinking": {"type": "enabled", "budget_tokens": 128},
            }
        ),
        response=response,
        model="test/model",
        start_time=start,
        end_time=start + timedelta(seconds=1),
    )
    tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: tokenizer
    )

    result = art.tokenize_trajectory(
        art.Trajectory(exchanges=TrajectoryExchanges(messages=[exchange])),
        base_model="base/model",
        chat_template="explicit-template",
        chat_template_kwargs={"explicit": True},
    )

    assert result.token_ids == [10, 11]
    assert result.assistant_mask == [False, True]
    assert math.isnan(result.logprobs[1])
    assert tokenizer.calls == [
        {
            "tools": None,
            "tokenize": True,
            "add_generation_prompt": True,
            "chat_template": "explicit-template",
            "request": True,
            "explicit": True,
            "enable_thinking": True,
            "thinking_budget": 128,
        },
        {
            "tools": None,
            "tokenize": True,
            "add_generation_prompt": False,
            "chat_template": "explicit-template",
            "request": True,
            "explicit": True,
            "enable_thinking": True,
            "thinking_budget": 128,
        },
    ]


def test_checkpoint_fallback_uses_latest_artifact_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_names: list[str] = []

    class Api:
        def artifact(self, name: str) -> SimpleNamespace:
            artifact_names.append(name)
            return SimpleNamespace(
                metadata={
                    "wandb.base_model": "base/model",
                    "renderer": {
                        "tokenizer_revision": "revision",
                        "chat_template": "template",
                        "chat_template_kwargs": {"thinking": True},
                    },
                }
            )

    monkeypatch.setattr("wandb.Api", Api)
    from art.trajectories._tokenize import _tokenizer_config

    config = _tokenizer_config("wandb-artifact:///entity/project/run", None)

    assert artifact_names == ["entity/project/run:latest"]
    assert config.base_model == "base/model"
    assert config.revision == "revision"
    assert config.chat_template == "template"
    assert config.chat_template_kwargs == {"thinking": True}


def test_anthropic_fallback_preserves_thinking_and_tool_history() -> None:
    from art.trajectories._tokenize import _anthropic_messages

    messages = _anthropic_messages(
        {
            "system": [{"type": "text", "text": "system"}],
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "reason"},
                        {"type": "text", "text": "calling"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "lookup",
                            "input": {"key": "value"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": [{"type": "text", "text": "result"}],
                        },
                        {"type": "text", "text": "continue"},
                    ],
                },
            ],
        }
    )

    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "calling",
            "reasoning": "reason",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key": "value"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        {"role": "user", "content": "continue"},
    ]


def test_choice_logprobs_survive_tokenizer_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _chat_exchange([], [])
    logprobs = exchange.response.choices[0].logprobs
    assert logprobs is not None
    exchange.response.choices[0].logprobs = logprobs.model_copy(
        update={
            "content": [
                ChatCompletionTokenLogprob(
                    token="answer",
                    logprob=-0.7,
                    bytes=list(b"answer"),
                    top_logprobs=[],
                )
            ]
        }
    )

    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [10, 11, 12] if messages[-1]["role"] == "assistant" else [10]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del text, kwargs
            return SimpleNamespace(input_ids=[11])

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    result = art.tokenize_trajectory(
        art.Trajectory(exchanges=TrajectoryExchanges(chat_completions=[exchange])),
        base_model="base/model",
    )
    assert result.token_ids == [10, 11, 12]
    assert result.logprobs[1] == -0.7
    assert math.isnan(result.logprobs[2])


def test_json_round_trip_preserves_exchange_types() -> None:
    original = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[_chat_exchange([1], [2])])
    )
    restored = art.Trajectory.model_validate_json(original.model_dump_json())
    assert restored.model_dump(mode="json") == original.model_dump(mode="json")
    assert isinstance(restored.exchanges.chat_completions[0].response, ChatCompletion)


def _response_exchange(
    response_id: str,
    output_id: int,
    *,
    previous_response_id: str | None = None,
    offset: int = 0,
) -> ResponsesExchange:
    response = Response.model_validate(
        {
            "id": response_id,
            "created_at": float(offset),
            "model": "test/model",
            "object": "response",
            "output": [
                {
                    "id": f"message-{response_id}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "answer",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "raw_output_tokens": [{"token_id": output_id, "logprob": -0.1}],
        }
    )
    request = {"model": "test/model", "input": f"turn {offset}"}
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    start = datetime(2026, 1, 1) + timedelta(seconds=offset)
    return ResponsesExchange(
        request=ResponsesRequest(request),
        response=response,
        model="test/model",
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )


def test_responses_previous_response_id_resolves_local_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [10] if len(messages) == 1 else [10, 20, 11]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    first = _response_exchange("resp-1", 20)
    second = _response_exchange("resp-2", 30, previous_response_id="resp-1", offset=1)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[first, second])
    )

    assert art.tokenize_trajectory(trajectory, base_model="base/model").token_ids == [
        10,
        20,
        11,
        30,
    ]

    second.request.root["previous_response_id"] = "missing"
    with pytest.raises(ValueError, match="outside this trajectory"):
        art.tokenize_trajectory(trajectory, base_model="base/model")


def test_exchange_trajectories_feed_existing_training_tokenizer() -> None:
    from art.preprocessing.tokenize import tokenize_trajectory_groups

    class Tokenizer:
        name_or_path = "test/model"

        def decode(self, token_id: int) -> str:
            return str(token_id)

    group = art.TrajectoryGroup(
        [
            art.Trajectory(
                exchanges=TrajectoryExchanges(
                    chat_completions=[_chat_exchange([1], [2])]
                ),
                reward=1,
            ),
            art.Trajectory(
                exchanges=TrajectoryExchanges(
                    chat_completions=[_chat_exchange([1], [3])]
                ),
                reward=0,
            ),
        ]
    )

    results = list(
        tokenize_trajectory_groups(
            # This path only calls decode; the minimal test double is intentional.
            Tokenizer(),  # type: ignore[arg-type]
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=False,
            shuffle_group_trajectories=False,
        )
    )

    assert [result.token_ids for result in results] == [[1, 2], [1, 3]]
    assert [result.assistant_mask for result in results] == [[0, 1], [0, 1]]
