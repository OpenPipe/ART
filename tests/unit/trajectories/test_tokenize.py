from __future__ import annotations

from datetime import datetime, timedelta
import math
from types import SimpleNamespace
from typing import Any

from anthropic.types import Message
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.responses import Response
import pytest

import art
from art.trajectories import (
    ChatCompletionsExchange,
    ChatCompletionsRequest,
    CompletionsExchange,
    CompletionsRequest,
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
            model=model,
            messages=[{"role": "user", "content": f"turn {offset}"}],
        ),
        response=response,
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )


def _completion_exchange(
    *,
    prompt: str | list[str] | list[int] | list[list[int]] = "question",
    echo: bool = False,
) -> CompletionsExchange:
    response = Completion.model_validate(
        {
            "id": "completion-1",
            "object": "text_completion",
            "created": 0,
            "model": "test/model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "text": "answer",
                    "prompt_token_ids": [1],
                    "token_ids": [2],
                    "logprobs": {
                        "tokens": ["token_id:2"],
                        "token_logprobs": [-0.2],
                        "top_logprobs": [{}],
                        "text_offset": [0],
                    },
                }
            ],
        }
    )
    request = CompletionsRequest(model="test/model", prompt="question", echo=echo)
    request["prompt"] = prompt
    start = datetime(2026, 1, 1)
    return CompletionsExchange(
        request=request,
        response=response,
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


def test_malformed_explicit_exact_token_metadata_fails_closed() -> None:
    chat = _chat_exchange([1], [2])
    chat_extra = chat.response.choices[0].model_extra
    assert chat_extra is not None
    chat_extra["prompt_token_ids"] = [1, "invalid"]

    completion = _completion_exchange()
    completion_extra = completion.response.choices[0].model_extra
    assert completion_extra is not None
    completion_extra["token_ids"] = [2, "invalid"]

    response = _response_exchange("response-invalid", 2)
    response_extra = response.response.model_extra
    assert response_extra is not None
    response_extra["raw_output_tokens"] = [{"token_id": "invalid"}]

    message_response = Message.model_validate(
        {
            "id": "message-invalid",
            "type": "message",
            "role": "assistant",
            "model": "test/model",
            "content": [{"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "token_ids": [2, "invalid"],
        }
    )
    start = datetime(2026, 1, 1)
    message = MessagesExchange(
        request=MessagesRequest(
            model="test/model",
            messages=[{"role": "user", "content": "question"}],
            max_tokens=16,
        ),
        response=message_response,
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )

    trajectories = [
        art.Trajectory(exchanges=TrajectoryExchanges(chat_completions=[chat])),
        art.Trajectory(exchanges=TrajectoryExchanges(completions=[completion])),
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[response])),
        art.Trajectory(exchanges=TrajectoryExchanges(messages=[message])),
    ]
    for trajectory in trajectories:
        with pytest.raises(ValueError, match="exact token"):
            art.tokenize_trajectory(trajectory, base_model="base/model")


@pytest.mark.parametrize(
    "exchange",
    [
        _completion_exchange(prompt=["batched"]),
        _completion_exchange(prompt=[[1, 2]]),
        _completion_exchange(echo=True),
    ],
)
def test_completions_reject_batch_prompts_and_echo(
    exchange: CompletionsExchange,
) -> None:
    with pytest.raises(ValueError, match="batched Completions|echo=True"):
        art.tokenize_trajectory(
            art.Trajectory(exchanges=TrajectoryExchanges(completions=[exchange]))
        )


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
            model="test/model",
            messages=[{"role": "user", "content": "question"}],
            chat_template="request-template",
            chat_template_kwargs={"request": True},
            thinking={"type": "enabled", "budget_tokens": 128},
        ),
        response=response,
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

    monkeypatch.setattr("wandb.apis.public.Api", Api)
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


def test_undecodable_visible_token_bytes_fall_back_to_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _chat_exchange([], [])
    logprobs = exchange.response.choices[0].logprobs
    assert logprobs is not None
    exchange.response.choices[0].logprobs = logprobs.model_copy(
        update={
            "content": [
                ChatCompletionTokenLogprob(
                    token="ordinary-token",
                    logprob=-0.7,
                    bytes=[0xF0],
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
            return [10, 11] if messages[-1]["role"] == "assistant" else [10]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    result = art.tokenize_trajectory(
        art.Trajectory(exchanges=TrajectoryExchanges(chat_completions=[exchange])),
        base_model="base/model",
    )

    assert result.token_ids == [10, 11]
    assert math.isnan(result.logprobs[1])


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
    request = ResponsesRequest(model="test/model", input=f"turn {offset}")
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    start = datetime(2026, 1, 1) + timedelta(seconds=offset)
    return ResponsesExchange(
        request=request,
        response=response,
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )


def _response_with_content_logprobs(*, exact_second: bool) -> ResponsesExchange:
    exchange = _response_exchange("response-content-logprobs", 0)
    data = exchange.response.model_dump(mode="python")
    data.pop("raw_output_tokens", None)

    def entry(token: str, token_id: int | None, logprob: float) -> dict[str, Any]:
        return {
            "token": token,
            "logprob": logprob,
            "bytes": list(("a" if token_id == 11 else "b").encode()),
            "top_logprobs": [],
            **({"token_id": token_id} if token_id is not None else {}),
        }

    data["output"][0]["content"] = [
        {
            "type": "output_text",
            "text": "a",
            "annotations": [],
            "logprobs": [entry("token_id:11", 11, -0.1)],
        },
        {
            "type": "output_text",
            "text": "b",
            "annotations": [],
            "logprobs": [
                entry(
                    "token_id:12" if exact_second else "b",
                    12 if exact_second else None,
                    -0.2,
                )
            ],
        },
    ]
    exchange.response = Response.model_validate(data)
    return exchange


def test_responses_aggregates_complete_exact_pairs_across_content_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del messages, kwargs
            return [10]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    result = art.tokenize_trajectory(
        art.Trajectory(
            exchanges=TrajectoryExchanges(
                responses=[_response_with_content_logprobs(exact_second=True)]
            )
        ),
        base_model="base/model",
    )

    assert result.token_ids == [10, 11, 12]
    assert result.logprobs[1:] == [-0.1, -0.2]


def test_responses_does_not_use_partial_exact_content_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del kwargs
            return [10, 11, 12] if messages[-1]["role"] == "assistant" else [10]

        def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                input_ids=[11 if text in {"a", "token_id:11"} else 12]
            )

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    result = art.tokenize_trajectory(
        art.Trajectory(
            exchanges=TrajectoryExchanges(
                responses=[_response_with_content_logprobs(exact_second=False)]
            )
        ),
        base_model="base/model",
    )

    assert result.token_ids == [10, 11, 12]
    assert result.logprobs[1:] == [-0.1, -0.2]


def test_responses_rejects_only_unrenderable_prompt_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: object
        ) -> list[int]:
            del messages, kwargs
            return [10]

    monkeypatch.setattr(
        "art.trajectories._tokenize._load_tokenizer", lambda _config: Tokenizer()
    )
    request_reasoning = _response_exchange("request-reasoning", 2)
    request_reasoning.request["input"] = [
        {"id": "reasoning-1", "summary": [], "type": "reasoning"}
    ]

    response_reasoning = _response_exchange("response-reasoning", 2)
    data = response_reasoning.response.model_dump(mode="python")
    data["output"] = [{"id": "reasoning-2", "summary": [], "type": "reasoning"}]
    response_reasoning.response = Response.model_validate(data)

    with pytest.raises(ValueError, match="Unsupported Responses input"):
        art.tokenize_trajectory(
            art.Trajectory(
                exchanges=TrajectoryExchanges(responses=[request_reasoning])
            ),
            base_model="base/model",
        )

    single = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[response_reasoning])
    )
    assert art.tokenize_trajectory(single, base_model="base/model").token_ids == [
        10,
        2,
    ]

    continuation = _response_exchange(
        "continuation",
        3,
        previous_response_id=response_reasoning.response.id,
        offset=1,
    )
    with pytest.raises(ValueError, match="Unsupported Responses output"):
        art.tokenize_trajectory(
            art.Trajectory(
                exchanges=TrajectoryExchanges(
                    responses=[response_reasoning, continuation]
                )
            ),
            base_model="base/model",
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

    second.request["previous_response_id"] = "missing"
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
            Tokenizer(),  # type: ignore[arg-type, ty:invalid-argument-type]
            [group],
            allow_training_without_logprobs=True,
            scale_rewards=False,
            shuffle_group_trajectories=False,
        )
    )

    assert [result.token_ids for result in results] == [[1, 2], [1, 3]]
    assert [result.assistant_mask for result in results] == [[0, 1], [0, 1]]
