from datetime import datetime, timedelta
import importlib

from anthropic.types import Message
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.responses import Response
import pytest

import art
from art.trajectories import (
    ChatCompletionsExchange,
    CompletionsExchange,
    MessagesExchange,
    ResponsesExchange,
    TrajectoryExchanges,
)


def _times(offset: int = 0) -> tuple[datetime, datetime]:
    start = datetime(2026, 1, 1) + timedelta(seconds=offset)
    return start, start + timedelta(milliseconds=1)


def _chat(
    messages: list[dict[str, object]],
    answer: str,
    *,
    model: str = "test/model",
    offset: int = 0,
) -> ChatCompletionsExchange:
    start, end = _times(offset)
    return ChatCompletionsExchange(
        request={"model": model, "messages": messages},
        response=ChatCompletion.model_validate(
            {
                "id": f"chat-{offset}",
                "object": "chat.completion",
                "created": offset,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": answer},
                    }
                ],
            }
        ),
        start_time=start,
        end_time=end,
    )


def _completion(
    prompt: list[int], output: list[int], *, offset: int = 0
) -> CompletionsExchange:
    start, end = _times(offset)
    return CompletionsExchange(
        request={"model": "test/model", "prompt": prompt},
        response=Completion.model_validate(
            {
                "id": f"completion-{offset}",
                "object": "text_completion",
                "created": offset,
                "model": "test/model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "text": "answer",
                        "prompt_token_ids": prompt,
                        "token_ids": output,
                    }
                ],
            }
        ),
        start_time=start,
        end_time=end,
    )


def _message() -> MessagesExchange:
    start, end = _times()
    return MessagesExchange(
        request={
            "model": "test/model",
            "system": "Be concise",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
        response=Message.model_validate(
            {
                "id": "message-1",
                "type": "message",
                "role": "assistant",
                "model": "test/model",
                "content": [{"type": "text", "text": "Hi"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ),
        start_time=start,
        end_time=end,
    )


def _response(
    response_id: str,
    text: str,
    *,
    previous_response_id: str | None = None,
    reasoning: str | None = None,
    offset: int = 0,
) -> ResponsesExchange:
    start, end = _times(offset)
    request: dict[str, object] = {
        "model": "test/model",
        "input": f"turn {offset}",
    }
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    return ResponsesExchange(
        request=request,
        response=Response.model_validate(
            {
                "id": response_id,
                "created_at": float(offset),
                "model": "test/model",
                "object": "response",
                "output": [
                    *(
                        [
                            {
                                "id": f"reasoning-{response_id}",
                                "type": "reasoning",
                                "summary": [
                                    {"type": "summary_text", "text": reasoning}
                                ],
                            }
                        ]
                        if reasoning is not None
                        else []
                    ),
                    {
                        "id": f"message-{response_id}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                                "logprobs": [],
                            }
                        ],
                    },
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        ),
        start_time=start,
        end_time=end,
    )


def test_chat_history_resolves_one_model_and_append_only_sequence() -> None:
    first = _chat([{"role": "user", "content": "one"}], "first")
    second = _chat(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
        ],
        "second",
        offset=1,
    )
    other = _chat(
        [{"role": "user", "content": "other"}],
        "other",
        model="other/model",
        offset=2,
    )
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second, other])
    )

    with pytest.raises(ValueError, match="exactly one model"):
        trajectory.history()
    history = trajectory.history(model="test/model")
    assert isinstance(history, art.ChatCompletionsHistory)
    assert [message["role"] for message in history.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert trajectory.chat_completions_history(model="test/model") == history

    second.request["cache_salt"] = "new-cache"
    with pytest.raises(ValueError, match="different cache_salt"):
        trajectory.chat_completions_history(model="test/model")
    second.request.pop("cache_salt")

    second.request["messages"] = [{"role": "user", "content": "branch"}]
    with pytest.raises(ValueError, match="append-only"):
        trajectory.chat_completions_history(model="test/model")


def test_protocol_histories_convert_to_chat_and_history_rejects_ambiguity() -> None:
    message_trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(messages=[_message()])
    )
    messages_history = message_trajectory.anthropic_messages_history()
    assert messages_history.system == "Be concise"
    assert (
        art.AnthropicMessagesHistory.model_validate_json(
            messages_history.model_dump_json()
        )
        == messages_history
    )
    assert [message["role"] for message in messages_history.messages] == [
        "user",
        "assistant",
    ]
    assert [message["role"] for message in message_trajectory.messages()] == [
        "system",
        "user",
        "assistant",
    ]

    mixed = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[_chat([{"role": "user", "content": "hi"}], "hi")],
            messages=[_message()],
        )
    )
    with pytest.raises(ValueError, match="multiple protocol histories"):
        mixed.history()
    assert isinstance(mixed.anthropic_messages_history(), art.AnthropicMessagesHistory)


def test_responses_history_expands_previous_response_chain() -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            responses=[
                _response("response-1", "first", reasoning="think"),
                _response(
                    "response-2",
                    "second",
                    previous_response_id="response-1",
                    offset=1,
                ),
            ]
        )
    )

    history = trajectory.responses_history()
    assert len(history.input) == 5
    assert (
        art.ResponsesHistory.model_validate_json(history.model_dump_json()) == history
    )
    chat_history = history.as_chat_completions_history()
    assert [message["role"] for message in chat_history.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert dict(chat_history.messages[1]).get("reasoning") == "think"

    trajectory.exchanges.responses[1].request["previous_response_id"] = "missing"
    with pytest.raises(ValueError, match="outside this history"):
        trajectory.responses_history()


def test_completions_history_preserves_exact_tokens_and_sampled_spans() -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            completions=[
                _completion([1], [2]),
                _completion([1, 2, 3], [4], offset=1),
            ]
        )
    )

    history = trajectory.completions_history()
    assert history.token_ids == [1, 2, 3, 4]
    assert history.sampled_spans == [(1, 2), (3, 4)]
    with pytest.raises(ValueError, match="no chat-message structure"):
        history.as_chat_completions_history()


def test_completions_history_uses_request_token_ids_and_rejects_echo() -> None:
    exchange = _completion([1], [2])
    response = exchange.response.model_dump(mode="python")
    response["choices"][0].pop("prompt_token_ids")
    exchange.response = Completion.model_validate(response)
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(completions=[exchange]))

    assert trajectory.completions_history().token_ids == [1, 2]

    exchange.request["echo"] = True
    with pytest.raises(ValueError, match="echo=True"):
        trajectory.completions_history()


def test_history_rejects_mutated_mixed_representation() -> None:
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "user", "content": "hi"}]
    )
    trajectory.exchanges.chat_completions.append(
        _chat([{"role": "user", "content": "hi"}], "hello")
    )

    with pytest.raises(ValueError, match="both exchanges and legacy histories"):
        trajectory.history()


def test_legacy_messages_delegate_through_history() -> None:
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "user", "content": "hello"}]
    )

    assert isinstance(trajectory.history(), art.History)
    assert trajectory.messages() == [{"role": "user", "content": "hello"}]
    with pytest.raises(ValueError, match="do not identify a model"):
        trajectory.history(model="test/model")


@pytest.mark.asyncio
async def test_ruler_accepts_exchange_trajectories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.rewards.ruler import TrajectoryScore, ruler_score_group

    ruler_module = importlib.import_module("art.rewards.ruler")
    captured: list[list[dict[str, object]]] = []

    async def score(
        message_lists: list[list[dict[str, object]]], **_: object
    ) -> list[TrajectoryScore]:
        captured.extend(message_lists)
        return [TrajectoryScore(trajectory_id="1", explanation="good", score=0.8)]

    monkeypatch.setattr(ruler_module, "ruler", score)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[_chat([{"role": "user", "content": "hi"}], "hello")]
        )
    )

    result = await ruler_score_group(art.TrajectoryGroup([trajectory]))

    assert result is not None
    assert result.trajectories[0].exchanges == trajectory.exchanges
    assert [message["role"] for message in captured[0]] == ["user", "assistant"]
