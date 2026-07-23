from datetime import datetime, timedelta
import importlib
from typing import Any

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
    assert [source.exchange for source in history.message_sources if source] == [
        first,
        first,
        second,
        second,
    ]
    assert history.messages is not first.request["messages"]
    assert trajectory.chat_completions_history(model="test/model") == history

    second.request["cache_salt"] = "new-cache"
    assert trajectory.chat_completions_history(model="test/model") == history
    second.request.pop("cache_salt")

    second.request["messages"] = [{"role": "user", "content": "branch"}]
    assert len(trajectory.chat_completions_histories(model="test/model")) == 2
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.chat_completions_history(model="test/model")


def test_chat_choices_branch_and_identical_continuation_uses_first_choice() -> None:
    first = _chat([{"role": "user", "content": "one"}], "same")
    response = first.response.model_dump(mode="python")
    second_choice = dict(response["choices"][0])
    second_choice["index"] = 1
    response["choices"].append(second_choice)
    first.response = ChatCompletion.model_validate(response)
    continuation = _chat(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "same"},
            {"role": "user", "content": "two"},
        ],
        "continued",
        offset=1,
    )
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, continuation])
    )

    histories = trajectory.chat_completions_histories()

    assert len(histories) == 2
    assert [len(history.messages) for history in histories] == [4, 2]
    assert histories[0].message_sources[1] is not None
    assert histories[0].message_sources[1].choice_index == 0
    assert histories[1].message_sources[1] is not None
    assert histories[1].message_sources[1].choice_index == 1


def test_history_mutation_must_keep_source_sidecar_consistent() -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[_chat([{"role": "user", "content": "one"}], "first")]
        )
    )
    history = trajectory.chat_completions_history()
    history.messages.append({"role": "user", "content": "next"})
    with pytest.raises(ValueError, match="differ in length"):
        history.tokenize()

    history.message_sources.append(None)
    history.messages[0] = {"role": "user", "content": "edited"}
    with pytest.raises(ValueError, match="no longer matches"):
        history.tokenize()


def test_history_accepts_user_authored_messages_with_none_source() -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            chat_completions=[_chat([{"role": "user", "content": "one"}], "first")]
        )
    )
    history = trajectory.chat_completions_history()
    history.messages.append({"role": "user", "content": "next"})
    history.message_sources.append(None)

    class Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return {"first": [20], "next": [30]}[text]

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: object,
            tokenize: bool,
            add_generation_prompt: bool,
            chat_template: str | None = None,
            **kwargs: object,
        ) -> list[int]:
            del tools, tokenize, add_generation_prompt, chat_template, kwargs
            return [10] if len(messages) == 1 else [10, 20, 30]

    tokenized = history.tokenize(tokenizer=Tokenizer())

    assert tokenized.token_ids == [10, 20, 30]
    assert tokenized.flags[1] == art.TokenFlag.SAMPLED


def test_protocol_histories_convert_to_chat_and_history_rejects_ambiguity() -> None:
    message_trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(messages=[_message()])
    )
    messages_history = message_trajectory.anthropic_messages_history()
    assert messages_history.system == "Be concise"
    assert not hasattr(messages_history, "model_dump")
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


def test_anthropic_chat_conversion_preserves_sources_for_expanded_messages() -> None:
    exchange = _message()
    exchange.request["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "result",
                },
                {"type": "text", "text": "continue"},
            ],
        }
    ]
    history = art.Trajectory(
        exchanges=TrajectoryExchanges(messages=[exchange])
    ).anthropic_messages_history()

    converted = history.as_chat_completions_history()

    assert [message["role"] for message in converted.messages] == [
        "system",
        "tool",
        "user",
        "assistant",
    ]
    for source in converted.message_sources[1:3]:
        assert source is not None
        assert source.exchange is exchange
        assert source.request_index == 0

    converted.messages[2] = {"role": "user", "content": "changed"}
    with pytest.raises(ValueError, match="no longer matches"):
        converted.tokenize()


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
    chat_history = history.as_chat_completions_history()
    assert [message["role"] for message in chat_history.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert dict(chat_history.messages[1]).get("reasoning") == "think"
    assert all(source is not None for source in chat_history.message_sources)
    assert chat_history.message_sources[1] is not None
    assert chat_history.message_sources[1].output_index == 0

    trajectory.exchanges.responses[1].request["previous_response_id"] = "missing"
    external = trajectory.responses_histories()
    assert len(external) == 2
    assert external[1].previous_response_id == "missing"


def test_completions_history_preserves_exact_tokens_and_sampled_spans() -> None:
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(
            completions=[
                _completion([1], [2]),
                _completion([1, 2, 3], [4], offset=1),
            ]
        )
    )

    history = trajectory.completions_token_history()
    assert history.prompt == [1, 2, 3, 4]
    assert history.sampled_spans == [(1, 2), (3, 4)]
    with pytest.raises(ValueError, match="no chat-message structure"):
        history.as_chat_completions_history()


def test_completions_history_uses_request_token_ids() -> None:
    exchange = _completion([1], [2])
    response = exchange.response.model_dump(mode="python")
    response["choices"][0].pop("prompt_token_ids")
    exchange.response = Completion.model_validate(response)
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(completions=[exchange]))

    assert trajectory.completions_token_history().prompt == [1, 2]


def test_batched_completions_create_every_prompt_choice_history() -> None:
    exchange = _completion([1], [10])
    exchange.request["prompt"] = ["first", "second"]
    response = exchange.response.model_dump(mode="python")
    response["choices"] = [
        {
            "index": index,
            "finish_reason": "stop",
            "text": f"answer-{index}",
            "prompt_token_ids": [prompt_id],
            "token_ids": [100 + index],
        }
        for index, prompt_id in enumerate((1, 1, 2, 2))
    ]
    exchange.response = Completion.model_validate(response)
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(completions=[exchange]))

    histories = trajectory.completions_token_histories()

    assert [history.prompt for history in histories] == [
        [1, 100],
        [1, 101],
        [2, 102],
        [2, 103],
    ]
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.history()


def test_completions_reject_ambiguous_batches_and_suffix() -> None:
    ambiguous = _completion([1], [10])
    ambiguous.request["prompt"] = ["first", "second"]
    with pytest.raises(ValueError, match="associate Completions choices"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(completions=[ambiguous])
        ).histories()

    insertion = _completion([1], [10])
    insertion.request["suffix"] = "tail"
    with pytest.raises(ValueError, match="suffix is not supported"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(completions=[insertion])
        ).histories()


def test_reasoning_stripping_produces_truthful_history_per_generation() -> None:
    def exchange(
        offset: int,
        request_messages: list[dict[str, object]],
        answer: str,
    ) -> MessagesExchange:
        start, end = _times(offset)
        return MessagesExchange(
            request={
                "model": "test/model",
                "messages": request_messages,
                "max_tokens": 16,
                "thinking": {"type": "enabled", "budget_tokens": 8},
            },
            response=Message.model_validate(
                {
                    "id": f"message-{offset}",
                    "type": "message",
                    "role": "assistant",
                    "model": "test/model",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": f"thought-{offset}",
                            "signature": "sig",
                        },
                        {"type": "text", "text": answer},
                    ],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
            start_time=start,
            end_time=end,
        )

    first = exchange(0, [{"role": "user", "content": "one"}], "first")
    second = exchange(
        1,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
        ],
        "second",
    )
    third = exchange(
        2,
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "three"},
        ],
        "third",
    )
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(messages=[first, second, third])
    )

    histories = trajectory.anthropic_messages_histories()

    assert len(histories) == 3
    assert [len(history.messages) for history in histories] == [2, 4, 6]
    assert histories[1].message_sources[1] is not None
    assert histories[1].message_sources[1].exchange is first
    assert histories[1].message_sources[1].request_index is None
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.tokenize()


def test_chat_template_stripped_reasoning_splits_exact_histories() -> None:
    first = _chat([{"role": "user", "content": "one"}], "first")
    first_data = first.response.model_dump(mode="python")
    first_data["prompt_token_ids"] = [1]
    first_data["choices"][0]["message"]["reasoning"] = "thought-one"
    first_data["choices"][0]["token_ids"] = [2, 3]
    first.response = ChatCompletion.model_validate(first_data)

    second = _chat(
        [
            {"role": "user", "content": "one"},
            {
                "role": "assistant",
                "content": "first",
                "reasoning": "thought-one",
            },
            {"role": "user", "content": "two"},
        ],
        "second",
        offset=1,
    )
    second_data = second.response.model_dump(mode="python")
    second_data["prompt_token_ids"] = [1, 3, 4]
    second_data["choices"][0]["message"]["reasoning"] = "thought-two"
    second_data["choices"][0]["token_ids"] = [5, 6]
    second.response = ChatCompletion.model_validate(second_data)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    )

    histories = trajectory.chat_completions_histories()

    assert len(histories) == 2
    assert [len(history.messages) for history in histories] == [2, 4]
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.tokenize()


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

    assert isinstance(trajectory.history(), art.LegacyHistory)
    assert trajectory.messages() == [{"role": "user", "content": "hello"}]
    assert isinstance(trajectory.history(model="test/model"), art.LegacyHistory)
    with pytest.raises(ValueError, match="requires model="):
        trajectory.tokenize()


def test_legacy_messages_preserve_primary_history_with_additional_histories() -> None:
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "user", "content": "primary"}],
        additional_histories=[
            art.LegacyHistory(
                messages_and_choices=[{"role": "user", "content": "alternate"}]
            )
        ],
    )

    assert trajectory.messages() == [{"role": "user", "content": "primary"}]
    assert len(trajectory.histories()) == 2
    assert len(trajectory.chat_completions_histories()) == 2
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.history()


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


@pytest.mark.asyncio
async def test_ruler_swallow_exceptions_covers_history_projection() -> None:
    from art.rewards.ruler import ruler_score_group

    group = art.TrajectoryGroup(
        [
            art.Trajectory(
                exchanges=TrajectoryExchanges(completions=[_completion([1], [2])])
            )
        ]
    )

    assert await ruler_score_group(group, swallow_exceptions=True) is None
    with pytest.raises(ValueError, match="no chat-message structure"):
        await ruler_score_group(group)
