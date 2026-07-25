from datetime import datetime, timedelta
import importlib
from statistics import median
from time import perf_counter
from typing import Any, cast

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
    MessagesRequest,
    ResponsesExchange,
    TrajectoryExchanges,
)
from art.types import Message as ChatMessage


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


def _growing_chat_trajectory(turn_count: int) -> art.Trajectory:
    exchanges: list[ChatCompletionsExchange] = []
    messages: list[dict[str, object]] = []
    for index in range(turn_count):
        messages.append({"role": "user", "content": f"question {index}"})
        exchanges.append(_chat(list(messages), f"answer {index}", offset=index))
        messages.append({"role": "assistant", "content": f"answer {index}"})
    return art.Trajectory(exchanges=TrajectoryExchanges(chat_completions=exchanges))


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


def test_chat_projection_keys_each_captured_message_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_module = importlib.import_module("art.trajectories._history")
    original = history_module._chat_message_key
    calls = 0

    def count(message: ChatMessage, *, visible_only: bool = False) -> str:
        nonlocal calls
        calls += 1
        return original(message, visible_only=visible_only)

    monkeypatch.setattr(history_module, "_chat_message_key", count)
    trajectory = _growing_chat_trajectory(16)

    trajectory.chat_completions_history()

    expected = sum(
        len(exchange.request["messages"]) + len(exchange.response.choices)
        for exchange in trajectory.exchanges.chat_completions
    )
    assert calls == expected


def test_chat_projection_scales_with_captured_messages() -> None:
    measurements: list[tuple[int, float, int]] = []
    for turn_count in (32, 64, 128):
        trajectory = _growing_chat_trajectory(turn_count)
        captured_bytes = sum(
            len(exchange.model_dump_json().encode())
            for exchange in trajectory.exchanges.chat_completions
        )
        trajectory.chat_completions_history()
        samples: list[float] = []
        for _ in range(5):
            started = perf_counter()
            trajectory.chat_completions_history()
            samples.append(perf_counter() - started)
        elapsed = median(samples)
        measurements.append((turn_count, elapsed, captured_bytes))

    # A growing transcript contains O(turns²) serialized evidence: doubling the
    # turn count roughly quadruples the bytes that projection must validate.
    # Preserve the turn-count curve as diagnostics, but gate near-linear cost in
    # the actual input size rather than imposing an impossible per-turn ratio.
    normalized = [
        elapsed / captured_bytes for _, elapsed, captured_bytes in measurements
    ]
    assert normalized[1] < normalized[0] * 2, measurements
    assert normalized[2] < normalized[1] * 2, measurements


def test_model_patterns_select_matching_histories_only() -> None:
    policy_12 = _chat(
        [{"role": "user", "content": "one"}],
        "first",
        model="policy@12",
    )
    judge = _chat(
        [{"role": "user", "content": "judge"}],
        "score",
        model="judge@4",
        offset=1,
    )
    policy_13 = _chat(
        [{"role": "user", "content": "two"}],
        "second",
        model="policy@13",
        offset=2,
    )
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[policy_12, judge, policy_13])
    )

    histories = trajectory.chat_completions_histories(model="policy@*")
    assert [history.model for history in histories] == ["policy@12", "policy@13"]
    generic_histories = trajectory.histories(model="policy@*")
    assert all(isinstance(history, art.History) for history in generic_histories)
    assert [cast(art.History, history).model for history in generic_histories] == [
        "policy@12",
        "policy@13",
    ]
    assert trajectory.chat_completions_history(model="policy@12").model == "policy@12"
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.history(model="policy@*")
    with pytest.raises(ValueError, match="no Chat Completions exchanges"):
        trajectory.chat_completions_histories(model="foreign@*")


def test_chat_history_preserves_provider_specific_nested_fields() -> None:
    exchange = _chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "one",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
        "first",
    )

    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[exchange])
    )
    history = trajectory.chat_completions_history()

    content = cast(list[dict[str, object]], history.messages[0]["content"])
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    dumped = trajectory.model_dump(mode="json", warnings="error")
    dumped_content = dumped["exchanges"]["chat_completions"][0]["request"]["messages"][
        0
    ]["content"]
    assert dumped_content[0]["cache_control"] == {"type": "ephemeral"}


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


def test_chat_history_normalizes_empty_response_only_fields() -> None:
    first = _chat([{"role": "user", "content": "one"}], "first")
    data = first.response.model_dump(mode="python")
    data["choices"][0]["message"]["tool_calls"] = []
    data["choices"][0]["message"]["annotations"] = []
    first.response = ChatCompletion.model_validate(data)
    second = _chat(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "two"},
        ],
        "second",
        offset=1,
    )

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    ).chat_completions_history()

    assert len(history.messages) == 4
    assert "tool_calls" not in history.messages[1]
    assert "annotations" not in history.messages[1]
    assert history.message_sources[1] is not None
    assert history.message_sources[1].exchange is first


def test_chat_history_normalizes_assistant_missing_content_to_empty() -> None:
    first = _chat([{"role": "user", "content": ""}], "")
    data = first.response.model_dump(mode="python")
    data["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "annotations": [],
    }
    first.response = ChatCompletion.model_validate(data)
    second = _chat(
        [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "next"},
        ],
        "second",
        offset=1,
    )

    history = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    ).chat_completions_history()

    assert [message["content"] for message in history.messages] == [
        "",
        "",
        "next",
        "second",
    ]
    assert "annotations" not in history.messages[1]
    assert history.message_sources[1] is not None
    assert history.message_sources[1].exchange is first


@pytest.mark.parametrize("indices", [[], [0, 0]])
def test_chat_history_rejects_missing_or_duplicate_choice_indices(
    indices: list[int],
) -> None:
    exchange = _chat([{"role": "user", "content": "one"}], "first")
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    data["choices"] = [{**choice, "index": index} for index in indices]
    exchange.response = ChatCompletion.model_validate(data)

    with pytest.raises(ValueError, match="choices|choice indices"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=[exchange])
        ).chat_completions_histories()


def test_same_content_seeded_inputs_remain_request_sourced() -> None:
    first_chat = _chat([{"role": "user", "content": "prompt"}], "same")
    seeded_chat = _chat([{"role": "assistant", "content": "same"}], "next", offset=1)
    chat_histories = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first_chat, seeded_chat])
    ).chat_completions_histories()
    seeded_chat_source = chat_histories[1].message_sources[0]
    assert seeded_chat_source is not None
    assert seeded_chat_source.exchange is seeded_chat
    assert seeded_chat_source.request_index == 0

    first_message = _message()
    seeded_message = _message()
    seeded_message.start_time, seeded_message.end_time = _times(1)
    seeded_message.request["messages"] = [{"role": "assistant", "content": "Hi"}]
    message_histories = art.Trajectory(
        exchanges=TrajectoryExchanges(messages=[first_message, seeded_message])
    ).anthropic_messages_histories()
    seeded_message_source = message_histories[1].message_sources[0]
    assert seeded_message_source is not None
    assert seeded_message_source.exchange is seeded_message
    assert seeded_message_source.request_index == 0

    first_response = _response("response-1", "same")
    seeded_response = _response("response-2", "next", offset=1)
    seeded_response.request["input"] = [
        first_response.response.output[0].model_dump(mode="json", exclude_none=True)
    ]
    response_histories = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[first_response, seeded_response])
    ).responses_histories()
    seeded_response_source = response_histories[1].input_sources[0]
    assert seeded_response_source is not None
    assert seeded_response_source.exchange is seeded_response
    assert seeded_response_source.request_index == 0


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
            return {"one": [10], "first": [20], "next": [30]}[text]

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
        assert source.output_indices is None
    response_source = converted.message_sources[-1]
    assert response_source is not None
    assert response_source.exchange is exchange
    assert response_source.output_indices == (0,)

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
    assert chat_history.message_sources[1].output_indices == (0, 1)

    trajectory.exchanges.responses[1].request["previous_response_id"] = "missing"
    external = trajectory.responses_histories()
    assert len(external) == 2
    assert external[1].previous_response_id == "missing"


def test_responses_chat_conversion_preserves_request_and_output_sources() -> None:
    exchange = _response("response-mixed-source", "answer")
    exchange.request["input"] = [
        {
            "id": "request-reasoning",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "prior thought"}],
        }
    ]

    converted = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )

    assert converted.messages == [
        {
            "role": "assistant",
            "content": "",
            "reasoning": "prior thought",
        },
        {
            "role": "assistant",
            "content": "answer",
        },
    ]
    request_source, output_source = converted.message_sources
    assert request_source is not None
    assert request_source.exchange is exchange
    assert request_source.request_index == 0
    assert request_source.output_indices is None
    assert output_source is not None
    assert output_source.exchange is exchange
    assert output_source.request_index is None
    assert output_source.output_indices == (0,)


def test_responses_chat_conversion_splits_cross_exchange_assistant_sources() -> None:
    first = _response("response-reasoning", "", reasoning="think")
    first_data = first.response.model_dump(mode="python")
    first_data["output"] = first_data["output"][:1]
    first.response = Response.model_validate(first_data)
    second = _response(
        "response-answer",
        "answer",
        previous_response_id="response-reasoning",
        offset=1,
    )
    second.request["input"] = []

    converted = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[first, second]))
        .responses_history()
        .as_chat_completions_history()
    )

    assert converted.messages[-2:] == [
        {"role": "assistant", "content": "", "reasoning": "think"},
        {"role": "assistant", "content": "answer"},
    ]
    first_source, second_source = converted.message_sources[-2:]
    assert first_source is not None and first_source.exchange is first
    assert first_source.output_indices == (0,)
    assert second_source is not None and second_source.exchange is second
    assert second_source.output_indices == (0,)


def test_responses_chat_conversion_owns_request_tool_group_by_first_item() -> None:
    exchange = _response("response-request-tools", "answer")
    exchange.request["input"] = [
        {
            "type": "function_call",
            "call_id": f"call-{index}",
            "name": f"tool_{index}",
            "arguments": "{}",
        }
        for index in range(2)
    ]

    converted = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )

    assert len(converted.messages[0].get("tool_calls", [])) == 2
    source = converted.message_sources[0]
    assert source is not None
    assert source.exchange is exchange
    assert source.request_index == 0
    assert source.output_indices is None


def test_responses_history_propagates_opaque_context_and_first_sources() -> None:
    first = _response(
        "response-1",
        "first",
        previous_response_id="outside-trajectory",
    )
    first.request["conversation"] = "conversation-1"
    second = _response(
        "response-2",
        "second",
        previous_response_id="response-1",
        offset=1,
    )
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[first, second])
    )

    history = trajectory.responses_history()

    assert history.previous_response_id == "outside-trajectory"
    assert history.conversation == "conversation-1"
    assert history.input_sources[1] is not None
    assert history.input_sources[1].exchange is first
    assert history.input_sources[1].output_index == 0


def test_branch_context_sources_follow_the_request_that_supplied_the_context() -> None:
    first_message = _message()
    second_message = _message()
    second_message.start_time, second_message.end_time = _times(1)
    second_message.request["system"] = "New instructions"
    second_message.request["messages"] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Again"},
    ]
    message_histories = art.Trajectory(
        exchanges=TrajectoryExchanges(messages=[first_message, second_message])
    ).anthropic_messages_histories()

    assert message_histories[-1].system == "New instructions"
    assert message_histories[-1].system_source is second_message

    first_response = _response("response-1", "first")
    first_response.request["instructions"] = "Old instructions"
    second_response = _response(
        "response-2",
        "second",
        previous_response_id="response-1",
        offset=1,
    )
    second_response.request["instructions"] = "New instructions"
    response_histories = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[first_response, second_response])
    ).responses_histories()

    assert response_histories[-1].instructions == "New instructions"
    assert response_histories[-1].instructions_source is second_response


def test_responses_history_maps_and_validates_generation_sources() -> None:
    exchange = _response("response-1", "first", reasoning="think")
    assert exchange.response.__pydantic_extra__ is not None
    exchange.response.__pydantic_extra__["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.1}],
            "output_indices": [0, 1],
        }
    ]

    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
    history = trajectory.responses_history()

    assert {
        source.generation_index
        for source in history.input_sources
        if source is not None and source.output_index is not None
    } == {0}

    exchange.response.__pydantic_extra__["token_generations"][0]["output_indices"] = [0]
    with pytest.raises(ValueError, match="every sampled output item"):
        trajectory.responses_history()

    exchange.response.__pydantic_extra__["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2}],
            "output_indices": [0],
        },
        {
            "prompt_token_ids": [1, 2],
            "output_tokens": [{"token_id": 3}],
            "output_indices": [0],
        },
    ]
    with pytest.raises(ValueError, match="nonoverlapping"):
        trajectory.responses_history()


def test_cross_exchange_responses_reasoning_stripping_splits_histories() -> None:
    first = _response("response-1", "first", reasoning="think")
    first_data = first.response.model_dump(mode="python")
    first_data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [
                {"token_id": 2, "logprob": -0.2},
                {"token_id": 3, "logprob": -0.3},
            ],
            "output_indices": [0, 1],
        }
    ]
    first.response = Response.model_validate(first_data)
    second = _response(
        "response-2",
        "second",
        previous_response_id="response-1",
        offset=1,
    )
    second_data = second.response.model_dump(mode="python")
    second_data["token_generations"] = [
        {
            "prompt_token_ids": [1, 3, 4],
            "output_tokens": [{"token_id": 5, "logprob": -0.5}],
            "output_indices": [0],
        }
    ]
    second.response = Response.model_validate(second_data)

    histories = art.Trajectory(
        exchanges=TrajectoryExchanges(responses=[first, second])
    ).responses_histories()

    assert len(histories) == 2
    assert any(item.get("type") == "reasoning" for item in histories[0].input)
    assert all(item.get("type") != "reasoning" for item in histories[1].input)
    first_answer_source = histories[1].input_sources[1]
    assert first_answer_source is not None
    assert first_answer_source.exchange is first
    assert first_answer_source.generation_index == 0


@pytest.mark.parametrize(
    ("second_prompt", "reasoning", "expected_histories"),
    [([1, 2, 3], False, 1), ([1, 3], True, 2)],
)
def test_responses_multi_generation_history_leaves_match_tokenization(
    second_prompt: list[int], reasoning: bool, expected_histories: int
) -> None:
    exchange = _response(
        "response-1", "first", reasoning="think" if reasoning else None
    )
    data = exchange.response.model_dump(mode="python")
    first_output_indices = list(range(len(data["output"])))
    tool_output_index = len(data["output"])
    second_output_index = tool_output_index + 1
    data["output"].extend(
        [
            {
                "id": "tool-output",
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "result",
                "status": "completed",
            },
            {
                "id": "message-second",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "second",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            },
        ]
    )
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2}],
            "output_indices": first_output_indices,
        },
        {
            "prompt_token_ids": second_prompt,
            "output_tokens": [{"token_id": 4, "logprob": -0.4}],
            "output_indices": [second_output_index],
        },
    ]
    exchange.response = Response.model_validate(data)
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))

    histories = trajectory.responses_histories()
    tokenized = trajectory.tokenize(multi_history=True)

    assert len(histories) == expected_histories
    assert len(tokenized.histories) == expected_histories
    final_sources = histories[-1].input_sources
    assert final_sources[-2] is not None
    assert final_sources[-2].generation_index is None
    assert final_sources[-1] is not None
    assert final_sources[-1].generation_index == 1
    if expected_histories == 2:
        assert final_sources[1] is not None
        assert final_sources[1].generation_index is None
        assert all(item.get("type") != "reasoning" for item in histories[-1].input)
    converted = histories[-1].as_chat_completions_history()
    assert any(
        source is not None
        and source.generation_index == 1
        and source.output_indices == (second_output_index,)
        for source in converted.message_sources
    )


def test_responses_generation_only_chat_source_has_empty_output_indices() -> None:
    exchange = _response("response-empty-generation", "")
    data = exchange.response.model_dump(mode="python")
    data["output"] = []
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2}],
            "output_indices": [],
        }
    ]
    exchange.response = Response.model_validate(data)

    converted = (
        art.Trajectory(exchanges=TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )

    source = converted.message_sources[-1]
    assert converted.messages[-1] == {"role": "assistant", "content": ""}
    assert source is not None
    assert source.output_indices == ()
    assert source.generation_index == 0


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


def test_batched_completions_prefer_exact_prompt_association() -> None:
    exchange = _completion([1], [10])
    exchange.request["prompt"] = [[1], [2]]
    response = exchange.response.model_dump(mode="python")
    response["choices"] = [
        {
            "index": 0,
            "finish_reason": "stop",
            "text": "second",
            "prompt_token_ids": [2],
            "token_ids": [20],
        },
        {
            "index": 1,
            "finish_reason": "stop",
            "text": "first",
            "prompt_token_ids": [1],
            "token_ids": [10],
        },
    ]
    exchange.response = Completion.model_validate(response)
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(completions=[exchange]))

    histories = trajectory.completions_token_histories()

    assert [history.prompt for history in histories] == [[1, 10], [2, 20]]


def test_batched_completions_honor_interleaved_explicit_prompt_indices() -> None:
    exchange = _completion([1], [10])
    exchange.request["prompt"] = ["first", "second"]
    response = exchange.response.model_dump(mode="python")
    response["choices"] = [
        {
            "index": choice_index,
            "finish_reason": "stop",
            "text": text,
            "prompt_index": prompt_index,
        }
        for choice_index, prompt_index, text in (
            (0, 1, "B0"),
            (1, 0, "A0"),
            (2, 1, "B1"),
            (3, 0, "A1"),
        )
    ]
    exchange.response = Completion.model_validate(response)

    histories = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exchange])
    ).completions_string_histories()

    assert [history.prompt for history in histories] == [
        "firstA0",
        "firstA1",
        "secondB0",
        "secondB1",
    ]


@pytest.mark.parametrize(("prompt_index", "raises"), [(0, False), (1, True)])
def test_batched_completions_validate_partial_prompt_index_fallback(
    prompt_index: int, raises: bool
) -> None:
    exchange = _completion([1], [10])
    exchange.request["prompt"] = ["first", "second"]
    response = exchange.response.model_dump(mode="python")
    response["choices"] = [
        {
            "index": index,
            "finish_reason": "stop",
            "text": text,
            **({"prompt_index": prompt_index} if index == 0 else {}),
        }
        for index, text in enumerate(("A0", "A1", "B0", "B1"))
    ]
    exchange.response = Completion.model_validate(response)
    trajectory = art.Trajectory(exchanges=TrajectoryExchanges(completions=[exchange]))

    if raises:
        with pytest.raises(ValueError, match="contradicts choice indices"):
            trajectory.completions_string_histories()
        return
    assert [
        history.prompt for history in trajectory.completions_string_histories()
    ] == ["firstA0", "firstA1", "secondB0", "secondB1"]


@pytest.mark.parametrize("prompt_index", [-1, 2, True, None])
def test_batched_completions_reject_invalid_explicit_prompt_index(
    prompt_index: object,
) -> None:
    exchange = _completion([1], [10])
    exchange.request["prompt"] = ["first", "second"]
    response = exchange.response.model_dump(mode="python")
    response["choices"] = [
        {
            "index": 0,
            "finish_reason": "stop",
            "text": "answer",
            "prompt_index": prompt_index,
        },
        {
            "index": 1,
            "finish_reason": "stop",
            "text": "answer",
            "prompt_index": 1,
        },
    ]
    exchange.response = Completion.model_validate(response)

    with pytest.raises(ValueError, match="prompt_index"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(completions=[exchange])
        ).completions_string_histories()


def test_batched_completions_reject_prompt_index_exact_evidence_contradiction() -> None:
    exchange = _completion([1], [10])
    exchange.request["prompt"] = [[1], [2]]
    response = exchange.response.model_dump(mode="python")
    response["choices"] = [
        {
            "index": 0,
            "finish_reason": "stop",
            "text": "answer-0",
            "prompt_index": 0,
            "prompt_token_ids": [2],
            "token_ids": [20],
        },
        {
            "index": 1,
            "finish_reason": "stop",
            "text": "answer-1",
            "prompt_index": 1,
            "prompt_token_ids": [1],
            "token_ids": [10],
        },
    ]
    exchange.response = Completion.model_validate(response)

    with pytest.raises(ValueError, match="contradicts exact prompt evidence"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(completions=[exchange])
        ).completions_token_histories()


def test_batched_completions_trust_explicit_string_prompt_index() -> None:
    from art.trajectories._history import _completion_choice_groups

    exchange = _completion([1], [10])
    # Captured provider payloads can be broader than the SDK's prompt union.
    exchange.request["prompt"] = cast(Any, ["same tokenization", [42]])
    response = exchange.response.model_dump(mode="python")
    response["choices"] = [
        {
            "index": index,
            "finish_reason": "stop",
            "text": f"answer-{index}",
            "prompt_index": index,
            "prompt_token_ids": [42],
            "token_ids": [index + 10],
        }
        for index in range(2)
    ]
    exchange.response = Completion.model_validate(response)

    assert [
        [choice.index for choice in group]
        for group in _completion_choice_groups(exchange)
    ] == [[0], [1]]


def test_completions_histories_never_silently_omit_mixed_evidence() -> None:
    exact = _completion([1], [2])
    missing = _completion([3], [4], offset=1)
    missing.request["prompt"] = "question"
    response = missing.response.model_dump(mode="python")
    response["choices"][0].pop("prompt_token_ids")
    response["choices"][0].pop("token_ids")
    missing.response = Completion.model_validate(response)
    trajectory = art.Trajectory(
        exchanges=TrajectoryExchanges(completions=[exact, missing])
    )

    with pytest.raises(ValueError, match="text prompts for every choice"):
        trajectory.histories()


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


def test_completions_reject_duplicate_choice_indices() -> None:
    exchange = _completion([1], [10])
    data = exchange.response.model_dump(mode="python")
    data["choices"].append(dict(data["choices"][0]))
    exchange.response = Completion.model_validate(data)

    with pytest.raises(ValueError, match="choice indices"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(completions=[exchange])
        ).completions_token_histories()


def test_malformed_anthropic_content_raises_value_error() -> None:
    exchange = _message()
    exchange.request = cast(
        MessagesRequest,
        {"messages": [{"role": "user", "content": None}]},
    )

    with pytest.raises(ValueError, match="Anthropic message content"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(messages=[exchange])
        ).anthropic_messages_histories()


def test_tokenless_responses_generation_raises_value_error() -> None:
    exchange = _response("response-1", "first")
    data = exchange.response.model_dump(mode="python")
    data["output"] = [
        {
            "id": "tool-output",
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "result",
            "status": "completed",
        }
    ]
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_indices": [0],
        }
    ]
    exchange.response = Response.model_validate(data)

    with pytest.raises(ValueError, match="without exact output tokens"):
        art.Trajectory(
            exchanges=TrajectoryExchanges(responses=[exchange])
        ).responses_histories()


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
            {"role": "assistant", "content": "first"},
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
    assert histories[1].message_sources[1] is not None
    assert histories[1].message_sources[1].exchange is first
    assert histories[1].message_sources[1].choice_index == 0
    with pytest.raises(ValueError, match="exactly one history"):
        trajectory.tokenize()


def test_reasoning_stripped_tool_call_keeps_first_sampled_source() -> None:
    first = _chat([{"role": "user", "content": "one"}], "")
    first_data = first.response.model_dump(mode="python")
    first_data["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "reasoning": "thought",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    }
    first.response = ChatCompletion.model_validate(first_data)
    second = _chat(
        [
            {"role": "user", "content": "one"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": first_data["choices"][0]["message"]["tool_calls"],
            },
            {"role": "user", "content": "two"},
        ],
        "second",
        offset=1,
    )

    histories = art.Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[first, second])
    ).chat_completions_histories()

    assert len(histories) == 2
    source = histories[1].message_sources[1]
    assert source is not None
    assert source.exchange is first
    assert source.choice_index == 0
    assert source.request_index is None


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
