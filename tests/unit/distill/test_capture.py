from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

from openai.types.chat import ChatCompletion
import pytest

from art import Trajectory, distill
from art.trajectories import ChatCompletionsExchange, TrajectoryExchanges


def _exchange(
    *,
    prompt: list[int],
    completion: list[int],
    offset: int,
    content: str = "answer",
    message_extra: dict[str, Any] | None = None,
    policy_spans: list[dict[str, Any]] | None = None,
) -> ChatCompletionsExchange:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    message.update(message_extra or {})
    choice: dict[str, Any] = {
        "index": 0,
        "finish_reason": "stop",
        "message": message,
        "prompt_token_ids": prompt,
        "token_ids": completion,
    }
    if policy_spans is not None:
        choice["policy_token_spans"] = policy_spans
    response = ChatCompletion.model_validate(
        {
            "id": f"response-{offset}",
            "object": "chat.completion",
            "created": offset,
            "model": "student",
            "choices": [choice],
        }
    )
    start = datetime(2026, 1, 1) + timedelta(seconds=offset)
    return ChatCompletionsExchange(
        request={
            "model": "student",
            "messages": [{"role": "user", "content": f"question-{offset}"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        },
        response=response,
        start_time=start,
        end_time=start + timedelta(milliseconds=1),
    )


def _trajectory() -> Trajectory:
    first = _exchange(
        prompt=[1, 2],
        completion=[3, 4],
        offset=0,
        policy_spans=[
            {
                "start_token": 0,
                "end_token": 2,
                "policy_version": 7,
                "lora_slot": "student@7",
                "update_seq": 9,
            }
        ],
    )
    second = _exchange(
        prompt=[1, 2, 3, 4, 5],
        completion=[6],
        offset=1,
    )
    return Trajectory(
        exchanges=TrajectoryExchanges(chat_completions=[second, first]),
        initial_policy_version=7,
        final_policy_version=7,
    )


def test_generations_preserve_exact_ids_order_offsets_and_revisions() -> None:
    captured = distill.generations(_trajectory())

    assert [item.continuation_token_ids for item in captured] == [(3, 4), (6,)]
    assert [item.trajectory_token_start for item in captured] == [2, 5]
    assert [item.event_index for item in captured] == [0, 1]
    assert captured[0].part_spans == (
        distill.PartSpan(
            start=0,
            end=2,
            part=distill.GenerationPart.ASSISTANT_TEXT,
        ),
    )
    assert captured[0].rollout_spans == (
        distill.RolloutRevisionSpan(
            start=0,
            end=2,
            revision=7,
            inference_name="student@7",
            update_seq=9,
        ),
    )
    assert captured[1].rollout_spans == (
        distill.RolloutRevisionSpan(
            start=0,
            end=1,
            revision=7,
        ),
    )
    assert distill.last_generation(_trajectory()) == captured[-1]


def test_generation_identities_and_order_are_stable_and_not_batch_relative() -> None:
    trajectory = _trajectory()

    first = distill.generations(trajectory)
    second = distill.generations(trajectory.model_copy(deep=True))

    assert [item.generation_id for item in first] == [
        item.generation_id for item in second
    ]
    assert [item.trajectory_fingerprint for item in first] == [
        item.trajectory_fingerprint for item in second
    ]
    assert first[0].generation_id != first[1].generation_id


def test_identical_exchange_timestamps_are_rejected_as_ambiguous() -> None:
    first = _exchange(prompt=[1], completion=[2], offset=0)
    second = _exchange(prompt=[1, 2, 3], completion=[4], offset=1)
    second.start_time = first.start_time
    second.end_time = first.end_time

    with pytest.raises(ValueError, match="unambiguous exchange order"):
        distill.generations(
            Trajectory(
                exchanges=TrajectoryExchanges(
                    chat_completions=[first, second],
                )
            )
        )


def test_teacher_view_edits_are_functional_and_mutation_isolated() -> None:
    generation = distill.last_generation(_trajectory())
    original = distill.captured_context(generation)
    original_request = original.request()
    assert isinstance(original_request, dict)
    cast(list[dict[str, Any]], original_request["messages"]).append(
        {"role": "system", "content": "mutated copy"}
    )

    hinted = distill.append_message(
        original,
        {"role": "system", "content": "private hint"},
    )
    tools = [{"type": "function", "function": {"name": "reviewed_lookup"}}]
    patched = distill.with_tools(hinted, tools)
    tools[0]["function"]["name"] = "mutated"

    unchanged = original.request()
    hinted_request = hinted.request()
    patched_request = patched.request()
    assert isinstance(unchanged, dict)
    assert isinstance(hinted_request, dict)
    assert isinstance(patched_request, dict)
    assert unchanged["messages"] == [{"role": "user", "content": "question-1"}]
    hinted_messages = hinted_request["messages"]
    patched_tools = patched_request["tools"]
    assert isinstance(hinted_messages, list)
    assert isinstance(patched_tools, list)
    assert hinted_messages[-1] == {
        "role": "system",
        "content": "private hint",
    }
    assert patched_tools[0]["function"]["name"] == "reviewed_lookup"
    assert patched.fingerprint != hinted.fingerprint != original.fingerprint


def test_append_message_keeps_late_system_hint_distinct_and_source_immutable() -> None:
    original = distill.TeacherView.from_request(
        "chat_completions",
        {
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {"role": "system", "content": "You are a booking assistant."},
                {"role": "user", "content": "Book dinner."},
                {"role": "assistant", "content": "Which restaurant?"},
                {"role": "user", "content": "The reviewed one."},
            ],
        },
    )

    hinted = distill.append_message(
        original,
        {"role": "system", "content": "Prefer the reviewed restaurant."},
    )

    original_request = cast(dict[str, Any], original.request())
    hinted_request = cast(dict[str, Any], hinted.request())
    assert original_request["messages"] == [
        {"role": "system", "content": "You are a booking assistant."},
        {"role": "user", "content": "Book dinner."},
        {"role": "assistant", "content": "Which restaurant?"},
        {"role": "user", "content": "The reviewed one."},
    ]
    assert hinted_request["messages"] == [
        *original_request["messages"],
        {"role": "system", "content": "Prefer the reviewed restaurant."},
    ]
    assert hinted.fingerprint != original.fingerprint


def test_missing_exact_metadata_is_rejected() -> None:
    exchange = _exchange(prompt=[1], completion=[2], offset=0)
    extra = exchange.response.choices[0].model_extra
    assert extra is not None
    extra.pop("prompt_token_ids")

    with pytest.raises(ValueError, match="exact prompt and completion token IDs"):
        distill.generations(
            Trajectory(
                exchanges=TrajectoryExchanges(chat_completions=[exchange]),
            )
        )


@pytest.mark.parametrize(
    "message_extra",
    [
        {"reasoning_content": "thinking"},
        {
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ]
        },
        {"refusal": "no"},
    ],
)
def test_mixed_output_is_explicitly_rejected(
    message_extra: dict[str, Any],
) -> None:
    exchange = _exchange(
        prompt=[1],
        completion=[2],
        offset=0,
        message_extra=message_extra,
    )

    with pytest.raises(ValueError, match="cannot infer token spans"):
        distill.generations(
            Trajectory(
                exchanges=TrajectoryExchanges(chat_completions=[exchange]),
            )
        )
