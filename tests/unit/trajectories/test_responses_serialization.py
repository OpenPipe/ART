from datetime import UTC, datetime
import json
from typing import Any, Literal

from openai.types.chat import ChatCompletion
from openai.types.responses import Response
import pytest

from art.trajectories import (
    ChatCompletionsExchange,
    ResponsesExchange,
    Trajectory,
)


def _response(status: str) -> dict[str, Any]:
    reasoning: dict[str, Any] = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "content": [],
        "encrypted_content": "opaque-reasoning",
        "provider_metadata": {"trace": None},
    }
    if status != "absent":
        reasoning["status"] = None if status == "null" else status
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "behavior",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "metadata": None,
        "output": [
            reasoning,
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"id":"item_1"}',
                "status": "completed",
            },
        ],
    }


def _trajectory(raw: dict[str, Any]) -> Trajectory:
    now = datetime.now(UTC)
    trajectory = Trajectory()
    trajectory.exchanges.responses.append(
        ResponsesExchange(
            request={"model": "behavior", "input": "Find the item."},
            response=Response.model_validate(raw),
            start_time=now,
            end_time=now,
        )
    )
    return trajectory


def _dump(
    trajectory: Trajectory,
    mode: Literal["python", "json", "json_string"],
    **kwargs: Any,
) -> dict[str, Any]:
    if mode == "json_string":
        return json.loads(trajectory.model_dump_json(**kwargs))
    return trajectory.model_dump(mode=mode, **kwargs)


@pytest.mark.parametrize("status", ["absent", "null", "completed"])
@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("mode", ["python", "json", "json_string"])
def test_responses_round_trip_preserves_provider_fields(
    status: str, compact: bool, mode: Literal["python", "json", "json_string"]
) -> None:
    raw = _response(status)
    dumped = _dump(_trajectory(raw), mode, exclude_defaults=compact)
    assert dumped["exchanges"]["responses"][0]["response"] == raw

    restored = Trajectory.model_validate(dumped)
    replayed = restored.exchanges.responses[0].response.model_copy(deep=True)
    assert replayed.model_dump(mode="json", exclude_unset=True) == raw


@pytest.mark.parametrize("mode", ["python", "json", "json_string"])
def test_responses_nested_include_and_exclude(
    mode: Literal["python", "json", "json_string"],
) -> None:
    trajectory = _trajectory(_response("null"))
    included = _dump(
        trajectory,
        mode,
        include={
            "exchanges": {
                "responses": {
                    0: {
                        "response": {
                            "output": {0: {"id", "status", "encrypted_content"}}
                        }
                    }
                }
            }
        },
        exclude={
            "exchanges": {"responses": {0: {"response": {"output": {0: {"id"}}}}}}
        },
    )
    assert included == {
        "exchanges": {
            "responses": [
                {
                    "response": {
                        "output": [
                            {"status": None, "encrypted_content": "opaque-reasoning"}
                        ]
                    }
                }
            ]
        }
    }
    excluded = _dump(trajectory, mode, exclude_none=True)
    response = excluded["exchanges"]["responses"][0]["response"]
    assert "metadata" not in response
    assert "status" not in response["output"][0]


def test_legacy_responses_tapes_keep_explicit_nulls() -> None:
    raw = Response.model_validate(_response("absent")).model_dump(mode="json")
    assert raw["output"][0]["status"] is None
    restored = Trajectory.model_validate_json(
        _trajectory(raw).model_dump_json(exclude_defaults=False)
    )
    assert _dump(restored, "json")["exchanges"]["responses"][0]["response"] == raw


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("mode", ["python", "json", "json_string"])
def test_chat_serialization_is_unchanged(
    compact: bool, mode: Literal["python", "json", "json_string"]
) -> None:
    now = datetime.now(UTC)
    response = ChatCompletion.model_validate(
        {
            "id": "chat_1",
            "object": "chat.completion",
            "created": 1,
            "model": "behavior",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Done."},
                }
            ],
        }
    )
    trajectory = Trajectory()
    trajectory.exchanges.chat_completions.append(
        ChatCompletionsExchange(
            request={"model": "behavior", "messages": []},
            response=response,
            start_time=now,
            end_time=now,
        )
    )
    expected = response.model_dump(
        mode="python" if mode == "python" else "json",
        exclude_defaults=compact if mode == "python" else False,
    )
    dumped = _dump(trajectory, mode, exclude_defaults=compact)
    assert dumped["exchanges"]["chat_completions"][0]["response"] == expected
