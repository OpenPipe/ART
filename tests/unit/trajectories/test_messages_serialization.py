from datetime import UTC, datetime
import json
from typing import Any, Literal

from anthropic.types import Message
import pytest

from art.trajectories import (
    MessagesExchange,
    Trajectory,
)


def _message(stop_sequence: str) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "id": "msg_01X9vhvwnz4Vzqz3X5V",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [
            {
                "type": "text",
                "text": "Here is the response.",
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 25,
            "output_tokens": 12,
        },
    }
    if stop_sequence != "absent":
        msg["stop_sequence"] = None if stop_sequence == "null" else stop_sequence
    return msg


def _trajectory(raw: dict[str, Any]) -> Trajectory:
    now = datetime.now(UTC)
    trajectory = Trajectory()
    trajectory.exchanges.messages.append(
        MessagesExchange(
            request={"model": "claude-3-5-sonnet-20241022", "messages": []},
            response=Message.model_validate(raw),
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


@pytest.mark.parametrize("stop_sequence", ["absent", "null", "STOP"])
@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("mode", ["python", "json", "json_string"])
def test_messages_round_trip_preserves_provider_fields(
    stop_sequence: str, compact: bool, mode: Literal["python", "json", "json_string"]
) -> None:
    raw = _message(stop_sequence)
    dumped = _dump(_trajectory(raw), mode, exclude_defaults=compact)
    assert dumped["exchanges"]["messages"][0]["response"] == raw

    restored = Trajectory.model_validate(dumped)
    replayed = restored.exchanges.messages[0].response.model_copy(deep=True)
    assert replayed.model_dump(mode="json", exclude_unset=True) == raw


@pytest.mark.parametrize("mode", ["python", "json", "json_string"])
def test_messages_nested_include_and_exclude(
    mode: Literal["python", "json", "json_string"],
) -> None:
    trajectory = _trajectory(_message("null"))
    included = _dump(
        trajectory,
        mode,
        include={
            "exchanges": {
                "messages": {
                    0: {
                        "response": {
                            "id",
                            "model",
                            "stop_sequence",
                        }
                    }
                }
            }
        },
        exclude={
            "exchanges": {"messages": {0: {"response": {"id"}}}}
        },
    )
    assert included == {
        "exchanges": {
            "messages": [
                {
                    "response": {
                        "model": "claude-3-5-sonnet-20241022",
                        "stop_sequence": None,
                    }
                }
            ]
        }
    }
    excluded = _dump(trajectory, mode, exclude_none=True)
    response = excluded["exchanges"]["messages"][0]["response"]
    assert "stop_sequence" not in response


def test_legacy_messages_tapes_keep_explicit_nulls() -> None:
    raw = Message.model_validate(_message("absent")).model_dump(mode="json")
    restored = Trajectory.model_validate_json(
        _trajectory(raw).model_dump_json(exclude_defaults=False)
    )
    assert _dump(restored, "json")["exchanges"]["messages"][0]["response"] == raw
