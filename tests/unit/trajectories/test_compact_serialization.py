from __future__ import annotations

import asyncio
import copy
from datetime import datetime
import json
import pickle
import random
import statistics
import sys
import time
from typing import Any

import pydantic
import pytest

import art
import art.trajectories as tr
from art.trajectories import _tokenize
from art.trajectories._capture.core import begin, reset
from art.trajectories._protocols import Endpoint, build_exchange


def _fresh(value: str) -> str:
    return value.encode().decode()


def _long() -> str:
    return "repeated provider value:" + " x" * 80


def _json_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def test_trajectory_construction_interns_nested_models_keys_and_cycles() -> None:
    class ProviderExtra(pydantic.BaseModel, extra="allow"):
        content: str

    class UserObject:
        def __init__(self, value: str) -> None:
            self.value = value

    repeated = _long()
    cycle: list[object] = [_fresh(repeated), _fresh(repeated)]
    cycle.append(cycle)
    shared_tuple = (_fresh(repeated),)
    first_key = _fresh(repeated)
    second_key = _fresh(repeated)
    provider = ProviderExtra(
        content=_fresh(repeated),
        **{second_key: _fresh(repeated)},
    )
    untouched = UserObject(_fresh(repeated))

    trajectory = art.Trajectory(
        metadata={
            first_key: cycle,
            "provider": provider,
            "untouched": untouched,
            "tuple_a": shared_tuple,
            "tuple_b": shared_tuple,
            "set": {_fresh(repeated)},
            "frozenset": frozenset({_fresh(repeated)}),
        }
    )

    canonical = next(key for key in trajectory.metadata if key == repeated)
    assert cycle[0] is canonical
    assert cycle[1] is canonical
    assert cycle[2] is cycle
    assert provider.content is canonical
    assert next(iter(provider.__pydantic_extra__ or {})) is canonical
    assert (provider.__pydantic_extra__ or {})[canonical] is canonical
    assert untouched.value is not canonical
    assert trajectory.metadata["tuple_a"] is trajectory.metadata["tuple_b"]
    assert trajectory.metadata["tuple_a"][0] is canonical
    assert next(iter(trajectory.metadata["set"])) is canonical
    assert next(iter(trajectory.metadata["frozenset"])) is canonical


def test_validation_finish_grouping_copy_and_pickle_preserve_sharing() -> None:
    repeated = _long()
    trajectory = art.Trajectory.model_validate(
        {"metadata": {"first": _fresh(repeated), "second": _fresh(repeated)}}
    )
    assert trajectory.metadata["first"] is trajectory.metadata["second"]
    from_json = art.Trajectory.model_validate_json(
        json.dumps({"metadata": {"first": repeated, "second": repeated}})
    )
    assert from_json.metadata["first"] is from_json.metadata["second"]

    trajectory.metadata["third"] = _fresh(repeated)
    assert trajectory.metadata["third"] is not trajectory.metadata["first"]
    trajectory.finish()
    assert trajectory.metadata["third"] is trajectory.metadata["first"]

    other = art.Trajectory(metadata={"value": _fresh(repeated)})
    group = art.TrajectoryGroup(
        [trajectory, other],
        exceptions=[ValueError(_fresh(repeated))],
        metadata={"value": _fresh(repeated)},
    )
    canonical = trajectory.metadata["first"]
    assert other.metadata["value"] is canonical
    assert group.metadata["value"] is canonical
    assert group.exceptions[0].message is canonical

    assert copy.copy(group).trajectories[0].metadata["first"] is canonical
    deep = copy.deepcopy(group)
    assert (
        deep.trajectories[0].metadata["first"] is deep.trajectories[1].metadata["value"]
    )
    restored = pickle.loads(pickle.dumps(group))
    assert (
        restored.trajectories[0].metadata["first"]
        is restored.trajectories[1].metadata["value"]
    )


def test_interning_does_not_change_model_equality() -> None:
    trajectory = art.Trajectory()
    trajectory.metadata["items"] = [_fresh(_long()), _fresh(_long())]
    before = copy.deepcopy(trajectory)

    trajectory._intern_strings()

    assert trajectory == before


def test_capture_uses_a_scope_pool_and_no_capture_hides_it() -> None:
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": _long(),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
            }
        ],
    }
    request = {
        "model": _fresh(_long()),
        "messages": [{"role": "user", "content": "hi"}],
    }

    with art.Trajectory() as trajectory:
        state, token = begin("POST", "https://example.com/v1/chat/completions", request)
        assert state is not None
        state.status_code = 200
        state.add(json.dumps(body).encode())
        state.finish()
        reset(token)
        exchange = trajectory.exchanges.chat_completions[0]
        assert exchange.request["model"] is exchange.response.model

        with art.no_capture():
            hidden, hidden_token = begin(
                "POST", "https://example.com/v1/chat/completions", request
            )
            assert hidden is None
            assert hidden_token is None


def test_nested_scopes_do_not_share_string_pools() -> None:
    repeated = _long()
    outer = art.Trajectory(metadata={"value": _fresh(repeated)})
    inner = art.Trajectory(metadata={"value": _fresh(repeated)})
    assert outer.metadata["value"] is not inner.metadata["value"]

    with outer:
        with inner:
            pass
        assert outer.metadata["value"] is not inner.metadata["value"]


async def test_concurrent_capture_scopes_keep_independent_pools() -> None:
    repeated = _long()
    ready = 0
    both_ready = asyncio.Event()

    async def capture() -> art.Trajectory:
        nonlocal ready
        with art.Trajectory(metadata={"value": _fresh(repeated)}) as trajectory:
            ready += 1
            if ready == 2:
                both_ready.set()
            await both_ready.wait()
            assert art.current_trajectory() is trajectory
        return trajectory

    first, second = await asyncio.gather(capture(), capture())
    assert first.metadata["value"] is not second.metadata["value"]


def test_normal_pydantic_dumps_are_unchanged() -> None:
    trajectory = art.Trajectory(
        reward=1,
        metadata={"nested": {"items": [_fresh(_long()), _fresh(_long())]}},
    )
    expected = {
        "reward": 1.0,
        "metadata": {"nested": {"items": [_long(), _long()]}},
    }

    assert trajectory.model_dump(mode="json") == expected
    assert json.loads(trajectory.model_dump_json()) == expected


def test_tokenization_boundaries_intern_manual_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repeated = _long()
    trajectory = art.Trajectory()
    trajectory.metadata["items"] = [_fresh(repeated), _fresh(repeated)]

    def tokenize_trajectory(value: art.Trajectory, **_: object) -> object:
        first, second = value.metadata["items"]
        assert first is second
        return object()

    monkeypatch.setattr(_tokenize, "tokenize_trajectory", tokenize_trajectory)
    trajectory.tokenize()

    group = art.TrajectoryGroup()
    group.metadata["items"] = [_fresh(repeated), _fresh(repeated)]

    def tokenize_group(value: art.TrajectoryGroup, **_: object) -> object:
        first, second = value.metadata["items"]
        assert first is second
        return object()

    monkeypatch.setattr(_tokenize, "tokenize_group", tokenize_group)
    group.tokenize()


def test_compact_trajectory_round_trip_and_literal_reference_collision() -> None:
    repeated = '\N{SNOWMAN} "quoted" \\ value ' + "long " * 40
    trajectory = art.Trajectory(
        metadata={
            "literal": "$0",
            "first": _fresh(repeated),
            "second": _fresh(repeated),
        }
    )

    payload = trajectory.compact_dump()

    assert payload["kind"] == "trajectory"
    assert payload["strings"]["$0"] == "$0"
    assert payload["strings"]["$1"] == repeated
    restored = art.trajectories.compact_validate(payload, kind="trajectory")
    assert restored.model_dump() == trajectory.model_dump()
    assert restored.metadata["first"] is restored.metadata["second"]
    assert restored.metadata["literal"] == "$0"

    trajectory.metadata["third"] = _fresh(repeated)
    trajectory.compact_dump()
    assert trajectory.metadata["third"] is trajectory.metadata["first"]


def test_compact_decode_is_one_level_and_unmatched_references_are_literal() -> None:
    payload: tr.CompactTrajectoryPayload = {
        "format": "art.trajectories",
        "version": 1,
        "kind": "trajectory",
        "strings": {"$0": "$1", "$1": "recursive value"},
        "data": {"metadata": {"mapped": "$0", "literal": "$2"}},
    }

    restored = art.trajectories.compact_validate(payload, kind="trajectory")

    assert restored.metadata == {"mapped": "$1", "literal": "$2"}


def test_compact_reference_literal_mapped_away_can_release_its_reference() -> None:
    values = [str(index) + chr(65 + index) * 100 for index in range(10)]
    trajectory = art.Trajectory(
        metadata={
            "references": ["$10"] * 100,
            **{f"value_{index}": [value] * 100 for index, value in enumerate(values)},
        }
    )

    payload = trajectory.compact_dump()

    assert payload["strings"] == {
        "$0": "$10",
        **{f"${index + 1}": value for index, value in enumerate(values)},
    }
    assert art.trajectories.compact_validate(
        payload, kind="trajectory"
    ).model_dump() == (trajectory.model_dump())


def test_compact_dictionary_keys_and_decode_collisions() -> None:
    repeated = _long()
    trajectory = art.Trajectory(
        metadata={
            "first": {_fresh(repeated): 1},
            "second": {_fresh(repeated): 2},
        }
    )
    payload = trajectory.compact_dump()
    restored = art.trajectories.compact_validate(payload, kind="trajectory")
    first = next(iter(restored.metadata["first"]))
    second = next(iter(restored.metadata["second"]))
    assert first is second

    with pytest.raises(ValueError, match="duplicate key"):
        art.trajectories.compact_validate(
            {
                "format": "art.trajectories",
                "version": 1,
                "kind": "trajectory",
                "strings": {"$0": "duplicate"},
                "data": {"metadata": {"$0": 1, "duplicate": 2}},
            },
            kind="trajectory",
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "contain exactly"),
        (
            {
                "format": "other",
                "version": 1,
                "kind": "trajectory",
                "strings": {},
                "data": {},
            },
            "format",
        ),
        (
            {
                "format": "art.trajectories",
                "version": True,
                "kind": "trajectory",
                "strings": {},
                "data": {},
            },
            "version",
        ),
        (
            {
                "format": "art.trajectories",
                "version": 1,
                "kind": "trajectory_group",
                "strings": {},
                "data": {},
            },
            "kind",
        ),
        (
            {
                "format": "art.trajectories",
                "version": 1,
                "kind": "trajectory",
                "strings": {"$00": "bad"},
                "data": {},
            },
            "reference",
        ),
        (
            {
                "format": "art.trajectories",
                "version": 1,
                "kind": "trajectory",
                "strings": {"$0": 1},
                "data": {},
            },
            "values",
        ),
        (
            {
                "format": "art.trajectories",
                "version": 1,
                "kind": "trajectory",
                "strings": {},
                "data": ("not", "json"),
            },
            "JSON-compatible",
        ),
    ],
)
def test_compact_validate_rejects_malformed_envelopes(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        art.trajectories.compact_validate(payload, kind="trajectory")


def test_compact_collection_dump_materializes_once_and_shares_strings() -> None:
    repeated = _long()
    consumed = 0

    def values() -> Any:
        nonlocal consumed
        for _ in range(2):
            consumed += 1
            yield art.Trajectory(metadata={"value": _fresh(repeated)})

    payload = art.trajectories.compact_dump(values())

    assert consumed == 2
    assert payload["kind"] == "trajectories"
    assert isinstance(payload["data"], list)
    for item in payload["data"]:
        assert isinstance(item, dict)
        assert "kind" not in item
    assert list(payload["strings"].values()) == [repeated]
    restored = art.trajectories.compact_validate(payload, kind="trajectories")
    assert len(restored) == 2
    assert restored[0].metadata["value"] is restored[1].metadata["value"]

    groups = [
        art.TrajectoryGroup([art.Trajectory(metadata={"value": _fresh(repeated)})]),
        art.TrajectoryGroup([art.Trajectory(metadata={"value": _fresh(repeated)})]),
    ]
    group_payload = art.trajectories.compact_dump(groups)
    restored_groups = art.trajectories.compact_validate(
        group_payload, kind="trajectory_groups"
    )
    assert (
        restored_groups[0].trajectories[0].metadata["value"]
        is restored_groups[1].trajectories[0].metadata["value"]
    )


def test_compact_singular_plural_kinds_and_empty_collections() -> None:
    trajectory = art.Trajectory(metadata={"a": _long(), "b": _fresh(_long())})
    group = art.TrajectoryGroup([trajectory])

    assert (
        art.trajectories.compact_validate(
            trajectory.compact_dump(), kind="trajectory"
        ).model_dump()
        == trajectory.model_dump()
    )
    assert (
        art.trajectories.compact_validate(
            group.compact_dump(), kind="trajectory_group"
        ).model_dump()
        == group.model_dump()
    )
    assert (
        art.trajectories.compact_validate(
            art.trajectories.compact_dump([trajectory]), kind="trajectories"
        )[0].model_dump()
        == trajectory.model_dump()
    )
    assert (
        art.trajectories.compact_validate(
            art.trajectories.compact_dump([group]), kind="trajectory_groups"
        )[0].model_dump()
        == group.model_dump()
    )
    with pytest.raises(ValueError, match="empty iterable"):
        art.trajectories.compact_dump([])
    empty: tr.CompactTrajectoryPayload = {
        "format": "art.trajectories",
        "version": 1,
        "kind": "trajectories",
        "strings": {},
        "data": [],
    }
    assert art.trajectories.compact_validate(empty, kind="trajectories") == []

    with pytest.raises(TypeError, match="homogeneous"):
        art.trajectories.compact_dump([trajectory, group])
    with pytest.raises(ValueError, match="kind"):
        art.trajectories.compact_validate(
            art.trajectories.compact_dump([trajectory]), kind="trajectory_groups"
        )


def test_compact_profitability_and_determinism_include_complete_envelope() -> None:
    short = art.Trajectory(metadata={"a": "x", "b": "x"})
    assert short.compact_dump()["strings"] == {}

    repeated = _long()
    trajectory = art.Trajectory(
        metadata={"first": repeated, "second": _fresh(repeated)}
    )
    first = trajectory.compact_dump()
    second = trajectory.compact_dump()
    plain = {**first, "strings": {}, "data": trajectory.model_dump(mode="json")}

    assert first == second
    assert _json_size(first) < _json_size(plain)


def test_compact_deterministic_property_cases_lose_no_json_data() -> None:
    rng = random.Random(20260811)
    strings = ["$0", "$01", "snowman \N{SNOWMAN}", _long(), "short"]

    def value(depth: int) -> Any:
        if depth == 0:
            return rng.choice([None, True, rng.randrange(100), *strings])
        if rng.randrange(2):
            return [value(depth - 1) for _ in range(rng.randrange(5))]
        return {
            f"key-{index}-{rng.choice(strings)}": value(depth - 1)
            for index in range(rng.randrange(5))
        }

    for _ in range(50):
        trajectory = art.Trajectory(metadata={"value": value(3)})
        payload = trajectory.compact_dump()
        assert art.trajectories.compact_validate(
            payload, kind="trajectory"
        ).model_dump() == (trajectory.model_dump())


def _protocol_trajectory() -> art.Trajectory:
    model = _long()
    requests_and_bodies: list[tuple[Endpoint, dict[str, Any], dict[str, Any]]] = [
        (
            "chat_completions",
            {"model": model, "messages": [{"role": "user", "content": model}]},
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": model},
                    }
                ],
            },
        ),
        (
            "completions",
            {"model": model, "prompt": model},
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "finish_reason": "stop", "text": model}],
            },
        ),
        (
            "responses",
            {"model": model, "input": model, "instructions": model},
            {
                "id": "resp_1",
                "created_at": 1.0,
                "model": model,
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        ),
        (
            "messages",
            {
                "model": model,
                "system": model,
                "messages": [{"role": "user", "content": model}],
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": model}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
    ]
    now = datetime.now()
    trajectory = art.Trajectory()
    for endpoint, request, body in requests_and_bodies:
        exchange = build_exchange(
            endpoint,
            request,
            json.dumps(body).encode(),
            start_time=now,
            end_time=now,
        )
        getattr(trajectory.exchanges, endpoint).append(exchange)
    trajectory.finish()
    return trajectory


def test_compact_round_trip_all_protocols_and_legacy_histories() -> None:
    trajectory = _protocol_trajectory()
    restored = art.trajectories.compact_validate(
        trajectory.compact_dump(), kind="trajectory"
    )
    assert restored.model_dump() == trajectory.model_dump()

    legacy = art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": _long()},
            {"role": "user", "content": _fresh(_long())},
        ]
    )
    restored_legacy = art.trajectories.compact_validate(
        legacy.compact_dump(), kind="trajectory"
    )
    assert restored_legacy.model_dump() == legacy.model_dump()


def test_tokenized_compact_round_trip_all_protocol_source_shapes() -> None:
    combined = _protocol_trajectory()
    for protocol in ("chat_completions", "completions", "responses", "messages"):
        source = art.Trajectory()
        source_exchanges = getattr(source.exchanges, protocol)
        source_exchanges.extend(getattr(combined.exchanges, protocol))
        history = source.histories()[0]
        assert isinstance(history, tr.History)
        assert history.model is not None
        tokenized = tr.TokenizedTrajectory(
            history=history,
            trajectory=source,
            model=history.model,
            token_ids=[1, 2],
            logprobs=[float("nan"), -0.1],
            flags=[
                tr.TokenFlag.EXACT,
                tr.TokenFlag.EXACT | tr.TokenFlag.SAMPLED,
            ],
            reward=1.0,
            metrics={},
            metadata={},
        )

        ordinary = tr.TokenizedTrajectory.model_validate_json(
            tokenized.model_dump_json(warnings="error")
        )
        assert ordinary.model_dump_json() == tokenized.model_dump_json()

        restored = art.trajectories.compact_validate(
            tokenized.compact_dump(), kind="tokenized_trajectory"
        )

        assert restored.model_dump() == tokenized.model_dump()
        canonical = getattr(restored.trajectory.exchanges, protocol)[0]
        if isinstance(restored.history, tr.ChatCompletionsHistory):
            history_source = next(
                item for item in restored.history.message_sources if item is not None
            )
            assert history_source.exchange is canonical
        elif isinstance(restored.history, tr.AnthropicMessagesHistory):
            assert restored.history.system_source is canonical
        elif isinstance(restored.history, tr.ResponsesHistory):
            assert restored.history.instructions_source is canonical
        elif isinstance(
            restored.history,
            (
                tr.CompletionsTokenHistory,
                tr.CompletionsStringHistory,
            ),
        ):
            history_source = next(
                span.source
                for span in restored.history.prompt_sources
                if span.source is not None
            )
            assert history_source.exchange is canonical
        else:
            raise AssertionError("Unexpected protocol history")
        if protocol == "chat_completions":
            assert _json_size(tokenized.compact_dump()) < len(
                tokenized.model_dump_json().encode()
            )


def test_interning_reduces_pickle_and_compact_json_sizes() -> None:
    trajectory = art.Trajectory()
    repeated = _long() * 4
    trajectory.metadata["items"] = [_fresh(repeated) for _ in range(200)]
    items = trajectory.metadata["items"]
    before_memory = sum(sys.getsizeof(item) for item in items)
    before_pickle = len(pickle.dumps(trajectory))
    trajectory.finish()
    after_memory = sum(
        sys.getsizeof(item) for item in {id(item): item for item in items}.values()
    )
    after_pickle = len(pickle.dumps(trajectory))
    compact = trajectory.compact_dump()
    plain = {**compact, "strings": {}, "data": trajectory.model_dump(mode="json")}

    assert after_memory < before_memory / 100
    assert after_pickle < before_pickle / 4
    assert _json_size(compact) < _json_size(plain) / 4


def test_cloudpickle_preserves_shared_references() -> None:
    cloudpickle = pytest.importorskip("cloudpickle")
    repeated = _long()
    trajectory = art.Trajectory(
        metadata={"items": [_fresh(repeated), _fresh(repeated)]}
    )
    cloud_restored = cloudpickle.loads(cloudpickle.dumps(trajectory))
    assert cloud_restored.metadata["items"][0] is cloud_restored.metadata["items"][1]


def test_interning_traversal_scales_near_linearly() -> None:
    repeated = _long()

    def duration(size: int) -> float:
        samples = []
        for _ in range(5):
            trajectory = art.Trajectory()
            trajectory.metadata["items"] = [_fresh(repeated) for _ in range(size)]
            start = time.perf_counter()
            trajectory._intern_strings()
            samples.append(time.perf_counter() - start)
        return statistics.median(samples)

    small = duration(4_000)
    large = duration(8_000)
    assert large < max(small * 3, 0.05)
