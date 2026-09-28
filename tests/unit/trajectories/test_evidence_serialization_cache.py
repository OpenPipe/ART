from __future__ import annotations

import copy
from typing import Any, cast

from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest
from test_tokenize import _chat_exchange

import art.trajectories._tokenize as core


@pytest.mark.parametrize(
    ("before", "after"), [(0.0, -0.0), (1, 1.0), (True, 1), (-0.2, -9.0)]
)
@pytest.mark.parametrize("selected", [False, True])
def test_each_guard_still_detects_changed_primitive_evidence(before, after, selected):
    source = cast(Any, _chat_exchange([1], [2]))
    row = source.response.choices[0].logprobs.content[0]
    row.logprob = before
    key = core._exchange_sampled_source_key(source)
    validate = core._sampled_source_validator({key: source})
    validate(None)
    validate(None)
    row.logprob = after
    assert core._exchange_sampled_source_key(source) != key
    with pytest.raises(ValueError, match="Sampled source changed"):
        validate(key if selected else None)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("field", ["prompt", "output", "bytes", "message"])
def test_equal_key_aliases_recheck_mutable_storage(which, field):
    first = cast(Any, _chat_exchange([1], [2]))
    sources = [first, copy.deepcopy(first)]
    key = core._exchange_sampled_source_key(first)
    validate = core._sampled_source_validator([(key, source) for source in sources])
    validate(None)
    validate(None)
    source = sources[which]
    choice = source.response.choices[0]
    if field == "prompt":
        choice.model_extra["prompt_token_ids"][0] = 9
    elif field == "output":
        choice.model_extra["token_ids"][0] = 9
    elif field == "bytes":
        choice.logprobs.content[0].bytes = [9]
    else:
        choice.message.content = "edited"
    with pytest.raises(ValueError, match="Sampled source changed"):
        validate(None)


def test_message_serialization_callback_still_runs_at_every_guard():
    calls = []
    source = cast(Any, _chat_exchange([1], [2]))

    class Message(ChatCompletionMessage):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(True)
            return super().model_dump(*args, **kwargs)

    choice = source.response.choices[0]
    choice.message = Message.model_validate(choice.message.model_dump())
    key = core._exchange_sampled_source_key(source)
    validate = core._sampled_source_validator({key: source})
    calls.clear()
    for _ in range(3):
        validate(None)
    assert len(calls) == 3


def test_fresh_equal_values_reuse_json_only_not_evidence_reads(monkeypatch):
    source = cast(Any, _chat_exchange([1], [2]))
    key = core._exchange_sampled_source_key(source)
    validate = core._sampled_source_validator({key: source})
    original = core._fingerprint
    calls = []

    def record(value):
        calls.append(True)
        return original(value)

    monkeypatch.setattr(core, "_fingerprint", record)
    validate(None)
    source.response = copy.deepcopy(source.response)
    validate(None)
    validate(None)
    assert len(calls) == 1
    source.response.choices[0].logprobs.content[0].logprob = -9.0
    with pytest.raises(ValueError, match="Sampled source changed"):
        validate(None)
    assert len(calls) == 2


def test_rich_values_preserve_original_json_without_executing_reducers(monkeypatch):
    reductions = []

    class Number(float):
        def __reduce_ex__(self, protocol):
            reductions.append(protocol)
            raise AssertionError("must not run reducer")

    source = cast(Any, _chat_exchange([1], [2]))
    row = source.response.choices[0].logprobs.content[0]
    row.logprob = Number(-0.2)
    key = core._exchange_sampled_source_key(source)
    validate = core._sampled_source_validator({key: source})
    original = core._fingerprint
    calls = []

    def record(value):
        calls.append(True)
        return original(value)

    monkeypatch.setattr(core, "_fingerprint", record)
    validate(None)
    validate(None)
    assert len(calls) == 2 and reductions == []
    row.logprob = Number(-9.0)
    with pytest.raises(ValueError, match="Sampled source changed"):
        validate(None)
    assert reductions == []


@pytest.mark.parametrize("evidence", [{"bad": b"bytes"}, {"bad": object()}])
def test_original_json_errors_are_not_hidden(evidence):
    cache = core._RevalidatedFingerprints()
    source = cast(Any, _chat_exchange([1], [2]))
    with pytest.raises(TypeError) as expected:
        core._fingerprint(evidence)
    with pytest.raises(TypeError) as actual:
        cache.fingerprint(source, "chat_completions", 0, evidence)
    assert str(actual.value) == str(expected.value)
    assert not cache.observations


def test_cycles_fall_back_to_original_error():
    evidence: dict[str, Any] = {}
    evidence["cycle"] = evidence
    with pytest.raises(ValueError, match="Circular reference detected"):
        core._RevalidatedFingerprints().fingerprint(
            _chat_exchange([1], [2]), "chat_completions", 0, evidence
        )


def test_observation_bound_and_value_semantics():
    cache = core._RevalidatedFingerprints()
    sources = [_chat_exchange([1], [2], offset=i) for i in range(257)]
    for source in sources:
        assert cache.fingerprint(
            source, "chat_completions", 0, {"a": [1]}
        ) == core._fingerprint({"a": [1]})
    assert len(cache.observations) == 256
    # Distinct layouts/order are permitted by JSON value equality; a miss must
    # recompute, never reject an otherwise unchanged source.
    shared = [1]
    evidence = {"a": shared, "b": shared}
    expected = cache.fingerprint(sources[0], "chat_completions", 0, evidence)
    assert (
        cache.fingerprint(sources[0], "chat_completions", 0, {"b": [1], "a": [1]})
        == expected
    )
    assert (
        cache.fingerprint(sources[0], "chat_completions", 0, {"b": [1], "a": [2]})
        != expected
    )


def test_large_valid_evidence_keeps_original_result_without_retention():
    cache = core._RevalidatedFingerprints()
    source = _chat_exchange([1], [2])
    evidence = {"text": "x" * (9 << 20)}
    expected = core._fingerprint(evidence)
    assert cache.fingerprint(source, "chat_completions", 0, evidence) == expected
    assert not cache.observations and cache.snapshot_bytes == 0


def test_rich_key_metadata_does_not_add_hash_or_equality_callbacks():
    calls = []

    class Index(int):
        def __hash__(self):
            calls.append("hash")
            raise AssertionError("cache must not hash rich key")

        def __eq__(self, other):
            calls.append("eq")
            raise AssertionError("cache must not compare rich key")

    cache = core._RevalidatedFingerprints()
    source = _chat_exchange([1], [2])
    for _ in range(2):
        assert cache.fingerprint(
            source, "chat_completions", Index(0), {"x": 1}
        ) == core._fingerprint({"x": 1})
    assert calls == [] and not cache.observations
