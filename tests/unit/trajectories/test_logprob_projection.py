from collections import UserDict
from types import SimpleNamespace
from typing import Any, cast

from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob,
    TopLogprob,
)
from pydantic import BaseModel
import pytest

from art.trajectories._tokenize import (
    _chat_logprob_fingerprint_evidence,
    _fingerprint,
    _token_logprob_data,
)

FIELDS = ("token", "token_id", "logprob", "bytes")


def evidence(row, refusal=False):
    choice = SimpleNamespace(
        logprobs=SimpleNamespace(
            content=None if refusal else [row], refusal=[row] if refusal else None
        )
    )
    result = _chat_logprob_fingerprint_evidence(cast(Any, choice))
    assert result is not None
    return cast(list[dict[str, object]], result["refusal" if refusal else "content"])[0]


def reference(row):
    data = _token_logprob_data(row)
    return {key: data[key] for key in FIELDS if key in data}


@pytest.mark.parametrize("refusal", [False, True])
@pytest.mark.parametrize(
    "extra", [None, {}, {"token_id": 7}, {"token": None, "logprob": -0.0, "bytes": [9]}]
)
@pytest.mark.parametrize("missing", [None, "token", "logprob", "bytes"])
def test_projection_preserves_order_precedence_missing_fields_and_aliases(
    refusal, extra, missing
):
    row = ChatCompletionTokenLogprob(
        token="word", logprob=-0.2, bytes=[1], top_logprobs=[]
    )
    object.__setattr__(row, "__pydantic_extra__", extra)
    if missing is not None:
        del row.__dict__[missing]
    expected = reference(row)
    actual = evidence(row, refusal)
    assert list(actual) == list(expected)
    assert _fingerprint(actual) == _fingerprint(expected)
    assert all(actual[key] is value for key, value in expected.items())
    assert actual is not row.__dict__ and actual is not extra


@pytest.mark.parametrize("row_type", [dict, UserDict])
def test_mapping_rows_keep_the_existing_projection(row_type):
    row = row_type(token="word", token_id=7, logprob=-0.2, bytes=[1], ignored=[2])
    assert evidence(row) == reference(row)
    assert evidence(row)["bytes"] is row["bytes"]


def test_typed_subclass_keeps_stored_field_and_extra_observation_order():
    calls = []

    class Row(ChatCompletionTokenLogprob):
        @property
        def model_extra(self):
            calls.append("extras")
            self.__dict__["logprob"] = -9.0
            return {"token_id": 7}

        def model_dump(self, *args, **kwargs):
            raise AssertionError("Typed row serialization must not execute")

    row = Row(token="word", logprob=-0.2, bytes=[1], top_logprobs=[])
    assert evidence(row) == {
        "token": "word",
        "token_id": 7,
        "logprob": -0.2,
        "bytes": [1],
    }
    assert row.logprob == -9.0 and calls == ["extras"]


def test_exact_row_captures_stored_fields_before_extra_property(monkeypatch):
    row = ChatCompletionTokenLogprob(
        token="word", logprob=-0.2, bytes=[1], top_logprobs=[]
    )
    calls = []

    def extras(self):
        calls.append("extras")
        self.__dict__["logprob"] = -9.0
        return {"token_id": 7}

    monkeypatch.setattr(ChatCompletionTokenLogprob, "model_extra", property(extras))
    assert evidence(row) == {
        "token": "word",
        "token_id": 7,
        "logprob": -0.2,
        "bytes": [1],
    }
    assert row.logprob == -9.0 and calls == ["extras"]


def test_rich_extra_mapping_keeps_unpacking_callbacks_and_precedence():
    calls = []

    class Extras(UserDict):
        def __getitem__(self, key):
            calls.append(key)
            return super().__getitem__(key)

    row = ChatCompletionTokenLogprob(
        token="word", logprob=-0.2, bytes=[1], top_logprobs=[]
    )
    object.__setattr__(
        row,
        "__pydantic_extra__",
        Extras({"ignored": 1, "token_id": 7, "logprob": -0.4}),
    )
    expected = reference(row)
    observed = list(calls)
    calls.clear()
    assert evidence(row) == expected
    assert calls == observed == ["ignored", "token_id", "logprob"]


def test_other_models_keep_serializer_output_and_exception_identity():
    failure = RuntimeError("row serialization failed")
    calls = []

    class Row(BaseModel):
        def model_dump(self, *args, **kwargs):
            calls.append(kwargs)
            raise failure

    with pytest.raises(RuntimeError) as caught:
        evidence(Row())
    assert caught.value is failure and calls == [{"mode": "python"}]


@pytest.mark.parametrize(
    "field,value",
    [("token", "changed"), ("token_id", 8), ("logprob", -9.0), ("bytes", [8])],
)
def test_projection_reads_fresh_mutable_evidence(field, value):
    row = ChatCompletionTokenLogprob(
        token="word", logprob=-0.2, bytes=[1], top_logprobs=[]
    )
    before = _fingerprint(evidence(row))
    if field == "token_id":
        cast(dict[str, Any], row.model_extra)[field] = value
    else:
        row.__dict__[field] = value
    assert _fingerprint(evidence(row)) != before


@pytest.mark.parametrize("refusal", [False, True])
def test_ignored_values_live_through_extra_observation(monkeypatch, refusal):
    events = []
    held = []

    class Alternative(TopLogprob):
        def __del__(self):
            events.append("finalizer")
            held[0].__dict__["logprob"] = -9.0

    row = ChatCompletionTokenLogprob(
        token="word",
        logprob=-0.2,
        bytes=[1],
        top_logprobs=[Alternative(token="word", logprob=-0.3, bytes=[2])],
    )
    held.append(row)

    def extras(self):
        events.append("extras-enter")
        self.__dict__.pop("top_logprobs")
        events.append("extras-read")
        return {"logprob": self.__dict__["logprob"]}

    monkeypatch.setattr(ChatCompletionTokenLogprob, "model_extra", property(extras))
    assert evidence(row, refusal)["logprob"] == -0.2
    assert events == ["extras-enter", "extras-read", "finalizer"]
    assert row.logprob == -9.0


@pytest.mark.parametrize("refusal", [False, True])
def test_rich_key_equality_follows_extra_observation(monkeypatch, refusal):
    events = []

    class Key(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            events.append("key-equality")
            return str.__eq__(self, other)

    row = ChatCompletionTokenLogprob(
        token="word", logprob=-0.2, bytes=[1], top_logprobs=[]
    )
    row.__dict__[Key("token_id")] = 7

    def extras(self):
        value = -9.0 if "key-equality" in events else -0.2
        events.append("extras-read")
        return {"logprob": value}

    monkeypatch.setattr(ChatCompletionTokenLogprob, "model_extra", property(extras))
    result = evidence(row, refusal)
    assert result["logprob"] == -0.2 and result["token_id"] == 7
    assert events == ["extras-read", "key-equality", "key-equality"]
