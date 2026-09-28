import pickle

from pydantic import BaseModel
import pytest

import art.trajectories as tr
from art.trajectories import _serialization as serialization

_calls = []


class EffectfulType(type):
    def __eq__(cls, other):
        _calls.append("equality")
        raise RuntimeError("Class equality is not a type check")

    __hash__ = type.__hash__


class Opaque(metaclass=EffectfulType):
    def __init__(self):
        self.value = [1, 2]


class NumericSubclass(int, metaclass=EffectfulType):
    pass


class ForgedModelType(type(BaseModel)):
    def __hash__(cls):
        _calls.append("hash")
        return hash(tr.ChatCompletionsHistory)

    def __eq__(cls, other):
        _calls.append("equality")
        return other is tr.ChatCompletionsHistory


@pytest.mark.parametrize("nested", [False, True])
def test_interning_does_not_compare_opaque_types(nested):
    opaque = Opaque()
    value = [opaque] if nested else opaque
    _calls.clear()
    assert serialization._intern_value(value, {}, {}) is value
    assert opaque.value == [1, 2]
    assert _calls == []


@pytest.mark.parametrize("in_trajectory", [False, True])
def test_opaque_pickle_fallback_retains_custom_metaclass(in_trajectory):
    opaque = Opaque()
    value = tr.Trajectory(metadata={"opaque": opaque}) if in_trajectory else opaque
    _calls.clear()
    restored = pickle.loads(pickle.dumps(value))
    actual = restored.metadata["opaque"] if in_trajectory else restored
    assert type(actual) is Opaque
    assert actual.value == [1, 2]
    assert _calls == []


def test_numeric_subclasses_and_plain_fast_values_are_unchanged():
    subclass = NumericSubclass(3)
    values = [None, True, 1, 1.5, subclass]
    _calls.clear()
    assert serialization._intern_value(values, {}, {}) is values
    assert values[-1] is subclass
    assert values == [None, True, 1, 1.5, 3]
    assert _calls == []


@pytest.mark.parametrize("base", [BaseModel, tr.ChatCompletionsHistory])
def test_history_kind_cannot_be_forged_by_metaclass(base):
    class ForgedHistory(base, metaclass=ForgedModelType):
        unrelated: int = 7

    value = ForgedHistory.model_construct()
    _calls.clear()
    with pytest.raises(TypeError, match="Unsupported history type"):
        serialization.serialize_history(value)
    assert _calls == []


_HISTORIES = [
    (tr.LegacyHistory, "legacy", {"messages_and_choices": []}),
    (
        tr.ChatCompletionsHistory,
        "chat_completions",
        {"model": "public", "messages": [], "message_sources": []},
    ),
    (
        tr.AnthropicMessagesHistory,
        "messages",
        {"model": "public", "messages": [], "message_sources": []},
    ),
    (
        tr.ResponsesHistory,
        "responses",
        {"model": "public", "input": [], "input_sources": []},
    ),
    (
        tr.CompletionsTokenHistory,
        "completions_token",
        {"model": "public", "prompt": [], "prompt_sources": [], "sampled_spans": []},
    ),
    (
        tr.CompletionsStringHistory,
        "completions_string",
        {"model": "public", "prompt": "", "prompt_sources": [], "sampled_spans": []},
    ),
]


@pytest.mark.parametrize("model,kind,arguments", _HISTORIES)
def test_concrete_history_roundtrip_and_existing_subclass_refusal(
    model, kind, arguments
):
    history = model(**arguments)
    encoded = serialization.serialize_history(history)
    assert encoded["kind"] == kind
    restored = serialization.validate_history(encoded)
    assert type(restored) is model
    assert restored == history
    subclass = type("HistorySubclass", (model,), {})(**arguments)
    # Direct model validation already accepts subclasses; serialization is exact.
    assert serialization.validate_history(subclass) is subclass
    with pytest.raises(TypeError, match="Unsupported history type"):
        serialization.serialize_history(subclass)
