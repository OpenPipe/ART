from enum import Enum
from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize


def _scalar(kind, override):
    accesses = []

    def dictionary(self):
        accesses.append("dictionary")
        if override == "raising":
            raise AssertionError("physical storage must bypass the override")
        return {}

    if kind in ("enum", "string_enum"):

        class Options(Enum):
            value = "stable"

            __dict__ = property(dictionary)

        class TextOptions(str, Enum):
            value = "stable"

            __dict__ = property(dictionary)

        options = TextOptions.value if kind == "string_enum" else Options.value
    else:
        base, value = {
            "str": (str, "stable"),
            "int": (int, 7),
            "float": (float, 0.5),
            "bytes": (bytes, b"stable"),
        }[kind]
        physical = type("Physical", (base,), {})
        hidden = type("Hidden", (physical,), {"__dict__": property(dictionary)})
        options = hidden(value)
    setattr(options, "revision", [0])
    return options, accesses


@pytest.mark.parametrize(
    "kind", ["str", "int", "float", "bytes", "enum", "string_enum"]
)
@pytest.mark.parametrize("override", ["hidden", "raising"])
def test_scalar_snapshot_tracks_physical_state_and_restoration(kind, override):
    options, accesses = _scalar(kind, override)
    context = [options, options]
    before = _tokenize._tokenization_context(context)
    validate = _tokenize._tokenization_context_validator(context)
    options.revision[0] = 1
    assert _tokenize._tokenization_context(context) != before
    with pytest.raises(ValueError, match="context changed"):
        validate(True)
    options.revision[0] = 0
    assert _tokenize._tokenization_context(context) == before
    validate(True)
    assert accesses == []


@pytest.mark.parametrize(
    "kind", ["str", "int", "float", "bytes", "enum", "string_enum"]
)
@pytest.mark.parametrize("override", ["hidden", "raising"])
@pytest.mark.parametrize("mutate", [False, True])
def test_scalar_callback_cannot_hide_mutation_or_restore_later(kind, override, mutate):
    options, accesses = _scalar(kind, override)
    _check_callback(options, accesses, mutate)


def _check_callback(options, accesses, mutate):
    exchange = _chat_exchange([1], [2])
    exchange.request["messages"] = [{"role": "user", "content": "q"}]
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    before = value.model_dump_json()
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["options"] is options
            calls.append("render")
            if mutate:
                options.revision[0] = 1
            return [3 if options.revision[0] else 1, 2]

        def __call__(self, text, **kwargs):
            calls.append("encode")
            if mutate:
                options.revision[0] = 0
            return [2 if text == "answer" else 1]

    def invoke():
        return value.tokenize(
            tokenizer=cast(Any, Tokenizer()),
            chat_template="custom",
            chat_template_kwargs={"options": options},
        )

    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            invoke()
        assert calls == ["render"] and options.revision == [1]
    else:
        actual = invoke()
        assert actual.tokens == [1, 2] and actual.logprobs[-1] == -0.2
        assert actual.flags[-1] & tr.TokenFlag.SAMPLED
        assert options.revision == [0] and "render" in calls
    assert accesses == []
    assert value.model_dump_json() == before


@pytest.mark.parametrize(
    "base,value", [(str, "x"), (int, 1), (float, 0.5), (bytes, b"x"), (dict, {})]
)
@pytest.mark.parametrize("override", ["hidden", "raising"])
@pytest.mark.parametrize("route", ["snapshot", "native", "render"])
def test_direct_hidden_dictionary_cannot_be_certified(base, value, override, route):
    accesses = []

    def dictionary(self):
        accesses.append("dictionary")
        if override == "raising":
            raise AssertionError("dictionary override must not be called")
        return {}

    direct = type("Direct", (base,), {"__dict__": property(dictionary)})
    options = direct(value)
    options.revision = [0]
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append("render")
            options.revision[0] = 1
            return [1, 2]

        def __call__(self, text, **kwargs):
            calls.append("encode")
            return [2 if text == "answer" else 1]

    if route == "snapshot":
        with pytest.raises(TypeError, match="Unsupported hidden instance dictionary"):
            _tokenize._tokenization_context(options)
        validate = _tokenize._tokenization_context_validator(options)
        validate(False)
        with pytest.raises(ValueError, match="cannot be checked"):
            validate(True)
    else:
        exchange = _chat_exchange([1], [2])
        trajectory = tr.Trajectory(
            exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
        )
        if route == "native":
            exchange.request["metadata"] = cast(Any, {"unused": options})
            assert trajectory.tokenize().tokens == [1, 2]
        else:
            with pytest.raises(
                TypeError, match="Unsupported hidden instance dictionary"
            ):
                trajectory.tokenize(
                    tokenizer=cast(Any, Tokenizer()),
                    chat_template="custom",
                    chat_template_kwargs={"options": options},
                )
    assert accesses == calls == []
    assert options.revision == [0]


@pytest.mark.parametrize(
    "base,value", [(str, "x"), (int, 1), (float, 0.5), (bytes, b"x"), (dict, {})]
)
def test_storage_free_scalar_and_mapping_do_not_call_dictionary_property(base, value):
    def dictionary(self):
        raise AssertionError("no dictionary exists")

    empty = type("Empty", (base,), {"__slots__": (), "__dict__": property(dictionary)})
    options = empty(value)
    _tokenize._tokenization_context(options)
    with pytest.raises(AttributeError):
        options.revision = 0


@pytest.mark.parametrize("base,value", [(str, "x"), (dict, {})])
@pytest.mark.parametrize("attribute", ["__mro__", "__dict__"])
@pytest.mark.parametrize("route", ["snapshot", "stable_render", "mutating_render"])
def test_metaclass_properties_cannot_hide_native_slots(base, value, attribute, route):
    accesses = []

    def hidden(cls):
        accesses.append(attribute)
        return () if attribute == "__mro__" else {}

    meta = type("Meta", (type,), {attribute: property(hidden)})
    physical = meta("Physical", (base,), {"__slots__": ("revision",)})
    options = physical(value)
    options.revision = [0]
    if route == "snapshot":
        before = _tokenize._tokenization_context([options, options])
        validate = _tokenize._tokenization_context_validator(options)
        options.revision[0] = 1
        assert _tokenize._tokenization_context([options, options]) != before
        with pytest.raises(ValueError, match="context changed"):
            validate(True)
        options.revision[0] = 0
        validate(True)
        assert _tokenize._tokenization_context([options, options]) == before
    else:
        _check_callback(options, accesses, route == "mutating_render")
    assert accesses == []
