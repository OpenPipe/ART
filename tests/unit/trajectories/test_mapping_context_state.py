from __future__ import annotations

from collections import UserDict
from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize


class _Dictionary(dict):
    revision: int


class _UserDictionary(UserDict):
    revision: int


class _Slots(dict):
    __slots__ = ("revision",)
    revision: int


class _ShadowSlots(_Slots):
    __slots__ = ("revision",)


class _HiddenDictionary(_Dictionary):
    @property
    def __dict__(self):
        raise AssertionError("context snapshots must inspect physical instance storage")


def _options(kind):
    types = {
        "dict": _Dictionary,
        "user_dict": _UserDictionary,
        "slots": _Slots,
        "shadowed_slot": _ShadowSlots,
        "hidden_dict": _HiddenDictionary,
    }
    options = types[kind](key="stable")
    options.revision = 0
    if kind == "shadowed_slot":
        descriptor = _Slots.__dict__["revision"]
        descriptor.__set__(options, 0)
        return (
            options,
            lambda: descriptor.__get__(options),
            lambda value: descriptor.__set__(options, value),
        )
    return (
        options,
        lambda: options.revision,
        lambda value: setattr(options, "revision", value),
    )


@pytest.mark.parametrize(
    "kind", ["dict", "user_dict", "slots", "shadowed_slot", "hidden_dict"]
)
def test_mapping_snapshot_observes_physical_attributes_and_slots(kind):
    options, read, write = _options(kind)
    original_items = list(options.items())
    before = _tokenize._tokenization_context([options, options])
    write(1)
    assert read() == 1 and list(options.items()) == original_items
    assert _tokenize._tokenization_context([options, options]) != before
    write(0)
    assert _tokenize._tokenization_context([options, options]) == before


@pytest.mark.parametrize(
    "kind", ["dict", "user_dict", "slots", "shadowed_slot", "hidden_dict"]
)
@pytest.mark.parametrize("mutate", [False, True])
def test_mapping_attribute_callback_cannot_change_rendering_or_restore_later(
    kind, mutate
):
    options, read, write = _options(kind)
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
                write(1)
            return [3 if read() else 1, 2]

        def __call__(self, text, **kwargs):
            calls.append("encode")
            # A later callback cannot hide the renderer's mutation.
            if mutate:
                write(0)
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
        assert calls == ["render"] and read() == 1
    else:
        actual = invoke()
        assert actual.tokens == [1, 2] and actual.logprobs[-1] == -0.2
        assert actual.flags[-1] & tr.TokenFlag.SAMPLED
        assert read() == 0 and "render" in calls
    assert value.model_dump_json() == before
