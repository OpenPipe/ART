from functools import cached_property
from typing import Any, cast

from pydantic import BaseModel, PrivateAttr
import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize


class _Model(BaseModel):
    value: int = 0
    _revision: list[int] = PrivateAttr(default_factory=lambda: [0])

    @cached_property
    def cached(self):
        return 0


class _Slots(_Model):
    __slots__ = ("revision",)


class _HiddenDictionary(_Model):
    @property
    def __dict__(self):
        raise AssertionError("model context must inspect physical instance storage")

    @__dict__.setter
    def __dict__(self, value):
        BaseModel.__dict__["__dict__"].__set__(self, value)


def _options(kind):
    value = _HiddenDictionary() if kind == "hidden_dict" else _Model()
    if kind in ("private", "hidden_dict"):
        return (
            value,
            lambda: value._revision[0],
            lambda n: value._revision.__setitem__(0, n),
        )
    if kind == "fields_set":

        def write(n):
            if n:
                value.value = 0
            else:
                value.model_fields_set.clear()

        return value, lambda: len(value.model_dump(exclude_unset=True)), write
    if kind == "cached":
        assert value.cached == 0
        return value, lambda: value.cached, lambda n: setattr(value, "cached", n)
    value = _Slots()
    object.__setattr__(value, "revision", 0)
    descriptor = _Slots.__dict__["revision"]
    return (
        value,
        lambda: descriptor.__get__(value),
        lambda n: descriptor.__set__(value, n),
    )


@pytest.mark.parametrize(
    "kind", ["private", "fields_set", "cached", "slots", "hidden_dict"]
)
def test_model_context_observes_physical_state_and_restoration(kind):
    options, read, write = _options(kind)
    before = _tokenize._tokenization_context([options, options])
    write(1)
    assert read() == 1
    assert _tokenize._tokenization_context([options, options]) != before
    write(0)
    assert _tokenize._tokenization_context([options, options]) == before


@pytest.mark.parametrize(
    "kind", ["private", "fields_set", "cached", "slots", "hidden_dict"]
)
@pytest.mark.parametrize("mutate", [False, True])
def test_model_state_callback_cannot_change_rendering_or_restore_later(kind, mutate):
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
