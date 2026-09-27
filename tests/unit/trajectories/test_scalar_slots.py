from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr


@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("kind", ["str", "float", "enum"])
def test_reused_scalar_slot_state_remains_part_of_context(mutate, kind):
    from enum import Enum

    class Text(str):
        __slots__ = ("state",)

    class Number(float):
        __slots__ = ("state",)

    class Choice(Enum):
        __slots__ = ("state",)
        FIRST = "first"

    setting = cast(
        Any,
        Text("stable")
        if kind == "str"
        else Number(1)
        if kind == "float"
        else Choice.FIRST,
    )
    setting.state = [1]
    options = {"value": setting}
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
                setting.state[0] = 3
            return [setting.state[0], 2]

        def __call__(self, text, **kwargs):
            calls.append("encode")
            return [2 if text == "answer" else setting.state[0]]

    def invoke():
        return value.tokenize(
            tokenizer=cast(Any, Tokenizer()),
            chat_template="custom",
            chat_template_kwargs={"options": options},
        )

    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            result = invoke()
            assert result.tokens == [3, 2]
            assert result.logprobs[-1] == -0.2
            assert setting.state == [3]
    else:
        result = invoke()
        assert result.tokens == [1, 2]
        assert result.logprobs[-1] == -0.2
        assert setting.state == [1]
    assert calls
    assert value.model_dump_json() == before


def test_scalar_slot_snapshot_distinguishes_missing_inherited_and_shadowed_state():
    from art.trajectories._tokenize import _tokenization_context

    class Parent(str):
        __slots__ = ("state",)

    class Child(Parent):
        __slots__ = ("state",)

    value = Child("stable")
    empty = _tokenization_context(value)
    vars(Parent)["state"].__set__(value, [1])
    inherited = _tokenization_context(value)
    assert empty != inherited
    vars(Child)["state"].__set__(value, [2])
    both = _tokenization_context(value)
    assert inherited != both
    vars(Parent)["state"].__get__(value)[0] = 3
    assert both != _tokenization_context(value)
    vars(Parent)["state"].__delete__(value)
    vars(Child)["state"].__delete__(value)
    assert empty == _tokenization_context(value)
