from typing import Any, cast

from pydantic import BaseModel
import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr
from art.trajectories._tokenize import _tokenization_context


def _observer(kind, failure):
    if kind in ("items", "partial_items"):

        class Options(dict):
            def items(self):
                if kind == "partial_items":
                    yield "stable", [1]
                raise failure

        return Options(key="stable")

    class Model(BaseModel):
        value: str = "stable"

        def __getattribute__(self, name):
            if name == kind:
                raise failure
            return super().__getattribute__(name)

    return Model()


@pytest.mark.parametrize("kind", ["items", "partial_items", "value", "model_extra"])
@pytest.mark.parametrize("error_type", [RecursionError, RuntimeError])
@pytest.mark.parametrize("route", ["snapshot", "native", "render"])
def test_observer_failure_is_not_an_unsupported_context(kind, error_type, route):
    failure = error_type("observer failure")
    options = _observer(kind, failure)
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append("render")
            return [1, 2]

        def __call__(self, text, **kwargs):
            calls.append("encode")
            return [2 if text == "answer" else 1]

    def invoke():
        if route == "snapshot":
            return _tokenization_context(options)
        exchange = _chat_exchange([1], [2])
        if route == "native":
            exchange.request["metadata"] = cast(Any, {"unused": options})
        value = tr.Trajectory(
            exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
        )
        return (
            value.tokenize()
            if route == "native"
            else value.tokenize(
                tokenizer=cast(Any, Tokenizer()),
                chat_template="custom",
                chat_template_kwargs={"options": options},
            )
        )

    with pytest.raises(error_type) as actual:
        invoke()
    assert actual.value is failure
    assert calls == []
