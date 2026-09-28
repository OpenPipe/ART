from typing import Any, cast

from pydantic import BaseModel
import pytest
from test_tokenize import _character_template_history, _chat_exchange

import art.trajectories as tr
from art.trajectories._tokenize import _RenderContextGuard, _tokenization_context


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
@pytest.mark.parametrize(
    "error_type",
    [
        TypeError,
        KeyError,
        NotImplementedError,
        ValueError,
        RecursionError,
        RuntimeError,
    ],
)
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


def _one_shot_observer(kind, armed):
    def check():
        if armed:
            raise armed.pop()

    if kind == "items":

        class Options(dict):
            def items(self):
                check()
                return super().items()

        return Options(key="stable")

    class Model(BaseModel):
        value: str = "stable"

        def __getattribute__(self, name):
            if name == "value":
                check()
            return super().__getattribute__(name)

    return Model()


@pytest.mark.parametrize("kind", ["items", "model"])
@pytest.mark.parametrize(
    "error_type",
    [
        TypeError,
        KeyError,
        NotImplementedError,
        ValueError,
        RecursionError,
        RuntimeError,
    ],
)
@pytest.mark.parametrize("observed", ["context", "arguments"])
def test_post_callback_observer_failure_is_original_and_sticky(
    kind, error_type, observed
):
    failure = error_type("one-shot observer failure")
    armed = []
    options = _one_shot_observer(kind, armed)
    guard = _RenderContextGuard(lambda: options if observed == "context" else [])
    calls = []

    def callback(value):
        calls.append("callback")
        armed.append(failure)
        return value

    with pytest.raises(error_type) as actual:
        guard.call(callback, options if observed == "arguments" else None)
    assert actual.value is failure
    assert armed == []
    # An optional-probe fallback cannot turn a transient observer error into
    # permission to accept the now-readable original context.
    with pytest.raises(error_type) as again:
        guard.call(callback, None)
    assert again.value is failure
    assert calls == ["callback"]


@pytest.mark.parametrize("kind", ["items", "model"])
@pytest.mark.parametrize(
    "error_type",
    [
        TypeError,
        KeyError,
        NotImplementedError,
        ValueError,
        RecursionError,
        RuntimeError,
    ],
)
def test_argument_observer_failure_cannot_be_swallowed_by_optional_probe(
    kind, error_type
):
    failure = error_type("argument observer")
    armed = [failure]
    options = _one_shot_observer(kind, armed)
    guard = _RenderContextGuard(lambda: [])
    calls = []
    with pytest.raises(error_type) as actual:
        guard.call(lambda value: calls.append(value), options)
    assert actual.value is failure and armed == []
    with pytest.raises(error_type) as again:
        guard.call(lambda: calls.append("fallback"))
    assert again.value is failure and calls == []


@pytest.mark.parametrize("kind", ["items", "model"])
@pytest.mark.parametrize(
    "error_type",
    [
        TypeError,
        KeyError,
        NotImplementedError,
        ValueError,
        RecursionError,
        RuntimeError,
    ],
)
@pytest.mark.parametrize("validator_kind", ["history", "sampled"])
def test_native_validator_cannot_lose_a_one_shot_observer_error(
    kind, error_type, validator_kind
):
    from art.trajectories import _tokenize

    failure = error_type("native observer")
    armed = []
    options = _one_shot_observer(kind, armed)
    if validator_kind == "history":
        history_check = _tokenize._tokenization_context_validator(options)

        def validate():
            history_check(True)
    else:
        exchange = _chat_exchange([1], [2])
        exchange.request["metadata"] = cast(Any, {"options": options})
        key = _tokenize._exchange_sampled_source_key(exchange)
        source_check = _tokenize._sampled_source_validator({key: exchange})

        def validate():
            source_check(None)

    armed.append(failure)
    with pytest.raises(error_type) as actual:
        validate()
    assert actual.value is failure and armed == []
    # Native capability handlers revalidate before declining to another path.
    with pytest.raises(error_type) as again:
        validate()
    assert again.value is failure


@pytest.mark.parametrize("kind", ["items", "model"])
@pytest.mark.parametrize(
    "error_type",
    [
        TypeError,
        KeyError,
        NotImplementedError,
        ValueError,
        RecursionError,
        RuntimeError,
    ],
)
def test_native_boundary_fallback_cannot_hide_request_observer_failure(
    monkeypatch, kind, error_type
):
    history, tokenizer, _ = _character_template_history()
    source = history.message_sources[3]
    assert source is not None
    armed = []
    failure = error_type("boundary request observer")
    source.exchange.request["metadata"] = cast(
        Any, {"options": _one_shot_observer(kind, armed)}
    )
    decode = tokenizer.decode
    calls = []

    def arm_once(tokens, **kwargs):
        calls.append("decode")
        if len(calls) == 1:
            armed.append(failure)
        return decode(tokens, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", arm_once)
    with pytest.raises(error_type) as actual:
        history.tokenize(tokenizer=tokenizer)
    assert actual.value is failure
    assert calls == ["decode"] and armed == []


@pytest.mark.parametrize("route", ["history", "sampled", "boundary"])
def test_transient_unsupported_state_after_admission_cannot_decline(monkeypatch, route):
    from art.trajectories import _tokenize

    armed = []

    class Transient(dict):
        __slots__ = ()

        def items(self):
            if armed:
                armed.pop()
                return [("temporary", object())]
            return []

    options = Transient()
    if route == "boundary":
        history, tokenizer, _ = _character_template_history()
        source = history.message_sources[3]
        assert source is not None
        source.exchange.request["metadata"] = cast(Any, {"options": options})
        decode = tokenizer.decode
        calls = []

        def arm_once(tokens, **kwargs):
            calls.append("decode")
            if len(calls) == 1:
                armed.append(True)
            return decode(tokens, **kwargs)

        monkeypatch.setattr(tokenizer, "decode", arm_once)
        with pytest.raises(ValueError, match="cannot be checked after admission"):
            history.tokenize(tokenizer=tokenizer)
        assert calls == ["decode"] and armed == []
        return
    if route == "history":
        history_check = _tokenize._tokenization_context_validator(options)

        def validate():
            history_check(True)
    else:
        exchange = _chat_exchange([1], [2])
        exchange.request["metadata"] = cast(Any, {"options": options})
        key = _tokenize._exchange_sampled_source_key(exchange)
        source_check = _tokenize._sampled_source_validator({key: exchange})

        def validate():
            source_check(None)

    armed.append(True)
    with pytest.raises(ValueError, match="cannot be checked after admission") as first:
        validate()
    assert armed == []
    assert isinstance(first.value.__cause__, _tokenize._UnsupportedTokenizationContext)
    with pytest.raises(ValueError) as again:
        validate()
    assert again.value is first.value
