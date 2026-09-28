from typing import Any, cast

import pytest
from test_tokenize import (
    _chat_exchange,
    _completion_exchange,
    _message_exchange,
    _response_exchange,
)

import art.trajectories as tr
from art.trajectories._tokenize import _tokenization_context


class _Options(dict):
    owner: object


class _Slots(dict):
    __slots__ = ("owner",)
    owner: object


def _recursive(kind):
    if kind == "list":
        value = []
        value.append(value)
        return value
    value = _Options(key="stable") if kind == "dictionary" else _Slots(key="stable")
    value.owner = value
    return value


@pytest.mark.parametrize("kind", ["dictionary", "slots", "list"])
def test_recursive_context_has_an_unsupported_proof_boundary(kind):
    value = _recursive(kind)
    with pytest.raises(TypeError, match="Unsupported recursive tokenization context"):
        _tokenization_context(value)


@pytest.mark.parametrize("kind", ["dictionary", "slots", "list"])
@pytest.mark.parametrize("authority", ["none", "plain", "dynamic"])
@pytest.mark.parametrize(
    "protocol", ["chat_completions", "messages", "responses", "completions"]
)
def test_native_recursive_context_bypasses_without_callbacks(
    kind, authority, protocol, monkeypatch
):
    calls = []

    class PlainTokenizer:
        eos_token_id = 2

    class DynamicTokenizer:
        @property
        def eos_token_id(self):
            calls.append("eos")
            return 2

    if protocol == "chat_completions":
        exchange = _chat_exchange([1], [2])
    elif protocol == "messages":
        exchange = _message_exchange(
            cast(Any, dict(model="test/model", max_tokens=5, messages=[])),
            prompt_token_ids=[1],
            token_ids=[2],
            logprobs=[-0.2],
        )
    elif protocol == "responses":
        exchange = _response_exchange("public-recursive", 2, prompt_token_ids=[1])
    else:
        exchange = _completion_exchange(prompt="question")
    context = _recursive(kind)
    exchange.request["metadata"] = cast(Any, {"unused": context})
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(**{protocol: [exchange]}))

    def unexpected(*args, **kwargs):
        pytest.fail("Complete native records must not load or render")

    monkeypatch.setattr("art.trajectories._tokenize._load_tokenizer", unexpected)
    tokenizer = (
        None
        if authority == "none"
        else PlainTokenizer()
        if authority == "plain"
        else DynamicTokenizer()
    )
    expected_calls = []
    if authority == "dynamic":
        # Preserve the existing opaque-context callback/refusal boundary.
        exchange.request["metadata"] = cast(Any, {"unused": object()})
        with pytest.raises(ValueError, match="[Cc]ontext"):
            value.tokenize(tokenizer=cast(Any, tokenizer))
        expected_calls = calls.copy()
        calls.clear()
        exchange.request["metadata"] = cast(Any, {"unused": context})
        with pytest.raises(ValueError, match="[Cc]ontext"):
            value.tokenize(tokenizer=cast(Any, tokenizer))
    else:
        result = value.tokenize(tokenizer=cast(Any, tokenizer))
        assert result.tokens == [1, 2]
        assert result.flags[-1] & tr.TokenFlag.SAMPLED
        assert bool(result.flags[-1] & tr.TokenFlag.STOP) == (authority == "plain")
    assert calls == expected_calls
    assert cast(Any, exchange.request["metadata"])["unused"] is context


def test_actual_renderer_recursion_error_is_not_context_admission():
    exchange = _chat_exchange([1], [2])
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    failure = RecursionError("renderer failure")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            raise failure

    with pytest.raises(RecursionError) as actual:
        value.tokenize(tokenizer=cast(Any, Tokenizer()), chat_template="custom")
    assert actual.value is failure
