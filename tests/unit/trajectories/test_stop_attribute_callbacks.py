from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr


@pytest.mark.parametrize(
    "attribute", ["eos_token_id", "eot_token_id", "special_tokens_map"]
)
@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_stop_metadata_lookup_preserves_consumed_logprobs(attribute, mutate, wrapped):
    from art.trajectories import _tokenize as module

    exchange = _chat_exchange([1], [2, 3])
    choice = exchange.response.choices[0]
    assert choice.logprobs is not None and choice.logprobs.content is not None
    logprobs = choice.logprobs.content
    calls = []

    def read(_self):
        calls.append(True)
        if mutate:
            logprobs[0].logprob = -9
        return {} if attribute == "special_tokens_map" else 3

    tokenizer = type("Tokenizer", (), {attribute: property(read)})()
    if wrapped:
        tokenizer = module._RenderingTokenizer(
            cast(Any, tokenizer), module._RenderContextGuard(lambda: [])
        )
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    before = value.model_dump_json()
    if mutate:
        with pytest.raises(ValueError, match="[Ss]ampled source changed"):
            actual = value.tokenize(tokenizer=cast(Any, tokenizer))
            assert actual.tokens == [1, 2, 3]
            assert actual.logprobs[-2:] == [-0.2, -0.3]
            assert logprobs[0].logprob == -9
    else:
        actual = value.tokenize(tokenizer=cast(Any, tokenizer))
        assert actual.tokens == [1, 2, 3]
        assert actual.logprobs[-2:] == [-0.2, -0.3]
        assert bool(actual.flags[-1] & tr.TokenFlag.STOP) == (
            attribute != "special_tokens_map"
        )
        assert value.model_dump_json() == before
    assert calls


@pytest.mark.parametrize("wrapped", [False, True])
def test_plain_eos_metadata_preserves_native_callback_free_path(monkeypatch, wrapped):
    from art.trajectories import _tokenize as module

    class Tokenizer:
        eos_token_id = 3

    def unexpected(*args, **kwargs):
        pytest.fail("Plain EOS metadata must not require a callback guard or renderer")

    monkeypatch.setattr(module, "_sampled_source_validator", unexpected)
    monkeypatch.setattr(module, "_load_tokenizer", unexpected)
    exchange = _chat_exchange([1], [2, 3])
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    tokenizer = (
        module._RenderingTokenizer(
            cast(Any, Tokenizer()), module._RenderContextGuard(lambda: [])
        )
        if wrapped
        else Tokenizer()
    )
    actual = value.tokenize(tokenizer=cast(Any, tokenizer))
    assert actual.tokens == [1, 2, 3]
    assert actual.logprobs[-2:] == [-0.2, -0.3]
    assert actual.flags[-1] & tr.TokenFlag.STOP


def test_recorded_numeric_stop_does_not_read_unused_tokenizer_metadata():
    class Tokenizer:
        @property
        def eos_token_id(self):
            pytest.fail("An explicit recorded STOP needs no tokenizer lookup")

    exchange = _chat_exchange([1], [2, 3])
    assert exchange.response.choices[0].model_extra is not None
    exchange.response.choices[0].model_extra["stop_reason"] = 3
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    actual = value.tokenize(tokenizer=cast(Any, Tokenizer()))
    assert actual.tokens == [1, 2, 3]
    assert actual.flags[-1] & tr.TokenFlag.STOP
