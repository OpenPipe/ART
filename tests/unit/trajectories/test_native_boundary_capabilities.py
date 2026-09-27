from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr


@pytest.mark.parametrize("decoder", ["absent", "property"])
@pytest.mark.parametrize(
    "case", ["terminal_stop", "terminal_length", "known_prior", "unproved_prior"]
)
def test_native_output_requires_decode_only_for_unproved_nonterminal_boundary(
    case, decoder
):
    calls = []

    class Tokenizer:
        eos_token_id = 9

        def __call__(self, *args, **kwargs):
            calls.append("encode")
            raise RuntimeError("unneeded encoding")

        def apply_chat_template(self, *args, **kwargs):
            calls.append("render")
            raise RuntimeError("boundary rendering required")

    class DecoderProperty(Tokenizer):
        @property
        def decode(self):
            calls.append("decode")
            raise RuntimeError("boundary decoder required")

    if case in ("known_prior", "unproved_prior"):
        first_output = [2, 9] if case == "known_prior" else [2]
        first = _chat_exchange([1], first_output)
        last = _chat_exchange([1, 2, 9, 3], [4], offset=1)
        exchanges = [first, last]
        expected = [1, 2, 9, 3, 4]
    else:
        last = _chat_exchange([1], [2])
        if case == "terminal_length":
            last.response.choices[0].finish_reason = "length"
        exchanges = [last]
        expected = [1, 2]
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=exchanges))
    original = value.model_dump_json()
    tokenizer = Tokenizer() if decoder == "absent" else DecoderProperty()
    if case == "unproved_prior":
        with pytest.raises(RuntimeError, match="boundary .* required"):
            value.tokenize(tokenizer=cast(Any, tokenizer))
        assert calls == (["render"] if decoder == "absent" else ["decode"])
    else:
        actual = value.tokenize(tokenizer=cast(Any, tokenizer))
        assert actual.tokens == expected
        assert actual.logprobs[-1] == -expected[-1] / 10
        assert actual.flags[-1] == (
            tr.TokenFlag.SAMPLED
            | tr.TokenFlag.EXACT
            | tr.TokenFlag.OUTPUT
            | tr.TokenFlag.ASSISTANT
        )
        if case == "known_prior":
            assert actual.flags[2] & tr.TokenFlag.STOP
            assert actual.logprobs[1:3] == [-0.2, -0.9]
        assert calls == []
    assert value.model_dump_json() == original
