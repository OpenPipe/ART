from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize as module


@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("internal", [False, True])
def test_completed_evidence_remains_guarded_after_tokenizer_loses_callbacks(
    mutate, internal
):
    first = _chat_exchange([1], [2, 9], model="public/first")
    second = _chat_exchange([3], [4, 9], model="public/second", offset=1)
    assert first.response.choices[0].model_extra is not None
    assert second.response.choices[0].model_extra is not None
    first.response.choices[0].model_extra["stop_reason"] = 9
    second.response.choices[0].model_extra["stop_reason"] = "§"
    calls = []
    lp = first.response.choices[0].logprobs
    assert lp is not None and lp.content
    first_lp = lp.content

    class PlainEOS:
        eos_token_id = 9

    class Tokenizer(PlainEOS):
        def __call__(self, text, **kwargs):
            assert text == "§"
            calls.append(text)
            if mutate:
                first_lp[0].logprob = -99
            self.__class__ = PlainEOS
            return [9]

    tokenizer = Tokenizer()
    value = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second])
    )

    def tokenize():
        if internal:
            return module._tokenize_trajectory_with_trace(
                value, tokenizer=cast(Any, tokenizer)
            )[0]
        return value.tokenize(tokenizer=cast(Any, tokenizer), multi_history=True)

    if mutate:
        with pytest.raises(ValueError, match="Sampled source changed"):
            actual = tokenize()
            assert actual.histories[0].logprobs[-2] == -0.2
            assert first_lp[0].logprob == -99
    else:
        actual = tokenize()
        assert [h.tokens for h in actual.histories] == [[1, 2, 9], [3, 4, 9]]
        assert [h.logprobs[-2:] for h in actual.histories] == [
            [-0.2, -0.9],
            [-0.4, -0.9],
        ]
        assert all(h.flags[-1] & tr.TokenFlag.STOP for h in actual.histories)
    assert calls == ["§"]
    assert type(tokenizer) is PlainEOS
