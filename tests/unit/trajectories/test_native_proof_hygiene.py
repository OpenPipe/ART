import math
from typing import Any

import pytest
from test_recorded_fallback_conditioning import _case
from test_tokenize import _chat_exchange, _StopTokenizer

import art.trajectories as tr
from art.trajectories import _tokenize as module


@pytest.mark.parametrize("route", ["history", "trajectory", "trace"])
@pytest.mark.parametrize("outcome", ["unchanged", "changed", "fatal"])
def test_stop_callback_does_not_leave_later_native_logprobs_cached(route, outcome):
    first = _chat_exchange([1], [2, 8, 9])
    assert first.response.choices[0].model_extra is not None
    first.response.choices[0].model_extra["stop_reason"] = "END"
    second = _chat_exchange([1, 2, 8, 9, 3], [4, 9], offset=1)
    second_lp = second.response.choices[0].logprobs
    assert second_lp is not None and second_lp.content is not None
    target_lp = second_lp.content[0]
    third = _chat_exchange([1, 2, 8, 9, 3, 4, 9, 5], [6, 9], offset=2)
    third.response.choices[0].finish_reason = "length"
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second, third])
    )
    failure = RuntimeError("exact STOP callback failure")

    class Tokenizer(_StopTokenizer):
        calls = 0

        def __call__(self, text: str, **kwargs: object) -> list[int]:
            if text == "END":
                self.calls += 1
                if self.calls == 2:
                    if outcome == "fatal":
                        raise failure
                    if outcome == "changed":
                        target_lp.logprob = -8.0
            return super().__call__(text, **kwargs)

    tokenizer = Tokenizer()

    def tokenize():
        if route == "history":
            return trajectory.chat_completions_history().tokenize(tokenizer=tokenizer)
        if route == "trace":
            value, traces = module._tokenize_trajectory_with_trace(
                trajectory, tokenizer=tokenizer
            )
            traces[0].validate(value.histories[0])
            return value.histories[0]
        return trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[0]

    if outcome == "fatal":
        with pytest.raises(RuntimeError) as caught:
            tokenize()
        assert caught.value is failure
    else:
        value = tokenize()
        assert value.tokens == [1, 2, 8, 9, 3, 4, 9, 5, 6, 9]
        assert value.logprobs[5] == second_lp.content[0].logprob
        assert value.logprobs[5] == (-8.0 if outcome == "changed" else -0.4)
        assert value.flags[5] & tr.TokenFlag.EXACT
        assert value.flags[5] & tr.TokenFlag.SAMPLED
    assert tokenizer.calls >= 2


@pytest.mark.parametrize("explicit", [False, True])
def test_explicit_render_does_not_repeat_recorded_projection_work(
    monkeypatch, explicit
):
    trajectory, tokenizer = _case("§")
    original = module._history_render_state
    calls = []

    def observed(history):
        calls.append(history)
        return original(history)

    monkeypatch.setattr(module, "_history_render_state", observed)
    options: dict[str, Any] = {"chat_template": "explicit"} if explicit else {}
    value = trajectory.chat_completions_history().tokenize(
        tokenizer=tokenizer, **options
    )
    assert value.tokens == tokenizer._encode("ua§vb§")
    sampled = [i for i, flag in enumerate(value.flags) if flag & tr.TokenFlag.SAMPLED]
    assert sampled == [1, 4, 5]
    assert all(math.isfinite(value.logprobs[i]) for i in sampled)
    assert len(calls) == (1 if explicit else 2)
