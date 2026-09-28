from copy import deepcopy
import math
from typing import Any, Literal

import pytest
from test_recorded_fallback_conditioning import _case
from test_tokenize import _character_template_history

import art.trajectories as tr
from art.trajectories import _tokenize as module


@pytest.mark.parametrize("route", ["history", "trajectory", "trace"])
@pytest.mark.parametrize(
    "finish,sampled_stop", [("stop", False), ("stop", True), ("length", False)]
)
def test_declined_boundary_fallback_ends_at_complete_native_record(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    finish: Literal["stop", "length"],
    sampled_stop: bool,
) -> None:
    trajectory, tokenizer = _case("§")
    first, last = trajectory.exchanges.chat_completions
    first.response.choices[0].finish_reason = "stop"
    choice = last.response.choices[0]
    choice.finish_reason = finish
    assert choice.model_extra is not None and choice.logprobs is not None
    assert choice.logprobs.content is not None
    output = tokenizer._encode("b§" if sampled_stop else "b")
    choice.model_extra["token_ids"] = output
    choice.logprobs.content = choice.logprobs.content[: len(output)]
    decode = tokenizer.decode

    def optional_decode(ids, **kwargs):
        if ids == tokenizer._encode("a"):
            raise ValueError("Public optional body decoding unavailable")
        return decode(ids, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", optional_decode)
    before = deepcopy(trajectory.model_dump())
    if route == "history":
        result = trajectory.chat_completions_history().tokenize(tokenizer=tokenizer)
    elif route == "trajectory":
        result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[
            0
        ]
    else:
        value, traces = module._tokenize_trajectory_with_trace(
            trajectory, tokenizer=tokenizer
        )
        result = value.histories[0]
        traces[0].validate(result)
    expected = tokenizer._encode("ua§v") + output
    assert result.tokens == expected
    assert result.logprobs[-len(output) :] == [
        entry.logprob for entry in choice.logprobs.content
    ]
    assert all(flag & tr.TokenFlag.SAMPLED for flag in result.flags[-len(output) :])
    assert bool(result.flags[-1] & tr.TokenFlag.STOP) is sampled_stop
    assert (
        result.flags[2]
        == tr.TokenFlag.EXACT
        | tr.TokenFlag.ASSISTANT
        | tr.TokenFlag.OUTPUT
        | tr.TokenFlag.STOP
    )
    assert math.isnan(result.logprobs[2])
    assert trajectory.model_dump() == before


@pytest.mark.parametrize("change", ["model", "stop", "context", "replacement"])
@pytest.mark.parametrize("outcome", ["changed", "stable", "fatal"])
def test_successful_fallback_rechecks_consumed_evidence(
    monkeypatch: pytest.MonkeyPatch, change: str, outcome: str
) -> None:
    history, tokenizer, expected = _character_template_history()
    decode, render = tokenizer.decode, tokenizer.apply_chat_template
    source = history.message_sources[-1 if change == "stop" else 3]
    assert source is not None
    exchange = source.exchange
    assert isinstance(exchange, module.ChatCompletionsExchange)
    failure = RuntimeError("Public renderer failed")
    fired = False

    def optional_decode(ids, **kwargs):
        if 7002 in ids:
            raise ValueError("Public optional body decoding unavailable")
        return decode(ids, **kwargs)

    def callback(*args: Any, **kwargs: Any):
        nonlocal fired
        value = render(*args, **kwargs)
        if not fired:
            fired = True
            if outcome != "stable":
                if change == "model":
                    exchange.request["model"] = "other/model"
                elif change == "context":
                    history.messages[0]["content"] = "changed"
                elif change == "replacement":
                    history.message_sources[3] = deepcopy(source)
                else:
                    choice = exchange.response.choices[0]
                    assert choice.model_extra is not None
                    choice.model_extra["stop_reason"] = "unmatched"
            if outcome == "fatal":
                raise failure
        return value

    monkeypatch.setattr(tokenizer, "decode", optional_decode)
    monkeypatch.setattr(tokenizer, "apply_chat_template", callback)
    if outcome == "stable":
        result = history.tokenize(tokenizer=tokenizer)
        assert result.tokens == expected
        assert result.model == "test/model"
    else:
        with pytest.raises(
            RuntimeError if outcome == "fatal" else ValueError
        ) as caught:
            history.tokenize(tokenizer=tokenizer)
        if outcome == "fatal":
            assert caught.value is failure
        else:
            assert "Sampled source changed" in str(caught.value)
    assert fired


def test_explicit_rendering_keeps_its_requested_terminal_markup() -> None:
    trajectory, tokenizer = _case("§")
    first, last = trajectory.exchanges.chat_completions
    first.response.choices[0].finish_reason = "stop"
    choice = last.response.choices[0]
    assert choice.model_extra is not None and choice.logprobs is not None
    assert choice.logprobs.content is not None
    choice.model_extra["token_ids"] = tokenizer._encode("b")
    choice.logprobs.content = choice.logprobs.content[:1]
    result = trajectory.tokenize(tokenizer=tokenizer, chat_template="explicit-renderer")
    assert result.tokens == tokenizer._encode("ua§vb§")
