from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, cast

import pytest
from test_recorded_fallback_conditioning import _case
from test_tokenize import _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize as module


def _reasoning_case(gap: str):
    trajectory, tokenizer = _case(gap)
    original = trajectory.exchanges.chat_completions[0]
    replacement = _chat_exchange(tokenizer._encode("u"), tokenizer._encode("ra"))
    replacement.request["messages"] = original.request["messages"]
    choice = replacement.response.choices[0]
    choice.message.content = "a"
    cast(Any, choice.message).reasoning_content = "r"
    choice.finish_reason = "length"
    trajectory.exchanges.chat_completions[0] = replacement
    return trajectory, tokenizer


@pytest.mark.parametrize("gap", ["", "different", "§"])
@pytest.mark.parametrize("route", ["trajectory", "trace"])
def test_reasoning_stripped_copy_recovers_complete_native_gap(gap, route):
    trajectory, tokenizer = _reasoning_case(gap)
    before = deepcopy(trajectory.model_dump())
    if route == "trace":
        result, traces = module._tokenize_trajectory_with_trace(
            trajectory, tokenizer=tokenizer
        )
        for history, trace in zip(result.histories, traces, strict=True):
            trace.validate(history)
    else:
        result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
        traces = None
    assert result.trajectory is trajectory
    assert len(result.histories) == 2
    original, copied = result.histories
    assert original.tokens == tokenizer._encode("ura")
    assert copied.tokens == tokenizer._encode("ua" + gap + "vb§")
    first, second = trajectory.exchanges.chat_completions
    first_lp = module._chat_tokens(first.response)[2]
    second_lp = module._chat_tokens(second.response)[2]
    assert original.logprobs[1:] == first_lp
    assert all(flag & tr.TokenFlag.SAMPLED for flag in original.flags[1:])
    assert (
        copied.flags[1]
        == tr.TokenFlag.EXACT | tr.TokenFlag.ASSISTANT | tr.TokenFlag.OUTPUT
    )
    assert math.isnan(copied.logprobs[1])
    final_start = len(tokenizer._encode("ua" + gap + "v"))
    assert copied.logprobs[final_start:] == second_lp
    assert all(flag & tr.TokenFlag.SAMPLED for flag in copied.flags[final_start:])
    if gap != "§":
        assert copied.flags[2:final_start] == [tr.TokenFlag.EXACT] * (final_start - 2)
        assert all(math.isnan(value) for value in copied.logprobs[2:final_start])
        for flag in [
            tr.TokenFlag.ASSISTANT,
            tr.TokenFlag.OUTPUT,
            tr.TokenFlag.STOP,
            tr.TokenFlag.SAMPLED,
        ]:
            assert not any(
                tr.first_occurrence_masks([copied], where=flag)[0][2:final_start]
            )
    assert (
        tr.first_occurrence_masks(result.histories, where=tr.TokenFlag.SAMPLED)[1][1]
        is False
    )
    if traces is not None:
        assert traces[1].source_keys[1] is None
        assert any(
            module._source_exchange(source) is first
            for source in traces[0].sources.values()
        )
        assert any(
            module._source_exchange(source) is second
            for source in traces[1].sources.values()
        )
    assert trajectory.model_dump() == before


@pytest.mark.parametrize("gap", ["", "different", "§"])
def test_copied_history_without_original_owner_still_fails(gap):
    trajectory, tokenizer = _reasoning_case(gap)
    history = trajectory.chat_completions_histories()[-1]
    with pytest.raises(ValueError, match="complete original sampled occurrence"):
        history.tokenize(tokenizer=tokenizer)


@pytest.mark.parametrize("field", ["tools", "chat_template", "chat_template_kwargs"])
def test_reasoning_projection_does_not_validate_changed_context(field):
    trajectory, _ = _reasoning_case("different")
    history = trajectory.chat_completions_histories()[-1]
    setattr(
        history,
        field,
        {
            "chat_template": "changed",
            "chat_template_kwargs": {"changed": True},
            "tools": [{"type": "function", "function": {"name": "changed"}}],
        }[field],
    )
    state = module._history_render_state(history)
    assert state.projection_matches is not True
    assert state.context_changed is True


def test_changed_sampled_model_still_fails_before_recovery():
    trajectory, tokenizer = _reasoning_case("different")
    history = trajectory.chat_completions_histories()[-1]
    history.message_sources[-1].exchange.request["model"] = "different/model"
    with pytest.raises(ValueError, match="model no longer matches"):
        history.tokenize(tokenizer=tokenizer)
