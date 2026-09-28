from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
import pytest
from test_tokenize import _CharacterTemplateTokenizer, _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize as module


def _case(gap: str):
    tokenizer = _CharacterTemplateTokenizer()
    encode = tokenizer._encode
    first = _chat_exchange(encode("u"), encode("a"))
    first.request["messages"] = [{"role": "user", "content": "u"}]
    first.response.choices[0].message.content = "a"
    first.response.choices[0].finish_reason = "length"
    second = _chat_exchange(encode("ua" + gap + "v"), encode("b§"), offset=1)
    second.request["messages"] = cast(
        list[ChatCompletionMessageParam],
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "v"},
        ],
    )
    second.response.choices[0].message.content = "b"
    second.response.choices[0].finish_reason = "stop"
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second])
    )
    return trajectory, tokenizer


@pytest.mark.parametrize("route", ["history", "trajectory", "trace"])
@pytest.mark.parametrize("gap", ["", "different", "§"])
def test_declined_boundary_proof_cannot_change_sampled_conditioning(route, gap):
    trajectory, tokenizer = _case(gap)
    before = deepcopy(trajectory.model_dump())
    options: dict[str, Any] = {"tokenizer": tokenizer}
    try:
        if route == "history":
            result = trajectory.chat_completions_history().tokenize(**options)
        elif route == "trajectory":
            result = trajectory.tokenize(multi_history=True, **options).histories[0]
        else:
            result = module._tokenize_trajectory_with_trace(trajectory, **options)[
                0
            ].histories[0]
    except ValueError as error:
        assert gap != "§"
        assert "original native conditioning" in str(error)
    else:
        for source in trajectory.chat_completions_history().message_sources:
            if source is None or not module._source_is_sampled(source):
                continue
            prompt, output, logprobs = module._source_native_record(source)
            assert prompt is not None and output is not None
            start, end = len(prompt), len(prompt) + len(output)
            assert result.tokens[:end] == prompt + output
            assert result.logprobs[start:end] == logprobs
            assert all(flag & tr.TokenFlag.SAMPLED for flag in result.flags[start:end])
    assert trajectory.model_dump() == before


@pytest.mark.parametrize("route", ["history", "trajectory", "trace"])
def test_matching_explicit_renderer_remains_supported(route):
    trajectory, tokenizer = _case("§")
    options: dict[str, Any] = {
        "tokenizer": tokenizer,
        "chat_template": "explicit-public-template",
    }
    if route == "history":
        result = trajectory.chat_completions_history().tokenize(**options)
    elif route == "trajectory":
        result = trajectory.tokenize(multi_history=True, **options).histories[0]
    else:
        result = module._tokenize_trajectory_with_trace(trajectory, **options)[
            0
        ].histories[0]
    assert result.tokens == tokenizer._encode("ua§vb§")


@pytest.mark.parametrize("route", ["history", "trajectory", "trace"])
@pytest.mark.parametrize("gap", ["", "different"])
def test_complete_native_gap_is_retained_without_inventing_roles(route, gap):
    trajectory, tokenizer = _case(gap)
    before = deepcopy(trajectory.model_dump())
    if route == "history":
        history = trajectory.chat_completions_history().tokenize(tokenizer=tokenizer)
    elif route == "trajectory":
        history = trajectory.tokenize(
            tokenizer=tokenizer, multi_history=True
        ).histories[0]
    else:
        value, traces = module._tokenize_trajectory_with_trace(
            trajectory, tokenizer=tokenizer
        )
        history = value.histories[0]
        traces[0].validate(history)
    assert history.tokens == tokenizer._encode("ua" + gap + "vb§")
    for source in trajectory.chat_completions_history().message_sources:
        if source is None or not module._source_is_sampled(source):
            continue
        prompt, output, lp = module._source_native_record(source)
        assert prompt is not None and output is not None
        start, end = len(prompt), len(prompt) + len(output)
        assert history.tokens[:end] == prompt + output
        assert history.logprobs[start:end] == lp
        assert all(flag & tr.TokenFlag.SAMPLED for flag in history.flags[start:end])
    end_gap = 3 + len(gap)
    assert history.flags[2:end_gap] == [tr.TokenFlag.EXACT] * (end_gap - 2)
    assert all(math.isnan(lp) for lp in history.logprobs[2:end_gap])
    for flag in (
        tr.TokenFlag.SAMPLED,
        tr.TokenFlag.ASSISTANT,
        tr.TokenFlag.OUTPUT,
        tr.TokenFlag.STOP,
    ):
        mask = tr.first_occurrence_masks([history], where=flag)[0]
        assert not any(mask[2:end_gap])
    assert trajectory.model_dump() == before


def test_conflicting_native_prefixes_remain_separate_owned_histories():
    trajectory, tokenizer = _case("")
    second = trajectory.exchanges.chat_completions[1]
    assert second.response.choices[0].model_extra is not None
    second.response.choices[0].model_extra["prompt_token_ids"][0] += 1
    before = deepcopy(trajectory.model_dump())
    value, traces = module._tokenize_trajectory_with_trace(
        trajectory, tokenizer=tokenizer
    )
    seen = set()
    for history, trace in zip(value.histories, traces, strict=True):
        trace.validate(history)
        for key, source in trace.sources.items():
            offsets = [i for i, owner in enumerate(trace.source_keys) if owner == key]
            if not offsets:
                continue
            prompt, output, lp = module._source_native_record(source)
            assert prompt is not None and output is not None
            assert offsets == list(range(len(prompt), len(prompt) + len(output)))
            assert history.tokens[: len(prompt) + len(output)] == prompt + output
            assert history.logprobs[len(prompt) : len(prompt) + len(output)] == lp
            seen.add(id(module._source_exchange(source)))
    assert seen == {id(exchange) for exchange in trajectory.exchanges.chat_completions}
    assert trajectory.model_dump() == before
