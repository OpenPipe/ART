from copy import deepcopy

import pytest
from test_recorded_fallback_conditioning import _case

import art.trajectories as tr
from art.trajectories import _tokenize as module


@pytest.mark.parametrize("route", ["history", "trajectory", "trace"])
@pytest.mark.parametrize("mutation", ["stable", "all", "first", "last", "content"])
def test_fallback_preserves_required_native_sources(monkeypatch, route, mutation):
    trajectory, tokenizer = _case("§")
    trajectory.exchanges.chat_completions[0].response.choices[0].finish_reason = "stop"
    before = deepcopy(trajectory.model_dump())
    decode, render = tokenizer.decode, tokenizer.apply_chat_template
    calls = 0

    def optional_decode(ids, **kwargs):
        if ids == tokenizer._encode("a"):
            raise ValueError("Optional body decoding unavailable")
        return decode(ids, **kwargs)

    def callback(messages, **kwargs):
        nonlocal calls
        calls += 1
        for message in messages:
            if message.get("role") != "assistant":
                continue
            if (
                mutation == "all"
                or (mutation == "first" and message.get("content") == "a")
                or (mutation == "last" and message.get("content") == "b")
            ):
                message["role"] = "user"
            elif mutation == "content":
                message["content"] = ""
        return render(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", optional_decode)
    monkeypatch.setattr(tokenizer, "apply_chat_template", callback)
    original = trajectory.chat_completions_history()
    trace = None
    if route == "history":
        result = original.tokenize(tokenizer=tokenizer)
    elif route == "trajectory":
        result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[
            0
        ]
    else:
        value, traces = module._tokenize_trajectory_with_trace(
            trajectory, tokenizer=tokenizer
        )
        result, trace = value.histories[0], traces[0]
        trace.validate(result)
    assert calls > 0
    assert result.tokens == tokenizer._encode("ua§vb§")
    expected_positions = set()
    for message, source in zip(
        original.messages, original.message_sources, strict=True
    ):
        if message.get("role") != "assistant" or source is None:
            continue
        prompt, output, logprobs = module._source_native_record(source)
        assert prompt is not None and output is not None
        start, end = len(prompt), len(prompt) + len(output)
        expected_positions.update(range(start, end))
        assert result.tokens[:end] == prompt + output
        assert result.logprobs[start:end] == logprobs
        if trace is not None:
            key = module._sampled_source_key(source)
            assert trace.source_keys[start:end] == [key] * len(output)
            assert module._source_exchange(trace.sources[key]) is source.exchange
    assert {
        i for i, flag in enumerate(result.flags) if flag & tr.TokenFlag.SAMPLED
    } == expected_positions
    assert tr.first_occurrence_masks([result], where=tr.TokenFlag.SAMPLED)[0] == [
        index in expected_positions for index in range(len(result.tokens))
    ]
    assert result.flags[-1] & tr.TokenFlag.STOP
    assert trajectory.model_dump() == before


def test_required_source_coverage_preserves_fatal_renderer_identity(monkeypatch):
    trajectory, tokenizer = _case("§")
    trajectory.exchanges.chat_completions[0].response.choices[0].finish_reason = "stop"
    decode = tokenizer.decode
    failure = RuntimeError("Renderer failed after changing its own message")
    before = deepcopy(trajectory.model_dump())

    def optional_decode(ids, **kwargs):
        if ids == tokenizer._encode("a"):
            raise ValueError("Optional body decoding unavailable")
        return decode(ids, **kwargs)

    def callback(messages, **kwargs):
        for message in messages:
            if message.get("role") == "assistant":
                message["role"] = "user"
        raise failure

    monkeypatch.setattr(tokenizer, "decode", optional_decode)
    monkeypatch.setattr(tokenizer, "apply_chat_template", callback)
    with pytest.raises(RuntimeError) as caught:
        trajectory.tokenize(tokenizer=tokenizer)
    assert caught.value is failure
    assert trajectory.model_dump() == before
