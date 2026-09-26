from __future__ import annotations

from typing import Any, cast

from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText
import pytest
from test_tokenize import (
    _completion_exchange,
    _multi_output_responses_chat_history,
    _response_exchange,
)

import art.trajectories as tr
import art.trajectories._tokenize as core


@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("target", ["logprob", "text"])
def test_responses_prefix_repair_retains_consumed_current_logprobs(
    monkeypatch, mutate, target
):
    monkeypatch.setattr(core, "_WARNED_PREFIX_RETOKENIZATION", False)
    exchange = _response_exchange("repair-consumption", 2, prompt_token_ids=[1])
    data = exchange.response.model_dump(mode="python")
    data["output"].append(
        {
            "id": "second",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "last",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    )
    data["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2}],
            "output_indices": [0],
        },
        {
            "prompt_token_ids": [1, 3, 4],
            "output_tokens": [{"token_id": 5, "logprob": -0.5}],
            "output_indices": [1],
        },
    ]
    exchange.response = Response.model_validate(data)
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
    history = value.responses_history(reconcile_text_equivalent_tokenizations=True)
    before = value.model_dump_json()
    calls = []

    class Tokenizer:
        def __call__(self, text, **kwargs):
            calls.append(text)
            if mutate and text == "answer":
                if target == "text":
                    output = exchange.response.output[0]
                    assert isinstance(output, ResponseOutputMessage)
                    content = output.content[0]
                    assert isinstance(content, ResponseOutputText)
                    content.text = "edited"
                else:
                    assert exchange.response.model_extra is not None
                    exchange.response.model_extra["token_generations"][1][
                        "output_tokens"
                    ][0]["logprob"] = -9.0
            return {"answer": [3], "last": [5]}[text]

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("native repair must not render")

    if mutate:
        with pytest.raises(
            ValueError, match="[Ss]ampled source changed|Consumed source text changed"
        ):
            history.tokenize(tokenizer=cast(Any, Tokenizer()))
    else:
        with pytest.warns(UserWarning, match="preserved the original sampled"):
            result = history.tokenize(tokenizer=cast(Any, Tokenizer()))
        assert result.tokens == [1, 2, 4, 5]
        assert result.logprobs[1] == -0.2 and result.logprobs[-1] == -0.5
        assert result.flags[-1] & tr.TokenFlag.SAMPLED
        assert value.model_dump_json() == before
    assert "answer" in calls


@pytest.mark.parametrize("mutate", [False, True])
def test_completions_rendered_only_logprobs_bind_before_encoding(monkeypatch, mutate):
    exchange = _completion_exchange(prompt="q")
    choice = exchange.response.choices[0]
    assert choice.model_extra is not None and choice.logprobs is not None
    choice.model_extra.pop("prompt_token_ids")
    choice.model_extra.pop("token_ids")
    choice.text = "a"
    choice.logprobs.tokens = ["a"]
    choice.logprobs.token_logprobs = [-0.2]
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(completions=[exchange]))
    before = value.model_dump_json()
    calls = []
    armed = False
    original = core._completion_visible_logprobs

    def observe(*args, **kwargs):
        nonlocal armed
        armed = True
        try:
            return original(*args, **kwargs)
        finally:
            armed = False

    monkeypatch.setattr(core, "_completion_visible_logprobs", observe)

    class Tokenizer:
        def __call__(self, text, **kwargs):
            if armed:
                calls.append(text)
                if mutate:
                    assert (
                        choice.logprobs is not None and choice.logprobs.token_logprobs
                    )
                    choice.logprobs.token_logprobs[0] = -9.0
            return {"q": [1], "a": [2]}[text]

    if mutate:
        with pytest.raises(ValueError, match="[Ss]ampled source changed"):
            value.tokenize(tokenizer=cast(Any, Tokenizer()))
    else:
        result = value.tokenize(tokenizer=cast(Any, Tokenizer()))
        assert result.tokens == [1, 2] and result.logprobs[-1] == -0.2
        assert result.flags[-1] & tr.TokenFlag.OUTPUT
        assert not result.flags[-1] & tr.TokenFlag.SAMPLED
        assert value.model_dump_json() == before
    assert calls == ["a"]


def test_new_ledger_callback_preserves_raised_object_after_mutation():
    exchange = _completion_exchange()
    key = core._exchange_sampled_source_key(exchange)
    builder = core._TraceBuilder()
    builder.consume_sources({key: exchange})
    error = RuntimeError("public callback failure")

    def callback():
        choice = exchange.response.choices[0]
        assert choice.logprobs is not None and choice.logprobs.token_logprobs
        choice.logprobs.token_logprobs[0] = -9.0
        raise error

    with pytest.raises(RuntimeError) as caught:
        builder.checked(callback)
    assert caught.value is error


@pytest.mark.parametrize("protocol", ["responses", "completions"])
def test_native_standalone_protocol_needs_no_callback_validator(monkeypatch, protocol):
    def unexpected(*args, **kwargs):
        raise AssertionError(
            "callback-free standalone native must not build callback guards"
        )

    monkeypatch.setattr(core, "_sampled_source_validator", unexpected)
    if protocol == "responses":
        exchange = _response_exchange("offline", 2, prompt_token_ids=[1])
        value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
        expected_lp = -0.1
    else:
        exchange = _completion_exchange()
        value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(completions=[exchange]))
        expected_lp = -0.2
    actual = value.tokenize()
    assert actual.tokens == [1, 2] and actual.logprobs[-1] == expected_lp
    assert actual.flags[-1] & tr.TokenFlag.SAMPLED


def test_declined_native_responses_keeps_original_consumed_key():
    projected = _multi_output_responses_chat_history()
    source = next(
        source
        for source in projected.message_sources
        if source is not None and source.generation_index == 0
    )
    exchange = source.exchange
    assert isinstance(exchange, tr.ResponsesExchange)
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
    history = value.responses_history()
    # One retained item cannot stand for the complete two-item native generation.
    history.input = history.input[:-1]
    history.input_sources = history.input_sources[:-1]
    builder = core._TraceBuilder(track_sources=False)
    assert (
        core._tokenize_exact_responses_history(
            history, base_model=None, tokenizer=None, _trace=builder
        )
        is None
    )
    assert builder.validate_sources is not None
    builder.validate_sources(None, require_supported_context=False)
    assert exchange.response.model_extra is not None
    exchange.response.model_extra["token_generations"][0]["output_tokens"][0][
        "logprob"
    ] = -9.0
    with pytest.raises(ValueError, match="[Ss]ampled source changed"):
        builder.validate_sources(None, require_supported_context=False)
