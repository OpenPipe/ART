"""Public explicit-rendered Responses item-ID consumption regression."""

from __future__ import annotations

import math
from typing import Any

from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputText
import pytest
from test_tokenize import _multi_output_responses_chat_history

import art.trajectories as tr


@pytest.mark.parametrize("route", ["history", "trajectory"])
@pytest.mark.parametrize("mutate", [False, True])
def test_multi_output_generation_keeps_consumed_item_token_id(route, mutate):
    projected = _multi_output_responses_chat_history()
    source = next(
        source
        for source in projected.message_sources
        if source is not None and source.output_indices == (0,)
    )
    exchange = source.exchange
    assert isinstance(exchange, tr.ResponsesExchange)
    data = exchange.response.model_dump(mode="python")
    # Retain one exact generation spanning both outputs: neither individual
    # rendered message owns all of its native output evidence.
    assert data["token_generations"][0]["output_indices"] == [0, 1]
    for output, text, lp in zip(
        data["output"], ["first", "second"], [-0.1, -0.2], strict=True
    ):
        output["content"][0]["logprobs"] = [
            {
                "token": text,
                "logprob": lp,
                "bytes": list(text.encode()),
                "top_logprobs": [],
            }
        ]
    data["output"][0]["content"][0]["logprobs"][0]["token_id"] = 20
    exchange.response = Response.model_validate(data)
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
    item = exchange.response.output[0]
    assert isinstance(item, ResponseOutputMessage)
    content = item.content[0]
    assert isinstance(content, ResponseOutputText) and content.logprobs
    first_lp = content.logprobs[0]
    assert first_lp.model_extra is not None
    extras = first_lp.model_extra
    calls = []

    class Tokenizer:
        def __call__(self, text: str, **kwargs: Any):
            token = {"turn 0": 1, "first": 20, "second": 30}[text]
            if kwargs.get("return_offsets_mapping"):
                calls.append(text)
                if mutate and text == "second":
                    assert first_lp.model_extra is not None
                    extras["token_id"] = 99
                return {"input_ids": [token], "offset_mapping": [(0, len(text))]}
            return [token]

        def apply_chat_template(self, messages, **kwargs):
            return [token for message in messages for token in self(message["content"])]

    def tokenize():
        if route == "history":
            return value.responses_history().tokenize(
                tokenizer=Tokenizer(), chat_template="custom"
            )
        return value.tokenize(tokenizer=Tokenizer(), chat_template="custom")

    if mutate:
        with pytest.raises(
            ValueError,
            match="[Ss]ampled source changed|[Cc]onsumed|[Cc]ontext changed|Rendered source evidence changed",
        ):
            actual = tokenize()
            # If the guard misses the edit, prove this is the consumed old
            # value being returned, not a benign pre-consumption refresh.
            assert extras["token_id"] == 99
            assert first_lp.logprob == -0.1
            assert actual.tokens == [1, 20, 30]
            assert math.isnan(actual.logprobs[0])
            assert actual.logprobs[1:] == [-0.1, -0.2]
            assert actual.flags == [
                tr.TokenFlag(0),
                tr.TokenFlag.ASSISTANT
                | tr.TokenFlag.OUTPUT
                | tr.TokenFlag.EXACT
                | tr.TokenFlag.SAMPLED,
                tr.TokenFlag.ASSISTANT | tr.TokenFlag.OUTPUT,
            ]
        assert extras["token_id"] == 99
        assert first_lp.logprob == -0.1
    else:
        actual = tokenize()
        assert actual.tokens == [1, 20, 30]
        assert math.isnan(actual.logprobs[0])
        assert actual.logprobs[1:] == [-0.1, -0.2]
        assert actual.flags == [
            tr.TokenFlag(0),
            tr.TokenFlag.ASSISTANT
            | tr.TokenFlag.OUTPUT
            | tr.TokenFlag.EXACT
            | tr.TokenFlag.SAMPLED,
            tr.TokenFlag.ASSISTANT | tr.TokenFlag.OUTPUT,
        ]
        assert first_lp.logprob == -0.1
        assert extras["token_id"] == 20
    assert "second" in calls
