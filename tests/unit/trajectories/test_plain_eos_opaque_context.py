from copy import deepcopy
from typing import Any, cast

import pytest
from test_tokenize import (
    _chat_exchange,
    _completion_exchange,
    _message_exchange,
    _response_exchange,
)

import art.trajectories as tr


@pytest.mark.parametrize("authority", ["none", "plain", "dynamic"])
@pytest.mark.parametrize(
    "protocol",
    ["chat_completions", "messages", "responses", "completions", "token_prompt"],
)
def test_callback_free_native_opaque_context_accepts_plain_eos(
    authority, protocol, monkeypatch
):
    class PlainTokenizer:
        eos_token_id = 2

    class DynamicTokenizer:
        @property
        def eos_token_id(self):
            return 2

    if protocol == "chat_completions":
        exchange = _chat_exchange([1], [2])
        expected_logprob = -0.2
    elif protocol == "messages":
        exchange = _message_exchange(
            cast(
                Any,
                dict(
                    model="test/model",
                    max_tokens=5,
                    messages=[dict(role="user", content="question")],
                ),
            ),
            prompt_token_ids=[1],
            token_ids=[2],
            logprobs=[-0.2],
        )
        expected_logprob = -0.2
    elif protocol == "responses":
        exchange = _response_exchange("public-opaque", 2, prompt_token_ids=[1])
        expected_logprob = -0.1
    else:
        exchange = _completion_exchange(
            prompt=[1] if protocol == "token_prompt" else "question"
        )
        expected_logprob = -0.2
        protocol = "completions"
    opaque = object()
    exchange.request["metadata"] = cast(Any, {"unused": opaque})
    # Independent model histories exercise final authority propagation too.
    second = deepcopy(exchange)
    second.request["model"] = "other/model"
    second.request["metadata"] = cast(Any, {"unused": opaque})
    value = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(**{protocol: [exchange, second]})
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Complete native records must not load or render")

    monkeypatch.setattr("art.trajectories._tokenize._load_tokenizer", unexpected)
    tokenizer = (
        None
        if authority == "none"
        else PlainTokenizer()
        if authority == "plain"
        else DynamicTokenizer()
    )
    if authority == "dynamic":
        with pytest.raises(ValueError, match="[Cc]ontext"):
            value.tokenize(tokenizer=cast(Any, tokenizer), multi_history=True)
        return
    actual = value.tokenize(tokenizer=cast(Any, tokenizer), multi_history=True)
    assert len(actual.histories) == 2
    for result in actual.histories:
        assert result.tokens == [1, 2]
        assert result.logprobs[-1] == expected_logprob
        assert result.flags[-1] & tr.TokenFlag.SAMPLED
        assert bool(result.flags[-1] & tr.TokenFlag.STOP) == (authority == "plain")
    assert cast(Any, exchange.request["metadata"])["unused"] is opaque
    assert cast(Any, second.request["metadata"])["unused"] is opaque
