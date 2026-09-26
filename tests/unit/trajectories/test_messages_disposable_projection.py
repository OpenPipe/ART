from typing import Any, cast

import pytest
from test_tokenize import _message_exchange

import art.trajectories as tr


@pytest.mark.parametrize("mutate_copy", [False, True])
def test_disposable_messages_renderer_is_prompt_authority(mutate_copy):
    exchange = _message_exchange(
        cast(
            Any,
            {
                "model": "test/model",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "question"}],
            },
        ),
        token_ids=[2],
        logprobs=[-0.2],
    )
    original = exchange.model_dump_json()
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "question"}]
            assert messages is not exchange.request["messages"]
            assert messages[0] is not exchange.request["messages"][0]
            if mutate_copy:
                messages[0]["content"] = "changed"
            calls.append(True)
            return [99]

    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(messages=[exchange]))
    actual = value.tokenize(tokenizer=cast(Any, Tokenizer()))
    assert calls == [True]
    assert actual.tokens == [99, 2]
    assert actual.logprobs[-1] == -0.2
    assert actual.flags[-1] & tr.TokenFlag.SAMPLED
    assert exchange.model_dump_json() == original
