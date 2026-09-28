from typing import Any, cast

import pytest
from test_tokenize import _message_exchange, _response_with_content_logprobs

import art.trajectories as tr


@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("mutate_copy", [False, True])
def test_template_selection_cannot_change_original_schema_then_restore(
    mutate, mutate_copy
):
    exchange = _message_exchange(
        cast(
            Any,
            {
                "model": "test/model",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "question"}],
                "tools": [{"name": "ask", "input_schema": {"type": "string"}}],
            },
        ),
        token_ids=[2],
        logprobs=[-0.2],
    )
    events = []

    class Tokenizer:
        chat_template = {"named": "unchanged body"}

        def get_chat_template(self, chat_template=None, tools: Any = None):
            schema = tools[0]["function"]["parameters"]
            assert schema is cast(Any, exchange.request)["tools"][0]["input_schema"]
            events.append("select")
            if mutate:
                schema["type"] = "integer"
            return "unchanged body"

        def apply_chat_template(self, messages, tools: Any = None, **kwargs):
            events.append("render")
            schema = tools[0]["function"]["parameters"]
            changed = schema["type"] == "integer"
            schema["type"] = "string"
            if mutate_copy:
                assert messages is not exchange.request["messages"]
                messages[0]["content"] = "disposable change"
            return [99 if changed else 1]

    trajectory = tr.Trajectory(exchanges=tr.TrajectoryExchanges(messages=[exchange]))
    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            trajectory.tokenize(tokenizer=cast(Any, Tokenizer()))
        assert events == ["select"]
    else:
        result = trajectory.tokenize(tokenizer=cast(Any, Tokenizer()))
        assert result.tokens == [1, 2]
        assert result.logprobs[-1] == -0.2
        assert result.flags[-1] & tr.TokenFlag.SAMPLED
        assert exchange.request["messages"][0]["content"] == "question"
        assert events == ["select", "render"]


@pytest.mark.parametrize("mutate", [False, True])
def test_responses_template_selection_validates_original_source(mutate):
    exchange = _response_with_content_logprobs(exact_second=True)
    original = exchange.request["input"]
    events = []

    class Tokenizer:
        chat_template = {"named": "unchanged body"}

        def get_chat_template(self, chat_template=None, tools=None):
            events.append("select")
            if mutate:
                exchange.request["input"] = "changed original"
            return "unchanged body"

        def apply_chat_template(self, messages, tools=None, **kwargs):
            events.append("render")
            changed = exchange.request["input"] != original
            exchange.request["input"] = original
            return [99 if changed else 1]

    trajectory = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
    if mutate:
        with pytest.raises(ValueError, match="[Cc]ontext changed"):
            trajectory.tokenize(tokenizer=cast(Any, Tokenizer()))
        assert events == ["select"]
    else:
        result = trajectory.tokenize(tokenizer=cast(Any, Tokenizer()))
        assert result.tokens == [1, 11, 12]
        assert result.logprobs[1:] == [-0.1, -0.2]
        assert all(flag & tr.TokenFlag.SAMPLED for flag in result.flags[1:])
        assert exchange.request["input"] == original
        assert events == ["select", "render"]
