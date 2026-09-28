from typing import Any, cast

import pytest
from test_tokenize import _response_exchange

import art.trajectories as tr


@pytest.mark.parametrize("mutate", [False, True])
def test_responses_renderer_cannot_change_a_reused_projection(mutate):
    exchange = _response_exchange("public-reused", 2)
    assert exchange.response.model_extra is not None
    exchange.response.model_extra.pop("token_generations")
    original = exchange.model_dump_json()
    observed = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            completed = any(message["role"] == "assistant" for message in messages)
            observed.append((completed, messages[0]["content"]))
            if not completed and mutate:
                messages[0]["content"] = "changed detached projection"
            return [99, 2] if completed else [99]

    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
    if mutate:
        with pytest.raises(
            ValueError, match="[Cc]ontext|[Pp]rojection|Consumed source text"
        ):
            actual = value.tokenize(tokenizer=cast(Any, Tokenizer()))
            assert actual.tokens == [99, 2]
            assert observed == [
                (False, "turn 0"),
                (True, "changed detached projection"),
            ]
            assert exchange.model_dump_json() == original
    else:
        actual = value.tokenize(tokenizer=cast(Any, Tokenizer()))
        assert actual.tokens == [99, 2]
        assert observed == [(False, "turn 0"), (True, "turn 0")]
        assert exchange.model_dump_json() == original
