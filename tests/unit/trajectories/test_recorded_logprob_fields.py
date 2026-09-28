from typing import Any, cast

from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr


@pytest.mark.parametrize("kind", ["standard", "pure", "mutating", "raising"])
@pytest.mark.parametrize("metadata", ["sentinel", "extra_id", "positional"])
@pytest.mark.parametrize("refusal", [False, True])
def test_native_logprobs_read_recorded_fields(
    kind: str, metadata: str, refusal: bool
) -> None:
    exchange = _chat_exchange([1], [2])
    choice = exchange.response.choices[0]
    assert choice.logprobs is not None and choice.logprobs.content
    calls: list[float] = []
    armed = False

    class Entry(ChatCompletionTokenLogprob):
        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            result = super().model_dump(*args, **kwargs)
            if armed:
                calls.append(self.logprob)
                if kind == "raising":
                    raise RuntimeError("Numeric evidence must not call serialization")
                if kind == "mutating" and len(calls) == 2:
                    self.logprob = -9.0
            return result

    row = choice.logprobs.content[0]
    if kind != "standard":
        row = Entry.model_validate(row.model_dump())
    if metadata != "sentinel":
        row.token = "visible text"
    if metadata == "extra_id":
        assert row.model_extra is not None
        row.model_extra["token_id"] = 2
    choice.logprobs.content = [] if refusal else [row]
    choice.logprobs.refusal = [row] if refusal else None
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    )
    history = trajectory.chat_completions_history()
    before = trajectory.model_dump_json()
    armed = True
    result = history.tokenize()
    assert calls == []
    assert result.tokens == [1, 2]
    assert result.logprobs[-1] == row.logprob == -0.2
    assert result.flags == [
        tr.TokenFlag.EXACT,
        tr.TokenFlag.EXACT
        | tr.TokenFlag.ASSISTANT
        | tr.TokenFlag.OUTPUT
        | tr.TokenFlag.SAMPLED,
    ]
    assert trajectory.model_dump_json() == before


@pytest.mark.parametrize("token_id", [True, -1, "invalid", 3])
def test_native_typed_extra_token_id_keeps_validation(token_id: object) -> None:
    exchange = _chat_exchange([1], [2])
    choice = exchange.response.choices[0]
    assert choice.logprobs is not None and choice.logprobs.content
    row = choice.logprobs.content[0]
    cast(dict[str, Any], row.model_extra)["token_id"] = token_id
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    )
    with pytest.raises(ValueError, match="invalid exact token ID|disagree"):
        trajectory.tokenize()
