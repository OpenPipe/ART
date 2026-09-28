from __future__ import annotations

from datetime import timedelta
import math
from typing import Any

import pytest
from test_native_protocol_terminal import _history, _opaque
from test_native_terminal_and_model import _FooterTokenizer

import art.trajectories as tr
from art.trajectories import _tokenize as module

_CASES = [
    ("responses_previous", "completed"),
    ("responses_conversation", "completed"),
    ("messages_image", "end_turn"),
    ("messages_redacted", "end_turn"),
    ("messages_image", "stop_sequence"),
]


def _normal(kind: str, reason: str) -> tr.Trajectory:
    trajectory = _opaque(kind)
    if trajectory.exchanges.responses:
        exchange = trajectory.exchanges.responses[0]
        data = exchange.response.model_dump()
        data.update(status="completed", incomplete_details=None)
        data["output"][0]["status"] = "completed"
        exchange.response = type(exchange.response).model_validate(data)
    else:
        response = trajectory.exchanges.messages[0].response
        data = response.model_dump()
        data["stop_reason"] = reason
        if reason == "stop_sequence":
            data["stop_sequence"] = "stop-here"
        trajectory.exchanges.messages[0].response = type(response).model_validate(data)
    return trajectory


@pytest.mark.parametrize("kind,reason", _CASES)
@pytest.mark.parametrize("route", ["history", "trajectory"])
@pytest.mark.parametrize("renderer", [False, True])
def test_complete_normal_stop_keeps_native_opaque_evidence(
    monkeypatch: pytest.MonkeyPatch, kind: str, reason: str, route: str, renderer: bool
) -> None:
    trajectory = _normal(kind, reason)
    before = trajectory.model_dump_json()
    history = _history(trajectory)
    monkeypatch.setattr(
        module, "_load_tokenizer", lambda *_: pytest.fail("loaded tokenizer")
    )
    monkeypatch.setattr(
        type(history),
        "as_chat_completions_history",
        lambda *_: pytest.fail("projected native history"),
    )
    tokenizer = _FooterTokenizer() if renderer else None
    if tokenizer is not None:
        monkeypatch.setattr(
            tokenizer,
            "apply_chat_template",
            lambda *a, **kw: pytest.fail("rendered native history"),
        )
    if route == "history":
        result = history.tokenize(tokenizer=tokenizer)
        assert result.history is history
    else:
        aggregate = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
        assert aggregate.trajectory is trajectory
        result = aggregate.histories[0]
    assert result.tokens == [117, 97]
    assert math.isnan(result.logprobs[0]) and result.logprobs[1:] == [-0.2]
    assert result.flags == [
        tr.TokenFlag.EXACT,
        tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.OUTPUT
        | tr.TokenFlag.ASSISTANT,
    ]
    assert tr.first_occurrence_masks([result], where=tr.TokenFlag.SAMPLED) == [
        [False, True]
    ]
    assert tr.first_occurrence_masks([result], where=tr.TokenFlag.STOP) == [
        [False, False]
    ]
    assert trajectory.model_dump_json() == before


@pytest.mark.parametrize("kind,reason", _CASES)
def test_real_opaque_normal_stop_without_mocked_conversion(
    kind: str, reason: str
) -> None:
    result = _normal(kind, reason).tokenize(tokenizer=_FooterTokenizer())
    assert result.tokens == [117, 97]
    assert result.logprobs[-1] == -0.2


@pytest.mark.parametrize("kind,reason", _CASES)
@pytest.mark.parametrize(
    "change", ["prompt", "output", "context", "template", "kwargs", "model", "copied"]
)
def test_normal_stop_does_not_bypass_incomplete_or_changed_context(
    kind: str, reason: str, change: str
) -> None:
    trajectory = _normal(kind, reason)
    exchange: Any = (trajectory.exchanges.responses or trajectory.exchanges.messages)[0]
    if change in ("prompt", "output"):
        data = exchange.response.model_dump()
        if trajectory.exchanges.responses:
            generation = data["token_generations"][0]
            generation.pop(
                "prompt_token_ids" if change == "prompt" else "output_tokens"
            )
        else:
            data.pop("prompt_token_ids" if change == "prompt" else "token_ids")
        exchange.response = type(exchange.response).model_validate(data)
        with pytest.raises(ValueError):
            trajectory.tokenize(tokenizer=_FooterTokenizer())
        return
    history = _history(trajectory)
    kwargs: dict[str, Any] = {}
    if change == "context":
        if isinstance(history, tr.ResponsesHistory):
            history.instructions = "changed"
        else:
            history.system = "changed"
    elif change == "template":
        kwargs["chat_template"] = "changed"
    elif change == "kwargs":
        kwargs["chat_template_kwargs"] = {"changed": True}
    elif change == "model":
        exchange.request["model"] = "different/model"
    with pytest.raises(ValueError):
        if change == "copied":
            module._tokenize_history(
                history,
                model=history.model,
                base_model=None,
                tokenizer=_FooterTokenizer(),
                chat_template=None,
                chat_template_kwargs=None,
                _copied_context=True,
            )
        else:
            history.tokenize(tokenizer=_FooterTokenizer(), **kwargs)


@pytest.mark.parametrize("kind,reason", _CASES)
def test_earlier_normal_stop_still_needs_turn_boundary(kind: str, reason: str) -> None:
    trajectory = _normal(kind, reason)
    first: Any = (trajectory.exchanges.responses or trajectory.exchanges.messages)[0]
    second = first.model_copy(deep=True)
    second.start_time += timedelta(seconds=1)
    second.end_time += timedelta(seconds=1)
    if trajectory.exchanges.responses:
        second.request["previous_response_id"] = first.response.id
        second.request["input"] = "v"
        data = second.response.model_dump()
        data["id"] = "second"
        data["token_generations"][0]["prompt_token_ids"] = [117, 97, 118]
        data["token_generations"][0]["output_tokens"][0]["token_id"] = 98
        second.response = type(second.response).model_validate(data)
        trajectory.exchanges.responses.append(second)
    else:
        second.request["messages"] += [
            {
                "role": "assistant",
                "content": [
                    block.model_dump(exclude_none=True)
                    for block in first.response.content
                ],
            },
            {"role": "user", "content": "v"},
        ]
        second.response.model_extra.update(
            prompt_token_ids=[117, 97, 118], token_ids=[98]
        )
        trajectory.exchanges.messages.append(second)
    history = _history(trajectory)
    assert module._history_needs_synthetic_stop(
        history, _FooterTokenizer(), include_complete_terminal=False
    )
    with pytest.raises(ValueError):
        history.tokenize(tokenizer=_FooterTokenizer())


@pytest.mark.parametrize("kind,reason", _CASES)
def test_missing_native_logprobs_remain_unknown(kind: str, reason: str) -> None:
    trajectory = _normal(kind, reason)
    exchange: Any = (trajectory.exchanges.responses or trajectory.exchanges.messages)[0]
    data = exchange.response.model_dump()
    if trajectory.exchanges.responses:
        data["token_generations"][0]["output_tokens"][0].pop("logprob")
    else:
        data.pop("logprobs")
    exchange.response = type(exchange.response).model_validate(data)
    result = trajectory.tokenize(tokenizer=_FooterTokenizer())
    assert result.tokens == [117, 97]
    assert all(math.isnan(value) for value in result.logprobs)


@pytest.mark.parametrize("kind,reason", _CASES[:3])
def test_recorded_eos_and_logprob_remain_sampled(kind: str, reason: str) -> None:
    trajectory = _normal(kind, reason)
    exchange: Any = (trajectory.exchanges.responses or trajectory.exchanges.messages)[0]
    data = exchange.response.model_dump()
    if trajectory.exchanges.responses:
        generation = data["token_generations"][0]
        generation["output_tokens"].append(
            {
                **generation["output_tokens"][0],
                "token_id": _FooterTokenizer.eos_token_id,
                "logprob": -0.3,
            }
        )
    else:
        data["token_ids"].append(_FooterTokenizer.eos_token_id)
        data["logprobs"].append(-0.3)
    exchange.response = type(exchange.response).model_validate(data)
    result = trajectory.tokenize(tokenizer=_FooterTokenizer())
    assert result.tokens == [117, 97, _FooterTokenizer.eos_token_id]
    assert result.logprobs[1:] == [-0.2, -0.3]
    assert result.flags[-1] & tr.TokenFlag.SAMPLED
    assert result.flags[-1] & tr.TokenFlag.STOP
