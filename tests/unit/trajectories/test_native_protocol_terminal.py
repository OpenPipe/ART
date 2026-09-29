from __future__ import annotations

from datetime import timedelta
import math
from typing import Any, cast

import pytest
from test_native_terminal_and_model import _FooterTokenizer, _terminal

import art.trajectories as tr
from art.trajectories import _tokenize as module


def _opaque(kind: str) -> tr.Trajectory:
    trajectory = _terminal("responses" if kind.startswith("responses") else "messages")
    if kind.startswith("responses"):
        exchange = trajectory.exchanges.responses[0]
        exchange.request[
            "previous_response_id" if kind == "responses_previous" else "conversation"
        ] = "external"
    else:
        exchange = trajectory.exchanges.messages[0]
        if kind == "messages_image":
            exchange.request["messages"][0]["content"] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "public",
                    },
                }
            ]
        else:
            data = exchange.response.model_dump()
            data["content"] = [{"type": "redacted_thinking", "data": "public"}]
            exchange.response = type(exchange.response).model_validate(data)
    return trajectory


_KINDS = [
    "responses_previous",
    "responses_conversation",
    "messages_image",
    "messages_redacted",
]


def _history(trajectory: tr.Trajectory):
    return (
        trajectory.responses_history()
        if trajectory.exchanges.responses
        else trajectory.anthropic_messages_history()
    )


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("route", ["history", "trajectory"])
@pytest.mark.parametrize("renderer", [False, True])
def test_opaque_terminal_native_records_need_no_projection(
    monkeypatch: pytest.MonkeyPatch, kind: str, route: str, renderer: bool
) -> None:
    trajectory = _opaque(kind)
    history = _history(trajectory)
    before = trajectory.model_dump_json()
    monkeypatch.setattr(
        module, "_load_tokenizer", lambda *_: pytest.fail("loaded tokenizer")
    )
    monkeypatch.setattr(
        type(history),
        "as_chat_completions_history",
        lambda *_: pytest.fail("projected opaque history"),
    )
    tokenizer = _FooterTokenizer() if renderer else None
    if tokenizer is not None:
        monkeypatch.setattr(
            tokenizer,
            "apply_chat_template",
            lambda *a, **kw: pytest.fail("rendered native record"),
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


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("change", ["context", "template", "kwargs"])
def test_explicit_changed_opaque_context_still_requires_projection(
    kind: str, change: str
) -> None:
    history = _history(_opaque(kind))
    kwargs = {}
    if change == "context":
        if isinstance(history, tr.ResponsesHistory):
            history.instructions = "changed"
        else:
            history.system = "changed"
    elif change == "template":
        kwargs["chat_template"] = "changed"
    else:
        kwargs["chat_template_kwargs"] = {"changed": True}
    with pytest.raises(ValueError):
        history.tokenize(**kwargs)


@pytest.mark.parametrize("kind", _KINDS)
def test_native_admission_keeps_source_validation(kind: str) -> None:
    history = _history(_opaque(kind))
    sources = (
        history.input_sources
        if isinstance(history, tr.ResponsesHistory)
        else history.message_sources
    )
    sources[-1].exchange.request["model"] = "different/model"
    with pytest.raises(ValueError, match="model no longer matches"):
        history.tokenize()


@pytest.mark.parametrize("kind", _KINDS)
def test_actual_opaque_terminal_path(kind: str) -> None:
    result = _opaque(kind).tokenize()
    assert result.tokens == [117, 97]
    assert result.logprobs[-1] == -0.2


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("missing", ["prompt", "output"])
def test_incomplete_opaque_terminal_still_needs_projection(
    kind: str, missing: str
) -> None:
    trajectory = _opaque(kind)
    if trajectory.exchanges.responses:
        exchange = trajectory.exchanges.responses[0]
        data = exchange.response.model_dump()
        generation = data["token_generations"][0]
        if missing == "prompt":
            generation["prompt_token_ids"] = None
        elif missing == "output":
            generation.pop("output_tokens")
    else:
        exchange = trajectory.exchanges.messages[0]
        data = exchange.response.model_dump()
        data.pop({"prompt": "prompt_token_ids", "output": "token_ids"}[missing])
    cast(Any, exchange).response = type(exchange.response).model_validate(data)
    with pytest.raises(ValueError):
        trajectory.tokenize()


@pytest.mark.parametrize("kind", _KINDS)
def test_missing_logprobs_remain_unknown_on_native_terminal(kind: str) -> None:
    trajectory = _opaque(kind)
    exchange = (trajectory.exchanges.responses or trajectory.exchanges.messages)[0]
    data = exchange.response.model_dump()
    if trajectory.exchanges.responses:
        data["token_generations"][0]["output_tokens"][0].pop("logprob")
    else:
        data.pop("logprobs")
    cast(Any, exchange).response = type(exchange.response).model_validate(data)
    result = trajectory.tokenize()
    assert result.tokens == [117, 97]
    assert all(math.isnan(value) for value in result.logprobs)


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_earlier_length_stop_keeps_protocol_boundary_proof(protocol: str) -> None:
    trajectory = _opaque(
        "responses_previous" if protocol == "responses" else "messages_image"
    )
    first = (trajectory.exchanges.responses or trajectory.exchanges.messages)[0]
    second = first.model_copy(deep=True)
    second.start_time += timedelta(seconds=1)
    second.end_time += timedelta(seconds=1)
    if protocol == "responses":
        assert isinstance(second, tr.ResponsesExchange)
        second.request["previous_response_id"] = first.response.id
        second.request["input"] = "v"
        data = second.response.model_dump()
        data["id"] = "second"
        data["token_generations"][0]["prompt_token_ids"] = [117, 97, 118]
        data["token_generations"][0]["output_tokens"][0]["token_id"] = 98
        second.response = type(second.response).model_validate(data)
        trajectory.exchanges.responses.append(second)
    else:
        assert isinstance(second, tr.MessagesExchange)
        second.request["messages"] += [
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": "v"},
        ]
        extra = second.response.model_extra
        assert extra is not None
        extra["prompt_token_ids"] = [117, 97, 118]
        extra["token_ids"] = [98]
        trajectory.exchanges.messages.append(second)
    history = _history(trajectory)
    assert module._history_has_length_stop(history)
    with pytest.raises(ValueError):
        history.tokenize()
