from __future__ import annotations

import math
from typing import Any

from openai.types.responses import Response
import pytest
from test_native_terminal_and_model import _FooterTokenizer, _terminal
from test_tokenize import (
    _CharacterTemplateTokenizer,
    _message_exchange,
    _response_exchange,
)

import art.trajectories as tr
from art.trajectories import _tokenize as module


def _history(trajectory: tr.Trajectory, protocol: str) -> Any:
    return (
        trajectory.anthropic_messages_history()
        if protocol == "messages"
        else trajectory.responses_history()
    )


def _result(
    trajectory: tr.Trajectory, protocol: str, route: str, tokenizer: Any
) -> Any:
    if route in ("history", "converted"):
        history = _history(trajectory, protocol)
        if route == "converted":
            history = history.as_chat_completions_history()
        result = history.tokenize(tokenizer=tokenizer)
        assert result.history is history
        return result
    if route == "trace":
        aggregate, traces = module._tokenize_trajectory_with_trace(
            trajectory, tokenizer=tokenizer
        )
        assert len(aggregate.histories) == len(traces) == 1
        traces[0].validate(aggregate.histories[0])
        return aggregate.histories[0]
    aggregate = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert aggregate.trajectory is trajectory and len(aggregate.histories) == 1
    return aggregate.histories[0]


@pytest.mark.parametrize("protocol", ["messages", "responses"])
@pytest.mark.parametrize("route", ["history", "converted", "trajectory", "trace"])
@pytest.mark.parametrize("termination", ["length", "stop"])
@pytest.mark.parametrize("renderer", [False, True])
def test_complete_native_terminal_never_adds_footer(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    route: str,
    termination: str,
    renderer: bool,
) -> None:
    trajectory = _terminal(protocol)
    if termination == "stop":
        if protocol == "messages":
            trajectory.exchanges.messages[0].response.stop_reason = "end_turn"
        else:
            response = trajectory.exchanges.responses[0]
            data = response.response.model_dump(mode="python")
            data.update(status="completed", incomplete_details=None)
            data["output"][0]["status"] = "completed"
            response.response = Response.model_validate(data)
    before = trajectory.model_dump()
    monkeypatch.setattr(module, "_load_tokenizer", lambda _: pytest.fail("native load"))
    tokenizer = _FooterTokenizer() if renderer else None
    if tokenizer is not None:
        monkeypatch.setattr(
            tokenizer,
            "apply_chat_template",
            lambda *a, **k: pytest.fail("native render"),
        )
    result = _result(trajectory, protocol, route, tokenizer)
    assert result.tokens == [117, 97]
    assert math.isnan(result.logprobs[0]) and result.logprobs[1:] == [-0.2]
    assert result.flags == [
        tr.TokenFlag.EXACT,
        tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.OUTPUT
        | tr.TokenFlag.ASSISTANT,
    ]
    assert trajectory.model_dump() == before


def _two_turns(protocol: str, boundary: str) -> tuple[tr.Trajectory, list[int]]:
    encode = _CharacterTemplateTokenizer._encode
    second_prompt = encode("ua" + boundary + "v")
    messages: list[Any] = [{"role": "user", "content": "u"}]
    later: list[Any] = [
        *messages,
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": "v"},
    ]
    if protocol == "messages":
        records = [
            _message_exchange(
                tr.MessagesRequest(model="test/model", messages=request, max_tokens=16),
                identifier=f"message-{i}",
                offset=i,
                content=[{"type": "text", "text": answer}],
                prompt_token_ids=prompt,
                token_ids=output,
                logprobs=[-0.2] * len(output),
                stop_reason=stop,
            )
            for i, request, prompt, answer, output, stop in [
                (0, messages, encode("u"), "a", encode("a"), "max_tokens"),
                (1, later, second_prompt, "b", encode("b§"), "end_turn"),
            ]
        ]
        trajectory = tr.Trajectory(exchanges=tr.TrajectoryExchanges(messages=records))
    else:
        records = []
        for i, request, prompt, answer, output in [
            (0, messages, encode("u"), "a", encode("a")),
            (1, later, second_prompt, "b", encode("b§")),
        ]:
            record = _response_exchange(
                f"response-{i}", output[0], offset=i, prompt_token_ids=prompt
            )
            record.request["input"] = "u" if i == 0 else "v"
            if i:
                record.request["previous_response_id"] = "response-0"
            data = record.response.model_dump(mode="python")
            data.update(
                status="incomplete" if i == 0 else "completed",
                incomplete_details={"reason": "max_output_tokens"} if i == 0 else None,
            )
            data["output"][0]["content"][0]["text"] = answer
            data["token_generations"][0]["output_tokens"] = [
                {"token_id": token, "logprob": -0.2} for token in output
            ]
            record.response = Response.model_validate(data)
            records.append(record)
        trajectory = tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=records))
    return trajectory, [*second_prompt, *encode("b§")]


@pytest.mark.parametrize("protocol", ["messages", "responses"])
@pytest.mark.parametrize("route", ["history", "converted", "trajectory", "trace"])
@pytest.mark.parametrize("boundary", ["", "§"])
def test_default_converted_native_conditioning(
    protocol: str, route: str, boundary: str
) -> None:
    trajectory, expected = _two_turns(protocol, boundary)
    before = trajectory.model_dump()
    result = _result(trajectory, protocol, route, _CharacterTemplateTokenizer())
    assert result.tokens == expected
    sampled = [i for i, flag in enumerate(result.flags) if flag & tr.TokenFlag.SAMPLED]
    assert sampled == [1, len(expected) - 2, len(expected) - 1]
    assert [result.logprobs[i] for i in sampled] == [-0.2] * 3
    assert all(
        math.isnan(lp) for i, lp in enumerate(result.logprobs) if i not in sampled
    )
    if not boundary:
        # A complete native gap needs no invented footer or role annotation.
        assert result.flags[2 : len(expected) - 2] == [tr.TokenFlag.EXACT]
        for role in (tr.TokenFlag.ASSISTANT, tr.TokenFlag.STOP, tr.TokenFlag.OUTPUT):
            assert not any(tr.first_occurrence_masks([result], where=role)[0][2:-2])
    assert trajectory.model_dump() == before


@pytest.mark.parametrize("protocol", ["messages", "responses"])
@pytest.mark.parametrize("route", ["history", "converted", "trajectory", "trace"])
def test_recorded_terminal_stop_is_preserved(protocol: str, route: str) -> None:
    trajectory = _terminal(protocol)
    if protocol == "messages":
        record = trajectory.exchanges.messages[0]
        data = record.response.model_dump(mode="python")
        data.update(token_ids=[97, 167], logprobs=[-0.2, -0.3], stop_reason="end_turn")
        record.response = type(record.response).model_validate(data)
    else:
        record = trajectory.exchanges.responses[0]
        data = record.response.model_dump(mode="python")
        data.update(status="completed", incomplete_details=None)
        data["output"][0]["status"] = "completed"
        data["token_generations"][0]["output_tokens"].append(
            {"token_id": 167, "logprob": -0.3}
        )
        record.response = Response.model_validate(data)
    result = _result(trajectory, protocol, route, _FooterTokenizer())
    assert result.tokens == [117, 97, 167]
    assert result.logprobs[1:] == [-0.2, -0.3]
    assert result.flags[-1] == (
        tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.OUTPUT
        | tr.TokenFlag.ASSISTANT
        | tr.TokenFlag.STOP
    )


@pytest.mark.parametrize("protocol", ["messages", "responses"])
@pytest.mark.parametrize("converted", [False, True])
@pytest.mark.parametrize("edit", ["message", "template", "kwargs"])
def test_edited_projection_keeps_validation_and_renderer_failures(
    protocol: str, converted: bool, edit: str
) -> None:
    history = _history(_terminal(protocol), protocol)
    if converted:
        history = history.as_chat_completions_history()
    if edit == "message":
        if isinstance(history, tr.ResponsesHistory):
            history.instructions = "edited"
        elif isinstance(history, tr.AnthropicMessagesHistory):
            history.system = "edited"
        else:
            history.messages[0]["content"] = "edited"
    elif edit == "template":
        history.chat_template = "explicit changed template"
    else:
        history.chat_template_kwargs = {"enable_thinking": False}
    failure = RuntimeError("exact renderer failure")

    class EditedTokenizer(_FooterTokenizer):
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> list[int]:
            raise failure

    if converted and edit == "message":
        with pytest.raises(ValueError, match="no longer matches its source exchange"):
            history.tokenize(tokenizer=EditedTokenizer())
    else:
        with pytest.raises(RuntimeError) as caught:
            history.tokenize(tokenizer=EditedTokenizer())
        assert caught.value is failure
