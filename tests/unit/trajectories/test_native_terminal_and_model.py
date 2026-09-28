from __future__ import annotations

import math
from typing import Any

from openai.types.responses import Response
import pytest
from test_recorded_prompt_roles import _case
from test_tokenize import _message_exchange, _response_exchange

import art.trajectories as tr
from art.trajectories import _tokenize as module


class _FooterTokenizer:
    eos_token_id = ord("§")

    def __call__(self, text: str, **kwargs: object) -> list[int]:
        return list(map(ord, text))

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: object
    ) -> list[int]:
        rendered = ""
        for message in messages:
            content = message.get("content") or ""
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            rendered += content + ("§" if message["role"] == "assistant" else "")
        return self(rendered)


def _terminal(protocol: str) -> tr.Trajectory:
    if protocol == "messages":
        message = _message_exchange(
            tr.MessagesRequest(
                model="test/model",
                messages=[{"role": "user", "content": "u"}],
                max_tokens=16,
            ),
            content=[{"type": "text", "text": "a"}],
            prompt_token_ids=[117],
            token_ids=[97],
            logprobs=[-0.2],
            stop_reason="max_tokens",
        )
        exchanges = tr.TrajectoryExchanges(messages=[message])
    else:
        response = _response_exchange("length", 97, prompt_token_ids=[117])
        response.request["input"] = "u"
        data = response.response.model_dump(mode="python")
        data.update(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
        data["output"][0]["status"] = "incomplete"
        data["output"][0]["content"][0]["text"] = "a"
        data["token_generations"][0]["output_tokens"][0]["logprob"] = -0.2
        response.response = Response.model_validate(data)
        exchanges = tr.TrajectoryExchanges(responses=[response])
    return tr.Trajectory(exchanges=exchanges)


@pytest.mark.parametrize("protocol", ["messages", "responses"])
@pytest.mark.parametrize("route", ["history", "trajectory"])
@pytest.mark.parametrize("renderer", [False, True])
def test_complete_terminal_length_uses_native_tokens_without_rendering(
    monkeypatch: pytest.MonkeyPatch, protocol: str, route: str, renderer: bool
) -> None:
    trajectory = _terminal(protocol)
    before = trajectory.model_dump()
    monkeypatch.setattr(
        module,
        "_load_tokenizer",
        lambda _: pytest.fail("native terminal loaded tokenizer"),
    )
    tokenizer = _FooterTokenizer() if renderer else None
    if tokenizer is not None:
        monkeypatch.setattr(
            tokenizer,
            "apply_chat_template",
            lambda *args, **kwargs: pytest.fail("native terminal rendered footer"),
        )
    if route == "history":
        history = (
            trajectory.anthropic_messages_history()
            if protocol == "messages"
            else trajectory.responses_history()
        )
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
    assert trajectory.model_dump() == before


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_changed_terminal_context_still_renders(protocol: str) -> None:
    trajectory = _terminal(protocol)
    if protocol == "messages":
        history = trajectory.anthropic_messages_history()
        history.system = "changed"
    else:
        history = trajectory.responses_history()
        history.instructions = "changed"
    error = RuntimeError("changed context reached renderer")

    class ChangedTokenizer(_FooterTokenizer):
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> list[int]:
            raise error

    with pytest.raises(RuntimeError) as caught:
        history.tokenize(tokenizer=ChangedTokenizer())
    assert caught.value is error


@pytest.mark.parametrize("model_source", ["request", "response"])
@pytest.mark.parametrize("outcome", ["return", "decline", "fatal"])
def test_role_proof_guards_effective_source_model(
    model_source: str, outcome: str
) -> None:
    trajectory, tokenizer, _, exchange = _case("Plain historical content")
    if model_source == "response":
        exchange.request.pop("model", None)
    history = trajectory.chat_completions_history()
    original = tokenizer.apply_chat_template
    fatal = RuntimeError("original role renderer failure")
    calls = 0

    def changed(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if model_source == "request":
            exchange.request["model"] = "other/model"
        else:
            exchange.response.model = "other/model"
        if outcome == "decline":
            raise NotImplementedError("optional renderer declined")
        if outcome == "fatal":
            raise fatal
        return result

    tokenizer.apply_chat_template = changed
    if outcome == "fatal":
        with pytest.raises(RuntimeError) as caught:
            history.tokenize(tokenizer=tokenizer)
        assert caught.value is fatal
    else:
        with pytest.raises(ValueError, match="Sampled source changed"):
            history.tokenize(tokenizer=tokenizer)
    assert calls > 0


def test_role_proof_keeps_unchanged_effective_model() -> None:
    trajectory, tokenizer, _, exchange = _case("Plain historical content")
    original = tokenizer.apply_chat_template

    def changed_unused_response(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        # Explicit request model remains the model consumed by this history.
        exchange.response.model = "provider/model-alias"
        return result

    tokenizer.apply_chat_template = changed_unused_response
    result = trajectory.chat_completions_history().tokenize(tokenizer=tokenizer)
    assert result.model == exchange.model == "public/qwen35"
