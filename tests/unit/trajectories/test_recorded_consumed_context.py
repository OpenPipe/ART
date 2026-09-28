"""Proof dependencies must be reread after public tokenizer callbacks."""

import math
from typing import Any

import pytest
from test_recorded_prompt_roles import _case
from test_tokenize import _character_template_history


@pytest.mark.parametrize("request_owns_model", [False, True])
@pytest.mark.parametrize("outcome", ["unchanged", "changed", "decline", "fatal"])
def test_boundary_proof_rechecks_effective_model(
    monkeypatch: pytest.MonkeyPatch, request_owns_model: bool, outcome: str
) -> None:
    history, tokenizer, tokens = _character_template_history()
    source = history.message_sources[3]
    assert source is not None
    exchange = source.exchange
    if not request_owns_model:
        exchange.request.pop("model")
    old_model = exchange.model
    assert old_model is not None
    original = tokenizer.apply_chat_template
    fatal = RuntimeError("Public boundary rendering failed")
    called = False

    def callback(*args: Any, **kwargs: Any):
        nonlocal called
        value = original(*args, **kwargs)
        called = True
        exchange.response.model = old_model if outcome == "unchanged" else "other/model"
        if outcome == "fatal":
            raise fatal
        if outcome == "decline":
            raise NotImplementedError("Public optional rendering unavailable")
        return value

    monkeypatch.setattr(tokenizer, "apply_chat_template", callback)
    if outcome == "fatal":
        with pytest.raises(RuntimeError) as caught:
            history.tokenize(tokenizer=tokenizer)
        assert caught.value is fatal
    elif not request_owns_model and outcome != "unchanged":
        with pytest.raises(ValueError, match="Sampled source changed"):
            history.tokenize(tokenizer=tokenizer)
    elif outcome == "decline":
        # An unchanged effective model does not turn ordinary capability failure
        # into an invented source-mutation error.
        with pytest.raises(NotImplementedError):
            history.tokenize(tokenizer=tokenizer)
    else:
        result = history.tokenize(tokenizer=tokenizer)
        assert result.model == old_model
        assert result.tokens == tokens
    assert called


@pytest.mark.parametrize(
    "setting", ["request-template", "request-kwargs", "tokenizer-template"]
)
@pytest.mark.parametrize("outcome", ["unchanged", "changed", "decline", "fatal"])
def test_role_proof_rechecks_current_effective_settings(
    monkeypatch: pytest.MonkeyPatch, setting: str, outcome: str
) -> None:
    trajectory, tokenizer, _, exchange = _case("Plain historical content")
    if setting == "tokenizer-template":
        exchange.request.pop("chat_template")
    expected = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[0]
    original = tokenizer.apply_chat_template
    original_template = tokenizer.chat_template
    original_kwargs = dict(exchange.request.get("chat_template_kwargs") or {})
    fatal = RuntimeError("Public role rendering failed")
    called = False

    def callback(*args: Any, **kwargs: Any):
        nonlocal called
        value = original(*args, **kwargs)
        if not called:
            called = True
            template = (
                original_template if outcome == "unchanged" else "{{ 'changed' }}"
            )
            if setting == "request-template":
                exchange.request["chat_template"] = template
            elif setting == "tokenizer-template":
                tokenizer.chat_template = template
            else:
                # Replacing a mapping with an equal fresh copy is permitted.
                exchange.request["chat_template_kwargs"] = {
                    **original_kwargs,
                    **({} if outcome == "unchanged" else {"enable_thinking": True}),
                }
            if outcome == "fatal":
                raise fatal
            if outcome == "decline":
                raise NotImplementedError("Public role rendering unavailable")
        return value

    monkeypatch.setattr(tokenizer, "apply_chat_template", callback)
    if outcome == "fatal":
        with pytest.raises(RuntimeError) as caught:
            trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
        assert caught.value is fatal
    elif outcome != "unchanged":
        with pytest.raises(ValueError, match="Sampled source changed"):
            trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    else:
        actual = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[
            0
        ]
        assert actual.tokens == expected.tokens and actual.flags == expected.flags
        assert all(
            left == right or math.isnan(left) and math.isnan(right)
            for left, right in zip(actual.logprobs, expected.logprobs, strict=True)
        )
    assert called
