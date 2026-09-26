from __future__ import annotations

import math
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
import pytest
from test_tokenize import _CharacterTemplateTokenizer, _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize as module


def branch(index: int, *, length: bool, model: str = "test/model", eos: int = 9):
    text = f"public question {index}"
    output = _CharacterTemplateTokenizer._encode("answer")
    if not length:
        output.append(eos)
    messages: list[dict[str, str]] = [{"role": "user", "content": text}]
    prompt = _CharacterTemplateTokenizer._encode(text)
    if length:
        # Loading is needed to prove request-owned assistant roles, independently
        # of the final length stop, which now ends at its recorded output.
        messages.insert(0, {"role": "assistant", "content": "historical context"})
        prompt = [
            *_CharacterTemplateTokenizer._encode("historical context"),
            eos,
            *prompt,
        ]
    exchange = _chat_exchange(prompt, output, model=model, offset=index)
    exchange.request["messages"] = cast(list[ChatCompletionMessageParam], messages)
    exchange.response.choices[0].finish_reason = "length" if length else "stop"
    return exchange


def trajectory(*exchanges):
    return tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=list(exchanges))
    )


def bind(monkeypatch: pytest.MonkeyPatch, tokenizers: dict[str, Any]):
    loads: list[str] = []

    def config(model: str, base_model: str | None):
        assert base_model is None
        return module._TokenizerConfig(model, chat_template="bound template")

    def load(config):
        loads.append(config.base_model)
        return tokenizers[config.base_model]

    monkeypatch.setattr(module, "_tokenizer_config", config)
    monkeypatch.setattr(module, "_load_tokenizer", load)
    return loads


def same_except_stop(left, right):
    assert len(left.histories) == len(right.histories)
    for a, b in zip(left.histories, right.histories, strict=True):
        assert a.model == b.model and a.tokens == b.tokens
        assert a.history.model_dump() == b.history.model_dump()
        assert all(
            x == y or math.isnan(x) and math.isnan(y)
            for x, y in zip(a.logprobs, b.logprobs, strict=True)
        )
        assert [f & ~tr.TokenFlag.STOP for f in a.flags] == [
            f & ~tr.TokenFlag.STOP for f in b.flags
        ]
    for flag in (tr.TokenFlag.SAMPLED, tr.TokenFlag.OUTPUT, tr.TokenFlag.ASSISTANT):
        assert tr.first_occurrence_masks(
            left.histories, where=flag
        ) == tr.first_occurrence_masks(right.histories, where=flag)


@pytest.mark.parametrize("length_first", [False, True])
def test_resolved_model_authority_completes_other_exact_history_stops(
    monkeypatch: pytest.MonkeyPatch, length_first: bool
) -> None:
    tokenizer = _CharacterTemplateTokenizer()
    loads = bind(monkeypatch, {"test/model": tokenizer})
    value = trajectory(
        branch(0, length=length_first), branch(1, length=not length_first)
    )
    original = value.model_dump()
    result = value.tokenize(multi_history=True)
    assert loads == ["test/model"]
    assert len(result.histories) == 2
    complete = result.histories[1 if length_first else 0]
    assert complete.flags[-1] & tr.TokenFlag.SAMPLED
    assert complete.flags[-1] & tr.TokenFlag.STOP
    supplied = value.tokenize(tokenizer=tokenizer, multi_history=True)
    same_except_stop(result, supplied)
    assert [h.flags for h in result.histories] == [h.flags for h in supplied.histories]
    assert value.model_dump() == original


def test_all_exact_histories_do_not_load_or_retain_prior_call_authority(monkeypatch):
    loads = bind(monkeypatch, {"test/model": _CharacterTemplateTokenizer()})
    trajectory(branch(0, length=True), branch(1, length=False)).tokenize(
        multi_history=True
    )
    assert loads == ["test/model"]
    loads.clear()
    result = trajectory(branch(0, length=False), branch(1, length=False)).tokenize(
        multi_history=True
    )
    assert loads == []
    assert all(not (h.flags[-1] & tr.TokenFlag.STOP) for h in result.histories)


def test_resolved_authority_never_crosses_model_identity(monkeypatch):
    loads = bind(monkeypatch, {"model/a": _CharacterTemplateTokenizer()})
    result = trajectory(
        branch(0, length=True, model="model/a"),
        branch(1, length=False, model="model/b"),
    ).tokenize(multi_history=True)
    assert loads == ["model/a"]
    assert not result.histories[1].flags[-1] & tr.TokenFlag.STOP


class OtherTokenizer(_CharacterTemplateTokenizer):
    eos_token_id = 8

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [
            8 if value == 9 else value
            for value in _CharacterTemplateTokenizer._encode(text)
        ]

    def convert_tokens_to_ids(self, token: str) -> int:
        return 8 if token == "§" else 0

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        return super().decode(
            [9 if token == 8 else token for token in token_ids], **kwargs
        )


def test_each_model_uses_its_own_resolved_tokenizer(monkeypatch):
    loads = bind(
        monkeypatch,
        {"model/a": _CharacterTemplateTokenizer(), "model/b": OtherTokenizer()},
    )
    result = trajectory(
        branch(0, length=True, model="model/a"),
        branch(1, length=True, model="model/b", eos=8),
        branch(2, length=False, model="model/a"),
        branch(3, length=False, model="model/b", eos=8),
    ).tokenize(multi_history=True)
    assert loads == ["model/a", "model/b"]
    assert [bool(h.flags[-1] & tr.TokenFlag.STOP) for h in result.histories] == [
        False,
        True,
        False,
        True,
    ]
    assert [h.model for h in result.histories] == [
        "model/a",
        "model/a",
        "model/b",
        "model/b",
    ]
    assert [h.tokens[-1] for h in result.histories] == [
        ord("r") + 100,
        9,
        ord("r") + 100,
        8,
    ]


def test_conflicting_resolved_tokenizers_do_not_authorize_another_history(monkeypatch):
    bind(monkeypatch, {})
    tokenizers = iter([_CharacterTemplateTokenizer(), OtherTokenizer()])
    monkeypatch.setattr(module, "_load_tokenizer", lambda config: next(tokenizers))
    result = trajectory(
        branch(0, length=True), branch(1, length=True, eos=8), branch(2, length=False)
    ).tokenize(multi_history=True)
    assert not result.histories[-1].flags[-1] & tr.TokenFlag.STOP


@pytest.mark.parametrize(
    "options",
    [
        {"chat_template": "caller override"},
        {"chat_template_kwargs": {"mode": "caller"}},
    ],
)
def test_stop_completion_does_not_select_or_change_render_overrides(
    monkeypatch, options
):
    tokenizer = _CharacterTemplateTokenizer()
    rendered = []
    original_render = tokenizer.apply_chat_template

    def render(messages, **kwargs):
        rendered.append(dict(kwargs))
        return original_render(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", render)
    bind(monkeypatch, {"test/model": tokenizer})
    value = trajectory(branch(0, length=True), branch(1, length=False))
    result = value.tokenize(multi_history=True, **options)
    actual_calls = list(rendered)
    rendered.clear()
    monkeypatch.setattr(module, "_complete_resolved_sampled_stops", lambda *args: None)
    baseline = value.tokenize(multi_history=True, **options)
    assert rendered == actual_calls
    same_except_stop(result, baseline)
    assert [h.flags for h in result.histories] == [h.flags for h in baseline.histories]


def test_nested_tokenization_does_not_share_authority(monkeypatch):
    tokenizer = _CharacterTemplateTokenizer()
    bind(monkeypatch, {"test/model": tokenizer})
    original_render = tokenizer.apply_chat_template
    nested = []

    def render(messages, **kwargs):
        if not nested:
            nested.append(
                trajectory(branch(7, length=False), branch(8, length=False)).tokenize(
                    multi_history=True
                )
            )
        return original_render(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", render)
    result = trajectory(branch(0, length=True), branch(1, length=False)).tokenize(
        multi_history=True
    )
    assert result.histories[-1].flags[-1] & tr.TokenFlag.STOP
    assert all(not (h.flags[-1] & tr.TokenFlag.STOP) for h in nested[0].histories)


def test_private_trace_and_public_results_agree(monkeypatch):
    bind(monkeypatch, {"test/model": _CharacterTemplateTokenizer()})
    value = trajectory(branch(0, length=False), branch(1, length=True))
    public = value.tokenize(multi_history=True)
    traced, traces = module._tokenize_trajectory_with_trace(value)
    same_except_stop(public, traced)
    assert [h.flags for h in public.histories] == [h.flags for h in traced.histories]
    for history, trace in zip(traced.histories, traces, strict=True):
        trace.validate(history)


def test_selected_model_does_not_load_unselected_authority(monkeypatch):
    loads = bind(monkeypatch, {"model/a": _CharacterTemplateTokenizer()})
    value = trajectory(
        branch(0, length=True, model="model/a"),
        branch(1, length=False, model="model/b"),
    )
    result = value.tokenize(model="model/b", multi_history=True)
    assert loads == [] and len(result.histories) == 1
    assert not result.histories[0].flags[-1] & tr.TokenFlag.STOP


def test_explicit_base_is_resolved_once_without_becoming_a_render_override(monkeypatch):
    calls = []
    tokenizer = _CharacterTemplateTokenizer()

    def config(model, base_model):
        calls.append((model, base_model))
        return module._TokenizerConfig(
            base_model,
            chat_template="artifact template",
            chat_template_kwargs={"public_flag": True},
        )

    monkeypatch.setattr(module, "_tokenizer_config", config)
    monkeypatch.setattr(module, "_load_tokenizer", lambda config: tokenizer)
    value = trajectory(branch(0, length=False), branch(1, length=True))
    result = value.tokenize(base_model="public/base", multi_history=True)
    assert calls == [("test/model", "public/base")]
    assert result.histories[0].flags[-1] & tr.TokenFlag.STOP


@pytest.mark.parametrize("reason", [9, "§"])
def test_recorded_stop_reason_precedence_is_preserved(monkeypatch, reason):
    loads = bind(monkeypatch, {"test/model": _CharacterTemplateTokenizer()})
    complete = branch(0, length=False)
    complete.response.choices[0].model_extra["stop_reason"] = reason
    value = trajectory(complete, branch(1, length=True))
    result = value.tokenize(multi_history=True)
    assert loads == ["test/model"]
    assert result.histories[0].flags[-1] & tr.TokenFlag.STOP


def test_marker_encoding_failure_keeps_exception_identity(monkeypatch):
    failure = RuntimeError("public stop encoder failure")

    class Tokenizer(_CharacterTemplateTokenizer):
        def __call__(self, text, **kwargs):
            if text == "public_stop_reason":
                raise failure
            return super().__call__(text, **kwargs)

    bind(monkeypatch, {"test/model": Tokenizer()})
    complete = branch(0, length=False)
    complete.response.choices[0].model_extra["stop_reason"] = "public_stop_reason"
    with pytest.raises(RuntimeError) as caught:
        trajectory(complete, branch(1, length=True)).tokenize(multi_history=True)
    assert caught.value is failure


def test_stop_postpass_keeps_copied_context_and_historical_roles(monkeypatch):
    bind(monkeypatch, {"test/model": _CharacterTemplateTokenizer()})
    first = _chat_exchange([1], [2, 9])
    second = _chat_exchange([1, 9, 4], [5, 9], offset=1)
    value = trajectory(first, second, branch(2, length=True))
    result = value.tokenize(multi_history=True)
    assert len(result.histories) == 3
    original, copied, length = result.histories
    assert original.flags[-1] & tr.TokenFlag.STOP
    assert copied.flags[-1] & tr.TokenFlag.STOP
    assert (
        copied.flags[1]
        == tr.TokenFlag.EXACT | tr.TokenFlag.ASSISTANT | tr.TokenFlag.OUTPUT
    )
    assert math.isnan(copied.logprobs[1])
    assert length.flags[-1] == (
        tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.ASSISTANT
        | tr.TokenFlag.OUTPUT
    )
    assert any(
        flag & tr.TokenFlag.STOP and not flag & tr.TokenFlag.SAMPLED
        for flag in length.flags
    )
    monkeypatch.setattr(module, "_complete_resolved_sampled_stops", lambda *args: None)
    baseline = value.tokenize(multi_history=True)
    same_except_stop(result, baseline)
    assert copied.flags[1] == baseline.histories[1].flags[1]
    assert length.flags == baseline.histories[2].flags


@pytest.mark.asyncio
async def test_public_async_default_dispatch_uses_resolved_stop_authority(monkeypatch):
    loads = bind(monkeypatch, {"test/model": _CharacterTemplateTokenizer()})
    value = trajectory(branch(0, length=True), branch(1, length=False))
    results = await tr.tokenize([value], multi_history=True)
    assert loads == ["test/model"]
    assert len(results) == 1 and results[0].trajectory is value
    assert results[0].histories[-1].flags[-1] & tr.TokenFlag.STOP
