"""Unavailable optional roles preserve complete native tokenization."""

import math
from typing import Any

import pytest
from test_recorded_prompt_roles import _case, _extras, _interior_historical_case
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


@pytest.mark.parametrize("route", ["history", "trajectory"])
def test_stock_tokenizer_without_template_keeps_native_result(route, monkeypatch):
    trajectory, _, prompt, exchange = _case("Plain historical content")
    exchange.request.pop("chat_template")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=BackendTokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    assert tokenizer.chat_template is None
    with pytest.raises(ValueError, match="chat_template"):
        tokenizer.apply_chat_template([{"role": "user", "content": "public"}])
    calls = []
    original = tokenizer.apply_chat_template

    def observed(*args: Any, **kwargs: Any):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", observed)
    expected = trajectory.tokenize(multi_history=True).histories[0]
    actual = (
        trajectory.chat_completions_history().tokenize(tokenizer=tokenizer)
        if route == "history"
        else trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[0]
    )
    assert actual.tokens == expected.tokens == prompt + _extras(exchange)["token_ids"]
    assert actual.flags == expected.flags
    assert all(
        a == b or math.isnan(a) and math.isnan(b)
        for a, b in zip(actual.logprobs, expected.logprobs, strict=True)
    )
    assert not calls


@pytest.mark.parametrize("error_kind", [ValueError, RuntimeError])
def test_unavailable_offsets_do_not_trigger_unused_tokenizing_render(
    monkeypatch, error_kind
):
    trajectory, tokenizer, _, _ = _case("Plain historical content")
    expected = trajectory.tokenize(multi_history=True).histories[0]
    encode = type(tokenizer).__call__
    render = tokenizer.apply_chat_template
    calls = []

    def no_offsets(self, text, **kwargs):
        result = encode(self, text, **kwargs)
        result.pop("offset_mapping", None)
        return result

    def observed(*args, **kwargs):
        calls.append(kwargs.get("tokenize"))
        if kwargs.get("tokenize") is True:
            raise error_kind("Unused render must not execute")
        return render(*args, **kwargs)

    monkeypatch.setattr(type(tokenizer), "__call__", no_offsets)
    monkeypatch.setattr(tokenizer, "apply_chat_template", observed)
    result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[0]
    assert calls == [False]
    assert result.tokens == expected.tokens and result.flags == expected.flags


@pytest.mark.parametrize("operation", ["render", "encode"])
@pytest.mark.parametrize("error_kind", [ValueError, RuntimeError])
def test_required_available_template_callbacks_retain_fatal_identity(
    monkeypatch, operation, error_kind
):
    trajectory, tokenizer, _, _ = _case("Plain historical content")
    fatal = error_kind("Actual optional role proof callback failed")

    def failed(*args, **kwargs):
        raise fatal

    if operation == "render":
        monkeypatch.setattr(tokenizer, "apply_chat_template", failed)
    else:
        monkeypatch.setattr(type(tokenizer), "__call__", failed)
    with pytest.raises(error_kind) as caught:
        trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert caught.value is fatal


@pytest.mark.parametrize("mutate", [False, True])
def test_declined_offsets_still_validate_consumed_source(monkeypatch, mutate):
    trajectory, tokenizer, prompt, exchange = _case("Plain historical content")
    encode = type(tokenizer).__call__

    def no_offsets(self, text, **kwargs):
        result = encode(self, text, **kwargs)
        result.pop("offset_mapping", None)
        settings = dict(exchange.request.get("chat_template_kwargs") or {})
        if mutate:
            settings["enable_thinking"] = True
        exchange.request["chat_template_kwargs"] = settings
        return result

    monkeypatch.setattr(type(tokenizer), "__call__", no_offsets)
    if mutate:
        with pytest.raises(ValueError, match="Sampled source changed"):
            trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    else:
        result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[
            0
        ]
        assert result.tokens == prompt + _extras(exchange)["token_ids"]


@pytest.mark.parametrize("selection", [None, "named"])
@pytest.mark.parametrize("mutate", [False, True])
def test_declined_offsets_validate_raw_template_dictionary(
    monkeypatch, selection, mutate
):
    trajectory, tokenizer, records = _interior_historical_case(selection=selection)
    encode = type(tokenizer).__call__
    calls = []

    def no_offsets(self, text, **kwargs):
        result = encode(self, text, **kwargs)
        result.pop("offset_mapping", None)
        templates = dict(self.chat_template)
        if mutate:
            templates[selection or "default"] += " changed"
        self.chat_template = templates
        calls.append(True)
        return result

    monkeypatch.setattr(type(tokenizer), "__call__", no_offsets)
    if mutate:
        with pytest.raises(ValueError, match="Sampled source changed"):
            trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    else:
        result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[
            0
        ]
        assert result.tokens == records[-1][0] + records[-1][1]
    assert calls == [True]
