import math
from typing import Any

import pytest
from test_recorded_prompt_roles import _case
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


def _stock(templates: dict[str, str]) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=BackendTokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template=templates,
    )


@pytest.mark.parametrize(
    "templates", [{}, {"named": "{{ messages }}"}, {"tool_use": "{{ messages }}"}]
)
@pytest.mark.parametrize("route", ["history", "trajectory"])
def test_unavailable_implicit_stock_selection_preserves_native_output(
    monkeypatch: pytest.MonkeyPatch, templates: dict[str, str], route: str
) -> None:
    trajectory, _, _, exchange = _case("Plain historical content")
    exchange.response.choices[0].finish_reason = "stop"
    exchange.request.pop("chat_template")
    tokenizer = _stock(templates)
    with pytest.raises(ValueError, match="no default specified"):
        tokenizer.get_chat_template()
    expected = trajectory.tokenize(multi_history=True).histories[0]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Unavailable optional rendering must not be entered")

    monkeypatch.setattr(tokenizer, "apply_chat_template", forbidden)
    result = (
        trajectory.chat_completions_history().tokenize(tokenizer=tokenizer)
        if route == "history"
        else trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[0]
    )
    assert result.tokens == expected.tokens and result.flags == expected.flags
    assert all(
        a == b or math.isnan(a) and math.isnan(b)
        for a, b in zip(result.logprobs, expected.logprobs, strict=True)
    )


@pytest.mark.parametrize("selection", ["default", "named", "literal", "tool_use"])
@pytest.mark.parametrize("operation", ["render", "encode"])
@pytest.mark.parametrize("error_kind", [ValueError, RuntimeError])
def test_available_stock_selection_retains_callback_error(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    operation: str,
    error_kind: type[Exception],
) -> None:
    trajectory, _, _, exchange = _case("Plain historical content")
    exchange.response.choices[0].finish_reason = "stop"
    exchange.request.pop("chat_template")
    template = "{{ messages }}"
    tokenizer = _stock({selection: template})
    argument = None
    if selection in {"named", "literal"}:
        argument = selection if selection == "named" else template
        exchange.request["chat_template"] = argument
    if selection == "tool_use":
        exchange.request["tools"] = []
    assert (
        tokenizer.get_chat_template(argument, exchange.request.get("tools")) == template
    )
    failure = error_kind("Actual stock rendering callback failed")

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise failure

    if operation == "render":
        monkeypatch.setattr(tokenizer, "apply_chat_template", failed)
    else:
        monkeypatch.setattr(type(tokenizer), "__call__", failed)
    with pytest.raises(error_kind) as caught:
        trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert caught.value is failure


@pytest.mark.parametrize("error_kind", [ValueError, RuntimeError])
def test_custom_named_selector_error_is_not_classified_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, error_kind: type[Exception]
) -> None:
    trajectory, _, _, exchange = _case("Plain historical content")
    exchange.response.choices[0].finish_reason = "stop"
    exchange.request.pop("chat_template")
    tokenizer = _stock({"named": "{{ messages }}"})
    failure = error_kind("Custom selector failed")

    def select(*args: Any, **kwargs: Any) -> str:
        raise failure

    monkeypatch.setattr(tokenizer, "get_chat_template", select)
    with pytest.raises(error_kind) as caught:
        trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert caught.value is failure
