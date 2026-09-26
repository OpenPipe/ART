from copy import deepcopy
from typing import Any, cast

import pytest
from test_literal_thinking_off import _history, _NamedTemplateTokenizer

import art.trajectories as tr
from art.trajectories import _tokenize
from art_inference.chat_template import configure_preserved_thinking_chat_template


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("selection", ["default", "named", "tools"])
@pytest.mark.parametrize("route", ["history", "prompt", "completed"])
def test_selected_body_normalizes_tool_arguments_without_reselecting(
    configured, selection, route
):
    tokenizer = _NamedTemplateTokenizer()
    tokenizer.chat_template = {
        name: body.replace("tool_call.arguments|items", "tool_call.arguments.items()")
        for name, body in tokenizer.chat_template.items()
    }
    if configured:
        configure_preserved_thinking_chat_template(tokenizer)
    # A body can itself be another name: forwarding it would select WRONG.
    selected = tokenizer.chat_template[
        "tool_use" if selection == "tools" else selection
    ]
    if configured:
        tokenizer.chat_template[selected] = "WRONG"
    templates = deepcopy(tokenizer.chat_template)
    selector = "named" if selection == "named" else None
    tools: list[Any] | None = (
        [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        if selection == "tools"
        else None
    )
    messages = [
        {"role": "user", "content": "Public query."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "public-call",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"city":"Paris","count":2}',
                    },
                }
            ],
        },
    ]
    kwargs = {"enable_thinking": True, "preserve_thinking": False}
    if route == "history":
        history = tr.ChatCompletionsHistory(
            model="public/qwen35",
            messages=messages,
            message_sources=[None, None],
            tools=tools,
        )
        before = history.model_dump()
        result = history.tokenize(
            tokenizer=tokenizer,
            chat_template=selector,
            chat_template_kwargs=kwargs,
        )
        rendered = tokenizer.decode(result.tokens)
        assert not any(flag & tr.TokenFlag.SAMPLED for flag in result.flags)
        assert history.model_dump() == before
    else:
        original, _ = _history()
        source = original.message_sources[-1]
        assert source is not None
        exchange = source.exchange.model_copy(deep=True)
        assert isinstance(exchange, tr.ChatCompletionsExchange)
        exchange.request.pop("chat_template", None)
        exchange.request["messages"] = cast(Any, messages)
        if tools is not None:
            exchange.request["tools"] = tools
        before = exchange.model_dump()
        result = _tokenize._template_ids(
            tokenizer,
            exchange,
            completed=route == "completed",
            config=_tokenize._TokenizerConfig(base_model="public/qwen35"),
            chat_template=selector,
            chat_template_kwargs=kwargs,
        )
        rendered = tokenizer.decode(result)
        assert exchange.model_dump() == before
    assert "<parameter=city>\nParis\n</parameter>" in rendered
    assert "<parameter=count>\n2\n</parameter>" in rendered
    assert "WRONG" not in rendered
    assert tokenizer.chat_template == templates
    assert tokenizer.selected
    assert all(
        settings["enable_thinking"] is True and settings["preserve_thinking"] is False
        for settings in tokenizer.settings
    )


@pytest.mark.parametrize("with_tools", [False, True])
def test_named_selection_normalizes_only_the_selected_body(monkeypatch, with_tools):
    tokenizer = _NamedTemplateTokenizer()
    original = deepcopy(tokenizer.chat_template)
    tools = (
        [{"type": "function", "function": {"name": "lookup"}}] if with_tools else None
    )
    selected = tokenizer.chat_template["tool_use" if with_tools else "default"]
    normalize = _tokenize.chat_template_with_preserved_thinking
    expected = normalize(selected)
    calls = []

    def observe(template):
        calls.append(template)
        return normalize(template)

    monkeypatch.setattr(_tokenize, "chat_template_with_preserved_thinking", observe)
    selector, body, defaults = _tokenize._resolved_chat_template(
        tokenizer, tokenizer.chat_template, tools
    )
    assert body == expected
    assert selector == (expected if expected != selected else tokenizer.chat_template)
    assert defaults == {}
    assert tokenizer.chat_template == original
    assert calls == [selected]
