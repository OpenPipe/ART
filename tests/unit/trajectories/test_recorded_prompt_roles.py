from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
import pytest
from test_literal_thinking_off import _TEMPLATE, _history

import art.trajectories as tr
from art.trajectories import _tokenize as module


def _extras(exchange: tr.ChatCompletionsExchange) -> dict[str, Any]:
    extra = exchange.response.choices[0].model_extra
    assert extra is not None
    return extra


def _case(content: str, *, output: str = "New recorded answer"):
    history, tokenizer = _history(content=output)
    source = history.message_sources[-1]
    assert source is not None
    exchange = source.exchange
    assert isinstance(exchange, tr.ChatCompletionsExchange)
    messages = [
        {"role": "user", "content": "Old public query"},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "New public query"},
    ]
    exchange.request["messages"] = cast(
        list[ChatCompletionMessageParam], deepcopy(messages)
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        chat_template=_TEMPLATE,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=True,
    )
    _extras(exchange)["prompt_token_ids"] = prompt
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    )
    return trajectory, tokenizer, prompt, exchange


@pytest.mark.parametrize(
    "content",
    [
        "Public reasoning\n\n\n\n\n</think>Public answer",
        "Public reasoning</think>Public answer",
        "Public reasoning</think>middle</think>Public answer",
    ],
)
def test_recorded_request_roles_follow_proved_historical_renderer(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    trajectory, tokenizer, prompt, exchange = _case(content)
    before = trajectory.model_dump()
    kwargs = dict(tokenizer=tokenizer, multi_history=True)
    result = trajectory.tokenize(**kwargs)
    assert len(result.histories) == 1
    actual = result.histories[0]
    output = _extras(exchange)["token_ids"]
    assert actual.tokens[: len(prompt)] == prompt
    assert actual.tokens[len(prompt) : len(prompt) + len(output)] == output
    assert actual.logprobs[len(prompt) : len(prompt) + len(output)] == [-0.5] * len(
        output
    )
    assert all(math.isnan(lp) for lp in actual.logprobs[: len(prompt)])
    required = (
        tr.TokenFlag.SAMPLED
        | tr.TokenFlag.EXACT
        | tr.TokenFlag.ASSISTANT
        | tr.TokenFlag.OUTPUT
    )
    assert all(
        flag & required == required
        for flag in actual.flags[len(prompt) : len(prompt) + len(output)]
    )
    assert not any(
        flag & (tr.TokenFlag.SAMPLED | tr.TokenFlag.OUTPUT)
        for flag in actual.flags[: len(prompt)]
    )
    assert any(flag & tr.TokenFlag.ASSISTANT for flag in actual.flags[: len(prompt)])
    assert sum(
        tr.first_occurrence_masks(result.histories, where=tr.TokenFlag.SAMPLED)[0]
    ) == len(output)
    assert trajectory.model_dump() == before
    # This original template is the successful historical rendering oracle for
    # this public request; new source normalization must not change its roles.
    with monkeypatch.context() as patch:
        patch.setattr(
            module, "chat_template_with_preserved_thinking", lambda value: value
        )
        historical = trajectory.tokenize(**kwargs).histories[0]
    assert actual.tokens == historical.tokens
    assert actual.flags == historical.flags
    assert all(
        a == b or math.isnan(a) and math.isnan(b)
        for a, b in zip(actual.logprobs, historical.logprobs, strict=True)
    )
    for flag in (
        tr.TokenFlag.SAMPLED,
        tr.TokenFlag.OUTPUT,
        tr.TokenFlag.ASSISTANT,
        tr.TokenFlag.STOP,
    ):
        assert tr.first_occurrence_masks(
            [actual], where=flag
        ) == tr.first_occurrence_masks([historical], where=flag)


def test_historical_context_does_not_reenable_literal_parsing_for_new_output() -> None:
    literal = "New literal </think> must remain content"
    trajectory, tokenizer, prompt, exchange = _case(
        "Public reasoning\n\n\n\n\n</think>Public answer", output=literal
    )
    result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    actual = result.histories[0]
    output = _extras(exchange)["token_ids"]
    assert actual.tokens[len(prompt) : len(prompt) + len(output)] == output
    assert literal in tokenizer.rendered[-1] or any(
        literal in text for text in tokenizer.rendered
    )


def test_historical_tool_serialization_uses_original_request_key_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, tokenizer, _, exchange = _case(
        "Public reasoning\n\n\n\n\n</think>Public answer"
    )
    # HF's template tojson filter preserves insertion order.
    tokenizer.env.policies["json.dumps_kwargs"] = {"sort_keys": False}
    tools = [
        {
            "type": "function",
            "function": {
                "parameters": {"type": "object", "properties": {}},
                "description": "Public tool",
                "name": "lookup",
            },
        }
    ]
    exchange.request["tools"] = tools
    prompt = tokenizer.apply_chat_template(
        exchange.request["messages"],
        tools=tools,
        chat_template=_TEMPLATE,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=True,
    )
    _extras(exchange)["prompt_token_ids"] = prompt
    history = trajectory.chat_completions_history()
    assert history.tools == tools
    assert json.dumps(history.tools) != json.dumps(tools)
    assert (
        tokenizer.apply_chat_template(
            history.messages[:-1],
            tools=history.tools,
            chat_template=_TEMPLATE,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=True,
        )
        != prompt
    )
    before = trajectory.model_dump()
    actual = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[0]
    with monkeypatch.context() as patch:
        patch.setattr(
            module, "chat_template_with_preserved_thinking", lambda value: value
        )
        historical = trajectory.tokenize(
            tokenizer=tokenizer, multi_history=True
        ).histories[0]
    assert actual.tokens == historical.tokens
    assert actual.flags == historical.flags
    assert actual.tokens[: len(prompt)] == prompt
    assert trajectory.model_dump() == before


@pytest.mark.parametrize("change", ["prompt", "edited", "override"])
def test_unproved_historical_renderer_does_not_certify_context(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    trajectory, tokenizer, _, exchange = _case(
        "Public reasoning\n\n\n\n\n</think>Public answer"
    )
    history = trajectory.chat_completions_history()
    kwargs: dict[str, Any] = {"tokenizer": tokenizer}
    if change == "prompt":
        _extras(exchange)["prompt_token_ids"][0] += 1
    elif change == "edited":
        history.messages[1]["content"] = "Edited public context"
    else:
        kwargs["chat_template"] = _TEMPLATE + "{# explicit renderer #}"
    original = module._recorded_prompt_role_masks
    admitted = []

    def observe(*args: Any, **options: Any):
        value = original(*args, **options)
        admitted.append(value)
        return value

    monkeypatch.setattr(module, "_recorded_prompt_role_masks", observe)

    def outcome():
        try:
            return history.tokenize(**kwargs)
        except ValueError as error:
            return type(error), str(error)

    actual = outcome()
    assert not any(value is not None for value in admitted)
    monkeypatch.setattr(
        module, "_recorded_prompt_role_masks", lambda *args, **kwargs: None
    )
    baseline = outcome()
    if isinstance(actual, tuple):
        assert actual == baseline
    else:
        assert actual.tokens == baseline.tokens
        assert actual.flags == baseline.flags


def test_sampled_history_is_not_reclassified_as_request_context() -> None:
    history, tokenizer = _history()
    assert (
        module._recorded_prompt_role_masks(
            cast(list[dict[str, Any]], history.messages),
            history.message_sources,
            [],
            tokenizer=tokenizer,
            template=_TEMPLATE,
            tools=None,
            kwargs={},
        )
        is None
    )


def test_unchanged_template_length_retry_preserves_request_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, tokenizer, prompt, _ = _case("Plain historical assistant content")
    monkeypatch.setattr(
        module, "chat_template_with_preserved_thinking", lambda value: value
    )
    result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[0]
    assert result.tokens[: len(prompt)] == prompt
    assert any(flag & tr.TokenFlag.ASSISTANT for flag in result.flags[: len(prompt)])
    assert not any(
        flag & (tr.TokenFlag.OUTPUT | tr.TokenFlag.SAMPLED)
        for flag in result.flags[: len(prompt)]
    )


@pytest.mark.parametrize(
    "change",
    ["message", "tools", "kwargs", "source", "tool_order", "source_tool_order"],
)
def test_historical_proof_refuses_renderer_mutation(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    trajectory, tokenizer, _, exchange = _case(
        "Public reasoning\n\n\n\n\n</think>Public answer"
    )
    render = tokenizer.apply_chat_template
    calls = 0

    def mutate(messages, **kwargs):
        nonlocal calls
        value = render(messages, **kwargs)
        if kwargs.get("chat_template") == _TEMPLATE:
            calls += 1
            if calls == 2:
                if change == "message":
                    messages[0]["content"] = "Changed public content"
                elif change == "source":
                    _extras(exchange)["prompt_token_ids"][0] += 1
                elif change == "tools":
                    kwargs["tools"][0]["function"]["description"] = "Changed"
                elif change in {"tool_order", "source_tool_order"}:
                    tools = (
                        kwargs["tools"]
                        if change == "tool_order"
                        else exchange.request["tools"]
                    )
                    function = tools[0]["function"]
                    function["name"] = function.pop("name")
                else:
                    # Mutate nested kwargs, which Python's ** expansion shares.
                    kwargs["public_option"]["changed"] = True
        return value

    if change in {"tools", "tool_order", "source_tool_order"}:
        exchange.request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Original",
                    "parameters": {},
                },
            }
        ]
        # Tools affect this template, so reconstruct the authoritative prompt.
        _extras(exchange)["prompt_token_ids"] = render(
            exchange.request["messages"],
            tools=exchange.request["tools"],
            chat_template=_TEMPLATE,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=True,
        )
    if change == "kwargs":
        exchange.request["chat_template_kwargs"]["public_option"] = {"changed": False}
    monkeypatch.setattr(tokenizer, "apply_chat_template", mutate)
    with pytest.raises(ValueError, match="changed|does not match|differs"):
        trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert calls >= 2


@pytest.mark.parametrize(
    "error", [RuntimeError("public callback"), KeyboardInterrupt(), SystemExit(9)]
)
def test_historical_proof_preserves_callback_exception_identity(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    trajectory, tokenizer, _, _ = _case(
        "Public reasoning\n\n\n\n\n</think>Public answer"
    )
    render = tokenizer.apply_chat_template

    def fail(messages, **kwargs):
        if kwargs.get("chat_template") == _TEMPLATE:
            raise error
        return render(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", fail)
    with pytest.raises(type(error)) as caught:
        trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert caught.value is error


@pytest.mark.parametrize("error", [TypeError(), KeyError(), NotImplementedError()])
def test_unsupported_historical_prefix_keeps_existing_translation(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    trajectory, tokenizer, _, _ = _case("Public reasoning</think>Public answer")
    with monkeypatch.context() as patch:
        patch.setattr(
            module, "_recorded_prompt_role_masks", lambda *args, **kwargs: None
        )
        expected = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    render = tokenizer.apply_chat_template
    failures = 0

    def unsupported(messages, **kwargs):
        nonlocal failures
        if kwargs.get("chat_template") == _TEMPLATE and len(messages) < 3:
            failures += 1
            raise error
        return render(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", unsupported)
    actual = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert failures == 1
    assert actual.histories[0].tokens == expected.histories[0].tokens
    assert actual.histories[0].flags == expected.histories[0].flags


@pytest.mark.parametrize("error", [TypeError(), NotImplementedError()])
def test_unsupported_historical_offsets_keep_existing_translation(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    trajectory, tokenizer, _, _ = _case("Public reasoning</think>Public answer")
    with monkeypatch.context() as patch:
        patch.setattr(
            module, "_recorded_prompt_role_masks", lambda *args, **kwargs: None
        )
        expected = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    encode = type(tokenizer).__call__
    failures = 0

    def unsupported(self, text, **kwargs):
        nonlocal failures
        if kwargs.get("return_offsets_mapping"):
            failures += 1
            raise error
        return encode(self, text, **kwargs)

    monkeypatch.setattr(type(tokenizer), "__call__", unsupported)
    actual = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert failures >= 1
    assert actual.histories[0].tokens == expected.histories[0].tokens
    assert actual.histories[0].flags == expected.histories[0].flags


@pytest.mark.parametrize("offset", [(0, 0), (-1, 1), (0, 100000), (False, 1)])
def test_unproved_historical_offset_declines(
    monkeypatch: pytest.MonkeyPatch, offset: tuple[int, int]
) -> None:
    trajectory, tokenizer, prompt, _ = _case("Public reasoning</think>Public answer")
    history = trajectory.chat_completions_history()
    encode = type(tokenizer).__call__

    def malformed(self, text, **kwargs):
        result = encode(self, text, **kwargs)
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"][0] = offset
        return result

    monkeypatch.setattr(type(tokenizer), "__call__", malformed)
    assert (
        module._recorded_prompt_role_masks(
            history.messages[:-1],
            history.message_sources[:-1],
            prompt,
            tokenizer=tokenizer,
            template=_TEMPLATE,
            tools=None,
            kwargs={"enable_thinking": False, "preserve_thinking": True},
        )
        is None
    )


@pytest.mark.parametrize("offsets", [False, True])
def test_interior_historical_parser_cannot_change_sample_conditioning(
    monkeypatch, offsets
):
    from openai.types.chat import ChatCompletion
    from test_literal_thinking_off import _TemplateTokenizer
    from test_tokenize import _chat_exchange

    from art_inference.chat_template import _QWEN_INLINE_STATEMENTS

    template = (
        "{% for message in messages %}{% set content = message.content %}"
        + "".join("{% " + part + " %}" for part in _QWEN_INLINE_STATEMENTS)
        + "{{ content }}{% if message.role == 'assistant' %}§{% endif %}{% endfor %}"
    )

    class Tokenizer(_TemplateTokenizer):
        eos_token_id = ord("§")

        def __call__(self, text, **kwargs):
            if kwargs.get("return_offsets_mapping") and not offsets:
                raise NotImplementedError("No public offsets")
            return super().__call__(text, **kwargs)

    tokenizer = Tokenizer()
    tokenizer.chat_template = template
    messages = [{"role": "user", "content": "first query"}]
    exchanges = []
    records = []
    for index in range(2):
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        assert isinstance(prompt, list)
        output = list(map(ord, f"answer{index}§"))
        exchange = _chat_exchange(prompt, output, offset=index)
        exchange.request["messages"] = cast(
            list[ChatCompletionMessageParam], deepcopy(messages)
        )
        exchange.request["chat_template"] = template
        payload = exchange.response.model_dump(mode="python")
        payload["choices"][0]["message"]["content"] = f"answer{index}"
        exchange.response = ChatCompletion.model_validate(payload)
        exchanges.append(exchange)
        records.append((prompt, output))
        messages.extend(
            [
                {"role": "assistant", "content": f"answer{index}"},
                {"role": "user", "content": "middle query"},
                {"role": "assistant", "content": "x</think>y"},
                {"role": "user", "content": "next query"},
            ]
        )
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=exchanges))
    original = value.model_dump()
    if not offsets:
        with pytest.raises(ValueError, match="Cannot preserve request roles"):
            value.tokenize(tokenizer=tokenizer, multi_history=True)
        assert value.model_dump() == original
        return
    actual = value.tokenize(tokenizer=tokenizer, multi_history=True)
    assert len(actual.histories) == 1
    result = actual.histories[0]
    assert result.tokens == records[-1][0] + records[-1][1]
    for prompt, output in records:
        assert result.tokens[: len(prompt)] == prompt
        assert result.tokens[len(prompt) : len(prompt) + len(output)] == output
        assert all(
            flag & tr.TokenFlag.SAMPLED
            for flag in result.flags[len(prompt) : len(prompt) + len(output)]
        )
    assert value.model_dump() == original

    with monkeypatch.context() as patch:
        patch.setattr(
            module, "chat_template_with_preserved_thinking", lambda value: value
        )
        expected = value.tokenize(tokenizer=tokenizer, multi_history=True)
    assert result.flags == expected.histories[0].flags
    assert all(
        a == b or math.isnan(a) and math.isnan(b)
        for a, b in zip(result.logprobs, expected.histories[0].logprobs, strict=True)
    )
    for flag in (
        tr.TokenFlag.SAMPLED,
        tr.TokenFlag.OUTPUT,
        tr.TokenFlag.ASSISTANT,
        tr.TokenFlag.STOP,
    ):
        assert tr.first_occurrence_masks(
            actual.histories, where=flag
        ) == tr.first_occurrence_masks(expected.histories, where=flag)
