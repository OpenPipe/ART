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
    # Complete recorded output is authoritative and need not be rendered.
    assert (
        "".join(map(chr, actual.tokens[len(prompt) : len(prompt) + len(output)]))
        == literal
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


def _interior_historical_case(offsets=True, selection="literal", turns=2):
    from openai.types.chat import ChatCompletion
    from test_literal_thinking_off import _TemplateTokenizer
    from test_tokenize import _chat_exchange

    from art_inference.chat_template import _QWEN_INLINE_STATEMENTS

    template = (
        "{% for message in messages %}{% set content = message.content %}"
        + "".join("{% " + part + " %}" for part in _QWEN_INLINE_STATEMENTS)
        + "{{ content }}{% if message.role == 'assistant' %}§{% endif %}{% endfor %}"
    )

    if selection != "literal":
        template = template.replace(
            "{{ content }}",
            "{{ content }}{% for tool_call in message.tool_calls or [] %}{% for key, arg in tool_call.function.arguments.items() %}{{ key }}={{ arg }}{% endfor %}{% endfor %}",
        )

    class Tokenizer(_TemplateTokenizer):
        chat_template: Any
        eos_token_id = ord("§")

        def get_chat_template(self, chat_template=None, tools=None):
            if isinstance(self.chat_template, dict):
                return self.chat_template.get(chat_template or "default", chat_template)
            return chat_template or self.chat_template

        def apply_chat_template(self, messages, **kwargs):
            selected = self.get_chat_template(
                kwargs.pop("chat_template", None), kwargs.get("tools")
            )
            return super().apply_chat_template(
                messages, chat_template=selected, **kwargs
            )

        def __call__(self, text, **kwargs):
            if kwargs.get("return_offsets_mapping") and not offsets:
                raise NotImplementedError("No public offsets")
            return super().__call__(text, **kwargs)

    tokenizer = Tokenizer()
    tokenizer.chat_template = (
        template if selection == "literal" else {"default": template, "named": template}
    )
    messages = [{"role": "user", "content": "first query"}]
    exchanges = []
    records = []
    for index in range(turns):
        prompt = tokenizer.apply_chat_template(
            module.normalize_tool_call_arguments_for_chat_template(messages, template),
            add_generation_prompt=True,
        )
        assert isinstance(prompt, list)
        output = list(map(ord, f"answer{index}§"))
        exchange = _chat_exchange(prompt, output, offset=index)
        exchange.request["messages"] = cast(
            list[ChatCompletionMessageParam], deepcopy(messages)
        )
        if selection is not None:
            exchange.request["chat_template"] = (
                template if selection == "literal" else selection
            )
        payload = exchange.response.model_dump(mode="python")
        payload["choices"][0]["message"]["content"] = f"answer{index}"
        exchange.response = ChatCompletion.model_validate(payload)
        exchanges.append(exchange)
        records.append((prompt, output))
        historical: dict[str, Any] = {"role": "assistant", "content": "x</think>y"}
        if selection != "literal":
            historical["tool_calls"] = [
                {
                    "id": "public-tool",
                    "type": "function",
                    "function": {
                        "name": "public",
                        "arguments": '{"public": "argument"}',
                    },
                }
            ]
        messages.extend(
            [
                {"role": "assistant", "content": f"answer{index}"},
                {"role": "user", "content": "middle query"},
                historical,
                {"role": "user", "content": "next query"},
            ]
        )
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=exchanges))
    return value, tokenizer, records


@pytest.mark.parametrize("offsets", [False, True])
@pytest.mark.parametrize("turns", [2, 3])
def test_interior_historical_parser_cannot_change_sample_conditioning(
    monkeypatch, offsets, turns
):
    value, tokenizer, records = _interior_historical_case(offsets, turns=turns)
    original = value.model_dump()
    actual = value.tokenize(tokenizer=tokenizer, multi_history=True)
    assert len(actual.histories) == 1
    result = actual.histories[0]
    assert result.tokens == records[-1][0] + records[-1][1]
    text = tokenizer.decode(result.tokens)
    # The original request template emitted only y; its recorded conditioning
    # stays authoritative even though new rendering preserves the literal text.
    historical = "y"
    cursor = 0
    for _ in range(turns - 1):
        start = text.index(historical + "§", cursor)
        end = start + len(historical)
        if offsets:
            assert all(
                flag & tr.TokenFlag.ASSISTANT for flag in result.flags[start:end]
            )
            assert result.flags[end] & tr.TokenFlag.STOP
        else:
            assert result.flags[start : end + 1] == [tr.TokenFlag.EXACT] * (
                end + 1 - start
            )
            assert all(math.isnan(lp) for lp in result.logprobs[start : end + 1])
        assert not any(flag & tr.TokenFlag.SAMPLED for flag in result.flags[start:end])
        cursor = end
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


@pytest.mark.parametrize("route", ["history", "single", "multi"])
def test_historical_role_callback_cannot_replace_consumed_earlier_logprob(
    monkeypatch, route
):
    value, tokenizer, _ = _interior_historical_case()
    first = value.exchanges.chat_completions[0]
    apply = tokenizer.apply_chat_template
    calls = []

    def mutate(messages, **kwargs):
        if kwargs.get("chat_template") == tokenizer.chat_template and len(messages) > 3:
            calls.append(True)
            logprobs = first.response.choices[0].logprobs
            assert logprobs is not None and logprobs.content is not None
            logprobs.content[0].logprob -= 0.5
        return apply(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", mutate)
    with pytest.raises(ValueError, match="[Ss]ampled source changed"):
        if route == "history":
            value.chat_completions_history().tokenize(tokenizer=tokenizer)
        else:
            value.tokenize(tokenizer=tokenizer, multi_history=route == "multi")
    assert calls


@pytest.mark.parametrize("selection", [None, "named"])
def test_named_original_request_proof_normalizes_tool_arguments(selection):
    value, tokenizer, records = _interior_historical_case(selection=selection)
    before = value.model_dump()
    result = value.tokenize(tokenizer=tokenizer, multi_history=True)
    assert result.histories[0].tokens == records[-1][0] + records[-1][1]
    assert value.model_dump() == before
    literal = value.model_copy(deep=True)
    for exchange in literal.exchanges.chat_completions:
        exchange.request["chat_template"] = tokenizer.chat_template["default"]
    expected = literal.tokenize(tokenizer=tokenizer, multi_history=True)
    assert result.histories[0].flags == expected.histories[0].flags
    assert all(
        a == b or math.isnan(a) and math.isnan(b)
        for a, b in zip(
            result.histories[0].logprobs, expected.histories[0].logprobs, strict=True
        )
    )
    for flag in (
        tr.TokenFlag.SAMPLED,
        tr.TokenFlag.OUTPUT,
        tr.TokenFlag.ASSISTANT,
        tr.TokenFlag.STOP,
    ):
        assert tr.first_occurrence_masks(
            result.histories, where=flag
        ) == tr.first_occurrence_masks(expected.histories, where=flag)
    for prompt, output in records:
        history = result.histories[0]
        assert history.tokens[: len(prompt)] == prompt
        assert history.tokens[len(prompt) : len(prompt) + len(output)] == output
        assert all(
            flag & tr.TokenFlag.SAMPLED
            for flag in history.flags[len(prompt) : len(prompt) + len(output)]
        )


@pytest.mark.parametrize("route", ["history", "trajectory", "trace"])
@pytest.mark.parametrize("historical", ["x</think>y", "y"])
@pytest.mark.parametrize("loader", [False, True])
def test_length_stop_keeps_original_conditioning_and_synthetic_stop_roles(
    monkeypatch, route, historical, loader
):
    trajectory, tokenizer, records = _interior_historical_case()
    options: dict[str, Any]
    if loader:
        monkeypatch.setattr(module, "_load_tokenizer", lambda config: tokenizer)
        options = {"base_model": "public/required-boundary-renderer"}
    else:
        options = {"tokenizer": tokenizer}
    first = trajectory.exchanges.chat_completions[0].response.choices[0]
    assert first.model_extra is not None and first.logprobs is not None
    assert first.logprobs.content is not None
    first.finish_reason = "length"
    first.model_extra["token_ids"] = first.model_extra["token_ids"][:-1]
    first.logprobs.content = first.logprobs.content[:-1]
    trajectory.exchanges.chat_completions[1].request["messages"][3]["content"] = (
        historical
    )
    before = trajectory.model_dump()
    try:
        if route == "history":
            result = trajectory.chat_completions_history().tokenize(**options)
        elif route == "trajectory":
            result = trajectory.tokenize(multi_history=True, **options).histories[0]
        else:
            value, _ = module._tokenize_trajectory_with_trace(trajectory, **options)
            result = value.histories[0]
    except ValueError as error:
        assert historical == "x</think>y"
        assert "recorded prompt" in str(error) or "native conditioning" in str(error)
    else:
        assert result.tokens == records[-1][0] + records[-1][1]
        boundary = len(records[0][0]) + len(records[0][1]) - 1
        assert result.flags[boundary] == (
            tr.TokenFlag.EXACT | tr.TokenFlag.STOP
            if historical == "y"
            else tr.TokenFlag.EXACT
        )
        assert math.isnan(result.logprobs[boundary])
    assert trajectory.model_dump() == before


@pytest.mark.parametrize("context", [None, [1, 2], (1, 2)])
def test_optional_initial_role_context_declines_before_callbacks(context):
    trajectory, tokenizer, prompt, exchange = _case(
        "Plain historical assistant content"
    )
    if context is not None:
        assert exchange.request.get("chat_template_kwargs") is not None
        exchange.request["chat_template_kwargs"]["unused"] = context
    before = trajectory.model_dump()
    tokenizer.calls.clear()
    value = trajectory.tokenize(tokenizer=tokenizer, multi_history=True).histories[0]
    assert value.tokens == prompt + _extras(exchange)["token_ids"]
    assert value.logprobs[len(prompt) :] == [-0.5] * (len(value.tokens) - len(prompt))
    if isinstance(context, tuple):
        # Preserve the maintained native fallback's unclassified initial roles;
        # an opaque optional proof must neither guess roles nor invoke callbacks.
        assert not tokenizer.calls
        assert value.flags[: len(prompt)] == [tr.TokenFlag.EXACT] * len(prompt)
    else:
        assert any(flag & tr.TokenFlag.ASSISTANT for flag in value.flags[: len(prompt)])
    assert trajectory.model_dump() == before


def test_unsupported_interior_role_context_keeps_native_unknown_roles():
    trajectory, tokenizer, _ = _interior_historical_case()
    for exchange in trajectory.exchanges.chat_completions:
        extra = exchange.response.choices[0].model_extra
        assert extra is not None
        extra["stop_reason"] = ord("§")
    trajectory.exchanges.chat_completions[-1].request["chat_template_kwargs"] = {
        "unused": (1, 2)
    }
    before = trajectory.model_dump()
    result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    expected = trajectory.tokenize(multi_history=True)
    actual = result.histories[0]
    assert actual.tokens == expected.histories[0].tokens
    assert actual.flags == expected.histories[0].flags
    assert all(
        a == b or math.isnan(a) and math.isnan(b)
        for a, b in zip(actual.logprobs, expected.histories[0].logprobs, strict=True)
    )
    assert tr.first_occurrence_masks(result.histories, where=tr.TokenFlag.SAMPLED) == (
        tr.first_occurrence_masks(expected.histories, where=tr.TokenFlag.SAMPLED)
    )
    assert trajectory.model_dump() == before


@pytest.mark.parametrize("route", ["history", "trajectory", "trace"])
@pytest.mark.parametrize(
    "case", ["leading-mismatch", "later-mismatch", "no-loader", "leading-empty"]
)
def test_optional_leading_roles_preserve_usable_native_history(
    monkeypatch: pytest.MonkeyPatch, case: str, route: str
) -> None:
    from jinja2 import UndefinedError
    from test_tokenize import _CharacterTemplateTokenizer, _chat_exchange

    class LeadingTokenizer(_CharacterTemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if case == "leading-empty" and not messages:
                raise UndefinedError("Public template requires messages[0]")
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = LeadingTokenizer()
    messages = [
        {"role": "assistant", "content": "few-shot"},
        {"role": "user", "content": "query"},
    ]
    if case == "later-mismatch":
        messages.insert(0, {"role": "user", "content": "earlier query"})
    prompt = (
        cast(
            list[int],
            tokenizer.apply_chat_template(messages, add_generation_prompt=True),
        )
        if case == "leading-empty"
        else [1]
    )
    exchange = _chat_exchange(prompt, [2, 9])
    exchange.request["messages"] = cast(list[ChatCompletionMessageParam], messages)
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    )
    before = trajectory.model_dump(mode="python")
    loads = []

    def forbidden_load(*args, **kwargs):
        loads.append(args)
        raise AssertionError("Optional leading roles must not load a tokenizer")

    monkeypatch.setattr(module, "_tokenizer_config", forbidden_load)
    options: dict[str, Any] = (
        {"base_model": "public/no-download"}
        if case == "no-loader"
        else {"tokenizer": tokenizer}
    )
    if route == "history":
        value = trajectory.chat_completions_history().tokenize(**options)
    elif route == "trajectory":
        value = trajectory.tokenize(multi_history=True, **options).histories[0]
    else:
        value = module._tokenize_trajectory_with_trace(trajectory, **options)[
            0
        ].histories[0]
    assert value.tokens == prompt + [2, 9]
    assert value.logprobs[-2:] == [-0.2, -0.9]
    assert value.flags[: len(prompt)] == [tr.TokenFlag.EXACT] * len(prompt)
    assert all(math.isnan(item) for item in value.logprobs[: len(prompt)])
    assert all(item & tr.TokenFlag.SAMPLED for item in value.flags[-2:])
    assert not loads
    assert trajectory.model_dump(mode="python") == before


@pytest.mark.parametrize("change", ["message", "kwargs"])
def test_edited_history_retains_successful_rendered_roles(change: str) -> None:
    trajectory, tokenizer, _ = _interior_historical_case()
    trajectory.exchanges.chat_completions[-1].request["messages"][3]["content"] = "y"
    history = trajectory.chat_completions_history()
    if change == "message":
        history.messages[0]["content"] = "edited public query"
        history.message_sources[0] = None
    else:
        history.chat_template_kwargs = {"public_unused_setting": True}
    before = history.model_dump(mode="python")
    expected = history.tokenize(
        tokenizer=tokenizer, chat_template=history.chat_template
    )
    actual = history.tokenize(tokenizer=tokenizer)
    assert actual.tokens == expected.tokens
    assert actual.flags == expected.flags
    assert all(
        left == right or math.isnan(left) and math.isnan(right)
        for left, right in zip(actual.logprobs, expected.logprobs, strict=True)
    )
    assert history.model_dump(mode="python") == before


@pytest.mark.parametrize("change", ["stop", "source-list"])
@pytest.mark.parametrize("outcome", ["unchanged", "success", "decline", "fatal"])
def test_role_proof_guards_consumed_source_identity_and_stop(
    monkeypatch: pytest.MonkeyPatch, change: str, outcome: str
) -> None:
    trajectory, tokenizer, prompt, exchange = _case("Plain historical content")
    exchange.response.choices[0].finish_reason = "stop"
    extra = _extras(exchange)
    extra["stop_reason"] = extra["token_ids"][-1]
    history = trajectory.chat_completions_history()
    original = type(tokenizer).__call__
    calls = []
    fatal = RuntimeError("Public historical encoder failed")

    def observed(self, text, **kwargs):
        value = original(self, text, **kwargs)
        calls.append(text)
        if outcome != "unchanged":
            if change == "stop":
                extra["stop_reason"] = 999999
            else:
                history.message_sources[-1] = deepcopy(history.message_sources[-1])
            if outcome == "fatal":
                raise fatal
            if outcome == "decline":
                raise TypeError("Public encoder does not support offsets")
        return value

    monkeypatch.setattr(type(tokenizer), "__call__", observed)
    if outcome == "fatal":
        with pytest.raises(RuntimeError) as raised:
            history.tokenize(tokenizer=tokenizer)
        assert raised.value is fatal
    elif outcome != "unchanged":
        with pytest.raises(ValueError, match="Sampled source changed"):
            history.tokenize(tokenizer=tokenizer)
    else:
        actual = history.tokenize(tokenizer=tokenizer)
        assert actual.tokens == prompt + extra["token_ids"]
        assert actual.flags[-1] & tr.TokenFlag.STOP
        assert actual.logprobs[-len(extra["token_ids"]) :] == [-0.5] * len(
            extra["token_ids"]
        )
    assert len(calls) == 1


@pytest.mark.parametrize("route", ["history", "trajectory", "trace"])
@pytest.mark.parametrize("proof", ["no-offsets", "mismatched-template", "no-loader"])
def test_exact_native_interior_roles_are_optional(monkeypatch, route, proof):
    trajectory, tokenizer, records = _interior_historical_case(
        offsets=proof != "no-offsets"
    )
    for exchange in trajectory.exchanges.chat_completions:
        extra = exchange.response.choices[0].model_extra
        assert extra is not None
        extra["stop_reason"] = ord("§")
    expected = trajectory.tokenize(multi_history=True).histories[0]
    if proof == "mismatched-template":
        # The supplied current renderer may differ from the server's renderer.
        apply = tokenizer.apply_chat_template

        def changed(messages, **kwargs):
            rendered = apply(messages, **kwargs)
            return rendered + ("different" if isinstance(rendered, str) else [99])

        monkeypatch.setattr(tokenizer, "apply_chat_template", changed)
    options: dict[str, Any]
    if proof == "no-loader":

        def no_load(*args, **kwargs):
            raise AssertionError("Optional roles attempted a model/tokenizer load")

        monkeypatch.setattr(module, "_tokenizer_config", no_load)
        monkeypatch.setattr(module, "_load_tokenizer", no_load)
        options = {"base_model": "public/no-role-loader"}
    else:
        options = {"tokenizer": tokenizer}
    before = trajectory.model_dump()
    if route == "history":
        actual = trajectory.chat_completions_history().tokenize(**options)
    elif route == "trajectory":
        actual = trajectory.tokenize(multi_history=True, **options).histories[0]
    else:
        actual = module._tokenize_trajectory_with_trace(trajectory, **options)[
            0
        ].histories[0]
    assert actual.tokens == records[-1][0] + records[-1][1] == expected.tokens
    assert actual.flags == expected.flags
    assert all(
        a == b or math.isnan(a) and math.isnan(b)
        for a, b in zip(actual.logprobs, expected.logprobs, strict=True)
    )
    for where in (tr.TokenFlag.SAMPLED, tr.TokenFlag.OUTPUT, tr.TokenFlag.ASSISTANT):
        assert tr.first_occurrence_masks([actual], where=where) == (
            tr.first_occurrence_masks([expected], where=where)
        )
    assert trajectory.model_dump() == before
