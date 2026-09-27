from copy import deepcopy
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam
import pytest
from test_literal_thinking_off import _TEMPLATE, _history

import art.trajectories as tr
from art.trajectories import _tokenize


@pytest.mark.parametrize("require_identity", [False, True])
def test_historical_role_proof_nested_options_identity(require_identity: bool) -> None:
    history, tokenizer = _history(content="New recorded answer")
    source = history.message_sources[-1]
    assert source is not None
    exchange = source.exchange
    assert isinstance(exchange, tr.ChatCompletionsExchange)
    messages = [
        {"role": "user", "content": "Old public query"},
        {"role": "assistant", "content": "Public reasoning</think>Public answer"},
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
    extra = exchange.response.choices[0].model_extra
    assert extra is not None
    extra["prompt_token_ids"] = prompt
    options = {"state": ["unchanged"]}
    kwargs = exchange.request["chat_template_kwargs"]
    assert isinstance(kwargs, dict)
    kwargs["options"] = options
    value = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    value = value.chat_completions_history()
    assert value.chat_template_kwargs is not None
    options = value.chat_template_kwargs["options"]
    original = value.model_dump_json()
    ordinary = tokenizer.apply_chat_template
    seen = []

    def render(messages, **kwargs: Any):
        same = kwargs.get("options") is options
        seen.append((kwargs.get("chat_template") == _TEMPLATE, same))
        if require_identity and not same:
            raise ValueError("nested option identity changed")
        return ordinary(messages, **kwargs)

    tokenizer.apply_chat_template = cast(Any, render)
    result = value.tokenize(tokenizer=tokenizer)
    assert result.tokens[: len(prompt)] == prompt
    assert (
        result.tokens[len(prompt) : len(prompt) + len(extra["token_ids"])]
        == extra["token_ids"]
    )
    assert value.model_dump_json() == original
    assert seen and all(same for _, same in seen)


@pytest.mark.parametrize("masks", [False, True])
@pytest.mark.parametrize("mutate", [False, True])
def test_historical_helpers_preserve_kwargs_identity_and_checks(
    masks: bool, mutate: bool
) -> None:
    messages = [{"role": "assistant", "content": "x"}, {"role": "user", "content": "q"}]
    tools = [{"type": "function", "function": {"name": "public_tool"}}]
    options = {"state": ["unchanged"]}
    calls = []

    class Tokenizer:
        def apply_chat_template(self, selected, *, tokenize=True, **kwargs):
            assert selected is not messages
            assert kwargs["tools"] is not tools
            assert kwargs["options"] is options
            calls.append(True)
            if mutate:
                options["state"][0] = "changed"
            text = "".join(message["content"] for message in selected)
            return list(map(ord, text)) if tokenize else text

        def __call__(self, text, **kwargs):
            return {
                "input_ids": list(map(ord, text)),
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }

    def invoke():
        kwargs: dict[str, Any] = dict(
            tokenizer=Tokenizer(),
            template="custom",
            tools=tools,
            kwargs={"options": options},
        )
        if masks:
            return _tokenize._recorded_prompt_role_masks(
                messages, [None, None], [120, 113], **kwargs
            )
        return _tokenize._recorded_prompt_tokens(messages, **kwargs)

    if mutate:
        with pytest.raises(ValueError, match="Renderer changed context"):
            invoke()
    else:
        result = invoke()
        assert result == (([True, False], [False, False]) if masks else [120, 113])
    assert calls
