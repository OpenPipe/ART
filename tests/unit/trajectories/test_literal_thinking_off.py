from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, cast

from jinja2.sandbox import ImmutableSandboxedEnvironment
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import pytest

import art.trajectories as tr
from art.trajectories import _tokenize

# Public Qwen3.5 template after ART's existing thinking-preservation rewrite.
_TEMPLATE = (
    Path(__file__).parents[2] / "fixtures/qwen35_preserved_thinking.jinja"
).read_text()
_LITERAL = "HEAD\n</think>DISCARDED_PUBLIC_SEGMENT</think>\n\nTAIL"


class _TemplateTokenizer:
    chat_template = _TEMPLATE

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.rendered: list[str] = []
        self.env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"input_ids": list(map(ord, text))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return "".join(map(chr, token_ids))

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> str | list[int]:
        self.calls.append(deepcopy(messages))
        text = self.env.from_string(chat_template or self.chat_template).render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        self.rendered.append(text)
        return list(map(ord, text)) if tokenize else text


def _history(
    *,
    thinking: bool | None = False,
    content: str = _LITERAL,
    reasoning: str | None = None,
    reasoning_field: str = "reasoning_content",
) -> tuple[tr.ChatCompletionsHistory, _TemplateTokenizer]:
    tokenizer = _TemplateTokenizer()
    request_kwargs: dict[str, Any] = {"preserve_thinking": True}
    if thinking is not None:
        request_kwargs["enable_thinking"] = thinking
    prompt_messages = [{"role": "user", "content": "Public query."}]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, **request_kwargs
    )
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message[reasoning_field] = reasoning
    output = list(map(ord, content))
    response = ChatCompletion.model_validate(
        {
            "id": "public-repro",
            "object": "chat.completion",
            "created": 0,
            "model": "public/qwen35",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": message,
                    "prompt_token_ids": prompt,
                    "token_ids": output,
                    "logprobs": {
                        "content": [
                            {
                                "token": f"token_id:{token}",
                                "logprob": -0.5,
                                "bytes": [],
                                "top_logprobs": [],
                            }
                            for token in output
                        ]
                    },
                }
            ],
        }
    )
    exchange = tr.ChatCompletionsExchange(
        request=tr.ChatCompletionsRequest(
            model="public/qwen35",
            messages=cast(list[ChatCompletionMessageParam], prompt_messages),
            chat_template=_TEMPLATE,
            chat_template_kwargs=request_kwargs,
        ),
        response=response,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    history = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    ).chat_completions_history()
    tokenizer.calls.clear()
    tokenizer.rendered.clear()
    return history, tokenizer


def _outcome(
    history: tr.ChatCompletionsHistory, tokenizer: _TemplateTokenizer
) -> object:
    try:
        value = history.tokenize(tokenizer=tokenizer)
    except ValueError as error:
        return type(error), str(error)
    return value.tokens, value.flags, [None if x != x else x for x in value.logprobs]


@pytest.mark.parametrize(
    "content", [_LITERAL, "literal </think> text", "π<think>one<think>two</think>end"]
)
def test_native_thinking_off_retains_literal_content(
    content: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    history, tokenizer = _history(content=content)
    original = history.model_dump(mode="python")
    # The pre-fix history path misrenders literal content even when later native
    # token splicing can recover the terminal output.
    with monkeypatch.context() as patch:
        patch.setattr(
            _tokenize, "_preserve_literal_thinking_off_content", lambda *args: None
        )
        patch.setattr(
            _tokenize, "chat_template_with_preserved_thinking", lambda value: value
        )
        _outcome(history, tokenizer)
    assert content not in tokenizer.rendered[0]
    tokenizer.calls.clear()
    tokenizer.rendered.clear()
    tokenized = history.tokenize(tokenizer=tokenizer)
    assert content in tokenizer.rendered[0]
    sampled = [
        i for i, flag in enumerate(tokenized.flags) if flag & tr.TokenFlag.SAMPLED
    ]
    assert "".join(chr(tokenized.tokens[i]) for i in sampled) == content
    assert all(tokenized.logprobs[i] == -0.5 for i in sampled)
    required = tr.TokenFlag.EXACT | tr.TokenFlag.ASSISTANT | tr.TokenFlag.OUTPUT
    assert all(tokenized.flags[i] & required == required for i in sampled)
    assert not any(flag & tr.TokenFlag.STOP for flag in tokenized.flags)
    assert tokenizer.calls[0][-1].get("reasoning_content", "") == ""
    assert tokenizer.calls[0][-1]["content"] == content
    assert history.model_dump(mode="python") == original


@pytest.mark.parametrize(
    "case",
    [
        "source_on",
        "source_unknown",
        "effective_on",
        "preserve_off",
        "other_template",
        "no_source",
        "request_source",
        "no_native_prompt",
        "structured",
        "alias",
        "visible_only",
    ],
)
def test_unrelated_histories_keep_original_rendering(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    history, tokenizer = _history(
        thinking=True
        if case == "source_on"
        else None
        if case == "source_unknown"
        else False,
        reasoning="explicit reasoning"
        if case in {"structured", "alias", "visible_only"}
        else None,
        reasoning_field="reasoning" if case == "alias" else "reasoning_content",
    )
    assert history.chat_template_kwargs is not None
    history.chat_template_kwargs["enable_thinking"] = case == "effective_on"
    if case == "preserve_off":
        history.chat_template_kwargs["preserve_thinking"] = False
    if case == "other_template":
        history.chat_template = _TEMPLATE + "{# different template #}"
    if case == "no_source":
        history.message_sources[-1] = None
    source = history.message_sources[-1]
    if case == "request_source":
        assert source is not None
        assert isinstance(source.exchange, tr.ChatCompletionsExchange)
        source.exchange.request["messages"].append(deepcopy(history.messages[-1]))
        history.message_sources[-1] = tr.ChatCompletionsMessageSource(
            exchange=source.exchange, request_index=1
        )
    if case == "no_native_prompt":
        assert source is not None
        assert isinstance(source.exchange, tr.ChatCompletionsExchange)
        extra = source.exchange.response.choices[0].model_extra
        assert extra is not None
        extra.pop("prompt_token_ids")
    if case == "visible_only":
        cast(dict[str, Any], history.messages[-1]).pop("reasoning")
    original = history.model_dump(mode="python")
    candidate = _outcome(history, tokenizer)
    calls = deepcopy(tokenizer.calls)
    tokenizer.calls.clear()
    with monkeypatch.context() as patch:
        patch.setattr(
            _tokenize, "_preserve_literal_thinking_off_content", lambda *args: None
        )
        baseline = _outcome(history, tokenizer)
    assert candidate == baseline
    assert len(calls) == len(tokenizer.calls)
    assert calls[0] == tokenizer.calls[0]
    assert history.model_dump(mode="python") == original


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_explicit_empty_reasoning_is_preserved(field: str) -> None:
    history, tokenizer = _history(reasoning="", reasoning_field=field)
    original = history.model_dump(mode="python")
    tokenized = history.tokenize(tokenizer=tokenizer)
    assert (
        "".join(
            chr(token)
            for token, flag in zip(tokenized.tokens, tokenized.flags, strict=True)
            if flag & tr.TokenFlag.SAMPLED
        )
        == _LITERAL
    )
    assert tokenizer.calls[0][-1].get("reasoning_content", "") == ""
    assert history.model_dump(mode="python") == original


def test_literal_adaptation_does_not_relax_source_validation() -> None:
    history, tokenizer = _history()
    cast(dict[str, Any], history.messages[-1])["content"] += "not in source"
    with pytest.raises(ValueError, match="source"):
        history.tokenize(tokenizer=tokenizer)
    assert not tokenizer.calls


@pytest.mark.parametrize("earlier_thinking", [True, None])
def test_mixed_history_uses_each_generations_own_request(
    earlier_thinking: bool | None,
) -> None:
    first, tokenizer = _history(thinking=earlier_thinking)
    second, _ = _history()
    history = tr.ChatCompletionsHistory(
        model=second.model,
        messages=[*first.messages, *second.messages],
        message_sources=[*first.message_sources, *second.message_sources],
        chat_template=second.chat_template,
        chat_template_kwargs=second.chat_template_kwargs,
    )
    original = history.model_dump(mode="python")
    _outcome(history, tokenizer)
    rendered_messages = tokenizer.calls[0]
    assert "reasoning_content" not in rendered_messages[1]
    assert "reasoning_content" not in rendered_messages[3]
    assert _LITERAL in tokenizer.rendered[0]
    assert [message["content"] for message in rendered_messages] == [
        message["content"] for message in history.messages
    ]
    assert history.model_dump(mode="python") == original


class _NewlineRunTokenizer(_TemplateTokenizer):
    """Reversible public codec that exposes the open/closed scaffold boundary."""

    all_special_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    all_special_ids = [200000, 200001, 200002, 200003]
    eos_token_id = 200001

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        pieces = list(
            re.finditer(r"<\|im_start\|>|<\|im_end\|>|<think>|</think>|\n+|[^\n]", text)
        )
        result = {
            "input_ids": [
                self.all_special_ids[self.all_special_tokens.index(piece.group())]
                if piece.group() in self.all_special_tokens
                else 300000 + len(piece.group())
                if piece.group().startswith("\n")
                else ord(piece.group())
                for piece in pieces
            ]
        }
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [piece.span() for piece in pieces]
        return result

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        return "".join(
            self.all_special_tokens[self.all_special_ids.index(token)]
            if token in self.all_special_ids
            else "\n" * (token - 300000)
            if token > 300000
            else chr(token)
            for token in token_ids
        )

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return (
            self.all_special_ids[self.all_special_tokens.index(token)]
            if token in self.all_special_tokens
            else None
        )

    def apply_chat_template(
        self, messages: list[dict[str, Any]], *, tokenize: bool = True, **kwargs: Any
    ) -> str | list[int]:
        text = super().apply_chat_template(messages, tokenize=False, **kwargs)
        assert isinstance(text, str)
        return self(text)["input_ids"] if tokenize else text


def test_literal_next_turn_preserves_preceding_length_stop_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _NewlineRunTokenizer()
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "unchanged preceding output"},
        {"role": "user", "content": "next query"},
        {"role": "assistant", "content": _LITERAL},
    ]
    exchanges = []
    for index in (1, 3):
        single, _ = _history(content=messages[index]["content"])
        source = single.message_sources[-1]
        assert source is not None and isinstance(
            source.exchange, tr.ChatCompletionsExchange
        )
        exchange = source.exchange
        exchange.request["messages"] = cast(
            list[ChatCompletionMessageParam], deepcopy(messages[:index])
        )
        data = exchange.response.model_dump(mode="python")
        data["id"] = f"public-length-{index}"
        choice = data["choices"][0]
        choice["prompt_token_ids"] = tokenizer.apply_chat_template(
            messages[:index],
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=True,
        )
        choice["token_ids"] = tokenizer(messages[index]["content"])["input_ids"]
        choice["logprobs"]["content"] = [
            {
                "token": f"token_id:{token}",
                "logprob": -0.5,
                "bytes": [],
                "top_logprobs": [],
            }
            for token in choice["token_ids"]
        ]
        exchange.response = ChatCompletion.model_validate(data)
        exchanges.append(exchange)
    history = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=exchanges)
    ).chat_completions_history()
    original = history.model_dump(mode="python")
    first = exchanges[0].response.choices[0].model_extra
    last = exchanges[1].response.choices[0].model_extra
    assert first is not None and last is not None
    end = len(first["prompt_token_ids"]) + len(first["token_ids"])
    native_boundary = last["prompt_token_ids"][end:]
    assert (
        last["prompt_token_ids"][:end] == first["prompt_token_ids"] + first["token_ids"]
    )
    source = history.message_sources[1]
    key = _tokenize._sampled_source_key(source)
    builder = _tokenize._tokenize_exact_projected_chat_history
    observed = []

    def observe(*args: Any, **kwargs: Any) -> tr.TokenizedHistory | None:
        result = builder(*args, **kwargs)
        if boundary := kwargs.get("length_stop_boundaries", {}).get(key):
            observed.append((boundary, result))
        return result

    monkeypatch.setattr(_tokenize, "_tokenize_exact_projected_chat_history", observe)
    with monkeypatch.context() as patch:
        patch.setattr(
            _tokenize, "_preserve_literal_thinking_off_content", lambda *args: None
        )
        patch.setattr(
            _tokenize, "chat_template_with_preserved_thinking", lambda value: value
        )
        _outcome(history, tokenizer)
    boundary, old_exact = observed[0]
    stored = list(boundary.tail + boundary.following)
    assert old_exact is None
    assert len(native_boundary) - len(stored) == 2
    assert stored[:-1] == native_boundary[:-3]
    assert tokenizer.decode(stored[-1:]) == "\n"
    assert tokenizer.decode(native_boundary[-3:]) == "\n\n</think>\n\n"
    observed.clear()

    value = history.tokenize(tokenizer=tokenizer)
    fixed_boundary, fixed_exact = observed[0]
    assert fixed_exact is value
    assert list(fixed_boundary.tail + fixed_boundary.following) == native_boundary
    assert (
        value.tokens[: len(last["prompt_token_ids"]) + len(last["token_ids"])]
        == last["prompt_token_ids"] + last["token_ids"]
    )
    assert value.flags[end] == tr.TokenFlag.EXACT | tr.TokenFlag.STOP
    assert not value.flags[end] & tr.TokenFlag.SAMPLED
    for exchange in exchanges:
        extra = exchange.response.choices[0].model_extra
        assert extra is not None
        start = len(extra["prompt_token_ids"])
        stop = start + len(extra["token_ids"])
        assert value.tokens[start:stop] == extra["token_ids"]
        assert value.logprobs[start:stop] == [-0.5] * (stop - start)
        assert all(
            flag
            == tr.TokenFlag.EXACT
            | tr.TokenFlag.SAMPLED
            | tr.TokenFlag.ASSISTANT
            | tr.TokenFlag.OUTPUT
            for flag in value.flags[start:stop]
        )
    assert history.model_dump(mode="python") == original
