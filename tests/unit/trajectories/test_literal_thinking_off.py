from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
import random
import re
from typing import Any, cast

from jinja2.sandbox import ImmutableSandboxedEnvironment
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.responses import Response
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
    template: str = _TEMPLATE,
) -> tuple[tr.ChatCompletionsHistory, _TemplateTokenizer]:
    tokenizer = _TemplateTokenizer()
    tokenizer.chat_template = template
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
            chat_template=template,
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
) -> tuple[Any, ...]:
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
        _outcome(history, tokenizer)
    assert content not in tokenizer.rendered[0]
    tokenizer.calls.clear()
    tokenizer.rendered.clear()
    tokenized = history.tokenize(tokenizer=tokenizer)
    # The behavioral proof probes the original view before selecting the
    # literal render copy. Its first completed candidate and later real render
    # must both retain the content.
    assert content in tokenizer.rendered[4]
    assert content in tokenizer.rendered[5]
    sampled = [
        i for i, flag in enumerate(tokenized.flags) if flag & tr.TokenFlag.SAMPLED
    ]
    assert "".join(chr(tokenized.tokens[i]) for i in sampled) == content
    assert all(tokenized.logprobs[i] == -0.5 for i in sampled)
    required = tr.TokenFlag.EXACT | tr.TokenFlag.ASSISTANT | tr.TokenFlag.OUTPUT
    assert all(tokenized.flags[i] & required == required for i in sampled)
    assert not any(flag & tr.TokenFlag.STOP for flag in tokenized.flags)
    assert tokenizer.calls[5][-1]["reasoning_content"] == ""
    assert tokenizer.calls[5][-1]["content"] == content
    assert history.model_dump(mode="python") == original


@pytest.mark.parametrize(
    "case",
    [
        "source_on",
        "source_unknown",
        "effective_on",
        "preserve_off",
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
    assert any(call[-1].get("reasoning_content") == "" for call in tokenizer.calls)
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
    rendered_messages = next(
        call
        for call in tokenizer.calls
        if len(call) == 4
        and call[-1].get("reasoning_content") == ""
        and call[-1]["content"] == _LITERAL
    )
    assert "reasoning_content" not in rendered_messages[1]
    assert rendered_messages[3]["reasoning_content"] == ""
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


@pytest.mark.parametrize("alias", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize(
    "tags", [("<think>", "</think>"), ("<analysis>", "</analysis>")]
)
def test_literal_contract_is_independent_of_template_hash_and_delimiters(
    alias: str, tags: tuple[str, str]
) -> None:
    template = (
        (_TEMPLATE + "{# semantically equivalent template revision #}")
        .replace("message.reasoning_content", f"message.{alias}")
        .replace("<think>", tags[0])
        .replace("</think>", tags[1])
    )
    # Public literal strings include nested, repeated, Unicode and whitespace
    # cases. Neither the content nor the delimiter spelling is an allowlist.
    rng = random.Random(0)
    for middle in [
        "",
        "\n",
        " π ",
        tags[0],
        tags[1],
        *["".join(rng.choices("ab\n Ω", k=8)) for _ in range(5)],
    ]:
        content = "head" + tags[1] + middle + tags[1] + "tail"
        history, tokenizer = _history(content=content, template=template)
        original = history.model_dump(mode="python")
        value = history.tokenize(tokenizer=tokenizer)
        source = history.message_sources[-1]
        assert source is not None and isinstance(
            source.exchange, tr.ChatCompletionsExchange
        )
        prompt, output, logprobs = _tokenize._chat_choice_tokens(
            source.exchange.response.choices[0], source.exchange.response
        )
        assert prompt is not None and output is not None
        assert value.tokens == prompt + output
        assert value.logprobs[len(prompt) :] == logprobs
        assert value.flags[len(prompt) :] == [
            tr.TokenFlag.EXACT
            | tr.TokenFlag.SAMPLED
            | tr.TokenFlag.ASSISTANT
            | tr.TokenFlag.OUTPUT
        ] * len(output)
        assert history.model_dump(mode="python") == original


@pytest.mark.parametrize("probe", ["decode_kwargs", "text", "empty", "scaffold"])
def test_unsupported_literal_proof_keeps_original_path(
    probe: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    history, tokenizer = _history(
        content=_LITERAL if probe != "text" else "ordinary public content"
    )
    decode, render = tokenizer.decode, tokenizer.apply_chat_template
    branches: list[str] = []

    def guarded_decode(ids: list[int], **kwargs: Any) -> str:
        if probe == "decode_kwargs" and kwargs:
            branches.append(probe)
            raise TypeError("unsupported decode keyword")
        return decode(ids, **kwargs)

    def guarded_render(
        messages: list[dict[str, Any]], **kwargs: Any
    ) -> str | list[int]:
        if probe == "text" and kwargs.get("tokenize") is False:
            branches.append(probe)
            raise TypeError("token-only template")
        if (
            probe == "empty"
            and messages
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("content")
        ):
            branches.append(probe)
            raise ValueError("empty assistant unsupported")
        value = render(messages, **kwargs)
        if probe == "scaffold" and any("reasoning_content" in m for m in messages):
            branches.append(probe)
            return "changed" + value if isinstance(value, str) else [999] + value
        return value

    monkeypatch.setattr(tokenizer, "decode", guarded_decode)
    monkeypatch.setattr(tokenizer, "apply_chat_template", guarded_render)
    candidate = _outcome(history, tokenizer)
    assert probe in branches
    with monkeypatch.context() as patch:
        patch.setattr(
            _tokenize, "_preserve_literal_thinking_off_content", lambda *args: False
        )
        baseline = _outcome(history, tokenizer)
    assert candidate == baseline
    if probe in {"decode_kwargs", "text"}:
        assert not isinstance(candidate[0], type)


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_literal_probe_does_not_swallow_control_exceptions(
    error: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    history, tokenizer = _history()

    def interrupt(*args: Any, **kwargs: Any) -> str:
        raise error()

    monkeypatch.setattr(tokenizer, "decode", interrupt)
    with pytest.raises(error):
        history.tokenize(tokenizer=tokenizer)


@pytest.mark.parametrize(
    "case",
    [
        "missing_output",
        "different_output",
        "nonempty_reasoning",
        "none_reasoning",
        "empty_reasoning",
    ],
)
def test_native_evidence_and_reasoning_are_independent_authorities(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    history, tokenizer = _history(
        reasoning="structured"
        if case == "nonempty_reasoning"
        else ""
        if case == "empty_reasoning"
        else None
    )
    source = history.message_sources[-1]
    assert source is not None and isinstance(
        source.exchange, tr.ChatCompletionsExchange
    )
    choice = source.exchange.response.choices[0]
    assert choice.model_extra is not None
    if case == "missing_output":
        choice.model_extra.pop("token_ids")
        choice.logprobs = None
    elif case == "different_output":
        choice.model_extra["token_ids"] = [12345]
        choice.logprobs = None
    elif case == "none_reasoning":
        assert choice.message.model_extra is not None
        choice.message.model_extra["reasoning_content"] = None
    original = history.model_dump(mode="python")
    value = _outcome(history, tokenizer)
    if case in {"none_reasoning", "empty_reasoning"}:
        assert not isinstance(value[0], type)
        assert any(call[-1].get("reasoning_content") == "" for call in tokenizer.calls)
    else:
        assert not any(
            call[-1].get("reasoning_content") == "" for call in tokenizer.calls
        )
        with monkeypatch.context() as patch:
            patch.setattr(
                _tokenize, "_preserve_literal_thinking_off_content", lambda *args: False
            )
            assert _outcome(history, tokenizer) == value
    assert history.model_dump(mode="python") == original


def _two_turn_literal_history(
    *, served_literal: bool
) -> tuple[tr.ChatCompletionsHistory, _TemplateTokenizer]:
    first, tokenizer = _history()
    first_source = first.message_sources[-1]
    assert first_source is not None and isinstance(
        first_source.exchange, tr.ChatCompletionsExchange
    )
    second, _ = _history(content="second answer")
    source = second.message_sources[-1]
    assert source is not None and isinstance(
        source.exchange, tr.ChatCompletionsExchange
    )
    exchange = source.exchange
    messages = [
        *[dict(m) for m in deepcopy(first.messages)],
        {"role": "user", "content": "follow up"},
    ]
    exchange.request["messages"] = cast(
        list[ChatCompletionMessageParam], deepcopy(messages)
    )
    if served_literal:
        messages[1]["reasoning_content"] = ""
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=True,
    )
    assert isinstance(prompt, list)
    exchange.response.id = "second-source"
    assert exchange.response.choices[0].model_extra is not None
    exchange.response.choices[0].model_extra["prompt_token_ids"] = prompt
    # Explicit History construction also exercises inconsistent native chains
    # which automatic history grouping would split into separate histories.
    history = tr.ChatCompletionsHistory(
        model=first.model,
        messages=[
            *first.messages,
            {"role": "user", "content": "follow up"},
            second.messages[-1],
        ],
        message_sources=[*first.message_sources, None, source],
        chat_template=first.chat_template,
        chat_template_kwargs=first.chat_template_kwargs,
    )
    tokenizer.calls.clear()
    tokenizer.rendered.clear()
    return history, tokenizer


@pytest.mark.parametrize(
    "case",
    [
        "exact",
        "lossy_later_prompt",
        "missing_later_prompt",
        "later_logprobs",
        "later_owner",
        "later_stop",
    ],
)
def test_adaptation_proves_every_turn_not_only_the_repaired_message(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    history, tokenizer = _two_turn_literal_history(
        served_literal=case != "lossy_later_prompt"
    )
    source = history.message_sources[-1]
    assert source is not None and isinstance(
        source.exchange, tr.ChatCompletionsExchange
    )
    if case == "missing_later_prompt":
        assert source.exchange.response.choices[0].model_extra is not None
        source.exchange.response.choices[0].model_extra.pop("prompt_token_ids")
    original = history.model_dump(mode="python")
    validate = _tokenize._require_native_render_conditioning
    extra = source.exchange.response.choices[0].model_extra
    assert extra is not None
    last_sampled = len(extra.get("prompt_token_ids", [])) + len(extra["token_ids"]) - 1

    def corrupt(h: Any, value: Any, trace: Any) -> Any:
        if case == "later_logprobs":
            value.logprobs[last_sampled] = -0.75
        elif case == "later_owner":
            trace.trace.source_keys[last_sampled] = trace.trace.source_keys[0]
        elif case == "later_stop":
            value.flags[last_sampled] |= tr.TokenFlag.STOP
        return validate(h, value, trace)

    monkeypatch.setattr(_tokenize, "_require_native_render_conditioning", corrupt)
    if case == "missing_later_prompt":
        candidate = _outcome(history, tokenizer)
        with monkeypatch.context() as patch:
            patch.setattr(
                _tokenize, "_preserve_literal_thinking_off_content", lambda *args: []
            )
            assert _outcome(history, tokenizer) == candidate
    elif case == "exact":
        value = history.tokenize(tokenizer=tokenizer)
        assert sum(bool(flag & tr.TokenFlag.SAMPLED) for flag in value.flags) == len(
            _LITERAL
        ) + len("second answer")
    else:
        with pytest.raises(ValueError, match="Literal render adaptation"):
            history.tokenize(tokenizer=tokenizer)
    assert history.model_dump(mode="python") == original


def test_behavioral_proof_has_a_fixed_render_budget() -> None:
    history, tokenizer = _history()
    messages = [dict(message) for message in history.messages]

    def render(
        selected_messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> str:
        value = tokenizer.apply_chat_template(
            selected_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
            preserve_thinking=True,
        )
        assert isinstance(value, str)
        return value

    assert _tokenize._preserve_literal_thinking_off_content(
        history,
        messages,
        {"enable_thinking": False, "preserve_thinking": True},
        tokenizer,
        render,
    )
    assert len(tokenizer.calls) == 5
    assert history.messages[-1].get("reasoning_content") is None


@pytest.mark.parametrize("partial", [False, True])
def test_mixed_responses_source_requires_complete_native_inventory(
    partial: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, tokenizer = _history()
    request_messages = [
        *[dict(m) for m in deepcopy(first.messages)],
        {"role": "user", "content": "next"},
    ]
    served = deepcopy(request_messages)
    served[1]["reasoning_content"] = ""
    prompt = tokenizer.apply_chat_template(
        served,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=True,
    )
    assert isinstance(prompt, list)
    texts = ["one", "two"] if partial else ["one"]
    output = list(map(ord, "".join(texts)))
    response = Response.model_validate(
        {
            "id": "mixed-responses",
            "created_at": 0,
            "model": "public/qwen35",
            "object": "response",
            "output": [
                {
                    "id": f"message-{i}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
                for i, text in enumerate(texts)
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "token_generations": [
                {
                    "prompt_token_ids": prompt,
                    "output_tokens": [
                        {"token_id": token, "logprob": -0.25} for token in output
                    ],
                    "output_indices": list(range(len(texts))),
                }
            ],
        }
    )
    exchange = tr.ResponsesExchange(
        request=tr.ResponsesRequest(model="public/qwen35", input="next"),
        response=response,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    projected = (
        tr.Trajectory(exchanges=tr.TrajectoryExchanges(responses=[exchange]))
        .responses_history()
        .as_chat_completions_history()
    )
    history = tr.ChatCompletionsHistory(
        model=first.model,
        messages=[*first.messages, *projected.messages],
        message_sources=[*first.message_sources, *projected.message_sources],
        chat_template=first.chat_template,
        chat_template_kwargs=first.chat_template_kwargs,
    )
    original = history.model_dump(mode="python")
    candidate = _outcome(history, tokenizer)
    if partial:
        with monkeypatch.context() as patch:
            patch.setattr(
                _tokenize, "_preserve_literal_thinking_off_content", lambda *args: []
            )
            assert _outcome(history, tokenizer) == candidate
    else:
        assert not isinstance(candidate[0], type)
        assert candidate[0][: len(prompt) + len(output)] == prompt + output
    assert history.model_dump(mode="python") == original


@pytest.mark.parametrize("finish", ["length", "stop"])
def test_literal_markers_do_not_own_sampled_stop_bits(finish: str) -> None:
    history, _ = _history()
    tokenizer = _NewlineRunTokenizer()
    source = history.message_sources[-1]
    assert source is not None and isinstance(
        source.exchange, tr.ChatCompletionsExchange
    )
    exchange = source.exchange
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    prompt = tokenizer.apply_chat_template(
        [dict(m) for m in exchange.request["messages"]],
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=True,
    )
    assert isinstance(prompt, list)
    output = tokenizer(_LITERAL)["input_ids"] + (
        [tokenizer.eos_token_id] if finish == "stop" else []
    )
    choice.update(prompt_token_ids=prompt, token_ids=output, finish_reason=finish)
    choice["logprobs"]["content"] = [
        {"token": f"token_id:{token}", "logprob": -0.5, "bytes": [], "top_logprobs": []}
        for token in output
    ]
    exchange.response = ChatCompletion.model_validate(data)
    original = history.model_dump(mode="python")
    value = history.tokenize(tokenizer=tokenizer)
    assert value.tokens[: len(prompt) + len(output)] == prompt + output
    sampled = value.flags[len(prompt) : len(prompt) + len(output)]
    assert [bool(flag & tr.TokenFlag.STOP) for flag in sampled] == [False] * (
        len(output) - (finish == "stop")
    ) + ([True] if finish == "stop" else [])
    assert value.logprobs[len(prompt) : len(prompt) + len(output)] == [-0.5] * len(
        output
    )
    assert history.model_dump(mode="python") == original


@pytest.mark.parametrize("limit", [0, 64, 8 << 20])
def test_literal_adaptation_reuses_only_unchanged_cache_prefixes(
    limit: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    history, tokenizer = _two_turn_literal_history(served_literal=True)
    expected = _outcome(history, tokenizer)
    # This fixture renderer is pure, but not stock HF. Exercise the admitted
    # integration explicitly; real HF eligibility is tested separately.
    monkeypatch.setattr(_tokenize, "cacheable_chat_template", lambda *args: True)
    monkeypatch.setattr(tokenizer, "special_tokens_map", {}, raising=False)
    monkeypatch.setattr(_tokenize._PrefixChatRenderCache, "_MAX_BYTES", limit)
    original = _tokenize._PrefixChatRenderCache.for_messages
    observed = []

    def check(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        observed.append(self)
        return result

    monkeypatch.setattr(_tokenize._PrefixChatRenderCache, "for_messages", check)
    assert _outcome(history, tokenizer) == expected
    assert observed and all(cache.bytes <= limit for cache in observed)


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
def test_earlier_structured_thinking_and_later_literal_content_remain_distinct(
    reasoning_field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, tokenizer = _history(
        thinking=True,
        content="reasoned answer",
        reasoning="explicit thought",
        reasoning_field=reasoning_field,
    )
    source = first.message_sources[-1]
    assert source is not None and isinstance(
        source.exchange, tr.ChatCompletionsExchange
    )
    first_exchange = source.exchange
    output = list(map(ord, "explicit thought</think>\n\nreasoned answer"))
    data = first_exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice["token_ids"] = output
    choice["logprobs"]["content"] = [
        {"token": f"token_id:{t}", "logprob": -0.5, "bytes": [], "top_logprobs": []}
        for t in output
    ]
    first_exchange.response = ChatCompletion.model_validate(data)
    second, _ = _history()
    source = second.message_sources[-1]
    assert source is not None and isinstance(
        source.exchange, tr.ChatCompletionsExchange
    )
    second_exchange = source.exchange
    request = [dict(m) for m in first.messages] + [{"role": "user", "content": "next"}]
    second_exchange.request["messages"] = cast(
        list[ChatCompletionMessageParam], deepcopy(request)
    )
    served = deepcopy(request)
    served[1]["reasoning_content"] = "explicit thought"
    second_exchange.response.id = "later-literal"
    extra = second_exchange.response.choices[0].model_extra
    assert extra is not None
    extra["prompt_token_ids"] = tokenizer.apply_chat_template(
        served,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=True,
    )
    # Different request thinking modes are normally separate automatic
    # histories. Explicit mixed histories are also supported and source checked.
    history = tr.ChatCompletionsHistory(
        model=first.model,
        messages=[
            *first.messages,
            {"role": "user", "content": "next"},
            second.messages[-1],
        ],
        message_sources=[*first.message_sources, None, second.message_sources[-1]],
        chat_template=second.chat_template,
        chat_template_kwargs=second.chat_template_kwargs,
    )
    first_prompt, first_output, _ = _tokenize._chat_choice_tokens(
        first_exchange.response.choices[0], first_exchange.response
    )
    assert first_prompt is not None and first_output is not None
    assert (
        extra["prompt_token_ids"][: len(first_prompt) + len(first_output)]
        == first_prompt + first_output
    )
    original = history.model_dump(mode="python")
    checked = []
    validate = _tokenize._require_native_render_conditioning

    def observe(*args: Any) -> Any:
        result = validate(*args)
        checked.append(len(args[0]))
        return result

    monkeypatch.setattr(_tokenize, "_require_native_render_conditioning", observe)
    value = history.tokenize(tokenizer=tokenizer)
    assert checked == [2]
    for e in (first_exchange, second_exchange):
        prompt, native, logprobs = _tokenize._chat_choice_tokens(
            e.response.choices[0], e.response
        )
        assert prompt is not None and native is not None
        assert value.tokens[: len(prompt) + len(native)] == prompt + native
        assert value.logprobs[len(prompt) : len(prompt) + len(native)] == logprobs
    assert history.model_dump(mode="python") == original


@pytest.mark.parametrize("literal", [False, True])
def test_stock_renderer_shares_proofs_without_stale_alias_prefixes(
    literal: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_prefix_render_cache import _tokenizer

    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = _tokenizer(_TEMPLATE)
    tokenizer.backend_tokenizer.decoder = tokenizers.decoders.Fuse()
    history, _ = _two_turn_literal_history(served_literal=True)
    if not literal:
        # Construct a clean equivalent, rather than changing captured messages.
        history, _ = _history(content="ordinary answer")
    for source in history.message_sources:
        if source is None or source.choice_index is None:
            continue
        assert isinstance(source.exchange, tr.ChatCompletionsExchange)
        choice = source.exchange.response.choices[0]
        extra = choice.model_extra
        assert extra is not None
        for field in ("prompt_token_ids", "token_ids"):
            extra[field] = tokenizer(
                "".join(map(chr, extra[field])), add_special_tokens=False
            )["input_ids"]
        assert choice.logprobs is not None and choice.logprobs.content
        # Keep actual Pydantic logprob objects, with the new codec's token IDs.
        pair = choice.logprobs.content[0]
        choice.logprobs.content = [
            pair.model_copy(update={"token": f"token_id:{token}"})
            for token in extra["token_ids"]
        ]
    kwargs = {"enable_thinking": False, "preserve_thinking": True}
    assert _tokenize.cacheable_chat_template(
        tokenizer, _TEMPLATE, history.tools, kwargs, list(history.messages)
    )
    original = history.model_dump(mode="python")
    calls = []
    cache = _tokenize._PrefixChatRenderCache.for_messages

    def observe(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(self)
        return cache(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_tokenize._PrefixChatRenderCache, "for_messages", observe)
        actual = history.tokenize(tokenizer=tokenizer)
    assert len(calls) >= 2 and all(item is calls[0] for item in calls)
    with monkeypatch.context() as patch:
        patch.setattr(_tokenize, "cacheable_chat_template", lambda *args: False)
        uncached = history.tokenize(tokenizer=tokenizer)
    assert actual.tokens == uncached.tokens
    assert actual.flags == uncached.flags
    assert [None if x != x else x for x in actual.logprobs] == [
        None if x != x else x for x in uncached.logprobs
    ]
    assert history.model_dump(mode="python") == original
