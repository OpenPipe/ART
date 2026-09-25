from copy import deepcopy
from datetime import datetime, timedelta
import json
import string

from openai.types.chat import ChatCompletion
import pytest

import art.trajectories as tr
from art.trajectories import _tokenize as tokenization


@pytest.fixture(autouse=True)
def restore_retokenization_warning(monkeypatch):
    monkeypatch.setattr(tokenization, "_WARNED_PREFIX_RETOKENIZATION", False)


def test_prefix_cache_preserves_order_types_generation_and_probe_context():
    calls = []

    def render(messages, *, add_generation_prompt):
        calls.append(deepcopy(messages))
        return json.dumps(messages) + str(add_generation_prompt)

    messages = [{"role": "user", "content": "snow雪"}, {"a": -0.0, "b": True}]
    cache = tokenization._PrefixChatRenderCache(render)
    expected = render(messages, add_generation_prompt=False)
    cached = cache.for_messages(messages, expected)
    for generation in (True, False):
        assert cached(messages[:1], add_generation_prompt=generation) == render(
            messages[:1], add_generation_prompt=generation
        )
    before = len(calls)
    assert cached(messages[:1], add_generation_prompt=True).endswith("True")
    assert len(calls) == before

    # Mutating a new probe, reordering mapping keys, and scalar equality must
    # never alias the baseline context, including -0.0 versus +0.0.
    for replacement in (
        {"a": 0.0, "b": True},
        {"b": True, "a": -0.0},
        {"a": -0.0, "b": 1},
    ):
        probe = [deepcopy(messages[0]), replacement]
        current = cache.for_messages(probe, render(probe, add_generation_prompt=False))
        assert current(probe, add_generation_prompt=False) == render(
            probe, add_generation_prompt=False
        )
    reordered = list(reversed(messages))
    assert cached(reordered, add_generation_prompt=False) == render(
        reordered, add_generation_prompt=False
    )


def test_prefix_cache_bounds_storage_and_does_not_cache_failures(monkeypatch):
    monkeypatch.setattr(tokenization._PrefixChatRenderCache, "_MAX_BYTES", 520)
    monkeypatch.setattr(tokenization._PrefixChatRenderCache, "_MAX_ENTRIES", 2)
    calls = 0

    def render(messages, *, add_generation_prompt):
        nonlocal calls
        calls += 1
        if not messages:
            raise ValueError("empty history")
        return "🙂" * len(messages) + ("?" if add_generation_prompt else "")

    messages = [{"content": str(i)} for i in range(20)]
    cache = tokenization._PrefixChatRenderCache(render)
    cached = cache.for_messages(messages, render(messages, add_generation_prompt=False))
    for _ in range(2):
        with pytest.raises(ValueError, match="empty history"):
            cached([], add_generation_prompt=True)
        for i in range(1, 21):
            assert cached(messages[:i], add_generation_prompt=True) == "🙂" * i + "?"
    assert len(cache.prefixes) <= 2
    assert cache.bytes <= 520
    assert calls > 20  # Full-cache misses and exceptions continue to render.


def test_cache_settings_and_mutated_messages_invalidate_previous_prefixes():
    settings = {"tool": "lookup"}

    def render(messages, *, add_generation_prompt):
        return settings["tool"] + json.dumps(messages) + str(add_generation_prompt)

    messages = [{"role": "user", "content": "first"}]
    cache = tokenization._PrefixChatRenderCache(render)
    for tool, content in (
        ("lookup", "first"),
        ("other", "first"),
        ("other", "changed"),
    ):
        settings["tool"], messages[0]["content"] = tool, content
        expected = render(messages, add_generation_prompt=True)
        current = cache.for_messages(
            messages, expected, settings=tokenization._render_context_key(settings)
        )
        assert current(messages, add_generation_prompt=True) == expected


def test_prefix_deltas_do_not_retain_quadratic_text():
    def render(messages, *, add_generation_prompt):
        return "".join(m["content"] for m in messages) + (
            "?" if add_generation_prompt else ""
        )

    messages = [{"content": "雪" * 4096} for _ in range(128)]
    cache = tokenization._PrefixChatRenderCache(render)
    cached = cache.for_messages(messages, render(messages, add_generation_prompt=False))
    for i in range(1, 129):
        assert (
            cached(messages[:i], add_generation_prompt=True) == "雪" * (4096 * i) + "?"
        )
    assert len(cache.prefixes) == 128
    assert sum(len(tail) for _, tail in cache.prefixes.values()) == 128


@pytest.mark.parametrize("value", [object(), float("nan"), float("inf"), {1: "x"}])
def test_prefix_cache_bypasses_non_json_context(value):
    def render(messages, *, add_generation_prompt):
        return "unchanged"

    cache = tokenization._PrefixChatRenderCache(render)
    assert cache.for_messages([{"content": value}], "unchanged") is render


def test_later_generation_split_is_recomputed_when_completed_suffix_is_equal():
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "y"},
        {"role": "assistant", "content": "HELLO"},
    ]

    def render(selected, *, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}!" for m in selected)
        if add_generation_prompt:
            text += "<assistant>"
            if any(m["content"] == "B" for m in selected):
                text += "H"
        return text

    cache = tokenization._PrefixChatRenderCache(render)
    original = render(messages, add_generation_prompt=False)
    tokenization._assistant_char_spans(
        messages,
        original,
        cache.for_messages(messages, original),
        add_generation_prompt=False,
    )
    probe = deepcopy(messages)
    probe[1]["content"] = "B"
    text = render(probe, add_generation_prompt=False)
    assert text == original.replace(">A!", ">B!")
    actual = tokenization._assistant_char_spans(
        probe, text, cache.for_messages(probe, text), add_generation_prompt=False
    )
    expected = tokenization._assistant_char_spans(
        probe, text, render, add_generation_prompt=False
    )
    assert actual == expected
    start, end = actual[-1]
    assert text[start:end] == "ELLO!"


_TEMPLATE = """{% for message in messages %}<{{ message.role }}>{{ message.content or '' }}
{% if message.reasoning_content %}<think>{{ message.reasoning_content }}</think>{% endif %}
{% for call in message.tool_calls or [] %}{% set tool_call = call.function %}<call>{{ tool_call.name }}({% for k, v in tool_call.arguments.items() %}{{ k }}={{ v|tojson }};{% endfor %})</call>{% endfor %}
{% if message.role == 'assistant' %}${% else %}</{{ message.role }}>{% endif %}{% endfor %}
{% if add_generation_prompt %}<assistant>{% endif %}"""


def _tokenizer(template):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {char: i for i, char in enumerate(dict.fromkeys(string.printable + "雪🙂"))}
    vocab["[UNK]"] = len(vocab)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="$",
        unk_token="[UNK]",
        chat_template=template,
    )


def _history(turns, *, reasoning=False, refusal=False, length=False, tokenizer=None):
    messages = [{"role": "user", "content": "snow雪🙂"}]
    exchanges = []
    for i in range(turns):
        message = {
            "role": "assistant",
            "content": " answer ",
            "tool_calls": [
                {
                    "id": f"call-{i}",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": json.dumps({"i": i})},
                }
            ],
        }
        if reasoning:
            message["reasoning"] = "consider 雪"
        if refusal:
            message["refusal"] = "cannot do that"
        choice = {
            "index": 0,
            "message": message,
            "finish_reason": "length" if length and i == turns - 1 else "tool_calls",
        }
        if tokenizer is not None:

            def tokens(selected, generation):
                return tokenizer.apply_chat_template(
                    tokenization.normalize_tool_call_arguments_for_chat_template(
                        selected, _TEMPLATE
                    ),
                    tokenize=True,
                    add_generation_prompt=generation,
                    return_dict=False,
                )

            prompt, completed = (
                tokens(messages, True),
                tokens([*messages, message], False),
            )
            assert completed[: len(prompt)] == prompt
            output = completed[len(prompt) :]
            choice.update(
                prompt_token_ids=prompt,
                token_ids=output,
                logprobs={
                    "content": [
                        {
                            "token": f"token_id:{token}",
                            "logprob": -0.1,
                            "bytes": [],
                            "top_logprobs": [],
                        }
                        for token in output
                    ]
                },
            )
        response = ChatCompletion.model_validate(
            {
                "id": f"response-{i}",
                "object": "chat.completion",
                "created": i,
                "model": "test/model",
                "choices": [choice],
            }
        )
        start = datetime(2026, 1, 1) + timedelta(seconds=i)
        exchanges.append(
            tr.ChatCompletionsExchange(
                request=tr.ChatCompletionsRequest(
                    model="test/model", messages=deepcopy(messages)
                ),
                response=response,
                start_time=start,
                end_time=start + timedelta(milliseconds=1),
            )
        )
        messages.extend(
            [message, {"role": "tool", "tool_call_id": f"call-{i}", "content": "ok"}]
        )
    return tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=exchanges)
    ).chat_completions_history()


@pytest.mark.parametrize(
    "reasoning,refusal,length",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("ends_with_assistant", [False, True])
def test_cached_tool_probes_match_all_tokenized_fields(
    monkeypatch, reasoning, refusal, length, ends_with_assistant
):
    history = _history(6, reasoning=reasoning, refusal=refusal, length=length)
    if not ends_with_assistant:
        history.messages.append(
            {"role": "tool", "tool_call_id": "call-5", "content": "ok"}
        )
        history.message_sources.append(None)
    tokenizer = _tokenizer(_TEMPLATE)
    trace = tokenization._TraceBuilder()
    actual = tokenization._tokenize_chat_view(
        history,
        tokenizer=tokenizer,
        base_model=None,
        chat_template=_TEMPLATE,
        chat_template_kwargs=None,
        _trace=trace,
    )
    monkeypatch.setattr(
        tokenization,
        "cacheable_chat_template",
        lambda *args: False,
    )
    expected_trace = tokenization._TraceBuilder()
    expected = tokenization._tokenize_chat_view(
        history,
        tokenizer=tokenizer,
        base_model=None,
        chat_template=_TEMPLATE,
        chat_template_kwargs=None,
        _trace=expected_trace,
    )
    assert actual.model_dump_json() == expected.model_dump_json()
    assert trace.trace is not None and expected_trace.trace is not None
    assert trace.trace.source_keys == expected_trace.trace.source_keys
    assert trace.trace.sources == expected_trace.trace.sources
    assert any(f & tr.TokenFlag.OUTPUT for f in actual.flags)


def test_cached_probes_preserve_native_logprobs_stop_and_source_bindings(monkeypatch):
    tokenizer = _tokenizer(_TEMPLATE)
    history = _history(4, tokenizer=tokenizer)
    actual = history.tokenize(tokenizer=tokenizer, chat_template=_TEMPLATE)
    monkeypatch.setattr(
        tokenization,
        "cacheable_chat_template",
        lambda *args: False,
    )
    expected = history.tokenize(tokenizer=tokenizer, chat_template=_TEMPLATE)
    assert actual.model_dump_json() == expected.model_dump_json()
    assert -0.1 in actual.logprobs
    assert any(flag & tr.TokenFlag.STOP for flag in actual.flags)
    assert any(flag & tr.TokenFlag.SAMPLED for flag in actual.flags)


@pytest.mark.parametrize("turns", [4, 8, 16])
def test_tool_probe_scaling_reuses_only_unchanged_prefixes(monkeypatch, turns):
    from transformers import tokenization_utils_base

    history = _history(turns)
    tokenizer = _tokenizer(_TEMPLATE)
    original = tokenization_utils_base.render_jinja_template
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tokenization_utils_base, "render_jinja_template", counted)
    actual = history.tokenize(tokenizer=tokenizer, chat_template=_TEMPLATE)
    cached_calls = calls
    calls = 0
    monkeypatch.setattr(
        tokenization,
        "cacheable_chat_template",
        lambda *args: False,
    )
    expected = history.tokenize(tokenizer=tokenizer, chat_template=_TEMPLATE)
    assert actual.model_dump_json() == expected.model_dump_json()
    # This removes repeated unchanged prefixes, not the remaining quadratic
    # changed-prefix work. A zero-use cache must not pass this regression.
    assert cached_calls <= calls - turns * (turns - 1)
