from copy import deepcopy
import hashlib
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
import pytest

from art_inference.chat_template import (
    chat_template_with_preserved_thinking,
    default_chat_template_kwargs_for_template,
)

_TEMPLATE = (
    Path(__file__).parents[1] / "fixtures/qwen35_preserved_thinking.jinja"
).read_text()
_FIXED = chat_template_with_preserved_thinking(_TEMPLATE)
_USER = {"role": "user", "content": "A public question."}
_LITERALS = (
    "plain answer",
    "<think>thought</think>answer",
    "prefix<think>literal</think>suffix",
    "answer<think>literal</think>",
    "prefix</think>middle</think>suffix",
    "<think>one</think><think>two</think>",
    "<think><think>nested</think></think>",
    "<think>unclosed",
    "unopened</think>",
    "<think>",
    "</think>",
    "\n  before <think> café 漢字 🦉 </think> after  \n",
    "",
)


def _render(template, messages, **kwargs):
    def refuse(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    return env.from_string(template).render(
        messages=messages, raise_exception=refuse, **kwargs
    )


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("content", _LITERALS)
def test_plain_content_is_literal_in_every_mode(content, thinking, preserve):
    messages = [_USER, {"role": "assistant", "content": content}]
    before = deepcopy(messages)
    rendered = _render(
        _FIXED, messages, enable_thinking=thinking, preserve_thinking=preserve
    )
    assert rendered.endswith(content + "<|im_end|>\n")
    assert messages == before
    # Changing the next turn's thinking mode never reinterprets history.
    assert rendered == _render(
        _FIXED, messages, enable_thinking=not thinking, preserve_thinking=preserve
    )


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize(
    "reasoning", [None, "", "reasoned\n", "<think>literal reasoning text</think>\n"]
)
def test_structured_reasoning_and_explicit_empty_field_keep_existing_behavior(
    thinking, preserve, reasoning
):
    messages = [
        _USER,
        {"role": "assistant", "content": "answer", "reasoning_content": reasoning},
        {"role": "user", "content": "next"},
    ]
    kwargs = dict(enable_thinking=thinking, preserve_thinking=preserve)
    before = deepcopy(messages)
    assert _render(_FIXED, messages, **kwargs) == _render(_TEMPLATE, messages, **kwargs)
    assert messages == before
    rendered = _render(_FIXED, messages, **kwargs)
    if reasoning:
        assert (reasoning in rendered) == preserve


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_proven_legacy_encoding_uses_existing_structured_fields(thinking, preserve):
    # This fixture declares the old encoding. The renderer cannot infer that
    # declaration from an indistinguishable literal string in plain content.
    legacy = {"role": "assistant", "content": "<think>\nthought\n</think>\n\nanswer"}
    structured = {
        "role": "assistant",
        "reasoning_content": "thought\n",
        "content": "answer",
    }
    kwargs = dict(enable_thinking=thinking, preserve_thinking=preserve)
    later = {"role": "user", "content": "next"}
    assert _render(_FIXED, [_USER, structured, later], **kwargs) == _render(
        _TEMPLATE, [_USER, legacy, later], **kwargs
    )
    assert legacy["content"] in _render(_FIXED, [_USER, legacy], **kwargs)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("content", [None, "", "before<think>literal</think>after"])
def test_tool_call_and_continuation_keep_content_and_arguments(
    thinking, preserve, content
):
    assistant = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"function": {"name": "lookup", "arguments": {"q": "</think>"}}}
        ],
    }
    messages = [_USER, assistant]
    before = deepcopy(messages)
    kwargs = dict(enable_thinking=thinking, preserve_thinking=preserve)
    rendered = _render(_FIXED, messages, **kwargs)
    assert "<function=lookup>" in rendered
    assert "<parameter=q>\n</think>\n</parameter>" in rendered
    if content:
        assert content in rendered
    continued = _render(
        _FIXED,
        [
            *messages,
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "next<think>literal</think>answer"},
        ],
        **kwargs,
    )
    assert continued.startswith(rendered)
    assert messages == before


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_generation_prompt_and_preserved_history_prefix_are_stable(thinking, preserve):
    kwargs = dict(
        enable_thinking=thinking, preserve_thinking=preserve, add_generation_prompt=True
    )
    assert _render(_FIXED, [_USER], **kwargs) == _render(_TEMPLATE, [_USER], **kwargs)
    messages = [
        _USER,
        {"role": "assistant", "content": "plain<think>literal</think>tail"},
    ]
    completed = _render(_FIXED, messages, preserve_thinking=preserve)
    continuation = _render(_FIXED, [*messages, _USER], **kwargs)
    if preserve:
        assert continuation.startswith(completed)
    else:
        # Explicit opt-out still removes the previous turn's reasoning scaffold;
        # the visible body is unchanged, not a newly promised full-token prefix.
        assert messages[-1]["content"] + "<|im_end|>\n" in continuation
        assert not continuation.startswith(completed)


def test_actual_template_operation_and_public_e2ac_shaped_regression():
    assert (
        hashlib.sha256(_TEMPLATE.encode()).hexdigest()
        == "098047d425a6673b1fe1a82a197a481616e53a283beaa8cb76cbb74d38ca6644"
    )
    # Public text with the captured branch's shape; no private text or IDs.
    prefix = "P" * 4149
    body = prefix + "<think>\n" + "R" * 747 + "\n</think>\n\n" + "A" * 1161
    messages = [_USER, {"role": "assistant", "content": body}]
    original = _render(
        _TEMPLATE, messages, enable_thinking=False, preserve_thinking=True
    )
    assert prefix not in original
    assert body in _render(
        _FIXED, messages, enable_thinking=False, preserve_thinking=True
    )


def test_configuration_is_idempotent_and_does_not_change_defaults_or_other_templates():
    assert _FIXED != _TEMPLATE
    assert chat_template_with_preserved_thinking(_FIXED) == _FIXED
    assert default_chat_template_kwargs_for_template(
        _FIXED
    ) == default_chat_template_kwargs_for_template(_TEMPLATE)
    other = "{% for message in messages %}{{ message.content }}{% endfor %}"
    assert chat_template_with_preserved_thinking(other) == other
    assert chat_template_with_preserved_thinking(
        {"default": _TEMPLATE, "other": other}
    ) == {"default": _FIXED, "other": other}


def test_unconfigured_template_receives_the_same_correction():
    # Reverse the prior preservation-only rewrite of this public fixture.
    raw = (
        _TEMPLATE.replace(
            "{%- set preserve_thinking = preserve_thinking | default(true) -%}", ""
        )
        .replace(
            "(render_content(message.content, true) if preserve_thinking and message.role == 'assistant' else render_content(message.content, true)|trim)",
            "render_content(message.content, true)|trim",
        )
        .replace(
            "{%- if not preserve_thinking or message.reasoning_content is not string %}{%- set reasoning_content = reasoning_content|trim %}{%- endif %}",
            "{%- set reasoning_content = reasoning_content|trim %}",
        )
        .replace(
            "('</think>\\n\\n' if preserve_thinking and message.reasoning_content is string and reasoning_content else '\\n</think>\\n\\n')",
            "'\\n</think>\\n\\n'",
        )
    )
    assert chat_template_with_preserved_thinking(raw) == _FIXED


@pytest.mark.parametrize("wrapper", [("{% raw %}", "{% endraw %}"), ("{#", "#}")])
def test_inline_operation_as_raw_or_comment_text_is_not_rewritten(wrapper):
    from art_inference.chat_template import _QWEN_INLINE_REASONING

    operation = _QWEN_INLINE_REASONING.search(_TEMPLATE).group()
    template = wrapper[0] + operation + wrapper[1]
    assert chat_template_with_preserved_thinking(template) == template


def test_other_structured_reasoning_condition_is_not_rewritten():
    gate = "{% if preserve_thinking and message.role == 'assistant' %}{{ message.reasoning_content }}{% endif %}"
    template = _TEMPLATE + "{% for message in messages %}" + gate + "{% endfor %}"
    fixed = chat_template_with_preserved_thinking(template)
    assert gate in fixed
    messages = [
        _USER,
        {"role": "assistant", "content": "answer", "reasoning_content": "prior reason"},
        _USER,
    ]
    assert "prior reason" not in _render(fixed, messages, preserve_thinking=False)


def test_inline_operation_inside_quoted_expression_is_literal():
    from art_inference.chat_template import _QWEN_INLINE_REASONING

    operation = _QWEN_INLINE_REASONING.search(_TEMPLATE).group().replace("\n", " ")
    template = '{{ "' + operation + '" }}'
    fixed = chat_template_with_preserved_thinking(template)
    assert fixed == template
    assert "set reasoning_content = content.split" in _render(fixed, [])


@pytest.mark.parametrize(
    "wrapper", [("{% raw %}", "{% endraw %}"), ("{#", "#}"), ('{{ "', '" }}')]
)
def test_mixed_executable_and_literal_operations_only_changes_executable(wrapper):
    from art_inference.chat_template import _QWEN_INLINE_REASONING

    operation = _QWEN_INLINE_REASONING.search(_TEMPLATE).group().replace("\n", " ")
    literal = wrapper[0] + operation + wrapper[1]
    template = _TEMPLATE + literal
    fixed = chat_template_with_preserved_thinking(template)
    assert fixed == _FIXED + literal
    assert "head<think>literal</think>tail" in _render(
        fixed,
        [_USER, {"role": "assistant", "content": "head<think>literal</think>tail"}],
    )
