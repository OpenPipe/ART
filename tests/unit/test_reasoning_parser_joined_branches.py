from jinja2.sandbox import ImmutableSandboxedEnvironment
import pytest

from art_inference.chat_template import (
    _QWEN_INLINE_REASONING,
    _without_inline_reasoning_parser,
)
from tests.unit.test_literal_reasoning_content import _TEMPLATE


@pytest.mark.parametrize("scope", ["top", "macro"])
@pytest.mark.parametrize("layout", ["if", "elif", "nested", "sequential"])
@pytest.mark.parametrize("content", ["  answer  ", "  before<think>x</think>after  "])
def test_joined_preview_retains_its_original_trim(scope, layout, content):
    match = _QWEN_INLINE_REASONING.search(_TEMPLATE)
    assert match is not None
    parser = match.group()
    trim = "{% set content = render_content(message.content, true)|trim %}"
    if layout == "if":
        branch = "{% if not preview_only %}" + parser + "{% endif %}"
    elif layout == "elif":
        branch = (
            "{% if preview_only %}{% elif mode == 'answer' %}" + parser + "{% endif %}"
        )
    elif layout == "nested":
        branch = (
            "{% if outer %}{% if not preview_only %}"
            + parser
            + "{% endif %}{% endif %}"
        )
    else:
        branch = (
            "{% if not preview_only %}" + parser + "{% endif %}"
            "{% if mode == 'other' %}" + parser + "{% endif %}"
        )
    body = trim + branch + "[{{ content }}]"
    if scope == "macro":
        body = (
            "{% macro answer(message) %}" + body + "{% endmacro %}{{ answer(message) }}"
        )
    template = (
        "{% macro render_content(content, count) %}{{ content }}{% endmacro %}" + body
    )
    fixed = _without_inline_reasoning_parser(template)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    kwargs = dict(
        message={"role": "assistant", "content": content},
        preview_only=True,
        mode="answer",
        outer=True,
    )
    assert env.from_string(fixed).render(**kwargs) == env.from_string(template).render(
        **kwargs
    )
    assert env.from_string(fixed).render(**kwargs) == "[" + content.strip() + "]"
    assert trim in fixed
    assert not _QWEN_INLINE_REASONING.search(fixed)
    # The parser path still treats the text as literal; its shared trim stays.
    kwargs["preview_only"] = False
    assert env.from_string(fixed).render(**kwargs) == "[" + content.strip() + "]"
    assert _without_inline_reasoning_parser(fixed) == fixed


@pytest.mark.parametrize("mode", ["a", "b", "c"])
def test_unknown_comparison_keeps_shared_trim_even_when_all_paths_have_parser(mode):
    match = _QWEN_INLINE_REASONING.search(_TEMPLATE)
    assert match is not None
    parser = match.group()
    template = (
        "{% macro render_content(content, count) %}{{ content }}{% endmacro %}"
        "{% set content = render_content(message.content, true)|trim %}"
        "{% if mode == 'a' %}"
        + parser
        + "{% elif mode == 'b' %}"
        + parser
        + "{% else %}"
        + parser
        + "{% endif %}[{{ content }}]"
    )
    fixed = _without_inline_reasoning_parser(template)
    content = "  before<think>x</think>after  "
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    assert (
        env.from_string(fixed).render(
            mode=mode, message={"role": "assistant", "content": content}
        )
        == "[" + content.strip() + "]"
    )
    assert _without_inline_reasoning_parser(fixed) == fixed
