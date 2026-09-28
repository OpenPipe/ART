from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from jinja2 import Environment
import pytest

from art_inference import chat_template as templates


@pytest.fixture(autouse=True)
def empty_cache():
    templates._cached_normalize_chat_template.cache_clear()
    yield
    templates._cached_normalize_chat_template.cache_clear()


@pytest.fixture
def raw_template():
    return (
        Path(__file__).parents[1] / "fixtures/qwen35_preserved_thinking.jinja"
    ).read_text()


def test_equal_raw_text_avoids_repeated_analysis(monkeypatch, raw_template):
    calls = []
    parse = Environment.parse

    def counted(self, source, *args, **kwargs):
        calls.append(source)
        return parse(self, source, *args, **kwargs)

    monkeypatch.setattr(Environment, "parse", counted)
    expected = templates._normalize_chat_template(raw_template)
    calls.clear()
    assert templates.chat_template_with_preserved_thinking(raw_template) == expected
    assert calls
    calls.clear()
    equal = raw_template.encode().decode()
    assert equal is not raw_template
    assert templates.chat_template_with_preserved_thinking(equal) == expected
    assert not calls
    assert (
        templates.chat_template_with_preserved_thinking(equal + "\n") == expected + "\n"
    )
    assert calls


def test_cached_source_still_obeys_each_render_options(raw_template):
    fixed = templates.chat_template_with_preserved_thinking(raw_template)
    assert isinstance(fixed, str)
    rendered = Environment().from_string(fixed)
    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "literal<think>text</think>answer",
            "reasoning_content": "structured",
        },
        {"role": "user", "content": "next"},
    ]
    for preserve in (True, False):
        for thinking in (True, False):
            options = dict(
                messages=messages,
                preserve_thinking=preserve,
                enable_thinking=thinking,
                add_generation_prompt=True,
            )
            expected = (
                Environment()
                .from_string(templates._normalize_chat_template(raw_template))
                .render(**options)
            )
            assert (
                templates.chat_template_with_preserved_thinking(raw_template) == fixed
            )
            assert rendered.render(**options) == expected


def test_dictionary_results_are_fresh_and_non_strings_pass_through(raw_template):
    marker = object()
    source: dict[str, Any] = {"nested": {"default": raw_template}, "marker": marker}
    first = cast(
        dict[str, Any], templates.chat_template_with_preserved_thinking(source)
    )
    assert isinstance(first, dict)
    first["nested"]["default"] = "changed"
    second = cast(
        dict[str, Any], templates.chat_template_with_preserved_thinking(source)
    )
    assert isinstance(second, dict)
    assert second["nested"]["default"] != "changed"
    assert isinstance(source["nested"], dict)
    assert source["nested"]["default"] == raw_template
    assert second["marker"] is marker


def test_string_subclass_keeps_uncached_method_behavior():
    class Template(str):
        visits = 0

        def __contains__(self, value):
            self.visits += 1
            return super().__contains__(value)

    value = Template("{{ messages }}")
    assert templates.chat_template_with_preserved_thinking(value) == value
    first = value.visits
    assert first > 0
    assert templates.chat_template_with_preserved_thinking(value) == value
    assert value.visits > first
    assert templates._cached_normalize_chat_template.cache_info().currsize == 0


@pytest.mark.parametrize("extra", [0, 1])
def test_input_size_boundary_preserves_output_and_oversized_fallback(extra):
    value = "🦉" * (templates._TEMPLATE_CACHE_MAX_CHARS + extra)
    for _ in range(2):
        assert templates.chat_template_with_preserved_thinking(value) == value
    info = templates._cached_normalize_chat_template.cache_info()
    assert info.currsize == (0 if extra else 1)
    assert info.hits == (0 if extra else 1)


@pytest.mark.parametrize("extra", [0, 1])
def test_output_size_boundary_does_not_retain_oversized_result(monkeypatch, extra):
    original = templates._normalize_chat_template
    stem = "clear_thinking"
    value = stem + "x" * (
        templates._TEMPLATE_CACHE_MAX_CHARS - len(original(stem)) + extra
    )
    result = original(value)
    assert len(value) < templates._TEMPLATE_CACHE_MAX_CHARS
    assert len(result) == templates._TEMPLATE_CACHE_MAX_CHARS + extra
    calls = []

    def normalize(template):
        calls.append(template)
        return result

    monkeypatch.setattr(templates, "_normalize_chat_template", normalize)
    for _ in range(2):
        assert templates.chat_template_with_preserved_thinking(value) is result
    assert calls == [value] * (2 if extra else 1)
    assert templates._cached_normalize_chat_template.cache_info().currsize == (
        0 if extra else 1
    )


def test_analysis_exceptions_propagate_and_are_not_cached(monkeypatch):
    failure = ValueError("normalizer failed")
    calls = []

    def fail(template):
        calls.append(template)
        raise failure

    monkeypatch.setattr(templates, "_normalize_chat_template", fail)
    for _ in range(2):
        with pytest.raises(ValueError) as raised:
            templates.chat_template_with_preserved_thinking("input")
        assert raised.value is failure
    assert calls == ["input", "input"]
    assert templates._cached_normalize_chat_template.cache_info().currsize == 0


def test_entry_limit_evicts_without_changing_normalization():
    for index in range(65):
        value = f"{{# template {index} #}}"
        assert templates.chat_template_with_preserved_thinking(value) == value
    before = templates._cached_normalize_chat_template.cache_info()
    assert before.currsize == before.maxsize == 64
    value = "{# template 0 #}"
    assert templates.chat_template_with_preserved_thinking(value) == value
    assert (
        templates._cached_normalize_chat_template.cache_info().misses
        == before.misses + 1
    )


def test_concurrent_calls_return_exact_same_source(raw_template):
    expected = templates._normalize_chat_template(raw_template)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                templates.chat_template_with_preserved_thinking, [raw_template] * 12
            )
        )
    assert results == [expected] * 12
    assert templates._cached_normalize_chat_template.cache_info().currsize == 1
