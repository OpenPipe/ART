from types import MethodType

import pytest

from art.trajectories._render_cache import cacheable_chat_template


@pytest.fixture
def tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", chat_template="{{ messages }}"
    )


def eligible(tokenizer, template, **context):
    return cacheable_chat_template(
        tokenizer,
        template,
        context.get("tools"),
        context.get("kwargs", {}),
        context.get("messages", [{"role": "user", "content": "hello"}]),
    )


def test_stock_render_with_local_macro_namespace_and_generation(tokenizer):
    template = """{% set ns = namespace(n=0) %}
{% macro content(message) %}{{ message.content|trim }}{% endmacro %}
{% for message in messages[::-1] %}{% set ns.n = ns.n + 1 %}
{% generation %}{{ content(message) }}{% endgeneration %}{% endfor %}
{% for key, value in tools[0]|items %}{{ key }}={{ value|tojson }}{% endfor %}
{{ ns.n }}{% if add_generation_prompt %}assistant{% endif %}"""
    tools = [{"name": "lookup"}]
    assert eligible(tokenizer, template, tools=tools)
    assert "hello" in tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tools=tools,
        chat_template=template,
        tokenize=False,
    )


@pytest.mark.parametrize(
    "template",
    [
        "{{ strftime_now('%s') }}",
        "{{ lipsum() }}",
        "{{ messages|random }}",
        "{{ messages|map('random')|list }}",
        "{{ messages|attr('clear')() }}",
        "{% set f = namespace %}{{ f() }}",
        "{% set f = messages.clear %}{{ f() }}",
        "{{ messages.clear() }}",
        "{{ messages[0].get }}",
        "{{ messages[0]|items }}",
        "{% for message in messages %}{{ loop.cycle }}{% endfor %}",
        "{{ messages[0]['get']('role') }}",
        "{{ messages[0] is sameas messages[1] }}",
        "{{ self }}",
        "{% include 'other' %}",
        "{% invalid %}",
    ],
)
def test_unsupported_or_nondeterministic_syntax_bypasses(tokenizer, template):
    assert not eligible(tokenizer, template)


@pytest.mark.parametrize(
    "value", [object(), float("nan"), float("inf"), {1: "x"}, (1,)]
)
@pytest.mark.parametrize("field", ["messages", "tools", "kwargs"])
def test_non_plain_context_bypasses(tokenizer, field, value):
    assert not eligible(tokenizer, "{{ messages }}", **{field: {"value": value}})


def test_mutable_context_is_rechecked_and_cycles_bypass(tokenizer):
    tools = [{"name": "lookup"}]
    assert eligible(tokenizer, "{{ tools|tojson }}", tools=tools)
    tools[0]["callback"] = lambda: None
    assert not eligible(tokenizer, "{{ tools|tojson }}", tools=tools)
    cycle = []
    cycle.append(cycle)
    assert not eligible(tokenizer, "{{ messages }}", messages=cycle)


@pytest.mark.parametrize("method", ["apply_chat_template", "get_chat_template"])
def test_custom_method_is_never_called(tokenizer, monkeypatch, method):
    def mutating(self, *args, **kwargs):
        raise AssertionError("eligibility must not invoke custom renderers")

    monkeypatch.setattr(tokenizer, method, MethodType(mutating, tokenizer))
    assert not eligible(tokenizer, "{{ messages }}")


def test_custom_class_named_template_and_special_objects_bypass(tokenizer, monkeypatch):
    class Custom(type(tokenizer)):
        pass

    monkeypatch.setattr(tokenizer, "__class__", Custom)
    assert not eligible(tokenizer, "{{ messages }}")
    monkeypatch.undo()
    monkeypatch.setattr(tokenizer, "chat_template", {"named": "{{ messages }}"})
    assert not eligible(tokenizer, "named")
    monkeypatch.undo()

    class Token:
        def __str__(self):
            raise AssertionError("do not stringify custom special tokens")

    monkeypatch.setitem(tokenizer._special_tokens_map, "eos_token", Token())
    assert not eligible(tokenizer, "{{ messages }}")


def test_mutated_compiled_environment_bypasses(tokenizer, monkeypatch):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = "{{ messages|length }}"
    assert eligible(tokenizer, template)
    env = _compile_jinja_template(template).environment
    monkeypatch.setitem(env.filters, "length", lambda value: 42)
    assert not eligible(tokenizer, template)


def test_other_stock_helper_cannot_replace_pure_helper(tokenizer, monkeypatch):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = "{{ messages|tojson }}"
    assert eligible(tokenizer, template)
    env = _compile_jinja_template(template).environment
    monkeypatch.setitem(env.filters, "tojson", env.globals["strftime_now"])
    assert not eligible(tokenizer, template)


def test_generation_extension_override_bypasses(tokenizer, monkeypatch):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    template = "{% generation %}{{ messages }}{% endgeneration %}"
    assert eligible(tokenizer, template)
    env = _compile_jinja_template(template).environment
    tracker = next(
        ext for ext in env.extensions.values() if hasattr(ext, "_generation_support")
    )
    monkeypatch.setattr(
        tracker, "_generation_support", lambda *args, **kwargs: "changed"
    )
    assert not eligible(tokenizer, template)
