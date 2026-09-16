from types import SimpleNamespace

import jinja2
import pytest

from art import get_tokenizer


def test_default_tokenizer_preserves_reasoning_and_honors_explicit_opt_out(monkeypatch):
    import transformers

    calls = []
    template = (
        "{% set enable_thinking = true %}"
        "{% set ns = namespace(last_query_index=2) %}"
        "{% for message in messages %}"
        "{%- if loop.index0 > ns.last_query_index %}{{ message.reasoning }}{% endif %}"
        "{{ message.content }}{% endfor %}"
    )

    def load(model, **kwargs):
        calls.append((model, kwargs))
        return SimpleNamespace(chat_template=template)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    tokenizer = get_tokenizer(
        "model:variant", revision="revision", local_files_only=True
    )
    assert calls == [
        (
            "model",
            {
                "revision": "revision",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]
    messages = [{"reasoning": "thought", "content": "answer"}]
    assert isinstance(tokenizer.chat_template, str)
    render = jinja2.Environment().from_string(tokenizer.chat_template).render
    assert render(messages=messages) == "thoughtanswer"
    assert render(messages=messages, preserve_thinking=False) == "answer"
    tokenizer.chat_template = "changed"
    assert get_tokenizer("model").chat_template != "changed"


def test_pinned_llama_revision_does_not_switch_repositories(monkeypatch):
    import transformers

    calls = []
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: calls.append((a, k)),
    )
    get_tokenizer("meta-llama/Llama-3.1-8B", revision="model-specific-commit")
    assert calls[0][0] == ("meta-llama/Llama-3.1-8B",)


@pytest.mark.parametrize("revision", [None, "pinned-commit"])
def test_ungated_llama_fallback_is_only_for_unpinned_loads(monkeypatch, revision):
    import transformers

    calls = []

    def load(model, **kwargs):
        calls.append(model)
        if model.startswith("meta-llama/"):
            raise OSError("gated repository")
        return SimpleNamespace(chat_template="native text template")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load)
    if revision:
        with pytest.raises(OSError, match="gated"):
            get_tokenizer("meta-llama/Llama-3.1-8B", revision=revision)
        assert len(calls) == 1
    else:
        get_tokenizer("meta-llama/Llama-3.1-8B")
        assert calls[-1] == "thinkingmachineslabinc/meta-llama-3-instruct-tokenizer"


def test_llama_base_retains_native_vocabulary_with_default_chat_template(monkeypatch):
    import transformers

    native = SimpleNamespace(chat_template=None)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda model, **_: (
            native
            if model.startswith("meta-llama/")
            else SimpleNamespace(chat_template="public chat template")
        ),
    )
    actual = get_tokenizer("meta-llama/Llama-3.1-8B")
    assert actual is native
    assert actual.chat_template == "public chat template"


@pytest.mark.parametrize(
    "template",
    [
        "reasoning_content and loop.index0 > ns.last_user_index",
        "thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls')",
        "message.thinking and not future_final_message.found",
    ],
)
def test_supported_template_patches_are_idempotent(template):
    from art.utils.chat_template import chat_template_with_preserved_thinking

    patched = chat_template_with_preserved_thinking(template)
    assert patched != template
    assert chat_template_with_preserved_thinking(patched) == patched


def test_glm_preserves_content_whitespace_and_remains_valid_jinja():
    from art.utils.chat_template import chat_template_with_preserved_thinking

    template = "{% if clear_thinking %}legacy{% endif %}{%- if content.strip() -%}{{ content.strip() }}{%- endif -%}"
    configured = chat_template_with_preserved_thinking(template)
    assert isinstance(configured, str)
    assert (
        jinja2.Environment().from_string(configured).render(content="\nanswer\n")
        == "\nanswer\n"
    )
    assert chat_template_with_preserved_thinking(configured) == configured


def test_kimi_preserves_old_reasoning_when_next_turn_does_not_think():
    from art.utils.chat_template import chat_template_with_preserved_thinking

    template = """{% set ns = namespace(last_non_tool_call_assistant_msg=0) %}
    {%- set hist_msgs = messages[:ns.last_non_tool_call_assistant_msg+1] -%}
    {%- set suffix_msgs = messages[ns.last_non_tool_call_assistant_msg+1:] -%}
    {% for message in hist_msgs %}{{ message.content }}{% endfor %}
    {% for message in suffix_msgs %}{%- if thinking is defined and thinking is false -%}{{ message.content }}{% else %}{{ message.reasoning_content }}{{ message.content }}{% endif %}{% endfor %}"""
    configured = chat_template_with_preserved_thinking(template)
    assert isinstance(configured, str)
    render = jinja2.Environment().from_string(configured).render
    messages = [{"reasoning_content": "thought", "content": "answer"}]
    assert "thoughtanswer" in render(messages=messages, thinking=False)
    assert "thought" not in render(
        messages=messages, thinking=False, preserve_thinking=False
    )
    assert chat_template_with_preserved_thinking(configured) == configured


@pytest.mark.parametrize("reasoning", [None, "", "\nthought\n"])
@pytest.mark.parametrize("next_thinking", [False, True])
@pytest.mark.parametrize(
    "tools",
    [
        None,
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    ],
)
def test_deepseek_v4_retains_prior_turns_when_generation_mode_changes(
    reasoning, next_thinking, tools
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from art.megatron.dsv4.tokenizer import get_dsv4_tokenizer

    tokenizer = get_dsv4_tokenizer(
        PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0})))
    )
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "reasoning_content": reasoning},
    ]
    completed = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, enable_thinking=reasoning is not None
    )
    continued = tokenizer.apply_chat_template(
        [*messages, {"role": "user", "content": "next"}],
        tools=tools,
        tokenize=False,
        enable_thinking=next_thinking,
    )
    assert isinstance(completed, str) and isinstance(continued, str)
    assert continued.startswith(completed)
    if reasoning:
        assert reasoning in continued
        assert reasoning not in tokenizer.apply_chat_template(
            [*messages, {"role": "user", "content": "next"}],
            tools=tools,
            tokenize=False,
            enable_thinking=next_thinking,
            preserve_thinking=False,
        )
