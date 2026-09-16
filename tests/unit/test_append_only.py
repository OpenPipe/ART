import asyncio
from types import SimpleNamespace

from pydantic import BaseModel
import pytest

from art.token_prefix import TokenPrefixCache, apply_prefix_edits
from art.utils.append_only import chat_prefix_observations, chat_response_prefixes
from art_inference.append_only import (
    output_prefix_observations,
    patch_deepseek_renderer,
)


@pytest.mark.parametrize("reasoning", [None, "", "\nthought\n"])
def test_dsv32_earlier_turn_framing_survives_a_new_user_turn(reasoning):
    def render(index, messages, thinking_mode):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        thinking = thinking_mode == "thinking"
        if messages[index]["role"] == "user":
            return "USER" + (
                "<think>" if thinking and index == last_user else "</think>"
            )
        if thinking and index > last_user:
            assert messages[index]["reasoning_content"], "Missing reasoning"
            return messages[index]["reasoning_content"] + "END"
        return "END"

    encoding = SimpleNamespace(render_message=render)
    preserve = True
    patch_deepseek_renderer(encoding, lambda: preserve, prefix_only=True)
    messages = [
        {"role": "user"},
        {"role": "assistant", "reasoning_content": reasoning},
    ]

    def encode(messages):
        return "".join(
            encoding.render_message(i, messages, "thinking")
            for i in range(len(messages))
        )

    completed = encode(messages)
    assert encode([*messages, {"role": "user"}]).startswith(completed)
    if reasoning:
        preserve = False
        assert not encode([*messages, {"role": "user"}]).startswith(completed)


class Tokenizer:
    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens, *, skip_special_tokens=False):
        return bytes(tokens).decode()


def test_many_protocol_markers_do_not_multiply_long_prompt_storage():
    tokenizer = SimpleNamespace(
        all_special_ids=[1, 2, 3],
        encode=lambda text, **_: [ord(char) for char in text],
        decode=lambda tokens, **_: "".join(chr(token) for token in tokens),
    )
    prompt = [100] * 128_000
    output = [101, 1, 102, 2, 103, 3] * 100
    entries = output_prefix_observations(tokenizer, prompt, prompt, output)
    assert len(entries) <= 2
    assert entries[-1][1] == prompt + output


@pytest.mark.parametrize("action", ["pass", '<tool>{"x": 1}</tool>'])
def test_normalized_reasoning_survives_an_edited_action(action):
    tokenizer = Tokenizer()
    encode = tokenizer.encode
    prompt = encode("USER question ASSISTANT <think>\n")
    rendered = prompt + encode(f"reasoning\n</think>\n\n{action}END\n")
    sampled = encode(f"\nreasoning\n</think>\n\n{action}END")
    reasoning = prompt + encode("reasoning\n</think>\n\nEND\n")
    observations = chat_prefix_observations(
        tokenizer, prompt, rendered, prompt, sampled, reasoning_prompt=reasoning
    )
    assert len(observations) == 2
    cache = TokenPrefixCache()
    for rendered_prefix, raw_prefix, edits in observations:
        assert apply_prefix_edits(rendered_prefix, edits) == raw_prefix
        cache.insert(rendered_prefix, raw_prefix, "rollout", edits)

    for next_action in (action, "corrected action"):
        next_prompt = prompt + encode(
            f"reasoning\n</think>\n\n{next_action}END\nUSER next ASSISTANT"
        )
        match = cache.lookup(next_prompt, "rollout")
        assert match is not None
        served = list(match.raw_prefix) + next_prompt[match.rendered_length :]
        assert served == prompt + encode(
            f"\nreasoning\n</think>\n\n{next_action}END\nUSER next ASSISTANT"
        )


def test_truncation_does_not_replace_completed_turn_framing():
    tokenizer = Tokenizer()
    encode = tokenizer.encode
    assert not chat_prefix_observations(
        tokenizer,
        encode("prompt"),
        encode("prompt partial END"),
        encode("prompt"),
        encode(" partial"),
        complete=False,
    )


def test_changed_prompt_cannot_be_registered_as_a_continuation():
    assert not chat_prefix_observations(Tokenizer(), [1], [2, 3], [4], [5])


@pytest.mark.parametrize(
    "options",
    [{"preserve_thinking": False}, {"clear_thinking": True}, {"drop_thinking": True}],
)
def test_explicit_rewriting_bypasses_observation(options):
    class Request(BaseModel):
        messages: list[dict]
        chat_template_kwargs: dict

    async def render(_):
        pytest.fail("Explicit rewriting must not create preservation mappings")

    assert (
        asyncio.run(
            chat_response_prefixes(
                Tokenizer(),
                Request(messages=[], chat_template_kwargs=options),
                [],
                [],
                render,
            )
        )
        == []
    )


def test_custom_stop_does_not_delete_the_template_terminator():
    encode = Tokenizer().encode
    assert (
        chat_prefix_observations(
            Tokenizer(),
            encode("prompt:"),
            encode("prompt:answerEND"),
            encode("prompt:"),
            encode("answer"),
            complete=True,
        )
        == []
    )


def test_reasoning_boundary_does_not_absorb_whitespace_merged_into_action():
    from art_inference.append_only import _whitespace_prefix

    tokenizer = Tokenizer()
    raw = tokenizer.encode("thought</think>\n\naction")
    boundary = _whitespace_prefix(tokenizer, tokenizer.encode("thought</think>"), raw)
    assert tokenizer.decode(raw[:boundary]) == "thought</think>"


def test_missing_lineage_never_selects_an_ambiguous_tokenization():
    cache = TokenPrefixCache()
    cache.insert([1, 2], [1, 3], "first")
    match = cache.lookup([1, 2], None)
    assert match is not None and match.raw_prefix == (1, 3)
    cache.insert([1, 2], [1, 4], "second")
    assert cache.lookup([1, 2], None) is None
    first = cache.lookup([1, 2], "first")
    second = cache.lookup([1, 2], "second")
    assert first is not None and first.raw_prefix == (1, 3)
    assert second is not None and second.raw_prefix == (1, 4)


def test_history_edits_leave_multimodal_markers_outside_replacements():
    from types import SimpleNamespace

    from art_inference.append_only import _rendering_edits
    from art_inference.vllm import replace_prompt_tokens

    rendered = [1, 2, 9, 9, 3, 4]
    raw = [1, 1, 2, 9, 9, 3, 4, 4]
    edits = _rendering_edits(SimpleNamespace(all_special_ids=[9]), rendered, raw)
    assert apply_prefix_edits(rendered, edits) == raw
    original = {
        "prompt_token_ids": rendered,
        "mm_placeholders": {"image": [SimpleNamespace(offset=2, length=2)]},
    }
    updated = replace_prompt_tokens(original, raw, edits)
    assert updated["mm_placeholders"]["image"][0].offset == 3
