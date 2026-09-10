from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, cast

import pytest

import art.trajectories as tr
from art.trajectories import _tokenize as tokenization


def _linear_prefix(left: str | list[int], right: str | list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def _sequence(text: str, token_ids: bool) -> Any:
    return [ord(character) for character in text] if token_ids else text


@pytest.mark.parametrize("token_ids", [False, True])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("", ""),
        ("", "abc"),
        ("abc", ""),
        ("same", "same"),
        ("a", "b"),
        ("prefix", "prefix-longer"),
        ("prefix-longer", "prefix"),
        ("aaaaab", "aaaaac"),
        ("ababaXaba", "ababaYaba"),
        ("é", "e\u0301"),
        ("\0雪🙂e\u0301𐀀x", "\0雪🙂e\u0301𐀀y"),
        ("🙂" * 4096 + "x", "🙂" * 4096 + "y"),
        ("x" + "a" * 8192, "y" + "a" * 8192),
    ],
)
def test_common_prefix_matches_linear_oracle(left, right, token_ids):
    left, right = _sequence(left, token_ids), _sequence(right, token_ids)
    before = deepcopy((left, right))
    assert tokenization._common_prefix_length(left, right) == _linear_prefix(
        left, right
    )
    assert (left, right) == before


def _random_pair[T](rng: random.Random, alphabet: list[T]) -> tuple[list[T], list[T]]:
    prefix = rng.choices(alphabet, k=rng.randrange(180))
    return (
        prefix + rng.choices(alphabet, k=rng.randrange(40)),
        prefix + rng.choices(alphabet, k=rng.randrange(40)),
    )


@pytest.mark.parametrize("token_ids", [False, True])
def test_common_prefix_seeded_differential_matrix(token_ids):
    rng = random.Random(305249)
    for _ in range(500):
        if token_ids:
            left, right = _random_pair(rng, [-(2**65), -1, 0, 1, 2, 2**65])
        else:
            left_chars, right_chars = _random_pair(rng, list("aa\0é雪🙂𐀀"))
            left, right = "".join(left_chars), "".join(right_chars)
        before = deepcopy((left, right))
        expected = _linear_prefix(left, right)
        assert tokenization._common_prefix_length(left, right) == expected
        assert tokenization._common_prefix_length(right, left) == expected
        assert (left, right) == before


@pytest.mark.parametrize("token_ids", [False, True])
def test_every_prefix_boundary_across_small_and_large_inputs(token_ids):
    for length in (1, 2, 3, 7, 16, 31, 64, 129, 1025):
        shared = "a" * length
        boundaries = (
            range(length + 1) if length < 130 else (0, 1, 511, 512, 1023, 1024, 1025)
        )
        for end in boundaries:
            left = _sequence(shared, token_ids)
            for text in (shared[:end], shared[:end] + "b" + shared[end:]):
                right = _sequence(text, token_ids)
                assert tokenization._common_prefix_length(left, right) == end


def _render(messages, *, add_generation_prompt, hint=""):
    result = "".join(
        ("<a>" + message["content"] + "§")
        if message["role"] == "assistant"
        else ("<u>" + message["content"] + "</u>")
        for message in messages
    )
    return result + ("<a>" + hint if add_generation_prompt else "")


@pytest.mark.parametrize("token_ids", [False, True])
@pytest.mark.parametrize("hint", ["", "↦generation-only"])
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_spans_and_token_masks_preserve_linear_oracle(
    monkeypatch, token_ids, hint, add_generation_prompt
):
    messages = [
        {"role": "user", "content": "雪🙂" * 256},
        {"role": "assistant", "content": "repeat🙂repeat"},
        {"role": "tool", "content": "repeat🙂repeat"},
        {"role": "assistant", "content": "e\u0301\0final"},
    ]

    def render(selected_messages, *, add_generation_prompt):
        return _sequence(
            _render(
                selected_messages,
                add_generation_prompt=add_generation_prompt,
                hint=hint,
            ),
            token_ids,
        )

    rendered = render(messages, add_generation_prompt=add_generation_prompt)
    before = deepcopy((messages, rendered))
    calls = []
    original = tokenization._common_prefix_length

    def observed(left, right):
        calls.append((left, right))
        return original(left, right)

    monkeypatch.setattr(tokenization, "_common_prefix_length", observed)
    actual = tokenization._assistant_char_spans(
        messages, rendered, render, add_generation_prompt=add_generation_prompt
    )
    assert len(calls) == (4 if hint else 2)  # Both shared/start call sites with hint.
    decoded = [
        "".join(map(chr, rendered[start:end])) if token_ids else rendered[start:end]
        for start, end in actual
    ]
    assert decoded == ["repeat🙂repeat§", "e\u0301\0final§"]
    monkeypatch.setattr(tokenization, "_common_prefix_length", _linear_prefix)
    assert actual == tokenization._assistant_char_spans(
        messages, rendered, render, add_generation_prompt=add_generation_prompt
    )
    if token_ids:
        expected = [
            any(start <= index < end for start, end in actual)
            for index in range(len(rendered))
        ]
        monkeypatch.setattr(tokenization, "_common_prefix_length", original)
        assert (
            tokenization._assistant_token_mask_from_ids(
                messages, rendered, render, add_generation_prompt=add_generation_prompt
            )
            == expected
        )
    assert (messages, rendered) == before


@pytest.mark.parametrize("token_ids", [False, True])
def test_rewritten_final_turn_uses_unique_suffix_fallback(monkeypatch, token_ids):
    messages = [
        {"role": "user", "content": "question雪"},
        {"role": "assistant", "content": "first🙂"},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "final🙂"},
    ]

    def render(selected_messages, *, add_generation_prompt):
        multiple = sum(item["role"] == "assistant" for item in selected_messages) > 1
        text = "".join(
            "<a>" + item["content"] + "END"
            if item["role"] == "assistant"
            else "<tool>" + item["content"] + ("END" if multiple else "CALL")
            if item["role"] == "tool"
            else "<u>" + item["content"]
            for item in selected_messages
        ) + ("<a>" if add_generation_prompt else "")
        return _sequence(text, token_ids)

    rendered = render(messages, add_generation_prompt=False)
    before = deepcopy((messages, rendered))
    suffix = tokenization._unique_prompt_suffix_end
    fallback_calls = []

    def observed(*args, **kwargs):
        fallback_calls.append(True)
        return suffix(*args, **kwargs)

    monkeypatch.setattr(tokenization, "_unique_prompt_suffix_end", observed)
    actual = tokenization._assistant_char_spans(
        messages, rendered, render, add_generation_prompt=False
    )
    assert fallback_calls
    assert [rendered[a:b] for a, b in actual] == [
        _sequence("first🙂END", token_ids),
        _sequence("final🙂END", token_ids),
    ]
    monkeypatch.setattr(tokenization, "_common_prefix_length", _linear_prefix)
    assert actual == tokenization._assistant_char_spans(
        messages, rendered, render, add_generation_prompt=False
    )
    assert (messages, rendered) == before


@pytest.mark.parametrize(
    ("values", "rendered", "error"),
    [
        (("R", "P?", "S!"), "S!", "Cannot map assistant message 1"),
        (("R", "R?<a>", "Q"), "RAN", "Assistant message 1 is not anchored"),
    ],
)
def test_unmappable_span_errors_match_linear_oracle(
    monkeypatch, values, rendered, error
):
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    def render(selected_messages, *, add_generation_prompt):
        return (
            values[2]
            if len(selected_messages) == 2
            else values[1]
            if add_generation_prompt
            else values[0]
        )

    before = deepcopy(messages)
    with pytest.raises(ValueError, match=error) as actual:
        tokenization._assistant_char_spans(
            messages, rendered, render, add_generation_prompt=False
        )
    monkeypatch.setattr(tokenization, "_common_prefix_length", _linear_prefix)
    with pytest.raises(ValueError) as expected:
        tokenization._assistant_char_spans(
            messages, rendered, render, add_generation_prompt=False
        )
    assert str(actual.value) == str(expected.value)
    assert messages == before


@pytest.mark.parametrize("token_ids", [False, True])
def test_overlap_error_is_preserved(monkeypatch, token_ids):
    messages = [
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "content": "a"},
    ]

    def render(selected_messages, *, add_generation_prompt):
        return _sequence("P" if add_generation_prompt else "Paa", token_ids)

    for prefix in (tokenization._common_prefix_length, _linear_prefix):
        monkeypatch.setattr(tokenization, "_common_prefix_length", prefix)
        with pytest.raises(ValueError, match="spans overlap"):
            tokenization._assistant_char_spans(
                messages,
                _sequence("Paa", token_ids),
                render,
                add_generation_prompt=False,
            )


@pytest.mark.parametrize("fallback", [None, TypeError, KeyError, ValueError])
@pytest.mark.parametrize("hint", ["", "↦generation-only"])
def test_history_flags_match_linear_oracle_through_text_and_id_paths(
    monkeypatch, fallback, hint
):
    class Tokenizer:
        def __call__(self, text, **kwargs):
            if fallback is ValueError and "<" in text:
                raise ValueError("segmented text encoding unavailable")
            return list(map(ord, text))

        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **kwargs
        ):
            if not tokenize and fallback in (TypeError, KeyError):
                raise fallback("text rendering unavailable")
            text = _render(
                messages, add_generation_prompt=add_generation_prompt, hint=hint
            )
            return list(map(ord, text)) if tokenize else text

    history = tr.ChatCompletionsHistory(
        model="test/prefix-oracle",
        messages=cast(
            Any,
            [
                {"role": "user", "content": "雪🙂"},
                {"role": "assistant", "content": "answer🙂"},
            ],
        ),
        message_sources=[None, None],
    )
    before = history.model_dump()
    actual = history.tokenize(tokenizer=Tokenizer())
    monkeypatch.setattr(tokenization, "_common_prefix_length", _linear_prefix)
    expected = history.tokenize(tokenizer=Tokenizer())
    assert actual.model_dump() == expected.model_dump()
    assert (
        "".join(
            chr(token)
            for token, flag in zip(actual.tokens, actual.flags)
            if flag & tr.TokenFlag.ASSISTANT
        )
        == "answer🙂§"
    )
    assert not any(flag & tr.TokenFlag.SAMPLED for flag in actual.flags)
    assert history.model_dump() == before


@pytest.mark.parametrize("token_ids", [False, True])
def test_rewritten_completion_uses_removed_message_fallback(monkeypatch, token_ids):
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    def render(selected_messages, *, add_generation_prompt):
        return _sequence("Px" if len(selected_messages) == 2 else "P", token_ids)

    rendered = _sequence("Py", token_ids)
    before = deepcopy((messages, rendered))
    original = tokenization._removed_message_span
    calls = []

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(tokenization, "_removed_message_span", observed)
    actual = tokenization._assistant_char_spans(
        messages, rendered, render, add_generation_prompt=False
    )
    assert calls and actual == [(1, 2)]
    monkeypatch.setattr(tokenization, "_common_prefix_length", _linear_prefix)
    assert (
        tokenization._assistant_char_spans(
            messages, rendered, render, add_generation_prompt=False
        )
        == actual
    )
    assert (messages, rendered) == before


@pytest.mark.parametrize(
    "messages",
    [[], [{"role": "user", "content": "u"}], [{"role": "assistant", "content": ""}]],
)
def test_no_assistant_or_no_generated_suffix_has_no_span(messages):
    def render(selected_messages, *, add_generation_prompt):
        return "same"

    assert (
        tokenization._assistant_char_spans(
            messages, "same", render, add_generation_prompt=False
        )
        == []
    )
