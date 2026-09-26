"""Byte-exact goldens for named tokenization cases.

Each case tokenizes a small history and compares ``(model, tokens, logprobs,
flags)`` with ``tokenize_golden_cases.json``. Failures name the first differing
field; full token lists are stored so a diff of the JSON shows the change.

Regenerate with ``ART_TOKENIZE_GOLDEN=update pytest tests/unit/trajectories/test_tokenize_golden.py``.
The companion ``conftest.py`` trace mode covers every other test in this
directory; these cases pin the paths the audit found repeatedly re-fixed:

``fake/*`` (deterministic character tokenizer, always run)
  chat_multi_turn_tool_calls_reasoning   render path: system/user/assistant with
                                          reasoning, tool_calls, tool result,
                                          multi-turn (#910 hidden demonstration text)
  chat_thinking_enabled                   same history, ``enable_thinking=True``
  chat_thinking_off_literal_think_markers thinking-off history whose assistant text
                                          contains literal ``<think>`` (#967)
  chat_multi_part_assistant_content       assistant ``content`` as a list of parts (#904)
  exchange_exact_sampled_multi_turn       exact projected path over three sampled
                                          exchanges (#886 sampled source identity)
  exchange_length_stop_nonterminal        ``finish_reason="length"`` with a synthetic
                                          stop (#829 #830 #847 #871)
  exchange_mixed_stop_captured            a length-stopped exchange whose captured
                                          history is re-prompted by a stopped one;
                                          exact coverage ends at the duplicated
                                          terminator (#882)
  exchange_nan_logprobs                   sampled logprobs containing ``NaN`` (#903)
  exchange_inexact_length_stop            length stop without token ids (inexact
                                          assistant attribution)

``qwen3/*`` (real ``Qwen/Qwen3-0.6B`` tokenizer; skipped when not cached offline)
  chat_multi_turn_tool_calls_reasoning, chat_thinking_enabled,
  chat_thinking_off_literal_think_markers, chat_multi_part_assistant_content
                                          the same render-path histories through
                                          ART's preserved-thinking Qwen3 template
"""

from __future__ import annotations

from collections.abc import Callable
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any

from _tokenize_golden import (
    FIELDS,
    describe_mismatch,
    field_values,
    input_digest,
    load_json,
    mode,
    output_digest,
)
from openai.types.chat import ChatCompletion
import pytest
from test_tokenize import (
    _QWEN_LIKE_TEMPLATE,
    _chat_exchange,
    _QwenLikeCharacterTokenizer,
    _StopTokenizer,
)

import art
import art.trajectories as tr
from art.trajectories import TrajectoryExchanges

GOLDEN_CASES = Path(__file__).with_name("tokenize_golden_cases.json")
_QWEN3 = "Qwen/Qwen3-0.6B"

_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is 2+2? Use the tool."},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I should call add.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call-1", "content": "4"},
    {"role": "assistant", "content": "It is 4.", "reasoning_content": "Done."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "6."},
]
_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]
_LITERAL_THINK: list[dict[str, Any]] = [
    {"role": "user", "content": "Echo the tag."},
    {"role": "assistant", "content": "The tag is <think> and it closes with </think>."},
]
_MULTI_PART: list[dict[str, Any]] = [
    {"role": "user", "content": [{"type": "text", "text": "Two parts please."}]},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "First part. "},
            {"type": "text", "text": "Second part."},
        ],
    },
]


def _chat_history(
    messages: list[dict[str, Any]],
    *,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tr.ChatCompletionsHistory:
    return tr.ChatCompletionsHistory(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        message_sources=[None] * len(messages),
        tools=tools,  # type: ignore[arg-type]
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
    )


def _fake(
    messages: list[dict[str, Any]], **kwargs: Any
) -> Callable[[Any], tr.TokenizedHistory]:
    def build(_: Any) -> tr.TokenizedHistory:
        return _chat_history(
            messages, model="test/qwen", chat_template=_QWEN_LIKE_TEMPLATE, **kwargs
        ).tokenize(tokenizer=_QwenLikeCharacterTokenizer())

    return build


def _qwen3(
    messages: list[dict[str, Any]], **kwargs: Any
) -> Callable[[Any], tr.TokenizedHistory]:
    def build(tokenizer: Any) -> tr.TokenizedHistory:
        if tokenizer is None:
            pytest.skip(f"{_QWEN3} tokenizer is not cached offline")
        return _chat_history(messages, model=_QWEN3, **kwargs).tokenize(
            base_model=_QWEN3, tokenizer=tokenizer
        )

    return build


def _with_finish_reason(exchange: Any, finish_reason: str) -> Any:
    exchange.response.choices[0].finish_reason = finish_reason
    return exchange


def _exchanges(*exchanges: Any) -> Callable[[Any], tr.TokenizedHistory]:
    def build(_: Any) -> tr.TokenizedHistory:
        return art.Trajectory(
            exchanges=TrajectoryExchanges(chat_completions=list(exchanges))
        ).tokenize(tokenizer=_StopTokenizer())

    return build


def _nan_exchange() -> Any:
    exchange = _chat_exchange([1], [2, 3, 9])
    data = exchange.response.model_dump(mode="python")
    content = data["choices"][0]["logprobs"]["content"]
    content[1]["logprob"] = math.nan
    exchange.response = ChatCompletion.model_validate(data)
    return exchange


def _inexact_length_exchange() -> Any:
    exchange = _chat_exchange([1], [2])
    data = exchange.response.model_dump(mode="python")
    choice = data["choices"][0]
    choice["finish_reason"] = "length"
    choice.pop("prompt_token_ids")
    choice.pop("token_ids")
    choice["logprobs"] = None
    exchange.response = ChatCompletion.model_validate(data)
    return exchange


CASES: dict[str, Callable[[Any], tr.TokenizedHistory]] = {
    "fake/chat_multi_turn_tool_calls_reasoning": _fake(_MESSAGES, tools=_TOOLS),
    "fake/chat_thinking_enabled": _fake(
        _MESSAGES, tools=_TOOLS, chat_template_kwargs={"enable_thinking": True}
    ),
    "fake/chat_thinking_off_literal_think_markers": _fake(
        _LITERAL_THINK, chat_template_kwargs={"enable_thinking": False}
    ),
    "fake/chat_multi_part_assistant_content": _fake(_MULTI_PART),
    "fake/exchange_exact_sampled_multi_turn": _exchanges(
        _chat_exchange([1], [2, 9]),
        _chat_exchange([1, 2, 9, 3], [4, 9], offset=1),
        _chat_exchange([1, 2, 9, 3, 4, 9, 5], [6, 9], offset=2),
    ),
    "fake/exchange_length_stop_nonterminal": _exchanges(
        _with_finish_reason(_chat_exchange([1], [2]), "length")
    ),
    "fake/exchange_mixed_stop_captured": _exchanges(
        _with_finish_reason(_chat_exchange([1], [2, 9]), "length"),
        _chat_exchange([1, 2, 9, 3], [4, 9], offset=1),
    ),
    "fake/exchange_nan_logprobs": _exchanges(_nan_exchange()),
    "fake/exchange_inexact_length_stop": _exchanges(_inexact_length_exchange()),
    "qwen3/chat_multi_turn_tool_calls_reasoning": _qwen3(_MESSAGES, tools=_TOOLS),
    "qwen3/chat_thinking_enabled": _qwen3(
        _MESSAGES, tools=_TOOLS, chat_template_kwargs={"enable_thinking": True}
    ),
    "qwen3/chat_thinking_off_literal_think_markers": _qwen3(
        _LITERAL_THINK, chat_template_kwargs={"enable_thinking": False}
    ),
    "qwen3/chat_multi_part_assistant_content": _qwen3(_MULTI_PART),
}


@pytest.fixture(scope="module")
def qwen3_tokenizer() -> Any:
    pytest.importorskip("transformers")
    from art.tokenizer import get_tokenizer

    try:
        return get_tokenizer(_QWEN3, local_files_only=True)
    except Exception:  # noqa: BLE001 - any loader failure means "not cached"
        return None


def _record(case_id: str, tokenized: tr.TokenizedHistory) -> dict[str, Any]:
    digest = output_digest(
        tokenized,
        input_hash=input_digest(
            tokenized.history,
            model=tokenized.model,
            base_model=None,
            tokenizer=None,
            chat_template=None,
            chat_template_kwargs=None,
        ),
    )
    return {"digest": digest, "values": field_values(tokenized)}


def _write_case(case_id: str, record: dict[str, Any]) -> None:
    GOLDEN_CASES.touch()
    with GOLDEN_CASES.open("r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = handle.read()
        golden = json.loads(text) if text.strip() else {}
        golden[case_id] = record
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(golden, sort_keys=True, indent=1) + "\n")


def _first_value_difference(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    for name in FIELDS:
        want, got = expected[name], actual[name]
        if want == got:
            continue
        if isinstance(want, list) and isinstance(got, list):
            for index, (a, b) in enumerate(zip(want, got, strict=False)):
                if a != b:
                    return f"{name}[{index}]: golden {a!r} != now {b!r}"
            return f"{name}: length golden {len(want)} != now {len(got)}"
        return f"{name}: golden {want!r} != now {got!r}"
    return "no field difference (digest inputs differ)"


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_tokenize_golden(case_id: str, qwen3_tokenizer: Any) -> None:
    tokenized = CASES[case_id](qwen3_tokenizer)
    record = _record(case_id, tokenized)
    if mode() == "update":
        _write_case(case_id, record)
        return
    golden = load_json(GOLDEN_CASES)
    assert case_id in golden, (
        f"{case_id} has no golden entry; run with ART_TOKENIZE_GOLDEN=update"
    )
    expected = golden[case_id]
    if expected["digest"]["output"] == record["digest"]["output"]:
        return
    pytest.fail(
        describe_mismatch(case_id, expected["digest"], record["digest"])
        + "\n  "
        + _first_value_difference(expected["values"], record["values"]),
        pytrace=False,
    )


def test_golden_cases_file_is_complete() -> None:
    """Every case has an entry, and no stale entries linger."""

    if mode() == "update":
        pytest.skip("golden cases are being rewritten")
    golden = load_json(GOLDEN_CASES)
    missing = sorted(set(CASES) - set(golden))
    stale = sorted(set(golden) - set(CASES))
    assert not missing and not stale, (
        f"missing goldens: {missing}; stale goldens: {stale}; "
        "run with ART_TOKENIZE_GOLDEN=update"
    )
    for case_id, record in golden.items():
        values = record["values"]
        assert (
            len(values["tokens"]) == len(values["logprobs"]) == len(values["flags"])
        ), case_id
        assert record["digest"]["n"] == len(values["tokens"]), case_id
        assert os.path.getsize(GOLDEN_CASES) < 2_000_000
