from __future__ import annotations

import copy
from datetime import datetime
import json
from typing import Any, cast

import pytest

from art.trajectories._content import content_mask_for_exact_prompt

pytest.importorskip("renderers")
transformers = pytest.importorskip("transformers")


# Every tokenizer load is revision-pinned. The 35B/397B Qwen3.5 repositories
# publish the same tokenizer_config blob as 4B, so they intentionally reuse its
# complete tokenizer snapshot while exercising their own ART model allowlist.
_QWEN_MODELS = [
    (
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        (False, True),
    ),
    (
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        (False, True),
    ),
    (
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "ac9c66cc9b46af7306746a9250f23d47083d689e",
        (False, True),
    ),
    (
        "OpenPipe/Qwen3-14B-Instruct",
        "OpenPipe/Qwen3-14B-Instruct",
        "99e4a359e990d8d00b77ac707c34bbfbc9d8c98a",
        (False,),
    ),
    (
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        (True,),
    ),
    (
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.5-27B",
        "fc05daec18b0a78c049392ed2e771dde82bdf654",
        (True,),
    ),
    (
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        (True,),
    ),
    (
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        (True,),
    ),
    (
        "Qwen/Qwen3.6-27B",
        "Qwen/Qwen3.6-27B",
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        (True,),
    ),
    (
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B",
        "995ad96eacd98c81ed38be0c5b274b04031597b0",
        (True,),
    ),
]

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

_MESSAGES = [
    {"role": "system", "content": "Be precise."},
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should call the tool.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": '{"city":"Paris"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
    {"role": "user", "content": "Summarize."},
]


def _tokenizer(model: str, revision: str) -> Any:
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            local_files_only=True,
        )
    except Exception as exc:
        pytest.skip(f"pinned tokenizer {model}@{revision} is not cached: {exc}")


def _input_ids(value: object) -> list[int]:
    ids = getattr(value, "input_ids", value)
    assert isinstance(ids, list)
    assert all(isinstance(item, int) for item in ids)
    return cast(list[int], ids)


def _engine_messages(
    messages: list[dict[str, Any]], *, parse_tool_arguments: bool
) -> list[dict[str, Any]]:
    messages = copy.deepcopy(messages)
    if parse_tool_arguments and len(messages) > 2:
        function = messages[2]["tool_calls"][0]["function"]
        function["arguments"] = json.loads(function["arguments"])
    return messages


@pytest.mark.parametrize(
    ("model", "tokenizer_model", "revision", "argument_dialects"), _QWEN_MODELS
)
def test_prime_renderer_matches_pinned_qwen_templates_with_full_history(
    model: str,
    tokenizer_model: str,
    revision: str,
    argument_dialects: tuple[bool, ...],
) -> None:
    tokenizer = _tokenizer(tokenizer_model, revision)
    messages = _MESSAGES[:2] if model == "OpenPipe/Qwen3-14B-Instruct" else _MESSAGES
    for parse_tool_arguments in argument_dialects:
        expected = _input_ids(
            tokenizer.apply_chat_template(
                _engine_messages(messages, parse_tool_arguments=parse_tool_arguments),
                tools=_TOOLS,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        mask = content_mask_for_exact_prompt(
            tokenizer=tokenizer,
            base_model=model,
            messages=messages,
            tools=_TOOLS,
            expected_prompt_ids=expected,
            chat_template_kwargs={},
            has_custom_chat_template=False,
        )

        assert mask is not None
        assert len(mask) == len(expected)
        assert any(mask)
        assert not mask[0]


def test_gpt_oss_renderer_matches_pinned_harmony_engine_format() -> None:
    from openai_harmony import (
        Conversation,
        HarmonyEncodingName,
        ReasoningEffort,
        Role,
        SystemContent,
        load_harmony_encoding,
    )
    from openai_harmony import (
        Message as HarmonyMessage,
    )

    revision = "6cee5e81ee83917806bbde320786a8fb61efebee"
    tokenizer = _tokenizer("openai/gpt-oss-20b", revision)
    date = datetime.now().strftime("%Y-%m-%d")
    messages = [{"role": "user", "content": "Hello!"}]
    system = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.MEDIUM)
        .with_conversation_start_date(date)
    )
    encoder = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    expected = encoder.render_conversation_for_training(
        Conversation.from_messages(
            [
                HarmonyMessage.from_role_and_content(Role.SYSTEM, system),
                HarmonyMessage.from_role_and_content(Role.USER, "Hello!"),
            ]
        )
    )
    expected.extend(
        tokenizer.encode(
            "<|start|>assistant<|channel|>analysis<|message|>",
            add_special_tokens=False,
        )
    )

    mask = content_mask_for_exact_prompt(
        tokenizer=tokenizer,
        base_model="openai/gpt-oss-20b",
        messages=messages,
        tools=None,
        expected_prompt_ids=expected,
        chat_template_kwargs={"conversation_start_date": date},
        has_custom_chat_template=False,
    )

    assert mask is not None
    assert len(mask) == len(expected)
    assert any(mask)
    assert not mask[0]


def test_irrelevant_standard_template_kwargs_are_proven_by_exact_parity() -> None:
    model, tokenizer_model, revision, _ = _QWEN_MODELS[0]
    tokenizer = _tokenizer(tokenizer_model, revision)
    messages = [{"role": "user", "content": "hello"}]
    kwargs = {"enable_thinking": True, "thinking_budget": 128}
    expected = _input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
    )

    mask = content_mask_for_exact_prompt(
        tokenizer=tokenizer,
        base_model=model,
        messages=messages,
        tools=None,
        expected_prompt_ids=expected,
        chat_template_kwargs=kwargs,
        has_custom_chat_template=False,
    )

    assert mask is not None
    assert len(mask) == len(expected)
    assert (
        content_mask_for_exact_prompt(
            tokenizer=tokenizer,
            base_model=model,
            messages=messages,
            tools=None,
            expected_prompt_ids=expected,
            chat_template_kwargs={"unqualified_renderer_knob": True},
            has_custom_chat_template=False,
        )
        is None
    )


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_responses_tool_schema_variants_require_exact_parity(engine: str) -> None:
    model, tokenizer_model, revision, _ = _QWEN_MODELS[0]
    tokenizer = _tokenizer(tokenizer_model, revision)
    function = cast(dict[str, Any], _TOOLS[0]["function"])
    engine_tools = cast(list[dict[str, Any]], copy.deepcopy(_TOOLS))
    engine_function = cast(dict[str, Any], engine_tools[0]["function"])
    if engine == "sglang":
        engine_function["strict"] = False
    else:
        engine_tools[0]["function"] = {
            "name": function["name"],
            "parameters": function["parameters"],
            "strict": None,
            "type": "function",
            "allowed_callers": None,
            "defer_loading": None,
            "description": function["description"],
            "output_schema": None,
        }
    expected = _input_ids(
        tokenizer.apply_chat_template(
            _engine_messages(_MESSAGES, parse_tool_arguments=True),
            tools=engine_tools,
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    mask = content_mask_for_exact_prompt(
        tokenizer=tokenizer,
        base_model=model,
        messages=_MESSAGES,
        tools=_TOOLS,
        expected_prompt_ids=expected,
        chat_template_kwargs={},
        has_custom_chat_template=False,
    )

    assert mask is not None
    assert len(mask) == len(expected)


def test_renderer_signal_is_rejected_without_full_exact_id_match() -> None:
    model, tokenizer_model, revision, _ = _QWEN_MODELS[0]
    tokenizer = _tokenizer(tokenizer_model, revision)
    messages = [{"role": "user", "content": "hello"}]
    expected = _input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    mismatches = [expected[:-1], [*expected, -1]]
    for index in range(len(expected)):
        changed = list(expected)
        changed[index] = -1
        mismatches.append(changed)
    for mismatch in mismatches:
        assert (
            content_mask_for_exact_prompt(
                tokenizer=tokenizer,
                base_model=model,
                messages=messages,
                tools=None,
                expected_prompt_ids=mismatch,
                chat_template_kwargs={},
                has_custom_chat_template=False,
            )
            is None
        )
    assert (
        content_mask_for_exact_prompt(
            tokenizer=tokenizer,
            base_model=model,
            messages=messages,
            tools=None,
            expected_prompt_ids=expected,
            chat_template_kwargs={},
            has_custom_chat_template=True,
        )
        is None
    )
