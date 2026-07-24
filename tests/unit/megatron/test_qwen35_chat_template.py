from hashlib import sha256
from types import SimpleNamespace
from typing import Any, cast

from transformers.utils.chat_template_utils import render_jinja_template

from art.megatron.model_support.handlers.qwen3_5 import QWEN3_5_DENSE_HANDLER
from art.megatron.model_support.qwen3_5_chat_template import (
    QWEN3_5_DISABLE_THINKING_MULTI_SYSTEM_CHAT_TEMPLATE,
)
from art.megatron.model_support.tokenizer import (
    configure_tokenizer_for_model_support,
)
from art.megatron.service import MegatronService


def test_qwen35_default_renders_late_system_hint_after_tool_history() -> None:
    template = QWEN3_5_DENSE_HANDLER.default_chat_template()
    assert template == QWEN3_5_DISABLE_THINKING_MULTI_SYSTEM_CHAT_TEMPLATE
    assert sha256(template.encode()).hexdigest() == (
        "4365cc0eaa2d39386ec2388abffbec318695e7fd61b30142a918ce5bf5a86cf1"
    )
    messages = [
        {"role": "system", "content": "Base booking policy."},
        {"role": "user", "content": "Find dinner in Paris."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "Le Jules Verne"},
        {"role": "assistant", "content": "I found one."},
        {"role": "user", "content": "Book it."},
        {"role": "system", "content": "Feedback: confirm the date first."},
    ]

    rendered, _ = render_jinja_template(
        conversations=[messages],
        chat_template=template,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    text = rendered[0]

    expected_fragments = [
        "<|im_start|>system\nBase booking policy.<|im_end|>",
        "<|im_start|>user\nFind dinner in Paris.<|im_end|>",
        "<function=search>",
        "<tool_response>\nLe Jules Verne\n</tool_response>",
        "<|im_start|>assistant\nI found one.<|im_end|>",
        "<|im_start|>user\nBook it.<|im_end|>",
        (
            "<|im_start|>system\nFeedback: confirm the date first."
            "<|im_end|>\n<|im_start|>assistant"
        ),
    ]
    positions = [text.index(fragment) for fragment in expected_fragments]
    assert positions == sorted(positions)
    assert text.endswith("<think>\n\n</think>\n\n")


def test_qwen35_training_tokenizer_uses_default_unless_explicitly_overridden() -> None:
    tokenizer = SimpleNamespace(chat_template="upstream-template")

    configured = configure_tokenizer_for_model_support(
        cast(Any, tokenizer),
        base_model="Qwen/Qwen3.5-4B",
        internal_config={},
    )
    assert (
        configured.chat_template == QWEN3_5_DISABLE_THINKING_MULTI_SYSTEM_CHAT_TEMPLATE
    )

    tokenizer.chat_template = "explicit-template"
    configured = configure_tokenizer_for_model_support(
        cast(Any, tokenizer),
        base_model="Qwen/Qwen3.5-4B",
        internal_config={"chat_template": "explicit-template"},
    )
    assert configured.chat_template == "explicit-template"


def test_qwen35_managed_vllm_default_and_explicit_override_are_consistent() -> None:
    service = object.__new__(MegatronService)
    service.base_model = "Qwen/Qwen3.5-4B"
    service.config = {}

    defaults = service._runtime_server_args(None)
    assert (
        defaults["chat_template"] == QWEN3_5_DISABLE_THINKING_MULTI_SYSTEM_CHAT_TEMPLATE
    )

    overridden = service._runtime_server_args(
        cast(
            Any,
            {
                "server_args": {
                    "chat_template": "explicit-template",
                    "tool_call_parser": "explicit-parser",
                }
            },
        )
    )
    assert overridden["chat_template"] == "explicit-template"
    assert overridden["tool_call_parser"] == "explicit-parser"
