import json

import pytest
from transformers import AutoTokenizer

from art.preprocessing.tokenize import tokenize_sft_batch
from art.trajectories import Trajectory

QWEN_LAST_ASSISTANT_TEMPLATE = r"""{%- for message in messages %}
{%- if message.role == 'assistant' %}
{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}
{{- message.content or '' }}
{%- for tool_call in message.tool_calls or [] %}
{%- set tool_call = tool_call.function %}
{{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
{%- for name, value in tool_call.arguments|items %}
{{- '<parameter=' + name + '>\n' + value + '\n</parameter>\n' }}
{%- endfor %}
{{- '</function>\n</tool_call>' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}
{%- endif %}"""


def test_qwen_last_assistant_tool_call_tokenization() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.6-35B-A3B",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Qwen3.6 tokenizer is not cached")
    tokenizer.chat_template = QWEN_LAST_ASSISTANT_TEMPLATE
    trajectory = Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "first system"},
            {"role": "system", "content": "final hint"},
            {"role": "user", "content": "Send the update"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "send_whatsapp",
                            "arguments": json.dumps({"body": "parking info"}),
                        },
                    }
                ],
            },
        ]
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[trajectory],
        learning_rate=1e-5,
        tokenizer=tokenizer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        assistant_turns="last",
    )

    input_ids = batch.trajectory_tensors[0]["input_ids"][0].tolist()
    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    target_start = next(index for index, label in enumerate(labels) if label != -100)
    target_ids = labels[target_start:]

    assert len(target_ids) == 32
    assert target_ids[0] == 271
    assert target_ids[-2:] == [tokenizer.eos_token_id, 198]
    assert tokenizer.eos_token_id == 248046
    assert input_ids[target_start - 1 : target_start + 1] == [271, 271]
    assert labels[target_start - 1 : target_start + 1] == [-100, 271]
    assert tokenizer.decode(target_ids) == (
        "\n\n<tool_call>\n<function=send_whatsapp>\n"
        "<parameter=body>\nparking info\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>\n"
    )


def test_qwen_last_assistant_preserves_stock_generation_boundary() -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.6-35B-A3B",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Qwen3.6 tokenizer is not cached")
    trajectory = Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hi"},
            {"role": "assistant", "content": "Hi"},
        ]
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[trajectory],
        learning_rate=1e-5,
        tokenizer=tokenizer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        chat_template_kwargs={"enable_thinking": True},
        assistant_turns="last",
    )

    input_ids = batch.trajectory_tensors[0]["input_ids"][0].tolist()
    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    target_start = next(index for index, label in enumerate(labels) if label != -100)
    target_ids = labels[target_start:]

    assert input_ids[target_start - 1 : target_start + 1] == [198, 198]
    assert labels[target_start - 1 : target_start + 1] == [-100, 198]
    assert tokenizer.decode(target_ids) == "\n</think>\n\nHi<|im_end|>\n"
