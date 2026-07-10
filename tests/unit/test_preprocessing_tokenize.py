import json
from typing import Any, cast

import pydantic
import pytest
from transformers.tokenization_utils_base import BatchEncoding

from art.preprocessing.tokenize import tokenize_sft_batch
from art.trajectories import Trajectory
from art.types import MessagesAndChoices, TrainSFTConfig

pytest.importorskip("torch")
pytest.importorskip("transformers")


class _FakeTokenizer:
    chat_template = ""
    eos_token = "\x00"
    eos_token_id = 0

    def __init__(self) -> None:
        self.apply_chat_template_kwargs: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        **kwargs,
    ):
        del tools
        self.apply_chat_template_kwargs.append(dict(kwargs))
        rendered_parts = []
        for message in messages:
            tool_calls = "".join(
                f"<tool>{tool_call['function']['name']}:{tool_call['function']['arguments']}"
                for tool_call in message.get("tool_calls", [])
            )
            rendered_parts.append(
                f"<{message['role']}>{tool_calls}{message.get('content', '')}"
            )
        rendered = "".join(rendered_parts)
        if not tokenize:
            return rendered
        token_ids = self.encode(rendered, add_special_tokens=False)
        if return_dict is False:
            return token_ids
        return BatchEncoding(
            {
                "input_ids": token_ids,
                "attention_mask": [1] * len(token_ids),
            }
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def __call__(self, text: str, add_special_tokens: bool = False):
        return type(
            "TokenizedText",
            (),
            {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)},
        )()

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            return chr(token_ids)
        return "".join(chr(token_id) for token_id in token_ids)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        if isinstance(tokens, str) and len(tokens) == 1:
            return ord(tokens)
        return self.eos_token_id


class _LastAssistantTokenizer(_FakeTokenizer):
    chat_template = "{{ tool_call.arguments|items }}"
    eos_token = "<eos>"

    assistant_prefix = "<assistant><think>\n\n</think>\n\n"

    def _render_chat(self, messages, *, add_generation_prompt: bool) -> str:
        rendered_parts: list[str] = []
        for message in messages:
            role = message["role"]
            content = message.get("content") or ""
            if role != "assistant":
                rendered_parts.append(f"<{role}>{content}{self.eos_token}\n")
                continue

            rendered_parts.append(f"{self.assistant_prefix}{content}")
            for tool_call in message.get("tool_calls", []):
                function = tool_call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                rendered_parts.append(
                    f"\n\n<tool_call>\n<function={function['name']}>\n"
                )
                for name, value in arguments.items():
                    rendered_parts.append(
                        f"<parameter={name}>\n{value}\n</parameter>\n"
                    )
                rendered_parts.append("</function>\n</tool_call>")
            rendered_parts.append(f"{self.eos_token}\n")

        if add_generation_prompt:
            rendered_parts.append(self.assistant_prefix)
        return "".join(rendered_parts)

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        add_generation_prompt=False,
        **kwargs,
    ):
        del tools
        self.apply_chat_template_kwargs.append(dict(kwargs))
        rendered = self._render_chat(
            messages,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return rendered
        token_ids = self.encode(rendered, add_special_tokens=False)
        if return_dict is False:
            return token_ids
        return BatchEncoding(
            {
                "input_ids": token_ids,
                "attention_mask": [1] * len(token_ids),
            }
        )


class _BoundaryMergingTokenizer(_LastAssistantTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        token_ids: list[int] = []
        index = 0
        while index < len(text):
            if text.startswith("\n\n\n\n", index):
                token_ids.append(987)
                index += 4
            elif text.startswith("\n\n", index):
                token_ids.append(271)
                index += 2
            else:
                token_ids.append(ord(text[index]))
                index += 1
        return token_ids


class _NonPrefixStableTokenizer(_LastAssistantTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        rendered = super().apply_chat_template(*args, **kwargs)
        if kwargs.get("tokenize", True) is False and not kwargs.get(
            "add_generation_prompt", False
        ):
            return f"<changed>{rendered}"
        return rendered


def test_tokenize_sft_batch_masks_response_tokens_without_unsloth_import() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OK"},
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages, reward=1.0)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "OK"
    assert batch.num_trainable_tokens == 2


def test_tokenize_sft_batch_all_trains_every_assistant_turn() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "B"},
            {"role": "assistant", "content": "second"},
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
        assistant_turns="all",
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "firstsecond"


def test_tokenize_sft_batch_passes_chat_template_kwargs() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OK"},
        ],
    )

    tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages, reward=1.0)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
        chat_template_kwargs={
            "enable_thinking": False,
            "preserve_thinking": True,
        },
    )

    assert tokenizer.apply_chat_template_kwargs[-1]["enable_thinking"] is False
    assert tokenizer.apply_chat_template_kwargs[-1]["preserve_thinking"] is True


def test_train_sft_config_selects_all_or_last_assistant_turns() -> None:
    assert TrainSFTConfig().assistant_turns == "all"
    assert TrainSFTConfig(assistant_turns="last").assistant_turns == "last"
    with pytest.raises(pydantic.ValidationError, match="assistant_turns"):
        TrainSFTConfig(assistant_turns="first")  # ty:ignore[invalid-argument-type]


def test_tokenize_sft_batch_trains_only_final_assistant_turn() -> None:
    tokenizer = _LastAssistantTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "system", "content": "first system"},
            {"role": "system", "content": "second system"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "system", "content": "final hint"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "final answer"},
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
        assistant_turns="last",
    )

    input_ids = batch.trajectory_tensors[0]["input_ids"][0].tolist()
    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    target_ids = tokenizer.encode("final answer<eos>\n")
    target_start = len(input_ids) - len(target_ids)

    assert labels[:target_start] == [-100] * target_start
    assert labels[target_start:] == target_ids
    assert input_ids[target_start:] == target_ids
    assert tokenizer.decode(target_ids) == "final answer<eos>\n"
    assert batch.num_trainable_tokens == len(target_ids)


def test_tokenize_sft_batch_trains_final_tool_call_and_turn_end() -> None:
    tokenizer = _LastAssistantTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
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
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
        assistant_turns="last",
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    target_text = tokenizer.decode(
        [token_id for token_id in labels if token_id != -100]
    )
    assert target_text == (
        "\n\n<tool_call>\n<function=send_whatsapp>\n"
        "<parameter=body>\nparking info\n</parameter>\n"
        "</function>\n</tool_call><eos>\n"
    )
    assert tokenizer.assistant_prefix not in target_text
    assert target_text.endswith("<eos>\n")


def test_tokenize_sft_batch_preserves_prompt_target_token_boundary() -> None:
    tokenizer = _BoundaryMergingTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
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
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
        assistant_turns="last",
    )

    input_ids = batch.trajectory_tensors[0]["input_ids"][0].tolist()
    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    target_start = next(index for index, label in enumerate(labels) if label != -100)
    assert input_ids[target_start - 1 : target_start + 1] == [271, 271]
    assert labels[target_start - 1 : target_start + 1] == [-100, 271]
    assert 987 not in input_ids


def test_tokenize_sft_batch_last_requires_final_assistant_message() -> None:
    tokenizer = _LastAssistantTokenizer()
    with pytest.raises(ValueError, match="final message.*assistant"):
        tokenize_sft_batch(
            trajectory_batch=[
                Trajectory(messages_and_choices=[{"role": "user", "content": "hello"}])
            ],
            learning_rate=1e-5,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            instruction_part="<user>",
            response_part="<assistant>",
            assistant_turns="last",
        )


def test_tokenize_sft_batch_last_rejects_non_prefix_stable_template() -> None:
    tokenizer = _NonPrefixStableTokenizer()
    with pytest.raises(ValueError, match="does not extend its generation prompt"):
        tokenize_sft_batch(
            trajectory_batch=[
                Trajectory(
                    messages_and_choices=[
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "world"},
                    ]
                )
            ],
            learning_rate=1e-5,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            instruction_part="<user>",
            response_part="<assistant>",
            assistant_turns="last",
        )
