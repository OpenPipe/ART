from typing import Any, cast

import pytest
from transformers.tokenization_utils_base import BatchEncoding

from art.preprocessing.tokenize import tokenize_sft_batch
from art.trajectories import Trajectory
from art.types import (
    MessagesAndChoices,
    SFTAllLossMask,
    SFTLastAssistantLossMask,
    SFTMessageMaskLossMask,
    SFTNoneLossMask,
    SFTRolesLossMask,
)
from art.utils.model_config import get_instruction_response_parts

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


class _QwenDisableThinkingTokenizer(_FakeTokenizer):
    chat_template = "<|im_start|>{% if enable_thinking is false %}<think>{% endif %}"

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
            role = message["role"]
            content = message.get("content", "")
            if role == "assistant":
                rendered_parts.append(
                    f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}"
                )
            else:
                rendered_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
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


def test_tokenize_sft_batch_loss_mask_last_assistant() -> None:
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
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
                loss_mask=SFTLastAssistantLossMask(),
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "second"
    assert batch.num_trainable_tokens == len("second")


def test_qwen_disable_thinking_config_excludes_scaffold_from_loss_mask() -> None:
    tokenizer = _QwenDisableThinkingTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "B"},
            {"role": "assistant", "content": "second"},
        ],
    )
    instruction_part, response_part = get_instruction_response_parts(
        "Qwen/Qwen3.5-35B-A3B",
        tokenizer,  # type: ignore[arg-type]
    )

    assert response_part == "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert (
        get_instruction_response_parts(
            "Qwen/Qwen3.6-27B",
            tokenizer,  # type: ignore[arg-type]
        )[1]
        == response_part
    )
    batch = tokenize_sft_batch(
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
                loss_mask=SFTLastAssistantLossMask(),
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part=instruction_part,
        response_part=response_part,
        chat_template_kwargs={"enable_thinking": False},
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "second"
    assert batch.num_trainable_tokens == len("second")


def test_tokenize_sft_batch_loss_mask_message_mask() -> None:
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
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
                loss_mask=SFTMessageMaskLossMask(mask=[False, True, False, False]),
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "first"


def test_tokenize_sft_batch_loss_mask_none() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
                loss_mask=SFTNoneLossMask(),
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    assert batch.num_trainable_tokens == 0


def test_tokenize_sft_batch_loss_mask_roles_assistant_default() -> None:
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
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
                loss_mask=SFTRolesLossMask(),
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "firstsecond"


def test_tokenize_sft_batch_loss_mask_all_does_not_label_first_token() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
                loss_mask=SFTAllLossMask(),
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    assert labels[0] == -100
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "system>S<user>Afirst"


def test_tokenize_sft_batch_rejects_bad_message_mask_length() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
        ],
    )

    with pytest.raises(ValueError, match="message_mask length"):
        tokenize_sft_batch(
            trajectory_batch=[
                Trajectory(
                    messages_and_choices=messages,
                    loss_mask=SFTMessageMaskLossMask(mask=[False]),
                )
            ],
            learning_rate=1e-5,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            instruction_part="<user>",
            response_part="<assistant>",
        )
