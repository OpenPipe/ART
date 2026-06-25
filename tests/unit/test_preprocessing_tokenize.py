from typing import Any, cast

import pytest
from transformers.tokenization_utils_base import BatchEncoding

from art.preprocessing.tokenize import tokenize_sft_batch
from art.trajectories import Trajectory
from art.types import MessagesAndChoices

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
