import sys
import types
from typing import cast

import pytest
from transformers.tokenization_utils_base import BatchEncoding

import art
from art.preprocessing.tokenize import tokenize_sft_batch, tokenize_trajectory
from art.trajectories import History, Trajectory
from art.types import MessagesAndChoices

pytest.importorskip("torch")
pytest.importorskip("transformers")


class _FakeTokenizer:
    vocab_size = 256
    eos_token = "\x00"
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        **kwargs,
    ):
        del tools, kwargs
        rendered = "".join(
            f"<{message['role']}>{message.get('content', '')}" for message in messages
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

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

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


def test_tokenize_trajectory_accepts_batchencoding_chat_template_output() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "OK"},
        ],
    )
    history = History(messages_and_choices=messages)
    trajectory = Trajectory(messages_and_choices=messages, reward=1.0)

    result = tokenize_trajectory(
        tokenizer=tokenizer,  # type: ignore[arg-type]
        image_processor=None,
        history=history,
        advantage=1.0,
        allow_training_without_logprobs=True,
        trajectory=trajectory,
    )

    assert result is not None
    assistant_ids = [
        token_id
        for token_id, mask in zip(result.token_ids, result.assistant_mask)
        if mask
    ]
    assert assistant_ids == tokenizer.encode("OK", add_special_tokens=False)


def test_tokenize_sft_batch_accepts_batchencoding_chat_template_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer()

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth_zoo = types.ModuleType("unsloth_zoo")
    fake_dataset_utils = types.ModuleType("unsloth_zoo.dataset_utils")

    def _train_on_responses_only(**kwargs):
        del kwargs

        def _labels_fn(batch):
            return {"labels": [list(batch["input_ids"][0])]}

        return _labels_fn

    fake_dataset_utils.train_on_responses_only = _train_on_responses_only  # type: ignore[attr-defined]
    fake_unsloth_zoo.dataset_utils = fake_dataset_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setitem(sys.modules, "unsloth_zoo", fake_unsloth_zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.dataset_utils", fake_dataset_utils)

    trajectory = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[trajectory],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        instruction_part="<user>",
        response_part="<assistant>",
    )

    expected_ids = tokenizer.encode(
        tokenizer.apply_chat_template(
            trajectory.messages_and_choices,
            tokenize=False,
            add_generation_prompt=False,
        ),
        add_special_tokens=False,
    )

    assert batch.trajectory_tensors[0]["input_ids"].tolist() == [expected_ids]
    assert batch.trajectory_tensors[0]["attention_mask"].tolist() == [
        [1] * len(expected_ids)
    ]
    assert batch.num_trainable_tokens == len(expected_ids)
