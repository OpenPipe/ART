from typing import Any, cast

import pydantic
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

pytest.importorskip("torch")
pytest.importorskip("transformers")


class _FakeTokenizer:
    chat_template = ""
    eos_token = "\x00"
    eos_token_id = 0

    def __init__(self) -> None:
        self.apply_chat_template_kwargs: list[dict[str, Any]] = []

    def _render_chat(self, messages) -> tuple[str, list[int]]:
        rendered_parts = []
        assistant_masks = []
        for message in messages:
            tool_calls = "".join(
                f"<tool>{tool_call['function']['name']}:{tool_call['function']['arguments']}"
                for tool_call in message.get("tool_calls", [])
            )
            prefix = f"<{message['role']}>"
            body = f"{tool_calls}{message.get('content', '')}"
            rendered_parts.append(f"{prefix}{body}")
            assistant_masks.extend([0] * len(prefix))
            assistant_masks.extend(
                [1 if message["role"] == "assistant" else 0] * len(body)
            )
        return "".join(rendered_parts), assistant_masks

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):
        del tools
        self.apply_chat_template_kwargs.append(dict(kwargs))
        rendered, assistant_masks = self._render_chat(messages)
        if not tokenize:
            return rendered
        token_ids = self.encode(rendered, add_special_tokens=False)
        if return_dict is False:
            return token_ids
        encoding = {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }
        if return_assistant_tokens_mask:
            encoding["assistant_masks"] = assistant_masks
        return BatchEncoding(encoding)

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

    def _render_chat(self, messages) -> tuple[str, list[int]]:
        rendered_parts = []
        assistant_masks = []
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            if role == "assistant":
                prefix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                rendered_parts.append(f"{prefix}{content}")
                assistant_masks.extend([0] * len(prefix))
                assistant_masks.extend([1] * len(content))
            else:
                rendered = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                rendered_parts.append(rendered)
                assistant_masks.extend([0] * len(rendered))
        return "".join(rendered_parts), assistant_masks

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=None,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):
        del tools
        self.apply_chat_template_kwargs.append(dict(kwargs))
        rendered, assistant_masks = self._render_chat(messages)
        if not tokenize:
            return rendered
        token_ids = self.encode(rendered, add_special_tokens=False)
        if return_dict is False:
            return token_ids
        encoding = {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }
        if return_assistant_tokens_mask:
            encoding["assistant_masks"] = assistant_masks
        return BatchEncoding(encoding)


class _NonPrefixStableTokenizer(_FakeTokenizer):
    def _render_chat(self, messages) -> tuple[str, list[int]]:
        rendered, assistant_masks = super()._render_chat(messages)
        prefix = f"<count={len(messages)}>"
        return f"{prefix}{rendered}", [0] * len(prefix) + assistant_masks


class _PartialAssistantMaskTokenizer(_FakeTokenizer):
    def _render_chat(self, messages) -> tuple[str, list[int]]:
        rendered, assistant_masks = super()._render_chat(messages)
        seen_assistant_span = False
        filtered_masks = []
        for mask_value in assistant_masks:
            if mask_value:
                filtered_masks.append(0 if seen_assistant_span else mask_value)
            else:
                if filtered_masks and filtered_masks[-1]:
                    seen_assistant_span = True
                filtered_masks.append(0)
        return rendered, filtered_masks


class _ShortAssistantMaskTokenizer(_FakeTokenizer):
    def _render_chat(self, messages) -> tuple[str, list[int]]:
        rendered, assistant_masks = super()._render_chat(messages)
        filtered_masks = []
        in_span = False
        emitted_in_span = False
        for mask_value in assistant_masks:
            if mask_value:
                if not in_span:
                    in_span = True
                    emitted_in_span = False
                filtered_masks.append(0 if emitted_in_span else 1)
                emitted_in_span = True
            else:
                in_span = False
                emitted_in_span = False
                filtered_masks.append(0)
        return rendered, filtered_masks


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
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "OK"
    assert batch.num_trainable_tokens == 2


def test_trajectory_default_loss_mask_is_roles_and_rejects_explicit_none() -> None:
    trajectory = Trajectory()

    assert trajectory.loss_mask == SFTRolesLossMask()
    with pytest.raises(pydantic.ValidationError, match="loss_mask"):
        Trajectory(loss_mask=None)  # ty:ignore[invalid-argument-type]


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
    batch = tokenize_sft_batch(
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
                loss_mask=SFTLastAssistantLossMask(),
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
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
    )

    assert batch.num_trainable_tokens == 0


def test_tokenize_sft_batch_default_loss_mask_equals_roles_assistant() -> None:
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

    default_batch = tokenize_sft_batch(
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
    )
    roles_batch = tokenize_sft_batch(
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
                loss_mask=SFTRolesLossMask(),
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
    )

    labels = roles_batch.trajectory_tensors[0]["labels"][0].tolist()
    assert labels == default_batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "firstsecond"


def test_tokenize_sft_batch_rejects_partial_assistant_generation_mask() -> None:
    tokenizer = _PartialAssistantMaskTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "B"},
            {"role": "assistant", "content": "second"},
        ],
    )

    with pytest.raises(ValueError, match="assistant token spans.*2"):
        tokenize_sft_batch(
            trajectory_batch=[Trajectory(messages_and_choices=messages)],
            learning_rate=1e-5,
            tokenizer=tokenizer,  # type: ignore[arg-type]
        )


def test_tokenize_sft_batch_rejects_short_assistant_generation_mask() -> None:
    tokenizer = _ShortAssistantMaskTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "B"},
            {"role": "assistant", "content": "second"},
        ],
    )

    with pytest.raises(ValueError, match="span shorter than the assistant payload"):
        tokenize_sft_batch(
            trajectory_batch=[Trajectory(messages_and_choices=messages)],
            learning_rate=1e-5,
            tokenizer=tokenizer,  # type: ignore[arg-type]
        )


def test_tokenize_sft_batch_rejects_short_tool_call_generation_mask() -> None:
    tokenizer = _ShortAssistantMaskTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Call the tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "book_table",
                            "arguments": '{"party_size": 4}',
                        },
                        "id": "call_123",
                        "type": "function",
                    }
                ],
            },
        ],
    )

    with pytest.raises(ValueError, match="span shorter than the assistant payload"):
        tokenize_sft_batch(
            trajectory_batch=[Trajectory(messages_and_choices=messages)],
            learning_rate=1e-5,
            tokenizer=tokenizer,  # type: ignore[arg-type]
        )


def test_tokenize_sft_batch_default_loss_mask_excludes_tool_result_messages() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "Use the tool"},
            {"role": "assistant", "content": "Calling"},
            {"role": "tool", "content": "secret tool result"},
            {"role": "assistant", "content": "Done"},
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_text = tokenizer.decode(
        [token_id for token_id in labels if token_id != -100]
    )
    assert trainable_text == "CallingDone"
    assert "secret tool result" not in trainable_text


def test_tokenize_sft_batch_loss_mask_all_labels_first_token() -> None:
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
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    assert labels[0] == tokenizer.encode("<")[0]
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "<system>S<user>Afirst"


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
        )


def test_tokenize_sft_batch_reuses_precomputed_prefix_boundaries() -> None:
    tokenizer = _FakeTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "B"},
            {"role": "assistant", "content": "second"},
        ],
    )

    tokenize_sft_batch(
        trajectory_batch=[
            Trajectory(
                messages_and_choices=messages,
                loss_mask=SFTAllLossMask(),
            )
        ],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
    )

    assert len(tokenizer.apply_chat_template_kwargs) == len(messages)


def test_tokenize_sft_batch_rejects_non_prefix_stable_template_for_spans() -> None:
    tokenizer = _NonPrefixStableTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
        ],
    )

    with pytest.raises(ValueError, match="semantic SFT loss masks.*prefix-stable"):
        tokenize_sft_batch(
            trajectory_batch=[
                Trajectory(
                    messages_and_choices=messages,
                    loss_mask=SFTAllLossMask(),
                )
            ],
            learning_rate=1e-5,
            tokenizer=tokenizer,  # type: ignore[arg-type]
        )


def test_tokenize_sft_batch_uses_generation_mask_for_non_prefix_stable_default() -> (
    None
):
    tokenizer = _NonPrefixStableTokenizer()
    messages = cast(
        MessagesAndChoices,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "first"},
        ],
    )

    batch = tokenize_sft_batch(
        trajectory_batch=[Trajectory(messages_and_choices=messages)],
        learning_rate=1e-5,
        tokenizer=tokenizer,  # type: ignore[arg-type]
    )

    labels = batch.trajectory_tensors[0]["labels"][0].tolist()
    trainable_token_ids = [token_id for token_id in labels if token_id != -100]
    assert tokenizer.decode(trainable_token_ids) == "first"
    assert len(tokenizer.apply_chat_template_kwargs) == 1
