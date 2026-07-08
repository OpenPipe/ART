from collections.abc import Callable
from typing import Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def token_ids_for_template_part(
    tokenizer: PreTrainedTokenizerBase,
    template_part: str,
) -> list[int]:
    return list(tokenizer(template_part, add_special_tokens=False).input_ids)


def _find_subsequence(
    values: list[int],
    pattern: list[int],
    *,
    start: int = 0,
) -> int | None:
    if not pattern:
        return None
    last_start = len(values) - len(pattern)
    for index in range(start, last_start + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return None


def response_only_labels(
    input_ids: list[int],
    *,
    instruction_ids: list[int],
    response_ids: list[int],
) -> list[int]:
    labels = [-100] * len(input_ids)
    index = 0
    while index < len(input_ids):
        response_start = _find_subsequence(input_ids, response_ids, start=index)
        if response_start is None:
            break

        trainable_start = response_start + len(response_ids)
        next_instruction_start = _find_subsequence(
            input_ids,
            instruction_ids,
            start=trainable_start,
        )
        trainable_end = (
            len(input_ids) if next_instruction_start is None else next_instruction_start
        )
        labels[trainable_start:trainable_end] = input_ids[trainable_start:trainable_end]
        index = trainable_end
    return labels


def _loss_mask_dict(loss_mask: Any) -> dict[str, Any]:
    if hasattr(loss_mask, "model_dump"):
        return loss_mask.model_dump(exclude_none=True)
    if isinstance(loss_mask, dict):
        return loss_mask
    raise TypeError(f"loss_mask must be a dict-like object, got {type(loss_mask)!r}")


def _selected_message_flags(
    messages: list[dict[str, Any]],
    loss_mask: Any,
) -> list[bool]:
    spec = _loss_mask_dict(loss_mask)
    mask_type = spec.get("type")

    if mask_type == "all":
        return [True] * len(messages)
    if mask_type == "none":
        return [False] * len(messages)
    if mask_type == "last_assistant":
        flags = [False] * len(messages)
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "assistant":
                flags[index] = True
                return flags
        raise ValueError("loss_mask type 'last_assistant' found no assistant message")
    if mask_type == "message_mask":
        mask = spec.get("mask")
        if not isinstance(mask, list):
            raise ValueError("loss_mask type 'message_mask' requires a list 'mask'")
        if len(mask) != len(messages):
            raise ValueError(
                "loss_mask message_mask length must match messages length: "
                f"{len(mask)} != {len(messages)}"
            )
        return [bool(value) for value in mask]
    if mask_type == "roles":
        selected_roles = {
            role
            for role in ("system", "user", "assistant", "tool")
            if bool(spec.get(role, role == "assistant"))
        }
        return [message.get("role") in selected_roles for message in messages]

    raise ValueError(f"Unknown SFT loss_mask type: {mask_type!r}")


def _validate_prefix(
    *,
    name: str,
    prefix_ids: list[int],
    full_ids: list[int],
) -> None:
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "Cannot compile SFT loss_mask because the chat template is not "
            f"prefix-stable at {name}."
        )


def semantic_loss_mask_labels(
    input_ids: list[int],
    *,
    messages: list[dict[str, Any]],
    loss_mask: Any,
    response_ids: list[int],
    render_message_prefix: Callable[[int], list[int]],
) -> list[int]:
    """Compile a semantic chat loss_mask into token labels.

    The tokenizer-side training code still consumes labels with -100 for ignored
    positions. This helper only decides which rendered message spans contribute
    loss; the trainer backends do not need to know about the semantic spec.
    """

    labels = [-100] * len(input_ids)
    selected_flags = _selected_message_flags(messages, loss_mask)

    for index, selected in enumerate(selected_flags):
        if not selected:
            continue

        before_ids = [] if index == 0 else render_message_prefix(index)
        after_ids = (
            input_ids
            if index + 1 == len(messages)
            else render_message_prefix(index + 1)
        )
        _validate_prefix(
            name=f"message {index} start",
            prefix_ids=before_ids,
            full_ids=input_ids,
        )
        _validate_prefix(
            name=f"message {index} end",
            prefix_ids=after_ids,
            full_ids=input_ids,
        )
        if len(after_ids) < len(before_ids):
            raise ValueError(
                "Cannot compile SFT loss_mask because a rendered message prefix "
                f"shrunk at message {index}."
            )

        label_start = len(before_ids)
        label_end = len(after_ids)
        if messages[index].get("role") == "assistant" and response_ids:
            response_start = _find_subsequence(
                input_ids,
                response_ids,
                start=label_start,
            )
            if response_start is None or response_start >= label_end:
                raise ValueError(
                    "Cannot compile SFT loss_mask because response_part was not "
                    f"found inside selected assistant message {index}."
                )
            if response_start + len(response_ids) > label_end:
                raise ValueError(
                    "Cannot compile SFT loss_mask because response_part extends "
                    f"past selected assistant message {index}."
                )
            label_start = response_start + len(response_ids)

        labels[label_start:label_end] = input_ids[label_start:label_end]

    if labels:
        labels[0] = -100
    return labels
