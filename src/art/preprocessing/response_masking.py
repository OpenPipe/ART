from collections.abc import Callable, Sequence
from typing import Any


def _loss_mask_dict(loss_mask: Any) -> dict[str, Any]:
    if hasattr(loss_mask, "model_dump"):
        return loss_mask.model_dump(exclude_none=True)
    if isinstance(loss_mask, dict):
        return loss_mask
    raise TypeError(f"loss_mask must be a dict-like object, got {type(loss_mask)!r}")


def _is_assistant_roles_only_loss_mask(spec: dict[str, Any]) -> bool:
    return (
        spec.get("type") == "roles"
        and bool(spec.get("assistant", True))
        and not any(bool(spec.get(role, False)) for role in ("system", "user", "tool"))
    )


def _assistant_messages_with_trainable_payload(
    messages: list[dict[str, Any]],
) -> int:
    return sum(
        1
        for message in messages
        if message.get("role") == "assistant"
        and (message.get("content") or message.get("tool_calls"))
    )


def _mask_span_lengths(mask: Sequence[int]) -> list[int]:
    lengths: list[int] = []
    current_length = 0
    in_span = False
    for mask_value in mask:
        if bool(mask_value):
            if not in_span:
                in_span = True
                current_length = 0
            current_length += 1
        else:
            if in_span:
                lengths.append(current_length)
            in_span = False
    if in_span:
        lengths.append(current_length)
    return lengths


def _assistant_token_mask_is_trusted(
    *,
    input_ids: list[int],
    messages: list[dict[str, Any]],
    assistant_token_mask: Sequence[int] | None,
    assistant_payload_token_counts: Sequence[int] | None,
) -> bool:
    if assistant_token_mask is None:
        return False
    if len(assistant_token_mask) != len(input_ids):
        raise ValueError(
            "Cannot compile semantic SFT loss masks because the tokenizer returned "
            "an assistant token mask with a different length than input_ids: "
            f"{len(assistant_token_mask)} != {len(input_ids)}."
        )
    expected_assistant_spans = _assistant_messages_with_trainable_payload(messages)
    assistant_mask_span_lengths = _mask_span_lengths(assistant_token_mask)
    assistant_mask_spans = len(assistant_mask_span_lengths)
    if expected_assistant_spans == 0:
        if assistant_mask_spans:
            raise ValueError(
                "Cannot compile semantic SFT loss masks because the tokenizer "
                "returned assistant token spans but the messages contain no "
                "assistant payloads."
            )
        return True
    if assistant_mask_spans == 0:
        return False
    if assistant_mask_spans != expected_assistant_spans:
        raise ValueError(
            "Cannot compile semantic SFT loss masks because the tokenizer returned "
            "assistant token spans that do not match the assistant messages with "
            "trainable payloads: "
            f"{assistant_mask_spans} != {expected_assistant_spans}."
        )
    if assistant_payload_token_counts is not None:
        if len(assistant_payload_token_counts) != expected_assistant_spans:
            raise ValueError(
                "Cannot compile semantic SFT loss masks because the number of "
                "assistant payload token counts does not match the assistant "
                "messages with trainable payloads: "
                f"{len(assistant_payload_token_counts)} != {expected_assistant_spans}."
            )
        for index, (span_length, payload_token_count) in enumerate(
            zip(assistant_mask_span_lengths, assistant_payload_token_counts)
        ):
            if span_length < payload_token_count:
                raise ValueError(
                    "Cannot compile semantic SFT loss masks because the tokenizer "
                    "returned an assistant token span shorter than the assistant "
                    f"payload tokenization at assistant payload {index}: "
                    f"{span_length} < {payload_token_count}."
                )
    return True


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
            "Cannot compile semantic SFT loss masks because the chat template is "
            f"not prefix-stable at {name}. Use tokenizer generation markers or a "
            "prefix-stable chat template."
        )


def _prefix_boundaries(
    message_prefix_token_ids: Sequence[Sequence[int]]
    | Callable[[], Sequence[Sequence[int]]],
) -> Sequence[Sequence[int]]:
    if callable(message_prefix_token_ids):
        return message_prefix_token_ids()
    return message_prefix_token_ids


def semantic_loss_mask_labels(
    input_ids: list[int],
    *,
    messages: list[dict[str, Any]],
    loss_mask: Any,
    message_prefix_token_ids: Sequence[Sequence[int]]
    | Callable[[], Sequence[Sequence[int]]],
    assistant_token_mask: Sequence[int] | None = None,
    assistant_payload_token_counts: Sequence[int] | None = None,
) -> list[int]:
    """Compile a semantic chat loss_mask into token labels.

    The tokenizer-side training code still consumes labels with -100 for ignored
    positions. This helper only decides which rendered message spans contribute
    loss; the trainer backends do not need to know about the semantic spec.
    """

    labels = [-100] * len(input_ids)
    spec = _loss_mask_dict(loss_mask)
    selected_flags = _selected_message_flags(messages, spec)
    assistant_mask_is_trusted = _assistant_token_mask_is_trusted(
        input_ids=input_ids,
        messages=messages,
        assistant_token_mask=assistant_token_mask,
        assistant_payload_token_counts=assistant_payload_token_counts,
    )

    if (
        assistant_mask_is_trusted
        and _is_assistant_roles_only_loss_mask(spec)
        and selected_flags
        == [message.get("role") == "assistant" for message in messages]
    ):
        assert assistant_token_mask is not None
        return [
            token_id if bool(mask_value) else -100
            for token_id, mask_value in zip(input_ids, assistant_token_mask)
        ]

    if not any(selected_flags):
        return labels

    boundaries = _prefix_boundaries(message_prefix_token_ids)
    if len(boundaries) != len(messages) + 1:
        raise ValueError(
            "Cannot compile semantic SFT loss masks because the number of rendered "
            "message prefixes does not match the message count: "
            f"{len(boundaries)} != {len(messages) + 1}."
        )

    previous_boundary_len = 0
    for boundary_index, boundary_ids in enumerate(boundaries):
        boundary = list(boundary_ids)
        _validate_prefix(
            name=f"message prefix {boundary_index}",
            prefix_ids=boundary,
            full_ids=input_ids,
        )
        if len(boundary) < previous_boundary_len:
            raise ValueError(
                "Cannot compile semantic SFT loss masks because a rendered message "
                f"prefix shrank at boundary {boundary_index}."
            )
        previous_boundary_len = len(boundary)

    for index, selected in enumerate(selected_flags):
        if not selected:
            continue

        label_start = len(boundaries[index])
        label_end = len(boundaries[index + 1])
        if messages[index].get("role") == "assistant" and assistant_mask_is_trusted:
            assert assistant_token_mask is not None
            for token_index in range(label_start, label_end):
                if bool(assistant_token_mask[token_index]):
                    labels[token_index] = input_ids[token_index]
            continue

        labels[label_start:label_end] = input_ids[label_start:label_end]

    return labels
