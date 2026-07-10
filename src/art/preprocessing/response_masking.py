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
