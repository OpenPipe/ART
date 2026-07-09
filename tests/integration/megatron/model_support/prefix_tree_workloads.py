from __future__ import annotations

from typing import Any

import torch


def build_complex_prefix_tree_packed_tensors(
    config: Any,
    seed: int,
) -> dict[str, Any]:
    """Build a deterministic nested prefix-tree packed workload.

    Each packed row repeats this tree while it fits:

    root
      -> mid_a -> leaf_a_short, leaf_a_long
      -> mid_b -> leaf_b
      -> direct_leaf

    Internal nodes are non-trainable. Each leaf starts with one non-trainable
    context token followed by trainable completion tokens, matching production
    prefix-tree packing's trainable-token boundary.
    """

    num_sequences = int(config.num_sequences)
    sequence_length = int(config.sequence_length)
    if num_sequences <= 1:
        raise ValueError("num_sequences must be greater than 1")
    if sequence_length < 16:
        raise ValueError("sequence_length must leave room for a nested prefix tree")

    generator = torch.Generator().manual_seed(seed)
    shape = (num_sequences, sequence_length)
    tokens = torch.zeros(shape, dtype=torch.long)
    group_ids = torch.full(shape, -1, dtype=torch.long)
    parent_ids = torch.full(shape, -1, dtype=torch.long)
    input_pos = torch.zeros(shape, dtype=torch.long)
    assistant_mask = torch.zeros(shape, dtype=torch.bool)
    logprobs = torch.full(shape, float("nan"), dtype=torch.float32)
    advantages = torch.zeros(shape, dtype=torch.float32)
    weights = torch.zeros(shape, dtype=torch.float32)

    token_low = 10
    vocab_high = max(token_low + 1, int(config.vocab_high))
    token_span = vocab_high - token_low
    prefill_tokens = max(4, min(sequence_length - 8, int(config.prefill_tokens)))
    root_len = max(2, prefill_tokens // 2)
    mid_budget = max(2, prefill_tokens - root_len)
    mid_a_len = max(1, mid_budget // 2)
    mid_b_len = max(1, mid_budget - mid_a_len)
    max_completion_tokens = max(1, sequence_length - root_len - mid_a_len - 2)
    base_completion_tokens = max(
        1,
        min(int(config.decode_tokens), max_completion_tokens),
    )
    jitter_width = min(int(config.decode_tokens_jitter), max_completion_tokens - 1)

    def sample_completion_length() -> int:
        jitter = (
            int(
                torch.randint(
                    low=-jitter_width,
                    high=jitter_width + 1,
                    size=(1,),
                    generator=generator,
                    dtype=torch.long,
                ).item()
            )
            if jitter_width > 0
            else 0
        )
        return max(1, min(max_completion_tokens, base_completion_tokens + jitter))

    def sample_tokens(length: int) -> torch.Tensor:
        return torch.randint(
            low=token_low,
            high=vocab_high,
            size=(length,),
            dtype=torch.long,
            generator=generator,
        )

    def sample_logprobs(length: int) -> torch.Tensor:
        return (
            torch.randn((length,), generator=generator, dtype=torch.float32) * 0.25
            - 1.75
        )

    def sample_advantage() -> float:
        return float(
            (torch.randn((1,), generator=generator, dtype=torch.float32) * 0.5).item()
        )

    def write_segment(
        *,
        sequence_index: int,
        cursor: int,
        group_id: int,
        parent_id: int,
        length: int,
        input_start: int,
        trainable_offset: int | None = None,
    ) -> int:
        end = min(sequence_length, cursor + length)
        take = end - cursor
        if take <= 0:
            return cursor
        tokens[sequence_index, cursor:end] = sample_tokens(take)
        group_ids[sequence_index, cursor:end] = group_id
        parent_ids[sequence_index, cursor:end] = parent_id
        input_pos[sequence_index, cursor:end] = torch.arange(
            input_start,
            input_start + take,
            dtype=torch.long,
        )
        if trainable_offset is not None and take > trainable_offset:
            train_start = cursor + trainable_offset
            train_len = end - train_start
            assistant_mask[sequence_index, train_start:end] = True
            logprobs[sequence_index, train_start:end] = sample_logprobs(train_len)
            advantages[sequence_index, train_start:end] = sample_advantage()
            weights[sequence_index, train_start:end] = 1.0
        return end

    def tree_token_budget(leaf_lengths: tuple[int, int, int, int]) -> int:
        return (
            root_len
            + mid_a_len
            + mid_b_len
            + sum(1 + length for length in leaf_lengths)
        )

    def fit_leaf_lengths(
        leaf_lengths: tuple[int, int, int, int],
        *,
        remaining: int,
    ) -> tuple[int, int, int, int] | None:
        completion_budget = (
            remaining - root_len - mid_a_len - mid_b_len - len(leaf_lengths)
        )
        if completion_budget < len(leaf_lengths):
            return None
        if sum(leaf_lengths) <= completion_budget:
            return leaf_lengths
        fitted = list(leaf_lengths)
        overflow = sum(fitted) - completion_budget
        while overflow > 0:
            index = max(range(len(fitted)), key=lambda candidate: fitted[candidate])
            reducible = fitted[index] - 1
            if reducible <= 0:
                return None
            reduction = min(reducible, overflow)
            fitted[index] -= reduction
            overflow -= reduction
        return (fitted[0], fitted[1], fitted[2], fitted[3])

    for sequence_index in range(num_sequences):
        cursor = 0
        next_group_id = 0
        while cursor < sequence_length:
            leaf_lengths = (
                sample_completion_length(),
                sample_completion_length(),
                sample_completion_length(),
                sample_completion_length(),
            )
            remaining = sequence_length - cursor
            if tree_token_budget(leaf_lengths) > remaining:
                fitted_leaf_lengths = fit_leaf_lengths(
                    leaf_lengths,
                    remaining=remaining,
                )
                if fitted_leaf_lengths is None:
                    break
                leaf_lengths = fitted_leaf_lengths
            root_group = next_group_id
            next_group_id += 1
            cursor = write_segment(
                sequence_index=sequence_index,
                cursor=cursor,
                group_id=root_group,
                parent_id=root_group,
                length=root_len,
                input_start=0,
            )
            mid_a_group = next_group_id
            next_group_id += 1
            cursor = write_segment(
                sequence_index=sequence_index,
                cursor=cursor,
                group_id=mid_a_group,
                parent_id=root_group,
                length=mid_a_len,
                input_start=root_len,
            )
            for leaf_length in leaf_lengths[:2]:
                leaf_group = next_group_id
                next_group_id += 1
                cursor = write_segment(
                    sequence_index=sequence_index,
                    cursor=cursor,
                    group_id=leaf_group,
                    parent_id=mid_a_group,
                    length=1 + leaf_length,
                    input_start=root_len + mid_a_len,
                    trainable_offset=1,
                )
            mid_b_group = next_group_id
            next_group_id += 1
            cursor = write_segment(
                sequence_index=sequence_index,
                cursor=cursor,
                group_id=mid_b_group,
                parent_id=root_group,
                length=mid_b_len,
                input_start=root_len,
            )
            leaf_b_group = next_group_id
            next_group_id += 1
            cursor = write_segment(
                sequence_index=sequence_index,
                cursor=cursor,
                group_id=leaf_b_group,
                parent_id=mid_b_group,
                length=1 + leaf_lengths[2],
                input_start=root_len + mid_b_len,
                trainable_offset=1,
            )
            direct_group = next_group_id
            next_group_id += 1
            cursor = write_segment(
                sequence_index=sequence_index,
                cursor=cursor,
                group_id=direct_group,
                parent_id=root_group,
                length=1 + leaf_lengths[3],
                input_start=root_len,
                trainable_offset=1,
            )

    half = num_sequences // 2
    if half > 0 and num_sequences % 2 == 0:
        valid_lengths = (group_ids != -1).sum(dim=1)
        for pair_index in range(half):
            left_index = pair_index
            right_index = pair_index + half
            left_valid = int(valid_lengths[left_index].item())
            right_valid = int(valid_lengths[right_index].item())
            if left_valid != right_valid or left_valid == 0:
                continue
            if torch.equal(
                tokens[left_index, :left_valid],
                tokens[right_index, :right_valid],
            ):
                tokens[right_index, 0] = (
                    (tokens[right_index, 0] - token_low + 1) % token_span
                ) + token_low

    weights = torch.where(assistant_mask, weights, torch.zeros_like(weights))
    if bool(assistant_mask.any().item()):
        weights[assistant_mask] /= weights[assistant_mask].mean()
        advantages = torch.where(
            assistant_mask,
            advantages,
            torch.zeros_like(advantages),
        )
        advantage_scale = (
            advantages[assistant_mask].abs() * weights[assistant_mask]
        ).mean()
        if float(advantage_scale.item()) > 0.0:
            advantages[assistant_mask] /= advantage_scale

    return {
        "tokens": tokens,
        "group_ids": group_ids,
        "parent_ids": parent_ids,
        "input_pos": input_pos,
        "assistant_mask": assistant_mask,
        "logprobs": logprobs,
        "advantages": advantages,
        "weights": weights,
        "pixel_values": [None] * num_sequences,
        "image_grid_thw": [None] * num_sequences,
    }
