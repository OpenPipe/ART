from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch

from .cases import GdnPhase0Case
from .parser_import import GdnPackedExecutionSpec, parse_gdn_shared_prefix_segments


class GdnCaseSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    total_tokens: int
    family_count: int
    completion_count: int
    max_segment_length: int
    suffix_shorter_than_conv: bool
    suffix_equal_to_conv: bool
    suffix_longer_than_conv: bool
    cp_boundary_prefix: bool
    cp_boundary_suffix: bool
    family_boundary_at_partition: bool
    empty_trailing_rank: bool
    valid_lengths: tuple[int, ...]


def build_phase0_packed_tensors(case: GdnPhase0Case) -> dict[str, Any]:
    shape = (len(case.rows), case.sequence_length)
    generator = torch.Generator().manual_seed(case.seed)
    tokens = torch.zeros(shape, dtype=torch.long)
    group_ids = torch.full(shape, -1, dtype=torch.long)
    parent_ids = torch.full(shape, -1, dtype=torch.long)
    input_pos = torch.zeros(shape, dtype=torch.long)
    assistant_mask = torch.zeros(shape, dtype=torch.bool)
    logprobs = torch.full(shape, float("nan"), dtype=torch.float32)
    advantages = torch.zeros(shape, dtype=torch.float32)
    weights = torch.zeros(shape, dtype=torch.float32)

    for row_index, row in enumerate(case.rows):
        cursor = 0
        next_group_id = row_index * 100_000
        for family in row.families:
            required = family.prefix_length + sum(family.suffix_lengths)
            if cursor + required > case.sequence_length:
                raise ValueError(
                    f"case {case.name} row {row_index}: family requires {required} "
                    f"tokens with only {case.sequence_length - cursor} remaining"
                )
            prefix_group_id = next_group_id
            next_group_id += 1
            prefix_end = cursor + family.prefix_length
            _write_tokens(tokens, row_index, cursor, prefix_end, generator)
            group_ids[row_index, cursor:prefix_end] = prefix_group_id
            parent_ids[row_index, cursor:prefix_end] = prefix_group_id
            input_pos[row_index, cursor:prefix_end] = torch.arange(
                family.prefix_length, dtype=torch.long
            )
            cursor = prefix_end

            for suffix_length in family.suffix_lengths:
                completion_group_id = next_group_id
                next_group_id += 1
                suffix_end = cursor + suffix_length
                _write_tokens(tokens, row_index, cursor, suffix_end, generator)
                group_ids[row_index, cursor:suffix_end] = completion_group_id
                parent_ids[row_index, cursor:suffix_end] = prefix_group_id
                input_pos[row_index, cursor:suffix_end] = torch.arange(
                    family.prefix_length,
                    family.prefix_length + suffix_length,
                    dtype=torch.long,
                )
                if suffix_length > 1:
                    trainable_start = cursor + 1
                    assistant_mask[row_index, trainable_start:suffix_end] = True
                    logprobs[row_index, trainable_start:suffix_end] = _sample_logprobs(
                        suffix_length - 1, generator
                    )
                    advantages[row_index, trainable_start:suffix_end] = (
                        _sample_advantage(generator)
                    )
                    weights[row_index, trainable_start:suffix_end] = 1.0 / (
                        suffix_length - 1
                    )
                cursor = suffix_end

    return {
        "tokens": tokens,
        "group_ids": group_ids,
        "parent_ids": parent_ids,
        "input_pos": input_pos,
        "assistant_mask": assistant_mask,
        "logprobs": logprobs,
        "advantages": advantages,
        "weights": weights,
        "pixel_values": [None] * len(case.rows),
        "image_grid_thw": [None] * len(case.rows),
    }


def build_gdn_group_parent_tensors(case: GdnPhase0Case) -> dict[str, torch.Tensor]:
    shape = (len(case.rows), case.sequence_length)
    group_ids = torch.full(shape, -1, dtype=torch.long)
    parent_ids = torch.full(shape, -1, dtype=torch.long)
    for row_index, row in enumerate(case.rows):
        cursor = 0
        next_group_id = row_index * 100_000
        for family in row.families:
            required = family.prefix_length + sum(family.suffix_lengths)
            if cursor + required > case.sequence_length:
                raise ValueError(
                    f"case {case.name} row {row_index}: family requires {required} "
                    f"tokens with only {case.sequence_length - cursor} remaining"
                )
            prefix_group_id = next_group_id
            next_group_id += 1
            prefix_end = cursor + family.prefix_length
            group_ids[row_index, cursor:prefix_end] = prefix_group_id
            parent_ids[row_index, cursor:prefix_end] = prefix_group_id
            cursor = prefix_end
            for suffix_length in family.suffix_lengths:
                completion_group_id = next_group_id
                next_group_id += 1
                suffix_end = cursor + suffix_length
                group_ids[row_index, cursor:suffix_end] = completion_group_id
                parent_ids[row_index, cursor:suffix_end] = prefix_group_id
                cursor = suffix_end
    return {"group_ids": group_ids, "parent_ids": parent_ids}


def summarize_case(
    case: GdnPhase0Case,
    tensors: dict[str, Any],
    *,
    conv_width: int,
    cp_sizes: tuple[int, ...] = (2, 4, 8),
) -> GdnCaseSummary:
    spec = parse_gdn_shared_prefix_segments(tensors["group_ids"], tensors["parent_ids"])
    suffix_lengths = [
        segment.length
        for index, segment in enumerate(spec.tree_segments)
        if spec.tree_parent_indices[index] >= 0
    ]
    boundary = _boundary_flags(spec, cp_sizes)
    return GdnCaseSummary(
        name=case.name,
        total_tokens=spec.real_token_count,
        family_count=spec.family_count,
        completion_count=sum(
            1 for parent_index in spec.tree_parent_indices if parent_index >= 0
        ),
        max_segment_length=max(
            (segment.length for segment in spec.tree_segments), default=0
        ),
        suffix_shorter_than_conv=any(length < conv_width for length in suffix_lengths),
        suffix_equal_to_conv=any(length == conv_width for length in suffix_lengths),
        suffix_longer_than_conv=any(length > conv_width for length in suffix_lengths),
        cp_boundary_prefix=boundary["cp_boundary_prefix"],
        cp_boundary_suffix=boundary["cp_boundary_suffix"],
        family_boundary_at_partition=boundary["family_boundary_at_partition"],
        empty_trailing_rank=boundary["empty_trailing_rank"],
        valid_lengths=spec.valid_lengths,
    )


def format_case_summary(summary: GdnCaseSummary) -> str:
    flags = []
    for name in (
        "suffix_shorter_than_conv",
        "suffix_equal_to_conv",
        "suffix_longer_than_conv",
        "cp_boundary_prefix",
        "cp_boundary_suffix",
        "family_boundary_at_partition",
        "empty_trailing_rank",
    ):
        if getattr(summary, name):
            flags.append(name)
    return (
        f"{summary.name}: tokens={summary.total_tokens} "
        f"families={summary.family_count} completions={summary.completion_count} "
        f"max_segment={summary.max_segment_length} flags={','.join(flags) or 'none'}"
    )


def _write_tokens(
    tokens: torch.Tensor,
    row_index: int,
    start: int,
    end: int,
    generator: torch.Generator,
) -> None:
    tokens[row_index, start:end] = torch.randint(
        low=10, high=8192, size=(end - start,), dtype=torch.long, generator=generator
    )


def _sample_logprobs(length: int, generator: torch.Generator) -> torch.Tensor:
    return (
        torch.randn((length,), generator=generator, dtype=torch.float32) * 0.25 - 1.75
    )


def _sample_advantage(generator: torch.Generator) -> float:
    return float(
        (torch.randn((1,), generator=generator, dtype=torch.float32) * 0.5).item()
    )


def _boundary_flags(
    spec: GdnPackedExecutionSpec, cp_sizes: tuple[int, ...]
) -> dict[str, bool]:
    real_index: dict[int, int] = {}
    cursor = 0
    for row_index, valid_length in enumerate(spec.valid_lengths):
        for position in range(valid_length):
            real_index[row_index * spec.sequence_length + position] = cursor
            cursor += 1
    flags = {
        "cp_boundary_prefix": False,
        "cp_boundary_suffix": False,
        "family_boundary_at_partition": False,
        "empty_trailing_rank": False,
    }
    if spec.real_token_count == 0:
        return flags
    for cp_size in cp_sizes:
        shard = (spec.real_token_count + cp_size - 1) // cp_size
        boundaries = {shard * rank for rank in range(1, cp_size)}
        if shard * (cp_size - 1) >= spec.real_token_count:
            flags["empty_trailing_rank"] = True
        for root in _root_segments(spec):
            descendants = _descendant_segments(spec, root.family_index)
            family_segments = (root, *descendants)
            family_start = min(
                _segment_real_start(segment, spec, real_index)
                for segment in family_segments
            )
            family_end = max(
                _segment_real_end(segment, spec, real_index)
                for segment in family_segments
            )
            if family_start in boundaries or family_end in boundaries:
                flags["family_boundary_at_partition"] = True
            if _crosses_boundary(root, spec, real_index, boundaries):
                flags["cp_boundary_prefix"] = True
            for completion in descendants:
                if _crosses_boundary(completion, spec, real_index, boundaries):
                    flags["cp_boundary_suffix"] = True
    return flags


def _root_segments(spec: GdnPackedExecutionSpec) -> tuple[Any, ...]:
    return tuple(
        segment
        for index, segment in enumerate(spec.tree_segments)
        if spec.tree_parent_indices[index] < 0
    )


def _descendant_segments(
    spec: GdnPackedExecutionSpec, root_index: int
) -> tuple[Any, ...]:
    descendants = []
    for index, segment in enumerate(spec.tree_segments):
        parent = spec.tree_parent_indices[index]
        while parent >= 0:
            if parent == root_index:
                descendants.append(segment)
                break
            parent = spec.tree_parent_indices[parent]
    return tuple(descendants)


def _segment_real_start(
    segment: Any, spec: GdnPackedExecutionSpec, real_index: dict[int, int]
) -> int:
    return real_index[segment.row_index * spec.sequence_length + segment.start]


def _segment_real_end(
    segment: Any, spec: GdnPackedExecutionSpec, real_index: dict[int, int]
) -> int:
    return real_index[segment.row_index * spec.sequence_length + segment.end - 1] + 1


def _crosses_boundary(
    segment: Any,
    spec: GdnPackedExecutionSpec,
    real_index: dict[int, int],
    boundaries: set[int],
) -> bool:
    start = _segment_real_start(segment, spec, real_index)
    end = _segment_real_end(segment, spec, real_index)
    return any(start < boundary < end for boundary in boundaries)
