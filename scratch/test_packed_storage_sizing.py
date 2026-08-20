import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import torch

from art.distributed.data_plane import (
    _flatten_packed_tensors,
    _layout,
    packed_plan_storage_byte_count,
    packed_text_storage_byte_count,
)
from art.megatron.runtime.data_plane import SFTBatchData
import art.preprocessing.pack as pack_module
from art.preprocessing.pack import (
    materialize_packed_tensors,
    prepare_packed_tensors_from_tokenized_results,
)
from art.preprocessing.tokenize import TokenizedResult
from art.trajectories import Trajectory


def test_text_storage_size_matches_physical_layout() -> None:
    rows, sequence_length, candidates = 3, 5, 2
    shape = (rows, sequence_length)
    flat = [
        ("tokens", torch.zeros(shape, dtype=torch.int64)),
        ("group_ids", torch.zeros(shape, dtype=torch.int64)),
        ("parent_ids", torch.zeros(shape, dtype=torch.int64)),
        ("input_pos", torch.zeros(shape, dtype=torch.int64)),
        ("assistant_mask", torch.zeros(shape, dtype=torch.bool)),
        ("logprobs", torch.zeros(shape, dtype=torch.float32)),
        ("advantages", torch.zeros(shape, dtype=torch.float32)),
        ("weights", torch.zeros(shape, dtype=torch.float32)),
        (
            "target_tokens",
            torch.zeros((*shape, candidates), dtype=torch.int64),
        ),
        ("loss_weights", torch.zeros((*shape, candidates), dtype=torch.float32)),
        (
            "behavior_logprobs",
            torch.zeros((*shape, candidates), dtype=torch.float32),
        ),
        (
            "token_advantages",
            torch.zeros((*shape, candidates), dtype=torch.float32),
        ),
        (
            "moe_routing_replay/expert_indices",
            torch.zeros((2, *shape, 3), dtype=torch.uint16),
        ),
    ]

    _, physical_bytes = _layout(flat)

    assert physical_bytes == packed_text_storage_byte_count(
        num_sequences=rows,
        sequence_length=sequence_length,
        candidate_capacity=candidates,
        moe_num_layers=2,
        moe_topk=3,
        moe_num_experts=257,
    )


def test_sft_storage_bound_contains_exact_canonical_tensors() -> None:
    trajectories = tuple(
        {
            "input_ids": torch.arange(length).reshape(1, -1),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
            "labels": torch.tensor([[-100, *range(1, length)]], dtype=torch.long),
        }
        for length in (3, 5)
    )
    batch = SFTBatchData(
        trajectory_tensors=trajectories,
        num_trajectories=2,
        num_tokens=8,
        num_trainable_tokens=6,
    )

    assert batch.storage_byte_count == 8 * 3 * 8
    assert batch.storage_byte_count <= SFTBatchData.storage_upper_bound(
        num_trajectories=2,
        max_sequence_length=5,
    )


def test_tool_schema_prefix_tree_shape_admits_exact_physical_storage() -> None:
    tool_schema = {
        "role": "system",
        "content": "Use the provided tools and return the final answer.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            argument: {"type": "string"},
                            "units": {"enum": ["metric", "imperial"]},
                        },
                        "required": [argument],
                    },
                },
            }
            for name, description, argument in (
                ("weather", "Get current weather", "location"),
                ("forecast", "Get a seven day forecast", "location"),
                ("geocode", "Resolve a place name", "query"),
            )
        ],
    }
    prefix = list(json.dumps(tool_schema, separators=(",", ":")).encode())
    completion_length = 12
    results = [
        TokenizedResult(
            advantage=1.0,
            chat="",
            token_ids=[
                *prefix,
                *(10_000 + choice * completion_length + offset for offset in range(completion_length)),
            ],
            input_pos=list(range(len(prefix) + completion_length)),
            assistant_mask=[0] * len(prefix) + [1] * completion_length,
            logprobs=[math.nan] * len(prefix) + [-0.1] * completion_length,
            pixel_values=None,
            image_grid_thw=None,
            trajectory=Trajectory(),
            choice_offsets=[len(prefix)],
            extra_logprobs={},
            _tokenizer=SimpleNamespace(decode=lambda token_id: str(token_id)),
            weight=1.0,
            prompt_id=17,
            prompt_length=len(prefix),
        )
        for choice in range(16)
    ]
    sequence_length = 1024

    planning_calls = 0
    canonical_planner = pack_module._prefix_tree_pack_rows

    def counted_planner(*args, **kwargs):
        nonlocal planning_calls
        planning_calls += 1
        return canonical_planner(*args, **kwargs)

    with patch.object(pack_module, "_prefix_tree_pack_rows", counted_planner):
        plan = prepare_packed_tensors_from_tokenized_results(
            results,
            seq_len=sequence_length,
            truncate_long_results=False,
            pack_results=True,
        )
        reservation = packed_plan_storage_byte_count(plan)
        packed = materialize_packed_tensors(plan)
    flat, _ = _flatten_packed_tensors(packed)
    _, exact_bytes = _layout(flat)

    assert planning_calls == 1
    assert plan.num_sequences == packed["tokens"].shape[0] == 1
    assert exact_bytes == reservation
    assert reservation < packed_text_storage_byte_count(
        num_sequences=len(results),
        sequence_length=sequence_length,
    )
