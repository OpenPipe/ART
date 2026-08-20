import torch

from art.distributed.data_plane import (
    _layout,
    packed_text_storage_byte_count,
)
from art.megatron.runtime.data_plane import SFTBatchData


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
