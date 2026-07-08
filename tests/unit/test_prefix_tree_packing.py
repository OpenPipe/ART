from __future__ import annotations

import pytest
import torch

from art.megatron.prefix_tree_packing import (
    _local_position_pairs,
    estimate_prefix_tree_packed_tokens,
    prefix_tree_pack,
)


def test_prefix_tree_pack_support_depth_one() -> None:
    inputs = (
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([1, 2, 5]),
        torch.tensor([9]),
    )

    pack = prefix_tree_pack(inputs, max_depth=1)

    assert pack.tokens.tolist() == [[1, 2, 3, 4, 5, 9]]
    assert pack.group_ids.tolist() == [[1, 1, 2, 2, 3, 4]]
    assert pack.parent_ids.tolist() == [[1, 1, 1, 1, 1, 4]]
    assert pack.position_ids.tolist() == [[0, 1, 2, 3, 2, 0]]
    assert [positions.tolist() for positions in pack.positions_by_sequence] == [
        [0, 1, 2, 3],
        [0, 1, 4],
        [5],
    ]


def test_prefix_tree_pack_support_zero_depth_without_sharing() -> None:
    pack = prefix_tree_pack(
        (
            torch.tensor([1, 2]),
            torch.tensor([1, 3]),
            torch.tensor([4]),
        ),
        max_depth=0,
    )

    assert pack.tokens.tolist() == [[1, 2, 1, 3, 4]]
    assert pack.group_ids.tolist() == [[1, 1, 2, 2, 3]]
    assert pack.parent_ids.tolist() == [[1, 1, 2, 2, 3]]
    assert pack.position_ids.tolist() == [[0, 1, 0, 1, 0]]
    assert [positions.tolist() for positions in pack.positions_by_sequence] == [
        [0, 1],
        [2, 3],
        [4],
    ]


def test_prefix_tree_pack_support_deeper_trees() -> None:
    pack = prefix_tree_pack(
        (
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 6, 7]),
        ),
        max_depth=2,
    )

    assert pack.tokens.tolist() == [[1, 2, 3, 4, 5, 6, 7]]
    assert pack.group_ids.tolist() == [[1, 2, 2, 3, 4, 5, 5]]
    assert pack.parent_ids.tolist() == [[1, 1, 1, 2, 2, 1, 1]]
    assert pack.position_ids.tolist() == [[0, 1, 2, 3, 3, 1, 2]]
    assert [positions.tolist() for positions in pack.positions_by_sequence] == [
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 5, 6],
    ]


def test_packing_preserves_first_seen_branch_order() -> None:
    pack = prefix_tree_pack(
        (torch.tensor([9]), torch.tensor([1])),
        max_depth=1,
    )

    assert pack.tokens.tolist() == [[9, 1]]
    assert [positions.tolist() for positions in pack.positions_by_sequence] == [
        [0],
        [1],
    ]


def test_packing_handles_empty_sequences() -> None:
    pack = prefix_tree_pack(
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        max_depth=1,
    )

    assert pack.tokens.tolist() == [[]]
    assert pack.group_ids.tolist() == [[]]
    assert pack.parent_ids.tolist() == [[]]
    assert [positions.tolist() for positions in pack.positions_by_sequence] == [[], []]


def test_prefix_tree_pack_respects_shareable_lengths() -> None:
    inputs = (
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 2, 4]),
    )

    pack = prefix_tree_pack(inputs, max_depth=4, shareable_lengths=(1, 1))

    assert pack.tokens.tolist() == [[1, 2, 3, 2, 4]]
    assert [positions.tolist() for positions in pack.positions_by_sequence] == [
        [0, 1, 2],
        [0, 3, 4],
    ]
    assert estimate_prefix_tree_packed_tokens(
        inputs,
        max_depth=4,
        shareable_lengths=(1, 1),
    ) == int(pack.tokens.numel())


def test_packed_token_estimator_matches_real_packing() -> None:
    cases = [
        (torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4]), torch.tensor([5])),
        (
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 2, 6, 7]),
            torch.tensor([1, 8]),
        ),
        (
            torch.tensor([9, 1, 2]),
            torch.tensor([9, 1, 3]),
            torch.tensor([9, 4, 5]),
            torch.tensor([6, 7]),
            torch.tensor([], dtype=torch.long),
        ),
    ]

    for inputs in cases:
        for depth in range(5):
            pack = prefix_tree_pack(inputs, max_depth=depth)

            assert estimate_prefix_tree_packed_tokens(inputs, max_depth=depth) == int(
                pack.tokens.numel()
            )


def test_packed_token_estimator_matches_randomized_packing() -> None:
    generator = torch.Generator().manual_seed(123)
    inputs = []
    for family in range(5):
        prefix = torch.randint(1, 100, (4,), generator=generator)
        for branch in range(4):
            middle = torch.tensor([family, branch])
            suffix = torch.randint(1, 100, (3,), generator=generator)
            inputs.append(torch.cat((prefix, middle, suffix)))

    for depth in range(5):
        pack = prefix_tree_pack(inputs, max_depth=depth)

        assert estimate_prefix_tree_packed_tokens(inputs, max_depth=depth) == int(
            pack.tokens.numel()
        )


def test_packing_rejects_non_1d_sequences() -> None:
    with pytest.raises(ValueError, match="expects 1-D tensors"):
        prefix_tree_pack((torch.tensor([[1, 2], [3, 4]]),), max_depth=1)


def test_local_position_pairs_preserve_requested_order_without_dense_match() -> None:
    local_global_positions = torch.tensor([[2, -1, 0, 4, 1]])
    item_positions = torch.tensor([0, 1, 2, 3, 4])

    local_positions, source_positions = _local_position_pairs(
        local_global_positions,
        item_positions,
    )

    assert local_positions.tolist() == [2, 4, 0, 3]
    assert source_positions.tolist() == [0, 1, 2, 4]
