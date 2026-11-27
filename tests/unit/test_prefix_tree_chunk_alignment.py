"""Unit tests for 64-token FLA chunk-aligned CP > 1 canonical prefix trees."""

from __future__ import annotations

import random

import pytest
import torch

from art.trainer_rank import _impl
from art.trainer_rank._prefix_tree_materializer import materialize_prefix_tree_layout
from art.trainer_rank._prefix_tree_planner import (
    GDN_CP_PREFIX_TREE_CHUNK_SIZE,
    CanonicalPrefixTree,
    build_canonical_prefix_tree,
    canonical_prefix_tree_chunk_size,
    plan_prefix_tree_layout,
    prefix_tree_layout_candidates,
)


def _assert_chunk_aligned_tree_valid(
    sequences: list[list[int]],
    *,
    chunk_size: int = GDN_CP_PREFIX_TREE_CHUNK_SIZE,
) -> CanonicalPrefixTree:
    tree = build_canonical_prefix_tree(sequences, chunk_size=chunk_size)
    parent_indices = {
        segment.parent_index
        for segment in tree.segments
        if segment.parent_index is not None
    }

    for segment in tree.segments:
        if len(segment.sequence_indices) > 1:
            assert segment.start % chunk_size == 0
            assert segment.end > segment.start
            if segment.index in parent_indices:
                assert segment.end % chunk_size == 0
                assert (segment.end - segment.start) % chunk_size == 0
            else:
                assert all(
                    len(sequences[seq_idx]) == segment.end
                    for seq_idx in segment.sequence_indices
                )
        if segment.parent_index is not None:
            assert segment.parent_index < segment.index
            assert tree.segments[segment.parent_index].end == segment.start

    for seq_idx, expected in enumerate(sequences):
        covering = [
            segment for segment in tree.segments if seq_idx in segment.sequence_indices
        ]
        covering.sort(key=lambda seg: seg.start)
        assert covering[0].start == 0
        assert covering[-1].end == len(expected)
        for prev_seg, next_seg in zip(covering, covering[1:], strict=False):
            assert prev_seg.end == next_seg.start
            assert next_seg.parent_index == prev_seg.index

    tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    for candidate in prefix_tree_layout_candidates(tree):
        packed = materialize_prefix_tree_layout(
            tensors, tree, candidate.layout, verify_shared_tokens=True
        )
        for seq_idx, expected_tensor in enumerate(tensors):
            reconstructed = packed.tokens[0].index_select(
                0, packed.positions_by_sequence[seq_idx]
            )
            assert torch.equal(reconstructed, expected_tensor)
        packed_parent_ids = {
            seg.parent_id for seg in packed.segments if seg.parent_id != seg.group_id
        }
        for seg in packed.segments:
            if len(seg.sequence_indices) > 1:
                assert seg.start % chunk_size == 0
                if seg.group_id in packed_parent_ids:
                    assert seg.end % chunk_size == 0
                    assert (seg.end - seg.start) % chunk_size == 0
                else:
                    assert all(
                        len(sequences[seq_idx]) == seg.end
                        for seq_idx in seg.sequence_indices
                    )
    return tree


def test_canonical_prefix_tree_chunk_size_gates_on_gdn_and_cp() -> None:
    assert canonical_prefix_tree_chunk_size(uses_gdn=True, cp_size=4) == 64
    assert canonical_prefix_tree_chunk_size(uses_gdn=True, cp_size=2) == 64
    assert canonical_prefix_tree_chunk_size(uses_gdn=True, cp_size=1) == 1
    assert canonical_prefix_tree_chunk_size(uses_gdn=False, cp_size=4) == 1
    assert canonical_prefix_tree_chunk_size(uses_gdn=False, cp_size=1) == 1


def test_chunk_aligned_prefix_tree_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        build_canonical_prefix_tree([[1, 2, 3]], chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        build_canonical_prefix_tree([[1, 2, 3]], chunk_size=-64)


def test_chunk_aligned_prefix_tree_s05_nested_duplicates_and_prefix_of_prefix() -> None:
    # Exact S0-5 shape: 4096-token aligned prefix + 7 shared tokens + 56-token tails.
    p4096 = list(range(100, 100 + 4096))
    s7 = [9001, 9002, 9003, 9004, 9005, 9006, 9007]
    s05_4 = [p4096 + s7 + [10000 + i * 100 + t for t in range(56)] for i in range(4)]
    tree_s05 = _assert_chunk_aligned_tree_valid(s05_4)
    assert [(s.start, s.end, s.sequence_indices) for s in tree_s05.segments] == [
        (0, 4096, (0, 1, 2, 3)),
        (4096, 4159, (0,)),
        (4096, 4159, (1,)),
        (4096, 4159, (2,)),
        (4096, 4159, (3,)),
    ]

    # Nested depth-3 tree: 4096 shared by all 4; +80 shared by (0, 1) -> rounds to +64;
    # +10 shared by (2, 3) -> rounds to 0 and splits directly into leaves.
    nested = [
        p4096 + list(range(5000, 5080)) + [7000 + t for t in range(64)],
        p4096 + list(range(5000, 5080)) + [8000 + t for t in range(64)],
        p4096 + list(range(6000, 6010)) + [8500 + t for t in range(64)],
        p4096 + list(range(6000, 6010)) + [8900 + t for t in range(64)],
    ]
    tree_nested = _assert_chunk_aligned_tree_valid(nested)
    assert [
        (s.start, s.end, s.sequence_indices)
        for s in tree_nested.segments
        if len(s.sequence_indices) > 1
    ] == [
        (0, 4096, (0, 1, 2, 3)),
        (4096, 4160, (0, 1)),
    ]

    # Identical sequences: unaligned length 4103, exact aligned 4096, and short 3/30
    # deduplicate cleanly into a single terminal shared leaf with zero chunk splits.
    tree_unaligned_dup = _assert_chunk_aligned_tree_valid([p4096 + s7, p4096 + s7])
    assert [
        (s.start, s.end, s.sequence_indices) for s in tree_unaligned_dup.segments
    ] == [(0, 4103, (0, 1))]
    tree_exact = _assert_chunk_aligned_tree_valid([p4096, p4096])
    assert [(s.start, s.end, s.sequence_indices) for s in tree_exact.segments] == [
        (0, 4096, (0, 1))
    ]
    tree_short_3 = _assert_chunk_aligned_tree_valid(
        [[3838, 198, 3838], [3838, 198, 3838]]
    )
    assert [(s.start, s.end, s.sequence_indices) for s in tree_short_3.segments] == [
        (0, 3, (0, 1))
    ]
    tree_short_30 = _assert_chunk_aligned_tree_valid([list(range(30)), list(range(30))])
    assert [(s.start, s.end, s.sequence_indices) for s in tree_short_30.segments] == [
        (0, 30, (0, 1))
    ]

    # Duplicate leaves alongside a continuing sequence: shared root stays aligned
    # at 4096, duplicate 4103-token leaves deduplicate cleanly without splitting a chunk.
    tree_dup_plus_cont = _assert_chunk_aligned_tree_valid(
        [p4096 + s7, p4096 + s7, p4096 + s7 + [999]]
    )
    assert [
        (s.start, s.end, s.sequence_indices) for s in tree_dup_plus_cont.segments
    ] == [
        (0, 4096, (0, 1, 2)),
        (4096, 4103, (0, 1)),
        (4096, 4104, (2,)),
    ]

    # Interleaved short duplicate leaves + continuing sequence preserve first-seen order.
    tree_interleaved = _assert_chunk_aligned_tree_valid(
        [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3]]
    )
    assert [
        (s.start, s.end, s.sequence_indices) for s in tree_interleaved.segments
    ] == [
        (0, 3, (0, 2)),
        (0, 4, (1,)),
    ]

    # Distinct short sequences sharing a sub-chunk prefix never split a chunk.
    tree_distinct_short = _assert_chunk_aligned_tree_valid([[1, 2, 3], [1, 2, 4]])
    assert [
        (s.start, s.end, s.sequence_indices) for s in tree_distinct_short.segments
    ] == [
        (0, 3, (0,)),
        (0, 3, (1,)),
    ]

    # Prefix-of-prefix: seq 0 ends at 4103, seqs 1 & 2 share 4200 tokens.
    tree_pop = _assert_chunk_aligned_tree_valid(
        [
            p4096 + s7,
            p4096 + s7 + list(range(2000, 2097)) + [1] * 50,
            p4096 + s7 + list(range(2000, 2097)) + [2] * 50,
        ]
    )
    assert [
        (s.start, s.end, s.sequence_indices)
        for s in tree_pop.segments
        if len(s.sequence_indices) > 1
    ] == [
        (0, 4096, (0, 1, 2)),
        (4096, 4160, (1, 2)),
    ]


@pytest.mark.parametrize("chunk_size", [4, 16, 64])
@pytest.mark.parametrize("seed", range(25))
def test_chunk_aligned_prefix_tree_random_invariants_and_materialization(
    chunk_size: int, seed: int
) -> None:
    rng = random.Random(20261126 + chunk_size * 1000 + seed)
    alphabet = rng.choice((2, 4, 32))
    row_count = rng.randint(2, 8)
    root_len = rng.choice(
        (0, 3, chunk_size - 1, chunk_size, chunk_size + 5, 2 * chunk_size + 7)
    )
    root = [rng.randrange(alphabet) for _ in range(root_len)]
    rows: list[list[int]] = []
    for _ in range(row_count):
        if rows and rng.random() < 0.2:
            rows.append(list(rows[rng.randrange(len(rows))]))
            continue
        if rows and rng.random() < 0.2:
            src = rows[rng.randrange(len(rows))]
            cut = rng.randint(1, len(src))
            rows.append(list(src[:cut]))
            continue
        tail_len = rng.randint(1, chunk_size + 12)
        rows.append(root + [rng.randrange(alphabet) for _ in range(tail_len)])
    _assert_chunk_aligned_tree_valid(rows, chunk_size=chunk_size)


def test_trainer_rank_compute_group_layout_aligns_gdn_cp_gt_1_and_isolates_tree_cache() -> (
    None
):
    from collections import OrderedDict
    import threading

    from art.trainer_rank._micro_batch_planner import _compute_group_layout

    rank = _impl.TrainerRank.__new__(_impl.TrainerRank)
    rank._layout_cache_lock = threading.Lock()
    rank._tree_cache = OrderedDict()
    rank._layout_selection_cache = OrderedDict()
    rank._planner_coefficients = None
    rank._planner_reranker = None

    shared = list(range(10, 10 + 71))  # 64 + 7 unaligned prefix
    rows = (
        torch.tensor(shared + [1001, 1002], dtype=torch.long),
        torch.tensor(shared + [2001, 2002], dtype=torch.long),
    )

    facts_cp4_gdn = _impl._PlannerFacts(
        cp_size=4,
        tp_size=1,
        layers=8,
        gdn_layers=6,
        uses_gdn=True,
        ep_size=1,
        etp_size=1,
        coefficient_version=1,
        coefficient_table=None,
        grad_enabled=True,
    )
    facts_cp1_gdn = _impl._PlannerFacts(
        cp_size=1,
        tp_size=1,
        layers=8,
        gdn_layers=6,
        uses_gdn=True,
        ep_size=1,
        etp_size=1,
        coefficient_version=1,
        coefficient_table=None,
        grad_enabled=True,
    )
    facts_cp4_dense = _impl._PlannerFacts(
        cp_size=4,
        tp_size=1,
        layers=8,
        gdn_layers=0,
        uses_gdn=False,
        ep_size=1,
        etp_size=1,
        coefficient_version=1,
        coefficient_table=None,
        grad_enabled=True,
    )

    tree_cp4_gdn, layout_cp4_gdn = _compute_group_layout(
        rank, rows, ("same-content", facts_cp4_gdn, "full_sharing")
    )
    tree_cp1_gdn, layout_cp1_gdn = _compute_group_layout(
        rank, rows, ("same-content", facts_cp1_gdn, "full_sharing")
    )
    tree_cp4_dense, layout_cp4_dense = _compute_group_layout(
        rank, rows, ("same-content", facts_cp4_dense, "full_sharing")
    )

    # GDN + CP=4 floors the 71-token shared prefix to 64 tokens.
    assert [(s.start, s.end, s.sequence_indices) for s in tree_cp4_gdn.segments] == [
        (0, 64, (0, 1)),
        (64, 73, (0,)),
        (64, 73, (1,)),
    ]
    assert layout_cp4_gdn.tree_fingerprint == tree_cp4_gdn.fingerprint

    # CP=1 and non-GDN CP=4 retain exact unaligned 71-token canonical prefix sharing.
    assert [(s.start, s.end, s.sequence_indices) for s in tree_cp1_gdn.segments] == [
        (0, 71, (0, 1)),
        (71, 73, (0,)),
        (71, 73, (1,)),
    ]
    assert tree_cp4_dense.fingerprint == tree_cp1_gdn.fingerprint
    assert layout_cp1_gdn.tree_fingerprint == tree_cp1_gdn.fingerprint
    assert layout_cp4_dense.tree_fingerprint == tree_cp4_dense.fingerprint
    assert (
        plan_prefix_tree_layout(
            tree_cp4_gdn, layout_cp4_gdn.selected_decisions
        ).fingerprint
        == layout_cp4_gdn.fingerprint
    )
    assert len(rank._tree_cache) == 2
