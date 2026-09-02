"""Layout features used by cost-model calibration and scoring.

Values are checked on the sealed GRPO ``primary_long_g8`` shape (2 groups x 8
completions, system 2048, prompt 8192, completion 512), whose canonical tree
has exactly three decisions and a four-layout mandatory family.
"""

from __future__ import annotations

import torch

from art.trainer_rank._planner_cost import LayoutFeatures, layout_features
from art.trainer_rank._prefix_tree_planner import (
    build_canonical_prefix_tree,
    prefix_tree_layout_candidates,
)


def _grpo_rows() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(6_001)

    def tokens(count: int) -> torch.Tensor:
        return torch.randint(10, 64_000, (count,), generator=generator)

    system = tokens(2048)
    rows = []
    for _ in range(2):
        prompt = torch.cat((system, tokens(8192)))
        for _ in range(8):
            rows.append(torch.cat((prompt, tokens(512))))
    return tuple(rows)


def test_layout_features_on_the_sealed_grpo_shape() -> None:
    tree = build_canonical_prefix_tree(_grpo_rows())
    by_label = {
        label: layout_features(candidate.layout)
        for candidate in prefix_tree_layout_candidates(tree)
        for label in candidate.labels
    }

    no_sharing = by_label["no_sharing"]
    assert no_sharing == LayoutFeatures(
        packed_tokens=172_032,
        segment_count=16,
        shared_segments=0,
        max_depth=1,
        shared_tokens=0,
        fanout_sum=0,
        small_segments=0,
        tiny_segments=0,
        attention_area=16 * (10_752 * 10_752 // 2),
    )
    depth_one = by_label["depth_one"]
    assert (depth_one.packed_tokens, depth_one.segment_count) == (141_312, 17)
    assert (depth_one.shared_segments, depth_one.max_depth) == (1, 2)
    assert (depth_one.shared_tokens, depth_one.fanout_sum) == (2_048, 16)
    full = by_label["full_sharing"]
    assert (full.packed_tokens, full.segment_count, full.shared_segments) == (
        26_624,
        19,
        3,
    )
    assert (full.max_depth, full.shared_tokens, full.fanout_sum) == (3, 18_432, 32)
    partial = by_label["minimum_effective_span_2049"]
    assert (partial.packed_tokens, partial.shared_segments, partial.max_depth) == (
        28_672,
        2,
        2,
    )
    assert partial.shared_tokens == 20_480 and partial.fanout_sum == 16
    # Sharing shrinks causal attention work: the shared layouts attend to a
    # fifth of the pairs the unshared layout does.
    assert full.attention_area < no_sharing.attention_area // 4
    assert full.as_dict()["attention_area"] == full.attention_area
