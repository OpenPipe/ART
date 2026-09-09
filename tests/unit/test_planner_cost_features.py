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

    assert by_label["no_sharing"] == LayoutFeatures(
        packed_tokens=172_032,
        segment_count=16,
        max_depth=1,
        segments_below=(0, 0, 0, 0, 0, 0, 0, 0),
    )
    depth_one = by_label["depth_one"]
    assert (depth_one.packed_tokens, depth_one.segment_count, depth_one.max_depth) == (
        141_312,
        17,
        2,
    )
    full = by_label["full_sharing"]
    assert (full.packed_tokens, full.segment_count, full.max_depth) == (26_624, 19, 3)
    # One 2,048-token system, two 8,192-token prompts, sixteen 512-token
    # completions: cumulative counts strictly below 64..8192 (the system counts
    # only below 4,096).
    assert full.segments_below == (0, 0, 0, 0, 16, 16, 17, 17)
    assert full.below(512) == 0 and full.below(1024) == 16
    # CP4 reads the bucket for 128 * 4 = 512 tokens per rank.
    assert full.below(128 * 4) == 0 and full.below(128 * 8) == 16
    partial = by_label["minimum_effective_span_2049"]
    assert (partial.packed_tokens, partial.segment_count, partial.max_depth) == (
        28_672,
        18,
        2,
    )
    assert full.as_dict()["segments_below"] == full.segments_below
