"""Layout features used by cost-model calibration and scoring.

Values are checked on the sealed GRPO ``primary_long_g8`` shape (2 groups x 8
completions, system 2048, prompt 8192, completion 512), whose canonical tree
has exactly three decisions and a four-layout mandatory family.
"""

from __future__ import annotations

import pytest
import torch

from art.trainer_rank import _planner_cost
from art.trainer_rank._planner_cost import (
    SEGMENT_LENGTH_THRESHOLDS,
    LayoutFeatures,
    layout_features,
    prefix_tree_layout_score,
)
from art.trainer_rank._prefix_tree_performance_search import (
    search_nonuniform_prefix_tree_layouts,
)
from art.trainer_rank._prefix_tree_planner import (
    PlannedSegment,
    PrefixTreeLayout,
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


@pytest.mark.parametrize(
    "lengths",
    [(), (0, 0, 1, 8192, 8193, 1 << 40)]
    + [(t + offset,) for t in SEGMENT_LENGTH_THRESHOLDS for offset in (-1, 0, 1)]
    + [
        tuple(t + offset for t in SEGMENT_LENGTH_THRESHOLDS for offset in (-1, 0, 1))
        * 2
    ],
)
def test_layout_features_strict_thresholds(lengths: tuple[int, ...]) -> None:
    layout = PrefixTreeLayout(
        tree_fingerprint="test",
        selected_decisions=frozenset(),
        segments=tuple(
            PlannedSegment((i,), 7, 7 + length, None, None)
            for i, length in enumerate(lengths)
        ),
        packed_tokens=sum(lengths),
        maximum_depth=int(bool(lengths)),
        fingerprint="test",
    )
    assert layout_features(layout) == LayoutFeatures(
        packed_tokens=sum(lengths),
        segment_count=len(lengths),
        max_depth=int(bool(lengths)),
        segments_below=tuple(
            sum(length < threshold for length in lengths)
            for threshold in SEGMENT_LENGTH_THRESHOLDS
        ),
    )


@pytest.mark.parametrize("cp_size", [1, 4])
def test_histogram_preserves_full_search(
    monkeypatch: pytest.MonkeyPatch, cp_size: int
) -> None:
    tree = build_canonical_prefix_tree(_grpo_rows())

    def search():
        return search_nonuniform_prefix_tree_layouts(
            tree,
            lambda layout: prefix_tree_layout_score(
                layout, cp_size=cp_size, layers=40, uses_gdn=True, gdn_layers=30
            ),
            mandatory_candidates=prefix_tree_layout_candidates(tree),
            refinement_work_budget=2000,
        )

    actual = search()

    def strict_counts(layout: PrefixTreeLayout) -> LayoutFeatures:
        lengths = [s.end - s.start for s in layout.segments]
        return LayoutFeatures(
            packed_tokens=layout.packed_tokens,
            segment_count=len(lengths),
            max_depth=layout.maximum_depth,
            segments_below=tuple(
                sum(length < threshold for length in lengths)
                for threshold in SEGMENT_LENGTH_THRESHOLDS
            ),
        )

    monkeypatch.setattr(_planner_cost, "layout_features", strict_counts)
    expected = search()
    assert actual.evaluated_refinements > 0
    assert actual == expected
    assert actual.fingerprint == expected.fingerprint
