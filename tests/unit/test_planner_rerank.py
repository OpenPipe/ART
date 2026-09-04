"""Two-stage layout selection on re-ranked parallel shapes.

On a table's ``reranked_shapes`` the ten-term score only shortlists: the
cheapest ``shortlist_size`` layouts plus the incumbent anchor are priced by
the context-parallel plan each produces (remote wave count, largest per-rank
token load), and the lowest second-stage score is selected. These tests use
a synthetic re-ranker and a synthetic plan structure; the certified
coefficients are bound to their certificates elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from art.trainer_rank._planner_cost import (
    COEFFICIENT_VERSION,
    COEFFICIENTS_MILLI_US,
    DENSE_H2560_TABLE,
    H200_CLASS,
    QWEN3_4B_GEOMETRY,
    CalibratedTable,
    DeviceClass,
    ParallelShape,
    ReRanker,
    prefix_tree_layout_score,
)
from art.trainer_rank._prefix_tree_planner import (
    CanonicalPrefixTree,
    LayoutCandidate,
    build_canonical_prefix_tree,
    prefix_tree_layout_candidates,
    select_prefix_tree_layout,
)


def _tree() -> CanonicalPrefixTree:
    # A two-level prefix tree: a shared 2,000-token prompt, two 600-token
    # branches, two 200-token leaves each, so the mandatory family has
    # several distinct layouts (no sharing, depth one, depth two, spans).
    prompt = list(range(1, 2_001))
    branches = [list(range(3_000 + b * 1_000, 3_600 + b * 1_000)) for b in range(2)]
    rows = [
        torch.tensor(
            prompt
            + branches[b]
            + list(range(9_000 + (2 * b + leaf) * 300, 9_200 + (2 * b + leaf) * 300))
        )
        for b in range(2)
        for leaf in range(2)
    ]
    return build_canonical_prefix_tree(tuple(rows))


def _reranker(
    *,
    shortlist_size: int = 3,
    wave_per_layer_milli_us: int = 1_000_000,
    max_rank_token_per_layer_milli_us: int = 1_000,
) -> ReRanker:
    return ReRanker(
        shortlist_size=shortlist_size,
        incumbent="depth_one",
        shortlist_coefficients_milli_us=COEFFICIENTS_MILLI_US,
        wave_per_layer_milli_us=wave_per_layer_milli_us,
        max_rank_token_per_layer_milli_us=max_rank_token_per_layer_milli_us,
    )


def _cheap_order(
    tree: CanonicalPrefixTree, cp_size: int = 4, layers: int = 8
) -> list[LayoutCandidate]:
    scored = sorted(
        prefix_tree_layout_candidates(tree),
        key=lambda c: prefix_tree_layout_score(
            c.layout, cp_size=cp_size, layers=layers, uses_gdn=False
        ),
    )
    return scored


def test_reranker_scores_waves_and_load_per_layer() -> None:
    reranker = _reranker(wave_per_layer_milli_us=7, max_rank_token_per_layer_milli_us=2)
    assert reranker.score(layers=10, wave_count=3, max_rank_tokens=100) == 10 * (
        21 + 200
    )


def _table(
    *,
    reranked_shapes: tuple[ParallelShape, ...] = (),
    reranker: ReRanker | None = None,
) -> CalibratedTable:
    return CalibratedTable(
        table_id="t",
        coefficients_milli_us=COEFFICIENTS_MILLI_US,
        device_classes=(H200_CLASS,),
        param_dtypes=("torch.bfloat16",),
        geometries=(QWEN3_4B_GEOMETRY,),
        shapes=(ParallelShape(),),
        reranked_shapes=reranked_shapes,
        reranker=reranker,
    )


def test_table_validation_ties_reranked_shapes_to_a_reranker() -> None:
    cp4 = ParallelShape(tp=1, cp=4)
    with pytest.raises(ValueError):
        _table(reranked_shapes=(cp4,))
    with pytest.raises(ValueError):
        _table(reranker=_reranker())
    with pytest.raises(ValueError):
        _table(reranked_shapes=(ParallelShape(),), reranker=_reranker())
    table = _table(reranked_shapes=(cp4,), reranker=_reranker())
    device = DeviceClass((9, 0), "hbm-141g")
    dtype = "torch.bfloat16"
    assert table.admits(
        device=device,
        param_dtype=dtype,
        geometry=QWEN3_4B_GEOMETRY,
        shape=ParallelShape(),
    )
    assert not table.admits(
        device=device, param_dtype=dtype, geometry=QWEN3_4B_GEOMETRY, shape=cp4
    )
    assert table.reranks(
        device=device, param_dtype=dtype, geometry=QWEN3_4B_GEOMETRY, shape=cp4
    )
    assert not table.reranks(
        device=device,
        param_dtype=dtype,
        geometry=QWEN3_4B_GEOMETRY,
        shape=ParallelShape(tp=1, cp=2),
    )
    assert not table.reranks(
        device=device,
        param_dtype="torch.float16",
        geometry=QWEN3_4B_GEOMETRY,
        shape=cp4,
    )
    assert DENSE_H2560_TABLE.reranker is None and not DENSE_H2560_TABLE.reranked_shapes


def test_without_a_reranker_selection_is_the_cheap_incumbent() -> None:
    tree = _tree()
    plain = select_prefix_tree_layout(
        tree, cp_size=4, layers=8, uses_gdn=False, refinement_work_budget=0
    )
    assert plain.layout == _cheap_order(tree)[0].layout


def test_reranking_picks_the_lowest_second_stage_score_in_the_shortlist() -> None:
    tree = _tree()
    order = _cheap_order(tree)
    assert len(order) >= 4
    # The plan structure makes the cheap order's third layout the clear winner
    # (one wave, smallest load) and the cheap winner the worst.
    structure = {c.layout: (2, 5_000) for c in order}
    structure[order[2].layout] = (1, 1_000)
    structure[order[0].layout] = (4, 9_000)
    selected = select_prefix_tree_layout(
        tree,
        cp_size=4,
        layers=8,
        uses_gdn=False,
        refinement_work_budget=0,
        reranker=_reranker(shortlist_size=3),
        plan_structure=lambda layout: structure[layout],
    )
    assert selected.layout == order[2].layout
    # Outside the shortlist a layout is never selected however good its plan.
    structure[order[-1].layout] = (0, 1)
    if order[-1] not in order[:3] and "depth_one" not in order[-1].labels:
        selected = select_prefix_tree_layout(
            tree,
            cp_size=4,
            layers=8,
            uses_gdn=False,
            refinement_work_budget=0,
            reranker=_reranker(shortlist_size=3),
            plan_structure=lambda layout: structure[layout],
        )
        assert selected.layout == order[2].layout


def test_ties_keep_the_cheaper_layout_and_the_incumbent_is_always_considered() -> None:
    tree = _tree()
    order = _cheap_order(tree)
    # Identical plan structure everywhere: the re-ranker must not disturb the
    # cheap order.
    flat = lambda layout: (1, 4_000)  # noqa: E731
    selected = select_prefix_tree_layout(
        tree,
        cp_size=4,
        layers=8,
        uses_gdn=False,
        refinement_work_budget=0,
        reranker=_reranker(shortlist_size=2),
        plan_structure=flat,
    )
    assert selected.layout == order[0].layout
    # A shortlist of one still considers the depth-one anchor.
    depth_one = next(c for c in order if "depth_one" in c.labels)
    if depth_one is not order[0]:
        structure = {c.layout: (3, 9_000) for c in order}
        structure[depth_one.layout] = (1, 1_000)
        selected = select_prefix_tree_layout(
            tree,
            cp_size=4,
            layers=8,
            uses_gdn=False,
            refinement_work_budget=0,
            reranker=_reranker(shortlist_size=1),
            plan_structure=lambda layout: structure[layout],
        )
        assert selected.layout == depth_one.layout


def test_reranked_selection_requires_a_plan_structure() -> None:
    with pytest.raises(ValueError):
        select_prefix_tree_layout(
            _tree(),
            cp_size=4,
            layers=8,
            uses_gdn=False,
            refinement_work_budget=0,
            reranker=_reranker(),
        )


def test_selection_version_is_unchanged_by_the_reranker_plumbing() -> None:
    assert COEFFICIENT_VERSION == 2
