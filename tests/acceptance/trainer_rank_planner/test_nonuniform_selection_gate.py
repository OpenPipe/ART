"""Landing acceptance: layout selection and bounded-search capability.

CORRECTED 2026-09-01 (pre-implementation, empirically verified against the
research tree): the original version of this gate required *production*
selection to pick nonuniform depth>1 layouts on the tiny sealed synthetic
families. Running the sealed research planner showed that is unsatisfiable by
a faithful physical cost model — on token sequences a few dozen tokens long,
sharing never pays under GDN/CP costs, and the research production score
correctly selects no-sharing there (its "nonuniform selection" results came
from the search-quality harness under an injected adversarial scorer). The
corrected gate matches what the research actually sealed:

1. PRODUCTION SELECTION, physical scale: on the sealed GPU win-cell shape
   (GRPO primary_long_g8: 2 groups x 8 completions, system 2048, prompt 8192,
   completion 512) the planner must select depth > 1 and pack dramatically
   fewer physical than logical tokens. Verified against the research planner:
   depth 3, 26,624 physical for 172,032 logical — identical to the sealed
   cold witness of the GPU campaign.
2. PRODUCTION SELECTION, declines-when-unprofitable: on the tiny sealed
   corpus families and on a heterogeneous control, the planner must select
   depth <= 1 / no sharing. (Verified: the research planner does exactly
   this.)
3. SEARCH CAPABILITY: with an injected scorer whose cost surface demands
   nonuniform decisions (the sealed adversarial branch-interaction scorer),
   the bounded search must recover the exhaustive-oracle optimum on the
   sealed corpus trees (all <= 14 decisions, so the oracle is enumerable).
4. Determinism at both scales.

ADAPTATION POINT: the ``_planner`` import shim may be adjusted once at landing
if module or function names differ. The required surface:
``build_canonical_prefix_tree(sequences)``,
``prefix_tree_layout_candidates(tree)`` (candidates carry ``.layout`` and
``.labels``), ``select_prefix_tree_layout(tree, *, cp_size, layers, uses_gdn,
refinement_work_budget)``, ``iter_all_prefix_tree_layouts(tree)``, and
``search_prefix_tree_layout(tree, scorer, *, refinement_work_budget)`` for
scorer injection. Assertion bodies are the acceptance criteria and must not
be weakened.

These tests are EXPECTED TO FAIL until the holistic planner lands.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import importlib
import random
from typing import Any

import pytest

BASE_SEED = 20260821
WORKLOAD_FAMILIES = (
    "balanced",
    "skewed",
    "deep_comb",
    "short_segment",
    "no_sharing",
    "grpo_like",
    "mixed_branch",
)
GATE_TOPOLOGY = {"cp_size": 4, "layers": 12, "uses_gdn": True}
WIN_CELL_TOPOLOGY = {"cp_size": 4, "layers": 2, "uses_gdn": True}
GENEROUS_BUDGET = 2_000

REQUIRED_SURFACE = (
    "build_canonical_prefix_tree",
    "prefix_tree_layout_candidates",
    "select_prefix_tree_layout",
    "iter_all_prefix_tree_layouts",
    "search_prefix_tree_layout",
)


def _planner() -> Any:
    """Import the landed planner surface, failing with a clear message."""

    try:
        module = importlib.import_module("art.trainer_rank._prefix_tree_planner")
    except ImportError as error:
        pytest.fail(f"the holistic planner surface has not landed yet: {error}")
    for required in REQUIRED_SURFACE:
        if not hasattr(module, required):
            pytest.fail(f"planner surface is missing {required}")
    return module


# --- Sealed corpus generators (verbatim port; do not modify) ---------------


def _edge_tokens(prefix_code: int, length: int) -> tuple[int, ...]:
    return tuple(prefix_code * 10_000 + offset for offset in range(length))


def _balanced_sequences(
    rng: random.Random,
    *,
    short_segments: bool,
) -> tuple[tuple[int, ...], ...]:
    depth = rng.choice((2, 3))
    root_length = 1 if short_segments else rng.randint(2, 8)
    root = _edge_tokens(1, root_length)
    rows: list[tuple[int, ...]] = []
    for leaf in range(1 << depth):
        row = list(root)
        prefix = 1
        for level in range(depth):
            bit = (leaf >> (depth - level - 1)) & 1
            prefix = prefix * 2 + bit
            length = 1 if short_segments else rng.randint(1, 7)
            row.extend(_edge_tokens(100 + prefix, length))
        row.extend(_edge_tokens(2_000 + leaf, rng.randint(1, 5)))
        rows.append(tuple(row))
    return tuple(rows)


def _comb_sequences(
    rng: random.Random,
    *,
    long_edges: bool,
) -> tuple[tuple[int, ...], ...]:
    leaf_count = rng.randint(7, 13)
    root = _edge_tokens(3, rng.randint(1, 6))
    continuation: list[int] = []
    rows: list[tuple[int, ...]] = []
    for leaf in range(leaf_count - 1):
        edge_length = rng.randint(2, 8) if long_edges else 1
        continuation.extend(_edge_tokens(300 + leaf, edge_length))
        rows.append(
            (*root, *continuation[:-edge_length], 900_000 + leaf, 910_000 + leaf)
        )
    continuation.extend(
        _edge_tokens(
            300 + leaf_count - 1,
            rng.randint(2, 8) if long_edges else 1,
        )
    )
    rows.append((*root, *continuation, 999_999))
    return tuple(rows)


def _no_sharing_sequences(rng: random.Random) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (700_000 + index, *_edge_tokens(700 + index, rng.randint(1, 12)))
        for index in range(rng.randint(2, 12))
    )


def _grpo_sequences(rng: random.Random) -> tuple[tuple[int, ...], ...]:
    group_count = rng.randint(2, 4)
    group_size = rng.choice((2, 4))
    rows: list[tuple[int, ...]] = []
    for group in range(group_count):
        prompt = _edge_tokens(800 + group, rng.randint(4, 18))
        for completion in range(group_size):
            rows.append(
                (
                    800_000 + group,
                    *prompt,
                    810_000 + group * 100 + completion,
                    *_edge_tokens(
                        900 + group * 10 + completion,
                        rng.randint(1, 8),
                    ),
                )
            )
    return tuple(rows)


def _mixed_sequences(rng: random.Random) -> tuple[tuple[int, ...], ...]:
    root = _edge_tokens(5, rng.randint(2, 6))
    balanced = _balanced_sequences(rng, short_segments=False)
    comb = _comb_sequences(rng, long_edges=True)
    balanced = balanced[:4]
    comb = comb[:7]
    return tuple((*root, 51_000, *row) for row in balanced) + tuple(
        (*root, 52_000, *row) for row in comb
    )


@dataclass(frozen=True)
class GateCase:
    family: str
    seed: int
    sequences: tuple[tuple[int, ...], ...]


def _gate_case(family: str) -> GateCase:
    family_index = WORKLOAD_FAMILIES.index(family)
    seed = BASE_SEED + family_index * 100_003
    rng = random.Random(seed)
    if family == "balanced":
        sequences = _balanced_sequences(rng, short_segments=False)
    elif family == "skewed":
        sequences = _comb_sequences(rng, long_edges=False)
    elif family == "deep_comb":
        sequences = _comb_sequences(rng, long_edges=True)
    elif family == "short_segment":
        sequences = _balanced_sequences(rng, short_segments=True)
    elif family == "no_sharing":
        sequences = _no_sharing_sequences(rng)
    elif family == "grpo_like":
        sequences = _grpo_sequences(rng)
    elif family == "mixed_branch":
        sequences = _mixed_sequences(rng)
    else:  # pragma: no cover - protected by the frozen family tuple
        raise AssertionError(f"unknown family {family}")
    return GateCase(family=family, seed=seed, sequences=sequences)


def _win_cell_sequences() -> tuple[tuple[int, ...], ...]:
    """GRPO primary_long_g8 shape from the sealed GPU win cell."""

    rows: list[tuple[int, ...]] = []
    system = tuple(range(1_000_000, 1_000_000 + 2_048))
    for group in range(2):
        prompt = tuple(
            range(2_000_000 + group * 100_000, 2_000_000 + group * 100_000 + 8_192)
        )
        for completion in range(8):
            suffix = tuple(
                range(
                    3_000_000 + (group * 100 + completion) * 10_000,
                    3_000_000 + (group * 100 + completion) * 10_000 + 512,
                )
            )
            rows.append(system + prompt + suffix)
    return tuple(rows)


def _heterogeneous_sequences() -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(row * 1_000_000, row * 1_000_000 + 4_000)) for row in range(16)
    )


# --- Sealed adversarial scorer (verbatim port; search-capability only) ------


def _decision_parents(tree: Any) -> dict[int, int | None]:
    decisions = frozenset(tree.decision_indices)
    nearest: list[int | None] = [None] * len(tree.segments)
    result: dict[int, int | None] = {}
    for segment in tree.segments:
        parent = None if segment.parent_index is None else nearest[segment.parent_index]
        if segment.index in decisions:
            result[segment.index] = parent
            nearest[segment.index] = segment.index
        else:
            nearest[segment.index] = parent
    return result


def _branch_interaction_score(tree: Any, layout: Any, *, seed: int) -> tuple[int, ...]:
    selected = layout.selected_decisions
    parents = _decision_parents(tree)
    transformer_work = layout.packed_tokens * 512
    attention_work = (
        sum(segment.length * segment.length for segment in layout.segments) * 16
    )
    exchange_work = 0
    barrier_work = 0
    saved_work_by_depth: dict[int, list[int]] = defaultdict(list)
    for ordinal, decision in enumerate(tree.decision_indices):
        segment = tree.segments[decision]
        local_saved = segment.length * (len(segment.sequence_indices) - 1)
        signature = seed ^ (decision * 1_315_423_911) ^ (ordinal * 2_654_435_761)
        expensive_lane = signature % 5 in (0, 1)
        if decision in selected:
            lane_factor = 160 if expensive_lane else 6
            exchange_work += (
                segment.end + segment.length * len(segment.sequence_indices)
            ) * lane_factor
            parent = parents[decision]
            if parent is not None and parent in selected:
                barrier_work += (segment.depth + 1) * (192 + 16 * lane_factor)
            saved_work_by_depth[segment.depth].append(local_saved)
        elif not expensive_lane:
            transformer_work += local_saved * 1_024
    bucket_padding = 0
    for saved_work in saved_work_by_depth.values():
        bucket_padding += max(saved_work) * len(saved_work) - sum(saved_work)
    compile_signatures = {
        (
            segment.length.bit_length(),
            segment.parent_segment_index is not None,
            segment.length % 8,
        )
        for segment in layout.segments
    }
    total = (
        transformer_work
        + attention_work
        + exchange_work * 64
        + barrier_work * 32
        + bucket_padding * 256
        + len(compile_signatures) * 2_048
    )
    return (total, layout.packed_tokens, len(layout.segments), layout.maximum_depth)


# --- Acceptance assertions --------------------------------------------------


def test_win_cell_shape_selects_deep_sharing() -> None:
    """Sealed cold witness: depth 3, 26,624 physical for 172,032 logical."""

    planner = _planner()
    tree = planner.build_canonical_prefix_tree(_win_cell_sequences())
    selected = planner.select_prefix_tree_layout(
        tree, refinement_work_budget=GENEROUS_BUDGET, **WIN_CELL_TOPOLOGY
    )
    assert selected.layout.maximum_depth > 1, (
        "the sealed win-cell shape must select deep sharing (sealed: depth 3);"
        f" got depth {selected.layout.maximum_depth}"
    )
    logical = sum(len(row) for row in _win_cell_sequences())
    assert selected.layout.packed_tokens < logical // 4, (
        "deep sharing must pack far fewer physical than logical tokens"
        f" (sealed: 26,624 of {logical}); got {selected.layout.packed_tokens}"
    )


def test_heterogeneous_control_declines_sharing() -> None:
    planner = _planner()
    tree = planner.build_canonical_prefix_tree(_heterogeneous_sequences())
    selected = planner.select_prefix_tree_layout(
        tree, refinement_work_budget=GENEROUS_BUDGET, cp_size=1, layers=2, uses_gdn=True
    )
    assert selected.layout.maximum_depth <= 1
    assert not selected.layout.selected_decisions


@pytest.mark.parametrize("family", ("grpo_like", "deep_comb", "mixed_branch"))
def test_tiny_corpus_families_decline_unprofitable_sharing(family: str) -> None:
    """Physical cost model must not share when token savings cannot pay.

    Empirically verified against the sealed research planner: on these tiny
    sealed-corpus trees the production score selects no sharing.
    """

    planner = _planner()
    case = _gate_case(family)
    tree = planner.build_canonical_prefix_tree(case.sequences)
    assert tree.decision_indices, f"{family}: corpus should offer decisions"
    selected = planner.select_prefix_tree_layout(
        tree, refinement_work_budget=GENEROUS_BUDGET, **GATE_TOPOLOGY
    )
    assert selected.layout.maximum_depth <= 1, (
        f"{family}: sharing tiny segments under GDN CP4 costs must not be"
        f" selected; got depth {selected.layout.maximum_depth}"
    )


def test_no_sharing_family_tree_has_no_decisions() -> None:
    planner = _planner()
    tree = planner.build_canonical_prefix_tree(_gate_case("no_sharing").sequences)
    assert not tree.decision_indices


@pytest.mark.parametrize("family", ("grpo_like", "deep_comb", "mixed_branch"))
def test_candidates_retain_mandatory_anchors(family: str) -> None:
    planner = _planner()
    tree = planner.build_canonical_prefix_tree(_gate_case(family).sequences)
    candidates = planner.prefix_tree_layout_candidates(tree)
    assert candidates, f"{family}: no layout candidates generated"
    layouts = [candidate.layout for candidate in candidates]
    decision_counts = {len(layout.selected_decisions) for layout in layouts}
    assert 0 in decision_counts, f"{family}: no-sharing anchor missing"
    assert len(tree.decision_indices) in decision_counts, (
        f"{family}: full-sharing anchor missing"
    )
    assert any(layout.maximum_depth == 1 for layout in layouts), (
        f"{family}: depth-one-equivalent anchor missing"
    )


@pytest.mark.parametrize(
    "family", ("grpo_like", "deep_comb", "mixed_branch", "balanced", "skewed")
)
def test_bounded_search_recovers_adversarial_oracle(family: str) -> None:
    """The sealed search-quality gate: injected-scorer optimum is found.

    The sealed corpus trees have <= 14 decisions, so the exhaustive oracle is
    enumerable. The adversarial scorer's cost surface demands nonuniform
    decisions on these families (sealed nonuniform selection rates: grpo_like
    1.0, deep_comb 0.875, mixed_branch 1.0), so passing proves the bounded
    search can represent and recover nonuniform optima.
    """

    planner = _planner()
    case = _gate_case(family)
    tree = planner.build_canonical_prefix_tree(case.sequences)
    assert len(tree.decision_indices) <= 14

    def scorer(layout: Any) -> tuple[int, ...]:
        return _branch_interaction_score(tree, layout, seed=case.seed)

    oracle = min(
        (scorer(layout) for layout in planner.iter_all_prefix_tree_layouts(tree)),
    )
    found = planner.search_prefix_tree_layout(
        tree, scorer, refinement_work_budget=GENEROUS_BUDGET
    )
    assert scorer(found.layout) == oracle, (
        f"{family}: bounded search missed the exhaustive oracle optimum"
    )


@pytest.mark.parametrize("scale", ("win_cell", "tiny"))
def test_selection_is_deterministic(scale: str) -> None:
    planner = _planner()
    if scale == "win_cell":
        sequences = _win_cell_sequences()
        topology = WIN_CELL_TOPOLOGY
    else:
        sequences = _gate_case("grpo_like").sequences
        topology = GATE_TOPOLOGY
    tree = planner.build_canonical_prefix_tree(sequences)
    first = planner.select_prefix_tree_layout(
        tree, refinement_work_budget=GENEROUS_BUDGET, **topology
    )
    second = planner.select_prefix_tree_layout(
        tree, refinement_work_budget=GENEROUS_BUDGET, **topology
    )
    assert first.layout.fingerprint == second.layout.fingerprint
