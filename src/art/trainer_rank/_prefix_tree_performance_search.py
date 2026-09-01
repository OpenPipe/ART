"""Bounded deterministic performance search over arbitrary prefix-tree layouts.

The canonical candidate family in :mod:`art.trainer_rank._prefix_tree_planner`
contains every mandatory anchor, but scalar depth/span policies cannot express
independent choices on different branches.  This module adds that missing
bounded search without introducing Megatron or model policy into the tree
primitives.

The caller supplies an integer-only cheap score.  Mandatory anchors are always
scored and retained.  Optional work is an anytime, bottom-up Pareto-beam search
whose budget is an integer count of refinement proposals; wall time, cache
state, and thread completion order never influence the result.  A downstream
planner can exact-lower ``shortlist`` while using ``incumbent`` as a safe cheap
score winner.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, TypeAlias

from ._prefix_tree_planner import (
    CanonicalPrefixTree,
    DecisionSet,
    FixedPointScore,
    LayoutCandidate,
    PrefixTreeLayout,
    _fingerprint,
    _normalize_score,
    plan_prefix_tree_layout,
    prefix_tree_layout_candidates,
)

SearchOperation: TypeAlias = Literal[
    "mandatory",
    "flip",
    "share_branch",
    "replay_branch",
    "neighborhood_flip",
]
SearchStopReason: TypeAlias = Literal[
    "no_shareable_decisions",
    "refinement_budget_exhausted",
    "search_complete",
]

_SEARCH_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class NonuniformSearchCandidate:
    """One scored layout and its deterministic discovery provenance."""

    candidate: LayoutCandidate
    score: tuple[int, ...]
    mandatory: bool
    discovery_tier: int
    operation: SearchOperation
    decision_index: int | None
    parent_selected_decisions: tuple[int, ...] | None
    refinement_decision_indices: tuple[int, ...] = ()
    # Derived, immutable orderings cached at construction: the Pareto beam
    # compares every candidate against the frontier many times, and
    # recomputing these tuples per comparison dominated search time.
    sorted_decisions: tuple[int, ...] = ()
    dominance: tuple[int, ...] = ()
    beam_key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        layout = self.candidate.layout
        sorted_decisions = tuple(sorted(layout.selected_decisions))
        dominance = (
            *self.score,
            layout.packed_tokens,
            len(layout.segments),
            len(layout.selected_decisions),
            layout.maximum_depth,
        )
        object.__setattr__(self, "sorted_decisions", sorted_decisions)
        object.__setattr__(self, "dominance", dominance)
        object.__setattr__(
            self,
            "beam_key",
            (
                self.score,
                layout.packed_tokens,
                len(layout.segments),
                len(layout.selected_decisions),
                layout.maximum_depth,
                sorted_decisions,
            ),
        )

    @property
    def layout(self) -> PrefixTreeLayout:
        return self.candidate.layout


@dataclass(frozen=True, slots=True)
class NonuniformLayoutSearch:
    """Anytime search result suitable for downstream exact-score shortlisting.

    ``refinement_work_budget`` applies only to optional branch proposals.
    Mandatory anchors are deliberately outside that budget: omitting an anchor
    is never an acceptable response to a small optimization budget.  Both
    components are exposed so the caller can price the complete planning miss.
    """

    candidates: tuple[NonuniformSearchCandidate, ...]
    shortlist: tuple[NonuniformSearchCandidate, ...]
    incumbent: NonuniformSearchCandidate
    mandatory_incumbent: NonuniformSearchCandidate
    mandatory_candidate_count: int
    refinement_work_budget: int
    refinement_work_used: int
    evaluated_refinements: int
    deduplicated_refinements: int
    completed_tiers: int
    completed_neighborhood_orders: int
    stop_reason: SearchStopReason
    fingerprint: str

    @property
    def total_scoring_work(self) -> int:
        """Number of layouts actually passed to the caller's scorer."""

        return self.mandatory_candidate_count + self.evaluated_refinements

    @property
    def improved_over_mandatory(self) -> bool:
        return _candidate_rank_key(self.incumbent) < _candidate_rank_key(
            self.mandatory_incumbent
        )


def _candidate_rank_key(
    candidate: NonuniformSearchCandidate,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return candidate.score, candidate.sorted_decisions


def _dominance_vector(candidate: NonuniformSearchCandidate) -> tuple[int, ...]:
    """Cheap dimensions retained for downstream-cost diversity.

    Score remains part of dominance, but token work, segment/edge overhead, and
    depth are kept independently.  Thus a slightly worse cheap score can remain
    in the beam when it exposes a substantially different physical schedule to
    the downstream exact lowerer.
    """

    return candidate.dominance


def _dominates(
    left: NonuniformSearchCandidate,
    right: NonuniformSearchCandidate,
) -> bool:
    left_values = left.dominance
    right_values = right.dominance
    if len(left_values) != len(right_values):
        raise ValueError("dominance vectors must have equal width")
    strictly_better = False
    for left_value, right_value in zip(left_values, right_values, strict=True):
        if left_value > right_value:
            return False
        if left_value < right_value:
            strictly_better = True
    return strictly_better


def _beam_key(candidate: NonuniformSearchCandidate) -> tuple[object, ...]:
    return candidate.beam_key


def _pareto_beam(
    candidates: Sequence[NonuniformSearchCandidate],
    *,
    beam_width: int,
) -> tuple[NonuniformSearchCandidate, ...]:
    """Return a bounded deterministic approximation of the Pareto frontier."""

    frontier: list[NonuniformSearchCandidate] = []
    for candidate in sorted(candidates, key=_beam_key):
        if any(_dominates(existing, candidate) for existing in frontier):
            continue
        frontier = [
            existing for existing in frontier if not _dominates(candidate, existing)
        ]
        frontier.append(candidate)
        frontier.sort(key=_beam_key)
        if len(frontier) > beam_width:
            del frontier[beam_width:]
    return tuple(frontier)


def _decision_subtrees(
    tree: CanonicalPrefixTree,
) -> dict[int, DecisionSet]:
    """Build decision-subtree memberships iteratively, with no depth ceiling."""

    decisions = frozenset(tree.decision_indices)
    nearest_decision: list[int | None] = [None] * len(tree.segments)
    decision_children: dict[int, list[int]] = {
        decision: [] for decision in tree.decision_indices
    }
    roots: list[int] = []
    for segment in tree.segments:
        parent_decision = (
            None
            if segment.parent_index is None
            else nearest_decision[segment.parent_index]
        )
        if segment.index in decisions:
            if parent_decision is None:
                roots.append(segment.index)
            else:
                decision_children[parent_decision].append(segment.index)
            nearest_decision[segment.index] = segment.index
        else:
            nearest_decision[segment.index] = parent_decision

    preorder: list[int] = []
    start: dict[int, int] = {}
    stop: dict[int, int] = {}
    stack: list[tuple[int, bool]] = [(root, False) for root in reversed(roots)]
    while stack:
        decision, exiting = stack.pop()
        if exiting:
            stop[decision] = len(preorder)
            continue
        start[decision] = len(preorder)
        preorder.append(decision)
        stack.append((decision, True))
        stack.extend((child, False) for child in reversed(decision_children[decision]))

    if len(preorder) != len(tree.decision_indices):
        raise AssertionError("decision forest did not cover every shareable node")
    return {
        decision: frozenset(preorder[start[decision] : stop[decision]])
        for decision in tree.decision_indices
    }


def search_nonuniform_prefix_tree_layouts(
    tree: CanonicalPrefixTree,
    score: Callable[[PrefixTreeLayout], FixedPointScore],
    *,
    refinement_work_budget: int,
    beam_width: int = 16,
    refinement_shortlist_size: int = 8,
    mandatory_candidates: Sequence[LayoutCandidate] | None = None,
) -> NonuniformLayoutSearch:
    """Generate and cheaply score bounded arbitrary subtree refinements.

    Search starts with the complete mandatory family.  It then visits decisions
    deepest-first.  For each Pareto-beam state, a tier may flip the current
    node, share its complete decision subtree, or replay that subtree.  States
    surviving one tier can be refined again at later tiers, so the search is
    not limited to one-flip layouts.

    One attempted optional operator consumes one unit from
    ``refinement_work_budget`` whether or not it deduplicates.  Mandatory score
    calls and optional score calls are reported separately.  The incumbent is
    the best fixed-point score across every evaluated layout and is therefore
    provably no worse, under the supplied cheap score, than the best mandatory
    anchor. A caller that already constructed the canonical mandatory family may
    pass it through ``mandatory_candidates``. The search validates and reuses
    that sequence without invoking :func:`prefix_tree_layout_candidates` again.
    """

    if refinement_work_budget < 0:
        raise ValueError("refinement_work_budget must be >= 0")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if refinement_shortlist_size < 1:
        raise ValueError("refinement_shortlist_size must be >= 1")

    mandatory_layouts = (
        prefix_tree_layout_candidates(tree)
        if mandatory_candidates is None
        else tuple(mandatory_candidates)
    )
    if not mandatory_layouts:
        raise ValueError("mandatory candidate family must not be empty")
    if any(
        not isinstance(candidate, LayoutCandidate) for candidate in mandatory_layouts
    ):
        raise TypeError("mandatory candidates must contain LayoutCandidate values")
    if any(
        candidate.layout.tree_fingerprint != tree.fingerprint
        for candidate in mandatory_layouts
    ):
        raise ValueError("mandatory candidate layout belongs to a different tree")
    decisions = frozenset(tree.decision_indices)
    if any(
        not candidate.layout.selected_decisions.issubset(decisions)
        for candidate in mandatory_layouts
    ):
        raise ValueError("mandatory candidate selects a decision absent from the tree")
    decision_keys = tuple(
        tuple(sorted(candidate.layout.selected_decisions))
        for candidate in mandatory_layouts
    )
    if decision_keys != tuple(sorted(decision_keys)):
        raise ValueError("mandatory candidates must use canonical decision-set order")
    if len(set(decision_keys)) != len(decision_keys):
        raise ValueError("mandatory candidates must contain unique decision sets")
    label_sets = tuple(candidate.labels for candidate in mandatory_layouts)
    if any(
        not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
        for labels in label_sets
    ):
        raise ValueError("mandatory candidate labels must be nonempty and unique")
    required_labels = frozenset(("no_sharing", "depth_one", "full_sharing"))
    observed_labels = frozenset(label for labels in label_sets for label in labels)
    if not required_labels.issubset(observed_labels):
        raise ValueError("mandatory candidates are missing a required anchor")
    expected_anchor_decisions = {
        "no_sharing": frozenset(),
        "depth_one": frozenset(
            index for index in tree.decision_indices if tree.segments[index].depth <= 1
        ),
        "full_sharing": decisions,
    }
    for label, expected_decisions in expected_anchor_decisions.items():
        anchors = tuple(
            candidate for candidate in mandatory_layouts if label in candidate.labels
        )
        if (
            len(anchors) != 1
            or anchors[0].layout.selected_decisions != expected_decisions
        ):
            raise ValueError(f"mandatory candidate {label!r} anchor is not canonical")
    if any(
        tuple(sorted(set(candidate.effective_span_thresholds)))
        != candidate.effective_span_thresholds
        or any(
            not isinstance(threshold, int)
            or isinstance(threshold, bool)
            or threshold < 0
            for threshold in candidate.effective_span_thresholds
        )
        for candidate in mandatory_layouts
    ):
        raise ValueError(
            "mandatory candidate effective-span thresholds must be canonical"
        )
    scored: list[NonuniformSearchCandidate] = []
    score_width: int | None = None

    def evaluate(
        candidate: LayoutCandidate,
        *,
        mandatory: bool,
        tier: int,
        operation: SearchOperation,
        decision_index: int | None,
        parent_selected_decisions: tuple[int, ...] | None,
        refinement_decisions: tuple[int, ...] = (),
    ) -> NonuniformSearchCandidate:
        nonlocal score_width
        normalized = _normalize_score(score(candidate.layout))
        if score_width is None:
            score_width = len(normalized)
        elif len(normalized) != score_width:
            raise ValueError("all layout scores must have the same tuple width")
        return NonuniformSearchCandidate(
            candidate=candidate,
            score=normalized,
            mandatory=mandatory,
            discovery_tier=tier,
            operation=operation,
            decision_index=decision_index,
            parent_selected_decisions=parent_selected_decisions,
            refinement_decision_indices=refinement_decisions,
        )

    for candidate in mandatory_layouts:
        scored.append(
            evaluate(
                candidate,
                mandatory=True,
                tier=0,
                operation="mandatory",
                decision_index=None,
                parent_selected_decisions=None,
            )
        )

    if not scored:
        raise AssertionError("mandatory prefix-tree family must not be empty")
    mandatory_incumbent = min(scored, key=_candidate_rank_key)
    incumbent = mandatory_incumbent
    seen = {candidate.layout.selected_decisions for candidate in scored}
    beam = _pareto_beam(scored, beam_width=beam_width)
    decision_subtrees = _decision_subtrees(tree)
    ordered_decisions = tuple(
        sorted(
            tree.decision_indices,
            key=lambda index: (-tree.segments[index].depth, index),
        )
    )

    work_used = 0
    evaluated_refinements = 0
    deduplicated_refinements = 0
    completed_tiers = 0
    stop_reason: SearchStopReason = "search_complete"

    for tier, decision in enumerate(ordered_decisions, start=1):
        tier_candidates: list[NonuniformSearchCandidate] = []
        branch = decision_subtrees[decision]
        # Snapshot ordering is important: thread timing or newly discovered
        # candidates cannot perturb the remainder of this tier.
        for parent in tuple(sorted(beam, key=_beam_key)):
            selected = parent.layout.selected_decisions
            proposals: tuple[tuple[SearchOperation, DecisionSet], ...] = (
                (
                    "flip",
                    (
                        selected.difference((decision,))
                        if decision in selected
                        else selected.union((decision,))
                    ),
                ),
                ("share_branch", selected.union(branch)),
                ("replay_branch", selected.difference(branch)),
            )
            for operation, proposal in proposals:
                if work_used >= refinement_work_budget:
                    stop_reason = "refinement_budget_exhausted"
                    break
                work_used += 1
                if proposal in seen:
                    deduplicated_refinements += 1
                    continue
                seen.add(proposal)
                layout = plan_prefix_tree_layout(tree, proposal)
                candidate = evaluate(
                    LayoutCandidate(
                        layout=layout,
                        labels=(
                            "nonuniform_refinement",
                            f"nonuniform_tier_{tier}",
                            f"nonuniform_{operation}_decision_{decision}",
                        ),
                        effective_span_thresholds=(),
                    ),
                    mandatory=False,
                    tier=tier,
                    operation=operation,
                    decision_index=decision,
                    parent_selected_decisions=tuple(
                        sorted(parent.layout.selected_decisions)
                    ),
                    refinement_decisions=(decision,),
                )
                scored.append(candidate)
                tier_candidates.append(candidate)
                evaluated_refinements += 1
                if _candidate_rank_key(candidate) < _candidate_rank_key(incumbent):
                    incumbent = candidate
            if stop_reason == "refinement_budget_exhausted":
                break
        if stop_reason == "refinement_budget_exhausted":
            break
        beam = _pareto_beam((*beam, *tier_candidates), beam_width=beam_width)
        completed_tiers += 1

    completed_neighborhood_orders = 0
    if stop_reason != "refinement_budget_exhausted":
        # A one-pass beam can prune a temporarily expensive partial assignment
        # even when a small coordinated branch change is profitable.  Spend
        # otherwise-unused budget on bounded Hamming neighborhoods around the
        # current incumbent.  Orders one through four cover common sibling and
        # parent/child rotations without turning the performance path into the
        # exhaustive feasibility search.
        for order in range(1, min(4, len(ordered_decisions)) + 1):
            base = incumbent
            neighborhood_best = incumbent
            for refinement in combinations(ordered_decisions, order):
                if work_used >= refinement_work_budget:
                    stop_reason = "refinement_budget_exhausted"
                    break
                work_used += 1
                proposal_values = set(base.layout.selected_decisions)
                for decision in refinement:
                    if decision in proposal_values:
                        proposal_values.remove(decision)
                    else:
                        proposal_values.add(decision)
                proposal = frozenset(proposal_values)
                if proposal in seen:
                    deduplicated_refinements += 1
                    continue
                seen.add(proposal)
                layout = plan_prefix_tree_layout(tree, proposal)
                tier = len(ordered_decisions) + order
                candidate = evaluate(
                    LayoutCandidate(
                        layout=layout,
                        labels=(
                            "nonuniform_refinement",
                            f"nonuniform_neighborhood_order_{order}",
                            "nonuniform_neighborhood_flip_"
                            + "_".join(str(value) for value in refinement),
                        ),
                        effective_span_thresholds=(),
                    ),
                    mandatory=False,
                    tier=tier,
                    operation="neighborhood_flip",
                    decision_index=None,
                    parent_selected_decisions=tuple(
                        sorted(base.layout.selected_decisions)
                    ),
                    refinement_decisions=refinement,
                )
                scored.append(candidate)
                evaluated_refinements += 1
                if _candidate_rank_key(candidate) < _candidate_rank_key(
                    neighborhood_best
                ):
                    neighborhood_best = candidate
            if _candidate_rank_key(neighborhood_best) < _candidate_rank_key(incumbent):
                incumbent = neighborhood_best
            if stop_reason == "refinement_budget_exhausted":
                break
            completed_neighborhood_orders += 1

    if not ordered_decisions:
        stop_reason = "no_shareable_decisions"

    # The mandatory family is never subjected to the optional shortlist cap.
    # Pin the optional incumbent as well, then fill the rest by stable cheap
    # rank.  This lets downstream exact scoring reconsider every semantic
    # anchor without losing the best result discovered by the bounded search.
    optional_ranked = sorted(
        (candidate for candidate in scored if not candidate.mandatory),
        key=_candidate_rank_key,
    )
    optional_shortlist = optional_ranked[:refinement_shortlist_size]
    if not incumbent.mandatory and incumbent not in optional_shortlist:
        optional_shortlist = [incumbent, *optional_shortlist]
        optional_shortlist = sorted(
            dict.fromkeys(optional_shortlist),
            key=_candidate_rank_key,
        )[:refinement_shortlist_size]
    shortlist = tuple(candidate for candidate in scored if candidate.mandatory) + tuple(
        optional_shortlist
    )

    if _candidate_rank_key(incumbent) > _candidate_rank_key(mandatory_incumbent):
        raise AssertionError("nonuniform search degraded its mandatory incumbent")

    fingerprint = _fingerprint(
        {
            "schema": _SEARCH_SCHEMA_VERSION,
            "tree_structure": tree.structure_fingerprint,
            "beam_width": beam_width,
            "refinement_shortlist_size": refinement_shortlist_size,
            "refinement_work_budget": refinement_work_budget,
            "refinement_work_used": work_used,
            "evaluated_refinements": evaluated_refinements,
            "deduplicated_refinements": deduplicated_refinements,
            "completed_tiers": completed_tiers,
            "completed_neighborhood_orders": completed_neighborhood_orders,
            "stop_reason": stop_reason,
            "incumbent": tuple(sorted(incumbent.layout.selected_decisions)),
            "incumbent_score": incumbent.score,
            "candidates": tuple(
                (
                    tuple(sorted(candidate.layout.selected_decisions)),
                    candidate.score,
                    candidate.mandatory,
                    candidate.discovery_tier,
                    candidate.operation,
                    candidate.decision_index,
                    candidate.parent_selected_decisions,
                    candidate.refinement_decision_indices,
                )
                for candidate in scored
            ),
            "shortlist": tuple(
                tuple(sorted(candidate.layout.selected_decisions))
                for candidate in shortlist
            ),
        }
    )
    return NonuniformLayoutSearch(
        candidates=tuple(scored),
        shortlist=shortlist,
        incumbent=incumbent,
        mandatory_incumbent=mandatory_incumbent,
        mandatory_candidate_count=len(mandatory_layouts),
        refinement_work_budget=refinement_work_budget,
        refinement_work_used=work_used,
        evaluated_refinements=evaluated_refinements,
        deduplicated_refinements=deduplicated_refinements,
        completed_tiers=completed_tiers,
        completed_neighborhood_orders=completed_neighborhood_orders,
        stop_reason=stop_reason,
        fingerprint=fingerprint,
    )


__all__ = [
    "NonuniformLayoutSearch",
    "NonuniformSearchCandidate",
    "SearchOperation",
    "SearchStopReason",
    "search_nonuniform_prefix_tree_layouts",
]
