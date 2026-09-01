"""Deterministic, shape-only building blocks for TrainerRank planning.

This module deliberately contains no Megatron, CUDA, distributed, or model
dependencies.  It describes prefix trees and local subforward plans using
immutable integer-valued records so every participating rank can independently
arrive at the same fingerprints and ordering.

The policy that predicts execution time lives above these primitives.  In
particular, :func:`prefix_tree_layout_candidates` constructs the mandatory
unbounded candidate family, while :func:`select_layout` supplies only stable
integer-score ordering and tie breaking.  :func:`select_prefix_tree_layout`
composes the candidate family, the calibrated production score, and the
bounded nonuniform refinement search into the one production selection entry
point.

Provenance: adopted from the holistic-planner research implementation frozen
2026-08-31, whose behavior was sealed by the final acceptance campaign
(oracle-exact bounded search on synthetic corpora; GPU win/tie cells).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import struct
from typing import Literal, TypeAlias, overload

Fingerprint: TypeAlias = str
DecisionSet: TypeAlias = frozenset[int]
FixedPointScore: TypeAlias = int | tuple[int, ...]

_TREE_SCHEMA_VERSION = 2
_LAYOUT_SCHEMA_VERSION = 2
_PARTITION_SCHEMA_VERSION = 1


_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")
_TOKEN_ROW_FINGERPRINT_DOMAIN = b"art.trainer_rank.canonical_token_row.v2\0"
_TOKEN_ROWS_FINGERPRINT_DOMAIN = b"art.trainer_rank.canonical_token_rows.v2\0"


def _fingerprint(payload: object) -> Fingerprint:
    """Hash a JSON-compatible canonical payload."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def canonical_token_row_fingerprint(tokens: Iterable[int]) -> Fingerprint:
    """Hash one token row with an explicit, cross-process int64 encoding."""

    digest = hashlib.sha256()
    digest.update(_TOKEN_ROW_FINGERPRINT_DOMAIN)
    count = 0
    for token in tokens:
        value = int(token)
        if value < -(1 << 63) or value >= 1 << 63:
            raise ValueError(f"token ID is outside signed int64: {value}")
        digest.update(_I64.pack(value))
        count += 1
    # Length suffixing distinguishes the domain even if the encoding changes
    # to a variable-width representation in a future schema.
    digest.update(_U64.pack(count))
    return digest.hexdigest()


def canonical_token_rows_fingerprint(
    rows: Sequence[tuple[int, str]],
) -> Fingerprint:
    """Compose ordered row digests without reading their tokens again."""

    digest = hashlib.sha256()
    digest.update(_TOKEN_ROWS_FINGERPRINT_DOMAIN)
    digest.update(_U64.pack(len(rows)))
    for length, fingerprint in rows:
        digest.update(_U64.pack(length))
        try:
            encoded = bytes.fromhex(fingerprint)
        except ValueError as exc:
            raise ValueError("canonical row fingerprint must be hexadecimal") from exc
        if len(encoded) != hashlib.sha256().digest_size:
            raise ValueError("canonical row fingerprint must be SHA-256")
        digest.update(encoded)
    return digest.hexdigest()


def _content_fingerprint(rows: Sequence[Sequence[int]]) -> Fingerprint:
    """Hash ordered rows through composable per-row int64 fingerprints."""

    return canonical_token_rows_fingerprint(
        tuple((len(row), canonical_token_row_fingerprint(row)) for row in rows)
    )


def _token_row(sequence: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(token) for token in sequence)


@dataclass(frozen=True, slots=True)
class CanonicalSegment:
    """One maximal segment of a full, unbounded radix tree."""

    index: int
    parent_index: int | None
    sequence_indices: tuple[int, ...]
    start: int
    end: int
    depth: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def shareable(self) -> bool:
        return len(self.sequence_indices) > 1


@dataclass(frozen=True, slots=True)
class CanonicalPrefixTree:
    """Canonical full radix tree for one checkpoint-equivalent sequence set."""

    sequence_lengths: tuple[int, ...]
    segments: tuple[CanonicalSegment, ...]
    terminal_segment_indices: tuple[int, ...]
    sequence_indices_by_terminal: tuple[tuple[int, ...], ...]
    decision_indices: tuple[int, ...]
    content_fingerprint: Fingerprint
    structure_fingerprint: Fingerprint
    fingerprint: Fingerprint

    @property
    def maximum_shared_depth(self) -> int:
        return max(
            (self.segments[index].depth for index in self.decision_indices),
            default=0,
        )


@dataclass(frozen=True, slots=True)
class CanonicalForestGroup:
    """One checkpoint-equivalent tree and its global sequence membership."""

    checkpoint_class: str
    sequence_indices: tuple[int, ...]
    tree: CanonicalPrefixTree


@dataclass(frozen=True, slots=True)
class CanonicalPrefixForest:
    """Stable forest spanning all checkpoint-equivalence classes in a call."""

    sequence_count: int
    groups: tuple[CanonicalForestGroup, ...]
    fingerprint: Fingerprint


@dataclass(frozen=True, slots=True)
class PlannedSegment:
    """One physical segment after applying arbitrary share/replay decisions."""

    sequence_indices: tuple[int, ...]
    start: int
    end: int
    parent_segment_index: int | None
    canonical_segment_index: int | None

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class PrefixTreeLayout:
    """A materialization-independent prefix-sharing layout."""

    tree_fingerprint: Fingerprint
    selected_decisions: DecisionSet
    segments: Sequence[PlannedSegment]
    packed_tokens: int
    maximum_depth: int
    fingerprint: Fingerprint


@dataclass(frozen=True, slots=True)
class _PlannedSegments(Sequence[PlannedSegment]):
    """Lazily expose one layout's segments without retaining every candidate.

    A deep radix tree has linearly many distinct uniform-depth and
    effective-span candidates.  Eagerly storing every candidate's linearly
    sized segment tuple makes the mandatory family quadratic in memory and
    spends most of its time allocating short-lived dataclasses.  The selected
    decision set and canonical tree determine the tuple exactly, so retain that
    compact representation and replay the deterministic parent-order walk only
    for candidates a scorer or materializer actually inspects.
    """

    tree: CanonicalPrefixTree
    selected_decisions: DecisionSet
    segment_count: int

    def __len__(self) -> int:
        return self.segment_count

    def __iter__(self) -> Iterator[PlannedSegment]:
        return _iter_planned_segments(self.tree, self.selected_decisions)

    @overload
    def __getitem__(self, index: int) -> PlannedSegment: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[PlannedSegment]: ...

    def __getitem__(
        self, index: int | slice
    ) -> PlannedSegment | Sequence[PlannedSegment]:
        # Random access is not used by the hot planner path.  Keeping it here
        # makes the object a normal Sequence for diagnostics and tests.
        values = tuple(self)
        return values[index]


@dataclass(frozen=True, slots=True)
class LayoutCandidate:
    """A distinct mandatory layout and all policies that produced it."""

    layout: PrefixTreeLayout
    labels: tuple[str, ...]
    effective_span_thresholds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LayoutSelection:
    """Deterministically selected layout plus the fixed-point score used."""

    candidate: LayoutCandidate
    score: tuple[int, ...]
    evaluated_candidates: int


def _finalize_canonical_prefix_tree(
    *,
    sequence_lengths: tuple[int, ...],
    segments: tuple[CanonicalSegment, ...],
    content_fingerprint: Fingerprint,
) -> CanonicalPrefixTree:
    """Finalize shared tree identities and terminal metadata exactly once."""

    structure_payload = {
        "schema": _TREE_SCHEMA_VERSION,
        "sequence_lengths": sequence_lengths,
        "segments": tuple(
            (
                segment.parent_index,
                segment.sequence_indices,
                segment.start,
                segment.end,
            )
            for segment in segments
        ),
    }
    structure_fingerprint = _fingerprint(structure_payload)
    tree_fingerprint = _fingerprint(
        {
            "schema": _TREE_SCHEMA_VERSION,
            "content": content_fingerprint,
            "structure": structure_fingerprint,
        }
    )
    terminal_segment_indices: list[int | None] = [None] * len(sequence_lengths)
    for segment in segments:
        for sequence_index in segment.sequence_indices:
            terminal_segment_indices[sequence_index] = segment.index
    if any(index is None for index in terminal_segment_indices):
        raise AssertionError("canonical tree does not cover every sequence")
    terminal_values = tuple(
        int(index) for index in terminal_segment_indices if index is not None
    )
    sequences_by_terminal: list[list[int]] = [[] for _ in segments]
    for sequence_index, terminal_index in enumerate(terminal_values):
        sequences_by_terminal[terminal_index].append(sequence_index)
    return CanonicalPrefixTree(
        sequence_lengths=sequence_lengths,
        segments=segments,
        terminal_segment_indices=terminal_values,
        sequence_indices_by_terminal=tuple(
            tuple(indices) for indices in sequences_by_terminal
        ),
        decision_indices=tuple(
            segment.index for segment in segments if segment.shareable
        ),
        content_fingerprint=content_fingerprint,
        structure_fingerprint=structure_fingerprint,
        fingerprint=tree_fingerprint,
    )


def build_canonical_prefix_tree(
    sequences: Iterable[Sequence[int]],
) -> CanonicalPrefixTree:
    """Build a full radix tree iteratively, with no depth limit.

    The input order is semantic: sequence indices and first-seen sibling order
    are retained.  Parents are always emitted before children, which permits
    all subsequent transforms to remain iterative as well.
    """

    rows = tuple(_token_row(sequence) for sequence in sequences)
    if not rows:
        raise ValueError("a canonical prefix tree requires at least one sequence")
    if any(not row for row in rows):
        raise ValueError("prefix-tree sequences must not be empty")

    lengths = tuple(len(row) for row in rows)
    segments: list[CanonicalSegment] = []
    tasks: list[tuple[tuple[int, ...], int, int | None]] = [
        (tuple(range(len(rows))), 0, None)
    ]
    while tasks:
        indices, start, parent_index = tasks.pop()
        active = tuple(index for index in indices if lengths[index] > start)
        if not active:
            continue
        depth = 1 if parent_index is None else segments[parent_index].depth + 1
        if len(active) == 1:
            segment_index = len(segments)
            segments.append(
                CanonicalSegment(
                    index=segment_index,
                    parent_index=parent_index,
                    sequence_indices=active,
                    start=start,
                    end=lengths[active[0]],
                    depth=depth,
                )
            )
            continue

        shared_end = min(lengths[index] for index in active)
        reference = rows[active[0]]
        cursor = start
        while cursor < shared_end and all(
            rows[index][cursor] == reference[cursor] for index in active[1:]
        ):
            cursor += 1
        if cursor > start:
            segment_index = len(segments)
            segments.append(
                CanonicalSegment(
                    index=segment_index,
                    parent_index=parent_index,
                    sequence_indices=active,
                    start=start,
                    end=cursor,
                    depth=depth,
                )
            )
            tasks.append((active, cursor, segment_index))
            continue

        # Preserve first-seen sibling order.  Pushing in reverse makes the
        # explicit stack produce the same order as a deterministic DFS.
        groups: dict[int, list[int]] = {}
        sibling_tokens: list[int] = []
        for sequence_index in active:
            token = rows[sequence_index][start]
            if token not in groups:
                groups[token] = []
                sibling_tokens.append(token)
            groups[token].append(sequence_index)
        for token in reversed(sibling_tokens):
            tasks.append((tuple(groups[token]), start, parent_index))

    return _finalize_canonical_prefix_tree(
        sequence_lengths=lengths,
        segments=tuple(segments),
        content_fingerprint=_content_fingerprint(rows),
    )


def build_canonical_prefix_forest(
    sequences: Sequence[Sequence[int]],
    checkpoint_classes: Sequence[str],
) -> CanonicalPrefixForest:
    """Build a stable forest without sharing across checkpoint classes."""

    if len(sequences) != len(checkpoint_classes):
        raise ValueError("one checkpoint class is required per sequence")
    if not sequences:
        return CanonicalPrefixForest(
            sequence_count=0,
            groups=(),
            fingerprint=_fingerprint(
                {"schema": _TREE_SCHEMA_VERSION, "sequence_count": 0, "groups": ()}
            ),
        )

    indices_by_class: dict[str, list[int]] = {}
    for sequence_index, checkpoint_class in enumerate(checkpoint_classes):
        if not isinstance(checkpoint_class, str):
            raise TypeError("checkpoint classes must be canonical strings")
        indices_by_class.setdefault(checkpoint_class, []).append(sequence_index)

    groups = tuple(
        CanonicalForestGroup(
            checkpoint_class=checkpoint_class,
            sequence_indices=tuple(indices),
            tree=build_canonical_prefix_tree(
                tuple(sequences[index] for index in indices)
            ),
        )
        for checkpoint_class, indices in sorted(indices_by_class.items())
    )
    return CanonicalPrefixForest(
        sequence_count=len(sequences),
        groups=groups,
        fingerprint=_fingerprint(
            {
                "schema": _TREE_SCHEMA_VERSION,
                "sequence_count": len(sequences),
                "groups": tuple(
                    (
                        group.checkpoint_class,
                        group.sequence_indices,
                        group.tree.fingerprint,
                    )
                    for group in groups
                ),
            }
        ),
    )


def plan_prefix_tree_layout(
    tree: CanonicalPrefixTree,
    selected_decisions: Iterable[int],
) -> PrefixTreeLayout:
    """Apply an arbitrary set of share/replay decisions to ``tree``."""

    selected = frozenset(int(index) for index in selected_decisions)
    unknown = selected.difference(tree.decision_indices)
    if unknown:
        raise ValueError(f"unknown or unshareable decisions: {sorted(unknown)}")

    packed_tokens, maximum_depth, segment_count = _layout_shape(tree, selected)
    layout_payload = {
        "schema": _LAYOUT_SCHEMA_VERSION,
        "tree": tree.fingerprint,
        # The full canonical tree plus this set uniquely determines segment
        # spans, ordering, and parents.  Hashing the derived segment tuple again
        # made deep candidate generation quadratic without adding identity.
        "selected": tuple(sorted(selected)),
    }
    return PrefixTreeLayout(
        tree_fingerprint=tree.fingerprint,
        selected_decisions=selected,
        segments=_PlannedSegments(tree, selected, segment_count),
        packed_tokens=packed_tokens,
        maximum_depth=maximum_depth,
        fingerprint=_fingerprint(layout_payload),
    )


def _nearest_selected_segments(
    tree: CanonicalPrefixTree,
    selected: DecisionSet,
) -> tuple[list[int | None], list[int | None]]:
    nearest: list[int | None] = [None] * len(tree.segments)
    selected_parents: list[int | None] = [None] * len(tree.segments)
    for canonical in tree.segments:
        parent = (
            None if canonical.parent_index is None else nearest[canonical.parent_index]
        )
        if canonical.index in selected:
            selected_parents[canonical.index] = parent
            nearest[canonical.index] = canonical.index
        else:
            nearest[canonical.index] = parent
    return nearest, selected_parents


def _layout_shape(
    tree: CanonicalPrefixTree,
    selected: DecisionSet,
) -> tuple[int, int, int]:
    """Compute layout metrics without allocating the derived segment tuple."""

    nearest, selected_parents = _nearest_selected_segments(tree, selected)
    selected_depths = [0] * len(tree.segments)
    packed_tokens = 0
    maximum_depth = 0
    segment_count = len(selected)
    for canonical in tree.segments:
        if canonical.index not in selected:
            continue
        parent = selected_parents[canonical.index]
        start = 0 if parent is None else tree.segments[parent].end
        if start >= canonical.end:
            raise ValueError("a selected canonical segment has an empty span")
        depth = 1 if parent is None else selected_depths[parent] + 1
        selected_depths[canonical.index] = depth
        if depth > maximum_depth:
            maximum_depth = depth
        packed_tokens += canonical.end - start

    for terminal, sequence_indices in enumerate(tree.sequence_indices_by_terminal):
        if not sequence_indices:
            continue
        deepest = nearest[terminal]
        start = 0 if deepest is None else tree.segments[deepest].end
        sequence_length = tree.sequence_lengths[sequence_indices[0]]
        if start >= sequence_length:
            continue
        segment_count += len(sequence_indices)
        packed_tokens += (sequence_length - start) * len(sequence_indices)
        depth = 1 if deepest is None else selected_depths[deepest] + 1
        if depth > maximum_depth:
            maximum_depth = depth
    return packed_tokens, maximum_depth, segment_count


def _iter_planned_segments(
    tree: CanonicalPrefixTree,
    selected: DecisionSet,
) -> Iterator[PlannedSegment]:
    """Yield the exact parent-before-child layout in canonical event order."""

    nearest, selected_parents = _nearest_selected_segments(tree, selected)
    planned_by_canonical: list[int | None] = [None] * len(tree.segments)
    planned_index = 0
    for canonical in tree.segments:
        if canonical.index in selected:
            parent_canonical = selected_parents[canonical.index]
            parent_segment = (
                None
                if parent_canonical is None
                else planned_by_canonical[parent_canonical]
            )
            if parent_canonical is not None and parent_segment is None:
                raise AssertionError("selected parent was not emitted before its child")
            start = (
                0 if parent_canonical is None else tree.segments[parent_canonical].end
            )
            yield PlannedSegment(
                sequence_indices=canonical.sequence_indices,
                start=start,
                end=canonical.end,
                parent_segment_index=parent_segment,
                canonical_segment_index=canonical.index,
            )
            planned_by_canonical[canonical.index] = planned_index
            planned_index += 1

        sequence_indices = tree.sequence_indices_by_terminal[canonical.index]
        if not sequence_indices:
            continue
        deepest = nearest[canonical.index]
        start = 0 if deepest is None else tree.segments[deepest].end
        sequence_length = tree.sequence_lengths[sequence_indices[0]]
        if start >= sequence_length:
            continue
        for sequence_index in sequence_indices:
            parent_segment = None if deepest is None else planned_by_canonical[deepest]
            if deepest is not None and parent_segment is None:
                raise AssertionError("selected parent was not emitted before its tail")
            yield PlannedSegment(
                sequence_indices=(sequence_index,),
                start=start,
                end=sequence_length,
                parent_segment_index=parent_segment,
                canonical_segment_index=None,
            )
            planned_index += 1


def effective_span_decisions(
    tree: CanonicalPrefixTree,
    minimum_span: int,
) -> DecisionSet:
    """Select every shared node whose effective span reaches a threshold."""

    if minimum_span < 0:
        raise ValueError("minimum_span must be >= 0")
    decisions = set(tree.decision_indices)
    selected: set[int] = set()
    nearest_selected: list[int | None] = [None] * len(tree.segments)
    for segment in tree.segments:
        selected_parent = (
            None
            if segment.parent_index is None
            else nearest_selected[segment.parent_index]
        )
        start = 0 if selected_parent is None else tree.segments[selected_parent].end
        if segment.index in decisions and segment.end - start >= minimum_span:
            selected.add(segment.index)
            nearest_selected[segment.index] = segment.index
        else:
            nearest_selected[segment.index] = selected_parent
    return frozenset(selected)


def effective_span_breakpoints(tree: CanonicalPrefixTree) -> tuple[int, ...]:
    """Return every distinct transition threshold for effective-span policy.

    A node's span can begin at any selected ancestor, so transitions include
    every positive node-to-decision-ancestor distance plus one.  This finite
    family covers all nonnegative integer thresholds without imposing a tree
    depth bound or sweeping all token positions.
    """

    transitions = {1}
    decisions = set(tree.decision_indices)
    for index in tree.decision_indices:
        segment = tree.segments[index]
        transitions.add(segment.end + 1)
        ancestor = segment.parent_index
        while ancestor is not None:
            if ancestor in decisions:
                span = segment.end - tree.segments[ancestor].end
                if span > 0:
                    transitions.add(span + 1)
            ancestor = tree.segments[ancestor].parent_index
    return tuple(sorted(transitions))


def _uniform_decisions(tree: CanonicalPrefixTree, depth: int) -> DecisionSet:
    return frozenset(
        index for index in tree.decision_indices if tree.segments[index].depth <= depth
    )


def _candidate_tie_key(candidate: LayoutCandidate) -> tuple[object, ...]:
    return (
        len(candidate.layout.selected_decisions),
        candidate.layout.maximum_depth,
        len(candidate.layout.segments),
        candidate.layout.packed_tokens,
        tuple(sorted(candidate.layout.selected_decisions)),
    )


def prefix_tree_layout_candidates(
    tree: CanonicalPrefixTree,
) -> tuple[LayoutCandidate, ...]:
    """Build the complete deterministic mandatory candidate family.

    Equivalent layouts are emitted once, with all originating labels retained.
    The family contains no sharing, depth one, full sharing, every uniform
    depth, every effective-span transition, and the 90/95/99 percent physical
    token-savings plateaus.
    """

    labels: dict[DecisionSet, list[str]] = {}
    thresholds: dict[DecisionSet, list[int]] = {}

    def add(
        decisions: DecisionSet,
        label: str,
        *,
        threshold: int | None = None,
    ) -> None:
        current_labels = labels.setdefault(decisions, [])
        if label not in current_labels:
            current_labels.append(label)
        if threshold is not None:
            current_thresholds = thresholds.setdefault(decisions, [])
            if threshold not in current_thresholds:
                current_thresholds.append(threshold)

    no_sharing = frozenset()
    add(no_sharing, "no_sharing")
    add(no_sharing, "uniform_depth_0")
    maximum_depth = tree.maximum_shared_depth
    depth_one = _uniform_decisions(tree, 1)
    add(depth_one, "depth_one")
    add(depth_one, "uniform_depth_1")
    for depth in range(2, maximum_depth + 1):
        add(_uniform_decisions(tree, depth), f"uniform_depth_{depth}")

    for threshold in effective_span_breakpoints(tree):
        add(
            effective_span_decisions(tree, threshold),
            f"minimum_effective_span_{threshold}",
            threshold=threshold,
        )

    full_sharing = frozenset(tree.decision_indices)
    add(full_sharing, "full_sharing")

    layouts = {
        decisions: plan_prefix_tree_layout(tree, decisions) for decisions in labels
    }
    maximum_savings = (
        layouts[no_sharing].packed_tokens - layouts[full_sharing].packed_tokens
    )
    for percentage in (90, 95, 99):
        eligible = []
        for decisions, layout in layouts.items():
            savings = layouts[no_sharing].packed_tokens - layout.packed_tokens
            if maximum_savings > 0 and savings * 100 < maximum_savings * percentage:
                continue
            eligible.append(
                LayoutCandidate(
                    layout=layout,
                    labels=tuple(labels[decisions]),
                    effective_span_thresholds=tuple(thresholds.get(decisions, ())),
                )
            )
        if not eligible:
            raise AssertionError("full sharing must satisfy every savings plateau")
        plateau = min(eligible, key=_candidate_tie_key)
        add(plateau.layout.selected_decisions, f"savings_plateau_{percentage}")

    return tuple(
        LayoutCandidate(
            layout=layouts[decisions],
            labels=tuple(labels[decisions]),
            effective_span_thresholds=tuple(thresholds.get(decisions, ())),
        )
        for decisions in sorted(labels, key=lambda value: tuple(sorted(value)))
    )


def iter_all_prefix_tree_layouts(
    tree: CanonicalPrefixTree,
    *,
    exclude: Iterable[DecisionSet] = (),
) -> Iterator[PrefixTreeLayout]:
    """Yield every legal share/replay layout without a depth or count cutoff.

    This iterator is reserved for memory-feasibility recovery.  Normal
    performance planning uses :func:`prefix_tree_layout_candidates`; if that
    bounded family cannot find a feasible complete call, correctness requires
    considering every independent internal-subtree decision before reporting
    an OOM.  Layouts with more sharing are visited first because they usually
    reduce physical rows, but the ordering is only an optimization: no layout
    is omitted and there is no elapsed-time or candidate-count budget.

    ``exclude`` lets a caller skip layouts already evaluated by the bounded
    family while retaining deterministic exhaustive coverage of the residual
    decision space.
    """

    decisions = tree.decision_indices
    excluded = frozenset(frozenset(value) for value in exclude)
    for selected_count in range(len(decisions), -1, -1):
        for selected_tuple in combinations(decisions, selected_count):
            selected = frozenset(selected_tuple)
            if selected in excluded:
                continue
            yield plan_prefix_tree_layout(tree, selected)


def _normalize_score(score: FixedPointScore) -> tuple[int, ...]:
    values = (
        (score,) if isinstance(score, int) and not isinstance(score, bool) else score
    )
    if not isinstance(values, tuple) or not values:
        raise TypeError("layout scores must be a nonempty int tuple")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise TypeError("layout scores must contain only fixed-point integers")
    return values


def select_layout(
    candidates: Sequence[LayoutCandidate],
    score: Callable[[PrefixTreeLayout], FixedPointScore],
) -> LayoutSelection:
    """Select by integer score, then by stable layout fingerprint."""

    if not candidates:
        raise ValueError("at least one layout candidate is required")
    evaluated = tuple(
        (_normalize_score(score(candidate.layout)), candidate)
        for candidate in candidates
    )
    selected_score, selected = min(
        evaluated,
        key=lambda value: (value[0], value[1].layout.fingerprint),
    )
    return LayoutSelection(
        candidate=selected,
        score=selected_score,
        evaluated_candidates=len(evaluated),
    )


@dataclass(frozen=True, slots=True)
class SubforwardMemory:
    """Conservative incremental memory for one graph-producing subforward."""

    retained_bytes: int
    forward_peak_bytes: int
    backward_workspace_bytes: int

    def __post_init__(self) -> None:
        if (
            min(
                self.retained_bytes,
                self.forward_peak_bytes,
                self.backward_workspace_bytes,
            )
            < 0
        ):
            raise ValueError("subforward memory values must be nonnegative")
        if self.forward_peak_bytes < self.retained_bytes:
            raise ValueError(
                "forward peak must include the subforward's retained bytes"
            )

    @property
    def ephemeral_tail_bytes(self) -> int:
        """Ordering priority for minimizing cumulative forward peak."""

        return self.forward_peak_bytes - self.retained_bytes


@dataclass(frozen=True, slots=True)
class LocalSubforward:
    """One local execution unit; item order is part of the plan."""

    item_indices: tuple[int, ...]
    memory: SubforwardMemory
    layout_fingerprints: tuple[Fingerprint, ...] = ()
    compile_signatures: tuple[Fingerprint, ...] = ()

    def __post_init__(self) -> None:
        if not self.item_indices:
            raise ValueError("a subforward must contain at least one item")
        if len(set(self.item_indices)) != len(self.item_indices):
            raise ValueError("an item may occur only once within a subforward")
        if any(index < 0 for index in self.item_indices):
            raise ValueError("item indices must be nonnegative")

    @property
    def fingerprint(self) -> Fingerprint:
        return _fingerprint(
            {
                "schema": _PARTITION_SCHEMA_VERSION,
                "items": self.item_indices,
                "memory": (
                    self.memory.retained_bytes,
                    self.memory.forward_peak_bytes,
                    self.memory.backward_workspace_bytes,
                ),
                "layouts": self.layout_fingerprints,
                "compile": self.compile_signatures,
            }
        )


@dataclass(frozen=True, slots=True)
class LocalForwardPlan:
    """Complete local partition for one user-visible ``dp_rank_forward``."""

    item_count: int
    subforwards: tuple[LocalSubforward, ...]
    fingerprint: Fingerprint

    @classmethod
    def build(
        cls,
        item_count: int,
        subforwards: Iterable[LocalSubforward],
    ) -> LocalForwardPlan:
        if item_count < 0:
            raise ValueError("item_count must be nonnegative")
        values = tuple(subforwards)
        observed = tuple(
            index for subforward in values for index in subforward.item_indices
        )
        if sorted(observed) != list(range(item_count)):
            raise ValueError("subforwards must contain every item exactly once")
        payload = {
            "schema": _PARTITION_SCHEMA_VERSION,
            "item_count": item_count,
            "subforwards": tuple(value.fingerprint for value in values),
        }
        return cls(
            item_count=item_count,
            subforwards=values,
            fingerprint=_fingerprint(payload),
        )

    @classmethod
    def empty(cls) -> LocalForwardPlan:
        return cls.build(0, ())


@dataclass(frozen=True, slots=True)
class CumulativeMemory:
    """Complete-call memory bounds while all returned graphs stay live."""

    forward_peak_bytes: int
    returned_live_bytes: int
    backward_upper_bound_bytes: int
    peak_bytes: int


def cumulative_memory(
    subforwards: Sequence[LocalSubforward],
    *,
    base_bytes: int = 0,
    gradient_bytes: int = 0,
) -> CumulativeMemory:
    """Evaluate the specified subforward order under the graph-live contract."""

    if base_bytes < 0 or gradient_bytes < 0:
        raise ValueError("base and gradient memory must be nonnegative")
    if not subforwards:
        return CumulativeMemory(
            forward_peak_bytes=base_bytes,
            returned_live_bytes=base_bytes,
            backward_upper_bound_bytes=base_bytes,
            peak_bytes=base_bytes,
        )
    retained = 0
    forward_peak = base_bytes
    maximum_backward_workspace = 0
    for subforward in subforwards:
        forward_peak = max(
            forward_peak,
            base_bytes + retained + subforward.memory.forward_peak_bytes,
        )
        retained += subforward.memory.retained_bytes
        maximum_backward_workspace = max(
            maximum_backward_workspace,
            subforward.memory.backward_workspace_bytes,
        )
    returned_live = base_bytes + retained
    backward_upper_bound = returned_live + gradient_bytes + maximum_backward_workspace
    return CumulativeMemory(
        forward_peak_bytes=forward_peak,
        returned_live_bytes=returned_live,
        backward_upper_bound_bytes=backward_upper_bound,
        peak_bytes=max(forward_peak, returned_live, backward_upper_bound),
    )


def memory_optimal_subforward_order(
    subforwards: Iterable[LocalSubforward],
) -> tuple[LocalSubforward, ...]:
    """Return the exact order minimizing the conservative forward peak.

    This is the standard maximum-lateness interchange rule: execute larger
    ``forward_peak - retained`` tails first.  Stable fingerprints break equal
    tails, so independent ranks produce an identical order.
    """

    return tuple(
        sorted(
            subforwards,
            key=lambda subforward: (
                -subforward.memory.ephemeral_tail_bytes,
                subforward.fingerprint,
            ),
        )
    )


def memory_feasible(
    subforwards: Sequence[LocalSubforward],
    *,
    limit_bytes: int,
    base_bytes: int = 0,
    gradient_bytes: int = 0,
) -> bool:
    """Return whether the complete forward/returned/backward bound fits."""

    if limit_bytes < 0:
        raise ValueError("limit_bytes must be nonnegative")
    return (
        cumulative_memory(
            subforwards,
            base_bytes=base_bytes,
            gradient_bytes=gradient_bytes,
        ).peak_bytes
        <= limit_bytes
    )


PlannerScope = Literal["global", "dp_local"]


@dataclass(frozen=True, slots=True)
class PlannerIdentity:
    """Versioned common-state identity included in distributed plan digests."""

    scope: PlannerScope
    input_fingerprint: Fingerprint
    topology_fingerprint: Fingerprint
    capability_fingerprint: Fingerprint
    planner_version: int
    coefficient_version: int
    seam_version: int

    def __post_init__(self) -> None:
        if self.scope not in ("global", "dp_local"):
            raise ValueError(f"unknown planner scope: {self.scope}")
        if (
            min(
                self.planner_version,
                self.coefficient_version,
                self.seam_version,
            )
            < 0
        ):
            raise ValueError("planner identity versions must be nonnegative")

    @property
    def fingerprint(self) -> Fingerprint:
        return _fingerprint(
            {
                "scope": self.scope,
                "input": self.input_fingerprint,
                "topology": self.topology_fingerprint,
                "capability": self.capability_fingerprint,
                "planner": self.planner_version,
                "coefficients": self.coefficient_version,
                "seam": self.seam_version,
            }
        )


def search_prefix_tree_layout(
    tree: CanonicalPrefixTree,
    score: Callable[[PrefixTreeLayout], FixedPointScore],
    *,
    refinement_work_budget: int,
    mandatory_candidates: Sequence[LayoutCandidate] | None = None,
):
    """Run the bounded nonuniform search and return its incumbent.

    The incumbent is provably no worse, under ``score``, than every mandatory
    anchor; with sufficient budget the bounded search recovers exhaustive
    optima on small trees (sealed research gate: oracle-exact on all corpus
    families at budget 2000).
    """

    from ._prefix_tree_performance_search import (
        search_nonuniform_prefix_tree_layouts,
    )

    return search_nonuniform_prefix_tree_layouts(
        tree,
        score,
        refinement_work_budget=refinement_work_budget,
        mandatory_candidates=mandatory_candidates,
    ).incumbent


def select_prefix_tree_layout(
    tree: CanonicalPrefixTree,
    *,
    cp_size: int,
    layers: int,
    uses_gdn: bool,
    refinement_work_budget: int,
):
    """Production layout selection for one canonical tree.

    Composes the mandatory candidate family, the calibrated production score,
    and the bounded nonuniform refinement search.  Deterministic: identical
    trees, topology facts, and budgets yield identical layouts on every rank.
    """

    from ._planner_cost import prefix_tree_layout_score

    def scorer(layout: PrefixTreeLayout) -> tuple[int, ...]:
        return prefix_tree_layout_score(
            layout,
            cp_size=cp_size,
            layers=layers,
            uses_gdn=uses_gdn,
        )

    return search_prefix_tree_layout(
        tree,
        scorer,
        refinement_work_budget=refinement_work_budget,
        mandatory_candidates=prefix_tree_layout_candidates(tree),
    )


__all__ = [
    "CanonicalForestGroup",
    "CanonicalPrefixForest",
    "CanonicalPrefixTree",
    "CanonicalSegment",
    "CumulativeMemory",
    "DecisionSet",
    "Fingerprint",
    "FixedPointScore",
    "LayoutCandidate",
    "LayoutSelection",
    "LocalForwardPlan",
    "LocalSubforward",
    "PlannedSegment",
    "PlannerIdentity",
    "PlannerScope",
    "PrefixTreeLayout",
    "SubforwardMemory",
    "build_canonical_prefix_forest",
    "build_canonical_prefix_tree",
    "cumulative_memory",
    "effective_span_breakpoints",
    "effective_span_decisions",
    "memory_feasible",
    "memory_optimal_subforward_order",
    "iter_all_prefix_tree_layouts",
    "plan_prefix_tree_layout",
    "prefix_tree_layout_candidates",
    "search_prefix_tree_layout",
    "select_layout",
    "select_prefix_tree_layout",
]
