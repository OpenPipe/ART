"""Calibrated integer cost model for prefix-tree layout selection.

The score is a lexicographic fixed-point tuple: predicted work first, then
packed tokens, segment count, and maximum depth as deterministic tie-breaks.
All terms are integers so every rank computes bit-identical scores.

Provenance: the formula and constants are the research implementation's
production layout score (frozen 2026-08-31), mirrored and test-locked by its
sealed nonuniform-search gate and validated end-to-end by the sealed GPU
acceptance cells (GRPO GDN CP4 win: automatic selected depth 3, packing
26,624 physical tokens for 172,032 logical, +47.2% paired median gain vs the
depth-one arm; heterogeneous/Ellavox CP1: correctly converged to the
depth-one-equivalent plan).

Known limitation carried from research (documented, not addressed here): the
GDN depth terms overprice deep sharing on some GRPO cells — the sealed
full-sharing arm measured faster than the automatic selection on the win
cell.  Constants are versioned via ``COEFFICIENT_VERSION`` so a future
recalibration invalidates cached recipes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._prefix_tree_planner import PrefixTreeLayout

COEFFICIENT_VERSION = 1

# Segment-length buckets for kernel-utilization effects: a segment shorter than
# these runs small-M kernels on every rank it is sharded over. Context
# parallelism shards a segment across ranks, so the scorer reads the bucket
# for ``threshold x cp`` (per-rank length); the histogram covers 64..8192.
SMALL_SEGMENT_TOKENS = 512
TINY_SEGMENT_TOKENS = 128
SEGMENT_LENGTH_THRESHOLDS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True, slots=True)
class LayoutFeatures:
    """Integer, O(segments) features of one layout that differ between layouts.

    Everything a call shares across its candidate layouts (logical tokens, the
    output head, model size) cancels in ranking, so only layout-dependent
    quantities are here. Each is derivable from the planned segments alone, so
    the calibration harness logs exactly what a scorer can price.
    """

    packed_tokens: int
    segment_count: int
    # Shared (fan-out > 1) segments; equals the number of selected decisions.
    shared_segments: int
    # Dependency levels including the root and tail levels: no sharing = 1.
    max_depth: int
    # Tokens held in shared segments: the physical rows whose state/KV is
    # handed on to more than one continuation.
    shared_tokens: int
    # Sum of fan-out over shared segments: state hand-offs per layer.
    fanout_sum: int
    small_segments: int
    tiny_segments: int
    # Causal attention pairs over physical rows: sum of len * (start + len/2).
    attention_area: int
    # Segments shorter than each SEGMENT_LENGTH_THRESHOLDS entry (cumulative),
    # and the tokens they hold: rows that run small-M kernels per rank.
    segments_below: tuple[int, ...] = ()
    tokens_below: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def below(self, tokens: int) -> int:
        """Segments shorter than ``tokens`` (nearest histogram bucket at or above)."""

        return _bucket(self.segments_below, tokens)

    def tokens_in_segments_below(self, tokens: int) -> int:
        """Tokens held in segments shorter than ``tokens``."""

        return _bucket(self.tokens_below, tokens)


def _bucket(cumulative: tuple[int, ...], tokens: int) -> int:
    for threshold, count in zip(SEGMENT_LENGTH_THRESHOLDS, cumulative, strict=False):
        if threshold >= tokens:
            return count
    return cumulative[-1] if cumulative else 0


def layout_features(layout: PrefixTreeLayout) -> LayoutFeatures:
    segment_count = 0
    shared_segments = 0
    shared_tokens = 0
    fanout_sum = 0
    small = 0
    tiny = 0
    area = 0
    below = [0] * len(SEGMENT_LENGTH_THRESHOLDS)
    tokens_below = [0] * len(SEGMENT_LENGTH_THRESHOLDS)
    for segment in layout.segments:
        length = segment.end - segment.start
        fanout = len(segment.sequence_indices)
        segment_count += 1
        if fanout > 1:
            shared_segments += 1
            shared_tokens += length
            fanout_sum += fanout
        if length < SMALL_SEGMENT_TOKENS:
            small += 1
        if length < TINY_SEGMENT_TOKENS:
            tiny += 1
        for index, threshold in enumerate(SEGMENT_LENGTH_THRESHOLDS):
            if length < threshold:
                below[index] += 1
                tokens_below[index] += length
        area += length * segment.start + (length * length) // 2
    return LayoutFeatures(
        packed_tokens=layout.packed_tokens,
        segment_count=segment_count,
        shared_segments=shared_segments,
        max_depth=layout.maximum_depth,
        shared_tokens=shared_tokens,
        fanout_sum=fanout_sum,
        small_segments=small,
        tiny_segments=tiny,
        attention_area=area,
        segments_below=tuple(below),
        tokens_below=tuple(tokens_below),
    )


# One integer work unit represents 1/1024 microsecond of predicted wall time.
WORK_PER_US = 1_024


@dataclass(frozen=True, slots=True)
class ScoringFacts:
    """Topology and model facts a term may scale by."""

    cp_size: int
    tp_size: int
    layers: int
    gdn_layers: int


# Interpretable cost terms: integer functions of (features, facts) in
# "feature units x WORK_PER_US" so a coefficient in microseconds per unit
# multiplies to integer work. The calibration fitter regresses measured
# within-cell timing deltas on exactly these functions, so a fitted table is
# consumed verbatim by ``score_terms``. Divisions are integer divisions of an
# already WORK_PER_US-scaled value, keeping every rank bit-identical.
#
# Topology enters through explicit interactions rather than per-topology
# tables: per-rank token work shrinks with tp x cp, while dependency levels,
# GDN state hand-offs and segment launches carry extra cost when they cross
# CP or TP ranks ((cp - 1) and (tp - 1) factors).
def _ranks(m: ScoringFacts) -> int:
    return max(1, m.tp_size) * max(1, m.cp_size)


def _token_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.packed_tokens * m.layers * WORK_PER_US // _ranks(m)


def _token_cp_exchange(f: LayoutFeatures, m: ScoringFacts) -> int:
    cp = max(1, m.cp_size)
    return f.packed_tokens * m.layers * (cp - 1) * WORK_PER_US // cp


def _token_tp_collective(f: LayoutFeatures, m: ScoringFacts) -> int:
    tp = max(1, m.tp_size)
    return f.packed_tokens * m.layers * (tp - 1) * WORK_PER_US // tp


def _segment_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.segment_count * m.layers * WORK_PER_US


def _segment_cross_rank_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.segment_count * m.layers * (_ranks(m) - 1) * WORK_PER_US


def _segment_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.segment_count * m.layers * WORK_PER_US // _ranks(m)


def _small_segment_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    # Small per rank: a segment is sharded across the CP ranks.
    return f.below(SMALL_SEGMENT_TOKENS * max(1, m.cp_size)) * m.layers * WORK_PER_US


def _tiny_segment_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.below(TINY_SEGMENT_TOKENS * max(1, m.cp_size)) * m.layers * WORK_PER_US


def _short_tokens_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    """Extra per-token cost for rows in segments that are short per rank."""

    tokens = f.tokens_in_segments_below(SMALL_SEGMENT_TOKENS * max(1, m.cp_size))
    return tokens * m.layers * WORK_PER_US // _ranks(m)


def _tiny_tokens_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    tokens = f.tokens_in_segments_below(TINY_SEGMENT_TOKENS * max(1, m.cp_size))
    return tokens * m.layers * WORK_PER_US // _ranks(m)


def _level_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return max(0, f.max_depth - 1) * m.layers * WORK_PER_US


def _level_cp_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return max(0, f.max_depth - 1) * m.layers * (max(1, m.cp_size) - 1) * WORK_PER_US


def _level_tp_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return max(0, f.max_depth - 1) * m.layers * (max(1, m.tp_size) - 1) * WORK_PER_US


def _gdn_level(f: LayoutFeatures, m: ScoringFacts) -> int:
    return max(0, f.max_depth - 1) * m.gdn_layers * WORK_PER_US


def _gdn_level_cp(f: LayoutFeatures, m: ScoringFacts) -> int:
    return (
        max(0, f.max_depth - 1) * m.gdn_layers * (max(1, m.cp_size) - 1) * WORK_PER_US
    )


def _gdn_level_tp(f: LayoutFeatures, m: ScoringFacts) -> int:
    return (
        max(0, f.max_depth - 1) * m.gdn_layers * (max(1, m.tp_size) - 1) * WORK_PER_US
    )


def _gdn_fanout(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.fanout_sum * m.gdn_layers * WORK_PER_US


def _gdn_fanout_cross_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.fanout_sum * m.gdn_layers * (_ranks(m) - 1) * WORK_PER_US


def _shared_tokens_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.shared_tokens * m.layers * WORK_PER_US // _ranks(m)


def _attention_area_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    attention_layers = max(0, m.layers - m.gdn_layers)
    return (f.attention_area * attention_layers * WORK_PER_US) // (
        1_000_000 * _ranks(m)
    )


TERM_FUNCTIONS = {
    "token_per_rank": _token_per_rank,
    "token_cp_exchange": _token_cp_exchange,
    "token_tp_collective": _token_tp_collective,
    "segment_per_layer": _segment_per_layer,
    "segment_cross_rank_per_layer": _segment_cross_rank_per_layer,
    "segment_per_rank": _segment_per_rank,
    "small_segment_per_layer": _small_segment_per_layer,
    "tiny_segment_per_layer": _tiny_segment_per_layer,
    "short_tokens_per_rank": _short_tokens_per_rank,
    "tiny_tokens_per_rank": _tiny_tokens_per_rank,
    "level_per_layer": _level_per_layer,
    "level_cp_per_layer": _level_cp_per_layer,
    "level_tp_per_layer": _level_tp_per_layer,
    "gdn_level": _gdn_level,
    "gdn_level_cp": _gdn_level_cp,
    "gdn_level_tp": _gdn_level_tp,
    "gdn_fanout": _gdn_fanout,
    "gdn_fanout_cross_rank": _gdn_fanout_cross_rank,
    "shared_tokens_per_rank": _shared_tokens_per_rank,
    "attention_area_per_rank": _attention_area_per_rank,
}


def term_values(features: LayoutFeatures, facts: ScoringFacts) -> dict[str, int]:
    """Every term's integer value (feature units x WORK_PER_US) for one layout."""

    return {name: fn(features, facts) for name, fn in TERM_FUNCTIONS.items()}


def score_terms(
    features: LayoutFeatures,
    facts: ScoringFacts,
    coefficients_us: dict[str, int],
) -> int:
    """Total predicted work under a coefficient table (microseconds per unit)."""

    return sum(
        TERM_FUNCTIONS[name](features, facts) * coefficient
        for name, coefficient in coefficients_us.items()
        if coefficient
    )


# Calibrated GDN pipeline penalties (microseconds per transformer layer): the
# first shared depth introduces segment-boundary state exchange; each depth
# beyond two adds bounded incremental barrier/bucket work.
GDN_FIRST_SHARED_PIPELINE_US_PER_LAYER = 768
GDN_EXCESS_DEPTH_PIPELINE_US_PER_LAYER = 256


def prefix_tree_layout_score(
    layout: PrefixTreeLayout,
    *,
    cp_size: int,
    layers: int,
    uses_gdn: bool,
    tp_size: int = 1,
    gdn_layers: int | None = None,
) -> tuple[int, int, int, int]:
    """Price one layout for the given topology and model facts.

    ``tp_size`` and ``gdn_layers`` are accepted so callers already pass the
    full topology; the coefficient-version-1 formula below does not yet use
    them (the recalibrated model does).
    """

    del tp_size, gdn_layers
    cp = max(1, cp_size)
    layer_count = max(1, layers)
    segment_count = len(layout.segments)
    parent_edges = len(layout.selected_decisions)
    transformer = layout.packed_tokens * WORK_PER_US
    imbalance = ((layout.packed_tokens + cp - 1) // cp) * (96 + 32 * cp)
    launch = segment_count * (96 + 32 * cp) * WORK_PER_US
    exchanges = parent_edges * (64 + 32 * cp) * WORK_PER_US
    gdn_work = (
        (
            min(1, max(0, layout.maximum_depth - 1))
            * layer_count
            * GDN_FIRST_SHARED_PIPELINE_US_PER_LAYER
            * WORK_PER_US
            + max(0, layout.maximum_depth - 2)
            * layer_count
            * GDN_EXCESS_DEPTH_PIPELINE_US_PER_LAYER
            * WORK_PER_US
        )
        if uses_gdn
        else 0
    )
    total = layer_count * transformer + (imbalance + launch + exchanges + gdn_work)
    return total, layout.packed_tokens, segment_count, layout.maximum_depth


__all__ = [
    "COEFFICIENT_VERSION",
    "GDN_EXCESS_DEPTH_PIPELINE_US_PER_LAYER",
    "GDN_FIRST_SHARED_PIPELINE_US_PER_LAYER",
    "LayoutFeatures",
    "SEGMENT_LENGTH_THRESHOLDS",
    "SMALL_SEGMENT_TOKENS",
    "ScoringFacts",
    "TERM_FUNCTIONS",
    "TINY_SEGMENT_TOKENS",
    "WORK_PER_US",
    "layout_features",
    "prefix_tree_layout_score",
    "score_terms",
    "term_values",
]
