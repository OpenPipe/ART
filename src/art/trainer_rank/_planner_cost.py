"""Fitted integer cost model for prefix-tree layout selection.

The score is a lexicographic fixed-point tuple: predicted work first, then
packed tokens, segment count, and maximum depth as deterministic tie-breaks.
All terms are integers so every rank computes bit-identical scores.

Provenance (coefficient version 2): fitted 2026-09-02 from a forced-candidate
calibration campaign on H200 bf16 (``dev/trainer_rank_landing_acceptance.py
--phase cost-calibrate`` and ``dev/trainer_rank_cost_fit.py``): every
mandatory candidate layout of each cell timed through the public API (forward
+ backward through an active LoRA slot, compile-free, max-rank), on Qwen3.5-4B
(GDN, 24 of 32 layers) and Qwen3-4B (attention) at TP1/TP2 and CP1/CP2/CP4,
over hierarchical GRPO shapes, heterogeneous controls and real Ellavox groups.
Coefficients are a non-negative least-squares fit on within-cell paired timing
deltas, refined by direct regret minimization, fitted on 45 cells and
evaluated on all 56 (the 11 odd Ellavox groups are the pre-registered holdout;
whole topologies, shapes and models were withheld in ablations). The table is
bound to its evidence by ``dev/trainer_rank_cost_calibration_certificate.json``.
The version-1 constants they replace were hand-set, not fitted.

What the data showed: the cost of a shared prefix level is a GDN effect that
grows when the level's state hand-offs cross CP or TP ranks (the attention
model pays almost nothing for it), per-rank token work shrinks with tp x cp,
and rows in segments that are short *per rank* run inefficient kernels.
``COEFFICIENT_VERSION`` is part of every layout cache key, so a new table
invalidates cached recipes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._prefix_tree_planner import PrefixTreeLayout

COEFFICIENT_VERSION = 2
# The pre-calibration score, kept as the fallback outside the calibrated domain.
COEFFICIENT_VERSION_FALLBACK = 1

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


def _attention_layers(m: ScoringFacts) -> int:
    return max(0, m.layers - m.gdn_layers)


def _attention_token_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.packed_tokens * _attention_layers(m) * WORK_PER_US // _ranks(m)


def _gdn_token_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.packed_tokens * m.gdn_layers * WORK_PER_US // _ranks(m)


def _attention_token_cp_exchange(f: LayoutFeatures, m: ScoringFacts) -> int:
    """Attention KV exchanged across CP ranks scales with rows and (cp - 1)."""

    cp = max(1, m.cp_size)
    return f.packed_tokens * _attention_layers(m) * (cp - 1) * WORK_PER_US // cp


def _attention_token_tp_collective(f: LayoutFeatures, m: ScoringFacts) -> int:
    tp = max(1, m.tp_size)
    return f.packed_tokens * _attention_layers(m) * (tp - 1) * WORK_PER_US // tp


def _segment_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.segment_count * m.layers * WORK_PER_US


def _segment_cross_rank_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.segment_count * m.layers * (_ranks(m) - 1) * WORK_PER_US


def _segment_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.segment_count * m.layers * WORK_PER_US // _ranks(m)


def _gdn_segment_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    """Per-segment work specific to GDN layers (state buckets per segment)."""

    return f.segment_count * m.gdn_layers * WORK_PER_US


def _gdn_small_segment_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return (
        f.below(SMALL_SEGMENT_TOKENS * max(1, m.cp_size)) * m.gdn_layers * WORK_PER_US
    )


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
    "attention_token_per_rank": _attention_token_per_rank,
    "gdn_token_per_rank": _gdn_token_per_rank,
    "attention_token_cp_exchange": _attention_token_cp_exchange,
    "attention_token_tp_collective": _attention_token_tp_collective,
    "segment_per_layer": _segment_per_layer,
    "segment_cross_rank_per_layer": _segment_cross_rank_per_layer,
    "segment_per_rank": _segment_per_rank,
    "gdn_segment_per_layer": _gdn_segment_per_layer,
    "gdn_small_segment_per_layer": _gdn_small_segment_per_layer,
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
    coefficients_milli_us: dict[str, int],
) -> int:
    """Total predicted work under a coefficient table (milli-microseconds per
    feature unit), in 1/(WORK_PER_US x 1000) microsecond units."""

    return sum(
        TERM_FUNCTIONS[name](features, facts) * coefficient
        for name, coefficient in coefficients_milli_us.items()
        if coefficient
    )


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """The capability domain the fitted table was measured on — exactly.

    Capability-based, never model-name-based, and narrowed to what the
    certificate actually contains: H200-class Hopper devices (compute
    capability 9.0 *and* an HBM3e-sized memory system, which separates H200
    from the 80 GB H100 that shares the capability), bf16 parameters, hidden
    size 2,560 (Qwen3.5-4B GDN and Qwen3-4B attention), dense (non-MoE)
    models, at TP1/TP2 and CP1/CP2/CP4. Hidden size is not a score feature,
    so every admitted width would get identical token-versus-boundary
    economics; only measured widths are admitted. Outside this domain the
    selector keeps the version-1 score. Extending the domain means running
    the calibration cells on the new device or width and regenerating the
    certificate, which the certificate test binds to these fields.
    """

    device_capabilities: tuple[tuple[int, int], ...] = ((9, 0),)
    # H200 reports ~143 GB; H100 (80 GB) is excluded by this bound.
    min_device_memory_bytes: int = 120 * 1024**3
    param_dtypes: tuple[str, ...] = ("torch.bfloat16",)
    hidden_sizes: tuple[int, ...] = (2_560,)
    allow_moe: bool = False
    # Documentation of the measured devices; matching never reads names.
    measured_device_names: tuple[str, ...] = ("NVIDIA H200",)

    def matches(
        self,
        *,
        device_capability: tuple[int, int] | None,
        device_memory_bytes: int | None,
        param_dtype: str,
        hidden_size: int,
        is_moe: bool,
    ) -> bool:
        if device_capability is not None and (
            tuple(device_capability) not in self.device_capabilities
        ):
            return False
        if device_memory_bytes is not None and (
            device_memory_bytes < self.min_device_memory_bytes
        ):
            return False
        if param_dtype not in self.param_dtypes:
            return False
        if hidden_size not in self.hidden_sizes:
            return False
        return self.allow_moe or not is_moe


CALIBRATION_PROFILE = CalibrationProfile()


def coefficient_version_for(
    *,
    device_capability: tuple[int, int] | None,
    param_dtype: str,
    hidden_size: int,
    is_moe: bool,
    device_memory_bytes: int | None = None,
) -> int:
    """Pick the score version for a runtime: the fitted table inside its
    calibrated capability profile, the fallback outside it.

    ``device_capability`` is ``None`` when the model is not on a CUDA device
    (CPU-only planning, as in the unit tests); such runtimes use the fitted
    table, since the profile describes GPU execution. On CUDA both the
    capability and the device memory size are checked.
    """

    if device_capability is None:
        return COEFFICIENT_VERSION
    inside = CALIBRATION_PROFILE.matches(
        device_capability=device_capability,
        device_memory_bytes=device_memory_bytes,
        param_dtype=param_dtype,
        hidden_size=hidden_size,
        is_moe=is_moe,
    )
    return COEFFICIENT_VERSION if inside else COEFFICIENT_VERSION_FALLBACK


# --- Version 1 (fallback): the landing's hand-shaped score ---------------------
# Provenance: the research implementation's production layout score (frozen
# 2026-08-31); constants were hand-set. Kept verbatim so runtimes outside the
# calibrated profile behave exactly as before the recalibration.
_V1_GDN_FIRST_SHARED_PIPELINE_US_PER_LAYER = 768
_V1_GDN_EXCESS_DEPTH_PIPELINE_US_PER_LAYER = 256


def _prefix_tree_layout_score_v1(
    layout: PrefixTreeLayout,
    *,
    cp_size: int,
    layers: int,
    uses_gdn: bool,
) -> tuple[int, int, int, int]:
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
            * _V1_GDN_FIRST_SHARED_PIPELINE_US_PER_LAYER
            * WORK_PER_US
            + max(0, layout.maximum_depth - 2)
            * layer_count
            * _V1_GDN_EXCESS_DEPTH_PIPELINE_US_PER_LAYER
            * WORK_PER_US
        )
        if uses_gdn
        else 0
    )
    total = layer_count * transformer + (imbalance + launch + exchanges + gdn_work)
    return total, layout.packed_tokens, segment_count, layout.maximum_depth


# Fitted coefficients in integer milli-microseconds per feature unit (see the
# module docstring for provenance; regenerate with
# ``dev/trainer_rank_cost_fit.py --integerize``). Zero means the term carried no
# weight on the calibration cells and is kept only so the table matches
# ``TERM_FUNCTIONS`` one to one.
COEFFICIENT_SCALE_PER_US = 1_000
COEFFICIENTS_MILLI_US: dict[str, int] = {
    "token_per_rank": 2015,
    "token_cp_exchange": 1,
    "token_tp_collective": 217,
    "attention_token_per_rank": 0,
    "gdn_token_per_rank": 1006,
    "attention_token_cp_exchange": 9,
    "attention_token_tp_collective": 128,
    "segment_per_layer": 0,
    "segment_per_rank": 0,
    "segment_cross_rank_per_layer": 0,
    "gdn_segment_per_layer": 0,
    "gdn_small_segment_per_layer": 0,
    "small_segment_per_layer": 0,
    "tiny_segment_per_layer": 224651,
    "short_tokens_per_rank": 0,
    "tiny_tokens_per_rank": 0,
    "level_per_layer": 0,
    "level_cp_per_layer": 167499,
    "level_tp_per_layer": 312400,
    "gdn_level": 3732660,
    "gdn_level_cp": 0,
    "gdn_level_tp": 1297375,
    "gdn_fanout": 0,
    "gdn_fanout_cross_rank": 0,
    "shared_tokens_per_rank": 0,
    "attention_area_per_rank": 33536,
}
assert set(COEFFICIENTS_MILLI_US) == set(TERM_FUNCTIONS)


def predicted_work(
    features: LayoutFeatures,
    facts: ScoringFacts,
    coefficients_milli_us: dict[str, int] = COEFFICIENTS_MILLI_US,
) -> int:
    """Total predicted work in 1/(WORK_PER_US x COEFFICIENT_SCALE_PER_US) us."""

    return score_terms(features, facts, coefficients_milli_us)


def predicted_us(features: LayoutFeatures, facts: ScoringFacts) -> float:
    """Predicted layout-dependent work in microseconds (for logging only)."""

    return predicted_work(features, facts) / (WORK_PER_US * COEFFICIENT_SCALE_PER_US)


def prefix_tree_layout_score(
    layout: PrefixTreeLayout,
    *,
    cp_size: int,
    layers: int,
    uses_gdn: bool,
    tp_size: int = 1,
    gdn_layers: int | None = None,
    coefficient_version: int = COEFFICIENT_VERSION,
) -> tuple[int, int, int, int]:
    """Price one layout for the given topology and model facts.

    ``gdn_layers`` defaults to ``layers`` when the model uses GDN and to zero
    otherwise; ``uses_gdn`` only matters for that default.
    ``coefficient_version`` selects the fitted table (2) or the fallback (1);
    see ``coefficient_version_for``.
    """

    if coefficient_version == COEFFICIENT_VERSION_FALLBACK:
        return _prefix_tree_layout_score_v1(
            layout, cp_size=cp_size, layers=layers, uses_gdn=uses_gdn
        )
    if coefficient_version != COEFFICIENT_VERSION:
        raise ValueError(f"unknown coefficient version {coefficient_version}")
    features = layout_features(layout)
    facts = ScoringFacts(
        cp_size=max(1, cp_size),
        tp_size=max(1, tp_size),
        layers=max(1, layers),
        gdn_layers=(
            max(0, gdn_layers)
            if gdn_layers is not None
            else (max(1, layers) if uses_gdn else 0)
        ),
    )
    return (
        predicted_work(features, facts),
        layout.packed_tokens,
        features.segment_count,
        layout.maximum_depth,
    )


__all__ = [
    "CALIBRATION_PROFILE",
    "COEFFICIENTS_MILLI_US",
    "COEFFICIENT_SCALE_PER_US",
    "COEFFICIENT_VERSION",
    "COEFFICIENT_VERSION_FALLBACK",
    "CalibrationProfile",
    "LayoutFeatures",
    "SEGMENT_LENGTH_THRESHOLDS",
    "SMALL_SEGMENT_TOKENS",
    "ScoringFacts",
    "TERM_FUNCTIONS",
    "TINY_SEGMENT_TOKENS",
    "WORK_PER_US",
    "coefficient_version_for",
    "layout_features",
    "predicted_us",
    "predicted_work",
    "prefix_tree_layout_score",
    "score_terms",
    "term_values",
]
