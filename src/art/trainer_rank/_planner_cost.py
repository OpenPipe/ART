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
bound to its evidence by ``dev/trainer_rank_cost_calibration_certificate_<table>.json``.
The version-1 constants they replace were hand-set, not fitted.

What the data showed: the cost of a shared prefix level is a GDN effect that
grows when the level's state hand-offs cross CP or TP ranks (the attention
model pays almost nothing for it), per-rank token work shrinks with tp x cp,
GDN layers cost more per token than attention layers, and segments that are
tiny *per rank* run inefficient kernels. Ten terms carried weight; candidates
that fitted to zero were dropped.
``COEFFICIENT_VERSION`` and the selected table's identity are part of every
layout cache key, so a new table invalidates cached recipes.

Calibrated domain: a table applies only to the execution classes it was
measured on — device class, parameter dtype, model geometry (widths, head
geometry, GDN state shape, expert geometry) and parallel shape (TP, CP, EP,
ETP) — see ``CalibratedTable`` and ``select_scoring``. Outside every table the
version-1 score applies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._prefix_tree_planner import PrefixTreeLayout

COEFFICIENT_VERSION = 2
# The pre-calibration score, kept as the fallback outside the calibrated domain.
COEFFICIENT_VERSION_FALLBACK = 1

# Segment-length histogram for kernel-utilization effects: a segment shorter
# than a threshold runs small-M kernels on every rank it is sharded over.
# Context parallelism shards a segment across ranks, so the scorer reads the
# bucket for ``threshold x cp`` (per-rank length); the histogram covers 64..8192.
TINY_SEGMENT_TOKENS = 128
SEGMENT_LENGTH_THRESHOLDS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True, slots=True)
class LayoutFeatures:
    """Integer, O(segments) features of one layout that differ between layouts.

    Everything a call shares across its candidate layouts (logical tokens, the
    output head, model size) cancels in ranking, so only layout-dependent
    quantities are here: exactly what the fitted terms read, plus the segment
    count used as a tie-break. Each is derivable from the planned segments
    alone, so the calibration harness logs exactly what a scorer can price.
    """

    packed_tokens: int
    segment_count: int
    # Dependency levels including the root and tail levels: no sharing = 1.
    max_depth: int
    # Segments shorter than each SEGMENT_LENGTH_THRESHOLDS entry (cumulative).
    segments_below: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def below(self, tokens: int) -> int:
        """Segments shorter than ``tokens`` (nearest histogram bucket at or above)."""

        for threshold, count in zip(
            SEGMENT_LENGTH_THRESHOLDS, self.segments_below, strict=False
        ):
            if threshold >= tokens:
                return count
        return self.segments_below[-1] if self.segments_below else 0


def layout_features(layout: PrefixTreeLayout) -> LayoutFeatures:
    segment_count = 0
    below = [0] * len(SEGMENT_LENGTH_THRESHOLDS)
    for segment in layout.segments:
        length = segment.end - segment.start
        segment_count += 1
        for index, threshold in enumerate(SEGMENT_LENGTH_THRESHOLDS):
            if length < threshold:
                below[index] += 1
    return LayoutFeatures(
        packed_tokens=layout.packed_tokens,
        segment_count=segment_count,
        max_depth=layout.maximum_depth,
        segments_below=tuple(below),
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
# "feature units x WORK_PER_US" so a coefficient in milli-microseconds per
# unit multiplies to integer work. The calibration fitter regresses measured
# within-cell timing deltas on exactly these functions, so a fitted table is
# consumed verbatim by ``score_terms``. Divisions are integer divisions of an
# already WORK_PER_US-scaled value, keeping every rank bit-identical.
#
# Topology enters through explicit interactions rather than per-topology
# tables: per-rank token work shrinks with tp x cp, while dependency levels
# and GDN state hand-offs carry extra cost when they cross CP or TP ranks
# ((cp - 1) and (tp - 1) factors). These are the terms that carried weight in
# the calibration; candidate terms that fitted to zero (segment launches,
# fan-out, shared tokens, attention area, small-M token surcharges) were
# dropped from the module and can be reintroduced by a future campaign.
def _ranks(m: ScoringFacts) -> int:
    return max(1, m.tp_size) * max(1, m.cp_size)


def _levels(f: LayoutFeatures) -> int:
    return max(0, f.max_depth - 1)


def _token_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    return f.packed_tokens * m.layers * WORK_PER_US // _ranks(m)


def _token_cp_exchange(f: LayoutFeatures, m: ScoringFacts) -> int:
    cp = max(1, m.cp_size)
    return f.packed_tokens * m.layers * (cp - 1) * WORK_PER_US // cp


def _token_tp_collective(f: LayoutFeatures, m: ScoringFacts) -> int:
    tp = max(1, m.tp_size)
    return f.packed_tokens * m.layers * (tp - 1) * WORK_PER_US // tp


def _gdn_token_per_rank(f: LayoutFeatures, m: ScoringFacts) -> int:
    """GDN layers cost more per token than attention layers."""

    return f.packed_tokens * m.gdn_layers * WORK_PER_US // _ranks(m)


def _attention_token_cp_exchange(f: LayoutFeatures, m: ScoringFacts) -> int:
    """Attention KV exchanged across CP ranks scales with rows and (cp - 1)."""

    cp = max(1, m.cp_size)
    attention_layers = max(0, m.layers - m.gdn_layers)
    return f.packed_tokens * attention_layers * (cp - 1) * WORK_PER_US // cp


def _tiny_segment_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    """Segments that are tiny per rank (threshold x cp) run inefficient kernels."""

    return f.below(TINY_SEGMENT_TOKENS * max(1, m.cp_size)) * m.layers * WORK_PER_US


def _level_cp_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return _levels(f) * m.layers * (max(1, m.cp_size) - 1) * WORK_PER_US


def _level_tp_per_layer(f: LayoutFeatures, m: ScoringFacts) -> int:
    return _levels(f) * m.layers * (max(1, m.tp_size) - 1) * WORK_PER_US


def _gdn_level(f: LayoutFeatures, m: ScoringFacts) -> int:
    """A shared prefix level costs GDN state hand-offs on every GDN layer."""

    return _levels(f) * m.gdn_layers * WORK_PER_US


def _gdn_level_tp(f: LayoutFeatures, m: ScoringFacts) -> int:
    return _levels(f) * m.gdn_layers * (max(1, m.tp_size) - 1) * WORK_PER_US


TERM_FUNCTIONS = {
    "token_per_rank": _token_per_rank,
    "token_cp_exchange": _token_cp_exchange,
    "token_tp_collective": _token_tp_collective,
    "gdn_token_per_rank": _gdn_token_per_rank,
    "attention_token_cp_exchange": _attention_token_cp_exchange,
    "tiny_segment_per_layer": _tiny_segment_per_layer,
    "level_cp_per_layer": _level_cp_per_layer,
    "level_tp_per_layer": _level_tp_per_layer,
    "gdn_level": _gdn_level,
    "gdn_level_tp": _gdn_level_tp,
}


def score_terms(
    features: LayoutFeatures,
    facts: ScoringFacts,
    coefficients_milli_us: Mapping[str, int],
) -> int:
    """Total predicted work under a coefficient table (milli-microseconds per
    feature unit), in 1/(WORK_PER_US x 1000) microsecond units."""

    return sum(
        TERM_FUNCTIONS[name](features, facts) * coefficient
        for name, coefficient in coefficients_milli_us.items()
        if coefficient
    )


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """Per-layer model geometry that sets kernel economics.

    Read from the Megatron ``TransformerConfig`` (never from a model name);
    layer *counts* are scoring facts, not identity. GDN and MoE fields are
    zero when the model has no such layers, so an attention-only model and a
    hybrid one never share a geometry.
    """

    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    gdn_value_heads: int = 0
    gdn_key_heads: int = 0
    gdn_key_head_dim: int = 0
    gdn_value_head_dim: int = 0
    gdn_conv_kernel: int = 0
    moe_experts: int = 0
    moe_topk: int = 0
    moe_ffn_hidden_size: int = 0
    moe_shared_expert_ffn: int = 0

    @classmethod
    def from_config(
        cls, config: Any, *, has_gdn: bool, has_moe: bool
    ) -> "ModelGeometry":
        def field(name: str, default: int = 0) -> int:
            value = getattr(config, name, None)
            return int(value) if value is not None else default

        hidden = field("hidden_size")
        heads = field("num_attention_heads")
        return cls(
            hidden_size=hidden,
            ffn_hidden_size=field("ffn_hidden_size"),
            num_attention_heads=heads,
            num_query_groups=field("num_query_groups", heads),
            kv_channels=field("kv_channels", hidden // heads if heads else 0),
            gdn_value_heads=field("linear_num_value_heads") if has_gdn else 0,
            gdn_key_heads=field("linear_num_key_heads") if has_gdn else 0,
            gdn_key_head_dim=field("linear_key_head_dim") if has_gdn else 0,
            gdn_value_head_dim=field("linear_value_head_dim") if has_gdn else 0,
            gdn_conv_kernel=field("linear_conv_kernel_dim") if has_gdn else 0,
            moe_experts=field("num_moe_experts") if has_moe else 0,
            moe_topk=field("moe_router_topk") if has_moe else 0,
            moe_ffn_hidden_size=field("moe_ffn_hidden_size") if has_moe else 0,
            moe_shared_expert_ffn=(
                field("moe_shared_expert_intermediate_size") if has_moe else 0
            ),
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeviceClass:
    """Accelerator class: compute capability plus a memory-system bucket.

    The 80 GB H100 shares capability 9.0 with the 141 GB H200 but not its
    memory system, so the bucket separates them. ``memory_class`` is ``None``
    when the runtime cannot read device memory (older drivers); such a device
    is admitted on capability alone.
    """

    capability: tuple[int, int]
    memory_class: str | None

    @staticmethod
    def memory_class_for(memory_bytes: int | None) -> str | None:
        if memory_bytes is None:
            return None
        if memory_bytes >= 120 * 1024**3:
            return "hbm-141g"
        if memory_bytes >= 70 * 1024**3:
            return "hbm-80g"
        return "hbm-small"

    @classmethod
    def for_device(
        cls, capability: tuple[int, int], memory_bytes: int | None
    ) -> "DeviceClass":
        return cls(
            (int(capability[0]), int(capability[1])),
            cls.memory_class_for(memory_bytes),
        )

    def admits(self, device: "DeviceClass") -> bool:
        return device.capability == self.capability and (
            device.memory_class is None or device.memory_class == self.memory_class
        )


@dataclass(frozen=True, slots=True)
class ParallelShape:
    """The parallel shape a table was measured at: TP x CP x EP x ETP."""

    tp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 1


@dataclass(frozen=True)
class ReRanker:
    """Second-stage score for a table's re-ranked parallel shapes.

    On these shapes the ten-term score cannot see the structure of the
    context-parallel plan a layout produces, which dominates real-data cells
    at CP4. It still shortlists well: the ``shortlist_size`` cheapest layouts
    of the search under ``shortlist_coefficients_milli_us`` (plus the
    ``incumbent`` anchor) are then priced per layer by the plan they produce,
    remote wave count and largest per-rank token load, and the lowest
    second-stage score is selected. Fitted and certified per class and shape
    like the tables (shortlist recall, ranking gates, non-regression against
    the cheap score and version 1, planning cost).
    """

    shortlist_size: int
    incumbent: str
    shortlist_coefficients_milli_us: Mapping[str, int]
    wave_per_layer_milli_us: int
    max_rank_token_per_layer_milli_us: int

    def score(self, *, layers: int, wave_count: int, max_rank_tokens: int) -> int:
        return max(1, layers) * (
            wave_count * self.wave_per_layer_milli_us
            + max_rank_tokens * self.max_rank_token_per_layer_milli_us
        )


@dataclass(frozen=True)
class CalibratedTable:
    """One fitted coefficient table and the execution classes it admits.

    Admission is exact: the runtime's device class, parameter dtype, model
    geometry and parallel shape must each appear in the table's measured
    sets (the certificate test binds these sets to the recorded cells). No
    configuration is admitted because a width falls between measured widths.
    ``reranked_shapes`` are admitted through the two-stage selection of the
    table's ``reranker`` instead of the ten-term score alone.
    """

    table_id: str
    coefficients_milli_us: Mapping[str, int]
    device_classes: tuple[DeviceClass, ...]
    param_dtypes: tuple[str, ...]
    geometries: tuple[ModelGeometry, ...]
    shapes: tuple[ParallelShape, ...]
    # Measured (geometry, shape) pairs the table does not admit: their cells
    # are in the certificate and the fit, but the score is known to misrank
    # that pair (a documented blind spot), so the fallback applies there.
    withheld: tuple[tuple[ModelGeometry, ParallelShape], ...] = ()
    reranked_shapes: tuple[ParallelShape, ...] = ()
    reranker: ReRanker | None = None

    def __post_init__(self) -> None:
        if set(self.reranked_shapes) & set(self.shapes):
            raise ValueError("a shape is either scored directly or re-ranked")
        if bool(self.reranked_shapes) != (self.reranker is not None):
            raise ValueError("re-ranked shapes and a re-ranker come together")

    def admits_class(
        self,
        *,
        device: DeviceClass,
        param_dtype: str,
        geometry: ModelGeometry,
    ) -> bool:
        return (
            any(known.admits(device) for known in self.device_classes)
            and param_dtype in self.param_dtypes
            and geometry in self.geometries
        )

    def admits(
        self,
        *,
        device: DeviceClass,
        param_dtype: str,
        geometry: ModelGeometry,
        shape: ParallelShape,
    ) -> bool:
        return (
            self.admits_class(device=device, param_dtype=param_dtype, geometry=geometry)
            and shape in self.shapes
            and (geometry, shape) not in self.withheld
        )

    def reranks(
        self,
        *,
        device: DeviceClass,
        param_dtype: str,
        geometry: ModelGeometry,
        shape: ParallelShape,
    ) -> bool:
        return (
            self.reranker is not None
            and self.admits_class(
                device=device, param_dtype=param_dtype, geometry=geometry
            )
            and shape in self.reranked_shapes
        )


@dataclass(frozen=True, slots=True)
class ScoringSelection:
    """The score a runtime uses: version 2 with one table (directly, or as
    the shortlist score of a two-stage re-ranker), or the fallback."""

    version: int
    table_id: str | None
    coefficients: Mapping[str, int] | None
    reranker: ReRanker | None = None


def select_scoring(
    *,
    device_capability: tuple[int, int] | None,
    device_memory_bytes: int | None,
    param_dtype: str,
    geometry: ModelGeometry,
    shape: ParallelShape,
) -> ScoringSelection:
    """Pick the score for a runtime: the table whose measured execution
    classes admit it, else the version-1 fallback.

    ``device_capability`` is ``None`` when the model is not on a CUDA device
    (CPU-only planning, as in the unit tests); such runtimes use the default
    table, since the calibrated domain describes GPU execution.
    """

    if device_capability is None:
        return ScoringSelection(
            COEFFICIENT_VERSION,
            DEFAULT_TABLE.table_id,
            DEFAULT_TABLE.coefficients_milli_us,
        )
    device = DeviceClass.for_device(device_capability, device_memory_bytes)
    for table in CALIBRATED_TABLES:
        if table.admits(
            device=device, param_dtype=param_dtype, geometry=geometry, shape=shape
        ):
            return ScoringSelection(
                COEFFICIENT_VERSION, table.table_id, table.coefficients_milli_us
            )
        if table.reranks(
            device=device, param_dtype=param_dtype, geometry=geometry, shape=shape
        ):
            assert table.reranker is not None
            return ScoringSelection(
                COEFFICIENT_VERSION,
                table.table_id,
                table.reranker.shortlist_coefficients_milli_us,
                table.reranker,
            )
    return ScoringSelection(COEFFICIENT_VERSION_FALLBACK, None, None)


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
# ``dev/trainer_rank_cost_fit.py --integerize`` and the checked-in certificate).
COEFFICIENT_SCALE_PER_US = 1_000
# Dense hidden-2,560 class (Qwen3.5-4B GDN and Qwen3-4B attention on H200 bf16).
COEFFICIENTS_MILLI_US: dict[str, int] = {
    "token_per_rank": 2002,
    "token_cp_exchange": 39,
    "token_tp_collective": 221,
    "gdn_token_per_rank": 594,
    "attention_token_cp_exchange": 27,
    "tiny_segment_per_layer": 219288,
    "level_cp_per_layer": 187255,
    "level_tp_per_layer": 684798,
    "gdn_level": 3336555,
    "gdn_level_tp": 1194872,
}

H200_CLASS = DeviceClass(capability=(9, 0), memory_class="hbm-141g")
# Megatron TransformerConfig geometry of the two measured models. Qwen3.5-4B:
# 16 attention heads in 4 query groups of 256 channels, GDN with 32 value /
# 16 key heads of 128 dims and a 4-tap conv; Qwen3-4B: 32 heads in 8 groups
# of 128 channels. Layer counts are scoring facts, not identity.
QWEN35_4B_GEOMETRY = ModelGeometry(
    hidden_size=2_560,
    ffn_hidden_size=9_216,
    num_attention_heads=16,
    num_query_groups=4,
    kv_channels=256,
    gdn_value_heads=32,
    gdn_key_heads=16,
    gdn_key_head_dim=128,
    gdn_value_head_dim=128,
    gdn_conv_kernel=4,
)
QWEN3_4B_GEOMETRY = ModelGeometry(
    hidden_size=2_560,
    ffn_hidden_size=9_728,
    num_attention_heads=32,
    num_query_groups=8,
    kv_channels=128,
)
DENSE_H2560_TABLE = CalibratedTable(
    table_id="dense-h2560-h200-bf16",
    coefficients_milli_us=COEFFICIENTS_MILLI_US,
    device_classes=(H200_CLASS,),
    param_dtypes=("torch.bfloat16",),
    geometries=(QWEN35_4B_GEOMETRY, QWEN3_4B_GEOMETRY),
    shapes=(
        ParallelShape(tp=1, cp=1),
        ParallelShape(tp=1, cp=2),
        ParallelShape(tp=1, cp=4),
        ParallelShape(tp=2, cp=1),
    ),
    # Qwen3-4B at TP1 x CP4 is measured but withheld: on real-data groups the
    # ten-term score cannot see the context-parallel exchange schedule and its
    # regret reached 35% (design brief, known limitation); version 1 is the
    # less harmful fallback there until the CP-aware re-ranker is certified.
    withheld=((QWEN3_4B_GEOMETRY, ParallelShape(tp=1, cp=4)),),
)
# Qwen3.5-35B-A3B class (GDN + MoE, hidden 2,048): 16 attention heads in 2 query groups of 256 channels, GDN 32 value / 16 key heads of 128 dims, 256 experts (top-8, expert FFN 512, shared expert 512), on H200 bf16; measured at TP1 x CP1/2/4 and TP2 with EP1, EP2 at CP2/CP4/TP2 and EP4 at CP4 (2026-09-04).
GDN_MOE_H2048_GEOMETRY = ModelGeometry(
    ffn_hidden_size=8_192,
    gdn_conv_kernel=4,
    gdn_key_head_dim=128,
    gdn_key_heads=16,
    gdn_value_head_dim=128,
    gdn_value_heads=32,
    hidden_size=2_048,
    kv_channels=256,
    moe_experts=256,
    moe_ffn_hidden_size=512,
    moe_shared_expert_ffn=512,
    moe_topk=8,
    num_attention_heads=16,
    num_query_groups=2,
)
GDN_MOE_H2048_TABLE = CalibratedTable(
    table_id="gdn-moe-h2048-h200-bf16",
    coefficients_milli_us={
        "attention_token_cp_exchange": 37,
        "gdn_level": 8_982_505,
        "gdn_level_tp": 0,
        "gdn_token_per_rank": 368,
        "level_cp_per_layer": 50_356,
        "level_tp_per_layer": 5_701_857,
        "tiny_segment_per_layer": 442_444,
        "token_cp_exchange": 37,
        "token_per_rank": 3_145,
        "token_tp_collective": 734,
    },
    device_classes=(H200_CLASS,),
    param_dtypes=("torch.bfloat16",),
    geometries=(GDN_MOE_H2048_GEOMETRY,),
    shapes=(
        ParallelShape(tp=1, cp=1),
        ParallelShape(tp=1, cp=2),
        ParallelShape(tp=1, cp=2, ep=2),
        ParallelShape(tp=1, cp=4),
        ParallelShape(tp=1, cp=4, ep=2),
        ParallelShape(tp=1, cp=4, ep=4),
        ParallelShape(tp=2, cp=1),
        ParallelShape(tp=2, cp=1, ep=2),
    ),
)

# Dense attention classes measured on the shape lattice (2026-09-04):
# Qwen3-1.7B (hidden 2,048), Qwen3-8B (4,096) and Qwen3-14B (5,120), each
# with 8 query groups of 128 channels, on H200 bf16. TP1 x CP4 is admitted
# through the two-stage re-ranker: the ten-term score cannot see the
# context-parallel plan a layout produces, which decides the recurring
# real-data groups there (design brief), so it only shortlists and the plan's
# remote wave count and largest per-rank token load select. TP2 x CP2 fails
# its gates for the two smaller classes and keeps the version-1 score.
_ATTENTION_DTYPES = ("torch.bfloat16",)
_ATTENTION_SHAPES = (
    ParallelShape(tp=1, cp=1),
    ParallelShape(tp=1, cp=2),
    ParallelShape(tp=2, cp=1),
)
QWEN3_1_7B_GEOMETRY = ModelGeometry(
    hidden_size=2_048,
    ffn_hidden_size=6_144,
    num_attention_heads=16,
    num_query_groups=8,
    kv_channels=128,
)
DENSE_ATTN_H2048_TABLE = CalibratedTable(
    table_id="dense-attn-h2048-h200-bf16",
    coefficients_milli_us={
        "attention_token_cp_exchange": 0,
        "gdn_level": 0,
        "gdn_level_tp": 0,
        "gdn_token_per_rank": 0,
        "level_cp_per_layer": 0,
        "level_tp_per_layer": 0,
        "tiny_segment_per_layer": 0,
        "token_cp_exchange": 46,
        "token_per_rank": 259,
        "token_tp_collective": 573,
    },
    device_classes=(H200_CLASS,),
    param_dtypes=_ATTENTION_DTYPES,
    geometries=(QWEN3_1_7B_GEOMETRY,),
    shapes=_ATTENTION_SHAPES,
    reranked_shapes=(ParallelShape(tp=1, cp=4),),
    reranker=ReRanker(
        shortlist_size=3,
        incumbent="depth_one",
        shortlist_coefficients_milli_us={
            "attention_token_cp_exchange": 3,
            "gdn_level": 0,
            "gdn_level_tp": 0,
            "gdn_token_per_rank": 0,
            "level_cp_per_layer": 69_746,
            "level_tp_per_layer": 0,
            "tiny_segment_per_layer": 0,
            "token_cp_exchange": 57,
            "token_per_rank": 1_656,
            "token_tp_collective": 143,
        },
        wave_per_layer_milli_us=1_144_207,
        max_rank_token_per_layer_milli_us=678,
    ),
)
QWEN3_8B_GEOMETRY = ModelGeometry(
    hidden_size=4_096,
    ffn_hidden_size=12_288,
    num_attention_heads=32,
    num_query_groups=8,
    kv_channels=128,
)
DENSE_ATTN_H4096_TABLE = CalibratedTable(
    table_id="dense-attn-h4096-h200-bf16",
    coefficients_milli_us={
        "attention_token_cp_exchange": 0,
        "gdn_level": 0,
        "gdn_level_tp": 0,
        "gdn_token_per_rank": 0,
        "level_cp_per_layer": 0,
        "level_tp_per_layer": 0,
        "tiny_segment_per_layer": 0,
        "token_cp_exchange": 262,
        "token_per_rank": 2_858,
        "token_tp_collective": 640,
    },
    device_classes=(H200_CLASS,),
    param_dtypes=_ATTENTION_DTYPES,
    geometries=(QWEN3_8B_GEOMETRY,),
    shapes=_ATTENTION_SHAPES,
    reranked_shapes=(ParallelShape(tp=1, cp=4),),
    reranker=ReRanker(
        shortlist_size=3,
        incumbent="depth_one",
        shortlist_coefficients_milli_us={
            "attention_token_cp_exchange": 57,
            "gdn_level": 0,
            "gdn_level_tp": 0,
            "gdn_token_per_rank": 0,
            "level_cp_per_layer": 17_614,
            "level_tp_per_layer": 0,
            "tiny_segment_per_layer": 1_508,
            "token_cp_exchange": 0,
            "token_per_rank": 2_925,
            "token_tp_collective": 573,
        },
        wave_per_layer_milli_us=249_653,
        max_rank_token_per_layer_milli_us=2_820,
    ),
)
QWEN3_14B_GEOMETRY = ModelGeometry(
    hidden_size=5_120,
    ffn_hidden_size=17_408,
    num_attention_heads=40,
    num_query_groups=8,
    kv_channels=128,
)
DENSE_ATTN_H5120_TABLE = CalibratedTable(
    table_id="dense-attn-h5120-h200-bf16",
    coefficients_milli_us={
        "attention_token_cp_exchange": 0,
        "gdn_level": 0,
        "gdn_level_tp": 0,
        "gdn_token_per_rank": 0,
        "level_cp_per_layer": 0,
        "level_tp_per_layer": 0,
        "tiny_segment_per_layer": 0,
        "token_cp_exchange": 0,
        "token_per_rank": 4_607,
        "token_tp_collective": 487,
    },
    device_classes=(H200_CLASS,),
    param_dtypes=_ATTENTION_DTYPES,
    geometries=(QWEN3_14B_GEOMETRY,),
    shapes=_ATTENTION_SHAPES + (ParallelShape(tp=2, cp=2),),
    reranked_shapes=(ParallelShape(tp=1, cp=4),),
    reranker=ReRanker(
        shortlist_size=3,
        incumbent="depth_one",
        shortlist_coefficients_milli_us={
            "attention_token_cp_exchange": 0,
            "gdn_level": 0,
            "gdn_level_tp": 0,
            "gdn_token_per_rank": 0,
            "level_cp_per_layer": 0,
            "level_tp_per_layer": 0,
            "tiny_segment_per_layer": 0,
            "token_cp_exchange": 0,
            "token_per_rank": 4_592,
            "token_tp_collective": 496,
        },
        wave_per_layer_milli_us=654_967,
        max_rank_token_per_layer_milli_us=4_642,
    ),
)

# Qwen3-30B-A3B class (attention + MoE, hidden 2,048): 32 attention heads in
# 4 query groups of 128 channels, 128 experts (top-8, expert FFN 768), on H200
# bf16 (2026-09-04/05). Admitted directly at TP1 x CP1, TP1 x CP2 (EP1, EP2)
# and TP2 x CP1 (EP1, EP2); TP1 x CP4 at every expert parallelism through the
# two-stage re-ranker (this attention class has the CP4 blind spot). The CP2
# and CP4 EP1/EP4 shapes were re-measured with sixteen rounds after their
# eight-round timings carried sporadic multi-second stalls (design brief).
ATTN_MOE_H2048_GEOMETRY = ModelGeometry(
    hidden_size=2_048,
    ffn_hidden_size=6_144,
    num_attention_heads=32,
    num_query_groups=4,
    kv_channels=128,
    moe_experts=128,
    moe_topk=8,
    moe_ffn_hidden_size=768,
    moe_shared_expert_ffn=0,
)
ATTN_MOE_H2048_TABLE = CalibratedTable(
    table_id="attn-moe-h2048-h200-bf16",
    coefficients_milli_us={
        "attention_token_cp_exchange": 0,
        "gdn_level": 0,
        "gdn_level_tp": 0,
        "gdn_token_per_rank": 0,
        "level_cp_per_layer": 127_543,
        "level_tp_per_layer": 0,
        "tiny_segment_per_layer": 0,
        "token_cp_exchange": 237,
        "token_per_rank": 6_648,
        "token_tp_collective": 105,
    },
    device_classes=(H200_CLASS,),
    param_dtypes=("torch.bfloat16",),
    geometries=(ATTN_MOE_H2048_GEOMETRY,),
    shapes=(
        ParallelShape(tp=1, cp=1),
        ParallelShape(tp=1, cp=2),
        ParallelShape(tp=1, cp=2, ep=2),
        ParallelShape(tp=2, cp=1),
        ParallelShape(tp=2, cp=1, ep=2),
    ),
    reranked_shapes=(
        ParallelShape(tp=1, cp=4),
        ParallelShape(tp=1, cp=4, ep=2),
        ParallelShape(tp=1, cp=4, ep=4),
    ),
    reranker=ReRanker(
        shortlist_size=3,
        incumbent="depth_one",
        shortlist_coefficients_milli_us={
            "attention_token_cp_exchange": 8,
            "gdn_level": 0,
            "gdn_level_tp": 0,
            "gdn_token_per_rank": 0,
            "level_cp_per_layer": 163_376,
            "level_tp_per_layer": 140_748,
            "tiny_segment_per_layer": 169_817,
            "token_cp_exchange": 98,
            "token_per_rank": 6_705,
            "token_tp_collective": 78,
        },
        wave_per_layer_milli_us=1_388_921,
        max_rank_token_per_layer_milli_us=1_347,
    ),
)

# Qwen3.5-27B class (dense GDN + attention, hidden 5,120): 24 attention heads
# in 4 query groups of 256 channels, GDN 48 value / 16 key heads of 128 dims
# and a 4-tap conv, 64 layers (48 GDN), on H200 bf16 (2026-09-04). Admitted at
# TP1 x CP1/2/4 and TP2 x CP1 (the ten terms rank this GDN class at CP4, as
# they do the 35B GDN MoE class); TP2 x CP2 fails its gates and keeps the
# version-1 score, which on this class lost up to 112% at CP4.
QWEN35_27B_GEOMETRY = ModelGeometry(
    hidden_size=5_120,
    ffn_hidden_size=17_408,
    num_attention_heads=24,
    num_query_groups=4,
    kv_channels=256,
    gdn_value_heads=48,
    gdn_key_heads=16,
    gdn_key_head_dim=128,
    gdn_value_head_dim=128,
    gdn_conv_kernel=4,
)
DENSE_GDN_H5120_TABLE = CalibratedTable(
    table_id="dense-gdn-h5120-h200-bf16",
    coefficients_milli_us={
        "attention_token_cp_exchange": 0,
        "gdn_level": 3_043_381,
        "gdn_level_tp": 0,
        "gdn_token_per_rank": 0,
        "level_cp_per_layer": 0,
        "level_tp_per_layer": 442_039,
        "tiny_segment_per_layer": 230_182,
        "token_cp_exchange": 0,
        "token_per_rank": 11_441,
        "token_tp_collective": 0,
    },
    device_classes=(H200_CLASS,),
    param_dtypes=("torch.bfloat16",),
    geometries=(QWEN35_27B_GEOMETRY,),
    shapes=(
        ParallelShape(tp=1, cp=1),
        ParallelShape(tp=1, cp=2),
        ParallelShape(tp=1, cp=4),
        ParallelShape(tp=2, cp=1),
    ),
)

CALIBRATED_TABLES: tuple[CalibratedTable, ...] = (
    DENSE_H2560_TABLE,
    GDN_MOE_H2048_TABLE,
    DENSE_ATTN_H2048_TABLE,
    DENSE_ATTN_H4096_TABLE,
    DENSE_ATTN_H5120_TABLE,
    ATTN_MOE_H2048_TABLE,
    DENSE_GDN_H5120_TABLE,
)
# CPU-only planning (unit tests) prices with this table.
DEFAULT_TABLE = DENSE_H2560_TABLE
assert set(COEFFICIENTS_MILLI_US) == set(TERM_FUNCTIONS)
assert all(
    set(table.coefficients_milli_us) == set(TERM_FUNCTIONS)
    for table in CALIBRATED_TABLES
)


def predicted_work(
    features: LayoutFeatures,
    facts: ScoringFacts,
    coefficients: Mapping[str, int] | None = None,
) -> int:
    """Integer predicted work under a table (default: the shipped dense table)."""

    return score_terms(
        features,
        facts,
        COEFFICIENTS_MILLI_US if coefficients is None else coefficients,
    )


def predicted_us(
    features: LayoutFeatures,
    facts: ScoringFacts,
    coefficients: Mapping[str, int] | None = None,
) -> float:
    return predicted_work(features, facts, coefficients) / (
        WORK_PER_US * COEFFICIENT_SCALE_PER_US
    )


def prefix_tree_layout_score(
    layout: PrefixTreeLayout,
    *,
    cp_size: int,
    layers: int,
    uses_gdn: bool,
    tp_size: int = 1,
    gdn_layers: int | None = None,
    coefficient_version: int = COEFFICIENT_VERSION,
    coefficients: Mapping[str, int] | None = None,
) -> tuple[int, int, int, int]:
    """Price one layout for the given topology and model facts.

    ``gdn_layers`` defaults to ``layers`` when the model uses GDN and to zero
    otherwise; ``uses_gdn`` only matters for that default.
    ``coefficient_version`` selects a fitted table (2) or the fallback (1) and
    ``coefficients`` names the table (default: the shipped dense table); see
    ``select_scoring``.
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
        predicted_work(features, facts, coefficients),
        layout.packed_tokens,
        features.segment_count,
        layout.maximum_depth,
    )


__all__ = [
    "CALIBRATED_TABLES",
    "COEFFICIENTS_MILLI_US",
    "COEFFICIENT_SCALE_PER_US",
    "COEFFICIENT_VERSION",
    "COEFFICIENT_VERSION_FALLBACK",
    "DEFAULT_TABLE",
    "ATTN_MOE_H2048_TABLE",
    "DENSE_ATTN_H2048_TABLE",
    "DENSE_ATTN_H4096_TABLE",
    "DENSE_ATTN_H5120_TABLE",
    "DENSE_GDN_H5120_TABLE",
    "DENSE_H2560_TABLE",
    "GDN_MOE_H2048_TABLE",
    "CalibratedTable",
    "DeviceClass",
    "LayoutFeatures",
    "ModelGeometry",
    "ParallelShape",
    "ReRanker",
    "ScoringSelection",
    "SEGMENT_LENGTH_THRESHOLDS",
    "ScoringFacts",
    "TERM_FUNCTIONS",
    "TINY_SEGMENT_TOKENS",
    "WORK_PER_US",
    "layout_features",
    "predicted_us",
    "predicted_work",
    "prefix_tree_layout_score",
    "score_terms",
    "select_scoring",
]
