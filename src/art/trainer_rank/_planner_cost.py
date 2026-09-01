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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._prefix_tree_planner import PrefixTreeLayout

COEFFICIENT_VERSION = 1

# One integer work unit represents 1/1024 microsecond of predicted wall time.
WORK_PER_US = 1_024

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
) -> tuple[int, int, int, int]:
    """Price one layout for the given topology and model facts."""

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
    "WORK_PER_US",
    "prefix_tree_layout_score",
]
