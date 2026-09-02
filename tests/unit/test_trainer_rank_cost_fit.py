"""The calibration fitter recovers coefficients from within-cell paired deltas."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_FIT = Path(__file__).resolve().parents[2] / "dev" / "trainer_rank_cost_fit.py"
_spec = importlib.util.spec_from_file_location("trainer_rank_cost_fit", _FIT)
assert _spec is not None and _spec.loader is not None
fit = importlib.util.module_from_spec(_spec)
sys.modules["trainer_rank_cost_fit"] = fit
_spec.loader.exec_module(fit)


def _candidate(
    cell: str, label: str, packed: int, segments: int, depth: int, ms: float
) -> object:
    features = {
        "packed_tokens": packed,
        "segment_count": segments,
        "shared_segments": max(0, depth - 1),
        "max_depth": depth,
        "shared_tokens": 0,
        "fanout_sum": 0,
        "small_segments": 0,
        "tiny_segments": 0,
        "attention_area": 0,
    }
    facts = {"layers": 2.0, "gdn_layers": 2.0, "tp": 1.0, "cp": 4.0, "uses_gdn": 1.0}
    return fit.Candidate(cell, label, features, facts, ms, 4, 1.0)


def test_paired_delta_nnls_recovers_known_coefficients() -> None:
    terms = ("token_work_per_layer", "level_per_layer")
    # True model: 5 us per token-layer, 20 ms per (level x layer x cp); a cell
    # constant of 300 ms cancels in paired deltas.
    true = np.array([5.0, 20_000.0])
    cells = []
    for cell in range(3):
        base = 300.0 + 50.0 * cell
        for label, packed, depth in (
            ("a", 20_000, 1),
            ("b", 12_000, 2),
            ("c", 8_000, 3),
        ):
            features = np.array([packed * 2 / 1, (depth - 1) * 2 * 4])
            ms = base + float(features @ true) / 1_000.0
            cells.append(
                _candidate(f"cell{cell}", label, packed, 16 + depth, depth, ms)
            )
    x, y = fit.paired_deltas(cells, terms)
    beta = fit.nnls(x, y)
    assert np.allclose(beta, true, rtol=1e-3)
    report = fit.evaluate(cells, fit.predict(cells, terms, beta))
    assert report["pairwise_accuracy"] == 1.0
    assert report["max_regret_pct"] == 0.0
    assert fit.gates_pass(report) == []


def test_evaluate_reports_regret_of_a_wrong_selection() -> None:
    cells = [
        _candidate("c", "fast", 10_000, 16, 1, 100.0),
        _candidate("c", "slow", 8_000, 17, 2, 120.0),
    ]
    # A scorer that prefers the slow layout has 20% regret and a clear miss.
    report = fit.evaluate(cells, np.array([2.0, 1.0]))
    assert report["per_cell"]["c"]["selected"] == "slow"
    assert abs(report["max_regret_pct"] - 20.0) < 1e-9
    assert report["clear_misses"] == ["c"]
    assert fit.gates_pass(report)
