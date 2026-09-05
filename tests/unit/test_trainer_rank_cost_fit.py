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
        "max_depth": depth,
        "segments_below": [0] * 8,
    }
    facts = {"layers": 2.0, "gdn_layers": 2.0, "tp": 1.0, "cp": 4.0, "uses_gdn": 1.0}
    return fit.Candidate(cell, label, features, facts, ms, 4, 1.0)


def test_paired_delta_nnls_recovers_known_coefficients() -> None:
    terms = ("token_per_rank", "level_cp_per_layer")
    # True model: 5 us per per-rank token-layer, 20 ms per (level x layer x
    # (cp - 1)); a cell constant of 300 ms cancels in paired deltas.
    true = np.array([5.0, 20_000.0])
    cells = []
    for cell in range(3):
        base = 300.0 + 50.0 * cell
        for label, packed, depth in (
            ("a", 20_000, 1),
            ("b", 12_000, 2),
            ("c", 8_000, 3),
        ):
            features = np.array([packed * 2 // 4, (depth - 1) * 2 * 3])
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


def test_manifest_validation_flags_missing_unexpected_and_mixed_cells(
    tmp_path: Path,
) -> None:
    """A whole missing cell, an unexpected cell, and duplicate cells with
    different execution fingerprints must all fail manifest validation."""

    import json

    def cell_row(cell: str, group: int, *, source: str = "s1") -> dict[str, object]:
        return {
            "record_type": "calibration_cell",
            "cell": cell,
            "model": "Qwen/Qwen3.5-4B",
            "layers": 32,
            "tp": 1,
            "cp": 1,
            "workload": {"group": group},
            "source": source,
            "requests_sha256": f"w{group}",
            "device": "NVIDIA H200",
            "param_dtype": "torch.bfloat16",
            "hidden_size": 2560,
            "candidates": [],
        }

    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                cell_row("cal-ellavox", 0),
                cell_row("cal-ellavox", 0, source="s2"),  # same key, other source
                cell_row("cal-ellavox", 9),  # not in the manifest
            )
        )
        + "\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": fit.MANIFEST_SCHEMA,
                "cells": [
                    {"key": "cal-ellavox|Qwen/Qwen3.5-4B|L32|tp1|cp1|g0"},
                    {"key": "cal-ellavox|Qwen/Qwen3.5-4B|L32|tp1|cp1|g1"},
                    {"key": "cal-ellavox|Qwen/Qwen3.5-4B|L32|tp1|cp1|g2"},
                ],
                "excluded": [{"key": "cal-ellavox|Qwen/Qwen3.5-4B|L32|tp1|cp1|g2"}],
            }
        )
    )
    problems, excluded = fit.validate_manifest(
        [evidence], manifest, excluded=["cp1|g2"]
    )
    assert excluded == ["cal-ellavox|Qwen/Qwen3.5-4B|L32|tp1|cp1|g2"]
    joined = "\n".join(problems)
    assert (
        "expected cell missing from the evidence: cal-ellavox|Qwen/Qwen3.5-4B|L32|tp1|cp1|g1"
        in joined
    )
    assert (
        "unexpected cell in the evidence: cal-ellavox|Qwen/Qwen3.5-4B|L32|tp1|cp1|g9"
        in joined
    )
    assert "different execution fingerprints" in joined
    # An exclusion that the manifest does not list is itself a problem.
    problems, _ = fit.validate_manifest([evidence], manifest, excluded=["cp1|g0"])
    assert any("not listed in the manifest" in p for p in problems)


def test_cell_key_carries_expert_parallelism_only_when_present() -> None:
    row = {
        "cell": "cal-grpo-g8",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "layers": 40,
        "tp": 1,
        "cp": 4,
        "workload": {"kind": "grpo"},
    }
    assert fit._cell_key(row) == "cal-grpo-g8|Qwen/Qwen3.5-35B-A3B|L40|tp1|cp4"
    assert fit._cell_key({**row, "ep": 2}) == (
        "cal-grpo-g8|Qwen/Qwen3.5-35B-A3B|L40|tp1|cp4|ep2"
    )
    assert fit._cell_key({**row, "ep": 4, "etp": 2, "workload": {"group": 3}}) == (
        "cal-grpo-g8|Qwen/Qwen3.5-35B-A3B|L40|tp1|cp4|ep4|etp2|g3"
    )
    # Older rows without expert fields keep their historical keys.
    assert fit._shape({"tp": 2, "cp": 1}) == (2, 1, 1, 1)


def test_group_reports_judge_gates_per_group() -> None:
    def candidates(cell: str, model: str, times: dict[str, float]) -> list:
        return [
            fit.Candidate(cell, label, {}, {}, ms, 8, 0.5, model, (1, 1, 1, 1))
            for label, ms in times.items()
        ]

    good = candidates("a|M1|L2|tp1|cp1", "M1", {"x": 100.0, "y": 120.0})
    bad = candidates("b|M2|L2|tp1|cp1", "M2", {"x": 100.0, "y": 130.0})
    # Predictions rank the good cell right and the bad cell wrong (30% regret).
    predicted = [1.0, 2.0, 2.0, 1.0]
    report = fit.evaluate_groups(
        good + bad, __import__("numpy").asarray(predicted), lambda c: c.model
    )
    assert report["M1"]["max_regret_pct"] == 0.0 and not report["M1"]["gate_problems"]
    assert report["M2"]["max_regret_pct"] > 10.0 and report["M2"]["gate_problems"]


def test_integerize_keeps_plain_rounding_when_it_preserves_rankings() -> None:
    import numpy as np

    cells = [
        fit.Candidate(
            "a|M|L2|tp1|cp1",
            label,
            {
                "packed_tokens": tokens,
                "segment_count": 1,
                "max_depth": 1,
                "segments_below": (),
            },
            {
                "layers": 2.0,
                "gdn_layers": 0.0,
                "tp": 1.0,
                "cp": 1.0,
                "ep": 1.0,
                "etp": 1.0,
                "uses_gdn": 0.0,
            },
            ms,
            8,
            0.5,
        )
        for label, (ms, tokens) in {"x": (100.0, 1000), "y": (120.0, 1500)}.items()
    ]
    assert fit.integerize(cells, ("token_per_rank",), np.asarray([2.0004])) == {
        "token_per_rank": 2000
    }


def test_integerize_never_loses_to_plain_rounding() -> None:
    import numpy as np

    def candidates(cell: str, times: dict[str, tuple[float, int]]) -> list:
        return [
            fit.Candidate(
                cell,
                label,
                {
                    "packed_tokens": tokens,
                    "segment_count": 1,
                    "max_depth": 1,
                    "segments_below": (),
                },
                {
                    "layers": 2.0,
                    "gdn_layers": 0.0,
                    "tp": 1.0,
                    "cp": 1.0,
                    "ep": 1.0,
                    "etp": 1.0,
                    "uses_gdn": 0.0,
                },
                ms,
                8,
                0.5,
            )
            for label, (ms, tokens) in times.items()
        ]

    cells = candidates(
        "a|M|L2|tp1|cp1", {"x": (100.0, 1000), "y": (110.0, 1100)}
    ) + candidates("b|M|L2|tp1|cp1", {"x": (100.0, 1000), "y": (99.0, 1001)})
    terms = ("token_per_rank",)
    beta = np.asarray([0.0004])  # rounds to 0 milli-us: every cell would tie
    table = fit.integerize(cells, terms, beta)
    assert all(isinstance(v, int) for v in table.values())
    matrix = fit.term_matrix(cells, terms)
    rounded = np.asarray([0.0])
    refined = np.asarray([table["token_per_rank"] / 1_000.0])
    assert fit.selection_loss(cells, matrix @ refined) <= fit.selection_loss(
        cells, matrix @ rounded
    )
