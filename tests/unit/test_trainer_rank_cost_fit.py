"""The calibration fitter recovers coefficients from within-cell paired deltas."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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


def test_production_regret_ignores_legacy_planner_rows(tmp_path: Path) -> None:
    """Paired planner A/B evidence times the ``automatic`` selection under both
    planner variants; only the current planner's rows describe production."""

    cell = {"cell": "cal-grpo-g8", "model": "m", "layers": 2, "tp": 1, "cp": 2}
    key = fit._cell_key(cell)
    rows = [
        {
            **cell,
            "record_type": "calibration_cell",
            "candidates": [
                {"label": "automatic", "matches": ["depth_one"]},
                {"label": "depth_one"},
            ],
        }
    ]
    for variant, ms in (
        ("current", 100.0),
        ("current", 100.0),
        ("legacy", 200.0),
        ("legacy", 200.0),
    ):
        rows.append(
            {
                **cell,
                "record_type": "calibration_sample",
                "role": "measured",
                "candidate_label": "automatic",
                "planner_variant": variant,
                "compile_statuses": ["none"],
                "ms_max_rank": ms,
            }
        )
    path = tmp_path / "evidence.jsonl"
    path.write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")
    candidates = [_candidate(key, "depth_one", 4096, 8, 1, 100.0)]
    report = fit.production_regret(candidates, [path])
    assert report[key]["automatic_ms"] == 100.0
    assert report[key]["regret_pct"] == 0.0


def _ab_rows(
    cell: dict, label: str, *, current: int, legacy: int, legacy_failed: bool = False
) -> list[dict]:
    rows = []
    for variant, count in (("current", current), ("legacy", legacy)):
        for i in range(count):
            rows.append(
                {
                    **cell,
                    "record_type": "calibration_sample",
                    "role": "measured",
                    "candidate_label": label,
                    "planner_variant": variant,
                    "compile_statuses": ["none"],
                    "round": i,
                    "ms_max_rank": 100.0,
                }
            )
    if legacy_failed:
        rows.append(
            {
                **cell,
                "record_type": "calibration_sample",
                "role": "measured",
                "candidate_label": label,
                "planner_variant": "legacy",
                "admission_failed": True,
            }
        )
    return rows


def test_completeness_counts_only_current_planner_rows(tmp_path: Path) -> None:
    """Legacy-planner rows of a paired A/B are not calibration evidence: they
    neither fill a candidate's row count nor fail a cell through their own
    admission failures."""

    import json

    cell = {"cell": "cal-grpo-g8", "model": "m", "layers": 2, "tp": 1, "cp": 2}
    header = {
        **cell,
        "record_type": "calibration_cell",
        "candidates": [{"label": "depth_one"}],
    }
    # Four current plus four legacy rows are four usable rows, not eight.
    thin = tmp_path / "thin.jsonl"
    thin.write_text(
        "\n".join(
            json.dumps(r)
            for r in [header, *_ab_rows(cell, "depth_one", current=4, legacy=4)]
        )
        + "\n"
    )
    gaps = fit.validate_completeness([thin], repeat=8)
    assert gaps == [f"{fit._cell_key(cell)}: depth_one has 4 usable rows (< 8)"]
    # Eight current rows are complete even when the legacy arm refused admission.
    full = tmp_path / "full.jsonl"
    full.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                header,
                *_ab_rows(cell, "depth_one", current=8, legacy=0, legacy_failed=True),
            ]
        )
        + "\n"
    )
    assert fit.validate_completeness([full], repeat=8) == []


def test_recertification_carries_only_the_named_shapes_and_refuses_missing_cells(
    tmp_path: Path,
) -> None:
    """--from-certificate with evidence: re-measured cells come from the rows;
    the certificate's cells of the explicitly named unchanged shapes keep their
    recorded aggregates; any other certificate cell the evidence does not
    re-measure refuses the re-certification (never a silent carry)."""

    import json

    cells = [
        {"cell": "cal-grpo-g8", "model": "m", "layers": 2, "tp": 1, "cp": cp}
        for cp in (1, 2, 4)
    ]
    keys = [fit._cell_key(c) for c in cells]
    features = {
        "packed_tokens": 4096,
        "segment_count": 8,
        "max_depth": 1,
        "segments_below": [0] * 8,
    }
    facts = {"layers": 2.0, "gdn_layers": 0.0, "tp": 1.0, "cp": 1.0, "uses_gdn": 0.0}
    fingerprint = {
        "source": "s",
        "requests_sha256": "r",
        "device": "d",
        "param_dtype": "bf16",
        "hidden_size": 8,
        "geometry": {"hidden_size": 8},
    }
    certificate = {
        "schema": fit.CERTIFICATE_SCHEMA,
        "cells": [
            {
                "cell": key,
                "facts": {**facts, "cp": float(cell["cp"])},
                "shape": [1, cell["cp"], 1, 1],
                **fingerprint,
                "candidates": [
                    {
                        "label": "depth_one",
                        "features": features,
                        "median_ms": 100.0,
                        "n": 8,
                        "spread_pct": 1.0,
                    }
                ],
            }
            for key, cell in zip(keys, cells)
        ],
    }
    cert_path = tmp_path / "certificate.json"
    cert_path.write_text(json.dumps(certificate))

    def evidence_for(*measured: dict) -> Path:
        rows: list[dict[str, Any]] = []
        for cell in measured:
            rows.append(
                {
                    **cell,
                    **fingerprint,
                    "record_type": "calibration_cell",
                    "candidates": [{"label": "depth_one", "features": features}],
                }
            )
            rows += [
                {
                    **cell,
                    "record_type": "calibration_sample",
                    "role": "measured",
                    "candidate_label": "depth_one",
                    "compile_statuses": ["none"],
                    "round": i,
                    "ms_max_rank": 90.0,
                }
                for i in range(8)
            ]
        path = tmp_path / f"evidence_{len(measured)}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    # The change touches CP > 1: CP1 may be carried, CP2 and CP4 must be re-measured.
    both = evidence_for(cells[1], cells[2])
    carried, records, problems = fit.carried_certificate_cells(
        cert_path, [both], carry_shapes={"tp1cp1"}
    )
    assert (
        problems == []
        and [c.cell for c in carried] == [keys[0]]
        and carried[0].ms == 100.0
    )
    assert fit.validate_completeness([both], repeat=8, carried=records) == []
    manifest = {
        "schema": fit.MANIFEST_SCHEMA,
        "cells": [{"key": k} for k in keys],
        "excluded": [],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert fit.validate_manifest(
        [both], manifest_path, excluded=[], carried=records
    ) == ([], [])
    # One affected cell re-measured, the other missing: refused, not carried.
    one = evidence_for(cells[1])
    _carried, _records, problems = fit.carried_certificate_cells(
        cert_path, [one], carry_shapes={"tp1cp1"}
    )
    assert problems == [f"affected cell not re-measured: {keys[2]}"]
    # Empty evidence: every affected cell is missing.
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    _carried, _records, problems = fit.carried_certificate_cells(
        cert_path, [empty], carry_shapes={"tp1cp1"}
    )
    assert problems == [f"affected cell not re-measured: {k}" for k in keys[1:]]
    # No shapes named: nothing may be carried.
    _carried, _records, problems = fit.carried_certificate_cells(
        cert_path, [both], carry_shapes=set()
    )
    assert problems and problems[0].startswith(
        "re-certification needs the configurations"
    )
    # A cell-key substring names a configuration the change cannot reach
    # (here the CP4 cell) regardless of its shape.
    _carried, records, problems = fit.carried_certificate_cells(
        cert_path, [one], carry_shapes={"tp1cp1"}, carry_cells=["|tp1|cp4"]
    )
    assert problems == [] and [r["cell"] for r in records] == [keys[0], keys[2]]
    # A carried-shape cell that is re-measured comes from the rows, not the certificate.
    all_three = evidence_for(*cells)
    carried, records, problems = fit.carried_certificate_cells(
        cert_path, [all_three], carry_shapes={"tp1cp1"}
    )
    assert problems == [] and carried == [] and records == []
