"""Fit and validate the prefix-tree layout cost model from calibration evidence.

Input: JSONL written by ``dev/trainer_rank_landing_acceptance.py --phase
cost-calibrate`` (one ``calibration_cell`` row per cell listing candidates and
their layout features, plus ``calibration_sample`` rows). A *cell* is one
(workload, model, layers, tp, cp) combination; only candidates of the same cell
are comparable, and only differences within a cell are informative (everything
a call shares across its layouts cancels).

Method
------
1. Aggregate compile-free, unsplit, admitted measured samples per (cell,
   candidate) into a median max-rank forward+backward time.
2. Build every within-cell candidate pair and fit non-negative coefficients on
   feature *deltas* by least squares (paired deltas, per the research thread's
   recommendation), so per-cell constants never enter.
3. Whole-cell holdout: fit on the remaining cells, then score the held-out
   cells on the gates below.  ``--holdout`` selects held-out cells by substring
   (default: every Ellavox group with an odd index, plus any cell name matching
   ``--holdout`` patterns).

Terms are interpretable products of a layout feature and a topology/model
factor (see ``TERMS``); the fit is a non-negative least squares over their
coefficients in microseconds per unit.  ``--integerize`` prints the frozen
integer table for ``_planner_cost.py`` and re-checks every ranking under the
integer model.

Gates (held-out cells; noise-qualified, per the research thread's review):
- pairwise ordering accuracy >= 90% for pairs separated by more than 3%;
- median selected regret <= 2%, p95 <= 5%, no cell above 10%;
- on cells whose best candidate leads the runner-up by more than the noise
  band, the selection is within 5% of the best.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

# Terms are the production module's integer term functions, so a fitted table
# is consumed verbatim by the scorer (single source of truth).
from art.trainer_rank._planner_cost import (  # noqa: E402
    TERM_FUNCTIONS,
    WORK_PER_US,
    LayoutFeatures,
    ScoringFacts,
)

TERMS = TERM_FUNCTIONS
DEFAULT_TERMS = tuple(TERMS)
NOISE_BAND_PCT = 3.0


@dataclass(frozen=True)
class Candidate:
    cell: str
    label: str
    features: dict[str, int]
    facts: dict[str, float]
    ms: float
    n: int
    spread_pct: float


def _cell_key(row: dict[str, Any]) -> str:
    return (
        f"{row['cell']}|{row['model']}|L{row['layers']}|tp{row['tp']}|cp{row['cp']}"
        + (
            f"|g{row['workload'].get('group')}"
            if row.get("workload", {}).get("group") is not None
            else ""
        )
    )


def load_candidates(paths: list[Path]) -> list[Candidate]:
    cells: dict[str, dict[str, Any]] = {}
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "calibration_cell":
                cells[_cell_key(row)] = row
            elif (
                row.get("record_type") == "calibration_sample"
                and row.get("role") == "measured"
            ):
                if row.get("admission_failed") or row.get("subforward_count", 1) != 1:
                    continue
                if any(status != "none" for status in row.get("compile_statuses", [])):
                    continue
                samples[(_cell_key(row), str(row["candidate_label"]))].append(
                    float(row["ms_max_rank"])
                )
    candidates: list[Candidate] = []
    for key, cell in cells.items():
        facts = {
            "layers": float(cell["layers"]),
            "gdn_layers": float(cell["gdn_layers"]),
            "tp": float(cell["tp"]),
            "cp": float(cell["cp"]),
            "uses_gdn": float(bool(cell["uses_gdn"])),
        }
        for candidate in cell["candidates"]:
            values = samples.get((key, candidate["label"]))
            if not values or len(values) < 2:
                continue
            median = statistics.median(values)
            spread = (max(values) - min(values)) / median * 100.0
            candidates.append(
                Candidate(
                    key,
                    candidate["label"],
                    candidate["features"],
                    facts,
                    median,
                    len(values),
                    spread,
                )
            )
    return candidates


def term_matrix(candidates: list[Candidate], terms: tuple[str, ...]) -> np.ndarray:
    """Term values in feature units (the integer functions divided by WORK_PER_US)."""

    rows = []
    for candidate in candidates:
        features = LayoutFeatures(**candidate.features)
        facts = ScoringFacts(
            cp_size=int(candidate.facts["cp"]),
            tp_size=int(candidate.facts["tp"]),
            layers=int(candidate.facts["layers"]),
            gdn_layers=int(candidate.facts["gdn_layers"]),
        )
        rows.append([TERMS[name](features, facts) / WORK_PER_US for name in terms])
    return np.asarray(rows, dtype=np.float64)


def paired_deltas(
    candidates: list[Candidate], terms: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    by_cell: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_cell[candidate.cell].append(candidate)
    features = term_matrix(candidates, terms)
    index = {id(candidate): i for i, candidate in enumerate(candidates)}
    xs, ys = [], []
    for members in by_cell.values():
        for a, b in itertools.combinations(members, 2):
            xs.append(features[index[id(a)]] - features[index[id(b)]])
            ys.append((a.ms - b.ms) * 1_000.0)  # microseconds
    return np.asarray(xs), np.asarray(ys)


def nnls(x: np.ndarray, y: np.ndarray, *, iterations: int = 20_000) -> np.ndarray:
    """Projected-gradient non-negative least squares (small problems)."""

    if x.size == 0:
        return np.zeros(x.shape[1] if x.ndim == 2 else 0)
    scale = np.maximum(np.abs(x).max(axis=0), 1e-12)
    xs = x / scale
    beta = np.zeros(xs.shape[1])
    step = 1.0 / (np.linalg.norm(xs, 2) ** 2 + 1e-12)
    for _ in range(iterations):
        gradient = xs.T @ (xs @ beta - y)
        updated = np.maximum(beta - step * gradient, 0.0)
        if np.max(np.abs(updated - beta)) < 1e-10 * (1 + np.max(np.abs(beta))):
            beta = updated
            break
        beta = updated
    return beta / scale


def predict(
    candidates: list[Candidate], terms: tuple[str, ...], beta: np.ndarray
) -> np.ndarray:
    return term_matrix(candidates, terms) @ beta


def evaluate(candidates: list[Candidate], predicted_us: np.ndarray) -> dict[str, Any]:
    by_cell: dict[str, list[tuple[Candidate, float]]] = defaultdict(list)
    for candidate, value in zip(candidates, predicted_us, strict=True):
        by_cell[candidate.cell].append((candidate, float(value)))
    ordered_pairs = 0
    ordered_correct = 0
    regrets: list[float] = []
    clear_misses: list[str] = []
    per_cell: dict[str, dict[str, Any]] = {}
    for cell, members in by_cell.items():
        best_measured = min(members, key=lambda item: item[0].ms)
        selected = min(members, key=lambda item: item[1])
        regret = (selected[0].ms - best_measured[0].ms) / best_measured[0].ms * 100.0
        regrets.append(regret)
        runner_up = (
            sorted(members, key=lambda item: item[0].ms)[1][0].ms
            if len(members) > 1
            else best_measured[0].ms
        )
        lead_pct = (runner_up - best_measured[0].ms) / best_measured[0].ms * 100.0
        noise = max(best_measured[0].spread_pct, NOISE_BAND_PCT)
        if lead_pct > noise and regret > 5.0:
            clear_misses.append(cell)
        for a, b in itertools.combinations(members, 2):
            separation = abs(a[0].ms - b[0].ms) / min(a[0].ms, b[0].ms) * 100.0
            if separation <= 3.0:
                continue
            ordered_pairs += 1
            if (a[0].ms < b[0].ms) == (a[1] < b[1]):
                ordered_correct += 1
        per_cell[cell] = {
            "selected": selected[0].label,
            "best": best_measured[0].label,
            "regret_pct": regret,
            "best_lead_pct": lead_pct,
            "candidates": {
                m[0].label: {"ms": m[0].ms, "pred_us": m[1], "n": m[0].n}
                for m in members
            },
        }
    regrets_sorted = sorted(regrets)
    return {
        "cells": len(by_cell),
        "pairwise_accuracy": (ordered_correct / ordered_pairs)
        if ordered_pairs
        else float("nan"),
        "ordered_pairs": ordered_pairs,
        "median_regret_pct": statistics.median(regrets) if regrets else float("nan"),
        "p95_regret_pct": regrets_sorted[
            min(len(regrets_sorted) - 1, int(0.95 * len(regrets_sorted)))
        ]
        if regrets
        else float("nan"),
        "max_regret_pct": max(regrets) if regrets else float("nan"),
        "clear_misses": clear_misses,
        "per_cell": per_cell,
    }


def gates_pass(report: dict[str, Any]) -> list[str]:
    problems = []
    if report["ordered_pairs"] and report["pairwise_accuracy"] < 0.9:
        problems.append(f"pairwise accuracy {report['pairwise_accuracy']:.3f} < 0.90")
    if report["cells"] and report["median_regret_pct"] > 2.0:
        problems.append(f"median regret {report['median_regret_pct']:.2f}% > 2%")
    if report["cells"] and report["p95_regret_pct"] > 5.0:
        problems.append(f"p95 regret {report['p95_regret_pct']:.2f}% > 5%")
    if report["cells"] and report["max_regret_pct"] > 10.0:
        problems.append(f"max regret {report['max_regret_pct']:.2f}% > 10%")
    if report["clear_misses"]:
        problems.append(
            f"clear-winner cells selected >5% off: {report['clear_misses']}"
        )
    return problems


def current_score_report(
    candidates: list[Candidate], paths: list[Path]
) -> dict[str, Any]:
    """Regret of the shipped scorer, from the ``current_score_us`` the harness logged."""

    scores: dict[tuple[str, str], float] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "calibration_cell":
                for candidate in row["candidates"]:
                    scores[(_cell_key(row), candidate["label"])] = float(
                        candidate["current_score_us"]
                    )
    predicted = np.asarray([scores[(c.cell, c.label)] for c in candidates])
    return evaluate(candidates, predicted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="+", type=Path)
    parser.add_argument("--terms", default=",".join(DEFAULT_TERMS))
    parser.add_argument(
        "--holdout",
        default="",
        help="comma-separated substrings selecting held-out cells (odd Ellavox groups are always held out)",
    )
    parser.add_argument("--report", default="", help="write the JSON report here")
    parser.add_argument("--integerize", action="store_true")
    arguments = parser.parse_args()
    terms = tuple(t for t in arguments.terms.split(",") if t)
    candidates = load_candidates(arguments.evidence)
    if not candidates:
        print("no usable candidates", file=sys.stderr)
        raise SystemExit(1)
    patterns = [p for p in arguments.holdout.split(",") if p]

    def held_out(cell: str) -> bool:
        if any(p in cell for p in patterns):
            return True
        if "cal-ellavox" in cell and "|g" in cell:
            return int(cell.rsplit("|g", 1)[1]) % 2 == 1
        return False

    train = [c for c in candidates if not held_out(c.cell)]
    test = [c for c in candidates if held_out(c.cell)]
    x, y = paired_deltas(train, terms)
    beta = nnls(x, y)
    report: dict[str, Any] = {
        "terms": dict(zip(terms, [float(b) for b in beta], strict=True)),
        "train_cells": len({c.cell for c in train}),
        "test_cells": len({c.cell for c in test}),
        "train": evaluate(train, predict(train, terms, beta)),
        "test": evaluate(test, predict(test, terms, beta)) if test else None,
        "current_all": current_score_report(candidates, arguments.evidence),
        "fit_all": evaluate(candidates, predict(candidates, terms, beta)),
    }
    if arguments.integerize:
        integer = {name: int(round(value)) for name, value in report["terms"].items()}
        report["integer_terms_us"] = integer
        beta_int = np.asarray([integer[name] for name in terms], dtype=np.float64)
        report["integer_all"] = evaluate(
            candidates, predict(candidates, terms, beta_int)
        )
    for split in ("train", "test", "current_all", "fit_all", "integer_all"):
        block = report.get(split)
        if not block:
            continue
        print(
            f"{split:12s} cells={block['cells']:3d} pairs={block['ordered_pairs']:4d} "
            f"acc={block['pairwise_accuracy']:.3f} regret median={block['median_regret_pct']:.2f}% "
            f"p95={block['p95_regret_pct']:.2f}% max={block['max_regret_pct']:.2f}% "
            f"clear_misses={len(block['clear_misses'])}"
        )
    print("coefficients (us/unit):", json.dumps(report["terms"], indent=1))
    if report["test"]:
        problems = gates_pass(report["test"])
        print(
            "held-out gates:",
            "PASS" if not problems else "FAIL: " + "; ".join(problems),
        )
    if arguments.report:
        Path(arguments.report).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
