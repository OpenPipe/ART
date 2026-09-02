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


def production_regret(
    candidates: list[Candidate], paths: list[Path]
) -> dict[str, dict[str, Any]]:
    """Measured regret of the timed ``automatic`` selection per cell, if present."""

    automatic: dict[str, tuple[list[float], list[str]]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "calibration_cell":
                for candidate in row["candidates"]:
                    if candidate["label"] == "automatic":
                        automatic.setdefault(
                            _cell_key(row), ([], candidate.get("matches", []))
                        )
            elif (
                row.get("record_type") == "calibration_sample"
                and row.get("role") == "measured"
                and row.get("candidate_label") == "automatic"
                and not row.get("admission_failed")
                and row.get("subforward_count", 1) == 1
                and all(status == "none" for status in row.get("compile_statuses", []))
            ):
                automatic.setdefault(_cell_key(row), ([], []))[0].append(
                    float(row["ms_max_rank"])
                )
    by_cell: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_cell[candidate.cell].append(candidate)
    report: dict[str, dict[str, Any]] = {}
    for cell, (values, matches) in automatic.items():
        if len(values) < 2 or cell not in by_cell:
            continue
        best = min(by_cell[cell], key=lambda c: c.ms)
        ms = statistics.median(values)
        report[cell] = {
            "automatic_ms": ms,
            "best": best.label,
            "best_ms": best.ms,
            "regret_pct": (ms - best.ms) / best.ms * 100.0,
            "matches": matches,
        }
    return report


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
            if candidate["label"] == "automatic":
                continue  # the production selection is reported, not fitted
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


def _normalized(features: dict[str, Any]) -> dict[str, Any]:
    """JSON round-trips turn tuples into lists; compare on a canonical form."""

    return {
        key: tuple(value) if isinstance(value, (list, tuple)) else value
        for key, value in features.items()
    }


def selector_check(candidates: list[Candidate]) -> dict[str, dict[str, Any]]:
    """Run the production selector on each cell's tree under the shipped table.

    Reports the chosen layout's features, whether it is one of the measured
    candidates (and that candidate's measured regret), or a nonuniform layout
    outside the mandatory family (which then needs measuring before the table
    can be trusted on that cell).
    """

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trainer_rank_landing_acceptance import _calibration_requests

    from art.trainer_rank._planner_cost import layout_features
    from art.trainer_rank._prefix_tree_planner import (
        build_canonical_prefix_tree,
        select_prefix_tree_layout,
    )

    by_cell: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_cell[candidate.cell].append(candidate)
    report: dict[str, dict[str, Any]] = {}
    for cell, members in by_cell.items():
        name, rest = cell.split("|", 1)
        group = int(cell.rsplit("|g", 1)[1]) if "|g" in cell else 0
        requests, _ = _calibration_requests(name, group=group)
        tree = build_canonical_prefix_tree(
            tuple(r.input_tokens.reshape(-1).to(torch.long) for r in requests)
        )
        facts = members[0].facts
        selected = select_prefix_tree_layout(
            tree,
            cp_size=int(facts["cp"]),
            layers=int(facts["layers"]),
            uses_gdn=bool(facts["uses_gdn"]),
            tp_size=int(facts["tp"]),
            gdn_layers=int(facts["gdn_layers"]),
            refinement_work_budget=2_000,
        ).layout
        features = _normalized(layout_features(selected).as_dict())
        best = min(members, key=lambda c: c.ms)
        match = next((c for c in members if _normalized(c.features) == features), None)
        report[cell] = {
            "selected": match.label if match else "nonuniform (unmeasured)",
            "selected_packed_tokens": features["packed_tokens"],
            "best": best.label,
            "regret_pct": ((match.ms - best.ms) / best.ms * 100.0) if match else None,
        }
    return report


def refresh_features(paths: list[Path]) -> None:
    """Recompute every cell's candidate features with the current extractor.

    Cells are reproducible from their workload description (GRPO shape and
    seed, heterogeneous seed, or Ellavox corpus group), so evidence recorded
    with an older feature set can be brought forward without re-timing.
    Rewrites the ``calibration_cell`` rows in place.
    """

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trainer_rank_landing_acceptance import _calibration_requests

    from art.trainer_rank._planner_cost import layout_features
    from art.trainer_rank._prefix_tree_planner import (
        build_canonical_prefix_tree,
        prefix_tree_layout_candidates,
    )

    for path in paths:
        lines = path.read_text().splitlines()
        out = []
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "calibration_cell":
                group = int(row["workload"].get("group", 0) or 0)
                requests, _ = _calibration_requests(row["cell"], group=group)
                tree = build_canonical_prefix_tree(
                    tuple(r.input_tokens.reshape(-1).to(torch.long) for r in requests)
                )
                by_label = {
                    label: layout_features(candidate.layout).as_dict()
                    for candidate in prefix_tree_layout_candidates(tree)
                    for label in candidate.labels
                }
                for candidate in row["candidates"]:
                    # The timed production selection keeps its recorded
                    # features; only mandatory-family labels are recomputed.
                    if candidate["label"] in by_label:
                        candidate["features"] = by_label[candidate["label"]]
            out.append(json.dumps(row, sort_keys=True, default=str))
        path.write_text("\n".join(out) + "\n")


def term_matrix(candidates: list[Candidate], terms: tuple[str, ...]) -> np.ndarray:
    """Term values in feature units (the integer functions divided by WORK_PER_US)."""

    rows = []
    for candidate in candidates:
        features = LayoutFeatures(
            **{
                **candidate.features,
                "segments_below": tuple(candidate.features.get("segments_below", ())),
                "tokens_below": tuple(candidate.features.get("tokens_below", ())),
            }
        )
        facts = ScoringFacts(
            cp_size=int(candidate.facts["cp"]),
            tp_size=int(candidate.facts["tp"]),
            layers=int(candidate.facts["layers"]),
            gdn_layers=int(candidate.facts["gdn_layers"]),
        )
        rows.append([TERMS[name](features, facts) / WORK_PER_US for name in terms])
    return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True)
class Pair:
    a: Candidate
    b: Candidate
    weight: float

    @property
    def separation_pct(self) -> float:
        return abs(self.a.ms - self.b.ms) / min(self.a.ms, self.b.ms) * 100.0


def candidate_pairs(candidates: list[Candidate]) -> list[Pair]:
    """Within-cell candidate pairs with base weights.

    Every cell contributes equally regardless of how many candidates it has,
    and a pair's error counts relative to the pair's magnitude (the ranking
    decision is between close candidates, not between no-sharing and full
    sharing).
    """

    by_cell: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_cell[candidate.cell].append(candidate)
    pairs: list[Pair] = []
    for members in by_cell.values():
        combos = list(itertools.combinations(members, 2))
        if not combos:
            continue
        cell_weight = 1.0 / math.sqrt(len(combos))
        for a, b in combos:
            pairs.append(Pair(a, b, cell_weight / (0.5 * (a.ms + b.ms))))
    return pairs


def paired_deltas(
    candidates: list[Candidate],
    terms: tuple[str, ...],
    pairs: list[Pair] | None = None,
    extra_weights: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted (feature delta, timing delta in us) rows for the pairs."""

    pairs = candidate_pairs(candidates) if pairs is None else pairs
    features = term_matrix(candidates, terms)
    index = {id(candidate): i for i, candidate in enumerate(candidates)}
    xs, ys = [], []
    for position, pair in enumerate(pairs):
        weight = pair.weight * (extra_weights or {}).get(position, 1.0)
        xs.append((features[index[id(pair.a)]] - features[index[id(pair.b)]]) * weight)
        ys.append((pair.a.ms - pair.b.ms) * 1_000.0 * weight)
    return np.asarray(xs), np.asarray(ys)


def fit_ranking(
    candidates: list[Candidate],
    terms: tuple[str, ...],
    *,
    rounds: int = 4,
    boost: float = 4.0,
) -> np.ndarray:
    """Non-negative fit with rank-focused reweighting.

    After each least-squares solve, separated pairs (> NOISE_BAND_PCT apart)
    whose predicted order is wrong have their weight multiplied by ``boost``
    and the fit repeats, so the coefficients concentrate on the ranking
    decisions rather than on the large easy deltas.
    """

    pairs = candidate_pairs(candidates)
    extra: dict[int, float] = {}
    beta = nnls(*paired_deltas(candidates, terms, pairs))
    for _ in range(rounds):
        predicted = predict(candidates, terms, beta)
        index = {id(candidate): i for i, candidate in enumerate(candidates)}
        wrong = [
            position
            for position, pair in enumerate(pairs)
            if pair.separation_pct > NOISE_BAND_PCT
            and (pair.a.ms < pair.b.ms)
            != (predicted[index[id(pair.a)]] < predicted[index[id(pair.b)]])
        ]
        if not wrong:
            break
        for position in wrong:
            extra[position] = extra.get(position, 1.0) * boost
        beta = nnls(*paired_deltas(candidates, terms, pairs, extra))
    return beta


def nnls(x: np.ndarray, y: np.ndarray, *, iterations: int = 500) -> np.ndarray:
    """Non-negative least squares (Lawson-Hanson active set, column-scaled)."""

    if x.size == 0:
        return np.zeros(x.shape[1] if x.ndim == 2 else 0)
    scale = np.maximum(np.abs(x).max(axis=0), 1e-12)
    xs = x / scale
    n = xs.shape[1]
    passive = np.zeros(n, dtype=bool)
    beta = np.zeros(n)
    tolerance = 1e-10 * max(1.0, float(np.abs(xs.T @ y).max()))
    for _ in range(iterations):
        gradient = xs.T @ (y - xs @ beta)
        gradient[passive] = -np.inf
        if gradient.max() <= tolerance:
            break
        passive[int(np.argmax(gradient))] = True
        while True:
            trial = np.zeros(n)
            trial[passive], *_ = np.linalg.lstsq(xs[:, passive], y, rcond=None)
            negative = passive & (trial <= 0)
            if not negative.any():
                beta = trial
                break
            # Step back toward the feasible boundary and drop what hits zero.
            ratios = beta[negative] / np.maximum(
                beta[negative] - trial[negative], 1e-300
            )
            alpha = float(min(1.0, ratios.min()))
            beta = beta + alpha * (trial - beta)
            passive &= beta > 1e-12
            beta[~passive] = 0.0
    return beta / scale


def selection_loss(
    candidates: list[Candidate], predicted_us: np.ndarray, *, pair_weight: float = 0.05
) -> float:
    """Sum of selected regret (percent) plus a small pairwise-ordering penalty."""

    report = evaluate(candidates, predicted_us)
    regret = sum(cell["regret_pct"] for cell in report["per_cell"].values())
    ordering = (
        (1.0 - report["pairwise_accuracy"]) * 100.0 if report["ordered_pairs"] else 0.0
    )
    return regret + pair_weight * ordering


def fit_regret(
    candidates: list[Candidate],
    terms: tuple[str, ...],
    start: np.ndarray,
    *,
    sweeps: int = 12,
) -> np.ndarray:
    """Minimize selected regret directly by coordinate search in log space.

    Starts from the least-squares solution (which fixes the scale) and tries
    multiplicative steps per coefficient, including switching a coefficient
    on at a small value or off; accepts a step only when the training loss
    falls. Greedy, so it runs from a few deterministic starts (the solution
    scaled up and down, and reversed coordinate order) and keeps the best.
    """

    matrix = term_matrix(candidates, terms)
    positive = start[start > 0]
    floor = float(positive.min()) * 1e-3 if positive.size else 1e-3
    factors = (0.25, 0.5, 0.8, 1.25, 2.0, 4.0)

    def search(beta: np.ndarray, order: list[int]) -> tuple[float, np.ndarray]:
        beta = beta.copy()
        best = selection_loss(candidates, matrix @ beta)
        for _ in range(sweeps):
            improved = False
            for j in order:
                current = beta[j]
                trials = [current * factor for factor in factors if current > 0]
                trials.append(0.0)
                if current == 0:
                    trials.extend(
                        floor * factor for factor in (1.0, 10.0, 100.0, 1000.0)
                    )
                for trial in trials:
                    if trial == current:
                        continue
                    candidate_beta = beta.copy()
                    candidate_beta[j] = trial
                    loss = selection_loss(candidates, matrix @ candidate_beta)
                    if loss < best - 1e-9:
                        best, beta, improved = loss, candidate_beta, True
            if not improved:
                break
        return best, beta

    forward = list(range(len(start)))
    results = [
        search(start * scale, order)
        for scale in (1.0, 0.5, 2.0)
        for order in (forward, forward[::-1])
    ]
    # Lowest loss wins; ties prefer the least-changed start (first in order).
    return min(results, key=lambda item: item[0])[1]


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
    parser.add_argument(
        "--rank-rounds",
        type=int,
        default=0,
        help="rank-focused reweighting rounds after the initial least squares",
    )
    parser.add_argument(
        "--objective",
        choices=("lsq", "regret"),
        default="regret",
        help="lsq: weighted non-negative least squares; regret: refine it by direct regret minimization",
    )
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="recompute candidate features from the reproducible workloads first",
    )
    parser.add_argument(
        "--selector-check",
        action="store_true",
        help="run the shipped production selector on every measured cell and report its choice",
    )
    arguments = parser.parse_args()
    if arguments.refresh_features:
        refresh_features(arguments.evidence)
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
    beta = fit_ranking(train, terms, rounds=arguments.rank_rounds)
    if arguments.objective == "regret":
        beta = fit_regret(train, terms, beta)
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
        # Production stores integer milli-microseconds per feature unit.
        integer = {
            name: int(round(value * 1_000)) for name, value in report["terms"].items()
        }
        report["integer_terms_milli_us"] = integer
        beta_int = np.asarray(
            [integer[name] / 1_000.0 for name in terms], dtype=np.float64
        )
        report["integer_all"] = evaluate(
            candidates, predict(candidates, terms, beta_int)
        )
        print("COEFFICIENTS_MILLI_US =", json.dumps(integer, indent=4))
    report["production"] = production_regret(candidates, arguments.evidence)
    if report["production"]:
        regrets = sorted(v["regret_pct"] for v in report["production"].values())
        print(
            f"production   cells={len(regrets):3d} (timed automatic selection) regret "
            f"median={statistics.median(regrets):.2f}% max={regrets[-1]:.2f}%"
        )
    if arguments.selector_check:
        report["selector_check"] = selector_check(candidates)
        unmeasured = [
            c for c, v in report["selector_check"].items() if v["regret_pct"] is None
        ]
        regrets = [
            v["regret_pct"]
            for v in report["selector_check"].values()
            if v["regret_pct"] is not None
        ]
        print(
            f"selector     cells={len(report['selector_check']):3d} shipped-table selection regret "
            f"median={statistics.median(regrets) if regrets else float('nan'):.2f}% "
            f"max={max(regrets) if regrets else float('nan'):.2f}% unmeasured_nonuniform={len(unmeasured)}"
        )
        for cell in unmeasured:
            print("   nonuniform selection not in measured family:", cell)
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
