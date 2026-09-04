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

Terms are the production module's integer term functions (``TERMS``); the fit
is a non-negative least squares over their coefficients in microseconds per
unit, refined by direct regret minimization.  ``--integerize`` prints the frozen
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
from collections.abc import Callable
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
# Evidence may carry features from an older, wider extractor; only the fields the
# current LayoutFeatures defines are kept (and compared).
FEATURE_FIELDS = tuple(LayoutFeatures.__dataclass_fields__)


def _project(features: dict[str, Any]) -> dict[str, Any]:
    return {
        key: tuple(value) if isinstance(value, (list, tuple)) else value
        for key, value in features.items()
        if key in FEATURE_FIELDS
    }


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
    # Grouping keys for per-model / per-shape reports (not fitted).
    model: str = ""
    shape: tuple[int, int, int, int] = (1, 1, 1, 1)
    # Structure of the CP plan the layout produces (wave_count,
    # max_rank_tokens, summary_ms) and the version-1 score: what the two-stage
    # re-ranker prices and what it is compared against. Absent in older rows.
    cp_plan: dict[str, float] | None = None
    fallback_work: int | None = None


def _shape(row: dict[str, Any]) -> tuple[int, int, int, int]:
    """(tp, cp, ep, etp); expert parallelism defaults to 1 for older rows."""

    return (
        int(row.get("tp", 1)),
        int(row.get("cp", 1)),
        int(row.get("ep", 1) or 1),
        int(row.get("etp", 1) or 1),
    )


def _cell_key(row: dict[str, Any]) -> str:
    tp, cp, ep, etp = _shape(row)
    return (
        f"{row['cell']}|{row['model']}|L{row['layers']}|tp{tp}|cp{cp}"
        + (f"|ep{ep}" if ep > 1 else "")
        + (f"|etp{etp}" if etp > 1 else "")
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


def validate_completeness(paths: list[Path], *, repeat: int) -> list[str]:
    """Every mandatory candidate of every cell must have ``repeat`` usable rows.

    The runners record failures, and the fitter silently skips thin candidates,
    so a failed cell or candidate could otherwise vanish while the reduced
    dataset still passes its gates. Returns the list of gaps (empty = complete).
    """

    expected: dict[str, list[str]] = {}
    usable: dict[tuple[str, str], int] = defaultdict(int)
    failed: set[tuple[str, str]] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = _cell_key(row) if "cell" in row else None
            if row.get("record_type") == "calibration_cell":
                expected[_cell_key(row)] = [
                    c["label"] for c in row["candidates"] if c["label"] != "automatic"
                ]
            elif row.get("record_type") == "calibration_sample" and key:
                label = str(row["candidate_label"])
                if row.get("admission_failed"):
                    failed.add((key, label))
                elif (
                    row.get("role") == "measured"
                    and row.get("subforward_count", 1) == 1
                    and all(s == "none" for s in row.get("compile_statuses", []))
                ):
                    usable[(key, label)] += 1
    gaps: list[str] = []
    for cell, labels in expected.items():
        for label in labels:
            if (cell, label) in failed:
                gaps.append(f"{cell}: {label} refused admission")
            elif usable[(cell, label)] < repeat:
                gaps.append(
                    f"{cell}: {label} has {usable[(cell, label)]} usable rows (< {repeat})"
                )
    return gaps


MANIFEST_SCHEMA = "art.dev.trainer_rank_cost_calibration_manifest.v1"
FINGERPRINT_FIELDS = (
    "source",
    "requests_sha256",
    "device",
    "param_dtype",
    "hidden_size",
    "geometry",
)


def _fingerprint(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(json.dumps(row.get(k), sort_keys=True) for k in FINGERPRINT_FIELDS)


def cell_fingerprints(paths: list[Path]) -> dict[str, set[tuple[Any, ...]]]:
    """Distinct execution fingerprints recorded per cell key across the evidence."""

    seen: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "calibration_cell":
                seen[_cell_key(row)].add(_fingerprint(row))
    return seen


def validate_manifest(
    paths: list[Path],
    manifest_path: Path,
    *,
    excluded: list[str],
    exact: bool = False,
) -> tuple[list[str], list[str]]:
    """Exact cell identities: every expected cell present unless excluded, no
    unexpected cells, exclusions listed in the manifest, and one execution
    fingerprint per cell. Returns (problems, resolved excluded keys)."""

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unknown manifest schema")
    expected = {cell["key"] for cell in manifest["cells"]}
    listed_exclusions = {cell["key"] for cell in manifest["excluded"]}
    present = cell_fingerprints(paths)
    problems: list[str] = []
    excluded_keys = {
        key
        for key in expected
        if (key in excluded if exact else any(p in key for p in excluded))
    }
    for key in sorted(excluded_keys - listed_exclusions):
        problems.append(f"excluded cell is not listed in the manifest: {key}")
    for key in sorted(expected - excluded_keys - set(present)):
        problems.append(f"expected cell missing from the evidence: {key}")
    for key in sorted(set(present) - expected):
        problems.append(f"unexpected cell in the evidence: {key}")
    for key, prints in sorted(present.items()):
        if len(prints) > 1 and key not in excluded_keys:
            problems.append(
                f"cell recorded with {len(prints)} different execution fingerprints "
                f"(source/workload/device/dtype/hidden): {key}"
            )
    return problems, sorted(excluded_keys)


CERTIFICATE_SCHEMA = "art.dev.trainer_rank_cost_calibration_certificate.v1"


def export_certificate(
    path: Path,
    candidates: list[Candidate],
    *,
    evidence: list[Path],
    arguments: dict[str, Any],
    integer_table: dict[str, int],
    report: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    table_id: str = "",
    reranked: list[Candidate] | None = None,
    reranker: dict[str, Any] | None = None,
) -> None:
    """Write the compact, reproducible record binding the table to its data.

    Per-cell candidate features, medians, counts, spreads and fingerprints
    (no tokens, no per-sample rows), the exact fit arguments, the integer table
    and its hash, and the headline metrics. ``--from-certificate`` re-fits from
    exactly this record.
    """

    import hashlib

    fingerprints: dict[str, dict[str, Any]] = {}
    for evidence_path in evidence:
        for line in evidence_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "calibration_cell":
                fingerprints[_cell_key(row)] = {
                    "requests_sha256": row.get("requests_sha256"),
                    "source": row.get("source"),
                    "workload": row.get("workload"),
                    "model": row.get("model"),
                    "device": row.get("device"),
                    "device_capability": row.get("device_capability"),
                    "device_memory_class": row.get("device_memory_class"),
                    "param_dtype": row.get("param_dtype"),
                    "hidden_size": row.get("hidden_size"),
                    "geometry": row.get("geometry"),
                    "shape": list(_shape(row)),
                }
    by_cell: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in [*candidates, *(reranked or [])]:
        by_cell[candidate.cell].append(candidate)
    cells = [
        {
            "cell": cell,
            "facts": members[0].facts,
            **fingerprints.get(cell, {}),
            "candidates": [
                {
                    "label": c.label,
                    "features": c.features,
                    "median_ms": c.ms,
                    "n": c.n,
                    "spread_pct": c.spread_pct,
                    **({"cp_plan": c.cp_plan} if c.cp_plan is not None else {}),
                    **(
                        {"fallback_work": c.fallback_work}
                        if c.fallback_work is not None
                        else {}
                    ),
                }
                for c in sorted(members, key=lambda c: c.label)
            ],
        }
        for cell, members in sorted(by_cell.items())
    ]
    table_hash = hashlib.sha256(
        json.dumps(integer_table, sort_keys=True).encode()
    ).hexdigest()
    envelope = {
        "device_names": sorted({str(c.get("device")) for c in cells}),
        "param_dtypes": sorted({str(c.get("param_dtype")) for c in cells}),
        "hidden_sizes": sorted({int(c.get("hidden_size") or 0) for c in cells}),
    }

    # The execution classes this table was measured on: exactly what the
    # production table admits (bound by tests/unit/test_planner_cost_certificate.py).
    def unique(values: Any) -> list[Any]:
        return [
            json.loads(v)
            for v in sorted({json.dumps(x, sort_keys=True) for x in values})
        ]

    admitted = {
        "device_classes": unique(
            {
                "capability": c.get("device_capability"),
                "memory_class": c.get("device_memory_class"),
            }
            for c in cells
        ),
        "param_dtypes": envelope["param_dtypes"],
        "geometries": unique(c.get("geometry") for c in cells),
        "shapes": unique(c.get("shape") for c in cells),
    }
    payload = {
        "schema": CERTIFICATE_SCHEMA,
        "table_id": table_id,
        "fit_arguments": arguments,
        "manifest": manifest,
        "measured_envelope": envelope,
        "admitted": admitted,
        "cells": cells,
        "integer_table_milli_us": integer_table,
        "integer_table_sha256": table_hash,
        # Two-stage selection on the re-ranked shapes (None when the table
        # scores every admitted shape directly).
        "reranker": reranker,
        "metrics": {
            split: {
                k: report[split][k]
                for k in (
                    "cells",
                    "ordered_pairs",
                    "pairwise_accuracy",
                    "median_regret_pct",
                    "p95_regret_pct",
                    "max_regret_pct",
                    "clear_misses",
                )
            }
            for split in ("integer_all", "test", "train")
            if report.get(split)
        },
    }
    path.write_text(_compact_json(payload) + "\n")


def _compact_json(payload: dict[str, Any]) -> str:
    """Pretty top level, one line per cell and per list entry: diff-friendly
    without pretty-printing every feature of every candidate."""

    def line(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    parts = []
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, list):
            body = ",\n".join("  " + line(item) for item in value)
            parts.append(f'"{key}": [\n{body}\n ]')
        elif isinstance(value, dict) and key in (
            "metrics",
            "fit_arguments",
            "manifest",
        ):
            body = ",\n".join(
                f"  {json.dumps(k)}: {line(v)}" for k, v in sorted(value.items())
            )
            parts.append(f'"{key}": {{\n{body}\n }}')
        else:
            parts.append(f'"{key}": {line(value)}')
    return "{\n " + ",\n ".join(parts) + "\n}"


def load_certificate(path: Path) -> tuple[list[Candidate], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != CERTIFICATE_SCHEMA:
        raise ValueError("unknown certificate schema")
    candidates = [
        Candidate(
            cell["cell"],
            c["label"],
            _project(c["features"]),
            cell["facts"],
            float(c["median_ms"]),
            int(c["n"]),
            float(c["spread_pct"]),
            str(cell.get("model") or cell["cell"].split("|")[1]),
            tuple(cell.get("shape") or _shape(cell["facts"])),
            c.get("cp_plan"),
            c.get("fallback_work"),
        )
        for cell in payload["cells"]
        for c in cell["candidates"]
    ]
    return sorted(candidates, key=lambda c: (c.cell, c.label)), payload


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
        shape = _shape(cell)
        facts = {
            "layers": float(cell["layers"]),
            "gdn_layers": float(cell["gdn_layers"]),
            "tp": float(shape[0]),
            "cp": float(shape[1]),
            "ep": float(shape[2]),
            "etp": float(shape[3]),
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
                    _project(candidate["features"]),
                    facts,
                    median,
                    len(values),
                    spread,
                    str(cell["model"]),
                    shape,
                    candidate.get("cp_plan"),
                    candidate.get("fallback_work"),
                )
            )
    # Canonical order: the regret refinement is a deterministic coordinate
    # search whose path depends on candidate order, and equivalent optima
    # exist, so evidence-file order and certificate order must agree.
    return sorted(candidates, key=lambda c: (c.cell, c.label))


def evaluate_groups(
    candidates: list[Candidate],
    predicted_us: np.ndarray,
    key: Callable[[Candidate], str],
) -> dict[str, dict[str, Any]]:
    """Headline metrics per group (model, parallel shape, workload family):
    gates are judged per group, never pooled, so many easy cells of one model
    cannot hide one poorly modeled architecture."""

    groups: dict[str, list[tuple[Candidate, float]]] = defaultdict(list)
    for candidate, value in zip(candidates, predicted_us, strict=True):
        groups[key(candidate)].append((candidate, float(value)))
    report: dict[str, dict[str, Any]] = {}
    for name, members in sorted(groups.items()):
        block = evaluate([m[0] for m in members], np.asarray([m[1] for m in members]))
        report[name] = {
            k: block[k]
            for k in (
                "cells",
                "ordered_pairs",
                "pairwise_accuracy",
                "median_regret_pct",
                "p95_regret_pct",
                "max_regret_pct",
                "clear_misses",
            )
        }
        report[name]["gate_problems"] = gates_pass(block)
    return report


def _shape_label(candidate: Candidate) -> str:
    tp, cp, ep, etp = candidate.shape
    return (
        f"tp{tp}cp{cp}"
        + (f"ep{ep}" if ep > 1 else "")
        + (f"etp{etp}" if etp > 1 else "")
    )


def _family_label(candidate: Candidate) -> str:
    return candidate.cell.split("|")[0]


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
        features = _project(layout_features(selected).as_dict())
        best = min(members, key=lambda c: c.ms)
        match = next((c for c in members if _project(c.features) == features), None)
        report[cell] = {
            "selected": match.label if match else "nonuniform (unmeasured)",
            "selected_packed_tokens": features["packed_tokens"],
            "best": best.label,
            "regret_pct": ((match.ms - best.ms) / best.ms * 100.0) if match else None,
        }
    return report


def _cp_plan_value(candidate: Candidate, key: str) -> float:
    if candidate.cp_plan is None:
        raise ValueError(
            f"{candidate.cell} {candidate.label}: no cp_plan recorded (re-run the "
            "harness or backfill the evidence before fitting a re-ranker)"
        )
    return float(candidate.cp_plan[key])


# The re-ranker's terms, in the units ReRanker.score multiplies by layers:
# remote wave count and largest per-rank token load of the CP plan.
RERANK_TERMS: dict[str, Callable[[Candidate], float]] = {
    "wave_per_layer": lambda c: (
        float(c.facts["layers"]) * _cp_plan_value(c, "wave_count")
    ),
    "max_rank_token_per_layer": lambda c: (
        float(c.facts["layers"]) * _cp_plan_value(c, "max_rank_tokens")
    ),
}


def term_matrix(candidates: list[Candidate], terms: tuple[str, ...]) -> np.ndarray:
    """Term values in feature units (the integer functions divided by
    WORK_PER_US; the re-ranker's terms in their own units)."""

    rows = []
    for candidate in candidates:
        features = LayoutFeatures(
            packed_tokens=int(candidate.features["packed_tokens"]),
            segment_count=int(candidate.features["segment_count"]),
            max_depth=int(candidate.features["max_depth"]),
            segments_below=tuple(candidate.features.get("segments_below", ())),
        )
        facts = ScoringFacts(
            cp_size=int(candidate.facts["cp"]),
            tp_size=int(candidate.facts["tp"]),
            layers=int(candidate.facts["layers"]),
            gdn_layers=int(candidate.facts["gdn_layers"]),
        )
        rows.append(
            [
                RERANK_TERMS[name](candidate)
                if name in RERANK_TERMS
                else TERMS[name](features, facts) / WORK_PER_US
                for name in terms
            ]
        )
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


def integerize(
    candidates: list[Candidate],
    terms: tuple[str, ...],
    beta: np.ndarray,
    *,
    sweeps: int = 12,
) -> dict[str, int]:
    """The production table: integer milli-microseconds per unit, repaired on
    the integer grid when rounding costs a ranking.

    Rounding the float optimum can flip close pairs (equivalent optima with
    tiny per-token coefficients are especially fragile). When the rounded
    table's selection loss on the training cells is no worse than the float
    optimum's, it is returned as is; otherwise a deterministic coordinate
    search over integer values continues minimizing the same loss from the
    rounded table, accepting only strict improvements.
    """

    matrix = term_matrix(candidates, terms)
    table = np.asarray([int(round(value * 1_000)) for value in beta], dtype=np.int64)

    def loss(values: np.ndarray) -> float:
        return selection_loss(candidates, matrix @ (values / 1_000.0))

    best = loss(table)
    if best <= selection_loss(candidates, matrix @ beta) + 1e-9:
        return {name: int(value) for name, value in zip(terms, table, strict=True)}
    for _ in range(sweeps):
        improved = False
        for j in range(len(table)):
            current = int(table[j])
            trials = {
                current + 1,
                max(0, current - 1),
                current + 2,
                max(0, current - 2),
                0,
            }
            if current > 0:
                trials |= {
                    int(round(current * factor)) for factor in (0.5, 0.9, 1.1, 2.0)
                }
            else:
                trials |= {1, 10, 100, 1_000}
            for trial in sorted(trials):
                if trial == current or trial < 0:
                    continue
                candidate_table = table.copy()
                candidate_table[j] = trial
                candidate_loss = loss(candidate_table)
                if candidate_loss < best - 1e-9:
                    best, table, improved = candidate_loss, candidate_table, True
        if not improved:
            break
    return {name: int(value) for name, value in zip(terms, table, strict=True)}


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


RERANK_TERM_NAMES = ("wave_per_layer", "max_rank_token_per_layer")


def two_stage_shortlist(
    members: list[Candidate],
    shortlist_us: np.ndarray,
    *,
    shortlist_size: int,
    incumbent: str,
) -> list[Candidate]:
    """The cheapest ``shortlist_size`` layouts of a cell under the shortlist
    score plus the incumbent anchor, in cheap order (production's rule)."""

    ranked = [
        member
        for _score, _label, member in sorted(
            zip(shortlist_us, [m.label for m in members], members, strict=True),
            key=lambda item: (item[0], item[1]),
        )
    ]
    shortlist = ranked[: max(1, shortlist_size)]
    shortlist += [m for m in ranked if m.label == incumbent and m not in shortlist]
    return shortlist


def evaluate_two_stage(
    candidates: list[Candidate],
    *,
    terms: tuple[str, ...],
    shortlist_beta: np.ndarray,
    rerank_beta: np.ndarray,
    shortlist_size: int,
    incumbent: str,
) -> dict[str, Any]:
    """Two-stage selection per cell: shortlist by the ten-term score, select
    the lowest re-ranker score (ties keep the cheaper layout). Reports the
    ranking gates over the shortlist, shortlist recall and its irrecoverable
    regret, non-regression against the cheap-only and version-1 selections,
    and the synchronous planning cost of pricing the shortlist."""

    by_cell: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_cell[candidate.cell].append(candidate)
    regrets: list[float] = []
    clear_misses: list[str] = []
    ordered_pairs = 0
    ordered_correct = 0
    recall = {
        "cells": 0,
        "best_in_shortlist": 0,
        "clear_winner_cells": 0,
        "clear_winners_in_shortlist": 0,
        "max_irrecoverable_pct": 0.0,
    }
    regression: dict[str, Any] = {
        "worse_than_cheap_cells": [],
        "worse_than_fallback_cells": [],
        "fallback_available": True,
        "savings_vs_cheap_pct": [],
        "savings_vs_fallback_pct": [],
    }
    economics: list[tuple[float, float, str]] = []
    per_cell: dict[str, dict[str, Any]] = {}
    for cell, members in sorted(by_cell.items()):
        shortlist_us = predict(members, terms, shortlist_beta)
        shortlist = two_stage_shortlist(
            members, shortlist_us, shortlist_size=shortlist_size, incumbent=incumbent
        )
        rerank_us = dict(
            zip(
                [m.label for m in shortlist],
                predict(shortlist, RERANK_TERM_NAMES, rerank_beta),
                strict=True,
            )
        )
        selected = min(
            enumerate(shortlist), key=lambda item: (rerank_us[item[1].label], item[0])
        )[1]
        best = min(members, key=lambda m: m.ms)
        regret = (selected.ms - best.ms) / best.ms * 100.0
        regrets.append(regret)
        by_ms = sorted(members, key=lambda m: m.ms)
        lead_pct = (by_ms[1].ms - best.ms) / best.ms * 100.0 if len(by_ms) > 1 else 0.0
        clear = lead_pct > max(best.spread_pct, NOISE_BAND_PCT)
        if clear and regret > 5.0:
            clear_misses.append(cell)
        recall["cells"] += 1
        recall["best_in_shortlist"] += best in shortlist
        recall["clear_winner_cells"] += clear
        recall["clear_winners_in_shortlist"] += clear and best in shortlist
        irrecoverable = (min(m.ms for m in shortlist) - best.ms) / best.ms * 100.0
        recall["max_irrecoverable_pct"] = max(
            recall["max_irrecoverable_pct"], irrecoverable
        )
        for a, b in itertools.combinations(shortlist, 2):
            separation = abs(a.ms - b.ms) / min(a.ms, b.ms) * 100.0
            if separation <= 3.0:
                continue
            ordered_pairs += 1
            ordered_correct += (a.ms < b.ms) == (
                rerank_us[a.label] < rerank_us[b.label]
            )
        cheap = shortlist[0]
        cheap_regret = (cheap.ms - best.ms) / best.ms * 100.0
        regression["savings_vs_cheap_pct"].append(cheap_regret - regret)
        if regret - cheap_regret > 2.0:
            regression["worse_than_cheap_cells"].append(cell)
        if all(m.fallback_work is not None for m in members):
            fallback = min(members, key=lambda m: (m.fallback_work or 0, m.label))
            fallback_regret = (fallback.ms - best.ms) / best.ms * 100.0
            regression["savings_vs_fallback_pct"].append(fallback_regret - regret)
            if regret - fallback_regret > 2.0:
                regression["worse_than_fallback_cells"].append(cell)
        else:
            regression["fallback_available"] = False
            fallback_regret = None
        summary_ms = [
            float((m.cp_plan or {}).get("summary_ms", 0.0)) for m in shortlist
        ]
        planning_ms = sum(summary_ms) - min(
            summary_ms
        )  # the selected plan is built anyway
        economics.append((planning_ms / best.ms * 100.0, planning_ms, cell))
        per_cell[cell] = {
            "selected": selected.label,
            "best": best.label,
            "regret_pct": regret,
            "best_lead_pct": lead_pct,
            "shortlist": [m.label for m in shortlist],
            "cheap_regret_pct": cheap_regret,
            "fallback_regret_pct": fallback_regret,
            "irrecoverable_pct": irrecoverable,
            "planning_ms": planning_ms,
        }
    regrets_sorted = sorted(regrets)
    ratios = sorted(economics)
    for key in ("savings_vs_cheap_pct", "savings_vs_fallback_pct"):
        values = regression.pop(key)
        regression[key.replace("_pct", "_mean_pct")] = (
            statistics.mean(values) if values else 0.0
        )
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
        "recall": recall,
        "non_regression": regression,
        "planning_cost": {
            "mean_pct_of_cell": statistics.mean(r[0] for r in ratios)
            if ratios
            else 0.0,
            "median_pct_of_cell": ratios[len(ratios) // 2][0] if ratios else 0.0,
            "max_pct_of_cell": ratios[-1][0] if ratios else 0.0,
            "max_ms": ratios[-1][1] if ratios else 0.0,
            "max_cell": ratios[-1][2] if ratios else "",
        },
        "per_cell": per_cell,
    }


def two_stage_gates(report: dict[str, Any]) -> list[str]:
    problems = gates_pass(report)
    recall = report["recall"]
    if recall["clear_winners_in_shortlist"] != recall["clear_winner_cells"]:
        problems.append(
            f"shortlist misses clear winners: {recall['clear_winners_in_shortlist']}/{recall['clear_winner_cells']}"
        )
    regression = report["non_regression"]
    if regression["worse_than_cheap_cells"]:
        problems.append(
            f"worse than the cheap selection by >2%: {regression['worse_than_cheap_cells']}"
        )
    if regression["worse_than_fallback_cells"]:
        problems.append(
            f"worse than the version-1 selection by >2%: {regression['worse_than_fallback_cells']}"
        )
    # A synchronous miss pays for the shortlist's plans; on the re-ranked
    # cells the execution saved over both single-stage selections must cover it.
    cost = report["planning_cost"]["mean_pct_of_cell"]
    savings = max(
        regression["savings_vs_cheap_mean_pct"],
        regression["savings_vs_fallback_mean_pct"],
    )
    if cost > savings:
        problems.append(
            f"planning cost {cost:.1f}% of cell time exceeds the mean saving {savings:.1f}%"
        )
    return problems


def fit_reranker(
    candidates: list[Candidate],
    reranked: list[Candidate],
    *,
    terms: tuple[str, ...],
    held_out: Callable[[str], bool],
    shortlist_size: int,
    incumbent: str,
    objective: str,
) -> dict[str, Any]:
    """Fit and evaluate the two-stage selector for the re-ranked shapes.

    The shortlist score is the ten-term table fitted on every measured shape
    (its job is recall, not selection); the re-ranker is fitted on paired
    deltas within the training cells' shortlists and refined on their regret,
    then integerized like a table.
    """

    shortlist_train = [c for c in candidates if not held_out(c.cell)]
    shortlist_beta = nnls(*paired_deltas(shortlist_train, terms))
    if objective == "regret":
        shortlist_beta = fit_regret(shortlist_train, terms, shortlist_beta)
    shortlist_table = integerize(shortlist_train, terms, shortlist_beta)
    shortlist_beta_int = np.asarray([shortlist_table[name] / 1_000.0 for name in terms])

    by_cell: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in reranked:
        by_cell[candidate.cell].append(candidate)
    train_shortlists: list[Candidate] = []
    for cell, members in sorted(by_cell.items()):
        if held_out(cell):
            continue
        train_shortlists += two_stage_shortlist(
            members,
            predict(members, terms, shortlist_beta_int),
            shortlist_size=shortlist_size,
            incumbent=incumbent,
        )
    rerank_beta = nnls(*paired_deltas(train_shortlists, RERANK_TERM_NAMES))
    if objective == "regret":
        rerank_beta = fit_regret(train_shortlists, RERANK_TERM_NAMES, rerank_beta)
    rerank_table = integerize(train_shortlists, RERANK_TERM_NAMES, rerank_beta)
    rerank_beta_int = np.asarray(
        [rerank_table[name] / 1_000.0 for name in RERANK_TERM_NAMES]
    )

    def two_stage(subset: list[Candidate]) -> dict[str, Any]:
        return evaluate_two_stage(
            subset,
            terms=terms,
            shortlist_beta=shortlist_beta_int,
            rerank_beta=rerank_beta_int,
            shortlist_size=shortlist_size,
            incumbent=incumbent,
        )

    held = [c for c in reranked if held_out(c.cell)]
    shapes = sorted({_shape_label(c) for c in reranked})
    return {
        "shortlist_size": shortlist_size,
        "incumbent": incumbent,
        "shapes": shapes,
        "shortlist_table_milli_us": shortlist_table,
        "wave_per_layer_milli_us": rerank_table["wave_per_layer"],
        "max_rank_token_per_layer_milli_us": rerank_table["max_rank_token_per_layer"],
        "metrics": {
            "all": two_stage(reranked),
            "held_out": two_stage(held) if held else None,
            "by_shape": {
                shape: two_stage([c for c in reranked if _shape_label(c) == shape])
                for shape in shapes
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="*", type=Path)
    parser.add_argument(
        "--from-certificate",
        default="",
        help="fit from a checked-in certificate's aggregates instead of raw evidence",
    )
    parser.add_argument(
        "--export-certificate",
        default="",
        help="write the compact reproducible certificate (aggregates, arguments, table, hash)",
    )
    parser.add_argument(
        "--exclude-cells",
        default="",
        help=(
            "comma-separated substrings of cells to leave out explicitly (recorded in the "
            "certificate), or @manifest for exactly the manifest's listed exclusions; "
            "the fitter never drops incomplete cells silently"
        ),
    )
    parser.add_argument(
        "--manifest",
        default="",
        help=(
            "expected-cell manifest (JSON): every listed cell must be present unless "
            "excluded, no unexpected cells, one execution fingerprint per cell"
        ),
    )
    parser.add_argument(
        "--require-complete",
        type=int,
        default=0,
        help="fail unless every mandatory candidate of every cell has at least this many usable rows",
    )
    parser.add_argument("--terms", default=",".join(DEFAULT_TERMS))
    parser.add_argument(
        "--holdout",
        default="",
        help="comma-separated substrings selecting held-out cells (odd Ellavox groups are always held out)",
    )
    parser.add_argument("--report", default="", help="write the JSON report here")
    parser.add_argument(
        "--table-id",
        default="",
        help="identity of the production table this certificate binds (required with --export-certificate)",
    )
    parser.add_argument("--integerize", action="store_true")
    parser.add_argument(
        "--rerank-shapes",
        default="",
        help=(
            "comma-separated shape labels (e.g. tp1cp4) selected by the two-stage "
            "re-ranker: the ten-term score shortlists, the CP plan structure decides; "
            "their cells leave the direct fit and need cp_plan in the evidence"
        ),
    )
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--incumbent", default="depth_one")
    parser.add_argument(
        "--objective",
        choices=("lsq", "regret"),
        default="regret",
        help="lsq: weighted non-negative least squares; regret: refine it by direct regret minimization",
    )
    parser.add_argument(
        "--selector-check",
        action="store_true",
        help="run the shipped production selector on every measured cell and report its choice",
    )
    arguments = parser.parse_args()
    terms = tuple(t for t in arguments.terms.split(",") if t)
    if arguments.exclude_cells == "@manifest":
        # Exactly the manifest's listed exclusions (each carries its reason).
        if not arguments.manifest:
            raise SystemExit("--exclude-cells @manifest requires --manifest")
        manifest_excluded = json.loads(Path(arguments.manifest).read_text())["excluded"]
        excluded = [cell["key"] for cell in manifest_excluded]
        exact = True
    else:
        excluded = [p for p in arguments.exclude_cells.split(",") if p]
        exact = False

    def excluded_cell(cell: str) -> bool:
        return cell in excluded if exact else any(p in cell for p in excluded)

    manifest_record: dict[str, Any] | None = None
    if arguments.manifest:
        problems, excluded_keys = validate_manifest(
            arguments.evidence,
            Path(arguments.manifest),
            excluded=excluded,
            exact=exact,
        )
        if problems:
            print("manifest validation failed:", file=sys.stderr)
            for problem in problems:
                print("  -", problem, file=sys.stderr)
            raise SystemExit(1)
        manifest_payload = json.loads(Path(arguments.manifest).read_text())
        manifest_record = {
            "path": Path(arguments.manifest).name,
            "expected": [cell["key"] for cell in manifest_payload["cells"]],
            "excluded": excluded_keys,
        }
        print(
            f"manifest ok: {len(manifest_record['expected'])} expected cells, "
            f"{len(excluded_keys)} excluded, identities and fingerprints consistent"
        )
    if arguments.require_complete:
        gaps = [
            gap
            for gap in validate_completeness(
                arguments.evidence, repeat=arguments.require_complete
            )
            if not excluded_cell(gap.split(":", 1)[0])
        ]
        if gaps:
            print("incomplete evidence:", file=sys.stderr)
            for gap in gaps:
                print("  -", gap, file=sys.stderr)
            raise SystemExit(1)
        print(
            f"evidence complete: every mandatory candidate has >= {arguments.require_complete} rows"
        )
    if arguments.from_certificate:
        candidates, _certificate = load_certificate(Path(arguments.from_certificate))
    else:
        candidates = load_candidates(arguments.evidence)
    if excluded:
        dropped = sorted({c.cell for c in candidates if excluded_cell(c.cell)})
        candidates = [c for c in candidates if not excluded_cell(c.cell)]
        print("excluded cells:", *dropped, sep="\n  ")
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

    # Cells on re-ranked shapes are selected by the two-stage selector; the
    # direct table is fitted and gated on the other shapes only.
    rerank_shapes = {label for label in arguments.rerank_shapes.split(",") if label}
    reranked = [c for c in candidates if _shape_label(c) in rerank_shapes]
    all_candidates = candidates
    if rerank_shapes:
        if not reranked:
            raise SystemExit(
                f"no cells on the re-ranked shapes {sorted(rerank_shapes)}"
            )
        candidates = [c for c in candidates if _shape_label(c) not in rerank_shapes]
        print(
            f"re-ranked shapes {sorted(rerank_shapes)}: {len({c.cell for c in reranked})} cells "
            f"leave the direct fit ({len({c.cell for c in candidates})} direct cells)"
        )
    train = [c for c in candidates if not held_out(c.cell)]
    test = [c for c in candidates if held_out(c.cell)]
    beta = nnls(*paired_deltas(train, terms))
    if arguments.objective == "regret":
        beta = fit_regret(train, terms, beta)
    report: dict[str, Any] = {
        "terms": dict(zip(terms, [float(b) for b in beta], strict=True)),
        "train_cells": len({c.cell for c in train}),
        "test_cells": len({c.cell for c in test}),
        "train": evaluate(train, predict(train, terms, beta)),
        "test": evaluate(test, predict(test, terms, beta)) if test else None,
        "fit_all": evaluate(candidates, predict(candidates, terms, beta)),
    }
    if arguments.integerize:
        # Production stores integer milli-microseconds per feature unit;
        # repaired on the integer grid (training cells) when rounding costs
        # a ranking.
        integer = integerize(train, terms, beta)
        report["integer_terms_milli_us"] = integer
        beta_int = np.asarray(
            [integer[name] / 1_000.0 for name in terms], dtype=np.float64
        )
        report["integer_all"] = evaluate(
            candidates, predict(candidates, terms, beta_int)
        )
        print("COEFFICIENTS_MILLI_US =", json.dumps(integer, indent=4))
    if reranked:
        if not arguments.integerize:
            raise SystemExit("--rerank-shapes requires --integerize")
        report["reranker"] = fit_reranker(
            all_candidates,
            reranked,
            terms=terms,
            held_out=held_out,
            shortlist_size=arguments.shortlist_size,
            incumbent=arguments.incumbent,
            objective=arguments.objective,
        )
        block = report["reranker"]
        print(
            "RERANKER =",
            json.dumps(
                {
                    k: block[k]
                    for k in (
                        "shortlist_size",
                        "incumbent",
                        "shapes",
                        "wave_per_layer_milli_us",
                        "max_rank_token_per_layer_milli_us",
                    )
                },
                indent=4,
            ),
        )
        print(
            "shortlist table (milli-us):", json.dumps(block["shortlist_table_milli_us"])
        )
        for name, metrics in (
            ("two-stage all", block["metrics"]["all"]),
            ("two-stage held-out", block["metrics"]["held_out"]),
            *(
                (f"two-stage {shape}", m)
                for shape, m in block["metrics"]["by_shape"].items()
            ),
        ):
            if not metrics:
                continue
            recall = metrics["recall"]
            problems = two_stage_gates(metrics)
            print(
                f"{name:22s} cells={metrics['cells']:3d} acc={metrics['pairwise_accuracy']:.3f} regret median={metrics['median_regret_pct']:.2f}% "
                f"p95={metrics['p95_regret_pct']:.2f}% max={metrics['max_regret_pct']:.2f}% | recall {recall['best_in_shortlist']}/{recall['cells']} "
                f"clear {recall['clear_winners_in_shortlist']}/{recall['clear_winner_cells']} irrecoverable max {recall['max_irrecoverable_pct']:.1f}% | "
                f"planning mean {metrics['planning_cost']['mean_pct_of_cell']:.1f}% max {metrics['planning_cost']['max_pct_of_cell']:.1f}% of cell vs saving "
                f"{max(metrics['non_regression']['savings_vs_cheap_mean_pct'], metrics['non_regression']['savings_vs_fallback_mean_pct']):.1f}% | "
                + (
                    "GATES PASS"
                    if not problems
                    else "GATES FAIL: " + "; ".join(problems)
                )
            )
    report["production"] = (
        production_regret(all_candidates, arguments.evidence)
        if arguments.evidence
        else {}
    )
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
    for split in ("train", "test", "fit_all", "integer_all"):
        block = report.get(split)
        if not block:
            continue
        print(
            f"{split:12s} cells={block['cells']:3d} pairs={block['ordered_pairs']:4d} "
            f"acc={block['pairwise_accuracy']:.3f} regret median={block['median_regret_pct']:.2f}% "
            f"p95={block['p95_regret_pct']:.2f}% max={block['max_regret_pct']:.2f}% "
            f"clear_misses={len(block['clear_misses'])}"
        )
    final_beta = np.asarray(
        [
            report.get("integer_terms_milli_us", {}).get(name, 0) / 1_000.0
            for name in terms
        ]
        if "integer_terms_milli_us" in report
        else beta,
        dtype=np.float64,
    )
    predicted_all = predict(candidates, terms, final_beta)
    report["by_model"] = evaluate_groups(candidates, predicted_all, lambda c: c.model)
    report["by_shape"] = evaluate_groups(candidates, predicted_all, _shape_label)
    report["by_family"] = evaluate_groups(candidates, predicted_all, _family_label)
    for title, blocks in (("model", report["by_model"]), ("shape", report["by_shape"])):
        for name, block in blocks.items():
            print(
                f"by {title:6s} {name:28s} cells={block['cells']:3d} acc="
                f"{block['pairwise_accuracy']:.3f} regret median={block['median_regret_pct']:.2f}% "
                f"max={block['max_regret_pct']:.2f}%"
                + (
                    f" GATES: {'; '.join(block['gate_problems'])}"
                    if block["gate_problems"]
                    else ""
                )
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
    if arguments.export_certificate:
        if "integer_terms_milli_us" not in report:
            raise SystemExit("--export-certificate requires --integerize")
        if not arguments.table_id:
            raise SystemExit("--export-certificate requires --table-id")
        export_certificate(
            Path(arguments.export_certificate),
            candidates,
            evidence=arguments.evidence,
            arguments={
                "terms": list(terms),
                "objective": arguments.objective,
                "holdout": arguments.holdout,
                "exclude_cells": excluded,
                "require_complete": arguments.require_complete,
                "rerank_shapes": sorted(rerank_shapes),
                "shortlist_size": arguments.shortlist_size,
                "incumbent": arguments.incumbent,
            },
            integer_table=report["integer_terms_milli_us"],
            report=report,
            manifest=manifest_record,
            table_id=arguments.table_id,
            reranked=reranked,
            reranker=report.get("reranker"),
        )
        print("certificate written:", arguments.export_certificate)


if __name__ == "__main__":
    main()
