"""The checked-in calibration certificate binds the production table to its data.

The certificate carries every cell's candidate features, median timings,
counts and fingerprints (no tokens), the exact fit arguments, the integer table
and its hash. This test verifies (cheaply) that the shipped table is the
certificate's table and that the certificate's headline metrics are what the
table produces on the recorded aggregates. Set ``ART_COST_CERTIFICATE_REFIT=1``
to also re-run the full fit from the aggregates and require the identical
integer table (slower: about a minute).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from art.trainer_rank._planner_cost import COEFFICIENTS_MILLI_US

_ROOT = Path(__file__).resolve().parents[2]
_CERTIFICATE = _ROOT / "dev" / "trainer_rank_cost_calibration_certificate.json"
_spec = importlib.util.spec_from_file_location(
    "trainer_rank_cost_fit", _ROOT / "dev" / "trainer_rank_cost_fit.py"
)
assert _spec is not None and _spec.loader is not None
fit = importlib.util.module_from_spec(_spec)
sys.modules["trainer_rank_cost_fit"] = fit
_spec.loader.exec_module(fit)


def test_shipped_table_is_the_certified_table() -> None:
    payload = json.loads(_CERTIFICATE.read_text())
    assert payload["schema"] == fit.CERTIFICATE_SCHEMA
    assert payload["integer_table_milli_us"] == COEFFICIENTS_MILLI_US
    digest = hashlib.sha256(
        json.dumps(payload["integer_table_milli_us"], sort_keys=True).encode()
    ).hexdigest()
    assert digest == payload["integer_table_sha256"]


def test_certified_metrics_hold_on_the_recorded_aggregates() -> None:
    candidates, payload = fit.load_certificate(_CERTIFICATE)
    terms = tuple(payload["fit_arguments"]["terms"])
    beta = np.asarray([COEFFICIENTS_MILLI_US[name] / 1_000.0 for name in terms])
    report = fit.evaluate(candidates, fit.predict(candidates, terms, beta))
    recorded = payload["metrics"]["integer_all"]
    assert report["cells"] == recorded["cells"] >= 50
    assert report["ordered_pairs"] == recorded["ordered_pairs"]
    assert abs(report["pairwise_accuracy"] - recorded["pairwise_accuracy"]) < 1e-9
    assert abs(report["max_regret_pct"] - recorded["max_regret_pct"]) < 1e-9
    assert report["max_regret_pct"] <= 5.0
    assert report["pairwise_accuracy"] >= 0.95
    assert not report["clear_misses"]


def test_full_refit_reproduces_the_table_when_requested() -> None:
    if os.environ.get("ART_COST_CERTIFICATE_REFIT") != "1":
        return
    candidates, payload = fit.load_certificate(_CERTIFICATE)
    arguments = payload["fit_arguments"]
    terms = tuple(arguments["terms"])

    def held_out(cell: str) -> bool:
        patterns = [p for p in arguments["holdout"].split(",") if p]
        if any(p in cell for p in patterns):
            return True
        return (
            "cal-ellavox" in cell
            and "|g" in cell
            and int(cell.rsplit("|g", 1)[1]) % 2 == 1
        )

    train = [c for c in candidates if not held_out(c.cell)]
    beta = fit.fit_ranking(train, terms, rounds=arguments["rank_rounds"])
    if arguments["objective"] == "regret":
        beta = fit.fit_regret(train, terms, beta)
    table = {
        name: int(round(value * 1_000)) for name, value in zip(terms, beta, strict=True)
    }
    assert table == payload["integer_table_milli_us"]
