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

from art.trainer_rank._planner_cost import CALIBRATION_PROFILE, COEFFICIENTS_MILLI_US

_ROOT = Path(__file__).resolve().parents[2]
_CERTIFICATE = _ROOT / "dev" / "trainer_rank_cost_calibration_certificate.json"
_MANIFEST = _ROOT / "dev" / "trainer_rank_cost_calibration_manifest.json"
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
    assert report["cells"] == recorded["cells"] == len(payload["cells"])
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
    beta = fit.nnls(*fit.paired_deltas(train, terms))
    if arguments["objective"] == "regret":
        beta = fit.fit_regret(train, terms, beta)
    table = {
        name: int(round(value * 1_000)) for name, value in zip(terms, beta, strict=True)
    }
    assert table == payload["integer_table_milli_us"]


def test_certificate_holds_exactly_the_manifest_cells() -> None:
    """Exact cell identities: every expected cell minus the explicit exclusions."""

    payload = json.loads(_CERTIFICATE.read_text())
    manifest = json.loads(_MANIFEST.read_text())
    expected = {cell["key"] for cell in manifest["cells"]}
    excluded = {cell["key"] for cell in manifest["excluded"]}
    assert excluded <= expected
    certified = {cell["cell"] for cell in payload["cells"]}
    assert certified == expected - excluded
    assert len(certified) == 56 and len(excluded) == 2
    assert set(payload["manifest"]["excluded"]) == excluded
    assert payload["manifest"]["path"] == _MANIFEST.name
    # Every retained cell carries its execution fingerprints.
    for cell in payload["cells"]:
        for key in (
            "source",
            "requests_sha256",
            "device",
            "param_dtype",
            "hidden_size",
        ):
            assert cell.get(key) not in (None, ""), (cell["cell"], key)


def test_profile_envelope_is_bound_to_the_certified_evidence() -> None:
    """The admitted domain is exactly what the certificate measured."""

    payload = json.loads(_CERTIFICATE.read_text())
    devices = {cell["device"] for cell in payload["cells"]}
    dtypes = {cell["param_dtype"] for cell in payload["cells"]}
    hidden = {int(cell["hidden_size"]) for cell in payload["cells"]}
    assert set(CALIBRATION_PROFILE.measured_device_names) == devices
    assert set(CALIBRATION_PROFILE.param_dtypes) == dtypes
    assert set(CALIBRATION_PROFILE.hidden_sizes) == hidden
    assert payload["measured_envelope"] == {
        "device_names": sorted(devices),
        "param_dtypes": sorted(dtypes),
        "hidden_sizes": sorted(hidden),
    }
    assert not CALIBRATION_PROFILE.allow_moe
    assert all(float(cell["facts"]["tp"]) in (1.0, 2.0) for cell in payload["cells"])
    assert all(
        float(cell["facts"]["cp"]) in (1.0, 2.0, 4.0) for cell in payload["cells"]
    )
