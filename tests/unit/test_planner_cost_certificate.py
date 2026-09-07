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

from art.trainer_rank._planner_cost import (
    CALIBRATED_TABLES,
    COEFFICIENTS_MILLI_US,
    DENSE_H2560_TABLE,
    DeviceClass,
    ModelGeometry,
    ParallelShape,
)

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
    assert payload["table_id"] == DENSE_H2560_TABLE.table_id
    assert payload["integer_table_milli_us"] == COEFFICIENTS_MILLI_US
    assert DENSE_H2560_TABLE.coefficients_milli_us == COEFFICIENTS_MILLI_US
    assert [table.table_id for table in CALIBRATED_TABLES] == [payload["table_id"]]
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
    """Exact cell identities: every expected cell minus the explicit exclusions
    (none since the two Ellavox CP4 cells were re-measured after issue #840)."""

    payload = json.loads(_CERTIFICATE.read_text())
    manifest = json.loads(_MANIFEST.read_text())
    expected = {cell["key"] for cell in manifest["cells"]}
    excluded = {cell["key"] for cell in manifest["excluded"]}
    assert excluded <= expected
    certified = {cell["cell"] for cell in payload["cells"]}
    assert certified == expected - excluded
    assert len(certified) == 58 and not excluded
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


def test_admitted_execution_classes_are_exactly_the_certified_ones() -> None:
    """The production table admits precisely the device classes, dtypes, model
    geometries and parallel shapes recorded in the certificate's cells."""

    payload = json.loads(_CERTIFICATE.read_text())
    cells = payload["cells"]
    for cell in cells:
        assert cell["geometry"] and cell["shape"] and cell["model"], cell["cell"]
        assert (
            cell["device_memory_class"]
            == DeviceClass.memory_class_for(
                None if cell.get("device_total_memory_bytes") is None else None
            )
            or cell["device_memory_class"] == "hbm-141g"
        )
    admitted = payload["admitted"]
    geometries = {ModelGeometry(**geometry) for geometry in admitted["geometries"]}
    assert geometries == set(DENSE_H2560_TABLE.geometries)
    assert geometries == {ModelGeometry(**cell["geometry"]) for cell in cells}
    shapes = {ParallelShape(*shape) for shape in admitted["shapes"]}
    assert shapes == set(DENSE_H2560_TABLE.shapes)
    assert shapes == {ParallelShape(*cell["shape"]) for cell in cells}
    devices = {
        DeviceClass(tuple(entry["capability"]), entry["memory_class"])
        for entry in admitted["device_classes"]
    }
    assert devices == set(DENSE_H2560_TABLE.device_classes)
    assert set(admitted["param_dtypes"]) == set(DENSE_H2560_TABLE.param_dtypes)
    # Every geometry was measured at every admitted shape (product admission).
    measured = {
        (ModelGeometry(**cell["geometry"]), ParallelShape(*cell["shape"]))
        for cell in cells
    }
    for geometry in DENSE_H2560_TABLE.geometries:
        for shape in DENSE_H2560_TABLE.shapes:
            assert (geometry, shape) in measured, (geometry.hidden_size, shape)
    # One geometry per model across all of its cells.
    by_model: dict[str, set[str]] = {}
    for cell in cells:
        by_model.setdefault(cell["model"], set()).add(
            json.dumps(cell["geometry"], sort_keys=True)
        )
    assert all(len(values) == 1 for values in by_model.values()), by_model
    assert payload["measured_envelope"] == {
        "device_names": sorted({cell["device"] for cell in cells}),
        "param_dtypes": sorted({cell["param_dtype"] for cell in cells}),
        "hidden_sizes": sorted({int(cell["hidden_size"]) for cell in cells}),
    }
