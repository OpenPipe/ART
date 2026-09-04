"""Each checked-in calibration certificate binds one production table to its data.

One certificate per calibrated table
(``dev/trainer_rank_cost_calibration_certificate_<table>.json``). A certificate
carries every cell's candidate features, median timings, counts and execution
fingerprints (no tokens), the exact fit arguments, the integer table and its
hash. These tests verify (cheaply) that each shipped table is its certificate's
table, that the certified headline metrics are what the table produces on the
recorded aggregates, and that the table admits exactly the execution classes
the certificate measured. Set ``ART_COST_CERTIFICATE_REFIT=1`` to also re-run
the full fit from the aggregates and require the identical integer table
(slower: about a minute per table).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from art.trainer_rank._planner_cost import (
    CALIBRATED_TABLES,
    COEFFICIENTS_MILLI_US,
    DENSE_H2560_TABLE,
    DeviceClass,
    ModelGeometry,
    ParallelShape,
)

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "trainer_rank_cost_fit", _ROOT / "dev" / "trainer_rank_cost_fit.py"
)
assert _spec is not None and _spec.loader is not None
fit = importlib.util.module_from_spec(_spec)
sys.modules["trainer_rank_cost_fit"] = fit
_spec.loader.exec_module(fit)


def certificate_path(table_id: str) -> Path:
    return _ROOT / "dev" / f"trainer_rank_cost_calibration_certificate_{table_id}.json"


def manifest_path(table_id: str) -> Path:
    return _ROOT / "dev" / f"trainer_rank_cost_calibration_manifest_{table_id}.json"


_TABLES = pytest.mark.parametrize("table", CALIBRATED_TABLES, ids=lambda t: t.table_id)


def _label(shape: ParallelShape) -> str:
    """The fitter's shape label (``_shape_label``): tp, cp, then ep/etp when > 1."""

    return (
        f"tp{shape.tp}cp{shape.cp}"
        + (f"ep{shape.ep}" if shape.ep > 1 else "")
        + (f"etp{shape.etp}" if shape.etp > 1 else "")
    )


def test_default_table_is_the_dense_one() -> None:
    assert DENSE_H2560_TABLE.coefficients_milli_us == COEFFICIENTS_MILLI_US
    assert len({table.table_id for table in CALIBRATED_TABLES}) == len(
        CALIBRATED_TABLES
    )


@_TABLES
def test_shipped_table_is_the_certified_table(table) -> None:
    payload = json.loads(certificate_path(table.table_id).read_text())
    assert payload["schema"] == fit.CERTIFICATE_SCHEMA
    assert payload["table_id"] == table.table_id
    assert payload["integer_table_milli_us"] == dict(table.coefficients_milli_us)
    digest = hashlib.sha256(
        json.dumps(payload["integer_table_milli_us"], sort_keys=True).encode()
    ).hexdigest()
    assert digest == payload["integer_table_sha256"]


@_TABLES
def test_certified_metrics_hold_on_the_recorded_aggregates(table) -> None:
    candidates, payload = fit.load_certificate(certificate_path(table.table_id))
    terms = tuple(payload["fit_arguments"]["terms"])
    beta = np.asarray(
        [table.coefficients_milli_us.get(name, 0) / 1_000.0 for name in terms]
    )
    reranked_labels = {_label(shape) for shape in table.reranked_shapes}
    candidates = [c for c in candidates if fit._shape_label(c) not in reranked_labels]
    report = fit.evaluate(candidates, fit.predict(candidates, terms, beta))
    recorded = payload["metrics"]["integer_all"]
    direct_cells = [
        cell
        for cell in payload["cells"]
        if ParallelShape(*cell["shape"]) not in table.reranked_shapes
    ]
    assert report["cells"] == recorded["cells"] == len(direct_cells)
    assert report["ordered_pairs"] == recorded["ordered_pairs"]
    assert abs(report["pairwise_accuracy"] - recorded["pairwise_accuracy"]) < 1e-9
    assert abs(report["max_regret_pct"] - recorded["max_regret_pct"]) < 1e-9
    assert report["max_regret_pct"] <= 5.0
    assert report["pairwise_accuracy"] >= 0.95
    assert not report["clear_misses"]


@_TABLES
def test_full_refit_reproduces_the_table_when_requested(table) -> None:
    if os.environ.get("ART_COST_CERTIFICATE_REFIT") != "1":
        return
    candidates, payload = fit.load_certificate(certificate_path(table.table_id))
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

    reranked_labels = {_label(shape) for shape in table.reranked_shapes}
    train = [
        c
        for c in candidates
        if not held_out(c.cell) and fit._shape_label(c) not in reranked_labels
    ]
    beta = fit.nnls(*fit.paired_deltas(train, terms))
    if arguments["objective"] == "regret":
        beta = fit.fit_regret(train, terms, beta)
    assert fit.integerize(train, terms, beta) == payload["integer_table_milli_us"]


@_TABLES
def test_reranked_shapes_are_certified_by_the_two_stage_selection(table) -> None:
    """Tables with a re-ranker: the shipped re-ranker is the certificate's, and
    the two-stage selection recomputed from the recorded aggregates (shortlist
    by the shortlist table, select by the plan structure) reproduces the
    certified metrics and passes the gates on every re-ranked shape; the
    direct metrics cover the directly scored shapes only."""

    candidates, payload = fit.load_certificate(certificate_path(table.table_id))
    block = payload.get("reranker")
    if table.reranker is None:
        assert block is None and not table.reranked_shapes
        assert all(
            ParallelShape(*cell["shape"]) in table.shapes for cell in payload["cells"]
        )
        return
    reranked_labels = {_label(shape) for shape in table.reranked_shapes}
    assert set(block["shapes"]) == reranked_labels
    assert block["shortlist_size"] == table.reranker.shortlist_size
    assert block["incumbent"] == table.reranker.incumbent
    assert block["shortlist_table_milli_us"] == dict(
        table.reranker.shortlist_coefficients_milli_us
    )
    assert block["wave_per_layer_milli_us"] == table.reranker.wave_per_layer_milli_us
    assert (
        block["max_rank_token_per_layer_milli_us"]
        == table.reranker.max_rank_token_per_layer_milli_us
    )
    terms = tuple(payload["fit_arguments"]["terms"])
    reranked = [c for c in candidates if fit._shape_label(c) in reranked_labels]
    direct = [c for c in candidates if fit._shape_label(c) not in reranked_labels]
    assert reranked and direct
    assert {fit._shape_label(c) for c in direct} == {
        _label(shape) for shape in table.shapes
    }
    recomputed = fit.evaluate_two_stage(
        reranked,
        terms=terms,
        shortlist_beta=np.asarray(
            [block["shortlist_table_milli_us"][name] / 1_000.0 for name in terms]
        ),
        rerank_beta=np.asarray(
            [
                block["wave_per_layer_milli_us"] / 1_000.0,
                block["max_rank_token_per_layer_milli_us"] / 1_000.0,
            ]
        ),
        shortlist_size=block["shortlist_size"],
        incumbent=block["incumbent"],
    )
    recorded = block["metrics"]["all"]
    for key in ("cells", "ordered_pairs", "clear_misses", "recall"):
        assert recomputed[key] == recorded[key], key
    for key in (
        "pairwise_accuracy",
        "median_regret_pct",
        "p95_regret_pct",
        "max_regret_pct",
    ):
        assert abs(recomputed[key] - recorded[key]) < 1e-9, key
    assert not fit.two_stage_gates(recomputed)
    for shape, metrics in block["metrics"]["by_shape"].items():
        assert not fit.two_stage_gates(metrics), shape
    # The direct metrics are the certificate's headline metrics: direct cells only.
    beta = np.asarray(
        [table.coefficients_milli_us.get(name, 0) / 1_000.0 for name in terms]
    )
    report = fit.evaluate(direct, fit.predict(direct, terms, beta))
    assert report["cells"] == payload["metrics"]["integer_all"]["cells"]


@_TABLES
def test_certificate_holds_exactly_the_manifest_cells(table) -> None:
    """Exact cell identities: every expected cell minus the explicit exclusions."""

    payload = json.loads(certificate_path(table.table_id).read_text())
    manifest = json.loads(manifest_path(table.table_id).read_text())
    assert manifest["table_id"] == table.table_id
    expected = {cell["key"] for cell in manifest["cells"]}
    excluded = {cell["key"] for cell in manifest["excluded"]}
    assert excluded <= expected
    certified = {cell["cell"] for cell in payload["cells"]}
    assert certified == expected - excluded
    assert set(payload["manifest"]["excluded"]) == excluded
    assert payload["manifest"]["path"] == manifest_path(table.table_id).name
    # Every retained cell carries its execution fingerprints.
    for cell in payload["cells"]:
        for key in (
            "source",
            "requests_sha256",
            "device",
            "param_dtype",
            "hidden_size",
            "geometry",
            "shape",
            "model",
        ):
            assert cell.get(key) not in (None, "", [], {}), (cell["cell"], key)


@_TABLES
def test_admitted_execution_classes_are_exactly_the_certified_ones(table) -> None:
    """The production table admits precisely the device classes, dtypes, model
    geometries and parallel shapes recorded in the certificate's cells."""

    payload = json.loads(certificate_path(table.table_id).read_text())
    cells = payload["cells"]
    memory_classes = {device.memory_class for device in table.device_classes}
    for cell in cells:
        assert cell["device_memory_class"] in memory_classes, cell["cell"]
    admitted = payload["admitted"]
    geometries = {ModelGeometry(**geometry) for geometry in admitted["geometries"]}
    assert geometries == set(table.geometries)
    assert geometries == {ModelGeometry(**cell["geometry"]) for cell in cells}
    shapes = {ParallelShape(*shape) for shape in admitted["shapes"]}
    assert shapes == set(table.shapes) | set(table.reranked_shapes)
    assert shapes == {ParallelShape(*cell["shape"]) for cell in cells}
    devices = {
        DeviceClass(tuple(entry["capability"]), entry["memory_class"])
        for entry in admitted["device_classes"]
    }
    assert devices == set(table.device_classes)
    assert set(admitted["param_dtypes"]) == set(table.param_dtypes)
    # Every geometry was measured at every admitted shape (product admission);
    # a withheld pair is a measured pair the manifest documents with a reason,
    # and the table does not admit it.
    measured = {
        (ModelGeometry(**cell["geometry"]), ParallelShape(*cell["shape"]))
        for cell in cells
    }
    for geometry in table.geometries:
        for shape in (*table.shapes, *table.reranked_shapes):
            assert (geometry, shape) in measured, (geometry.hidden_size, shape)
    manifest = json.loads(manifest_path(table.table_id).read_text())
    geometry_of_model = {
        cell["model"]: ModelGeometry(**cell["geometry"]) for cell in cells
    }
    withheld = {
        (geometry_of_model[entry["model"]], ParallelShape(*entry["shape"]))
        for entry in manifest.get("withheld", [])
    }
    assert withheld == set(table.withheld)
    assert withheld <= measured
    assert all(entry.get("reason") for entry in manifest.get("withheld", []))
    for geometry, shape in withheld:
        assert not table.admits(
            device=table.device_classes[0],
            param_dtype=table.param_dtypes[0],
            geometry=geometry,
            shape=shape,
        )
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
