"""Each calibration manifest is exactly what its checked-in launches produce.

One manifest per calibrated table (``dev/trainer_rank_cost_calibration_manifest_<table>.json``).
A manifest names the fixed recipes it was run from (the local script and the
two SkyPilot recipes in both ``CELL_SET`` modes, parsed here with their
Ellavox group loops expanded) and/or the shape-lattice launches (model, shapes,
cells) of the generic lattice recipe; the resulting cell keys must equal the
manifest's, so a clean rerun of the documented launches reproduces the
certified cell set (minus the explicit exclusions).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from art.trainer_rank._planner_cost import CALIBRATED_TABLES

_ROOT = Path(__file__).resolve().parents[2]
_DEV = _ROOT / "dev"
_FULL_LAYERS = {
    "Qwen/Qwen3.5-4B": 32,
    "Qwen/Qwen3-4B": 36,
    "Qwen/Qwen3.5-35B-A3B": 40,
    "Qwen/Qwen3-8B": 36,
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-14B": 40,
    "Qwen/Qwen3.5-27B": 64,
    "Qwen/Qwen3-30B-A3B": 48,
}
_ELLAVOX_GROUPS = range(8)
_LATTICE_RECIPE = "dev/trainer_rank_cost_calibration_lattice.sky.yaml"


def manifest_path(table_id: str) -> Path:
    return _DEV / f"trainer_rank_cost_calibration_manifest_{table_id}.json"


def _key(
    cell: str,
    model: str,
    layers: int,
    tp: int,
    cp: int,
    group: int | None,
    ep: int = 1,
    etp: int = 1,
) -> str:
    key = f"{cell}|{model}|L{layers}|tp{tp}|cp{cp}"
    key += f"|ep{ep}" if ep > 1 else ""
    key += f"|etp{etp}" if etp > 1 else ""
    return key + (f"|g{group}" if group is not None else "")


def _expand(
    cell: str, model: str, layers: str, tp: int, cp: int, group: str | None
) -> set[str]:
    depth = _FULL_LAYERS[model] if int(layers) == 0 else int(layers)
    if cell == "cal-ellavox":
        assert group is not None and group.startswith("$")
        return {_key(cell, model, depth, tp, cp, g) for g in _ELLAVOX_GROUPS}
    return {_key(cell, model, depth, tp, cp, None)}


def _local_cells() -> set[str]:
    text = (_DEV / "trainer_rank_cost_calibration_local.sh").read_text()
    cells: set[str] = set()
    for match in re.finditer(
        r"^\s*run_cell\s+(\S+)\s+(\S+)\s+(\d+)(?:\s+\"?(\$g)\"?)?", text, re.M
    ):
        cell, model, layers, group = match.groups()
        cells |= _expand(cell, model, layers, 1, 1, group)
    return cells


def _cp4_cells() -> set[str]:
    text = (_DEV / "trainer_rank_cost_calibration_cp4.sky.yaml").read_text()
    cells: set[str] = set()
    for match in re.finditer(
        r"^\s*run_cell\s+(\S+)\s+(\S+)\s+(\d+)(?:\s+\"?(\$g)\"?)?", text, re.M
    ):
        cell, model, layers, group = match.groups()
        cells |= _expand(cell, model, layers, 1, 4, group)
    return cells


def _two_gpu_cells() -> set[str]:
    text = (_DEV / "trainer_rank_cost_calibration_2gpu.sky.yaml").read_text()
    cells: set[str] = set()
    for match in re.finditer(
        r"^\s*run_cell\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)(?:\s+\"?(\$g)\"?)?",
        text,
        re.M,
    ):
        tp, cp, cell, model, layers, group = match.groups()
        cells |= _expand(cell, model, layers, int(tp), int(cp), group)
    return cells


_RECIPE_CELLS = {
    "dev/trainer_rank_cost_calibration_local.sh": _local_cells,
    "dev/trainer_rank_cost_calibration_cp4.sky.yaml": _cp4_cells,
    "dev/trainer_rank_cost_calibration_2gpu.sky.yaml": _two_gpu_cells,
}


def launch_cells(launch: dict) -> set[str]:
    """Cells one lattice launch runs: every shape x every cell spec."""

    assert launch["recipe"] == _LATTICE_RECIPE, launch["recipe"]
    model = launch["model"]
    cells: set[str] = set()
    for shape in launch["shapes"]:
        tp, cp, ep, etp = (list(shape) + [1, 1])[:4]
        for spec in launch["cells"]:
            name, *rest = spec.split(":")
            layers = int(rest[0]) if rest else 0
            group = int(rest[1]) if len(rest) > 1 else None
            depth = _FULL_LAYERS[model] if layers == 0 else layers
            cells.add(_key(name, model, depth, tp, cp, group, ep, etp))
    return cells


def manifest_cells(manifest: dict) -> set[str]:
    launched: set[str] = set()
    for recipe in manifest.get("recipes", []):
        launched |= _RECIPE_CELLS[recipe]()
    for launch in manifest.get("launches", {}).values():
        launched |= launch_cells(launch)
    return launched


@pytest.mark.parametrize("table", CALIBRATED_TABLES, ids=lambda t: t.table_id)
def test_launches_produce_exactly_the_manifest_cells(table) -> None:
    manifest = json.loads(manifest_path(table.table_id).read_text())
    assert manifest["table_id"] == table.table_id
    expected = {cell["key"] for cell in manifest["cells"]}
    launched = manifest_cells(manifest)
    assert launched, "no launches parsed from the manifest"
    assert launched == expected, {
        "only in launches": sorted(launched - expected),
        "only in manifest": sorted(expected - launched),
    }
    excluded = {cell["key"] for cell in manifest["excluded"]}
    assert excluded <= expected
    assert all(cell.get("reason") for cell in manifest["excluded"])
    campaigns = {cell["campaign"] for cell in manifest["cells"]}
    assert campaigns <= set(manifest.get("campaign_labels", [])) | set(
        manifest.get("launches", {})
    )


def test_dense_table_manifest_is_the_original_58_cells_plus_blind_spot_evidence() -> (
    None
):
    """The 58 fitted cells of the original recipes, plus the Qwen3-4B real-data
    launch at TP1 x CP4 and TP2 x CP2 (16 cells) that documents the
    context-parallel blind spot; none of those 16 is fitted."""

    manifest = json.loads(manifest_path("dense-h2560-h200-bf16").read_text())
    expected = {cell["key"] for cell in manifest["cells"]}
    excluded = {cell["key"] for cell in manifest["excluded"]}
    assert len(expected - excluded) == 58
    assert excluded == launch_cells(manifest["launches"]["tr-cost-q3-4b-cp4"])
    assert len(excluded) == 16
    assert [entry["shape"] for entry in manifest["withheld"]] == [[1, 4, 1, 1]]
    for recipe in manifest["recipes"]:
        assert (_ROOT / recipe).is_file(), recipe


def test_lattice_launch_expansion() -> None:
    launch = {
        "recipe": _LATTICE_RECIPE,
        "model": "Qwen/Qwen3.5-35B-A3B",
        "shapes": [[1, 2, 2], [2, 1, 1, 1]],
        "cells": ["cal-grpo-g8:0", "cal-ellavox:0:3"],
    }
    assert launch_cells(launch) == {
        "cal-grpo-g8|Qwen/Qwen3.5-35B-A3B|L40|tp1|cp2|ep2",
        "cal-ellavox|Qwen/Qwen3.5-35B-A3B|L40|tp1|cp2|ep2|g3",
        "cal-grpo-g8|Qwen/Qwen3.5-35B-A3B|L40|tp2|cp1",
        "cal-ellavox|Qwen/Qwen3.5-35B-A3B|L40|tp2|cp1|g3",
    }
