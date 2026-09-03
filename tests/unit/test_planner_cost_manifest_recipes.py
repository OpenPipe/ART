"""The calibration manifest is exactly what the checked-in recipes launch.

Parses the three runners (the local script and the two SkyPilot recipes in
both ``CELL_SET`` modes), expands the Ellavox group loops, resolves ``layers=0``
to each model's depth, and requires the resulting cell keys to equal the
manifest's, so a clean rerun of every documented recipe reproduces the
certified cell set (minus the explicit exclusions).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

_ROOT = Path(__file__).resolve().parents[2]
_DEV = _ROOT / "dev"
_MANIFEST = _DEV / "trainer_rank_cost_calibration_manifest.json"
_FULL_LAYERS = {"Qwen/Qwen3.5-4B": 32, "Qwen/Qwen3-4B": 36}
_ELLAVOX_GROUPS = range(8)


def _key(
    cell: str, model: str, layers: int, tp: int, cp: int, group: int | None
) -> str:
    key = f"{cell}|{model}|L{layers}|tp{tp}|cp{cp}"
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


def test_recipes_launch_exactly_the_manifest_cells() -> None:
    manifest = json.loads(_MANIFEST.read_text())
    expected = {cell["key"] for cell in manifest["cells"]}
    launched = _local_cells() | _cp4_cells() | _two_gpu_cells()
    assert launched, "no run_cell invocations parsed from the recipes"
    assert launched == expected, {
        "only in recipes": sorted(launched - expected),
        "only in manifest": sorted(expected - launched),
    }
    excluded = {cell["key"] for cell in manifest["excluded"]}
    assert excluded <= expected
    assert len(expected - excluded) == 58


def test_manifest_campaign_labels_name_checked_in_recipes() -> None:
    manifest = json.loads(_MANIFEST.read_text())
    assert {cell["campaign"] for cell in manifest["cells"]} <= {
        "local",
        "tr-cost-cp4",
        "tr-cost-cp4-2",
        "tr-cost-2gpu",
        "tr-cost-2gpu-2",
        "tr-cost-cp4-840",
    }
