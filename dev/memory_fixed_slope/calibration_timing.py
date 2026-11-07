"""Estimate additional boot rungs from run6 and v7 GPU evidence (no GPU run)."""

import csv
import gzip
import json
from pathlib import Path

OUT = Path(__file__).parent


def main():
    cases = json.loads(
        (
            OUT.parents[1] / "tests/unit/fixtures/memory_fixed_slope/sweeps.json"
        ).read_text()
    )
    run6 = Path("/tmp/stark-gpu/cp8-perf-profile/run6-analysis")
    matrix = {r["case"]: r for r in csv.DictReader((run6 / "main-matrix.csv").open())}
    dense = {}
    for op in ["forward_backward", "forward"]:
        for n in [8192, 32768]:
            samples = [
                r["wall_seconds"]
                for r in cases
                if r["model"] == 0 and r["operation"] == op and r["length"] == n
            ]
            dense[(op, n)] = min(
                samples
            )  # warmest supplied rank variant; first-use charged separately below
    moe = {
        ("forward_backward", 8192): float(matrix["single-8192"]["warm_max_wall_mean"]),
        ("forward_backward", 32768): float(
            matrix["single-32768"]["warm_max_wall_mean"]
        ),
    }
    forward = []
    cold = []
    for line in gzip.open(run6 / "main-all-rank-records.jsonl.gz", "rt"):
        r = json.loads(line)
        if r.get("rank") != 0 or r.get("event") != "op" or r.get("status") != "ok":
            continue
        if r.get("name") == "calibration/forward" and r.get("lengths") in [
            [8192],
            [32768],
        ]:
            forward.append(r)
        if r.get("name") == "calibration/forward_backward" and r.get("lengths") == [
            8192
        ]:
            cold.append(r["wall_seconds"])
    for n in [8192, 32768]:
        moe[("forward", n)] = min(
            r["wall_seconds"] for r in forward if r["lengths"] == [n]
        )
    result = {}
    for model, families, rates, firstuse in [
        (
            "dense",
            2,
            dense,
            max(
                r["wall_seconds"]
                for r in cases
                if r["model"] == 0 and r["length"] == 8192
            ),
        ),
        ("moe", 6, moe, max(cold)),
    ]:
        rung_count = 3 * 2 * 2 * families
        execution = 3 * families * sum(rates.values())
        planning = rung_count * 0.9  # supplied production per-wave plan figure
        compile_budget = (
            families * firstuse
        )  # one cold-size-equivalent allowance per output mix; not an observed multi-root compile duration
        seconds = execution + planning + compile_budget
        result[model] = dict(
            estimate=True,
            new_rungs=rung_count,
            warm_execution_seconds=execution,
            planning_seconds=planning,
            first_use_allowance_seconds=compile_budget,
            extra_seconds=seconds,
            baseline_seconds=780,
            total_seconds=780 + seconds,
            extra_seconds_no_cross_rank_compile_reuse=execution
            + planning
            + 3 * compile_budget,
            total_seconds_no_cross_rank_compile_reuse=780
            + execution
            + planning
            + 3 * compile_budget,
            deadline_seconds=3600,
            per_rung_gpu_seconds={f"{op}-{n}": v for (op, n), v in rates.items()},
            limitation="Assumes compiled graph reuse across configured LoRA ranks; 64/256-root first-use costs and StableReinforce/route-replay head costs require GPU validation. Other target layouts are not included.",
        )
    (OUT / "calibration_timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
