"""Frozen boot-anchor replay; held-out cells never update the fitted profile.

A/B use only measured 8K / 8K+32K anchors. C is a fixture projection of the
configured root-only ladder: 512, 2048 and 81920 are not in the fixture, so only
its measured 8K/32K rungs can be replayed. C does not qualify RL signatures or
claim an exact prediction of the unavailable 81920 observation.
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from replay import FIXTURES, fit, required, scalar_rank, sig

from art.trainer_rank import _impl

OUT = Path(__file__).parent


def main():
    groups = defaultdict(list)
    for row in json.loads((FIXTURES / "sweeps.json").read_text()):
        groups[row["model"], row["operation"], row["lora_rank"]].append(row)
    records: list[dict[str, Any]] = []
    summary = []
    timings = json.loads((OUT / "calibration_timing.json").read_text())
    for choice, sizes in (
        ("A", (8192,)),
        ("B", (8192, 32768)),
        ("C-proxy", (8192, 32768)),
    ):
        for key, rows in groups.items():
            rank = scalar_rank(key[0])
            signature = sig(key)
            # Max over actual rank-local predictions, as admission's MAX does.
            locals_ = []
            for rank_id in range(8):
                rank._memory_profiles.clear()
                locals_.append(
                    fit(
                        rank,
                        signature,
                        [r for r in rows if r["length"] in sizes],
                        rank_id=rank_id,
                    )
                )
            for row in [*rows, dict(length=262144)]:
                n = row["length"]
                predictions = []
                raw_predictions = []
                for p in locals_:
                    rank._memory_profiles[signature] = p
                    with patch.object(_impl, "_MEMORY_SAFETY_FACTOR", 1.0):
                        raw_predictions.append(
                            rank._estimate_required_memory_bytes_from_values(
                                packed_tokens=n,
                                logical_tokens=n,
                                output_bytes=4 * n,
                                signature=signature,
                                group_rows=((n // 8, signature.grad_enabled),),
                                gdn_segments=int(signature.grad_enabled),
                                include_cold_extrapolation=False,
                            )
                        )
                    predictions.append(required(rank, signature, row))
                raw = max(raw_predictions)
                effective = max(predictions)
                measured = row.get("measured_increment_bytes")
                usable = row.get("usable_limit_bytes")
                record = dict(
                    choice=choice,
                    model="dense" if key[0] == 0 else "MoE",
                    operation=key[1],
                    lora_rank=key[2],
                    tokens=n,
                    raw_bytes=raw,
                    effective_bytes=effective,
                    measured_bytes=measured,
                    usable_limit_bytes=usable,
                    raw_ratio=raw / measured if measured else None,
                    effective_ratio=effective / measured if measured else None,
                    refused=effective > usable if usable else None,
                    cold_factor=1.75 if n > max(sizes) else 1.0,
                    within_trust=n <= max(sizes) * 8,
                )
                records.append(record)
                if measured:
                    assert effective >= measured, record
        for model, families in (("dense", 2), ("MoE", 6)):
            cells = [
                r
                for r in records
                if r["choice"] == choice and r["model"] == model and r["measured_bytes"]
            ]
            ranges = {}
            for length in (0, 65536, 131072):
                values = [
                    r["effective_ratio"]
                    for r in cells
                    if not length or r["tokens"] == length
                ]
                ranges[str(length)] = [min(values), max(values)]
            timing = timings["dense" if model == "dense" else "moe"]
            gpu = timing["per_rung_gpu_seconds"]
            count = families * 3 * 2 * len(sizes) if choice != "C-proxy" else 0
            execution = (
                families
                * 3
                * sum(
                    gpu[f"{op}-{n}"]
                    for op in ("forward", "forward_backward")
                    for n in sizes
                )
                if count
                else 0
            )
            first = timing["first_use_allowance_seconds"] if count else 0
            summary.append(
                dict(
                    choice=choice,
                    model=model,
                    cells=len(cells),
                    effective_ranges=ranges,
                    refused_fitting_100k=[
                        r for r in cells if r["tokens"] >= 100000 and r["refused"]
                    ],
                    added_boot_seconds_estimate=execution + count * 0.9 + first,
                    added_boot_seconds_no_rank_compile_reuse_estimate=execution
                    + count * 0.9
                    + first * 3,
                    new_rl_rungs=count,
                )
            )
    with (OUT / "anchor-choices.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (OUT / "anchor-choices.json").write_text(
        json.dumps(dict(assumptions=__doc__, summary=summary, cells=records), indent=2)
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
