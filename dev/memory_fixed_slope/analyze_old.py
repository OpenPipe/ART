"""Summarize supplied GPU evidence; no GPU or ART import required."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/unit/fixtures/memory_fixed_slope"
OUTPUT = Path(__file__).parent


def main():
    rows = []
    for case in json.loads((FIXTURES / "sweeps.json").read_text()):
        measured = case["measured_increment_bytes"]
        effective = case["ART_predicted_increment_bytes"]
        rows.append(
            dict(
                model=case["model"],
                operation=case["operation"],
                lora_rank=case["lora_rank"],
                length=case["length"],
                measured_bytes=measured,
                old_raw_bytes=effective / 1.1,
                old_effective_bytes=effective,
                old_raw_ratio=effective / 1.1 / measured,
                old_effective_ratio=effective / measured,
                old_under=effective < measured,
            )
        )
    with (OUTPUT / "old.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Recorded OLD estimates",
        "",
        "Raw is reconstructed as recorded effective / 1.10 (integer rounding uncertainty below two bytes). Effective is the recorded admission operand against usable limit. The 3% device reserve is already subtracted from usable limit; adding it to this operand would count it twice.",
        "",
        "| Model | Op | LoRA | Tokens | Measured bytes | OLD raw / measured | OLD effective / measured | Under? |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['operation']} | {row['lora_rank']} | {row['length']} | {row['measured_bytes']} | {row['old_raw_ratio']:.6f} | {row['old_effective_ratio']:.6f} | {'**YES**' if row['old_under'] else ''} |"
        )
    lines += [
        "",
        f"{sum(r['old_under'] for r in rows)} / {len(rows)} OLD effective estimates under measured. Full byte values: old.csv.",
    ]
    (OUTPUT / "OLD.md").write_text("\n".join(lines) + "\n")
    print(lines[-1])
    print(
        "effective range",
        min(r["old_effective_ratio"] for r in rows),
        max(r["old_effective_ratio"] for r in rows),
    )
    for r in rows:
        if r["old_under"]:
            print(r)


if __name__ == "__main__":
    main()
