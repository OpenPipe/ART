"""Compare per-rank collective traces from ``trainer_rank_collective_trace.py``.

Prints the layout label each rank ran per forward, then the first collective at
which the ranks disagree on operation, participation or split sizes (rank a's
send to b must equal rank b's expected receive from a), and any stall dumps.

Usage: python dev/trainer_rank_collective_diff.py <log-dir>
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import sys


def _load(root: Path) -> dict[int, list[dict]]:
    logs: dict[int, list[dict]] = {}
    for path in sorted(root.glob("collectives.rank*.jsonl")):
        rank = int(path.stem.split("rank")[1])
        logs[rank] = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    return logs


def main() -> None:
    root = Path(sys.argv[1])
    logs = _load(root)
    ranks = sorted(logs)
    print("records per rank:", {rank: len(logs[rank]) for rank in ranks})
    for rank in ranks:
        forwards = collections.OrderedDict()
        for record in logs[rank]:
            if record["op"] == "forward_start":
                forwards[record["forward_index"]] = record["anchor"]
        print(
            f"rank {rank} forwards: "
            + ", ".join(f"f{index}={anchor}" for index, anchor in forwards.items())
        )
    sequences = {
        rank: [r for r in logs[rank] if not r["op"].startswith("forward_")]
        for rank in ranks
    }
    length = max(len(sequence) for sequence in sequences.values())
    for index in range(length):
        records = {
            rank: sequences[rank][index] if index < len(sequences[rank]) else None
            for rank in ranks
        }
        signatures = {
            rank: (
                (record["op"], record["caller"].split("/")[0], record["group_size"])
                if record
                else None
            )
            for rank, record in records.items()
        }
        reason = None
        if len(set(signatures.values())) != 1:
            reason = f"different operations: {signatures}"
        else:
            mismatches = []
            for a in ranks:
                for b in ranks:
                    if a == b or records[a] is None or records[b] is None:
                        continue
                    sent, expected = (
                        records[a].get("in_splits"),
                        records[b].get("out_splits"),
                    )
                    if sent is None or expected is None:
                        continue
                    if sent[b] != expected[a]:
                        mismatches.append(
                            f"rank{a}->rank{b}: sends {sent[b]}, rank{b} expects {expected[a]}"
                        )
            if mismatches:
                reason = "split mismatch: " + "; ".join(mismatches[:4])
        if reason is None:
            continue
        print(f"\nFIRST DIVERGENCE at collective #{index + 1}: {reason}")
        for rank in ranks:
            for j in range(max(0, index - 3), min(len(sequences[rank]), index + 4)):
                record = sequences[rank][j]
                detail = (
                    f"in={record.get('in_splits')} out={record.get('out_splits')}"
                    if record["op"] == "all_to_all_single"
                    else f"numel={record.get('numel')}"
                )
                print(
                    f"  rank{rank} #{j + 1} f{record['forward_index']} {record['anchor']} "
                    f"{record['op']} {record['caller'][:70]} {detail}"
                )
        break
    else:
        print("\nno divergence in the common prefix of the traces")
    for path in sorted(root.glob("stall.rank*.txt")):
        print(f"\n=== {path.name}")
        print(path.read_text()[:1200])


if __name__ == "__main__":
    main()
