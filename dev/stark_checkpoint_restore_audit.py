"""Audit committed FP32 gradient restoration from an immutable snapshot, without Torch."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(snapshot, receipts, run, expected_sha, expected_seq=1, expected_count=994):
    with snapshot.open("rb") as source:
        actual_sha = hashlib.file_digest(source, "sha256").hexdigest()
    require(actual_sha == expected_sha, "snapshot content address does not match")
    peers = [json.loads(path.read_text()) for path in receipts]
    require(
        bool(peers) and all(peer == peers[0] for peer in peers), "rank receipts differ"
    )
    record = peers[0]
    prefix = f"run/{run}/gradient/"
    nonzero = 0
    elements = 0
    with safe_open(snapshot, framework="np") as tensors:
        state = json.loads(tensors.metadata()["state"])
        require(state["seq"] == expected_seq, "wrong committed snapshot sequence")
        keys = [key for key in tensors.keys() if key.startswith(prefix)]
        require(len(keys) == expected_count, "incomplete gradient tensor coverage")
        require(
            {key.removeprefix(prefix) for key in keys} == record.keys(),
            "receipt coverage differs",
        )
        for key in sorted(keys):
            tensor = tensors.get_tensor(key)
            require(tensor.dtype == np.float32, f"gradient is not FP32: {key}")
            require(
                bool(np.isfinite(tensor).all()), f"nonfinite original gradient: {key}"
            )
            with np.errstate(over="raise", invalid="raise"):
                doubled = np.multiply(tensor, np.float32(2), dtype=np.float32)
            expected = record[key.removeprefix(prefix)]
            digest = hashlib.sha256(doubled.tobytes()).hexdigest()
            require(
                digest == expected["sha256"],
                f"pending gradient was not doubled exactly: {key}",
            )
            count = int(np.count_nonzero(doubled))
            require(count == expected["nonzero"], f"nonzero count differs: {key}")
            nonzero += count
            elements += tensor.size
    require(nonzero > 0, "all restored gradients are zero")
    return {
        "snapshot_sha256": actual_sha,
        "committed_seq": expected_seq,
        "gradient_tensors": len(keys),
        "gradient_elements": int(elements),
        "nonzero_elements": nonzero,
        "ranks": len(peers),
        "rank_receipts_equal": True,
        "doubled_gradient_bytes_exact": True,
        "torch_imported": "torch" in sys.modules,
        "numpy": np.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--sha256", required=True, help="Digest from original immutable S3 address"
    )
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--expected-seq", type=int, default=1)
    parser.add_argument("--expected-count", type=int, default=994)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.snapshot,
        args.receipts,
        args.run,
        args.sha256,
        args.expected_seq,
        args.expected_count,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
