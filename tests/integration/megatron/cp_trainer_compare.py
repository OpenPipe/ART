"""Compare rank-sharded CPU TrainerRank probe artifacts in canonical token order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def compare(reference: Path, candidate: Path) -> dict[str, Any]:
    ref = torch.load(reference / "rank0.pt", weights_only=True)
    ranks = [
        torch.load(path, weights_only=True)
        for path in sorted(candidate.glob("rank*.pt"))
    ]
    result: dict[str, Any] = {}
    for name, value in ref.items():
        if (
            not isinstance(value, torch.Tensor)
            or name.endswith("uids")
            or name in ("token_uids", "position_ids")
        ):
            continue
        if name == "logprobs":
            actual = ranks[0][name]
            expected = value
        else:
            uid_key = name + ".uids" if name + ".uids" in ref else "token_uids"
            ref_ids = ref[uid_key].reshape(-1)
            valid = ref_ids >= 0
            n = int(ref_ids[valid].max()) + 1
            expected = value.new_zeros((n, *value.shape[1:]))
            actual = torch.zeros_like(expected)
            expected[ref_ids[valid]] = value[valid]
            seen = torch.zeros(n, dtype=torch.long)
            for rank in ranks:
                ids = rank[uid_key].reshape(-1)
                keep = (ids >= 0) & (ids < n)
                actual[ids[keep]] = rank[name][keep]
                seen[ids[keep]] += 1
            assert (seen > 0).all(), (name, int((seen == 0).sum()))
        delta = (expected - actual).abs().double()
        per_token = delta.reshape(delta.shape[0], -1).amax(-1)
        bad = torch.nonzero(per_token > 1e-5).flatten()
        result[name] = {
            "mean": float(delta.mean()),
            "max": float(delta.max()),
            "p50": float(delta.quantile(0.5)),
            "p99": float(delta.quantile(0.99)),
            "rel_l2": float(
                (actual - expected).double().norm()
                / expected.double().norm().clamp_min(1e-30)
            ),
            "first_bad_1e5": int(bad[0]) if bad.numel() else None,
            "frac_gt_005": float((delta > 0.05).double().mean()),
            "frac_gt_1": float((delta > 1).double().mean()),
            "close": torch.allclose(actual, expected, atol=2e-5, rtol=2e-5),
        }
        if name == "logprobs":
            offset = 0
            documents = []
            for length in ref["lengths"]:
                d = per_token[offset : offset + length]
                bad = torch.nonzero(d > 0.05).flatten()
                documents.append(
                    {
                        "length": length,
                        "first_gt_005": int(bad[0]) if bad.numel() else None,
                        "mean": float(d.mean()),
                        "frac_gt_1": float((d > 1).double().mean()),
                    }
                )
                offset += length
            result[name]["documents"] = documents
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-close", action="store_true")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload)
    if args.assert_close:
        assert all(record["close"] for record in result.values())


if __name__ == "__main__":
    main()
