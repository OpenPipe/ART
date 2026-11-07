"""CPU replay of supplied peaks through ART's fixed/slope and scalar admission.

Fits are in-sample. Scalar replay uses model geometry, balanced CP8 root rows,
and target-only outputs (4 B/row), with a metadata-only model. Unsupported native
operator-specific floors are not synthesized; full native floors remain in ART
and their existing tests. The empirical line alone covers all supplied peaks.
"""

from collections import defaultdict
import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from art.trainer_rank import TrainerRank, _impl, _memory, _memory_model

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/unit/fixtures/memory_fixed_slope"
OUT = Path(__file__).parent


def scalar_rank(model):
    cfg = json.loads(
        (
            FIXTURES / ("dense-config.json" if model == 0 else "moe-config.json")
        ).read_text()
    )
    p = SimpleNamespace(**cfg)
    p.num_layers = cfg["num_hidden_layers"]
    p.ffn_hidden_size = cfg.get("intermediate_size", 4 * cfg["hidden_size"])
    p.kv_channels = cfg["head_dim"]
    p.num_query_groups = cfg["num_key_value_heads"]
    p.num_moe_experts = cfg.get("num_experts", 0)
    p.moe_router_topk = cfg.get("num_experts_per_tok", 0)
    p.moe_ffn_hidden_size = cfg.get("moe_intermediate_size", 0)
    p.moe_shared_expert_intermediate_size = cfg.get(
        "shared_expert_intermediate_size", 0
    )
    p.recompute_granularity = "full"
    p.params_dtype = torch.bfloat16
    m = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
    m.requires_grad_(False)
    m.config = p
    r = TrainerRank(
        SimpleNamespace(
            model=[m],
            optimizer=None,
            provider=p,
            model_support_handler=SimpleNamespace(
                build_gdn_execution_spec=True, is_moe=bool(p.num_moe_experts)
            ),
        )
    )
    r._topology_key = lambda: (1, 1, 8, 1)
    r._parallel_shape = replace(r._parallel_shape, cp=8)
    r._dense_fc1_adapted = not p.num_moe_experts
    r._gdn_layers = sum(t == "linear_attention" for t in cfg["layer_types"])
    return r


def sig(key):
    _, operation, lora = key
    grad = operation == "forward_backward"
    return _impl._MemorySignature(
        topology=(1, 1, 8, 1),
        planner_coefficients=(1, None),
        slot_group_count=1,
        request_mix=("target:single",),
        grad_enabled=grad,
        grad_modes=(grad,),
        slot_shapes=((grad, ((2, 2048, lora, lora, 2048),)),),
    )


def fit(rank, signature, rows, *, rank_id=None):
    for row in rows:
        measured = row["measured_increment_bytes"]
        if rank_id is not None:
            per_rank = next(v for v in row["per_rank"] if v["rank"] == rank_id)
            measured = max(
                a["measured_increment_bytes"]
                for a in per_rank["attempts"]
                if a["status"] == "complete"
            )
        n = row["length"]
        plan = SimpleNamespace(
            signature=signature,
            packed_tokens=n,
            active_logical_tokens=n,
            output_bytes=4 * n,
        )
        with _memory.fixed_slope_memory_profiles():
            _memory._update_memory_profile(rank, plan, measured, retained_bytes=None)
    return rank._memory_profiles[signature]


def required(rank, signature, row):
    n = row["length"]
    return rank._estimate_required_memory_bytes_from_values(
        packed_tokens=n,
        logical_tokens=n,
        output_bytes=4 * n,
        signature=signature,
        group_rows=((n // 8, signature.grad_enabled),),
        gdn_segments=int(signature.grad_enabled),
    )


def main():
    cases = json.loads((FIXTURES / "sweeps.json").read_text())
    groups = defaultdict(list)
    for row in cases:
        groups[(row["model"], row["operation"], row["lora_rank"])].append(row)
    records = []
    sensitivity = []
    for key, rows in groups.items():
        rank = scalar_rank(key[0])
        signature = sig(key)
        profile = fit(rank, signature, rows)
        rank_profiles = []
        for rank_id in range(8):
            rank._memory_profiles.clear()
            rank_profiles.append(fit(rank, signature, rows, rank_id=rank_id))
        for row in rows:
            n = row["length"]
            measured = row["measured_increment_bytes"]
            rank._memory_profiles[signature] = profile
            line = 4 * n + _memory._profile_compute_peak(profile, n)
            effective = required(rank, signature, row)
            with patch.object(_impl, "_MEMORY_SAFETY_FACTOR", 1.0):
                raw = required(rank, signature, row)
            per_rank_effective = []
            for p in rank_profiles:
                rank._memory_profiles[signature] = p
                per_rank_effective.append(required(rank, signature, row))
            old = row["ART_predicted_increment_bytes"]
            records.append(
                dict(
                    model=key[0],
                    operation=key[1],
                    lora_rank=key[2],
                    length=n,
                    measured_bytes=measured,
                    old_raw_bytes=old / 1.1,
                    old_effective_bytes=old,
                    old_raw_ratio=old / 1.1 / measured,
                    old_effective_ratio=old / measured,
                    old_under=old < measured,
                    new_line_raw_bytes=line,
                    new_line_raw_ratio=line / measured,
                    new_raw_bytes=raw,
                    new_effective_bytes=effective,
                    new_raw_ratio=raw / measured,
                    new_effective_ratio=effective / measured,
                    rank_local_max_effective_bytes=max(per_rank_effective),
                    rank_local_effective_ratio=max(per_rank_effective) / measured,
                    usable_limit_bytes=row["usable_limit_bytes"],
                )
            )
            assert min(effective, max(per_rank_effective)) >= measured
        for threshold in (2048, 4096, 8192, 16384, 32768):
            fixed, slope = _memory_model.fit_points(
                profile.compute_observations, min_tokens=threshold
            )
            ratios = [
                (4 * r["length"] + _memory_model.predict(fixed, slope, r["length"]))
                * 1.1
                / r["measured_increment_bytes"]
                for r in rows
            ]
            sensitivity.append(
                dict(
                    model=key[0],
                    operation=key[1],
                    lora_rank=key[2],
                    threshold=threshold,
                    min=min(ratios),
                    max=max(ratios),
                )
            )
    records.sort(
        key=lambda r: (r["model"], r["length"], r["lora_rank"], r["operation"])
    )
    for name, rows in [("comparison", records), ("thresholds", sensitivity)]:
        with (OUT / f"{name}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = [
        "# OLD vs NEW supplied-sweep replay",
        "",
        __doc__.strip(),
        "",
        "OLD raw is reconstructed from the recorded effective value. NEW raw uses the identical scalar calculation with safety=1 to expose the pre-factor demand; `new_line_raw_ratio` in CSV additionally isolates the empirical line. Reserve is already on the usable-budget side. These are post-fit replays, not predictions made before observing each case. See DESIGN.md for native-floor limitations.",
        "",
        "| Model | Op | LoRA | Tokens | Measured GiB | OLD raw ratio | OLD effective ratio | NEW raw ratio | NEW effective ratio | OLD under? |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['model']} | {r['operation']} | {r['lora_rank']} | {r['length']} | {r['measured_bytes'] / 2**30:.6f} | {r['old_raw_ratio']:.6f} | {r['old_effective_ratio']:.6f} | {r['new_raw_ratio']:.6f} | {r['new_effective_ratio']:.6f} | {'**YES**' if r['old_under'] else ''} |"
        )
    summary = dict(
        cases=len(records),
        old_under=sum(r["old_under"] for r in records),
        new_effective_min=min(r["new_effective_ratio"] for r in records),
        new_effective_max=max(r["new_effective_ratio"] for r in records),
        new_raw_min=min(r["new_raw_ratio"] for r in records),
        new_raw_max=max(r["new_raw_ratio"] for r in records),
        new_line_effective_min=min(r["new_line_raw_ratio"] * 1.1 for r in records),
        new_line_effective_max=max(r["new_line_raw_ratio"] * 1.1 for r in records),
        effective_over_1_5=sum(r["new_effective_ratio"] > 1.5 for r in records),
    )
    lines += ["", f"```json\n{json.dumps(summary, indent=2)}\n```"]
    (OUT / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
