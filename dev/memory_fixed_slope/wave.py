"""Approximate dense RL width: production CPU planner, supplied root peak fit.

568 datums, families of 16 sharing a 240-token prompt, independent 213-token
completions. CP8, LoRA 8, 24 GiB usable (host-overhead harness budget). A synthetic
8-token 2 MB/token observation reconstructs that harness's poisoned OLD rate.
The root-sweep fit is transferred to the RL signature for this what-if only;
this transfer is NOT permitted by production signature matching or a GPU bound.
"""

import json
from pathlib import Path
import time

from cpu_planner import make_rank, requests
from replay import fit

from art.trainer_rank import _memory

OUT = Path(__file__).parent


def main():
    rows = [
        r
        for r in json.loads(
            (
                OUT.parents[1] / "tests/unit/fixtures/memory_fixed_slope/sweeps.json"
            ).read_text()
        )
        if (r["model"], r["operation"], r["lora_rank"]) == (0, "forward_backward", 8)
    ]
    inputs = requests()
    results = {}
    for version in ("old", "new"):
        rank = make_rank(lora_rank=8)
        seed = rank._plan_flat_forward(inputs[:32])
        if version == "new":
            fit(rank, seed.signature, rows)
        else:
            for row in rows:
                from types import SimpleNamespace

                plan = SimpleNamespace(
                    signature=seed.signature,
                    packed_tokens=row["length"],
                    active_logical_tokens=row["length"],
                    output_bytes=4 * row["length"],
                )
                rank._update_memory_profile(
                    plan, row["measured_increment_bytes"], retained_bytes=None
                )
        # Same tiny workspace observation and trust extent in each version.
        from types import SimpleNamespace

        tiny = SimpleNamespace(
            signature=seed.signature,
            packed_tokens=8,
            active_logical_tokens=8,
            output_bytes=0,
        )
        rank._update_memory_profile(tiny, 16_000_000, retained_bytes=None)
        prices = []
        for width in (32, 64, 128, 256, 512, 568):
            plan = rank._plan_flat_forward(inputs[:width])
            check = rank._memory_check(plan)
            prices.append(
                dict(
                    width=width,
                    packed_tokens=plan.packed_tokens,
                    logical_tokens=plan.active_logical_tokens,
                    required_bytes=check.estimated_required_bytes,
                    usable_bytes=check.available_bytes,
                    fits=check.fits,
                )
            )
        start = time.perf_counter()
        selected = rank._select_next_micro_batch(inputs, 0)
        results[version] = dict(
            selected_width=selected.stats_global_count,
            seconds=time.perf_counter() - start,
            prices=prices,
            selected_packed_tokens=selected.plan.packed_tokens,
            selected_required_bytes=selected.check.estimated_required_bytes,
            profile=vars(rank._memory_profiles[seed.signature]),
        )
        print(version, results[version]["selected_width"], flush=True)
    result = dict(
        approximate=True,
        description=__doc__,
        usable_bytes=24 * 2**30,
        lora_rank=8,
        cp=8,
        results=results,
    )
    (OUT / "wave.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v["selected_width"] for k, v in results.items()}), flush=True)


if __name__ == "__main__":
    main()
