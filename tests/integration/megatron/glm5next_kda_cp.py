"""M4 single-root envelope around M2's unchanged CP8 worker and kernels."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch

from tests.integration.megatron import kda_parity as lane


def cp_checks(module: Any, x: torch.Tensor, cu: torch.Tensor, group: Any) -> dict:
    """M2's packed CP-vs-CP1 case, without constructing a branching tree."""
    world, rank = lane.dist.get_world_size(group), lane.dist.get_rank(group)
    module.cp_group = None
    full = copy.deepcopy(module)
    module.cp_group = group
    lane.adapt(module)
    lane.adapt(full, source=module)
    local = x.chunk(world)[rank].clone().requires_grad_()
    probe = torch.randn_like(x, dtype=torch.float32)
    actual = module(local, packed_seq_params=lane.packed(cu))[0]
    expected = full(x, packed_seq_params=lane.packed(cu))[0]
    output = lane.metrics(actual, expected.chunk(world)[rank])
    gradients = lane.gradients(module, actual, probe.chunk(world)[rank])
    expected_gradients = lane.gradients(full, expected, probe)
    comparisons = {}
    for name, value in gradients.items():
        lane.dist.all_reduce(value, group=group)
        comparisons[name] = lane.metrics(value, expected_gradients[name])
    return {
        "gate": "rel-l2-1e-3-no-envelope-fallback",
        "packing": "single-root",
        "tree_case": "not run: post-MVP",
        "world_size": world,
        "rank": rank,
        "output": output,
        "gradients": comparisons,
    }


def gate_policy(record: dict, world: int) -> None:
    assert world > 1 and record["packing"] == "single-root"
    record["output_pass"] = record["output"]["rel_l2"] <= 1e-3
    record["gradients_pass"] = all(
        v["rel_l2"] <= 1e-3 for v in record["gradients"].values()
    )
    output = record["output"]
    record["blocking_pass"] = (
        all(math.isfinite(value) for value in output.values())
        and output["rel_l2"] <= 1e-2
        and output["cosine"] >= 0.999
    )
    record["report_only_pass"] = record["output_pass"] and record["gradients_pass"]
    record["pass"] = record["blocking_pass"] and record["report_only_pass"]
    record["output_policy"] = "report-only; original 1e-3 threshold retained"
    record["hard_bug_policy"] = "blocking: rel-L2 >1e-2, cosine <0.999 or nonfinite"
    record["gradient_policy"] = "report-only; original 1e-3 threshold retained"


def main() -> None:
    # The lane's parser, loading, CP communication and result aggregation stay intact.
    native_lane: Any = lane
    original_run = native_lane.run

    def run(args: Any) -> dict:
        native_lane.cp_checks = cp_checks
        native_lane.apply_gate_policy = gate_policy
        result = original_run(args)
        result["packing"] = "single-root"
        return result

    native_lane.run = run
    native_lane.main()


if __name__ == "__main__":
    main()
