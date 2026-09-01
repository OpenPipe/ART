from __future__ import annotations

from typing import Any, Literal


def precision(handler: Any) -> Literal["bf16", "fp32"]:
    return "bf16" if type(handler).__name__ in {"Dsv4Handler", "Glm52Handler"} else "fp32"


def use_fp32_lora_reference(handler: Any) -> bool:
    return precision(handler) == "fp32"


def phase_pass_fns(handler: Any, oracle_harness: Any) -> dict[str, Any] | None:
    if precision(handler) == "fp32":
        return None
    nonzero = {"typical_abs_scale": 0.0, "candidate_abs_scale": 0.0}
    forward = oracle_harness.MetricThresholdRule(
        limits={"mean_abs_pct": 3.0}, minimums=nonzero
    )
    gradients = oracle_harness.MetricThresholdRule(
        limits={"mean_abs_pct": 5.0}, minimums=nonzero
    )
    return {
        "forward": forward,
        "outputs": forward,
        "losses": oracle_harness.MetricThresholdRule(limits={"mean_abs_pct": 3.0}),
        "grads": gradients,
        "deltas": gradients,
        "router_scores": forward,
        "router_topk_ids": oracle_harness.MetricThresholdRule(
            limits={"topk_mismatch_fraction": 0.0, "top1_mismatch_fraction": 0.0}
        ),
    }
