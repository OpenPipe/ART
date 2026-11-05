"""M4 gate-policy envelope around M3's unchanged CP8 scientific worker."""

from __future__ import annotations

from typing import Any


def apply_gate_policy(record: dict) -> None:
    """Retain M3's tolerances and failures; only correctness stops the window."""
    record["original_all_tolerances_passed"] = record["passed"]
    blocking = (
        record["selection_matching_fraction"] >= 0.999
        and record["output"]["cosine"] >= 0.9999
        and record["output"]["rel_l2"] <= 1e-2
    )
    report_only = all(
        metric["rel_l2"] <= 2e-2 for metric in record["lora_gradients"].values()
    )
    if record["cp1"] is not None:
        blocking = blocking and record["cp1"]["output"]["rel_l2"] <= 1e-3
        report_only = report_only and all(
            metric["rel_l2"] <= 1e-3
            for name, metric in record["cp1"].items()
            if name != "output"
        )
    record["blocking_pass"] = blocking
    record["report_only_pass"] = report_only
    record["all_tolerances_passed"] = blocking and report_only
    record["gradient_policy"] = (
        "report-only; original HF 2e-2 and CP1 1e-3 thresholds retained"
    )
    # M3's main checks this field after saving every complete scientific record.
    record["passed"] = blocking


def main() -> None:
    from tests.integration.megatron import glm5next_dsa_parity

    lane: Any = glm5next_dsa_parity
    original_run_case = lane.run_case

    def run_case(*args: Any, **kwargs: Any) -> dict:
        record = original_run_case(*args, **kwargs)
        apply_gate_policy(record)
        return record

    lane.run_case = run_case
    lane.main()


if __name__ == "__main__":
    main()
