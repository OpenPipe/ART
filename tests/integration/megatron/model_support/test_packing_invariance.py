from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")

from .packing_invariance import run_packing_invariance


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for packing invariance validation",
)
def test_run_packing_invariance_qwen35() -> None:
    report = run_packing_invariance(
        base_model="Qwen/Qwen3.5-35B-A3B",
    )

    assert len(report.scenarios) == 4
    assert all(scenario.matched for scenario in report.scenarios)
    assert all(scenario.checked_token_count > 0 for scenario in report.scenarios)
    assert all(scenario.prompt_family_count >= 2 for scenario in report.scenarios)
    assert (
        next(
            scenario.max_tree_depth
            for scenario in report.scenarios
            if scenario.name == "deep_nested"
        )
        == 3
    )
    assert all(scenario.rotary_grouping_checked for scenario in report.scenarios)
    assert all(
        scenario.repeated_position_key_count > 0 for scenario in report.scenarios
    )
    assert all(scenario.completion_pair_count > 0 for scenario in report.scenarios)
    assert report.precision == "fp32"
    assert all(
        scenario.logits_mean_abs_pct <= scenario.logits_mean_abs_pct_limit
        for scenario in report.scenarios
    )
