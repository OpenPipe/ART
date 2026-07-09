import pytest
import torch

from art.loss import (
    LossOffPolicyDiagnostics,
    LossOffPolicyDiagnosticsAccumulator,
)


def test_offpolicy_diagnostics_accumulates_cispo_metrics() -> None:
    ratios = torch.tensor([0.5, 1.0, 2.0, 6.0])
    diagnostics = LossOffPolicyDiagnostics.from_tensors(
        prob_ratio=ratios,
        advantages=torch.ones_like(ratios),
        assistant_mask=torch.ones_like(ratios),
        weights=torch.ones_like(ratios),
        ppo=False,
        epsilon=1.0,
        epsilon_high=4.0,
    )
    accumulator = LossOffPolicyDiagnosticsAccumulator()
    accumulator.add(diagnostics)

    metrics = accumulator.to_metrics()

    assert metrics["loss/importance_ratio_mean"] == pytest.approx(2.375)
    assert metrics["loss/clipped_token_fraction"] == pytest.approx(0.25)
    assert metrics["loss/importance_ratio_p99"] >= metrics["loss/importance_ratio_p95"]
    assert metrics["loss/importance_ratio_p95"] > 2.0


def test_offpolicy_diagnostics_uses_ppo_advantage_sign_for_clipping() -> None:
    diagnostics = LossOffPolicyDiagnostics.from_tensors(
        prob_ratio=torch.tensor([1.3, 0.7, 1.3, 0.7]),
        advantages=torch.tensor([1.0, -1.0, -1.0, 1.0]),
        assistant_mask=torch.ones(4),
        weights=torch.ones(4),
        ppo=True,
        epsilon=0.2,
        epsilon_high=0.2,
    )
    accumulator = LossOffPolicyDiagnosticsAccumulator()
    accumulator.add(diagnostics)

    metrics = accumulator.to_metrics()

    assert metrics["loss/clipped_token_fraction"] == pytest.approx(0.5)


def test_offpolicy_diagnostics_ignores_masked_tokens() -> None:
    diagnostics = LossOffPolicyDiagnostics.from_tensors(
        prob_ratio=torch.tensor([1.0, 100.0]),
        advantages=torch.ones(2),
        assistant_mask=torch.tensor([1.0, 1.0]),
        weights=torch.tensor([1.0, 0.0]),
        ppo=False,
        epsilon=1.0,
        epsilon_high=4.0,
    )
    accumulator = LossOffPolicyDiagnosticsAccumulator()
    accumulator.add(diagnostics)

    metrics = accumulator.to_metrics()

    assert metrics["loss/importance_ratio_mean"] == pytest.approx(1.0)
    assert metrics["loss/clipped_token_fraction"] == 0.0
