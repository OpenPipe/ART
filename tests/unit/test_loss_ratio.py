from typing import Literal

import pytest
import torch

from art.loss import AlignedLossInputs, loss_fn

ImportanceSamplingLevel = Literal["token", "sequence", "average", "geometric_average"]


def _inputs(old_logprobs: torch.Tensor) -> AlignedLossInputs:
    return AlignedLossInputs(
        assistant_mask=torch.ones_like(old_logprobs, dtype=torch.bool),
        old_logprobs=old_logprobs,
        advantages=-torch.ones_like(old_logprobs),
        weights=torch.ones_like(old_logprobs),
        group_ids=torch.ones_like(old_logprobs, dtype=torch.long),
    )


@pytest.mark.parametrize(
    "old_logprob", [-float("inf"), -9999.0, -101.0, float("inf"), float("nan")]
)
@pytest.mark.parametrize("level", ["token", "sequence", "average", "geometric_average"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ppo_extreme_old_logprobs_have_finite_loss_and_gradients(
    old_logprob: float, level: ImportanceSamplingLevel, dtype: torch.dtype
) -> None:
    old = torch.tensor([[old_logprob, -1.0]], dtype=dtype)
    new = torch.full_like(old, -1.0, requires_grad=True)
    result = loss_fn(
        _inputs(old),
        new,
        None,
        None,
        {"ppo": True, "importance_sampling_level": level},
    )
    result.policy_loss.backward()
    assert torch.isfinite(result.policy_loss)
    assert new.grad is not None
    assert torch.isfinite(new.grad).all()
    assert result.policy_loss.dtype == dtype


@pytest.mark.parametrize("ppo", [False, True])
@pytest.mark.parametrize("level", ["token", "sequence", "average", "geometric_average"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_normal_ratios_preserve_loss_and_gradients_exactly(
    ppo: bool, level: ImportanceSamplingLevel, dtype: torch.dtype
) -> None:
    old = torch.tensor([[-0.5, -1.0, -3.0, -5.0]], dtype=dtype)
    new = torch.tensor([[-0.4, -1.5, -2.0, -6.0]], dtype=dtype, requires_grad=True)
    inputs = _inputs(old)
    inputs.advantages = torch.tensor([[1.0, -1.0, 0.5, -0.5]], dtype=dtype)
    expected_new = new.detach().clone().requires_grad_()
    difference = expected_new - old
    ratio = difference.exp()
    if level != "token":
        sequence_ratio = inputs.group_mean(difference, by=inputs.group_ids).exp()
        if level == "sequence":
            ratio = sequence_ratio
        elif level == "average":
            ratio = (ratio + sequence_ratio) / 2
        else:
            ratio = (ratio**0.5) * (sequence_ratio**0.5)
    if ppo:
        expected = -torch.min(
            ratio * inputs.advantages, ratio.clamp(0.8, 1.2) * inputs.advantages
        ).mean()
    else:
        expected = -(
            ratio.detach().clamp(0.0, 5.0) * inputs.advantages * expected_new
        ).mean()
    actual = loss_fn(
        inputs, new, None, None, {"ppo": ppo, "importance_sampling_level": level}
    )
    actual.policy_loss.backward()
    expected.backward()
    assert torch.equal(actual.policy_loss, expected)
    assert torch.equal(new.grad, expected_new.grad)


@pytest.mark.parametrize("level", ["token", "sequence", "average", "geometric_average"])
def test_ignored_extreme_logprobs_do_not_affect_active_tokens(
    level: ImportanceSamplingLevel,
) -> None:
    old = torch.tensor([[-1.0, -float("inf"), -9999.0]])
    new = torch.tensor([[-1.0, float("nan"), -float("inf")]], requires_grad=True)
    inputs = _inputs(old)
    inputs.assistant_mask = torch.tensor([[True, False, False]])
    result = loss_fn(
        inputs, new, None, None, {"ppo": True, "importance_sampling_level": level}
    )
    result.policy_loss.backward()
    assert result.policy_loss.item() == 1.0
    assert torch.equal(new.grad, torch.tensor([[1.0, 0.0, 0.0]]))


def test_sequence_ratio_aggregates_before_clamping() -> None:
    old = torch.tensor([[-32.0, -1.0]])
    new = torch.tensor([[-2.0, -11.0]], requires_grad=True)
    result = loss_fn(
        _inputs(old),
        new,
        None,
        None,
        {"ppo": True, "importance_sampling_level": "sequence"},
    )
    assert result.policy_loss.item() == torch.exp(torch.tensor(10.0)).item()
    result.policy_loss.backward()
    torch.testing.assert_close(
        new.grad, torch.full_like(new, result.policy_loss.item() / 2)
    )


def test_sequence_ratio_handles_both_infinity_signs() -> None:
    old = torch.tensor([[-float("inf"), float("inf")]])
    new = torch.full_like(old, -1.0, requires_grad=True)
    result = loss_fn(
        _inputs(old),
        new,
        None,
        None,
        {"ppo": True, "importance_sampling_level": "sequence"},
    )
    assert result.policy_loss.item() == 1.0
    result.policy_loss.backward()
    assert torch.equal(new.grad, torch.full_like(new, 0.5))
