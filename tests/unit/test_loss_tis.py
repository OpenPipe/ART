from typing import Literal
import warnings

import pytest
import torch

from art import dev
from art.loss import AlignedLossInputs, LossInputs, loss_fn

ImportanceSamplingLevel = Literal["token", "sequence", "average", "geometric_average"]


def _inputs(original_logprobs: torch.Tensor | None = None) -> AlignedLossInputs:
    return AlignedLossInputs(
        assistant_mask=torch.ones(1, 3, dtype=torch.bool),
        old_logprobs=torch.full((1, 3), -2.0),
        advantages=torch.tensor([[1.0, -1.0, -0.5]]),
        weights=torch.tensor([[1.0, 0.5, 2.0]]),
        group_ids=torch.ones(1, 3, dtype=torch.long),
        original_logprobs=original_logprobs,
    )


@pytest.mark.parametrize("ppo", [False, True])
@pytest.mark.parametrize(
    "level", ["token", "sequence", "average", "geometric_average"]
)
def test_tis_without_original_logprobs_preserves_loss_and_gradient(
    ppo: bool, level: ImportanceSamplingLevel
) -> None:
    inputs = _inputs()
    plain_logprobs = torch.tensor([[-1.0, -1.5, -2.5]], requires_grad=True)
    tis_logprobs = plain_logprobs.detach().clone().requires_grad_()
    config: dev.TrainConfig = {"ppo": ppo, "importance_sampling_level": level}
    plain = loss_fn(inputs, plain_logprobs, None, None, config)
    with pytest.warns(UserWarning, match="original_logprobs"):
        tis = loss_fn(
            inputs,
            tis_logprobs,
            None,
            None,
            {**config, "truncated_importance_sampling": 2.0},
        )
    plain.policy_loss.backward()
    tis.policy_loss.backward()
    assert torch.equal(tis.policy_loss, plain.policy_loss)
    assert torch.equal(tis_logprobs.grad, plain_logprobs.grad)


@pytest.mark.parametrize("ppo", [False, True])
def test_tis_uses_trainer_sampler_ratio_when_original_logprobs_exist(
    ppo: bool,
) -> None:
    inputs = _inputs(torch.tensor([[-3.0, float("nan"), -2.0]]))
    new_logprobs = torch.full((1, 3), -1.0, requires_grad=True)
    ratio = torch.exp(new_logprobs - inputs.old_logprobs)
    if ppo:
        per_token_loss = -torch.min(
            ratio * inputs.advantages,
            ratio.clamp(0.8, 1.2) * inputs.advantages,
        )
    else:
        per_token_loss = -ratio.detach().clamp(0.0, 5.0) * inputs.advantages * new_logprobs
    correction = torch.tensor([[2.0, torch.exp(torch.tensor(-1.0)).item(), 1.0]])
    expected = (per_token_loss * correction * inputs.weights).mean()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = loss_fn(
            inputs,
            new_logprobs,
            None,
            None,
            {"ppo": ppo, "truncated_importance_sampling": 2.0},
        )
    assert not caught
    torch.testing.assert_close(actual.policy_loss, expected)
    actual_gradient = torch.autograd.grad(actual.policy_loss, new_logprobs)[0]
    expected_gradient = torch.autograd.grad(expected, new_logprobs)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_missing_tis_logprobs_warn_once_with_default_warning_filter() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            loss_fn(
                _inputs(),
                torch.full((1, 3), -1.0),
                None,
                None,
                {"truncated_importance_sampling": 2.0},
            )
    assert len(caught) == 1
    assert "original_logprobs" in str(caught[0].message)


@pytest.mark.parametrize("upper_bound", [None, 0.0])
def test_disabled_tis_does_not_warn(upper_bound: float | None) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss_fn(
            _inputs(),
            torch.full((1, 3), -1.0),
            None,
            None,
            {"truncated_importance_sampling": upper_bound},
        )
    assert not caught


def test_packed_inputs_without_original_logprobs_skip_tis() -> None:
    inputs = LossInputs(
        inputs={
            "assistant_mask": torch.tensor([[False, True]]),
            "logprobs": torch.tensor([[float("nan"), -2.0]]),
            "advantages": torch.tensor([[0.0, -1.0]]),
            "weights": torch.ones(1, 2),
            "group_ids": torch.tensor([[0, 1]]),
        }
    )
    new_logprobs = torch.tensor([[-1.0, 0.0]], requires_grad=True)
    plain = loss_fn(inputs, new_logprobs, None, None, {})
    with pytest.warns(UserWarning, match="original_logprobs"):
        tis = loss_fn(
            inputs,
            new_logprobs,
            None,
            None,
            {"truncated_importance_sampling": 2.0},
        )
    assert torch.equal(tis.policy_loss, plain.policy_loss)
