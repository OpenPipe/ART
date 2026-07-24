from __future__ import annotations

from dataclasses import fields
import math

import pytest
import torch

from art import dev
from art.loss import AlignedLossInputs, loss_fn
from art.megatron.composite_loss import (
    CompositePreparedLoss,
    PreparedDistillationSidecars,
    PreparedPolicySidecars,
    composite_prepared_loss_from_logits,
)


def _logits(*, physical_vocab_size: int = 8) -> torch.Tensor:
    logical = torch.tensor(
        [
            [
                [0.3, -0.2, 1.1, 0.7, -0.8, 0.4],
                [-0.6, 0.9, 0.1, 1.3, -0.4, 0.2],
                [1.0, 0.5, -0.3, 0.2, 0.8, -0.7],
            ],
            [
                [0.4, 1.2, -0.5, 0.3, -0.2, 0.6],
                [-0.1, 0.2, 0.8, -0.9, 1.4, 0.5],
                [0.7, -0.4, 0.3, 1.0, 0.1, -0.6],
            ],
        ],
        dtype=torch.float64,
    )
    if physical_vocab_size == logical.shape[-1]:
        return logical
    padding = torch.tensor(
        [[[100.0, -100.0]]],
        dtype=torch.float64,
    ).expand(2, 3, physical_vocab_size - logical.shape[-1])
    return torch.cat((logical, padding), dim=-1)


def _policy(
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> PreparedPolicySidecars:
    sampled = torch.tensor([[2, 3, 0], [1, 4, 3]])
    old_logprobs = torch.tensor(
        [[-1.0, -0.7, -1.4], [-0.8, -0.9, -0.6]],
        dtype=torch.float64,
    )
    advantages = torch.tensor(
        [[1.2, -0.5, 0.7], [-1.1, 0.4, 1.5]],
        dtype=torch.float64,
    )
    return PreparedPolicySidecars(
        sampled_token_ids=sampled,
        old_logprobs=old_logprobs,
        advantages=advantages,
        weights=(
            weights
            if weights is not None
            else torch.tensor(
                [[1.0, 0.25, 1.5], [0.8, 2.0, 0.4]],
                dtype=torch.float64,
            )
        ),
        mask=mask,
        group_ids=torch.tensor([[1, 1, 1], [2, 2, 2]]),
        original_logprobs=old_logprobs - 0.2,
        ref_logprobs=old_logprobs - 0.1,
    )


def _distillation(
    mask: torch.Tensor,
    *,
    temperature: float = 1.7,
    weights: torch.Tensor | None = None,
) -> PreparedDistillationSidecars:
    teacher_ids = torch.tensor(
        [
            [[2, 3], [3, 1], [0, 4]],
            [[1, 5], [4, 2], [3, 0]],
        ]
    )
    selected_probs = torch.tensor(
        [
            [[0.42, 0.23], [0.51, 0.12], [0.34, 0.26]],
            [[0.47, 0.18], [0.39, 0.21], [0.44, 0.16]],
        ],
        dtype=torch.float64,
    )
    tail_probs = 1.0 - selected_probs.sum(dim=-1)
    return PreparedDistillationSidecars(
        teacher_topk_ids=teacher_ids,
        teacher_topk_logprobs=selected_probs.log(),
        teacher_tail_logprob=tail_probs.log(),
        mask=mask,
        weights=(
            weights
            if weights is not None
            else torch.tensor(
                [[0.5, 1.0, 2.0], [0.75, 1.5, 0.25]],
                dtype=torch.float64,
            )
        ),
        temperature=temperature,
        compensate_temperature_squared=True,
    )


def _gradient(
    *,
    policy: PreparedPolicySidecars | None,
    distillation: PreparedDistillationSidecars | None,
    coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, CompositePreparedLoss]:
    logits = _logits().requires_grad_(True)
    result = composite_prepared_loss_from_logits(
        logits,
        logical_vocab_size=6,
        policy=policy,
        distillation=distillation,
        policy_config={
            "epsilon": 0.2,
            "epsilon_high": 0.4,
            "importance_sampling_level": "token",
        },
        distillation_coefficient=coefficient,
    )
    result.loss.backward()
    assert logits.grad is not None
    return result.loss.detach(), logits.grad.detach(), result


@pytest.mark.parametrize(
    ("policy_mask", "kd_mask"),
    [
        (
            torch.tensor([[True, False, True], [True, False, False]]),
            torch.tensor([[False, True, False], [False, True, True]]),
        ),
        (
            torch.tensor([[True, True, False], [False, True, False]]),
            torch.tensor([[False, True, True], [False, True, False]]),
        ),
    ],
    ids=("disjoint", "overlapping"),
)
def test_additive_gradient_is_sum_of_independently_normalized_gradients(
    policy_mask: torch.Tensor,
    kd_mask: torch.Tensor,
) -> None:
    coefficient = 0.65
    policy_loss, policy_grad, policy_result = _gradient(
        policy=_policy(policy_mask),
        distillation=None,
        coefficient=coefficient,
    )
    kd_loss, kd_grad, kd_result = _gradient(
        policy=None,
        distillation=_distillation(kd_mask),
        coefficient=coefficient,
    )
    combined_loss, combined_grad, combined = _gradient(
        policy=_policy(policy_mask),
        distillation=_distillation(kd_mask),
        coefficient=coefficient,
    )

    torch.testing.assert_close(combined_loss, policy_loss + kd_loss)
    torch.testing.assert_close(combined_grad, policy_grad + kd_grad)
    assert combined.policy is not None
    assert combined.distillation is not None
    assert policy_result.policy is not None
    assert kd_result.distillation is not None
    assert combined.policy.token_count.item() == int(policy_mask.sum())
    assert combined.distillation.token_count.item() == int(kd_mask.sum())
    assert torch.isfinite(combined_grad).all()


def test_unequal_counts_and_weights_do_not_cross_normalize_components() -> None:
    policy_mask = torch.tensor([[True, True, True], [True, False, False]])
    kd_mask = torch.tensor([[False, True, False], [False, False, True]])
    coefficient = 0.3
    logits = _logits().requires_grad_(True)
    result = composite_prepared_loss_from_logits(
        logits,
        logical_vocab_size=6,
        policy=_policy(
            policy_mask,
            weights=torch.tensor(
                [[3.0, 0.1, 2.0], [0.25, 99.0, 99.0]],
                dtype=torch.float64,
            ),
        ),
        distillation=_distillation(
            kd_mask,
            weights=torch.tensor(
                [[99.0, 0.5, 99.0], [99.0, 99.0, 4.0]],
                dtype=torch.float64,
            ),
        ),
        distillation_coefficient=coefficient,
    )

    assert result.policy is not None
    assert result.distillation is not None
    assert result.policy.token_count.item() == 4
    assert result.distillation.token_count.item() == 2
    expected = (
        result.policy.loss_sum / 4 + coefficient * result.distillation.loss_sum / 2
    )
    torch.testing.assert_close(result.loss, expected)


def test_every_enabled_objective_requires_a_nonzero_denominator() -> None:
    empty = torch.zeros((2, 3), dtype=torch.bool)
    active = torch.tensor([[True, False, False], [False, False, False]])
    with pytest.raises(ValueError, match="policy objective.*zero"):
        composite_prepared_loss_from_logits(
            _logits(),
            logical_vocab_size=6,
            policy=_policy(empty),
        )
    with pytest.raises(ValueError, match="distillation mask selects zero"):
        composite_prepared_loss_from_logits(
            _logits(),
            logical_vocab_size=6,
            distillation=_distillation(empty),
        )
    with pytest.raises(ValueError, match="distillation mask selects zero"):
        composite_prepared_loss_from_logits(
            _logits(),
            logical_vocab_size=6,
            policy=_policy(active),
            distillation=_distillation(empty),
        )


def _sidecar_bytes(sidecars: PreparedPolicySidecars) -> tuple[bytes | None, ...]:
    return tuple(
        (
            value.detach().cpu().contiguous().numpy().tobytes()
            if isinstance(value, torch.Tensor)
            else None
        )
        for field in fields(sidecars)
        if (value := getattr(sidecars, field.name)) is not None
    )


def test_teacher_failure_clears_only_kd_and_preserves_policy_bytes_and_gradient() -> (
    None
):
    policy_mask = torch.tensor([[True, True, False], [True, False, True]])
    full_kd_mask = torch.tensor([[True, True, False], [False, True, True]])
    failed_kd_mask = torch.tensor([[True, False, False], [False, True, False]])
    policy = _policy(policy_mask)
    policy_bytes = _sidecar_bytes(policy)

    policy_gradients = []
    policy_sums = []
    for kd_mask in (full_kd_mask, failed_kd_mask):
        logits = _logits().requires_grad_(True)
        result = composite_prepared_loss_from_logits(
            logits,
            logical_vocab_size=6,
            policy=policy,
            distillation=_distillation(kd_mask),
            distillation_coefficient=0.4,
        )
        assert result.policy is not None
        policy_loss = result.policy_loss
        assert policy_loss is not None
        policy_gradient = torch.autograd.grad(policy_loss, logits)[0]
        policy_gradients.append(policy_gradient)
        policy_sums.append(result.policy.loss_sum.detach())
        assert _sidecar_bytes(policy) == policy_bytes
        assert result.policy.token_count.item() == int(policy_mask.sum())

    torch.testing.assert_close(policy_sums[0], policy_sums[1])
    torch.testing.assert_close(policy_gradients[0], policy_gradients[1])


def test_physical_vocab_padding_is_ignored_by_both_objectives() -> None:
    policy_mask = torch.tensor([[True, False, True], [False, True, False]])
    kd_mask = torch.tensor([[False, True, False], [True, False, True]])

    padded = _logits(physical_vocab_size=8).requires_grad_(True)
    padded_result = composite_prepared_loss_from_logits(
        padded,
        logical_vocab_size=6,
        policy=_policy(policy_mask),
        distillation=_distillation(kd_mask),
        distillation_coefficient=0.8,
    )
    padded_result.loss.backward()

    logical = _logits(physical_vocab_size=6).requires_grad_(True)
    logical_result = composite_prepared_loss_from_logits(
        logical,
        logical_vocab_size=6,
        policy=_policy(policy_mask),
        distillation=_distillation(kd_mask),
        distillation_coefficient=0.8,
    )
    logical_result.loss.backward()

    torch.testing.assert_close(padded_result.loss, logical_result.loss)
    assert padded.grad is not None
    assert logical.grad is not None
    torch.testing.assert_close(padded.grad[..., :6], logical.grad)
    assert torch.count_nonzero(padded.grad[..., 6:]).item() == 0
    assert torch.isfinite(padded.grad).all()


def test_kd_temperature_never_changes_cispo_scores_or_gradient() -> None:
    policy_mask = torch.tensor([[True, False, True], [False, True, False]])
    kd_mask = torch.tensor([[False, True, False], [True, False, True]])
    policy_scores = []
    policy_gradients = []
    for temperature in (0.6, 2.4):
        logits = _logits().requires_grad_(True)
        result = composite_prepared_loss_from_logits(
            logits,
            logical_vocab_size=6,
            policy=_policy(policy_mask),
            distillation=_distillation(kd_mask, temperature=temperature),
        )
        assert result.policy is not None
        policy_scores.append(result.policy.new_logprobs.detach())
        policy_loss = result.policy_loss
        assert policy_loss is not None
        policy_gradients.append(torch.autograd.grad(policy_loss, logits)[0])

    torch.testing.assert_close(policy_scores[0], policy_scores[1])
    torch.testing.assert_close(policy_gradients[0], policy_gradients[1])


def test_cispo_component_matches_legacy_loss_numerically() -> None:
    mask = torch.tensor([[True, True, False], [True, True, False]])
    sidecars = _policy(mask)
    config: dev.TrainConfig = {
        "epsilon": 0.25,
        "epsilon_high": 0.5,
        "importance_sampling_level": "sequence",
        "kimi_k2_tau": 0.15,
        "kl_penalty_coef": 0.2,
        "kl_penalty_source": "sample",
        "truncated_importance_sampling": 1.8,
    }

    actual_logits = _logits().requires_grad_(True)
    actual = composite_prepared_loss_from_logits(
        actual_logits,
        logical_vocab_size=6,
        policy=sidecars,
        policy_config=config,
    )
    actual.loss.backward()

    reference_logits = _logits().requires_grad_(True)
    sampled = sidecars.sampled_token_ids.unsqueeze(-1)
    reference_new_logprobs = (
        torch.log_softmax(
            reference_logits[..., :6].to(torch.float32),
            dim=-1,
        )
        .gather(-1, sampled)
        .squeeze(-1)
    )
    reference = loss_fn(
        AlignedLossInputs(
            assistant_mask=sidecars.mask,
            old_logprobs=sidecars.old_logprobs,
            advantages=sidecars.advantages,
            weights=sidecars.weights,
            group_ids=sidecars.group_ids,
            original_logprobs=sidecars.original_logprobs,
            entropies_are_aligned=True,
        ),
        new_logprobs=reference_new_logprobs,
        ref_logprobs=sidecars.ref_logprobs,
        entropies=None,
        experimental_config={**config, "ppo": False},
        reduction="sum",
    )
    reference_loss = reference.policy_loss_sum / int(mask.sum())
    reference_loss.backward()

    assert actual.policy is not None
    torch.testing.assert_close(actual.policy.loss_sum, reference.policy_loss_sum)
    torch.testing.assert_close(actual.loss, reference_loss)
    assert actual_logits.grad is not None
    assert reference_logits.grad is not None
    torch.testing.assert_close(actual_logits.grad, reference_logits.grad)
    assert actual.policy.details.offpolicy_diagnostics is not None
    assert math.isfinite(actual.policy.details.probs_corr.item())
