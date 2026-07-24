"""Pure policy-plus-distillation loss composition for prepared Megatron data.

This module owns no packing, distributed collectives, optimizer state, or model
lifecycle.  It provides the TP=1 numerical contract that a later
vocabulary-parallel scorer can satisfy by producing the same objective
components from selected-token and tail log-probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import torch

from art import dev
from art.loss import AlignedLossInputs, Loss, loss_fn

from .distillation_loss import (
    TopKPlusTailForwardKLResult,
    topk_plus_tail_forward_kl,
)


@dataclass(frozen=True)
class PreparedPolicySidecars:
    """Independent token-aligned inputs for the CISPO objective."""

    sampled_token_ids: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor
    mask: torch.Tensor
    group_ids: torch.Tensor
    original_logprobs: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None


@dataclass(frozen=True)
class PreparedDistillationSidecars:
    """Independent token-aligned top-k-plus-tail teacher targets."""

    teacher_topk_ids: torch.Tensor
    teacher_topk_logprobs: torch.Tensor
    teacher_tail_logprob: torch.Tensor
    mask: torch.Tensor
    weights: torch.Tensor | None = None
    temperature: float = 1.0
    compensate_temperature_squared: bool = False


@dataclass(frozen=True)
class PreparedPolicyLoss:
    """Unreduced CISPO component and its student-score diagnostics."""

    loss_sum: torch.Tensor
    token_count: torch.Tensor
    new_logprobs: torch.Tensor
    details: Loss


@dataclass(frozen=True)
class PreparedDistillationLoss:
    """Unreduced KD component produced by any compatible student scorer."""

    loss_sum: torch.Tensor
    token_count: torch.Tensor
    details: TopKPlusTailForwardKLResult


@dataclass(frozen=True)
class CompositePreparedLoss:
    """Normalized policy/KD objective with independent component denominators."""

    loss: torch.Tensor
    policy_loss: torch.Tensor | None
    distillation_loss: torch.Tensor | None
    policy: PreparedPolicyLoss | None
    distillation: PreparedDistillationLoss | None
    distillation_coefficient: float


def prepared_cispo_loss_from_logprobs(
    new_logprobs: torch.Tensor,
    *,
    sidecars: PreparedPolicySidecars,
    config: dev.TrainConfig | None = None,
) -> PreparedPolicyLoss:
    """Evaluate legacy CISPO semantics from already selected student scores.

    This is the objective-composition seam for a future vocabulary-parallel
    scorer.  Such a scorer need only supply sampled-token log-probabilities at
    temperature one; it does not need to reproduce composition or reduction.
    """

    row_shape = sidecars.mask.shape
    if sidecars.mask.dtype != torch.bool:
        raise ValueError("policy mask must be boolean")
    if new_logprobs.shape != row_shape:
        raise ValueError("new_logprobs must match the policy token-row dimensions")
    _validate_policy_sidecars(sidecars, row_shape=row_shape, device=new_logprobs.device)

    active_count = int(sidecars.mask.sum().item())
    if active_count == 0:
        raise ValueError("enabled policy objective has a zero token denominator")

    resolved_config = cast(dev.TrainConfig, dict(config or {}))
    if resolved_config.get("ppo", False):
        raise ValueError("prepared policy composition supports CISPO, not PPO")
    resolved_config["ppo"] = False

    mask = sidecars.mask
    sanitized_new = torch.where(mask, new_logprobs, new_logprobs.new_zeros(()))
    sanitized_old = torch.where(
        mask,
        sidecars.old_logprobs,
        sidecars.old_logprobs.new_zeros(()),
    )
    sanitized_advantages = torch.where(
        mask,
        sidecars.advantages,
        sidecars.advantages.new_zeros(()),
    )
    sanitized_weights = torch.where(
        mask,
        sidecars.weights,
        sidecars.weights.new_zeros(()),
    )
    sanitized_groups = torch.where(
        mask,
        sidecars.group_ids,
        sidecars.group_ids.new_zeros(()),
    )
    sanitized_original = (
        torch.where(
            mask,
            sidecars.original_logprobs,
            sidecars.original_logprobs.new_zeros(()),
        )
        if sidecars.original_logprobs is not None
        else None
    )
    sanitized_ref = (
        torch.where(
            mask,
            sidecars.ref_logprobs,
            sidecars.ref_logprobs.new_zeros(()),
        )
        if sidecars.ref_logprobs is not None
        else None
    )

    details = loss_fn(
        AlignedLossInputs(
            assistant_mask=mask,
            old_logprobs=sanitized_old,
            advantages=sanitized_advantages,
            weights=sanitized_weights,
            group_ids=sanitized_groups,
            original_logprobs=sanitized_original,
            entropies_are_aligned=True,
        ),
        new_logprobs=sanitized_new,
        ref_logprobs=sanitized_ref,
        entropies=None,
        experimental_config=resolved_config,
        reduction="sum",
    )
    if not bool(torch.isfinite(details.policy_loss_sum.detach()).item()):
        raise ValueError("CISPO produced a non-finite unreduced loss")
    return PreparedPolicyLoss(
        loss_sum=details.policy_loss_sum,
        token_count=torch.tensor(
            active_count,
            dtype=torch.int64,
            device=new_logprobs.device,
        ),
        new_logprobs=sanitized_new,
        details=details,
    )


def compose_prepared_objective_losses(
    *,
    policy: PreparedPolicyLoss | None,
    distillation: PreparedDistillationLoss | None,
    distillation_coefficient: float = 1.0,
) -> CompositePreparedLoss:
    """Normalize and add enabled objective components exactly once."""

    if policy is None and distillation is None:
        raise ValueError("at least one prepared objective must be enabled")
    if not math.isfinite(distillation_coefficient) or distillation_coefficient <= 0:
        raise ValueError("distillation_coefficient must be finite and positive")

    policy_mean = _normalized_component(policy, label="policy")
    distillation_mean = _normalized_component(
        distillation,
        label="distillation",
    )
    if policy_mean is None:
        assert distillation_mean is not None
        total = distillation_mean * distillation_coefficient
    elif distillation_mean is None:
        total = policy_mean
    else:
        total = policy_mean + distillation_mean * distillation_coefficient
    if not bool(torch.isfinite(total.detach()).item()):
        raise ValueError("composite prepared loss is non-finite")
    return CompositePreparedLoss(
        loss=total,
        policy_loss=policy_mean,
        distillation_loss=distillation_mean,
        policy=policy,
        distillation=distillation,
        distillation_coefficient=distillation_coefficient,
    )


def composite_prepared_loss_from_logits(
    student_logits: torch.Tensor,
    *,
    logical_vocab_size: int,
    policy: PreparedPolicySidecars | None = None,
    distillation: PreparedDistillationSidecars | None = None,
    policy_config: dev.TrainConfig | None = None,
    distillation_coefficient: float = 1.0,
) -> CompositePreparedLoss:
    """Evaluate policy-only, KD-only, or additive loss from one raw-logit tensor.

    CISPO always reads temperature-one sampled-token log-probabilities.  The KD
    scorer independently applies the temperature recorded in its sidecars.
    """

    _validate_student_logits(
        student_logits,
        logical_vocab_size=logical_vocab_size,
    )
    policy_component = None
    if policy is not None:
        new_logprobs = _sampled_token_logprobs(
            student_logits,
            sampled_token_ids=policy.sampled_token_ids,
            mask=policy.mask,
            logical_vocab_size=logical_vocab_size,
        )
        policy_component = prepared_cispo_loss_from_logprobs(
            new_logprobs,
            sidecars=policy,
            config=policy_config,
        )

    distillation_component = None
    if distillation is not None:
        details = topk_plus_tail_forward_kl(
            student_logits,
            teacher_topk_ids=distillation.teacher_topk_ids,
            teacher_topk_logprobs=distillation.teacher_topk_logprobs,
            teacher_tail_logprob=distillation.teacher_tail_logprob,
            mask=distillation.mask,
            weights=distillation.weights,
            logical_vocab_size=logical_vocab_size,
            temperature=distillation.temperature,
            compensate_temperature_squared=(
                distillation.compensate_temperature_squared
            ),
        )
        distillation_component = PreparedDistillationLoss(
            loss_sum=details.loss_sum,
            token_count=details.token_count,
            details=details,
        )

    return compose_prepared_objective_losses(
        policy=policy_component,
        distillation=distillation_component,
        distillation_coefficient=distillation_coefficient,
    )


def _sampled_token_logprobs(
    student_logits: torch.Tensor,
    *,
    sampled_token_ids: torch.Tensor,
    mask: torch.Tensor,
    logical_vocab_size: int,
) -> torch.Tensor:
    row_shape = student_logits.shape[:-1]
    if sampled_token_ids.shape != row_shape:
        raise ValueError("sampled_token_ids must match the logits token rows")
    if mask.shape != row_shape or mask.dtype != torch.bool:
        raise ValueError("policy mask must be boolean and match the logits token rows")
    if sampled_token_ids.dtype == torch.bool or sampled_token_ids.is_floating_point():
        raise ValueError("sampled_token_ids must have an integer dtype")
    if sampled_token_ids.device != student_logits.device or mask.device != (
        student_logits.device
    ):
        raise ValueError("policy score tensors must be on the logits device")

    flat_mask = mask.reshape(-1)
    active_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    if active_indices.numel() == 0:
        raise ValueError("enabled policy objective has a zero token denominator")
    active_ids = (
        sampled_token_ids.reshape(-1)
        .index_select(0, active_indices)
        .to(dtype=torch.int64)
    )
    if bool(((active_ids < 0) | (active_ids >= logical_vocab_size)).any().item()):
        raise ValueError("active sampled token ID exceeds the logical vocabulary")

    physical_vocab_size = student_logits.shape[-1]
    active_logits = (
        student_logits.reshape(-1, physical_vocab_size)[..., :logical_vocab_size]
        .index_select(0, active_indices)
        .to(dtype=torch.float32)
    )
    if not bool(torch.isfinite(active_logits).all().item()):
        raise ValueError("active logical student logits must be finite")
    active_logprobs = torch.log_softmax(active_logits, dim=-1).gather(
        dim=-1,
        index=active_ids.unsqueeze(-1),
    )
    flat_result = student_logits.new_zeros(
        flat_mask.numel(),
        dtype=torch.float32,
    )
    return flat_result.scatter(0, active_indices, active_logprobs.squeeze(-1)).reshape(
        row_shape
    )


def _validate_student_logits(
    student_logits: torch.Tensor,
    *,
    logical_vocab_size: int,
) -> None:
    if student_logits.ndim < 1 or not student_logits.is_floating_point():
        raise ValueError(
            "student_logits must be floating point with a vocabulary dimension"
        )
    if logical_vocab_size <= 1 or logical_vocab_size > student_logits.shape[-1]:
        raise ValueError("logical_vocab_size must be in [2, student_logits.shape[-1]]")


def _validate_policy_sidecars(
    sidecars: PreparedPolicySidecars,
    *,
    row_shape: torch.Size,
    device: torch.device,
) -> None:
    required = {
        "sampled_token_ids": sidecars.sampled_token_ids,
        "old_logprobs": sidecars.old_logprobs,
        "advantages": sidecars.advantages,
        "weights": sidecars.weights,
        "mask": sidecars.mask,
        "group_ids": sidecars.group_ids,
    }
    optional = {
        "original_logprobs": sidecars.original_logprobs,
        "ref_logprobs": sidecars.ref_logprobs,
    }
    for name, tensor in (*required.items(), *optional.items()):
        if tensor is None:
            continue
        if tensor.shape != row_shape:
            raise ValueError(f"{name} must match the policy token-row dimensions")
        if tensor.device != device:
            raise ValueError("all policy sidecars must be on the score device")
    if sidecars.group_ids.dtype == torch.bool or sidecars.group_ids.is_floating_point():
        raise ValueError("group_ids must have an integer dtype")

    mask = sidecars.mask
    active_old = sidecars.old_logprobs[mask]
    if bool(torch.isinf(active_old).any().item()):
        raise ValueError("active old log-probabilities must be finite or NaN")
    for name, tensor in (
        ("advantages", sidecars.advantages),
        ("weights", sidecars.weights),
    ):
        if not bool(torch.isfinite(tensor[mask]).all().item()):
            raise ValueError(f"active policy {name} must be finite")
    if bool((sidecars.weights[mask] < 0).any().item()):
        raise ValueError("active policy weights must be non-negative")
    for name, tensor in optional.items():
        if tensor is not None and bool(torch.isinf(tensor[mask]).any().item()):
            raise ValueError(f"active {name} must be finite or NaN")


def _normalized_component(
    component: PreparedPolicyLoss | PreparedDistillationLoss | None,
    *,
    label: str,
) -> torch.Tensor | None:
    if component is None:
        return None
    count = int(component.token_count.detach().item())
    if count <= 0:
        raise ValueError(f"enabled {label} objective has a zero token denominator")
    if not bool(torch.isfinite(component.loss_sum.detach()).item()):
        raise ValueError(f"enabled {label} objective has a non-finite loss sum")
    return component.loss_sum / count
