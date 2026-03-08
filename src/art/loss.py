from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
import torch

from art.utils.group_aggregate import group_aggregate

from . import dev

if TYPE_CHECKING:
    from art.unsloth.service import TrainInputs


class Loss(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mean_policy_loss: torch.Tensor
    mean_entropy: torch.Tensor | None
    policy_loss_sum: torch.Tensor
    probs_corr: torch.Tensor
    kl_policy_ref: torch.Tensor | None = None


def loss_fn(
    inputs: "TrainInputs",
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    entropies: torch.Tensor | None,
    experimental_config: dev.TrainConfig,
) -> Loss:
    old_logprobs = shift_tensor(inputs["logprobs"], float("nan"))
    advantages = shift_tensor(inputs["advantages"], 0.0)
    assistant_mask = shift_tensor(inputs["assistant_mask"], False).to(
        new_logprobs.dtype
    )
    weights = shift_tensor(inputs["weights"], 0.0)
    old_logprobs_mask = ~torch.isnan(old_logprobs)
    probs_corr = torch.corrcoef(
        torch.stack(
            [
                torch.exp(old_logprobs[old_logprobs_mask]),
                torch.exp(new_logprobs[old_logprobs_mask]),
            ]
        )
    )[0, 1]
    # Assume missing old logprobs were sampled under the current policy
    old_logprobs = torch.where(
        torch.isnan(old_logprobs),
        new_logprobs.detach(),
        old_logprobs,
    )
    logprob_diff = new_logprobs - old_logprobs
    importance_sampling_level = experimental_config.get(
        "importance_sampling_level", "token"
    )
    prob_ratio = torch.exp(logprob_diff)
    if importance_sampling_level != "token":
        sequence_prob_ratio = torch.exp(
            group_aggregate(
                logprob_diff,
                by=shift_tensor(inputs["group_ids"], 0) * assistant_mask,
                reduce="mean",
            )
        )
        if importance_sampling_level == "sequence":
            prob_ratio = sequence_prob_ratio
        elif importance_sampling_level == "average":
            prob_ratio = (prob_ratio + sequence_prob_ratio) / 2
        elif importance_sampling_level == "geometric_average":
            prob_ratio = (prob_ratio**0.5) * (sequence_prob_ratio**0.5)
    ppo = experimental_config.get("ppo", False)
    if ppo:
        epsilon_default = 0.2
        epsilon_high_default = None
    else:
        epsilon_default = 1.0
        epsilon_high_default = 4.0
    epsilon = experimental_config.get("epsilon", epsilon_default)
    epsilon_high = experimental_config.get("epsilon_high", epsilon_high_default)
    if epsilon_high is None:
        epsilon_high = epsilon
    if max_negative_advantage_importance_sampling_weight := experimental_config.get(
        "max_negative_advantage_importance_sampling_weight", None
    ):
        prob_ratio = torch.clamp(
            prob_ratio, max=max_negative_advantage_importance_sampling_weight
        )
    if experimental_config.get("mask_prob_ratio", False):
        prob_ratio = torch.where(
            (prob_ratio > 1 - epsilon) & (prob_ratio < 1 + epsilon_high),
            prob_ratio,
            0.0,
        )
    if tau := experimental_config.get("kimi_k2_tau", None):
        advantages -= tau * logprob_diff.detach()
    kl_policy_ref: torch.Tensor | None = None
    kl_penalty_coef = experimental_config.get("kl_penalty_coef", 0.0)
    if kl_penalty_coef > 0 and ref_logprobs is not None:
        kl_per_token = (new_logprobs - ref_logprobs).detach() * assistant_mask
        avg_kl = kl_per_token.sum() / (assistant_mask.sum() + 1e-6)
        kl_penalty = kl_penalty_coef * (avg_kl - kl_per_token) * assistant_mask
        advantages = advantages + kl_penalty
        kl_policy_ref = avg_kl
    if ppo:
        policy_loss = -torch.min(
            prob_ratio * advantages,
            torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
        )
    else:
        # Modified REINFORCE or Clipped IS-weight Policy Optimization (CISPO)
        policy_loss = -(
            torch.clip(prob_ratio.detach(), 1 - epsilon, 1 + epsilon_high)
            * advantages
            * new_logprobs
        )
    if upper_bound := experimental_config.get("truncated_importance_sampling", None):
        if "original_logprobs" in inputs:
            original_logprobs = shift_tensor(inputs["original_logprobs"], 0.0)  # ty:ignore[invalid-key]
            original_logprobs = torch.where(
                torch.isnan(original_logprobs),
                new_logprobs.detach(),
                original_logprobs,
            )
            logprob_diff = old_logprobs - original_logprobs
            prob_ratio = torch.exp(logprob_diff)
        policy_loss *= torch.clamp(prob_ratio, max=upper_bound).detach()
    policy_loss = policy_loss * weights * assistant_mask
    mean_policy_loss = policy_loss.sum() / (assistant_mask.sum() + 1e-6)
    # Compute mean entropy for the current step
    if entropies is not None:
        shifted_entropies = shift_tensor(entropies, 0.0)
        mean_entropy = (shifted_entropies * weights * assistant_mask).sum() / (
            assistant_mask.sum() + 1e-6
        )
    else:
        mean_entropy = None
    return Loss(
        mean_policy_loss=mean_policy_loss,
        mean_entropy=mean_entropy,
        policy_loss_sum=policy_loss.sum(),
        probs_corr=probs_corr,
        kl_policy_ref=kl_policy_ref,
    )


def shift_tensor(tensor: torch.Tensor, pad: int | float | bool) -> torch.Tensor:
    return torch.nn.functional.pad(tensor[:, 1:], (0, 1), value=pad)
