from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict
import torch

from art.utils.group_aggregate import group_aggregate

from . import dev

PROBABILITY_CORRELATION_STAT_COUNT = 6
IMPORTANCE_RATIO_HISTOGRAM_BUCKETS = 258

if TYPE_CHECKING:
    from art.preprocessing.inputs import TrainInputs
    from art.preprocessing.pack import PackedTensors

    PackedLossInput = PackedTensors | TrainInputs
else:
    PackedLossInput = object


class LossOffPolicyDiagnostics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ratio_sum: torch.Tensor
    clipped_count: torch.Tensor
    token_count: torch.Tensor
    ratio_histogram: torch.Tensor

    @classmethod
    def from_tensors(
        cls,
        *,
        prob_ratio: torch.Tensor,
        advantages: torch.Tensor,
        assistant_mask: torch.Tensor,
        weights: torch.Tensor,
        ppo: bool,
        epsilon: float,
        epsilon_high: float,
    ) -> "LossOffPolicyDiagnostics":
        train_mask = (assistant_mask > 0) & (weights != 0)
        ratios = prob_ratio.detach()[train_mask].float()
        edges = importance_ratio_histogram_edges(device=prob_ratio.device)
        zero = prob_ratio.new_zeros((), dtype=torch.float32)
        if ratios.numel() == 0:
            return cls(
                ratio_sum=zero,
                clipped_count=zero,
                token_count=zero,
                ratio_histogram=zero.new_zeros(edges.numel() + 1),
            )

        lower = 1.0 - epsilon
        upper = 1.0 + epsilon_high
        if ppo:
            train_advantages = advantages.detach()[train_mask]
            clipped = ((train_advantages > 0) & (ratios > upper)) | (
                (train_advantages < 0) & (ratios < lower)
            )
        else:
            clipped = (ratios < lower) | (ratios > upper)

        return cls(
            ratio_sum=ratios.sum(),
            clipped_count=clipped.float().sum(),
            token_count=ratios.new_tensor(float(ratios.numel())),
            ratio_histogram=torch.bincount(
                torch.bucketize(ratios, edges),
                minlength=edges.numel() + 1,
            ).to(dtype=torch.float32),
        )


class LossOffPolicyDiagnosticsAccumulator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ratio_sum: torch.Tensor | None = None
    clipped_count: torch.Tensor | None = None
    token_count: torch.Tensor | None = None
    ratio_histogram: torch.Tensor | None = None

    def add(self, diagnostics: LossOffPolicyDiagnostics | None) -> None:
        if diagnostics is None:
            return
        if self.ratio_sum is None:
            self.ratio_sum = diagnostics.ratio_sum.detach()
            self.clipped_count = diagnostics.clipped_count.detach()
            self.token_count = diagnostics.token_count.detach()
            self.ratio_histogram = diagnostics.ratio_histogram.detach()
            return
        assert self.clipped_count is not None
        assert self.token_count is not None
        assert self.ratio_histogram is not None
        self.ratio_sum = self.ratio_sum + diagnostics.ratio_sum.detach()
        self.clipped_count = self.clipped_count + diagnostics.clipped_count.detach()
        self.token_count = self.token_count + diagnostics.token_count.detach()
        self.ratio_histogram = (
            self.ratio_histogram + diagnostics.ratio_histogram.detach()
        )

    def extend(self, other: "LossOffPolicyDiagnosticsAccumulator") -> None:
        if other.ratio_sum is None:
            return
        assert other.clipped_count is not None
        assert other.token_count is not None
        assert other.ratio_histogram is not None
        self.add(
            LossOffPolicyDiagnostics(
                ratio_sum=other.ratio_sum,
                clipped_count=other.clipped_count,
                token_count=other.token_count,
                ratio_histogram=other.ratio_histogram,
            )
        )

    def tensor(self, *, device: torch.device | None = None) -> torch.Tensor:
        if self.ratio_sum is None:
            if device is None:
                raise ValueError("empty diagnostics require an explicit device")
            return torch.zeros(
                3 + IMPORTANCE_RATIO_HISTOGRAM_BUCKETS,
                device=device,
                dtype=torch.float64,
            )
        assert self.clipped_count is not None
        assert self.token_count is not None
        assert self.ratio_histogram is not None
        return torch.cat(
            (
                self.ratio_sum.reshape(1),
                self.clipped_count.reshape(1),
                self.token_count.reshape(1),
                self.ratio_histogram,
            )
        ).to(dtype=torch.float64)

    @classmethod
    def from_tensor(cls, values: torch.Tensor) -> "LossOffPolicyDiagnosticsAccumulator":
        if values.shape != (3 + IMPORTANCE_RATIO_HISTOGRAM_BUCKETS,):
            raise ValueError("invalid off-policy diagnostics vector shape")
        return cls(
            ratio_sum=values[0],
            clipped_count=values[1],
            token_count=values[2],
            ratio_histogram=values[3:],
        )

    def to_metrics(self, *, group: Any | None = None) -> dict[str, float]:
        if (
            self.ratio_sum is None
            or self.clipped_count is None
            or self.token_count is None
            or self.ratio_histogram is None
        ):
            return {}

        vector = self.tensor()
        if group is not None:
            torch.distributed.all_reduce(vector, group=group)  # ty: ignore[possibly-missing-attribute]

        ratio_sum, clipped_count, token_count = vector[:3]
        denominator = token_count.clamp_min(1.0)
        histogram = vector[3:]
        edges = importance_ratio_histogram_edges(device=vector.device)
        return {
            "loss/importance_ratio_mean": float((ratio_sum / denominator).item()),
            "loss/importance_ratio_p95": float(
                histogram_quantile(histogram, edges, 0.95).item()
            ),
            "loss/importance_ratio_p99": float(
                histogram_quantile(histogram, edges, 0.99).item()
            ),
            "loss/clipped_token_fraction": float((clipped_count / denominator).item()),
        }


def importance_ratio_histogram_edges(*, device: torch.device) -> torch.Tensor:
    return torch.logspace(-4, 4, steps=257, device=device, dtype=torch.float32)


def histogram_quantile(
    histogram: torch.Tensor,
    edges: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    total = histogram.sum()
    if total <= 0:
        return histogram.new_zeros(())
    cumulative = torch.cumsum(histogram, dim=0)
    target = total * quantile
    bucket = torch.searchsorted(cumulative, target).clamp(max=histogram.numel() - 1)
    lower_count = torch.where(
        bucket > 0,
        cumulative[(bucket - 1).clamp_min(0)],
        cumulative.new_zeros(()),
    )
    bucket_count = histogram[bucket].clamp_min(1.0)
    lower = torch.where(
        bucket == 0,
        edges.new_zeros(()),
        edges[(bucket - 1).clamp(max=edges.numel() - 1)],
    )
    upper = torch.where(
        bucket >= edges.numel(),
        edges[-1],
        edges[bucket.clamp(max=edges.numel() - 1)],
    )
    fraction = ((target - lower_count) / bucket_count).clamp(0.0, 1.0)
    return lower + (upper - lower) * fraction


class Loss(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reduction: Literal["mean", "sum"]
    policy_loss: torch.Tensor
    entropy: torch.Tensor | None
    policy_loss_sum: torch.Tensor
    probs_corr: torch.Tensor
    probs_corr_stats: torch.Tensor
    kl_policy_ref: torch.Tensor | None = None
    kl_policy_ref_sum: torch.Tensor | None = None
    kl_policy_ref_count: torch.Tensor | None = None
    offpolicy_diagnostics: LossOffPolicyDiagnostics | None = None


class AlignedLossInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    assistant_mask: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor
    group_ids: torch.Tensor
    original_logprobs: torch.Tensor | None = None
    entropies_are_aligned: bool = False

    def align_inputs(self) -> "AlignedLossInputs":
        return self

    def group_mean(self, values: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        return group_aggregate(values, by=by, reduce="mean")

    def masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values.sum() / (mask.sum() + 1e-18)

    def denominator(self, mask: torch.Tensor, reduction: Literal["mean", "sum"]):
        if reduction == "sum":
            return 1.0
        return mask.sum() + 1e-18

    def aligned_entropies(self, entropies: torch.Tensor | None) -> torch.Tensor | None:
        if entropies is None:
            return None
        if self.entropies_are_aligned:
            return entropies
        return shift_tensor(entropies, 0.0)


class LossInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    inputs: PackedLossInput

    def align_inputs(self) -> AlignedLossInputs:
        inputs = self.inputs
        return AlignedLossInputs(
            assistant_mask=shift_tensor(inputs["assistant_mask"], False),
            old_logprobs=shift_tensor(inputs["logprobs"], float("nan")),
            advantages=shift_tensor(inputs["advantages"], 0.0),
            weights=shift_tensor(inputs["weights"], 0.0),
            group_ids=shift_tensor(inputs["group_ids"], 0),
            original_logprobs=(
                shift_tensor(inputs["original_logprobs"], 0.0)
                if "original_logprobs" in inputs
                else None
            ),
        )


def compute_probs_corr(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
) -> torch.Tensor:
    return probability_correlation_from_stats(
        compute_probs_corr_stats(old_logprobs, new_logprobs)
    ).to(dtype=new_logprobs.dtype)


def compute_probs_corr_stats(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
) -> torch.Tensor:
    old_logprobs_mask = torch.isfinite(old_logprobs) & torch.isfinite(new_logprobs)
    old_probs = torch.exp(old_logprobs[old_logprobs_mask]).to(torch.float64)
    new_probs = torch.exp(new_logprobs[old_logprobs_mask]).to(torch.float64)
    return torch.stack(
        (
            old_probs.new_tensor(old_probs.numel()),
            old_probs.sum(),
            new_probs.sum(),
            old_probs.square().sum(),
            new_probs.square().sum(),
            (old_probs * new_probs).sum(),
        )
    )


def probability_correlation_from_stats(stats: torch.Tensor) -> torch.Tensor:
    if stats.shape != (PROBABILITY_CORRELATION_STAT_COUNT,):
        raise ValueError("invalid probability-correlation statistics shape")
    count, sum_x, sum_y, sum_x2, sum_y2, sum_xy = stats
    covariance = count * sum_xy - sum_x * sum_y
    variance_x = count * sum_x2 - sum_x.square()
    variance_y = count * sum_y2 - sum_y.square()
    denominator = (variance_x.clamp_min(0) * variance_y.clamp_min(0)).sqrt()
    valid = (
        (count >= 2)
        & torch.isfinite(covariance)
        & torch.isfinite(denominator)
        & (denominator > 0)
    )
    return torch.where(valid, covariance / denominator.clamp_min(1e-30), 0.0)


def _mask_ignored_tokens(
    tensor: torch.Tensor,
    assistant_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.where(assistant_mask, tensor, tensor.new_zeros(()))


def loss_fn(
    inputs: "LossInputs | AlignedLossInputs",
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    entropies: torch.Tensor | None,
    experimental_config: dev.TrainConfig,
    reduction: Literal["mean", "sum"] = "mean",
) -> Loss:
    aligned_inputs = inputs.align_inputs()
    old_logprobs = aligned_inputs.old_logprobs
    advantages = aligned_inputs.advantages
    assistant_mask_bool = aligned_inputs.assistant_mask.to(dtype=torch.bool)
    new_logprobs = _mask_ignored_tokens(new_logprobs, assistant_mask_bool)
    old_logprobs = _mask_ignored_tokens(old_logprobs, assistant_mask_bool)
    if ref_logprobs is not None:
        ref_logprobs = _mask_ignored_tokens(ref_logprobs, assistant_mask_bool)
    assistant_mask = assistant_mask_bool.to(new_logprobs.dtype)
    weights = aligned_inputs.weights
    probs_corr_stats = compute_probs_corr_stats(
        torch.where(
            assistant_mask_bool,
            old_logprobs,
            old_logprobs.new_full((), float("nan")),
        ),
        new_logprobs,
    )
    probs_corr = probability_correlation_from_stats(probs_corr_stats).to(
        dtype=new_logprobs.dtype
    )
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
            aligned_inputs.group_mean(
                logprob_diff,
                by=aligned_inputs.group_ids * assistant_mask,
            )
        )
        if importance_sampling_level == "sequence":
            prob_ratio = sequence_prob_ratio
        elif importance_sampling_level == "average":
            prob_ratio = (prob_ratio + sequence_prob_ratio) / 2
        elif importance_sampling_level == "geometric_average":
            prob_ratio = (prob_ratio**0.5) * (sequence_prob_ratio**0.5)
    raw_prob_ratio = prob_ratio
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
        advantages = advantages - tau * logprob_diff.detach()
    kl_policy_ref: torch.Tensor | None = None
    kl_policy_ref_sum: torch.Tensor | None = None
    kl_policy_ref_count: torch.Tensor | None = None
    kl_penalty_coef = experimental_config.get("kl_penalty_coef", 0.0)
    if kl_penalty_coef > 0 and ref_logprobs is not None:
        match experimental_config.get("kl_penalty_source", "current_learner"):
            case "sample":
                kl_source_logprobs = old_logprobs.detach()
            case "current_learner":
                kl_source_logprobs = new_logprobs.detach()
            case other:
                raise AssertionError(other)
        kl_per_token = (kl_source_logprobs - ref_logprobs).detach() * assistant_mask
        kl_policy_ref_sum = kl_per_token.sum()
        kl_policy_ref_count = assistant_mask.sum()
        avg_kl = kl_policy_ref_sum / kl_policy_ref_count.clamp_min(1e-18)
        kl_penalty = kl_penalty_coef * (avg_kl - kl_per_token) * assistant_mask
        advantages = advantages + kl_penalty
        kl_policy_ref = avg_kl
    offpolicy_diagnostics = LossOffPolicyDiagnostics.from_tensors(
        prob_ratio=raw_prob_ratio,
        advantages=advantages,
        assistant_mask=assistant_mask,
        weights=weights,
        ppo=ppo,
        epsilon=epsilon,
        epsilon_high=epsilon_high,
    )
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
        if aligned_inputs.original_logprobs is not None:
            original_logprobs = aligned_inputs.original_logprobs
            original_logprobs = torch.where(
                torch.isnan(original_logprobs),
                new_logprobs.detach(),
                original_logprobs,
            )
            logprob_diff = old_logprobs - original_logprobs
            prob_ratio = torch.exp(logprob_diff)
        policy_loss *= torch.clamp(prob_ratio, max=upper_bound).detach()
    policy_loss = policy_loss * weights * assistant_mask
    denominator = aligned_inputs.denominator(assistant_mask, reduction)
    reduced_policy_loss = policy_loss.sum() / denominator
    # Compute reduced entropy for the current step.
    aligned_entropies = aligned_inputs.aligned_entropies(entropies)
    if aligned_entropies is not None:
        aligned_entropies = _mask_ignored_tokens(aligned_entropies, assistant_mask_bool)
        entropy = (aligned_entropies * weights * assistant_mask).sum() / denominator
    else:
        entropy = None
    return Loss(
        reduction=reduction,
        policy_loss=reduced_policy_loss,
        entropy=entropy,
        policy_loss_sum=policy_loss.sum(),
        probs_corr=probs_corr,
        probs_corr_stats=probs_corr_stats,
        kl_policy_ref=kl_policy_ref,
        kl_policy_ref_sum=kl_policy_ref_sum,
        kl_policy_ref_count=kl_policy_ref_count,
        offpolicy_diagnostics=offpolicy_diagnostics,
    )


def shift_tensor(tensor: torch.Tensor, pad: int | float | bool) -> torch.Tensor:
    return torch.nn.functional.pad(tensor[:, 1:], (0, 1), value=pad)
