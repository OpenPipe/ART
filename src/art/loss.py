from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict
import torch

from art.utils.group_aggregate import group_aggregate

from . import dev
from .dev.train import LossType

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

    def to_metrics(self, *, group: Any | None = None) -> dict[str, float]:
        if (
            self.ratio_sum is None
            or self.clipped_count is None
            or self.token_count is None
            or self.ratio_histogram is None
        ):
            return {}

        vector = torch.cat(
            [
                self.ratio_sum.reshape(1),
                self.clipped_count.reshape(1),
                self.token_count.reshape(1),
                self.ratio_histogram,
            ]
        )
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
    # The scalar used to normalize a raw (``reduction="sum"``) loss.  Megatron
    # uses this value to keep gradient and metric normalization identical to
    # the local reducer.  It is optional for backwards compatibility with
    # callers that construct ``Loss`` objects directly.
    normalization_denominator: torch.Tensor | None = None
    kl_policy_ref: torch.Tensor | None = None
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
    # Distributed schedules pad incomplete global batches with zero-gradient
    # microbatches.  Keep that scheduling fact separate from an ordinary
    # completion whose assistant mask happens to be empty: the latter still
    # participates in GRPO's batch denominator, while the former must not.
    is_dummy: bool = False
    # Context-parallel dispatch can retain the complete logical batch shape
    # while each rank only owns a token slice.  In that case the local
    # ``group_ids`` do not contain enough information to recover the number of
    # completion tails, so the dispatcher may provide the globally computed
    # count here.  Ordinary dense callers leave it unset.
    logical_sequence_count_override: int | None = None
    # Optional per-rank contribution to the denominator used by a raw
    # ``reduction="sum"`` loss.  Context-parallel dispatch uses an integer
    # owner rank for fixed sequence denominators so M-Core can all-reduce the
    # value without rounding fractional counts.
    normalization_denominator_override: torch.Tensor | None = None

    def align_inputs(self) -> "AlignedLossInputs":
        return self

    def group_mean(self, values: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        return group_aggregate(values, by=by, reduce="mean")

    def masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values.sum() / (mask.sum() + 1e-18)

    def denominator(self, mask: torch.Tensor, reduction: Literal["mean", "sum"]):
        """Historical active-token denominator helper.

        Keep this method for callers that used the aligned-input utility
        directly.  ``loss_fn`` now uses :meth:`normalization_denominator` so
        alternate GRPO reductions can share one source of truth.
        """

        if reduction == "sum":
            return 1.0
        return mask.sum() + 1e-18

    def _logical_sequence_groups(self, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Return a group index map and logical completion count.

        Unpacked backends have one row per completion.  Megatron can pack
        several completions into one row, with ``group_ids`` identifying each
        prefix-tree segment.  Grouping by both row and segment keeps the
        reduction invariant under either representation while avoiding any
        gradient-bearing operations on the metadata tensors.
        """

        if mask.ndim != 2:
            # Loss tensors are normally [batch, sequence].  Flattening a
            # lightweight one-dimensional fixture as one sequence keeps the
            # helper well-defined without introducing a host-side branch in
            # the training path.
            mask = mask.reshape(1, -1)

        batch_size = int(mask.shape[0])
        if self.group_ids.shape != mask.shape:
            # A missing/mismatched group map means one logical completion per
            # physical row.  Return an empty index map; callers that need
            # per-group values use the row-wise fallback below.
            empty = mask.new_empty((0,), dtype=torch.long)
            return empty, batch_size

        active = mask > 0
        row_grid = torch.arange(batch_size, device=mask.device).unsqueeze(1)
        row_ids = row_grid.expand_as(mask)[active]
        group_ids = self.group_ids.to(device=mask.device, dtype=torch.long)[active]
        if row_ids.numel() == 0:
            empty = mask.new_empty((0,), dtype=torch.long)
            return empty, batch_size

        keys = torch.stack((row_ids, group_ids), dim=1)
        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
        group_count = int(unique_keys.shape[0])
        active_rows = torch.unique(row_ids)
        empty_rows = batch_size - int(active_rows.numel())
        return inverse, group_count + empty_rows

    def logical_sequence_count(self, mask: torch.Tensor) -> int:
        """Number of completion denominators represented by ``mask``.

        A packed row may contain multiple independent completion tails.  The
        prefix-tree ``group_ids`` metadata lets Dr. GRPO retain the paper's
        ``B * max_completion_length`` denominator instead of accidentally
        using the number of physical rows.
        """

        if self.logical_sequence_count_override is not None:
            return max(int(self.logical_sequence_count_override), 1)
        _, count = self._logical_sequence_groups(mask)
        return max(count, 1)

    def _grouped_mean_sum(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Return the sum of per-completion means and its denominator count."""

        if mask.ndim != 2:
            values = values.reshape(1, -1)
            mask = mask.reshape(1, -1)
        masked_values = values * mask
        if self.group_ids.shape != mask.shape:
            row_values = masked_values.reshape(mask.shape[0], -1).sum(dim=1)
            row_counts = mask.reshape(mask.shape[0], -1).sum(dim=1)
            return (row_values / (row_counts + 1e-18)).sum(), int(mask.shape[0])

        active = mask > 0
        row_grid = torch.arange(mask.shape[0], device=mask.device).unsqueeze(1)
        row_ids = row_grid.expand_as(mask)[active]
        if row_ids.numel() == 0:
            return masked_values.sum(), int(mask.shape[0])
        group_ids = self.group_ids.to(device=mask.device, dtype=torch.long)[active]
        keys = torch.stack((row_ids, group_ids), dim=1)
        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
        group_count = int(unique_keys.shape[0])
        sums = values.new_zeros((group_count,))
        counts = values.new_zeros((group_count,))
        sums.scatter_add_(0, inverse, values[active])
        counts.scatter_add_(0, inverse, mask[active])
        means_sum = (sums / (counts + 1e-18)).sum()
        empty_rows = int(mask.shape[0]) - int(torch.unique(row_ids).numel())
        return means_sum, group_count + empty_rows

    def reduce_masked(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        loss_type: LossType,
        max_completion_length: int | None,
        reduction: Literal["mean", "sum"],
    ) -> torch.Tensor:
        """Reduce token values using the selected GRPO normalization.

        ``bnpo`` is ART's historical active-token mean.  ``grpo`` first takes
        a mean within each completion and then averages completions.  Dr.
        GRPO (``dr_grpo``) uses a fixed ``B * max_completion_length``
        denominator, removing the response-length term from the objective.
        ``reduction="sum"`` intentionally remains a raw sum for distributed
        training; its matching denominator is exposed separately through
        :meth:`normalization_denominator`.
        """

        if reduction == "sum":
            if loss_type == "grpo":
                grouped_sum, _ = self._grouped_mean_sum(values, mask)
                return grouped_sum
            return (values * mask).sum()

        if loss_type == "bnpo":
            if reduction == "mean":
                return self.masked_mean(values * mask, mask)
            return (values * mask).sum() / (mask.sum() + 1e-18)

        if loss_type == "dr_grpo":
            assert max_completion_length is not None
            denominator = torch.tensor(
                float(self.logical_sequence_count(mask) * max_completion_length),
                device=mask.device,
                dtype=torch.float32,
            )
            return (values * mask).sum() / denominator.clamp_min(1.0)

        # GRPO: average each logical completion independently.  The grouped
        # path also handles Megatron prefix-tree rows; ordinary unpacked rows
        # simply contribute one group each.
        if loss_type != "grpo":
            raise ValueError(f"Unknown loss_type: {loss_type!r}")
        grouped_sum, denominator = self._grouped_mean_sum(values, mask)
        return grouped_sum / max(denominator, 1)

    def normalization_denominator(
        self,
        mask: torch.Tensor,
        *,
        loss_type: LossType,
        max_completion_length: int | None,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> torch.Tensor:
        """Return the denominator paired with a raw loss sum."""

        if reduction == "sum" and self.is_dummy:
            return mask.new_zeros(())

        if (
            self.normalization_denominator_override is not None
            and loss_type
            in {
                "grpo",
                "dr_grpo",
            }
            and reduction == "sum"
        ):
            denominator = self.normalization_denominator_override.to(device=mask.device)
            if loss_type == "dr_grpo":
                assert max_completion_length is not None
                denominator = denominator * max_completion_length
            return denominator

        if loss_type == "bnpo":
            denominator = mask.sum()
            return denominator if reduction == "sum" else denominator.clamp_min(1.0)
        if loss_type == "dr_grpo":
            assert max_completion_length is not None
            return torch.tensor(
                float(self.logical_sequence_count(mask) * max_completion_length),
                device=mask.device,
                dtype=torch.float32,
            ).clamp_min(1.0)
        if loss_type == "grpo":
            return torch.tensor(
                float(self.logical_sequence_count(mask)),
                device=mask.device,
                dtype=torch.float32,
            )
        raise ValueError(f"Unknown loss_type: {loss_type!r}")

    def aligned_entropies(self, entropies: torch.Tensor | None) -> torch.Tensor | None:
        if entropies is None:
            return None
        if self.entropies_are_aligned:
            return entropies
        return shift_tensor(entropies, 0.0)


class LossInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    inputs: PackedLossInput
    is_dummy: bool = False

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
            is_dummy=self.is_dummy,
        )


def compute_probs_corr(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor,
) -> torch.Tensor:
    old_logprobs_mask = torch.isfinite(old_logprobs) & torch.isfinite(new_logprobs)
    old_probs = torch.exp(old_logprobs[old_logprobs_mask])
    new_probs = torch.exp(new_logprobs[old_logprobs_mask])
    if old_probs.numel() < 2:
        return new_logprobs.new_zeros(())
    old_std = old_probs.std(unbiased=False)
    new_std = new_probs.std(unbiased=False)
    if (
        not torch.isfinite(old_std).item()
        or not torch.isfinite(new_std).item()
        or old_std.item() == 0.0
        or new_std.item() == 0.0
    ):
        return new_logprobs.new_zeros(())
    return torch.corrcoef(torch.stack([old_probs, new_probs]))[0, 1]


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
    *,
    loss_type: LossType | None = None,
    max_completion_length: int | None = None,
) -> Loss:
    aligned_inputs = inputs.align_inputs()
    configured_loss_type = (
        loss_type
        if loss_type is not None
        else experimental_config.get("loss_type", "bnpo")
    )
    if configured_loss_type is None:
        configured_loss_type = "bnpo"
    if not isinstance(configured_loss_type, str) or configured_loss_type not in {
        "grpo",
        "bnpo",
        "dr_grpo",
    }:
        raise ValueError(
            f"Unknown loss_type: {configured_loss_type!r}. "
            "Expected one of 'grpo', 'bnpo', or 'dr_grpo'."
        )
    resolved_loss_type = configured_loss_type
    resolved_max_completion_length = (
        max_completion_length
        if max_completion_length is not None
        else experimental_config.get("max_completion_length")
    )
    if resolved_loss_type == "dr_grpo":
        if resolved_max_completion_length is None:
            resolved_max_completion_length = dev.DEFAULT_MAX_COMPLETION_LENGTH
        if (
            isinstance(resolved_max_completion_length, bool)
            or not isinstance(resolved_max_completion_length, int)
            or resolved_max_completion_length <= 0
        ):
            raise ValueError("max_completion_length must be a positive integer")
    old_logprobs = aligned_inputs.old_logprobs
    advantages = aligned_inputs.advantages
    assistant_mask_bool = aligned_inputs.assistant_mask.to(dtype=torch.bool)
    new_logprobs = _mask_ignored_tokens(new_logprobs, assistant_mask_bool)
    old_logprobs = _mask_ignored_tokens(old_logprobs, assistant_mask_bool)
    if ref_logprobs is not None:
        ref_logprobs = _mask_ignored_tokens(ref_logprobs, assistant_mask_bool)
    assistant_mask = assistant_mask_bool.to(new_logprobs.dtype)
    weights = aligned_inputs.weights
    probs_corr = compute_probs_corr(
        torch.where(
            assistant_mask_bool,
            old_logprobs,
            old_logprobs.new_full((), float("nan")),
        ),
        new_logprobs,
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
        avg_kl = aligned_inputs.masked_mean(kl_per_token, assistant_mask)
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
    reduced_policy_loss = aligned_inputs.reduce_masked(
        policy_loss,
        assistant_mask,
        loss_type=resolved_loss_type,
        max_completion_length=resolved_max_completion_length,
        reduction=reduction,
    )
    normalization_denominator = aligned_inputs.normalization_denominator(
        assistant_mask,
        loss_type=resolved_loss_type,
        max_completion_length=resolved_max_completion_length,
        reduction=reduction,
    )
    # Compute reduced entropy for the current step.
    aligned_entropies = aligned_inputs.aligned_entropies(entropies)
    if aligned_entropies is not None:
        aligned_entropies = _mask_ignored_tokens(aligned_entropies, assistant_mask_bool)
        entropy_values = aligned_entropies * weights * assistant_mask
        # Entropy is a diagnostic in ART (it is not added to ``policy_loss``),
        # so retain the historical active-token mean regardless of the policy
        # normalization selected above.  The sum path remains a raw sum for
        # callers that explicitly requested ``reduction="sum"``.
        entropy = (
            entropy_values.sum()
            if reduction == "sum"
            else aligned_inputs.masked_mean(entropy_values, assistant_mask)
        )
    else:
        entropy = None
    return Loss(
        reduction=reduction,
        policy_loss=reduced_policy_loss,
        entropy=entropy,
        policy_loss_sum=policy_loss.sum(),
        probs_corr=probs_corr,
        normalization_denominator=normalization_denominator,
        kl_policy_ref=kl_policy_ref,
        offpolicy_diagnostics=offpolicy_diagnostics,
    )


def shift_tensor(tensor: torch.Tensor, pad: int | float | bool) -> torch.Tensor:
    return torch.nn.functional.pad(tensor[:, 1:], (0, 1), value=pad)
