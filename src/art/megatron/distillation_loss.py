"""Sparse top-k-plus-tail distillation loss for Megatron training.

The teacher target has ``K`` explicit token probabilities and one bucket for
all other logical-vocabulary tokens.  This implementation only materializes
``O(rows * K)`` teacher state; it never expands the teacher distribution to
the vocabulary dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TopKPlusTailForwardKLResult:
    """Unreduced KD loss and diagnostics for a batch of token rows."""

    loss_sum: torch.Tensor
    token_count: torch.Tensor
    per_token_loss: torch.Tensor
    per_token_weighted_loss: torch.Tensor
    selected_loss_sum: torch.Tensor
    tail_loss_sum: torch.Tensor
    teacher_selected_mass_sum: torch.Tensor
    teacher_tail_mass_sum: torch.Tensor
    student_selected_mass_sum: torch.Tensor
    student_tail_mass_sum: torch.Tensor
    numerical_clamp_count: torch.Tensor


def _active_any(value: torch.Tensor) -> bool:
    """Return a scalar validation result without retaining an autograd edge."""

    return bool(value.detach().any().item())


def topk_plus_tail_forward_kl(
    student_logits: torch.Tensor,
    *,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_tail_logprob: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    logical_vocab_size: int | None = None,
    temperature: float = 1.0,
    compensate_temperature_squared: bool = False,
) -> TopKPlusTailForwardKLResult:
    """Evaluate coarsened ``KL(teacher || student)`` over ``K + 1`` buckets.

    Leading dimensions identify token rows.  The final student dimension is
    the physical (possibly padded) vocabulary and the final teacher dimension
    is the fixed top-k width.  Teacher log-probabilities are assumed to have
    already had their target temperature applied.

    The returned ``loss_sum`` is weighted and unreduced. ``token_count`` is an
    independent integer denominator: the number of true entries in ``mask``.
    Values in inactive rows are ignored, including sentinel IDs and nonfinite
    sidecar values.
    """

    if student_logits.ndim < 1 or not student_logits.is_floating_point():
        raise ValueError(
            "student_logits must be floating point with a vocabulary dimension"
        )
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    physical_vocab_size = student_logits.shape[-1]
    vocab_size = (
        physical_vocab_size if logical_vocab_size is None else int(logical_vocab_size)
    )
    if vocab_size <= 1 or vocab_size > physical_vocab_size:
        raise ValueError("logical_vocab_size must be in [2, student_logits.shape[-1]]")

    row_shape = student_logits.shape[:-1]
    if teacher_topk_ids.ndim != student_logits.ndim:
        raise ValueError("teacher_topk_ids must have one trailing top-k dimension")
    if teacher_topk_logprobs.shape != teacher_topk_ids.shape:
        raise ValueError("teacher_topk_logprobs must match teacher_topk_ids exactly")
    if teacher_topk_ids.shape[:-1] != row_shape:
        raise ValueError("teacher top-k rows must match student_logits")
    if teacher_tail_logprob.shape != row_shape:
        raise ValueError("teacher_tail_logprob must match the token-row dimensions")
    if mask.shape != row_shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean and match the token-row dimensions")
    if weights is not None and weights.shape != row_shape:
        raise ValueError("weights must match the token-row dimensions")
    if teacher_topk_ids.dtype == torch.bool or (teacher_topk_ids.is_floating_point()):
        raise ValueError("teacher_topk_ids must have an integer dtype")

    device = student_logits.device
    tensors = (
        teacher_topk_ids,
        teacher_topk_logprobs,
        teacher_tail_logprob,
        mask,
    )
    if weights is not None:
        tensors = (*tensors, weights)
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all distillation tensors must be on the logits device")

    active = mask
    flat_active = active.reshape(-1)
    active_indices = torch.nonzero(flat_active, as_tuple=False).squeeze(-1)
    active_count = int(active_indices.numel())
    if active_count == 0:
        raise ValueError("distillation mask selects zero tokens")
    token_count = torch.tensor(active_count, dtype=torch.int64, device=device)

    k = teacher_topk_ids.shape[-1]
    if k <= 0 or k >= vocab_size:
        raise ValueError("top-k width must be in [1, logical_vocab_size)")

    active_ids = (
        teacher_topk_ids.reshape(-1, k).index_select(0, active_indices).to(torch.int64)
    )
    invalid_ids = (active_ids < 0) | (active_ids >= vocab_size)
    if _active_any(invalid_ids):
        raise ValueError("active teacher_topk_ids contain an out-of-range token ID")
    if k > 1:
        sorted_ids = active_ids.sort(dim=-1).values
        duplicates = sorted_ids[..., 1:] == sorted_ids[..., :-1]
        if _active_any(duplicates):
            raise ValueError("active teacher_topk_ids must be unique within each row")

    active_logits = (
        student_logits.reshape(-1, physical_vocab_size)[..., :vocab_size]
        .index_select(0, active_indices)
        .to(torch.float32)
    )
    if _active_any(~torch.isfinite(active_logits)):
        raise ValueError("active logical student logits must be finite")
    teacher_logprobs = (
        teacher_topk_logprobs.reshape(-1, k)
        .index_select(0, active_indices)
        .to(torch.float32)
    )
    if _active_any(~torch.isfinite(teacher_logprobs)):
        raise ValueError("active teacher top-k log-probabilities must be finite")
    raw_tail_logprob = (
        teacher_tail_logprob.reshape(-1)
        .index_select(0, active_indices)
        .to(torch.float32)
    )
    invalid_tail = torch.isnan(raw_tail_logprob) | torch.isposinf(raw_tail_logprob)
    if _active_any(invalid_tail):
        raise ValueError("active teacher tail log-probabilities must be finite or -inf")

    if weights is None:
        active_weights = torch.ones(active_count, device=device, dtype=torch.float32)
    else:
        active_weights = (
            weights.reshape(-1).index_select(0, active_indices).to(torch.float32)
        )
        if _active_any(~torch.isfinite(active_weights)):
            raise ValueError("active distillation weights must be finite")
        if _active_any(active_weights < 0):
            raise ValueError("active distillation weights must be non-negative")

    teacher_selected_probs = teacher_logprobs.exp()
    teacher_tail_prob = raw_tail_logprob.exp()
    # A zero teacher tail is encoded as -inf. Replace only its logarithm after
    # exponentiation so 0 * log(0) is evaluated as zero rather than NaN.
    tail_logprob = torch.where(
        torch.isneginf(raw_tail_logprob),
        torch.zeros_like(raw_tail_logprob),
        raw_tail_logprob,
    )
    teacher_total = teacher_selected_probs.sum(dim=-1) + teacher_tail_prob
    if not torch.allclose(
        teacher_total,
        torch.ones_like(teacher_total),
        rtol=2e-4,
        atol=2e-5,
    ):
        raise ValueError("active teacher top-k probabilities plus tail must sum to one")

    scaled_logits = active_logits / temperature
    log_normalizer = torch.logsumexp(scaled_logits, dim=-1)
    selected_logits = torch.gather(scaled_logits, dim=-1, index=active_ids)
    selected_student_logprobs = selected_logits - log_normalizer.unsqueeze(-1)

    # Mask only the active M*V student rows. This computes the complement
    # directly, preserving gradients even when selected mass rounds to one.
    complement_logits = scaled_logits.scatter(
        dim=-1,
        index=active_ids,
        value=-torch.inf,
    )
    student_tail_logprob = torch.logsumexp(complement_logits, dim=-1) - log_normalizer
    selected_student_logmass = torch.logsumexp(selected_student_logprobs, dim=-1)

    selected_loss = (
        teacher_selected_probs * (teacher_logprobs - selected_student_logprobs)
    ).sum(dim=-1)
    tail_loss = teacher_tail_prob * (tail_logprob - student_tail_logprob)

    scale = 1.0
    if compensate_temperature_squared:
        scale *= temperature**2
    active_per_token_loss = (selected_loss + tail_loss) * scale
    active_weighted_loss = active_per_token_loss * active_weights
    flat_per_token_loss = torch.zeros(
        flat_active.numel(), dtype=torch.float32, device=device
    ).scatter(
        0,
        active_indices,
        active_per_token_loss,
    )
    flat_weighted_loss = torch.zeros(
        flat_active.numel(), dtype=torch.float32, device=device
    ).scatter(
        0,
        active_indices,
        active_weighted_loss,
    )
    per_token_loss = flat_per_token_loss.reshape(row_shape)
    per_token_weighted_loss = flat_weighted_loss.reshape(row_shape)

    selected_student_mass = selected_student_logmass.exp()
    student_tail_mass = student_tail_logprob.exp()
    return TopKPlusTailForwardKLResult(
        loss_sum=active_weighted_loss.sum(),
        token_count=token_count,
        per_token_loss=per_token_loss,
        per_token_weighted_loss=per_token_weighted_loss,
        selected_loss_sum=(selected_loss * active_weights).sum() * scale,
        tail_loss_sum=(tail_loss * active_weights).sum() * scale,
        teacher_selected_mass_sum=teacher_selected_probs.sum(),
        teacher_tail_mass_sum=teacher_tail_prob.sum(),
        student_selected_mass_sum=selected_student_mass.sum(),
        student_tail_mass_sum=student_tail_mass.sum(),
        numerical_clamp_count=torch.zeros((), dtype=torch.int64, device=device),
    )
