"""Numerically explicit reference losses for distillation.

This module is a correctness oracle, not the eventual fused training kernel.  It
intentionally performs the probability calculation in float64 and materializes
the complement mask so optimized implementations can be checked against it.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopKForwardKLResult:
    """Unreduced forward-KL loss and diagnostics for one batch of token rows."""

    loss_sum: torch.Tensor
    token_count: torch.Tensor
    per_token_loss: torch.Tensor
    per_token_weighted_loss: torch.Tensor
    selected_loss_sum: torch.Tensor
    tail_loss_sum: torch.Tensor
    teacher_tail_mass_sum: torch.Tensor
    student_tail_mass_sum: torch.Tensor


def topk_plus_tail_forward_kl(
    student_logits: torch.Tensor,
    *,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    teacher_tail_prob: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    logical_vocab_size: int | None = None,
    student_temperature: float = 1.0,
    compensate_temperature_squared: bool = False,
) -> TopKForwardKLResult:
    """Compute ``KL(teacher || student)`` over top-k tokens plus a tail bucket.

    The teacher distribution consists of explicit probabilities for arbitrary
    unique token IDs and one probability for all remaining logical-vocabulary
    tokens.  The student's tail probability is computed with a logsumexp over
    that exact complement.  Leading dimensions are token-row dimensions and the
    final dimensions of the logits and top-k inputs are vocabulary and ``k``.

    ``loss_sum`` is weighted but unreduced. ``token_count`` is the number of
    true mask entries and is deliberately independent of optional weights.
    Teacher probabilities are assumed to have already had their own target
    temperature applied.
    """

    if student_logits.ndim < 1:
        raise ValueError("student_logits must have a vocabulary dimension")
    if not student_logits.is_floating_point():
        raise ValueError("student_logits must be floating point")
    if not torch.isfinite(torch.tensor(student_temperature)):
        raise ValueError("student_temperature must be finite")
    if student_temperature <= 0:
        raise ValueError("student_temperature must be greater than zero")

    physical_vocab_size = student_logits.shape[-1]
    vocab_size = (
        physical_vocab_size if logical_vocab_size is None else logical_vocab_size
    )
    if vocab_size <= 0 or vocab_size > physical_vocab_size:
        raise ValueError("logical_vocab_size must be in [1, student_logits.shape[-1]]")

    row_shape = student_logits.shape[:-1]
    if teacher_topk_ids.ndim != student_logits.ndim:
        raise ValueError("teacher_topk_ids must have one trailing top-k dimension")
    if teacher_topk_probs.shape != teacher_topk_ids.shape:
        raise ValueError("teacher_topk_probs must match teacher_topk_ids")
    if teacher_topk_ids.shape[:-1] != row_shape:
        raise ValueError("teacher top-k row dimensions must match student_logits")
    if teacher_tail_prob.shape != row_shape:
        raise ValueError("teacher_tail_prob must match the token-row dimensions")
    if mask.shape != row_shape:
        raise ValueError("mask must match the token-row dimensions")
    if mask.dtype != torch.bool:
        raise ValueError("mask must have boolean dtype")
    if weights is not None and weights.shape != row_shape:
        raise ValueError("weights must match the token-row dimensions")
    if teacher_topk_ids.device != student_logits.device:
        raise ValueError("teacher_topk_ids must be on the student_logits device")

    train_mask = mask.to(device=student_logits.device)
    token_count = train_mask.sum(dtype=torch.int64)
    if token_count.item() == 0:
        raise ValueError("distillation mask selects zero tokens")

    k = teacher_topk_ids.shape[-1]
    if k <= 0 or k > vocab_size:
        raise ValueError("top-k width must be in [1, logical_vocab_size]")
    if teacher_topk_ids.dtype == torch.bool or teacher_topk_ids.is_floating_point():
        raise ValueError("teacher_topk_ids must have an integer dtype")
    active_rows = train_mask.unsqueeze(-1)
    invalid_ids = (teacher_topk_ids < 0) | (teacher_topk_ids >= vocab_size)
    if (invalid_ids & active_rows).any():
        raise ValueError("teacher_topk_ids contains an out-of-range token ID")
    safe_ids = torch.where(active_rows, teacher_topk_ids, 0).to(torch.int64)
    sorted_ids = safe_ids.sort(dim=-1).values
    duplicate_ids = sorted_ids[..., 1:] == sorted_ids[..., :-1]
    if k > 1 and (duplicate_ids & active_rows[..., 0].unsqueeze(-1)).any():
        raise ValueError("teacher_topk_ids must be unique within each row")

    compute_device = student_logits.device
    teacher_probs = teacher_topk_probs.to(device=compute_device, dtype=torch.float64)
    teacher_tail = teacher_tail_prob.to(device=compute_device, dtype=torch.float64)
    if (~torch.isfinite(teacher_probs) & active_rows).any() or (
        ~torch.isfinite(teacher_tail) & train_mask
    ).any():
        raise ValueError("teacher probabilities must be finite")
    if ((teacher_probs < 0) & active_rows).any() or (
        (teacher_tail < 0) & train_mask
    ).any():
        raise ValueError("teacher probabilities must be non-negative")
    safe_probs = torch.where(active_rows, teacher_probs, 0.0)
    safe_tail = torch.where(train_mask, teacher_tail, 0.0)
    # Give inactive rows a harmless normalized target. Their values remain
    # excluded from all returned sums by ``train_mask``.
    safe_probs[..., 0] = torch.where(
        train_mask, safe_probs[..., 0], torch.ones_like(safe_probs[..., 0])
    )
    teacher_total = safe_probs.sum(dim=-1) + safe_tail
    if not torch.allclose(
        teacher_total[train_mask],
        torch.ones_like(teacher_total[train_mask]),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError("teacher top-k probabilities plus tail must sum to one")
    if k == vocab_size and not torch.allclose(
        safe_tail[train_mask],
        torch.zeros_like(safe_tail[train_mask]),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("teacher tail probability must be zero when k == vocab")

    if weights is None:
        row_weights = torch.ones(row_shape, device=compute_device, dtype=torch.float64)
    else:
        row_weights = weights.to(device=compute_device, dtype=torch.float64)
        if (~torch.isfinite(row_weights) & train_mask).any():
            raise ValueError("weights must be finite")
        if ((row_weights < 0) & train_mask).any():
            raise ValueError("weights must be non-negative")
        row_weights = torch.where(train_mask, row_weights, 0.0)

    logical_logits = student_logits[..., :vocab_size].to(torch.float64)
    if not torch.isfinite(logical_logits).all():
        raise ValueError("logical student logits must be finite")
    scaled_logits = logical_logits / student_temperature
    log_normalizer = torch.logsumexp(scaled_logits, dim=-1)
    selected_logits = torch.gather(scaled_logits, dim=-1, index=safe_ids)
    selected_log_probs = selected_logits - log_normalizer.unsqueeze(-1)

    # xlogy defines the 0 * log(0) teacher-entropy term as zero.
    selected_loss = (
        torch.xlogy(safe_probs, safe_probs) - safe_probs * selected_log_probs
    ).sum(dim=-1)

    if k == vocab_size:
        student_tail = torch.zeros_like(teacher_tail)
        tail_loss = torch.zeros_like(teacher_tail)
    else:
        complement = torch.ones_like(logical_logits, dtype=torch.bool)
        complement.scatter_(-1, safe_ids, False)
        tail_logit_lse = torch.logsumexp(
            scaled_logits.masked_fill(~complement, -torch.inf), dim=-1
        )
        student_tail_log_prob = tail_logit_lse - log_normalizer
        student_tail = student_tail_log_prob.exp()
        tail_loss = torch.xlogy(safe_tail, safe_tail) - (
            safe_tail * student_tail_log_prob
        )

    temperature_factor = (
        student_temperature**2 if compensate_temperature_squared else 1.0
    )
    per_token_loss = (selected_loss + tail_loss) * temperature_factor
    effective_weight = row_weights * train_mask.to(torch.float64)
    per_token_weighted_loss = per_token_loss * effective_weight

    return TopKForwardKLResult(
        loss_sum=per_token_weighted_loss.sum(),
        token_count=token_count,
        per_token_loss=per_token_loss,
        per_token_weighted_loss=per_token_weighted_loss,
        selected_loss_sum=(selected_loss * effective_weight).sum() * temperature_factor,
        tail_loss_sum=(tail_loss * effective_weight).sum() * temperature_factor,
        teacher_tail_mass_sum=(safe_tail * train_mask).sum(),
        student_tail_mass_sum=(student_tail * train_mask).sum(),
    )
