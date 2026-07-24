"""Vocabulary-parallel scoring for prepared policy and distillation losses.

Megatron returns one padded vocabulary shard per tensor-parallel rank.  This
module scores arbitrary global token IDs without gathering the vocabulary.
The forward collectives replicate scalar/token-row scores while their backward
is intentionally local: every TP rank evaluates the same loss and owns the
gradient for only its vocabulary shard.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from art import dev

from .composite_loss import (
    CompositePreparedLoss,
    PreparedDistillationLoss,
    PreparedDistillationSidecars,
    PreparedPolicySidecars,
    compose_prepared_objective_losses,
    prepared_cispo_loss_from_logprobs,
)
from .distillation_loss import TopKPlusTailForwardKLResult


class _ForwardAllReduce(torch.autograd.Function):
    """All-reduce in forward while preserving rank-local backward ownership."""

    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        group: Any | None,
        op: Any,
    ) -> torch.Tensor:
        del ctx
        reduced = value.clone()
        torch.distributed.all_reduce(reduced, op=op, group=group)  # ty: ignore[possibly-missing-attribute]
        return reduced

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> tuple[torch.Tensor, None, None]:
        del ctx
        (grad_output,) = grad_outputs
        return grad_output, None, None


def _all_reduce_forward(
    value: torch.Tensor,
    *,
    group: Any | None,
    op: Any,
    world_size: int,
) -> torch.Tensor:
    if world_size == 1:
        return value
    return _ForwardAllReduce.apply(value, group, op)


def _validate_partition(
    local_logits: torch.Tensor,
    *,
    logical_vocab_size: int,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
) -> tuple[int, int]:
    if local_logits.ndim < 1 or not local_logits.is_floating_point():
        raise ValueError(
            "local_logits must be floating point with a vocabulary dimension"
        )
    if tensor_parallel_world_size <= 0:
        raise ValueError("tensor_parallel_world_size must be positive")
    if tensor_parallel_world_size > 1 and not torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        raise RuntimeError(
            "tensor-parallel scoring requires an initialized process group"
        )
    if not 0 <= tensor_parallel_rank < tensor_parallel_world_size:
        raise ValueError("tensor_parallel_rank is outside the TP world")
    local_width = int(local_logits.shape[-1])
    physical_vocab_size = local_width * tensor_parallel_world_size
    if logical_vocab_size <= 1 or logical_vocab_size > physical_vocab_size:
        raise ValueError(
            "logical_vocab_size must fit the padded tensor-parallel vocabulary"
        )
    start = tensor_parallel_rank * local_width
    return start, min(start + local_width, logical_vocab_size)


def _active_rows(
    tensor: torch.Tensor,
    *,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.shape != tensor.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean and match the token-row dimensions")
    if mask.device != tensor.device:
        raise ValueError("mask and logits must be on the same device")
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
    if int(indices.numel()) == 0:
        raise ValueError("prepared scorer mask selects zero tokens")
    rows = tensor.reshape(-1, tensor.shape[-1]).index_select(0, indices).float()
    return indices, rows


def _global_logsumexp(
    local_values: torch.Tensor,
    *,
    group: Any | None,
    world_size: int,
) -> torch.Tensor:
    local_max = local_values.detach().amax(dim=-1)
    global_max = _all_reduce_forward(
        local_max,
        group=group,
        op=torch.distributed.ReduceOp.MAX,  # ty: ignore[possibly-missing-attribute]
        world_size=world_size,
    )
    finite_max = torch.where(
        torch.isfinite(global_max),
        global_max,
        torch.zeros_like(global_max),
    )
    local_sum = torch.exp(local_values - finite_max.unsqueeze(-1)).sum(dim=-1)
    global_sum = _all_reduce_forward(
        local_sum,
        group=group,
        op=torch.distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
        world_size=world_size,
    )
    if bool(torch.any(global_sum <= 0).detach().item()):
        raise ValueError("vocabulary-parallel score has no finite logical logits")
    return finite_max + torch.log(global_sum)


def _logical_local_logits(
    local_logits: torch.Tensor,
    *,
    logical_vocab_size: int,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
) -> tuple[torch.Tensor, int, int]:
    start, end = _validate_partition(
        local_logits,
        logical_vocab_size=logical_vocab_size,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_world_size=tensor_parallel_world_size,
    )
    valid_width = max(end - start, 0)
    if valid_width > 0:
        valid = local_logits[..., :valid_width]
        if bool(torch.any(~torch.isfinite(valid)).detach().item()):
            raise ValueError("active logical student logits must be finite")
    else:
        valid = local_logits[..., :0]
    if valid_width == int(local_logits.shape[-1]):
        return valid.float(), start, end
    padded = local_logits.float().clone()
    padded[..., valid_width:] = -torch.inf
    return padded, start, end


def _selected_global_logits(
    local_values: torch.Tensor,
    *,
    global_ids: torch.Tensor,
    partition_start: int,
    partition_end: int,
    group: Any | None,
    world_size: int,
) -> torch.Tensor:
    local_ids = global_ids - partition_start
    owned = (global_ids >= partition_start) & (global_ids < partition_end)
    safe_ids = local_ids.clamp(min=0, max=max(int(local_values.shape[-1]) - 1, 0))
    selected = torch.gather(local_values, dim=-1, index=safe_ids)
    selected = torch.where(owned, selected, torch.zeros_like(selected))
    return _all_reduce_forward(
        selected,
        group=group,
        op=torch.distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
        world_size=world_size,
    )


def vocabulary_parallel_sampled_logprobs(
    local_logits: torch.Tensor,
    *,
    sampled_token_ids: torch.Tensor,
    mask: torch.Tensor,
    logical_vocab_size: int,
    tensor_parallel_group: Any | None,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
) -> torch.Tensor:
    """Score sampled global token IDs from one padded vocabulary shard."""

    if tensor_parallel_world_size > 1 and tensor_parallel_group is None:
        raise RuntimeError(
            "tensor-parallel scoring requires an explicit tensor-parallel group"
        )
    if sampled_token_ids.shape != mask.shape:
        raise ValueError("sampled_token_ids must match the policy mask")
    if sampled_token_ids.dtype == torch.bool or sampled_token_ids.is_floating_point():
        raise ValueError("sampled_token_ids must have an integer dtype")
    if sampled_token_ids.device != local_logits.device:
        raise ValueError("sampled token IDs and logits must be on the same device")
    indices, active_logits = _active_rows(local_logits, mask=mask)
    logical_logits, start, end = _logical_local_logits(
        active_logits,
        logical_vocab_size=logical_vocab_size,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_world_size=tensor_parallel_world_size,
    )
    active_ids = sampled_token_ids.reshape(-1).index_select(0, indices).long()
    if bool(
        torch.any((active_ids < 0) | (active_ids >= logical_vocab_size)).detach().item()
    ):
        raise ValueError("active sampled token ID exceeds the logical vocabulary")
    normalizer = _global_logsumexp(
        logical_logits,
        group=tensor_parallel_group,
        world_size=tensor_parallel_world_size,
    )
    selected = _selected_global_logits(
        logical_logits,
        global_ids=active_ids.unsqueeze(-1),
        partition_start=start,
        partition_end=end,
        group=tensor_parallel_group,
        world_size=tensor_parallel_world_size,
    ).squeeze(-1)
    result = local_logits.new_zeros(mask.numel(), dtype=torch.float32)
    return result.scatter(0, indices, selected - normalizer).reshape(mask.shape)


def vocabulary_parallel_topk_plus_tail_forward_kl(
    local_logits: torch.Tensor,
    *,
    sidecars: PreparedDistillationSidecars,
    logical_vocab_size: int,
    tensor_parallel_group: Any | None,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
) -> TopKPlusTailForwardKLResult:
    """Evaluate exact top-k-plus-residual forward KL over sharded logits."""

    if tensor_parallel_world_size > 1 and tensor_parallel_group is None:
        raise RuntimeError(
            "tensor-parallel scoring requires an explicit tensor-parallel group"
        )
    if not math.isfinite(sidecars.temperature) or sidecars.temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    indices, active_logits = _active_rows(local_logits, mask=sidecars.mask)
    logical_logits, start, end = _logical_local_logits(
        active_logits,
        logical_vocab_size=logical_vocab_size,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_world_size=tensor_parallel_world_size,
    )
    scaled_logits = logical_logits / sidecars.temperature
    k = int(sidecars.teacher_topk_ids.shape[-1])
    if k <= 0 or k >= logical_vocab_size:
        raise ValueError("top-k width must be in [1, logical_vocab_size)")
    if sidecars.teacher_topk_ids.shape[:-1] != sidecars.mask.shape:
        raise ValueError("teacher top-k rows must match the distillation mask")
    if (
        sidecars.teacher_topk_ids.dtype == torch.bool
        or sidecars.teacher_topk_ids.is_floating_point()
    ):
        raise ValueError("teacher top-k IDs must have an integer dtype")
    if sidecars.teacher_topk_logprobs.shape != sidecars.teacher_topk_ids.shape:
        raise ValueError("teacher top-k log-probabilities must match token IDs")
    if sidecars.teacher_tail_logprob.shape != sidecars.mask.shape:
        raise ValueError("teacher tail log-probability must match the mask")
    if sidecars.weights is not None and sidecars.weights.shape != sidecars.mask.shape:
        raise ValueError("distillation weights must match the mask")
    tensors = (
        sidecars.teacher_topk_ids,
        sidecars.teacher_topk_logprobs,
        sidecars.teacher_tail_logprob,
    )
    if sidecars.weights is not None:
        tensors = (*tensors, sidecars.weights)
    if any(tensor.device != local_logits.device for tensor in tensors):
        raise ValueError("all distillation tensors must be on the logits device")

    ids = sidecars.teacher_topk_ids.reshape(-1, k).index_select(0, indices).long()
    if bool(torch.any((ids < 0) | (ids >= logical_vocab_size)).detach().item()):
        raise ValueError("active teacher top-k IDs exceed the logical vocabulary")
    if k > 1 and bool(
        torch.any(ids.sort(dim=-1).values[:, 1:] == ids.sort(dim=-1).values[:, :-1])
        .detach()
        .item()
    ):
        raise ValueError("active teacher top-k IDs must be unique within each row")

    teacher_logprobs = (
        sidecars.teacher_topk_logprobs.reshape(-1, k).index_select(0, indices).float()
    )
    raw_tail_logprob = (
        sidecars.teacher_tail_logprob.reshape(-1).index_select(0, indices).float()
    )
    if bool(torch.any(~torch.isfinite(teacher_logprobs)).detach().item()):
        raise ValueError("active teacher top-k log-probabilities must be finite")
    invalid_tail = torch.isnan(raw_tail_logprob) | torch.isposinf(raw_tail_logprob)
    if bool(torch.any(invalid_tail).detach().item()):
        raise ValueError("active teacher tail log-probabilities must be finite or -inf")
    teacher_selected_probs = teacher_logprobs.exp()
    teacher_tail_prob = raw_tail_logprob.exp()
    teacher_total = teacher_selected_probs.sum(dim=-1) + teacher_tail_prob
    if not torch.allclose(
        teacher_total,
        torch.ones_like(teacher_total),
        rtol=2e-4,
        atol=2e-5,
    ):
        raise ValueError("active teacher top-k probabilities plus tail must sum to one")

    normalizer = _global_logsumexp(
        scaled_logits,
        group=tensor_parallel_group,
        world_size=tensor_parallel_world_size,
    )
    selected_logits = _selected_global_logits(
        scaled_logits,
        global_ids=ids,
        partition_start=start,
        partition_end=end,
        group=tensor_parallel_group,
        world_size=tensor_parallel_world_size,
    )
    selected_student_logprobs = selected_logits - normalizer.unsqueeze(-1)

    owned = (ids >= start) & (ids < end)
    safe_local_ids = (ids - start).clamp(
        min=0,
        max=max(int(scaled_logits.shape[-1]) - 1, 0),
    )
    selected_local_count = torch.zeros_like(scaled_logits, dtype=torch.int8)
    selected_local_count.scatter_reduce_(
        dim=-1,
        index=safe_local_ids,
        src=owned.to(dtype=torch.int8),
        reduce="amax",
        include_self=True,
    )
    local_tail_logits = scaled_logits.masked_fill(
        selected_local_count.bool(), -torch.inf
    )
    student_tail_logprob = (
        _global_logsumexp(
            local_tail_logits,
            group=tensor_parallel_group,
            world_size=tensor_parallel_world_size,
        )
        - normalizer
    )
    tail_logprob = torch.where(
        torch.isneginf(raw_tail_logprob),
        torch.zeros_like(raw_tail_logprob),
        raw_tail_logprob,
    )
    selected_loss = (
        teacher_selected_probs * (teacher_logprobs - selected_student_logprobs)
    ).sum(dim=-1)
    tail_loss = teacher_tail_prob * (tail_logprob - student_tail_logprob)
    scale = sidecars.temperature**2 if sidecars.compensate_temperature_squared else 1.0
    per_token = (selected_loss + tail_loss) * scale
    if sidecars.weights is None:
        weights = torch.ones_like(per_token)
    else:
        weights = sidecars.weights.reshape(-1).index_select(0, indices).float()
        if bool(torch.any(~torch.isfinite(weights)).detach().item()):
            raise ValueError("active distillation weights must be finite")
        if bool(torch.any(weights < 0).detach().item()):
            raise ValueError("active distillation weights must be non-negative")
    weighted = per_token * weights
    flat_loss = local_logits.new_zeros(sidecars.mask.numel(), dtype=torch.float32)
    flat_weighted = local_logits.new_zeros(sidecars.mask.numel(), dtype=torch.float32)
    selected_mass = torch.logsumexp(selected_student_logprobs, dim=-1).exp()
    tail_mass = student_tail_logprob.exp()
    return TopKPlusTailForwardKLResult(
        loss_sum=weighted.sum(),
        token_count=torch.tensor(
            int(indices.numel()),
            device=local_logits.device,
            dtype=torch.int64,
        ),
        per_token_loss=flat_loss.scatter(0, indices, per_token).reshape(
            sidecars.mask.shape
        ),
        per_token_weighted_loss=flat_weighted.scatter(0, indices, weighted).reshape(
            sidecars.mask.shape
        ),
        selected_loss_sum=(selected_loss * weights).sum() * scale,
        tail_loss_sum=(tail_loss * weights).sum() * scale,
        teacher_selected_mass_sum=teacher_selected_probs.sum(),
        teacher_tail_mass_sum=teacher_tail_prob.sum(),
        student_selected_mass_sum=selected_mass.sum(),
        student_tail_mass_sum=tail_mass.sum(),
        numerical_clamp_count=torch.zeros(
            (), device=local_logits.device, dtype=torch.int64
        ),
    )


def vocabulary_parallel_composite_prepared_loss(
    local_logits: torch.Tensor,
    *,
    logical_vocab_size: int,
    tensor_parallel_group: Any | None,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
    policy: PreparedPolicySidecars | None = None,
    distillation: PreparedDistillationSidecars | None = None,
    policy_config: dev.TrainConfig | None = None,
    distillation_coefficient: float = 1.0,
) -> CompositePreparedLoss:
    """Compose CISPO and KD from one tensor-parallel raw-logit forward."""

    if tensor_parallel_world_size > 1 and tensor_parallel_group is None:
        raise RuntimeError(
            "tensor-parallel scoring requires an explicit tensor-parallel group"
        )

    policy_loss = None
    if policy is not None:
        new_logprobs = vocabulary_parallel_sampled_logprobs(
            local_logits,
            sampled_token_ids=policy.sampled_token_ids,
            mask=policy.mask,
            logical_vocab_size=logical_vocab_size,
            tensor_parallel_group=tensor_parallel_group,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_world_size=tensor_parallel_world_size,
        )
        policy_loss = prepared_cispo_loss_from_logprobs(
            new_logprobs,
            sidecars=policy,
            config=policy_config,
        )

    distillation_loss = None
    if distillation is not None:
        details = vocabulary_parallel_topk_plus_tail_forward_kl(
            local_logits,
            sidecars=distillation,
            logical_vocab_size=logical_vocab_size,
            tensor_parallel_group=tensor_parallel_group,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_world_size=tensor_parallel_world_size,
        )
        distillation_loss = PreparedDistillationLoss(
            loss_sum=details.loss_sum,
            token_count=details.token_count,
            details=details,
        )
    return compose_prepared_objective_losses(
        policy=policy_loss,
        distillation=distillation_loss,
        distillation_coefficient=distillation_coefficient,
    )
