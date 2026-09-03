from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict
import torch
import torch.nn.functional as F

from .token_matrix import NamedLossRequest


class TokenMatrixLossOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reported_loss: torch.Tensor
    backward_loss: torch.Tensor
    learner_logprobs: torch.Tensor
    probability_ratio: torch.Tensor | None = None


def logical_active_position_count(
    logical_value_mask: torch.Tensor,
    logical_advantages: torch.Tensor,
) -> int:
    """Count one activity marker per accepted logical token position."""

    if logical_value_mask.shape != logical_advantages.shape:
        raise ValueError("TokenMatrix logical activity tensors must have one shape")
    return int(((logical_advantages != 0) & logical_value_mask).sum().item())


def gather_logical_projection_values(
    *,
    local_values: torch.Tensor,
    local_projection_ids: torch.Tensor,
    logical_projection_ids: torch.Tensor,
    logical_value_mask: torch.Tensor,
    projection_count: int,
    cp_group: Any | None,
) -> torch.Tensor:
    """Gather only selected values and scatter them to logical occurrences."""

    if local_values.shape != local_projection_ids.shape:
        raise ValueError("local projection values and IDs must have one shape")
    if logical_projection_ids.shape != logical_value_mask.shape:
        raise ValueError("logical projection IDs and mask must have one shape")
    if projection_count < 0:
        raise ValueError("projection_count must be nonnegative")
    flat_ids = local_projection_ids.reshape(-1)
    flat_values = local_values.reshape(-1)
    valid = flat_ids >= 0
    selected_ids = flat_ids[valid].to(dtype=torch.long)
    zero_dependency = flat_values.sum() * 0.0
    if projection_count == 0:
        return local_values.new_zeros(logical_projection_ids.shape) + zero_dependency
    projection_values = local_values.new_zeros((projection_count,)) + zero_dependency
    if selected_ids.numel():
        projection_values = projection_values.index_add(
            0, selected_ids, flat_values[valid]
        )
    if cp_group is not None and torch.distributed.get_world_size(cp_group) > 1:
        from torch.distributed.nn.functional import all_reduce

        projection_values = all_reduce(projection_values, group=cp_group)
    logical_ids = logical_projection_ids.masked_fill(~logical_value_mask, 0)
    result = projection_values.index_select(0, logical_ids.reshape(-1)).reshape_as(
        logical_projection_ids
    )
    return result.masked_fill(~logical_value_mask, 0.0)


def execute_token_matrix_loss(
    request: NamedLossRequest,
    *,
    learner_logprobs: torch.Tensor,
    loss_weights: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    logical_value_mask: torch.Tensor,
    logical_matrix_indices: torch.Tensor,
    matrix_pairs: tuple[tuple[int, int], ...] = (),
    cp_group: Any | None = None,
) -> TokenMatrixLossOutput:
    """Evaluate one complete logical loss view on every context-parallel rank."""

    tensors = (
        learner_logprobs,
        loss_weights,
        behavior_logprobs,
        advantages,
        logical_value_mask,
        logical_matrix_indices,
    )
    if any(tensor.shape != learner_logprobs.shape for tensor in tensors):
        raise ValueError("TokenMatrix logical loss tensors must have one shape")
    weights = loss_weights.masked_fill(~logical_value_mask, 0.0)
    ratio = None
    if request.name == "cross_entropy":
        reported_loss = -(learner_logprobs * weights).sum()
    elif request.name == "dpo":
        reported_loss = _dpo_loss(
            request,
            learner_logprobs=learner_logprobs,
            behavior_logprobs=behavior_logprobs,
            loss_weights=weights,
            matrix_indices=logical_matrix_indices.masked_fill(~logical_value_mask, -1),
            matrix_pairs=matrix_pairs,
        )
    else:
        ratio = torch.exp(learner_logprobs - behavior_logprobs)
        if request.name == "importance_sampling":
            reported_loss = -(ratio * advantages * weights).sum()
        elif request.name == "cispo":
            low = _numeric_setting(request, "clip_low_threshold", 0.0)
            high = _numeric_setting(request, "clip_high_threshold", 4.0)
            clipped = ratio.clamp(min=low, max=high)
            reported_loss = -(
                clipped.detach() * learner_logprobs * advantages * weights
            ).sum()
        else:
            raise AssertionError(request.name)
    cp_size = torch.distributed.get_world_size(cp_group) if cp_group is not None else 1
    return TokenMatrixLossOutput(
        reported_loss=reported_loss,
        backward_loss=reported_loss / cp_size,
        learner_logprobs=learner_logprobs,
        probability_ratio=ratio,
    )


def _dpo_loss(
    request: NamedLossRequest,
    *,
    learner_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    loss_weights: torch.Tensor,
    matrix_indices: torch.Tensor,
    matrix_pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    if not matrix_pairs:
        raise ValueError("dpo requires resolved matrix pairs")
    matrix_count = max(max(pair) for pair in matrix_pairs) + 1
    values = (learner_logprobs - behavior_logprobs) * loss_weights
    zero_dependency = values.sum() * 0.0
    logratios = learner_logprobs.new_zeros((matrix_count,)) + zero_dependency
    residency = torch.zeros(matrix_count, device=values.device, dtype=torch.long)
    valid = (matrix_indices >= 0) & (matrix_indices < matrix_count)
    logratios = logratios.index_add(
        0,
        matrix_indices[valid].to(dtype=torch.long),
        values[valid],
    )
    residency = residency.index_add(
        0,
        matrix_indices[valid].to(dtype=torch.long),
        torch.ones_like(matrix_indices[valid], dtype=torch.long),
    )
    chosen = torch.tensor(
        [pair[0] for pair in matrix_pairs], device=values.device, dtype=torch.long
    )
    rejected = torch.tensor(
        [pair[1] for pair in matrix_pairs], device=values.device, dtype=torch.long
    )
    chosen_present = residency.index_select(0, chosen) > 0
    rejected_present = residency.index_select(0, rejected) > 0
    if bool(torch.logical_xor(chosen_present, rejected_present).any().item()):
        raise RuntimeError("dpo component was split across packed microbatches")
    resident_pairs = chosen_present & rejected_present
    beta = _numeric_setting(request, "beta", 0.1)
    if not bool(resident_pairs.any().item()):
        return zero_dependency
    resident_chosen = chosen[resident_pairs]
    resident_rejected = rejected[resident_pairs]
    return (
        -F.logsigmoid(
            beta
            * (
                logratios.index_select(0, resident_chosen)
                - logratios.index_select(0, resident_rejected)
            )
        ).sum()
        + zero_dependency
    )


def _numeric_setting(request: NamedLossRequest, name: str, default: float) -> float:
    raw = request.values.get(name, default)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise TypeError(f"{name} must be numeric")
    return float(raw)
