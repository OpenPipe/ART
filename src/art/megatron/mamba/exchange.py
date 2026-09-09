from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.distributed.nn.functional import all_to_all_single

from .exchange_kernels import assemble_head_shards, pack_projected
from .plan import MambaTokenExchangePlan


class MambaShardShape(BaseModel):
    model_config = ConfigDict(frozen=True)

    inner: int = Field(gt=0)
    heads: int = Field(gt=0)
    groups: int = Field(gt=0)
    state_dim: int = Field(gt=0)


def projected_tokens_to_recurrent_layout(
    projected: torch.Tensor,
    plan: MambaTokenExchangePlan,
    shape: MambaShardShape,
    group: object,
) -> torch.Tensor:
    """Exchange token-sharded projections into source-ordered CP head shards."""

    if projected.ndim != 3:
        raise ValueError(
            f"Mamba projection must be [sequence, batch, width], got {projected.shape}"
        )
    flat = projected.flatten(0, 1)
    if plan.cp_size == 1:
        return flat.index_select(0, plan.physical_token_positions)
    if int(projected.shape[1]) != 1:
        raise ValueError("ART Mamba CP supports exactly one packed sequence")
    local = flat[: plan.local_token_count]
    send = pack_projected(
        local,
        inner=shape.inner,
        heads=shape.heads,
        groups=shape.groups,
        state_dim=shape.state_dim,
        cp_size=plan.cp_size,
    )
    local_width = int(send.shape[-1])
    received = _all_to_all_flat(
        send.flatten(),
        send_splits=(plan.local_token_count * local_width,) * plan.cp_size,
        receive_splits=tuple(count * local_width for count in plan.source_token_counts),
        group=group,
    ).view(plan.token_count, local_width)
    return received


def recurrent_layout_to_token_layout(
    recurrent: torch.Tensor,
    projected_shape: tuple[int, int, int],
    plan: MambaTokenExchangePlan,
    shape: MambaShardShape,
    group: object,
) -> torch.Tensor:
    """Return source-ordered recurrent head shards to their token owners."""

    local_inner = shape.inner // plan.cp_size
    expected = (plan.token_count, local_inner)
    if tuple(recurrent.shape) != expected:
        raise ValueError(f"Mamba recurrent output must have shape {expected}")
    if plan.cp_size == 1:
        flat = recurrent.new_zeros(
            (projected_shape[0] * projected_shape[1], shape.inner)
        )
        return flat.index_copy_(0, plan.physical_token_positions, recurrent).view(
            *projected_shape[:2], shape.inner
        )
    return _recurrent_send_to_token_layout(
        recurrent, projected_shape, plan, local_inner, group
    )


def _recurrent_send_to_token_layout(
    send: torch.Tensor,
    projected_shape: tuple[int, int, int],
    plan: MambaTokenExchangePlan,
    local_inner: int,
    group: object,
) -> torch.Tensor:
    received = _all_to_all_flat(
        send.flatten(),
        send_splits=tuple(count * local_inner for count in plan.source_token_counts),
        receive_splits=(plan.local_token_count * local_inner,) * plan.cp_size,
        group=group,
    )
    flat_size = projected_shape[0] * projected_shape[1]
    if flat_size < plan.local_token_count:
        raise ValueError(
            "Mamba output token layout is smaller than its real token count"
        )
    flat = assemble_head_shards(
        received,
        flat_tokens=flat_size,
        tokens=plan.local_token_count,
        cp_size=plan.cp_size,
        local_inner=local_inner,
    )
    return flat.view(*projected_shape)


def _all_to_all_flat(
    tensor: torch.Tensor,
    *,
    send_splits: tuple[int, ...],
    receive_splits: tuple[int, ...],
    group: object,
) -> torch.Tensor:
    output = tensor.new_empty(sum(receive_splits))
    return all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=list(receive_splits),
        input_split_sizes=list(send_splits),
        group=group,
    )
