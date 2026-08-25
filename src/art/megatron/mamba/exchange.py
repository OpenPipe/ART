from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.distributed.nn.functional import all_to_all_single

from .plan import MambaTokenExchangePlan


class MambaShardShape(BaseModel):
    model_config = ConfigDict(frozen=True)

    inner: int = Field(gt=0)
    heads: int = Field(gt=0)
    groups: int = Field(gt=0)
    state_dim: int = Field(gt=0)


def projected_tokens_to_canonical_head_shard(
    projected: torch.Tensor,
    plan: MambaTokenExchangePlan,
    shape: MambaShardShape,
    group: object,
) -> torch.Tensor:
    """Exchange token-sharded projections into canonical tokens and CP-sharded heads."""

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
    z, x, b, c, dt = torch.split(
        local,
        [
            shape.inner,
            shape.inner,
            shape.groups * shape.state_dim,
            shape.groups * shape.state_dim,
            shape.heads,
        ],
        dim=-1,
    )
    send_chunks = [
        torch.cat(
            (
                _even_slice(z, rank, plan.cp_size),
                _even_slice(x, rank, plan.cp_size),
                _group_slice(b, rank, plan.cp_size, shape.groups, shape.state_dim),
                _group_slice(c, rank, plan.cp_size, shape.groups, shape.state_dim),
                _even_slice(dt, rank, plan.cp_size),
            ),
            dim=-1,
        )
        for rank in range(plan.cp_size)
    ]
    local_width = int(send_chunks[0].shape[-1])
    received = _all_to_all_flat(
        torch.cat(send_chunks, dim=0).flatten(),
        send_splits=(plan.local_token_count * local_width,) * plan.cp_size,
        receive_splits=tuple(count * local_width for count in plan.source_token_counts),
        group=group,
    ).view(plan.token_count, local_width)
    return received.new_zeros(received.shape).index_copy(
        0, plan.received_global_positions, received
    )


def canonical_head_shard_to_token_layout(
    canonical: torch.Tensor,
    projected_shape: tuple[int, int, int],
    plan: MambaTokenExchangePlan,
    shape: MambaShardShape,
    group: object,
) -> torch.Tensor:
    """Return CP head-sharded Mamba output to ART's attention token layout."""

    local_inner = shape.inner // plan.cp_size
    if tuple(canonical.shape) != (plan.token_count, local_inner):
        raise ValueError(
            f"canonical Mamba output has shape {tuple(canonical.shape)}, expected "
            f"{(plan.token_count, local_inner)}"
        )
    if plan.cp_size == 1:
        flat = canonical.new_zeros(
            (projected_shape[0] * projected_shape[1], shape.inner)
        )
        return flat.index_copy(0, plan.physical_token_positions, canonical).view(
            projected_shape[0], projected_shape[1], shape.inner
        )
    send_chunks = [
        canonical.index_select(0, positions)
        for positions in plan.global_positions_by_rank
    ]
    received = _all_to_all_flat(
        torch.cat(send_chunks, dim=0).flatten(),
        send_splits=tuple(count * local_inner for count in plan.source_token_counts),
        receive_splits=(plan.local_token_count * local_inner,) * plan.cp_size,
        group=group,
    )
    assembled = torch.cat(
        tuple(
            received.narrow(
                0,
                rank * plan.local_token_count * local_inner,
                plan.local_token_count * local_inner,
            ).view(plan.local_token_count, local_inner)
            for rank in range(plan.cp_size)
        ),
        dim=-1,
    )
    flat_size = projected_shape[0] * projected_shape[1]
    if flat_size < plan.local_token_count:
        raise ValueError(
            "Mamba output token layout is smaller than its real token count"
        )
    flat = canonical.new_zeros((flat_size, shape.inner))
    flat = flat.index_copy(
        0,
        torch.arange(plan.local_token_count, device=canonical.device),
        assembled,
    )
    return flat.view(projected_shape[0], projected_shape[1], shape.inner)


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


def _even_slice(tensor: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    if int(tensor.shape[-1]) % size:
        raise ValueError("Mamba head features must divide evenly across CP")
    width = int(tensor.shape[-1]) // size
    return tensor.narrow(-1, rank * width, width)


def _group_slice(
    tensor: torch.Tensor,
    rank: int,
    cp_size: int,
    groups: int,
    state_dim: int,
) -> torch.Tensor:
    if groups >= cp_size:
        if groups % cp_size:
            raise ValueError("Mamba groups must divide evenly across CP")
        local_groups = groups // cp_size
        return tensor.narrow(
            -1, rank * local_groups * state_dim, local_groups * state_dim
        )
    if cp_size % groups:
        raise ValueError("Mamba CP must divide evenly across replicated groups")
    group = rank // (cp_size // groups)
    return tensor.narrow(-1, group * state_dim, state_dim)
