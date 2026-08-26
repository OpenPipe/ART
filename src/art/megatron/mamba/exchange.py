from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch.distributed.nn.functional import all_to_all_single

from .exchange_kernels import assemble_head_shards, pack_canonical_pair, pack_projected
from .permutation import permute_rows
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
    return permute_rows(received, plan.received_canonical_order)


def canonical_head_shard_to_token_layout(
    canonical: torch.Tensor,
    projected_shape: tuple[int, int, int],
    plan: MambaTokenExchangePlan,
    shape: MambaShardShape,
    group: object,
) -> torch.Tensor:
    """Return CP head-sharded Mamba output to ART's attention token layout."""

    local_inner = shape.inner // plan.cp_size
    if canonical.ndim != 2:
        raise ValueError(f"canonical Mamba output must be 2D, got {canonical.shape}")
    components, remainder = divmod(int(canonical.shape[-1]), local_inner)
    if int(canonical.shape[0]) != plan.token_count or not components or remainder:
        raise ValueError(
            "canonical Mamba output must contain whole CP-local head components; "
            f"got {tuple(canonical.shape)}"
        )
    local_width = components * local_inner
    output_width = components * shape.inner
    if plan.cp_size == 1:
        flat = canonical.new_zeros(
            (projected_shape[0] * projected_shape[1], output_width)
        )
        return flat.index_copy_(0, plan.physical_token_positions, canonical).view(
            projected_shape[0], projected_shape[1], output_width
        )
    send = permute_rows(canonical, plan.received_global_positions)
    return _canonical_send_to_token_layout(
        send, projected_shape, plan, components, local_inner, group
    )


def canonical_head_shard_pair_to_token_layout(
    first: torch.Tensor,
    second: torch.Tensor,
    projected_shape: tuple[int, int, int],
    plan: MambaTokenExchangePlan,
    shape: MambaShardShape,
    group: object,
) -> torch.Tensor:
    """Fuse Mamba recurrent/gate packing into the return token exchange."""

    local_inner = shape.inner // plan.cp_size
    expected = (plan.token_count, local_inner)
    if tuple(first.shape) != expected or tuple(second.shape) != expected:
        raise ValueError(f"canonical Mamba components must both have shape {expected}")
    if plan.cp_size == 1:
        return canonical_head_shard_to_token_layout(
            torch.cat((first, second), dim=-1),
            projected_shape,
            plan,
            shape,
            group,
        )
    send = pack_canonical_pair(first, second, plan.received_global_positions)
    return _canonical_send_to_token_layout(
        send, projected_shape, plan, 2, local_inner, group
    )


def _canonical_send_to_token_layout(
    send: torch.Tensor,
    projected_shape: tuple[int, int, int],
    plan: MambaTokenExchangePlan,
    components: int,
    local_inner: int,
    group: object,
) -> torch.Tensor:
    local_width = components * local_inner
    output_width = components * plan.cp_size * local_inner
    received = _all_to_all_flat(
        send.flatten(),
        send_splits=tuple(count * local_width for count in plan.source_token_counts),
        receive_splits=(plan.local_token_count * local_width,) * plan.cp_size,
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
        components=components,
        local_inner=local_inner,
    )
    return flat.view(projected_shape[0], projected_shape[1], output_width)


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
