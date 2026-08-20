from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch
import torch.distributed as dist


class MambaHeadShardExchangePlan(BaseModel):
    """CPU plan for ART token ownership <-> Mamba head ownership."""

    model_config = ConfigDict(frozen=True)

    cp_size: int = Field(gt=0)
    token_positions_by_rank: tuple[tuple[int, ...], ...]
    total_token_count: int = Field(ge=0)
    heads_local_tp: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    groups_local_tp: int = Field(gt=0)
    state_dim: int = Field(gt=0)
    projected_feature_positions_by_rank: tuple[tuple[int, ...], ...]
    conv_feature_positions_by_rank: tuple[tuple[int, ...], ...]
    head_positions_by_rank: tuple[tuple[int, ...], ...]
    group_positions_by_rank: tuple[tuple[int, ...], ...]
    head_feature_positions_by_rank: tuple[tuple[int, ...], ...]
    canonical_to_received_positions: tuple[int, ...]
    canonical_flat_token_positions: tuple[int, ...]

    @model_validator(mode="after")
    def _validate_plan(self) -> MambaHeadShardExchangePlan:
        rank_fields = (
            self.token_positions_by_rank,
            self.projected_feature_positions_by_rank,
            self.conv_feature_positions_by_rank,
            self.head_positions_by_rank,
            self.group_positions_by_rank,
            self.head_feature_positions_by_rank,
        )
        if any(len(field) != self.cp_size for field in rank_fields):
            raise ValueError("Mamba exchange rank metadata must match cp_size")
        received = tuple(
            position for rank in self.token_positions_by_rank for position in rank
        )
        if sorted(received) != list(range(self.total_token_count)):
            raise ValueError(
                "Mamba token ownership must contain every canonical token exactly once"
            )
        if len(self.canonical_to_received_positions) != self.total_token_count:
            raise ValueError("canonical receive permutation has the wrong length")
        if sorted(self.canonical_to_received_positions) != list(
            range(self.total_token_count)
        ):
            raise ValueError("canonical receive positions must be a permutation")
        if len(self.canonical_flat_token_positions) != self.total_token_count:
            raise ValueError("canonical flat token positions have the wrong length")
        if len(set(self.canonical_flat_token_positions)) != self.total_token_count:
            raise ValueError("canonical flat token positions must be unique")
        if any(position < 0 for position in self.canonical_flat_token_positions):
            raise ValueError("canonical flat token positions must be nonnegative")
        projected_widths = {
            len(positions) for positions in self.projected_feature_positions_by_rank
        }
        conv_widths = {
            len(positions) for positions in self.conv_feature_positions_by_rank
        }
        head_widths = {
            len(positions) for positions in self.head_feature_positions_by_rank
        }
        if len(projected_widths) != 1 or len(conv_widths) != 1 or len(head_widths) != 1:
            raise ValueError("Mamba CP shards must have equal feature widths")
        return self

    @property
    def projected_width_local_tp(self) -> int:
        inner = self.heads_local_tp * self.head_dim
        return (
            2 * inner + 2 * self.groups_local_tp * self.state_dim + self.heads_local_tp
        )

    @property
    def inner_width_local_tp(self) -> int:
        return self.heads_local_tp * self.head_dim


class MambaHeadShardDevicePlan(BaseModel):
    """Device-materialized indices; construction belongs to CPU lookahead."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cpu: MambaHeadShardExchangePlan
    token_positions_by_rank: tuple[torch.Tensor, ...]
    projected_feature_positions_by_rank: tuple[torch.Tensor, ...]
    conv_feature_positions_by_rank: tuple[torch.Tensor, ...]
    head_positions_by_rank: tuple[torch.Tensor, ...]
    group_positions_by_rank: tuple[torch.Tensor, ...]
    head_feature_positions_by_rank: tuple[torch.Tensor, ...]
    canonical_to_received_positions: torch.Tensor
    canonical_flat_token_positions: torch.Tensor


def build_mamba_head_shard_exchange_plan(
    token_positions_by_rank: tuple[tuple[int, ...], ...],
    *,
    canonical_flat_token_positions: tuple[int, ...],
    heads_local_tp: int,
    head_dim: int,
    groups_local_tp: int,
    state_dim: int,
) -> MambaHeadShardExchangePlan:
    """Build the combined projected-stream split without touching CUDA."""

    cp_size = len(token_positions_by_rank)
    if cp_size == 0 or heads_local_tp % cp_size:
        raise ValueError("Mamba heads_local_tp must be divisible by CP size")
    if groups_local_tp < cp_size:
        if cp_size % groups_local_tp:
            raise ValueError("CP size must be divisible by Mamba groups_local_tp")
        group_repeat = cp_size // groups_local_tp
        groups_per_rank = 1
    else:
        if groups_local_tp % cp_size:
            raise ValueError("Mamba groups_local_tp must be divisible by CP size")
        group_repeat = 1
        groups_per_rank = groups_local_tp // cp_size

    heads_per_rank = heads_local_tp // cp_size
    inner = heads_local_tp * head_dim
    group_width = groups_local_tp * state_dim
    z_start, x_start = 0, inner
    b_start, c_start, dt_start = (
        2 * inner,
        2 * inner + group_width,
        2 * inner + 2 * group_width,
    )
    projected_positions = []
    conv_positions = []
    head_positions = []
    group_positions = []
    for rank in range(cp_size):
        head_start = rank * heads_per_rank
        inner_start = head_start * head_dim
        inner_end = inner_start + heads_per_rank * head_dim
        group_start = (rank // group_repeat) * groups_per_rank
        group_feature_start = group_start * state_dim
        group_feature_end = group_feature_start + groups_per_rank * state_dim
        projected_positions.append(
            tuple(range(z_start + inner_start, z_start + inner_end))
            + tuple(range(x_start + inner_start, x_start + inner_end))
            + tuple(range(b_start + group_feature_start, b_start + group_feature_end))
            + tuple(range(c_start + group_feature_start, c_start + group_feature_end))
            + tuple(
                range(dt_start + head_start, dt_start + head_start + heads_per_rank)
            )
        )
        conv_positions.append(
            tuple(range(inner_start, inner_end))
            + tuple(range(inner + group_feature_start, inner + group_feature_end))
            + tuple(
                range(
                    inner + group_width + group_feature_start,
                    inner + group_width + group_feature_end,
                )
            )
        )
        head_positions.append(tuple(range(head_start, head_start + heads_per_rank)))
        group_positions.append(tuple(range(group_start, group_start + groups_per_rank)))

    received_positions = tuple(
        position for positions in token_positions_by_rank for position in positions
    )
    canonical_to_received = [0] * len(received_positions)
    for received_index, canonical_position in enumerate(received_positions):
        canonical_to_received[canonical_position] = received_index
    return MambaHeadShardExchangePlan(
        cp_size=cp_size,
        token_positions_by_rank=token_positions_by_rank,
        total_token_count=len(received_positions),
        heads_local_tp=heads_local_tp,
        head_dim=head_dim,
        groups_local_tp=groups_local_tp,
        state_dim=state_dim,
        projected_feature_positions_by_rank=tuple(projected_positions),
        conv_feature_positions_by_rank=tuple(conv_positions),
        head_positions_by_rank=tuple(head_positions),
        group_positions_by_rank=tuple(group_positions),
        head_feature_positions_by_rank=tuple(
            tuple(
                range(
                    rank * heads_per_rank * head_dim,
                    (rank + 1) * heads_per_rank * head_dim,
                )
            )
            for rank in range(cp_size)
        ),
        canonical_to_received_positions=tuple(canonical_to_received),
        canonical_flat_token_positions=canonical_flat_token_positions,
    )


def materialize_mamba_head_shard_exchange_plan(
    plan: MambaHeadShardExchangePlan,
    device: torch.device | str,
) -> MambaHeadShardDevicePlan:
    """Move every runtime index during lookahead, before GPU forward."""

    def indices(rows: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(rows, dtype=torch.long, device=device)

    return MambaHeadShardDevicePlan(
        cpu=plan,
        token_positions_by_rank=tuple(map(indices, plan.token_positions_by_rank)),
        projected_feature_positions_by_rank=tuple(
            map(indices, plan.projected_feature_positions_by_rank)
        ),
        conv_feature_positions_by_rank=tuple(
            map(indices, plan.conv_feature_positions_by_rank)
        ),
        head_positions_by_rank=tuple(map(indices, plan.head_positions_by_rank)),
        group_positions_by_rank=tuple(map(indices, plan.group_positions_by_rank)),
        head_feature_positions_by_rank=tuple(
            map(indices, plan.head_feature_positions_by_rank)
        ),
        canonical_to_received_positions=indices(plan.canonical_to_received_positions),
        canonical_flat_token_positions=indices(plan.canonical_flat_token_positions),
    )


@torch.compiler.disable
def exchange_mamba_projected_to_head_shards(
    local_projected: torch.Tensor,
    plan: MambaHeadShardDevicePlan,
    *,
    group: Any | None = None,
) -> torch.Tensor:
    """One variable all-to-all for combined z/x/B/C/dt streams."""

    rank = _validate_runtime(local_projected, plan, group=group)
    if tuple(local_projected.shape) != (
        len(plan.cpu.token_positions_by_rank[rank]),
        plan.cpu.projected_width_local_tp,
    ):
        raise ValueError(
            "local projected Mamba tensor does not match token/feature plan: "
            f"got {tuple(local_projected.shape)}"
        )
    return _ProjectedToHead.apply(local_projected, plan, rank, group)


@torch.compiler.disable
def exchange_mamba_head_shards_to_attention(
    global_head_values: torch.Tensor,
    plan: MambaHeadShardDevicePlan,
    *,
    group: Any | None = None,
) -> torch.Tensor:
    """One inverse variable all-to-all restoring ART token ownership."""

    rank = _validate_runtime(global_head_values, plan, group=group)
    expected = (
        plan.cpu.total_token_count,
        len(plan.cpu.head_feature_positions_by_rank[rank]),
    )
    if tuple(global_head_values.shape) != expected:
        raise ValueError(
            f"global Mamba head values must be {expected}, got {tuple(global_head_values.shape)}"
        )
    return _HeadToAttention.apply(global_head_values, plan, rank, group)


class _ProjectedToHead(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        local_projected: torch.Tensor,
        plan: MambaHeadShardDevicePlan,
        rank: int,
        group: Any | None,
    ) -> torch.Tensor:
        ctx.plan, ctx.rank, ctx.group = plan, rank, group
        return _projected_to_head(local_projected, plan, rank=rank, group=group)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_output,) = grad_outputs
        return (
            _projected_head_gradient_to_attention(
                grad_output.contiguous(), ctx.plan, rank=ctx.rank, group=ctx.group
            ),
            None,
            None,
            None,
        )


class _HeadToAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        global_head_values: torch.Tensor,
        plan: MambaHeadShardDevicePlan,
        rank: int,
        group: Any | None,
    ) -> torch.Tensor:
        ctx.plan, ctx.rank, ctx.group = plan, rank, group
        return _head_to_attention(global_head_values, plan, rank=rank, group=group)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_output,) = grad_outputs
        return (
            _attention_gradient_to_head(
                grad_output.contiguous(), ctx.plan, rank=ctx.rank, group=ctx.group
            ),
            None,
            None,
            None,
        )


def _projected_to_head(
    local: torch.Tensor,
    plan: MambaHeadShardDevicePlan,
    *,
    rank: int,
    group: Any | None,
) -> torch.Tensor:
    send_parts = [
        local.index_select(1, features).reshape(-1)
        for features in plan.projected_feature_positions_by_rank
    ]
    width = len(plan.cpu.projected_feature_positions_by_rank[rank])
    recv = _all_to_all_flat(
        torch.cat(send_parts),
        send_splits=tuple(part.numel() for part in send_parts),
        recv_splits=tuple(
            len(tokens) * width for tokens in plan.cpu.token_positions_by_rank
        ),
        cp_size=plan.cpu.cp_size,
        group=group,
    )
    received = recv.view(plan.cpu.total_token_count, width)
    return received.index_select(0, plan.canonical_to_received_positions)


def _projected_head_gradient_to_attention(
    grad: torch.Tensor,
    plan: MambaHeadShardDevicePlan,
    *,
    rank: int,
    group: Any | None,
) -> torch.Tensor:
    send_parts = [
        grad.index_select(0, positions).reshape(-1)
        for positions in plan.token_positions_by_rank
    ]
    local_tokens = len(plan.cpu.token_positions_by_rank[rank])
    recv_splits = tuple(
        local_tokens * len(features)
        for features in plan.cpu.projected_feature_positions_by_rank
    )
    recv = _all_to_all_flat(
        torch.cat(send_parts),
        send_splits=tuple(part.numel() for part in send_parts),
        recv_splits=recv_splits,
        cp_size=plan.cpu.cp_size,
        group=group,
    )
    output = grad.new_zeros((local_tokens, plan.cpu.projected_width_local_tp))
    offset = 0
    for features, split in zip(
        plan.projected_feature_positions_by_rank, recv_splits, strict=True
    ):
        peer = recv[offset : offset + split].view(local_tokens, features.numel())
        output = output.index_add(1, features, peer)
        offset += split
    return output


def _head_to_attention(
    local: torch.Tensor,
    plan: MambaHeadShardDevicePlan,
    *,
    rank: int,
    group: Any | None,
) -> torch.Tensor:
    send_parts = [
        local.index_select(0, positions).reshape(-1)
        for positions in plan.token_positions_by_rank
    ]
    local_tokens = len(plan.cpu.token_positions_by_rank[rank])
    recv_splits = tuple(
        local_tokens * len(features)
        for features in plan.cpu.head_feature_positions_by_rank
    )
    recv = _all_to_all_flat(
        torch.cat(send_parts),
        send_splits=tuple(part.numel() for part in send_parts),
        recv_splits=recv_splits,
        cp_size=plan.cpu.cp_size,
        group=group,
    )
    output = local.new_zeros((local_tokens, plan.cpu.inner_width_local_tp))
    offset = 0
    for features, split in zip(
        plan.head_feature_positions_by_rank, recv_splits, strict=True
    ):
        peer = recv[offset : offset + split].view(local_tokens, features.numel())
        output = output.index_copy(1, features, peer)
        offset += split
    return output


def _attention_gradient_to_head(
    grad: torch.Tensor,
    plan: MambaHeadShardDevicePlan,
    *,
    rank: int,
    group: Any | None,
) -> torch.Tensor:
    send_parts = [
        grad.index_select(1, features).reshape(-1)
        for features in plan.head_feature_positions_by_rank
    ]
    width = len(plan.cpu.head_feature_positions_by_rank[rank])
    recv = _all_to_all_flat(
        torch.cat(send_parts),
        send_splits=tuple(part.numel() for part in send_parts),
        recv_splits=tuple(
            len(tokens) * width for tokens in plan.cpu.token_positions_by_rank
        ),
        cp_size=plan.cpu.cp_size,
        group=group,
    )
    received = recv.view(plan.cpu.total_token_count, width)
    return received.index_select(0, plan.canonical_to_received_positions)


def _all_to_all_flat(
    send: torch.Tensor,
    *,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
    cp_size: int,
    group: Any | None,
) -> torch.Tensor:
    if cp_size == 1:
        return send
    if (
        len(send_splits) != cp_size
        or len(recv_splits) != cp_size
        or sum(send_splits) != send.numel()
    ):
        raise ValueError("Mamba all-to-all split metadata does not match its buffer")
    padded_send, padded_send_splits = _pad_peer_splits(send, send_splits)
    padded_recv_splits = tuple(split + 1 for split in recv_splits)
    padded_recv = send.new_empty(sum(padded_recv_splits))
    dist.all_to_all_single(
        padded_recv,
        padded_send,
        output_split_sizes=list(padded_recv_splits),
        input_split_sizes=list(padded_send_splits),
        group=group,
    )
    return torch.cat(
        [peer[:-1] for peer in torch.split(padded_recv, list(padded_recv_splits))]
    )


def _pad_peer_splits(
    send: torch.Tensor, splits: tuple[int, ...]
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Keep NCCL all-to-all collective participation valid for empty ranks."""

    sentinel = send.new_zeros(1)
    peers = torch.split(send, list(splits))
    return (
        torch.cat([torch.cat((peer, sentinel)) for peer in peers]).contiguous(),
        tuple(split + 1 for split in splits),
    )


def _validate_runtime(
    tensor: torch.Tensor,
    plan: MambaHeadShardDevicePlan,
    *,
    group: Any | None,
) -> int:
    device_indices = (
        *plan.token_positions_by_rank,
        *plan.projected_feature_positions_by_rank,
        *plan.conv_feature_positions_by_rank,
        *plan.head_positions_by_rank,
        *plan.group_positions_by_rank,
        *plan.head_feature_positions_by_rank,
        plan.canonical_to_received_positions,
        plan.canonical_flat_token_positions,
    )
    if any(index.device != tensor.device for index in device_indices):
        raise ValueError(
            "Mamba exchange indices must be materialized on the tensor device"
        )
    if plan.cpu.cp_size == 1:
        return 0
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for Mamba CP")
    if dist.get_world_size(group) != plan.cpu.cp_size:
        raise ValueError("Mamba CP process-group size does not match the exchange plan")
    return dist.get_rank(group)
