from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
import torch.distributed as dist


def chunk_gated_delta_rule_native_cp(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor,
    group: Any,
    output_final_state: bool,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_cpu: Tensor | None = None,
    lengths_by_rank_cpu: Tensor | None = None,
    scale: float | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Run FLA gated-delta recurrence on one CP-sharded logical chain.

    This is the ART-owned extension missing from FLA's public CP surface:
    parent recurrent state is injected at rank 0, FLA summary scans seed every
    rank-local shard, and chain-tail state is emitted on every rank.
    """

    if group is None:
        raise ValueError("native FLA CP GDN requires a process group")
    if not dist.is_available() or not dist.is_initialized():  # ty: ignore[possibly-missing-attribute]
        raise RuntimeError("torch.distributed must be initialized for native FLA CP")
    if q.ndim != 4 or int(q.shape[0]) != 1:
        raise ValueError(f"q must be [1, T, H, K], got {tuple(q.shape)}")
    if tuple(k.shape) != tuple(q.shape):
        raise ValueError(f"k shape must match q, got {tuple(k.shape)}")
    if v.ndim != 4 or tuple(v.shape[:3]) != tuple(q.shape[:3]):
        raise ValueError(f"v must be [1, T, H, V], got {tuple(v.shape)}")
    if tuple(g.shape) != tuple(q.shape[:3]):
        raise ValueError(f"g must be [1, T, H], got {tuple(g.shape)}")
    if tuple(beta.shape) != tuple(q.shape[:3]):
        raise ValueError(f"beta must be [1, T, H], got {tuple(beta.shape)}")
    if int(q.shape[1]) <= 0:
        raise ValueError("native FLA CP GDN currently requires non-empty rank shards")
    if initial_state.ndim != 4:
        raise ValueError(
            f"initial_state must be [N, H, K, V], got {tuple(initial_state.shape)}"
        )
    if cu_seqlens is None and int(initial_state.shape[0]) != 1:
        raise ValueError("single-chain native FLA CP requires one initial state")
    if cu_seqlens is not None:
        if cu_seqlens_cpu is None:
            raise ValueError("native FLA CP varlen requires CPU cu_seqlens metadata")
        if cu_seqlens.ndim != 1:
            raise ValueError(
                f"cu_seqlens must be rank 1, got {tuple(cu_seqlens.shape)}"
            )
        if cu_seqlens_cpu.ndim != 1:
            raise ValueError(
                f"cu_seqlens_cpu must be rank 1, got {tuple(cu_seqlens_cpu.shape)}"
            )
        if cu_seqlens_cpu.device.type != "cpu":
            raise ValueError("native FLA CP cu_seqlens_cpu must stay on CPU")
        if int(cu_seqlens.numel()) != int(initial_state.shape[0]) + 1:
            raise ValueError(
                "cu_seqlens entries must equal initial_state batch + 1, got "
                f"{int(cu_seqlens.numel())} and {int(initial_state.shape[0])}"
            )
        if int(cu_seqlens_cpu.numel()) != int(cu_seqlens.numel()):
            raise ValueError(
                "cu_seqlens_cpu entries must match cu_seqlens, got "
                f"{int(cu_seqlens_cpu.numel())} and {int(cu_seqlens.numel())}"
            )
    if tuple(initial_state.shape[1:3]) != tuple(q.shape[2:4]):
        raise ValueError(
            "initial_state H/K must match q, got "
            f"{tuple(initial_state.shape)} for q {tuple(q.shape)}"
        )
    if int(initial_state.shape[-1]) != int(v.shape[-1]):
        raise ValueError(
            "initial_state V must match v, got "
            f"{tuple(initial_state.shape)} for v {tuple(v.shape)}"
        )
    if scale is None:
        scale = float(k.shape[-1] ** -0.5)
    if lengths_by_rank_cpu is None:
        raise ValueError("native FLA CP requires static all-rank sequence lengths")
    if lengths_by_rank_cpu.device.type != "cpu":
        raise ValueError("native FLA CP lengths_by_rank_cpu must stay on CPU")
    if tuple(lengths_by_rank_cpu.shape) != (
        dist.get_world_size(group),  # ty: ignore[possibly-missing-attribute]
        int(initial_state.shape[0]),
    ):
        raise ValueError(
            "native FLA CP lengths_by_rank_cpu must be [world_size, segments], got "
            f"{tuple(lengths_by_rank_cpu.shape)}"
        )
    if not _fla_chunk_boundaries_aligned_cpu(lengths_by_rank_cpu):
        raise ValueError(
            "native FLA CP GDN requires 64-token aligned non-final rank "
            f"boundaries; lengths_by_rank={lengths_by_rank_cpu.tolist()}"
        )
    return _NativeCpChunkGatedDeltaRule.apply(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        cu_seqlens_cpu,
        group,
        bool(output_final_state),
        float(scale),
    )


def _fla_chunk_boundaries_aligned_cpu(lengths_by_rank: Tensor) -> bool:
    if int(lengths_by_rank.shape[0]) <= 1:
        return True
    starts = torch.cumsum(lengths_by_rank, dim=0)[:-1]
    return bool(torch.all(starts.remainder(64) == 0).item())


class _NativeCpChunkGatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        beta: Tensor,
        initial_state: Tensor,
        cu_seqlens: Tensor | None,
        cu_seqlens_cpu: Tensor | None,
        group: Any,
        output_final_state: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor | None]:
        from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from fla.ops.common.chunk_o import chunk_fwd_o
        from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
        from fla.ops.utils import chunk_local_cumsum, prepare_chunk_indices, solve_tril

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu)
            if cu_seqlens is not None
            else None
        )
        chunk_local_cumsum = cast(Any, chunk_local_cumsum)
        chunk_fwd_o = cast(Any, chunk_fwd_o)
        chunk_scaled_dot_kkt_fwd = cast(Any, chunk_scaled_dot_kkt_fwd)
        solve_tril = cast(Any, solve_tril)
        recompute_w_u_fwd = cast(Any, recompute_w_u_fwd)
        g_cumsum = chunk_local_cumsum(
            g,
            chunk_size=64,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        a = chunk_scaled_dot_kkt_fwd(
            k=k,
            g=g_cumsum,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=torch.float32,
        )
        a = solve_tril(
            A=a,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=k.dtype,
        )
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=a,
            g=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        summary = _fwd_summary(k=k, w=w, u=u, g=g_cumsum, cu_seqlens=cu_seqlens)
        prefix_summary = _prefix_summary_exclusive(summary, group)
        local_initial = _scan_fwd_initial_state(
            prefix_summary,
            initial_state,
        )
        h, v_new, local_final_state = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=local_initial,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        out = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g_cumsum,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        final_state = (
            _broadcast_chain_final_state(local_final_state, group)
            if output_final_state
            else None
        )
        ctx.save_for_backward(q, k, v, g_cumsum, beta, a, local_initial)
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.group = group
        ctx.scale = scale
        ctx.output_final_state = output_final_state
        return out.to(q.dtype), final_state

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor | None) -> tuple[Any, ...]:
        from fla.ops.common.chunk_o import chunk_bwd_dv_local
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_bwd
        from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd

        q, k, v, g, beta, a, local_initial = ctx.saved_tensors
        do = grad_outputs[0]
        if do is None:
            do = v.new_zeros(v.shape)
        dht = grad_outputs[1]
        do = cast(Tensor, do)
        recompute_w_u_fwd = cast(Any, recompute_w_u_fwd)
        chunk_bwd_dv_local = cast(Any, chunk_bwd_dv_local)
        chunk_gated_delta_rule_bwd = cast(Any, chunk_gated_delta_rule_bwd)
        w, _ = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=a,
            g=g,
            cu_seqlens=ctx.cu_seqlens,
            chunk_indices=ctx.chunk_indices,
        )
        dv_local = chunk_bwd_dv_local(
            q=q,
            k=k,
            g=g,
            do=do,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
            chunk_indices=ctx.chunk_indices,
        )
        external_dht = _external_final_state_grad(
            dht,
            local_initial,
            group=ctx.group,
            enabled=ctx.output_final_state,
        )
        bwd_summary = _bwd_summary(
            q=q,
            k=k,
            w=w,
            g=g,
            do=do,
            dv=dv_local,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
        )
        suffix_summary, full_bwd_summary = _suffix_summary_exclusive_and_full(
            bwd_summary, ctx.group
        )
        local_dht = _scan_bwd_local_final_grad(
            suffix_summary,
            external_dht,
        )
        dq, dk, dv, db, dg, _dh0, _dA_log, _ddt_bias = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=a,
            scale=ctx.scale,
            initial_state=local_initial,
            do=do,
            dht=local_dht,
            cu_seqlens=ctx.cu_seqlens,
            chunk_indices=ctx.chunk_indices,
            use_exp2=False,
        )
        dh0 = _scan_bwd_initial_state_grad(full_bwd_summary, external_dht)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            dh0.to(local_initial),
            None,
            None,
            None,
            None,
            None,
        )


def _fwd_summary(
    *, k: Tensor, w: Tensor, u: Tensor, g: Tensor, cu_seqlens: Tensor | None = None
) -> Tensor:
    from fla.ops.cp.chunk_delta_h import pre_process_fwd_kernel_merged
    import triton

    _, token_count, head_count, key_dim = k.shape
    value_dim = u.shape[-1]
    sequence_count = 1 if cu_seqlens is None else int(cu_seqlens.numel()) - 1
    summary_shape = (
        (head_count, key_dim, value_dim + key_dim)
        if cu_seqlens is None
        else (sequence_count, head_count, key_dim, value_dim + key_dim)
    )
    summary = k.new_zeros(*summary_shape, dtype=torch.float32)
    block_size = 32 if key_dim <= 64 else 64
    grid = (
        triton.cdiv(value_dim, block_size) + triton.cdiv(key_dim, block_size),
        head_count,
        *(() if cu_seqlens is None else (sequence_count,)),
    )
    pre_process_fwd_kernel_merged[grid](
        k=k,
        v=u,
        w=w,
        g=g,
        gk=None,
        bg=None,
        u=u,
        hm=summary,
        cu_seqlens=cu_seqlens,
        T=token_count,
        H=head_count,
        HV=head_count,
        K=key_dim,
        V=value_dim,
        BT=64,
        BK1=max(16, triton.next_power_of_2(key_dim)),
        USE_EXP2=False,
        BLOCK_SIZE=block_size,
        MULTI_SEQS=cu_seqlens is not None,
    )
    return summary


def _bwd_summary(
    *,
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g: Tensor,
    do: Tensor,
    dv: Tensor,
    scale: float,
    cu_seqlens: Tensor | None = None,
) -> Tensor:
    from fla.ops.cp.chunk_delta_h import pre_process_bwd_kernel_merged
    import triton

    if cu_seqlens is not None:
        from .fla_cp_kernels import pre_process_bwd_summary_multi

        return pre_process_bwd_summary_multi(
            q=q,
            k=k,
            w=w,
            g=g,
            do=do,
            dv=dv,
            cu_seqlens=cu_seqlens,
            scale=scale,
        )

    _, token_count, head_count, key_dim = q.shape
    value_dim = do.shape[-1]
    sequence_count = 1 if cu_seqlens is None else int(cu_seqlens.numel()) - 1
    summary_shape = (
        (head_count, key_dim, value_dim + key_dim)
        if cu_seqlens is None
        else (sequence_count, head_count, key_dim, value_dim + key_dim)
    )
    summary = q.new_zeros(*summary_shape, dtype=torch.float32)
    block_size = 32 if key_dim <= 64 else 64
    grid = (
        triton.cdiv(value_dim, block_size) + triton.cdiv(key_dim, block_size),
        head_count,
        *(() if cu_seqlens is None else (sequence_count,)),
    )
    pre_process_bwd_kernel_merged[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=None,
        do=do,
        dhm=summary,
        dv=dv,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=token_count,
        H=head_count,
        HV=head_count,
        K=key_dim,
        V=value_dim,
        BT=64,
        BK1=max(16, triton.next_power_of_2(key_dim)),
        USE_BG=False,
        USE_EXP2=False,
        BLOCK_SIZE=block_size,
    )
    return summary


def _prefix_summary_exclusive(summary: Tensor, group: Any) -> Tensor | None:
    inclusive = _prefix_summary_inclusive(summary, group)
    rank = dist.get_rank(group)  # ty: ignore[possibly-missing-attribute]
    world_size = dist.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    return _exchange_summary(
        inclusive,
        group=group,
        send_to=rank + 1 if rank + 1 < world_size else None,
        recv_from=rank - 1 if rank > 0 else None,
    )


def _prefix_summary_inclusive(summary: Tensor, group: Any) -> Tensor:
    rank = dist.get_rank(group)  # ty: ignore[possibly-missing-attribute]
    world_size = dist.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    aggregate = summary.contiguous()
    offset = 1
    while offset < world_size:
        received = _exchange_summary(
            aggregate,
            group=group,
            send_to=rank + offset if rank + offset < world_size else None,
            recv_from=rank - offset if rank >= offset else None,
        )
        if received is not None:
            aggregate = _compose_summary(aggregate, received)
        offset *= 2
    return aggregate


def _suffix_summary_exclusive_and_full(
    summary: Tensor, group: Any
) -> tuple[Tensor | None, Tensor]:
    inclusive = _suffix_summary_inclusive(summary, group)
    rank = dist.get_rank(group)  # ty: ignore[possibly-missing-attribute]
    world_size = dist.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    exclusive = _exchange_summary(
        inclusive,
        group=group,
        send_to=rank - 1 if rank > 0 else None,
        recv_from=rank + 1 if rank + 1 < world_size else None,
    )
    full = inclusive if rank == 0 else torch.empty_like(summary)
    dist.broadcast(full, src=0, group=group)  # ty: ignore[possibly-missing-attribute]
    return exclusive, full


def _suffix_summary_inclusive(summary: Tensor, group: Any) -> Tensor:
    rank = dist.get_rank(group)  # ty: ignore[possibly-missing-attribute]
    world_size = dist.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    aggregate = summary.contiguous()
    offset = 1
    while offset < world_size:
        received = _exchange_summary(
            aggregate,
            group=group,
            send_to=rank - offset if rank >= offset else None,
            recv_from=rank + offset if rank + offset < world_size else None,
        )
        if received is not None:
            aggregate = _compose_summary(aggregate, received)
        offset *= 2
    return aggregate


def _exchange_summary(
    summary: Tensor,
    *,
    group: Any,
    send_to: int | None,
    recv_from: int | None,
) -> Tensor | None:
    world_size = dist.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    count = int(summary.numel())
    input_splits = [0] * world_size
    output_splits = [0] * world_size
    if send_to is None:
        send_buffer = summary.new_empty((0,))
    else:
        input_splits[int(send_to)] = count
        send_buffer = summary.reshape(-1).contiguous()
    if recv_from is not None:
        output_splits[int(recv_from)] = count
    recv_buffer = summary.new_empty((sum(output_splits),))
    dist.all_to_all_single(  # ty: ignore[possibly-missing-attribute]
        recv_buffer,
        send_buffer,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    if recv_from is None:
        return None
    return recv_buffer.view_as(summary)


def _compose_summary(after: Tensor, before: Tensor) -> Tensor:
    value_dim = int(before.shape[-1]) - int(before.shape[-2])
    before_he = before[..., :value_dim]
    before_transition = before[..., value_dim:]
    after_he = after[..., :value_dim]
    after_transition = after[..., value_dim:]
    he = torch.matmul(after_transition.float(), before_he.float()) + after_he.float()
    transition = torch.matmul(
        after_transition.float(),
        before_transition.float(),
    )
    return torch.cat((he, transition), dim=-1).contiguous()


def _scan_fwd_initial_state(summary: Tensor | None, h0: Tensor) -> Tensor:
    if summary is None:
        return h0.float()
    multi = summary.ndim == 4
    state = h0.float() if multi else h0[0].float()
    state = _apply_summary(summary, state)
    return state if multi else state.unsqueeze(0)


def _broadcast_chain_final_state(final_state: Tensor | None, group: Any) -> Tensor:
    if final_state is None:
        raise RuntimeError("native FLA CP did not produce a local final state")
    owner = dist.get_world_size(group) - 1  # ty: ignore[possibly-missing-attribute]
    final_state = final_state.contiguous()
    dist.broadcast(final_state, src=owner, group=group)  # ty: ignore[possibly-missing-attribute]
    return final_state


def _scan_bwd_local_final_grad(
    summary: Tensor | None,
    dht: Tensor,
) -> Tensor:
    if summary is None:
        return dht.float()
    multi = summary.ndim == 4
    state = dht.float() if multi else dht[0].float()
    state = _apply_summary(summary, state)
    return state if multi else state.unsqueeze(0)


def _scan_bwd_initial_state_grad(summary: Tensor, dht: Tensor) -> Tensor:
    multi = summary.ndim == 4
    state = dht.float() if multi else dht[0].float()
    state = _apply_summary(summary, state)
    return state if multi else state.unsqueeze(0)


def _apply_summary(summary: Tensor, state: Tensor) -> Tensor:
    value_dim = state.shape[-1]
    he = summary[..., :value_dim]
    transition = summary[..., value_dim:]
    return torch.matmul(transition.float(), state.float()) + he.float()


def _external_final_state_grad(
    dht: Tensor | None,
    reference: Tensor,
    *,
    group: Any,
    enabled: bool,
) -> Tensor:
    grad = reference.new_zeros(reference.shape, dtype=torch.float32)
    if not enabled:
        return grad
    if dht is not None:
        grad = dht.contiguous().float()
    dist.all_reduce(  # ty: ignore[possibly-missing-attribute]
        grad,
        op=dist.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
        group=group,
    )
    return grad
