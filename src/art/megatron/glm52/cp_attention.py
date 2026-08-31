from __future__ import annotations

from typing import Any, cast

import torch

from art.megatron.context_parallel.types import ArtContextParallelState
from art.megatron.glm52.cp_stage import (
    drain_stage_fetches,
    launch_remote_stage_fetches,
    launch_remote_stage_reduce,
    reduce_local_stage_rows_,
    stage_kv_rows,
)
from art.megatron.glm52.indexer import Glm52RoutedTopk
from art.megatron.glm52.sparse_mla import (
    reduce_tensor_parallel_dkv,
    sparse_mla_backward,
    sparse_mla_forward,
)
from art.megatron.glm52.state import Glm52PrefixTreeState

_LATENT_DIM = 512


def _combined_stage_kv(
    kv: torch.Tensor,
    cp_state: ArtContextParallelState,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    fetches = launch_remote_stage_fetches(kv, cp_state)
    parts = tuple(
        stage_kv_rows(kv, stage, cp_state, fetches)
        for stage in cp_state.rank_plan.stage_plans
    )
    drain_stage_fetches(fetches)
    if not parts:
        raise RuntimeError("GLM-5.2 CP plan has no KV stages.")
    return (
        parts[0] if len(parts) == 1 else torch.cat(parts),
        tuple(int(part.shape[0]) for part in parts),
    )


def _forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    state: Glm52PrefixTreeState,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cp_state = cast(ArtContextParallelState, state.context_parallel_state)
    valid = int(sum(cp_state.rank_plan.local_valid_lengths))
    kv_flat = kv[0, :valid].contiguous()
    combined_kv, _ = _combined_stage_kv(kv_flat, cp_state)
    combined_out, lse = sparse_mla_forward(
        q[:, :valid].contiguous(),
        combined_kv.unsqueeze(0),
        indices[:, :valid].contiguous(),
        scale=scale,
    )
    if valid == q.shape[1]:
        return combined_out, combined_out[0], lse[0]
    output = q.new_zeros((q.shape[0], q.shape[1], q.shape[2], _LATENT_DIM))
    output[:, :valid].copy_(combined_out)
    return output, combined_out[0], lse[0]


def _backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    global_out: torch.Tensor,
    global_lse: torch.Tensor,
    state: Glm52PrefixTreeState,
    scale: float,
    tp_group: Any | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cp_state = cast(ArtContextParallelState, state.context_parallel_state)
    valid = int(sum(cp_state.rank_plan.local_valid_lengths))
    kv_flat = kv[0, :valid].contiguous()
    dkv = torch.zeros_like(kv_flat)
    combined_kv, stage_sizes = _combined_stage_kv(kv_flat, cp_state)
    dq, combined_dkv = sparse_mla_backward(
        q[:, :valid].contiguous(),
        combined_kv.unsqueeze(0),
        indices[:, :valid].contiguous(),
        global_out.unsqueeze(0),
        global_lse.unsqueeze(0),
        grad_output[:, :valid].contiguous(),
        scale=scale,
    )
    combined_dkv = reduce_tensor_parallel_dkv(
        combined_dkv, tp_group=tp_group, dtype=kv.dtype
    )
    stage_starts = [0]
    for size in stage_sizes:
        stage_starts.append(stage_starts[-1] + size)
    reductions = []
    for stage_index in cp_state.rank_plan.backward_stage_indices:
        stage_plan = cp_state.rank_plan.stage_plans[int(stage_index)]
        start, end = stage_starts[int(stage_index) : int(stage_index) + 2]
        dkv_stage = combined_dkv[0, start:end]
        if stage_plan.is_local_stage:
            reduction = reduce_local_stage_rows_(dkv, dkv_stage, stage_plan, cp_state)
            if reduction is not None:
                reductions.append(reduction)
        else:
            reductions.append(
                launch_remote_stage_reduce(dkv_stage, stage_plan, cp_state, dkv)
            )
    for reduction in reductions:
        reduction.wait_post_process()
    if valid == q.shape[1]:
        return dq, dkv.unsqueeze(0)
    dq_padded, dkv_padded = torch.zeros_like(q), torch.zeros_like(kv)
    dq_padded[:, :valid].copy_(dq)
    dkv_padded[0, :valid].copy_(dkv)
    return dq_padded, dkv_padded


class _ContextParallelSparseMla(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        state: Glm52PrefixTreeState,
        scale: float,
        tp_group: Any | None,
    ) -> torch.Tensor:
        output, global_out, global_lse = _forward(q, kv, indices, state, scale)
        ctx.save_for_backward(q, kv, indices, global_out, global_lse)
        ctx.state = state
        ctx.scale = float(scale)
        ctx.tp_group = tp_group
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = cast(tuple[torch.Tensor], grad_outputs)
        q, kv, indices, global_out, global_lse = ctx.saved_tensors
        dq, dkv = _backward(
            grad_output,
            q,
            kv,
            indices,
            global_out,
            global_lse,
            ctx.state,
            ctx.scale,
            ctx.tp_group,
        )
        return dq, dkv, None, None, None, None


def context_parallel_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk: Glm52RoutedTopk,
    state: Glm52PrefixTreeState,
    *,
    scale: float,
    tp_group: Any | None = None,
) -> torch.Tensor:
    """Run sparse MLA once over the union of ART-planned KV stages."""
    if q.ndim != 4 or kv.ndim != 3 or q.shape[:2] != kv.shape[:2]:
        raise ValueError("GLM-5.2 CP sparse MLA expects q[B,S,H,576], kv[B,S,576].")
    if q.shape[0] != 1:
        raise ValueError("GLM-5.2 context parallel supports one packed row.")
    return _ContextParallelSparseMla.apply(
        q.contiguous(),
        kv.contiguous(),
        topk.indices.contiguous(),
        state,
        float(scale),
        tp_group,
    )
