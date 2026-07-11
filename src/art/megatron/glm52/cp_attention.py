from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl

from art.megatron.context_parallel.types import ArtContextParallelState
from art.megatron.glm52.cp_stage import (
    drain_stage_fetches,
    launch_remote_stage_fetches,
    launch_remote_stage_reduce,
    reduce_local_stage_rows_,
    stage_kv_rows,
    stage_query_rows,
)
from art.megatron.glm52.indexer import Glm52RoutedTopk
from art.megatron.glm52.sparse_mla import sparse_mla_backward, sparse_mla_forward
from art.megatron.glm52.state import Glm52PrefixTreeState

_LATENT_DIM = 512


@triton.jit
def _merge_stage_kernel(
    stage_out_ptr,
    stage_lse_ptr,
    owner_rows_ptr,
    accum_out_ptr,
    accum_lse_ptr,
    heads,
    stride_soq,
    stride_soh,
    stride_sod,
    stride_slq,
    stride_slh,
    stride_aoq,
    stride_aoh,
    stride_aod,
    stride_alq,
    stride_alh,
    latent_dim: tl.constexpr,
    block_d: tl.constexpr,
):
    query = tl.program_id(0)
    head = tl.program_id(1)
    d_block = tl.program_id(2)
    owner = tl.load(owner_rows_ptr + query)
    dimensions = d_block * block_d + tl.arange(0, block_d)
    mask = (head < heads) & (dimensions < latent_dim)
    stage_lse = tl.load(stage_lse_ptr + query * stride_slq + head * stride_slh)
    accum_lse_offset = owner * stride_alq + head * stride_alh
    previous_lse = tl.load(accum_lse_ptr + accum_lse_offset)
    both_empty = (previous_lse == float("-inf")) & (stage_lse == float("-inf"))
    maximum = tl.maximum(previous_lse, stage_lse)
    merged_lse = maximum + tl.log(
        tl.exp(previous_lse - maximum) + tl.exp(stage_lse - maximum)
    )
    merged_lse = tl.where(both_empty, float("-inf"), merged_lse)
    previous_delta = tl.where(
        (previous_lse == float("-inf")) & (merged_lse == float("-inf")),
        float("-inf"),
        previous_lse - merged_lse,
    )
    stage_delta = tl.where(
        (stage_lse == float("-inf")) & (merged_lse == float("-inf")),
        float("-inf"),
        stage_lse - merged_lse,
    )
    previous = tl.load(
        accum_out_ptr
        + owner * stride_aoq
        + head * stride_aoh
        + dimensions * stride_aod,
        mask=mask,
        other=0.0,
    )
    stage = tl.load(
        stage_out_ptr
        + query * stride_soq
        + head * stride_soh
        + dimensions * stride_sod,
        mask=mask,
        other=0.0,
    )
    merged = previous * tl.exp(previous_delta) + stage * tl.exp(stage_delta)
    tl.store(
        accum_out_ptr
        + owner * stride_aoq
        + head * stride_aoh
        + dimensions * stride_aod,
        merged,
        mask=mask,
    )
    tl.store(
        accum_lse_ptr + accum_lse_offset,
        merged_lse,
        mask=(d_block == 0) & (head < heads),
    )


def _merge_stage_(
    accum_out: torch.Tensor,
    accum_lse: torch.Tensor,
    stage_out: torch.Tensor,
    stage_lse: torch.Tensor,
    owner_rows: torch.Tensor,
) -> None:
    q_len, heads = stage_lse.shape
    _merge_stage_kernel[(q_len, heads, 1)](
        stage_out,
        stage_lse,
        owner_rows,
        accum_out,
        accum_lse,
        heads,
        *stage_out.stride(),
        *stage_lse.stride(),
        *accum_out.stride(),
        *accum_lse.stride(),
        latent_dim=_LATENT_DIM,  # ty: ignore[invalid-argument-type]
        block_d=_LATENT_DIM,  # ty: ignore[invalid-argument-type]
        num_warps=8,  # ty: ignore[unknown-argument]
    )


def _forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    state: Glm52PrefixTreeState,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cp_state = cast(ArtContextParallelState, state.context_parallel_state)
    valid = int(sum(cp_state.rank_plan.local_valid_lengths))
    q_flat = q[0, :valid].contiguous()
    kv_flat = kv[0, :valid].contiguous()
    accum_out = torch.zeros(
        (valid, q.shape[2], _LATENT_DIM), device=q.device, dtype=torch.float32
    )
    accum_lse = torch.full(
        (valid, q.shape[2]), float("-inf"), device=q.device, dtype=torch.float32
    )
    fetches = launch_remote_stage_fetches(kv_flat, cp_state)
    for stage_plan, stage in zip(
        cp_state.rank_plan.stage_plans, state.stages, strict=True
    ):
        if not stage.slices:
            continue
        q_stage = stage_query_rows(q_flat, stage_plan, cp_state)
        kv_stage = stage_kv_rows(kv_flat, stage_plan, cp_state, fetches)
        indices_stage = stage_query_rows(indices[0], stage_plan, cp_state)
        offsets_stage = stage_query_rows(offsets[0], stage_plan, cp_state)
        stage_out, stage_lse = sparse_mla_forward(
            q_stage.unsqueeze(0),
            kv_stage.unsqueeze(0),
            indices_stage.unsqueeze(0),
            scale=scale,
            route_offsets=offsets_stage.unsqueeze(0),
            stage_index=int(stage.stage_index),
            fp32_output=True,
        )
        _merge_stage_(
            accum_out,
            accum_lse,
            stage_out[0],
            stage_lse[0],
            stage.owner_q_rows,
        )
        del q_stage, kv_stage, indices_stage, offsets_stage, stage_out, stage_lse
    drain_stage_fetches(fetches)
    output = q.new_zeros((q.shape[0], q.shape[1], q.shape[2], _LATENT_DIM))
    output[0, :valid].copy_(accum_out)
    return output, output[0, :valid], accum_lse


def _backward(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    global_out: torch.Tensor,
    global_lse: torch.Tensor,
    state: Glm52PrefixTreeState,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cp_state = cast(ArtContextParallelState, state.context_parallel_state)
    valid = int(sum(cp_state.rank_plan.local_valid_lengths))
    q_flat = q[0, :valid].contiguous()
    kv_flat = kv[0, :valid].contiguous()
    grad_flat = grad_output[0, :valid].contiguous()
    dq = torch.zeros_like(q_flat, dtype=torch.float32)
    dkv = torch.zeros_like(kv_flat)
    fetches = launch_remote_stage_fetches(kv_flat, cp_state)
    reductions = []
    for stage_index in cp_state.rank_plan.backward_stage_indices:
        stage_plan = cp_state.rank_plan.stage_plans[int(stage_index)]
        stage = state.stages[int(stage_index)]
        kv_stage = stage_kv_rows(kv_flat, stage_plan, cp_state, fetches)
        if stage.slices:
            q_stage = stage_query_rows(q_flat, stage_plan, cp_state)
            indices_stage = stage_query_rows(indices[0], stage_plan, cp_state)
            offsets_stage = stage_query_rows(offsets[0], stage_plan, cp_state)
            out_stage = global_out.index_select(0, stage.owner_q_rows)
            lse_stage = global_lse.index_select(0, stage.owner_q_rows)
            grad_stage = grad_flat.index_select(0, stage.owner_q_rows)
            dq_stage, dkv_stage = sparse_mla_backward(
                q_stage.unsqueeze(0),
                kv_stage.unsqueeze(0),
                indices_stage.unsqueeze(0),
                out_stage.unsqueeze(0),
                lse_stage.unsqueeze(0),
                grad_stage.unsqueeze(0),
                scale=scale,
                route_offsets=offsets_stage.unsqueeze(0),
                stage_index=int(stage.stage_index),
                fp32_grad_q=True,
            )
            dq.index_add_(0, stage.owner_q_rows, dq_stage[0])
            dkv_stage = dkv_stage[0]
        else:
            dkv_stage = torch.zeros_like(kv_stage)
        if stage_plan.is_local_stage:
            reduce_local_stage_rows_(dkv, dkv_stage, stage_plan, cp_state)
        else:
            reductions.append(
                launch_remote_stage_reduce(dkv_stage, stage_plan, cp_state, dkv)
            )
    drain_stage_fetches(fetches)
    for reduction in reductions:
        reduction.wait_post_process()
    dq_padded = torch.zeros_like(q)
    dkv_padded = torch.zeros_like(kv)
    dq_padded[0, :valid].copy_(dq)
    dkv_padded[0, :valid].copy_(dkv)
    return dq_padded, dkv_padded


class _ContextParallelSparseMla(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        state: Glm52PrefixTreeState,
        scale: float,
    ) -> torch.Tensor:
        output, global_out, global_lse = _forward(q, kv, indices, offsets, state, scale)
        ctx.save_for_backward(q, kv, indices, offsets, global_out, global_lse)
        ctx.state = state
        ctx.scale = float(scale)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        (grad_output,) = cast(tuple[torch.Tensor], grad_outputs)
        q, kv, indices, offsets, global_out, global_lse = ctx.saved_tensors
        dq, dkv = _backward(
            grad_output,
            q,
            kv,
            indices,
            offsets,
            global_out,
            global_lse,
            ctx.state,
            ctx.scale,
        )
        return dq, dkv, None, None, None, None


def context_parallel_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk: Glm52RoutedTopk,
    state: Glm52PrefixTreeState,
    *,
    scale: float,
) -> torch.Tensor:
    """Run stage-routed sparse MLA with global-LSE backward replay."""
    if q.ndim != 4 or kv.ndim != 3 or q.shape[:2] != kv.shape[:2]:
        raise ValueError("GLM-5.2 CP sparse MLA expects q[B,S,H,576], kv[B,S,576].")
    if q.shape[0] != 1:
        raise ValueError("GLM-5.2 context parallel supports one packed row.")
    return _ContextParallelSparseMla.apply(
        q.contiguous(),
        kv.contiguous(),
        topk.indices.contiguous(),
        topk.stage_offsets.contiguous(),
        state,
        float(scale),
    )
