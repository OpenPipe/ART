from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict
import torch
import triton
import triton.language as tl

from art.megatron.context_parallel.types import ArtContextParallelState
from art.megatron.glm52.cp_stage import (
    drain_stage_fetches,
    launch_remote_stage_fetches,
    stage_kv_rows,
    stage_query_rows,
)
from art.megatron.glm52.state import (
    Glm52IndexerRowPlan,
    Glm52PrefixTreeState,
    Glm52StageState,
)

_MAX_SCORE_WORKSPACE_BYTES = 256 * 1024 * 1024
_MAX_K_CHUNK = 32 * 1024


class Glm52RoutedTopk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    indices: torch.Tensor


@triton.jit
def _canonicalize_topk_kernel(
    ids_ptr,
    topk: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    ids = tl.load(
        ids_ptr + row * topk + columns,
        mask=columns < topk,
        other=0x7FFF_FFFF,
    )
    ids = tl.where(ids >= 0, ids, 0x7FFF_FFFF)
    ids = tl.sort(ids)
    tl.store(
        ids_ptr + row * topk + columns,
        tl.where(ids == 0x7FFF_FFFF, -1, ids),
        mask=columns < topk,
    )


def _canonicalize_topk_(ids: torch.Tensor) -> None:
    topk = int(ids.shape[-1])
    block = triton.next_power_of_2(topk)
    _canonicalize_topk_kernel[(ids.numel() // topk,)](
        ids,
        topk=topk,  # ty: ignore[invalid-argument-type]
        block=block,  # ty: ignore[invalid-argument-type]
        num_warps=4,  # ty: ignore[unknown-argument]
    )


@triton.jit
def _round_bf16(value):
    bits = value.to(tl.int32, bitcast=True)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded & -0x10000).to(tl.float32, bitcast=True)


@triton.jit
def _index_rope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    tokens,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_rb,
    stride_rs,
    stride_rd,
    heads: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    batch = row // tokens
    token = row - batch * tokens
    half = tl.arange(0, 32)
    passthrough = tl.arange(0, 64)
    rope_base = batch * stride_rb + token * stride_rs
    cos = tl.load(cos_ptr + rope_base + half * stride_rd)
    sin = tl.load(sin_ptr + rope_base + half * stride_rd)

    q_base = batch * stride_qb + token * stride_qs + head * stride_qh
    q_first = tl.load(q_ptr + q_base + half * stride_qd)
    q_second = tl.load(q_ptr + q_base + (32 + half) * stride_qd)
    q_ac = _round_bf16(q_first.to(tl.float32) * cos.to(tl.float32))
    q_bs = _round_bf16(q_second.to(tl.float32) * sin.to(tl.float32))
    q_bc = _round_bf16(q_second.to(tl.float32) * cos.to(tl.float32))
    q_as = _round_bf16(q_first.to(tl.float32) * sin.to(tl.float32))
    tl.store(
        q_out_ptr + q_base + half * stride_qd,
        _round_bf16(q_ac - q_bs),
    )
    tl.store(
        q_out_ptr + q_base + (32 + half) * stride_qd,
        _round_bf16(q_bc + q_as),
    )
    tl.store(
        q_out_ptr + q_base + (64 + passthrough) * stride_qd,
        tl.load(q_ptr + q_base + (64 + passthrough) * stride_qd),
    )

    k_base = batch * stride_kb + token * stride_ks
    k_mask = head == 0
    k_first = tl.load(k_ptr + k_base + half * stride_kd, mask=k_mask, other=0.0)
    k_second = tl.load(k_ptr + k_base + (32 + half) * stride_kd, mask=k_mask, other=0.0)
    k_ac = _round_bf16(k_first.to(tl.float32) * cos.to(tl.float32))
    k_bs = _round_bf16(k_second.to(tl.float32) * sin.to(tl.float32))
    k_bc = _round_bf16(k_second.to(tl.float32) * cos.to(tl.float32))
    k_as = _round_bf16(k_first.to(tl.float32) * sin.to(tl.float32))
    tl.store(
        k_out_ptr + k_base + half * stride_kd,
        _round_bf16(k_ac - k_bs),
        mask=k_mask,
    )
    tl.store(
        k_out_ptr + k_base + (32 + half) * stride_kd,
        _round_bf16(k_bc + k_as),
        mask=k_mask,
    )
    tl.store(
        k_out_ptr + k_base + (64 + passthrough) * stride_kd,
        tl.load(
            k_ptr + k_base + (64 + passthrough) * stride_kd,
            mask=k_mask,
            other=0.0,
        ),
        mask=k_mask,
    )


def indexer_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply half-split RoPE with the indexer's eager-BF16 rounding contract."""
    if q.ndim != 4 or k.ndim != 3 or q.shape[:2] != k.shape[:2]:
        raise ValueError("GLM-5.2 indexer RoPE expects q[B,S,H,128], k[B,S,128].")
    if q.shape[-1] != 128 or k.shape[-1] != 128:
        raise ValueError("GLM-5.2 indexer RoPE requires head_dim=128.")
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
        raise TypeError("GLM-5.2 indexer RoPE requires BF16 q/k.")
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    batch, tokens, heads, _ = q.shape
    _index_rope_kernel[(batch * tokens, heads)](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        tokens,
        *q.stride(),
        *k.stride(),
        *cos.stride(),
        heads=heads,  # ty: ignore[invalid-argument-type]
        num_warps=1,  # ty: ignore[unknown-argument]
    )
    return q_out, k_out


@triton.jit
def _index_scores_kernel(
    q_ptr,
    k_ptr,
    weights_ptr,
    scores_ptr,
    q_len,
    k_len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kd,
    stride_wt,
    stride_wh,
    stride_st,
    stride_sk,
    q_position_offset: tl.constexpr,
    k_position_offset: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    causal: tl.constexpr,
    ranking_keys: tl.constexpr,
):
    q_block = tl.program_id(0)
    k_block = tl.program_id(1)
    q_offsets = q_block * block_q + tl.arange(0, block_q)
    h_offsets = tl.arange(0, heads)
    d_offsets = tl.arange(0, head_dim)
    k_offsets = k_block * block_k + tl.arange(0, block_k)

    qh_offsets = q_offsets[:, None] * heads + h_offsets[None, :]
    qh_offsets = qh_offsets.reshape((block_q * heads,))
    q = tl.load(
        q_ptr
        + (qh_offsets // heads)[:, None] * stride_qt
        + (qh_offsets % heads)[:, None] * stride_qh
        + d_offsets[None, :] * stride_qd,
        mask=(qh_offsets[:, None] // heads < q_len),
        other=0.0,
    )
    k = tl.load(
        k_ptr + k_offsets[None, :] * stride_kt + d_offsets[:, None] * stride_kd,
        mask=k_offsets[None, :] < k_len,
        other=0.0,
    )
    dots = tl.dot(q, k).reshape((block_q, heads, block_k))
    weights = tl.load(
        weights_ptr + q_offsets[:, None] * stride_wt + h_offsets[None, :] * stride_wh,
        mask=q_offsets[:, None] < q_len,
        other=0.0,
    )
    scores = tl.sum(tl.maximum(dots, 0.0) * weights[:, :, None], axis=1)
    valid = (q_offsets[:, None] < q_len) & (k_offsets[None, :] < k_len)
    if causal:
        valid &= (
            k_position_offset + k_offsets[None, :]
            <= q_position_offset + q_offsets[:, None]
        )
    scores = tl.where(valid, scores, float("-inf"))
    output_offsets = (
        scores_ptr + q_offsets[:, None] * stride_st + k_offsets[None, :] * stride_sk
    )
    output_mask = (q_offsets[:, None] < q_len) & (k_offsets[None, :] < k_len)
    if ranking_keys:
        canonical_scores = tl.where(scores == 0.0, 0.0, scores)
        bits = canonical_scores.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFF_FFFF
        ordered = tl.where(
            (bits >> 31) != 0,
            (~bits) & 0xFFFF_FFFF,
            bits ^ 0x8000_0000,
        )
        primary = ordered - 0x8000_0000
        global_ids = (k_position_offset + k_offsets[None, :]).to(tl.int64)
        keys = (primary << 32) | (0xFFFF_FFFF - global_ids)
        keys = tl.where(valid, keys, -0x8000_0000_0000_0000)
        tl.store(output_offsets, keys, mask=output_mask)
    else:
        tl.store(output_offsets, scores, mask=output_mask)


def _index_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    q_position_offset: int,
    k_position_offset: int,
    causal: bool,
) -> torch.Tensor:
    if q.dtype is not torch.bfloat16 or k.dtype is not torch.bfloat16:
        raise TypeError(f"GLM-5.2 index q/k must be bf16, got {q.dtype}/{k.dtype}.")
    if weights.dtype is not torch.float32:
        raise TypeError(f"GLM-5.2 index weights must be fp32, got {weights.dtype}.")
    if q.ndim != 3 or k.ndim != 2 or weights.shape != q.shape[:2]:
        raise ValueError(
            "GLM-5.2 index score shapes must be q[Q,H,D], k[K,D], w[Q,H], "
            f"got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(weights.shape)}."
        )
    q_len, heads, head_dim = q.shape
    k_len = int(k.shape[0])
    if int(k.shape[1]) != head_dim or 128 % heads:
        raise ValueError(
            f"Unsupported GLM-5.2 index shape heads={heads}, head_dim={head_dim}."
        )
    block_q = 128 // heads
    block_k = 64
    scores = torch.empty((q_len, k_len), device=q.device, dtype=torch.float32)
    _index_scores_kernel[(triton.cdiv(q_len, block_q), triton.cdiv(k_len, block_k))](
        q,
        k,
        weights,
        scores,
        q_len,
        k_len,
        *q.stride(),
        *k.stride(),
        *weights.stride(),
        *scores.stride(),
        q_position_offset=q_position_offset,  # ty: ignore[invalid-argument-type]
        k_position_offset=k_position_offset,  # ty: ignore[invalid-argument-type]
        heads=heads,  # ty: ignore[invalid-argument-type]
        head_dim=head_dim,  # ty: ignore[invalid-argument-type]
        block_q=block_q,  # ty: ignore[invalid-argument-type]
        block_k=block_k,  # ty: ignore[invalid-argument-type]
        causal=causal,  # ty: ignore[invalid-argument-type]
        ranking_keys=False,  # ty: ignore[invalid-argument-type]
        num_warps=8,  # ty: ignore[unknown-argument]
        num_stages=3,  # ty: ignore[unknown-argument]
    )
    return scores


def _index_score_keys(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    q_position_offset: int,
    k_position_offset: int,
    causal: bool,
) -> torch.Tensor:
    q_len, heads, head_dim = q.shape
    k_len = int(k.shape[0])
    block_q = 128 // heads
    block_k = 64
    keys = torch.empty((q_len, k_len), device=q.device, dtype=torch.int64)
    _index_scores_kernel[(triton.cdiv(q_len, block_q), triton.cdiv(k_len, block_k))](
        q,
        k,
        weights,
        keys,
        q_len,
        k_len,
        *q.stride(),
        *k.stride(),
        *weights.stride(),
        *keys.stride(),
        q_position_offset=q_position_offset,  # ty: ignore[invalid-argument-type]
        k_position_offset=k_position_offset,  # ty: ignore[invalid-argument-type]
        heads=heads,  # ty: ignore[invalid-argument-type]
        head_dim=head_dim,  # ty: ignore[invalid-argument-type]
        block_q=block_q,  # ty: ignore[invalid-argument-type]
        block_k=block_k,  # ty: ignore[invalid-argument-type]
        causal=causal,  # ty: ignore[invalid-argument-type]
        ranking_keys=True,  # ty: ignore[invalid-argument-type]
        num_warps=8,  # ty: ignore[unknown-argument]
        num_stages=3,  # ty: ignore[unknown-argument]
    )
    return keys


def _decode_score_keys(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    invalid = keys == torch.iinfo(torch.int64).min
    ordered = (keys >> 32) + 0x8000_0000
    bits = torch.where(
        (ordered & 0x8000_0000).bool(),
        ordered ^ 0x8000_0000,
        (~ordered) & 0xFFFF_FFFF,
    )
    scores = bits.to(torch.int32).view(torch.float32)
    ids = (0xFFFF_FFFF - (keys & 0xFFFF_FFFF)).to(torch.int32)
    return scores.masked_fill(invalid, float("-inf")), ids.masked_fill(invalid, -1)


def _merge_topk(
    scores: torch.Tensor,
    ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_scores = torch.cat((scores, candidate_scores), dim=1)
    all_ids = torch.cat((ids, candidate_ids), dim=1)
    keep = min(topk, int(all_scores.shape[1]))
    scores, positions = torch.topk(all_scores, keep, dim=1, sorted=False)
    return scores, torch.gather(all_ids, 1, positions)


def _stable_topk(
    scores: torch.Tensor,
    global_ids: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select score-descending/id-ascending top-k using an exact integer key."""
    if scores.dtype is not torch.float32 or global_ids.dtype is not torch.int32:
        raise TypeError("GLM-5.2 stable top-k expects fp32 scores and int32 ids.")
    keep = min(int(topk), int(scores.shape[1]))
    key_scores = torch.where(scores == 0, torch.zeros_like(scores), scores)
    bits = key_scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFF_FFFF
    ordered = torch.where(
        (bits >> 31).bool(),
        (~bits) & 0xFFFF_FFFF,
        bits ^ 0x8000_0000,
    )
    primary = ordered - 0x8000_0000
    secondary = torch.where(
        global_ids >= 0,
        0xFFFF_FFFF - global_ids.to(torch.int64),
        torch.zeros_like(primary),
    )
    positions = torch.topk((primary << 32) | secondary, keep, dim=1).indices
    return torch.gather(scores, 1, positions), torch.gather(global_ids, 1, positions)


def _merge_stable_topk(
    scores: torch.Tensor,
    global_ids: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _stable_topk(
        torch.cat((scores, candidate_scores), dim=1),
        torch.cat((global_ids, candidate_ids), dim=1),
        topk=topk,
    )


def _stage_topk_update(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    stage: Glm52StageState,
    best_scores: torch.Tensor,
    best_ids: torch.Tensor,
    *,
    topk: int,
) -> None:
    max_score_elements = _MAX_SCORE_WORKSPACE_BYTES // torch.int64.itemsize
    for slice_ in stage.slices:
        max_k_len = int(slice_.k_end - slice_.k_start)
        k_chunk_size = min(max_k_len, _MAX_K_CHUNK)
        q_chunk_size = max(1, max_score_elements // max(k_chunk_size, 1))
        for q_start in range(slice_.q_start, slice_.q_end, q_chunk_size):
            q_end = min(q_start + q_chunk_size, slice_.q_end)
            owner_rows = stage.owner_q_rows[q_start:q_end]
            scores = best_scores.index_select(0, owner_rows)
            ids = best_ids.index_select(0, owner_rows)
            for k_start in range(slice_.k_start, slice_.k_end, k_chunk_size):
                k_end = min(k_start + k_chunk_size, slice_.k_end)
                score_keys = _index_score_keys(
                    q[q_start:q_end].contiguous(),
                    k[k_start:k_end].contiguous(),
                    weights[q_start:q_end].contiguous(),
                    q_position_offset=slice_.q_position_offset
                    + q_start
                    - slice_.q_start,
                    k_position_offset=slice_.k_position_offset
                    + k_start
                    - slice_.k_start,
                    causal=slice_.causal,
                )
                keep = min(topk, k_end - k_start)
                candidate_keys = torch.topk(
                    score_keys, keep, dim=1, sorted=False
                ).values
                candidate_scores, candidate_ids = _decode_score_keys(candidate_keys)
                scores, ids = _merge_stable_topk(
                    scores,
                    ids,
                    candidate_scores,
                    candidate_ids,
                    topk=topk,
                )
            best_scores.index_copy_(0, owner_rows, scores)
            best_ids.index_copy_(0, owner_rows, ids)


@torch.no_grad()
def context_parallel_tree_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    state: Glm52PrefixTreeState,
    *,
    topk: int,
) -> Glm52RoutedTopk:
    """Accumulate exact GLM indexer top-k on query owners across ART stages."""
    cp_state = cast(ArtContextParallelState, state.context_parallel_state)
    valid_tokens = int(sum(cp_state.rank_plan.local_valid_lengths))
    q = q[:, :valid_tokens].reshape(valid_tokens, *q.shape[2:]).contiguous()
    k = k[:, :valid_tokens].reshape(valid_tokens, k.shape[-1]).contiguous()
    weights = weights[:, :valid_tokens].reshape(valid_tokens, weights.shape[-1])
    best_scores = torch.full(
        (valid_tokens, topk), float("-inf"), device=q.device, dtype=torch.float32
    )
    best_ids = torch.full((valid_tokens, topk), -1, device=q.device, dtype=torch.int32)
    works = launch_remote_stage_fetches(k, cp_state)
    for stage_plan, stage in zip(
        cp_state.rank_plan.stage_plans, state.stages, strict=True
    ):
        if not stage.slices:
            continue
        q_stage = stage_query_rows(q, stage_plan, cp_state)
        weights_stage = stage_query_rows(weights, stage_plan, cp_state)
        k_stage = stage_kv_rows(k, stage_plan, cp_state, works)
        _stage_topk_update(
            q_stage,
            k_stage,
            weights_stage,
            stage,
            best_scores,
            best_ids,
            topk=topk,
        )
        del q_stage, weights_stage, k_stage
    drain_stage_fetches(works)
    del best_scores
    _canonicalize_topk_(best_ids)
    route_map = state.route_by_global_id
    if route_map is None:
        raise RuntimeError("GLM-5.2 CP route map is missing.")
    indices = torch.where(
        best_ids >= 0,
        route_map[best_ids.clamp_min(0).to(torch.int64)],
        torch.full_like(best_ids, state.combined_k_rows),
    ).view(1, valid_tokens, topk)
    del best_ids
    return Glm52RoutedTopk(indices=indices)


@torch.compiler.disable
def streaming_tree_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    rows: tuple[Glm52IndexerRowPlan, ...],
    *,
    topk: int,
) -> torch.Tensor:
    """Exact tree-aware topk with bounded score workspace and no square logits."""
    if not q.is_cuda or q.device != k.device or q.device != weights.device:
        raise RuntimeError("GLM-5.2 indexer requires colocated CUDA tensors.")
    if q.ndim != 4 or k.ndim != 3 or weights.ndim != 3:
        raise ValueError("GLM-5.2 indexer expects q[B,S,H,D], k[B,S,D], w[B,S,H].")
    batch, seq_len, _, _ = q.shape
    if len(rows) != batch or k.shape[:2] != (batch, seq_len):
        raise ValueError("GLM-5.2 index plan does not match the packed tensor shape.")
    result = torch.full(
        (batch, seq_len, topk),
        -1,
        device=q.device,
        dtype=torch.int32,
    )
    max_score_elements = _MAX_SCORE_WORKSPACE_BYTES // torch.float32.itemsize
    for row in rows:
        for query in row.queries:
            max_k_len = max(slice_.k_end - slice_.k_start for slice_ in query.slices)
            k_chunk_size = min(max_k_len, _MAX_K_CHUNK)
            q_chunk_size = max(1, max_score_elements // max(k_chunk_size, 1))
            for q_start in range(query.q_start, query.q_end, q_chunk_size):
                q_end = min(q_start + q_chunk_size, query.q_end)
                q_chunk = q[row.row_index, q_start:q_end].contiguous()
                w_chunk = weights[row.row_index, q_start:q_end].contiguous()
                best_scores = torch.empty(
                    (q_end - q_start, 0), device=q.device, dtype=torch.float32
                )
                best_ids = torch.empty(
                    (q_end - q_start, 0), device=q.device, dtype=torch.int32
                )
                for slice_ in query.slices:
                    for k_start in range(slice_.k_start, slice_.k_end, k_chunk_size):
                        k_end = min(k_start + k_chunk_size, slice_.k_end)
                        score_chunk = _index_scores(
                            q_chunk,
                            k[row.row_index, k_start:k_end].contiguous(),
                            w_chunk,
                            q_position_offset=q_start,
                            k_position_offset=k_start,
                            causal=slice_.causal,
                        )
                        keep = min(topk, k_end - k_start)
                        candidate_scores, candidate_ids = torch.topk(
                            score_chunk,
                            keep,
                            dim=1,
                            sorted=False,
                        )
                        candidate_ids = (candidate_ids + k_start).to(torch.int32)
                        candidate_ids.masked_fill_(torch.isneginf(candidate_scores), -1)
                        best_scores, best_ids = _merge_topk(
                            best_scores,
                            best_ids,
                            candidate_scores,
                            candidate_ids,
                            topk=topk,
                        )
                result[row.row_index, q_start:q_end, : best_ids.shape[1]] = best_ids
    _canonicalize_topk_(result)
    result.masked_fill_(result < 0, seq_len)
    return result
