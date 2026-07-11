from __future__ import annotations

import torch
import triton
import triton.language as tl

from art.megatron.glm52.state import Glm52IndexerRowPlan

_MAX_SCORE_WORKSPACE_BYTES = 256 * 1024 * 1024
_MAX_K_CHUNK = 32 * 1024
_ROUTE_STAGE_BITS = 3


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
    tl.store(
        scores_ptr + q_offsets[:, None] * stride_st + k_offsets[None, :] * stride_sk,
        scores,
        mask=(q_offsets[:, None] < q_len) & (k_offsets[None, :] < k_len),
    )


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
        num_warps=8,  # ty: ignore[unknown-argument]
        num_stages=3,  # ty: ignore[unknown-argument]
    )
    return scores


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


@triton.jit
def _group_stage_routes_kernel(
    routes_ptr,
    grouped_ptr,
    offsets_ptr,
    topk: tl.constexpr,
    stage_count: tl.constexpr,
    stage_bits: tl.constexpr,
    stage_mask: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = row * topk
    offset_base = row * (stage_count + 1)
    cursor = 0
    tl.store(offsets_ptr + offset_base, 0)
    for stage in range(stage_count):
        for block_start in range(0, topk, block):
            columns = block_start + tl.arange(0, block)
            in_bounds = columns < topk
            routes = tl.load(routes_ptr + row_base + columns, mask=in_bounds, other=-1)
            matches = in_bounds & (routes >= 0) & ((routes & stage_mask) == stage)
            compact = tl.cumsum(matches.to(tl.int32), axis=0) - 1
            tl.store(
                grouped_ptr + row_base + cursor + compact,
                routes >> stage_bits,
                mask=matches,
            )
            cursor += tl.sum(matches.to(tl.int32), axis=0)
        tl.store(offsets_ptr + offset_base + stage + 1, cursor)


def group_stage_routes(
    routes: torch.Tensor,
    *,
    stage_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group packed `(stage, local_row)` routes without a global-id remap."""
    if routes.dtype is not torch.int32 or not routes.is_cuda or routes.ndim != 3:
        raise TypeError("GLM-5.2 routes must be CUDA int32 [B,S,topk].")
    if not 1 <= stage_count <= 1 << _ROUTE_STAGE_BITS:
        raise ValueError(f"GLM-5.2 route stage count is invalid: {stage_count}.")
    topk = int(routes.shape[-1])
    if topk % 32:
        raise ValueError("GLM-5.2 route topk must be divisible by 32.")
    routes = routes.contiguous()
    grouped = torch.empty_like(routes)
    offsets = torch.empty(
        (*routes.shape[:2], stage_count + 1),
        device=routes.device,
        dtype=torch.int32,
    )
    _group_stage_routes_kernel[(routes.numel() // topk,)](
        routes,
        grouped,
        offsets,
        topk=topk,  # ty: ignore[invalid-argument-type]
        stage_count=stage_count,  # ty: ignore[invalid-argument-type]
        stage_bits=_ROUTE_STAGE_BITS,  # ty: ignore[invalid-argument-type]
        stage_mask=(1 << _ROUTE_STAGE_BITS) - 1,  # ty: ignore[invalid-argument-type]
        block=256,  # ty: ignore[invalid-argument-type]
        num_warps=8,  # ty: ignore[unknown-argument]
    )
    return grouped, offsets


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
    return result
