"""Frozen Flash k-pool selection, ported from slime and HF Glm5NextTextIndexer.

The budget counts tokens: 2048 means 512 complete pools plus at most three tail
tokens. No FP8/Hadamard quantization is used by the HF training oracle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

INDEX_BLOCK = 64


def index_width(topk: int, pool_size: int = 4) -> int:
    if topk <= 0 or pool_size <= 0 or topk % pool_size:
        raise ValueError("The positive token budget must be divisible by pool size.")
    return (topk + pool_size - 1 + INDEX_BLOCK - 1) // INDEX_BLOCK * INDEX_BLOCK


def compress_keys(
    keys: torch.Tensor,
    gates: torch.Tensor,
    ape: torch.Tensor,
    pool_starts: torch.Tensor,
) -> torch.Tensor:
    """HF's slot softmax, including its cast before multiplication/reduction."""
    slots = torch.arange(ape.shape[0], device=keys.device)
    tokens = pool_starts[:, None] + slots
    grouped_keys = keys[tokens]
    logits = gates[tokens].float() + ape.float()[None]
    probabilities = logits.softmax(dim=1).to(grouped_keys.dtype)
    return (probabilities * grouped_keys).sum(dim=1)


def _pool_scores(
    q: torch.Tensor, pools: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    # Reuse the bounded GLM-5.2 score kernel on GPU. Its weights include both
    # positive scales; for Flash's H=32,D=128 their product is exactly 1/64.
    scale = q.shape[-1] ** -0.5
    if q.is_cuda and q.dtype == pools.dtype == torch.bfloat16:
        from art.megatron.glm52.indexer import _index_scores

        return _index_scores(
            q.contiguous(),
            pools.contiguous(),
            (weights * scale).contiguous(),
            q_position_offset=0,
            k_position_offset=0,
            causal=False,
        )
    scores = torch.matmul(q.float(), pools.T.float()) * scale
    return torch.matmul(weights[:, None, :], scores.relu()).squeeze(1)


@torch.no_grad()
def select_pools(
    q: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    gates: torch.Tensor,
    ape: torch.Tensor,
    *,
    pool_starts: torch.Tensor,
    query_ids: torch.Tensor,
    tail_origin: int,
    topk: int = 2048,
    query_chunk_size: int = 128,
) -> torch.Tensor:
    """Select from one query segment's complete ancestor and own pools.

    All ids address the same global packed row. Candidate pools never span a
    segment boundary. Exact ties use torch.topk's ordering, as HF does; selected
    sets can differ on tied boundary scores between devices/backends.
    """
    pool_size = int(ape.shape[0])
    width = index_width(topk, pool_size)
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive.")
    if q.ndim != 3 or weights.shape != q.shape[:2] or query_ids.shape != q.shape[:1]:
        raise ValueError("Expected q[Q,H,D], weights[Q,H], and query_ids[Q].")
    if keys.ndim != 2 or gates.shape != keys.shape or ape.shape[1] != keys.shape[1]:
        raise ValueError("Expected matching keys/gates[K,D] and ape[pool_size,D].")
    result = torch.full((q.shape[0], width), -1, device=q.device, dtype=torch.int32)
    pools = compress_keys(keys, gates, ape, pool_starts)
    count = min(topk // pool_size, int(pool_starts.numel()))
    slots = torch.arange(pool_size, device=q.device)
    # Score storage is [chunk,pools], rather than [heads,sequence,sequence].
    for first in range(0, q.shape[0], query_chunk_size):
        last = min(first + query_chunk_size, q.shape[0])
        ids = query_ids[first:last]
        if count:
            scores = _pool_scores(q[first:last], pools, weights[first:last])
            visible = pool_starts[None] + pool_size - 1 <= ids[:, None]
            scores.masked_fill_(~visible, torch.finfo(scores.dtype).min)
            selected = scores.topk(count, dim=-1).indices
            tokens = pool_starts[selected, None] + slots
            tokens.masked_fill_(~visible.gather(1, selected)[..., None], -1)
            result[first:last, : count * pool_size] = tokens.flatten(1).int()
        tail_count = (ids - tail_origin + 1).remainder(pool_size)
        tail_start = ids + 1 - tail_count
        tail = tail_start[:, None] + slots[:-1]
        tail.masked_fill_(slots[:-1] >= tail_count[:, None], -1)
        result[first:last, topk : topk + pool_size - 1] = tail.int()
    return result


@torch.no_grad()
def pooled_indexer_topk(
    q: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    gates: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: Sequence[int],
    *,
    topk: int = 2048,
    query_start: int = 0,
    query_chunk_size: int = 128,
) -> torch.Tensor:
    """Packed independent documents; q may be a contiguous CP query interval.

    Inputs are q[Q,H,D], keys/gates[K,D], weights[Q,H]. Weights already contain
    H**-0.5. Returns int32 [1,Q,ceil((topk+3)/64)*64], with -1 sentinels.
    """
    boundaries = tuple(int(value) for value in cu_seqlens)
    if (
        not boundaries
        or boundaries[0] != 0
        or boundaries[-1] != keys.shape[0]
        or any(b <= a for a, b in zip(boundaries[:-1], boundaries[1:], strict=True))
        or query_start < 0
        or query_start + q.shape[0] > keys.shape[0]
    ):
        raise ValueError("Invalid packed document boundaries or query interval.")
    pool_size = int(ape.shape[0])
    result = torch.full(
        (1, q.shape[0], index_width(topk, pool_size)),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        first, last = max(start - query_start, 0), min(end - query_start, q.shape[0])
        if first >= last:
            continue
        starts = (
            torch.arange((end - start) // pool_size, device=q.device) * pool_size
            + start
        )
        result[0, first:last] = select_pools(
            q[first:last],
            keys,
            weights[first:last],
            gates,
            ape,
            pool_starts=starts,
            query_ids=torch.arange(
                first + query_start, last + query_start, device=q.device
            ),
            tail_origin=start,
            topk=topk,
            query_chunk_size=query_chunk_size,
        )
    return result


def _tensor(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


class KPoolIndexer(nn.Module):
    """Checkpoint-shaped projections; sequence-major interface matches GLM-5.2.

    MCore injects duplicated TE linears and a LayerNorm builder. Plain PyTorch
    builders also support the CPU HF oracle without importing Megatron/CUDA.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 4096,
        q_lora_rank: int = 1536,
        num_heads: int = 32,
        head_dim: int = 128,
        topk: int = 2048,
        pool_size: int = 4,
        linear_factory: Callable[[int, int], nn.Module] | None = None,
        norm_factory: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        index_width(topk, pool_size)
        self.heads, self.head_dim = num_heads, head_dim
        self.topk, self.pool_size = topk, pool_size
        linear = linear_factory or (lambda n, m: nn.Linear(n, m, bias=False))
        norm = norm_factory or (lambda n: nn.LayerNorm(n, eps=1e-6))
        self.linear_wq_b = linear(q_lora_rank, num_heads * head_dim)
        self.linear_wk = linear(hidden_size, head_dim)
        self.k_norm = norm(head_dim)
        self.linear_weights_proj = linear(hidden_size, num_heads)
        weight = cast(torch.Tensor, self.linear_wk.weight)
        self.index_kpool_compress_gate = nn.Parameter(
            weight.new_zeros(head_dim, hidden_size)
        )
        self.index_kpool_compress_ape = nn.Parameter(
            weight.new_zeros(pool_size, head_dim)
        )
        self.requires_grad_(False)

    @torch.no_grad()
    def project(
        self, hidden_states: torch.Tensor, q_residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = _tensor(self.linear_wq_b(q_residual)).unflatten(
            -1, (self.heads, self.head_dim)
        )
        keys = _tensor(self.k_norm(_tensor(self.linear_wk(hidden_states)))).to(q.dtype)
        weights = (
            _tensor(self.linear_weights_proj(hidden_states)).float() * self.heads**-0.5
        )
        gates = F.linear(hidden_states, self.index_kpool_compress_gate)
        return (
            q.transpose(0, 1).contiguous(),
            keys.transpose(0, 1).contiguous(),
            weights.transpose(0, 1).contiguous(),
            gates.transpose(0, 1).contiguous(),
        )

    @torch.no_grad()
    def forward(
        self, hidden_states: torch.Tensor, q_residual: torch.Tensor, state: Any
    ) -> Any:
        from art.megatron.glm5next.dsa_cp import tree_pooled_indexer_topk

        return tree_pooled_indexer_topk(
            self, self.project(hidden_states, q_residual), state
        )
