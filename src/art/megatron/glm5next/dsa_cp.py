"""K-pool metadata and indexer routing over GLM-5.2's existing CP stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
from art.megatron.context_parallel.types import ArtContextParallelState, AttnMaskKind
from art.megatron.glm5next.kpool_indexer import KPoolIndexer, index_width, select_pools
from art.megatron.glm52.cp_stage import (
    drain_stage_fetches,
    launch_remote_stage_fetches,
    stage_kv_rows,
)
from art.megatron.glm52.indexer import Glm52RoutedTopk
from art.megatron.glm52.state import (
    Glm52PrefixTreeState,
    build_glm52_context_parallel_state,
    build_glm52_prefix_tree_state,
)


def dsa_pack_alignment(cp_size: int) -> int:
    if cp_size < 1:
        raise ValueError("CP size must be positive.")
    return 4 * 2 * cp_size


@dataclass(frozen=True)
class KPoolQueryPlan:
    batch: int
    owner_rows: torch.Tensor
    query_ids: torch.Tensor
    pool_starts: torch.Tensor
    tail_origin: int


class Glm5NextDsaState(Glm52PrefixTreeState):
    pool_plans: tuple[KPoolQueryPlan, ...] = ()
    combined_global_ids: torch.Tensor | None = None


def build_glm5next_dsa_state(
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    device: torch.device,
    context_parallel_state: ArtContextParallelState | None = None,
) -> Glm5NextDsaState:
    """Build once in M4's handler, stored at attention_bias.model_state['glm5next_dsa'].

    CP uses its original global tree metadata and exact stage token ids. Segment
    starts and ancestor ends must align to four; pad packing to 8*CP. Padding is
    identified by tree metadata, never by the model's pad/EOS token id.
    """
    cp = context_parallel_state
    combined_ids = None
    global_to_local = None
    if cp is None:
        base = build_glm52_prefix_tree_state(
            position_ids=position_ids,
            group_ids=group_ids,
            parent_ids=parent_ids,
            device=device,
        )
    else:
        base = build_glm52_context_parallel_state(
            position_ids=position_ids, context_parallel_state=cp, device=device
        )
        group_ids, parent_ids = cp.group_ids[None], cp.parent_ids[None]
        global_to_local = torch.full(
            (cp.rank_plan.original_seq_len,), -1, dtype=torch.long
        )
        k_ids = []
        for plan in cp.rank_plan.stage_plans:
            metadata = plan.mask_metadata
            if metadata is None:
                continue
            nq = sum(r.size() for r in plan.owner_local_q_ranges)
            nk = sum(r.size() for r in plan.owner_local_k_ranges)
            rows = (
                torch.cat(
                    [torch.arange(r.start, r.end) for r in plan.owner_local_q_ranges]
                )
                if plan.owner_local_q_ranges
                else torch.empty(0, dtype=torch.long)
            )
            qids = metadata.q_token_indices[:nq].cpu().long()
            global_to_local[qids] = rows
            ids = metadata.k_token_indices[:nk].cpu().long()
            # Individual stages may contain several separated aligned ranges.
            for ranges in (plan.global_q_ranges, plan.global_k_ranges):
                if any(r.start % 4 or r.end % 4 for r in ranges):
                    raise ValueError(
                        "Flash CP stage boundaries must align to four-token pools."
                    )
            k_ids.append(ids)
        combined_ids = torch.cat(k_ids) if k_ids else torch.empty(0, dtype=torch.long)
        if combined_ids.unique().numel() != combined_ids.numel():
            raise ValueError(
                "Flash CP stages must assign each KV token exactly one route."
            )

    spec = build_prefix_tree_attention_spec(
        group_ids=group_ids.cpu(), parent_ids=parent_ids.cpu()
    )
    pool_plans = []
    available = None
    if cp is not None:
        available = torch.zeros(cp.rank_plan.original_seq_len, dtype=torch.bool)
        available[combined_ids] = True
    for row in spec.rows:
        by_query: dict[tuple[int, int], list[Any]] = {}
        for slice_ in row.slices:
            by_query.setdefault((slice_.q_range.start, slice_.q_range.end), []).append(
                slice_
            )
        for (first, last), slices in sorted(by_query.items()):
            query_ids = torch.arange(first, last)
            owners = (
                query_ids if global_to_local is None else global_to_local[query_ids]
            )
            query_ids, owners = query_ids[owners >= 0], owners[owners >= 0]
            if not query_ids.numel():
                continue
            starts = []
            tail_origin = None
            for slice_ in slices:
                start, end = slice_.k_range.start, slice_.k_range.end
                if start % 4 or (slice_.mask_kind is AttnMaskKind.FULL and end % 4):
                    raise ValueError(
                        "Flash tree segments/ancestors must align to four-token pools."
                    )
                pools = torch.arange((end - start) // 4) * 4 + start
                if available is not None and pools.numel():
                    present = available[pools[:, None] + torch.arange(4)].all(1)
                    if bool((~present & (pools + 3 <= query_ids.max())).any()):
                        raise ValueError(
                            "CP stages omit part of a visible four-token pool."
                        )
                    # Keep masked future slots: HF top-k tie ordering depends
                    # on candidate width. Missing future keys are zero-filled
                    # and remain invisible to every local query.
                starts.append(pools)
                if slice_.mask_kind is AttnMaskKind.CAUSAL:
                    tail_origin = start
            if tail_origin is None:
                raise ValueError(
                    "Flash query segment is missing its causal tail origin."
                )
            pool_plans.append(
                KPoolQueryPlan(
                    batch=row.row_index,
                    owner_rows=owners.to(device),
                    query_ids=query_ids.to(device),
                    pool_starts=torch.cat(starts).sort().values.to(device),
                    tail_origin=tail_origin,
                )
            )
    return Glm5NextDsaState(
        **{name: getattr(base, name) for name in Glm52PrefixTreeState.model_fields},
        pool_plans=tuple(pool_plans),
        combined_global_ids=None if combined_ids is None else combined_ids.to(device),
    )


@torch.no_grad()
def tree_pooled_indexer_topk(
    indexer: KPoolIndexer,
    projected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    state: Glm5NextDsaState,
) -> torch.Tensor | Glm52RoutedTopk:
    q, keys, weights, gates = projected
    if not isinstance(state, Glm5NextDsaState):
        raise TypeError("Flash's indexer requires Glm5NextDsaState.")
    if state.position_ids.shape != q.shape[:2]:
        raise ValueError("Flash DSA state/token shapes differ.")
    cp = state.context_parallel_state
    if cp is not None:
        if q.shape[0] != 1:
            raise ValueError("Flash CP supports one packed row.")
        valid = sum(cp.rank_plan.local_valid_lengths)
        local = torch.cat((keys[0, :valid], gates[0, :valid]), -1).contiguous()
        works = launch_remote_stage_fetches(local, cp)
        try:
            parts = [
                stage_kv_rows(local, stage, cp, works)
                for stage in cp.rank_plan.stage_plans
            ]
        finally:
            drain_stage_fetches(works)
        global_k_gate = local.new_zeros(
            (cp.rank_plan.original_seq_len, local.shape[-1])
        )
        if state.combined_global_ids is None:
            raise RuntimeError("Flash CP global route ids are missing.")
        global_k_gate.index_copy_(0, state.combined_global_ids, torch.cat(parts))
        keys, gates = (part[None] for part in global_k_gate.chunk(2, -1))
    result = torch.full(
        (*q.shape[:2], index_width(indexer.topk, indexer.pool_size)),
        -1,
        dtype=torch.int32,
        device=q.device,
    )
    for plan in state.pool_plans:
        selected = select_pools(
            q[plan.batch, plan.owner_rows],
            keys[plan.batch],
            weights[plan.batch, plan.owner_rows],
            gates[plan.batch],
            indexer.index_kpool_compress_ape,
            pool_starts=plan.pool_starts,
            query_ids=plan.query_ids,
            tail_origin=plan.tail_origin,
            topk=indexer.topk,
        )
        result[plan.batch, plan.owner_rows] = selected
    if cp is None:
        return result.masked_fill(result < 0, q.shape[1])
    route = state.route_by_global_id
    if route is None:
        raise RuntimeError("Flash CP routes are missing.")
    result = torch.where(
        result >= 0, route[result.clamp_min(0).long()], state.combined_k_rows
    )
    if bool((result < 0).any()):
        raise RuntimeError(
            "Flash selection includes a token absent from its CP stages."
        )
    return Glm52RoutedTopk(indices=result)
