from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
import hashlib
import json
from threading import Lock
from typing import Any, cast

from pydantic import BaseModel
import torch

from art.loss import shift_tensor
from art.megatron.selective_lm_head import LmHeadTokenSelection
from art.preprocessing.pack import PackedTensors

from .builder import build_prefix_tree_attention_spec
from .layout_index import TokenLayoutIndex
from .types import (
    ArtContextParallelState,
    AttnMaskKind,
    AttnSlice,
    ContextParallelConfig,
    ContextParallelWorkloadProfile,
    CpBlockMaskVariant,
    DispatchedPackedTensors,
    DkvReducePlan,
    ExactMaskMetadata,
    KvFetchPlan,
    PackedBatchAttentionSpec,
    PackedRowAttentionSpec,
    ParallelTopology,
    PreparedMegatronBatch,
    RankRuntimePlan,
    StagePlan,
    TokenRange,
    TrainingMicrobatchWorkload,
)

_CHUNK_MASK_STATS_TORCH_THRESHOLD = 1024
_CP4_SEARCH_PROBE_CANDIDATE_LIMIT = 2
_CP4_SEARCH_PROBE_IMPROVEMENT_MS = 1.0
_PLAN_CACHE_MAX_ENTRIES = 128

StagePiece = tuple[TokenRange, TokenRange, AttnMaskKind, int | None]
StageSliceKey = tuple[int, int, int, int, int, str, int]
ProfiledChunkPiece = tuple[int, int, int, int, int, int, str, int | None]


@dataclass(frozen=True)
class _PlanningBundle:
    spec: PackedBatchAttentionSpec
    row_spec: PackedRowAttentionSpec
    chunk_ranges: tuple[TokenRange, ...]
    owners: tuple[int, ...]
    wave_assignment: tuple[int, ...]
    token_layout_index: TokenLayoutIndex
    config: ContextParallelConfig
    gdn_execution_spec: Any | None = None


_PLANNING_BUNDLE_CACHE: dict[str, _PlanningBundle] = {}
_RUNTIME_PLAN_CACHE: dict[str, tuple[RankRuntimePlan, ...]] = {}
_RANK_RUNTIME_PLAN_CACHE: dict[tuple[str, int], RankRuntimePlan] = {}
_GDN_GLOBAL_DECISION_CACHE: dict[tuple[str, str], Any] = {}
_GDN_RANK_PLAN_CACHE: dict[tuple[str, str, int | None, int, str], Any] = {}
_PLAN_CACHE_LOCK = Lock()


def _json_cache_key(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _cache_put(cache: dict[Any, Any], key: Any, value: Any) -> None:
    with _PLAN_CACHE_LOCK:
        if key not in cache and len(cache) >= _PLAN_CACHE_MAX_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = value


def _metadata_tensor_digest(tensor: torch.Tensor) -> str:
    """Digest planning metadata without touching CUDA in the normal path.

    CP lookahead depends on this function receiving CPU metadata. CUDA input is
    still accepted for compatibility, but the device-to-host copy below will
    synchronize and break host-ahead overlap with the previous microbatch's GPU
    work.
    """
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha1()
    digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
    digest.update(str(cpu_tensor.dtype).encode("utf-8"))
    digest.update(cpu_tensor.numpy().tobytes())
    return digest.hexdigest()


def _planning_metadata_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Return metadata on CPU for the no-sync CP planning boundary.

    Production lookahead callers should pass CPU tensors, making this a cheap
    detach/contiguous operation. CUDA input is a compatibility fallback and
    necessarily synchronizes.
    """
    return tensor.detach().to(device="cpu").contiguous()


def _planning_bundle_cache_key(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    topology: ParallelTopology,
    config: ContextParallelConfig,
    original_seq_len: int,
    build_gdn_execution_spec: bool,
) -> str:
    return _json_cache_key(
        {
            "group_ids": _metadata_tensor_digest(group_ids),
            "parent_ids": _metadata_tensor_digest(parent_ids),
            "topology": _dataclass_payload(topology),
            "config": _dataclass_payload(config),
            "original_seq_len": int(original_seq_len),
            "build_gdn_execution_spec": bool(build_gdn_execution_spec),
        }
    )


def _get_or_build_planning_bundle(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    topology: ParallelTopology,
    config: ContextParallelConfig,
    original_seq_len: int,
    build_gdn_execution_spec: bool,
) -> tuple[str, _PlanningBundle, torch.Tensor, torch.Tensor]:
    group_ids_cpu = _planning_metadata_cpu(group_ids)
    parent_ids_cpu = _planning_metadata_cpu(parent_ids)
    planning_key = _planning_bundle_cache_key(
        group_ids=group_ids_cpu,
        parent_ids=parent_ids_cpu,
        topology=topology,
        config=config,
        original_seq_len=original_seq_len,
        build_gdn_execution_spec=build_gdn_execution_spec,
    )
    bundle = _PLANNING_BUNDLE_CACHE.get(planning_key)
    if bundle is not None:
        return planning_key, bundle, group_ids_cpu, parent_ids_cpu

    spec = build_prefix_tree_attention_spec(
        group_ids=group_ids_cpu,
        parent_ids=parent_ids_cpu,
    )
    gdn_execution_spec = None
    if build_gdn_execution_spec:
        from art.megatron.gdn.gdn_prefix_tree import parse_gdn_prefix_tree_segments

        gdn_execution_spec = parse_gdn_prefix_tree_segments(
            group_ids_cpu,
            parent_ids_cpu,
        )
    row_spec, chunk_ranges, owners, wave_assignment = _runtime_plan_assignment(
        spec,
        topology=topology,
        config=config,
    )
    bundle = _PlanningBundle(
        spec=spec,
        row_spec=row_spec,
        chunk_ranges=chunk_ranges,
        owners=owners,
        wave_assignment=wave_assignment,
        token_layout_index=_build_runtime_token_layout_index(
            chunk_ranges=chunk_ranges,
            owners=owners,
            cp_size=max(int(topology.cp), 1),
        ),
        config=config,
        gdn_execution_spec=gdn_execution_spec,
    )
    _cache_put(_PLANNING_BUNDLE_CACHE, planning_key, bundle)
    return planning_key, bundle, group_ids_cpu, parent_ids_cpu


def _get_or_build_bundle_rank_plan(
    *,
    planning_key: str,
    bundle: _PlanningBundle,
    original_seq_len: int,
    target_rank: int,
    block_size: int,
) -> RankRuntimePlan:
    """Materialize only the caller's rank plan at the host-ahead boundary.

    Building every rank plan here scales CPU work with CP size, can exceed the
    planning budget, and exposes planning after lookahead GPU work completes.
    Each rank plan depends only on the shared assignment, so peer plans are not
    part of this runtime boundary.
    """
    cache_key = (planning_key, int(target_rank))
    cached = _RANK_RUNTIME_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    plan = _build_rank_runtime_plan(
        row_spec=bundle.row_spec,
        chunk_ranges=bundle.chunk_ranges,
        owners=bundle.owners,
        wave_assignment=bundle.wave_assignment,
        token_layout_index=bundle.token_layout_index,
        cp_size=len(bundle.token_layout_index.token_counts_by_rank),
        original_seq_len=original_seq_len,
        target_rank=target_rank,
        block_size=block_size,
        config=bundle.config,
    )
    _cache_put(_RANK_RUNTIME_PLAN_CACHE, cache_key, plan)
    return plan


def _gdn_planner_config_cache_key(gdn_planner_config: Any | None) -> str:
    return (
        _json_cache_key(_dataclass_payload(gdn_planner_config))
        if gdn_planner_config is not None
        else ""
    )


def _gdn_rank_plan_cache_key(
    *,
    planning_key: str,
    cp_rank: int,
    gdn_planner_config: Any | None,
    device: torch.device,
) -> tuple[str, str, int | None, int, str]:
    return (
        planning_key,
        device.type,
        device.index,
        int(cp_rank),
        _gdn_planner_config_cache_key(gdn_planner_config),
    )


def _plan_gdn_global_execution(
    *,
    planning_key: str,
    bundle: _PlanningBundle,
    topology: ParallelTopology,
    gdn_planner_config: Any | None,
) -> Any:
    """Select one all-rank GDN decision without rank-local tensors."""
    if bundle.gdn_execution_spec is None:
        raise RuntimeError("GDN CP planning requires a parsed execution spec")
    cache_key = (
        planning_key,
        _gdn_planner_config_cache_key(gdn_planner_config),
    )
    cached = _GDN_GLOBAL_DECISION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from art.megatron.gdn.gdn_prefix_tree import (
        build_gdn_global_execution_decision,
    )

    decision = build_gdn_global_execution_decision(
        bundle.gdn_execution_spec,
        cp_size=int(topology.cp),
        attention_token_layout_index=bundle.token_layout_index,
        planner_config=gdn_planner_config,
    )
    _cache_put(_GDN_GLOBAL_DECISION_CACHE, cache_key, decision)
    return decision


def _plan_gdn_rank_execution(
    *,
    planning_key: str,
    bundle: _PlanningBundle,
    topology: ParallelTopology,
    cp_rank: int,
    gdn_planner_config: Any | None,
) -> Any:
    """Plan one GDN rank on CPU at the explicit CP planning boundary."""
    if bundle.gdn_execution_spec is None:
        raise RuntimeError("GDN CP planning requires a parsed execution spec")
    cache_key = _gdn_rank_plan_cache_key(
        planning_key=planning_key,
        cp_rank=cp_rank,
        gdn_planner_config=gdn_planner_config,
        device=torch.device("cpu"),
    )
    cached = _GDN_RANK_PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    decision = _plan_gdn_global_execution(
        planning_key=planning_key,
        bundle=bundle,
        topology=topology,
        gdn_planner_config=gdn_planner_config,
    )
    from art.megatron.gdn.gdn_prefix_tree import materialize_gdn_rank_execution_plan

    plan = materialize_gdn_rank_execution_plan(
        bundle.gdn_execution_spec,
        decision,
        device="cpu",
        cp_rank=int(cp_rank),
        planner_config=gdn_planner_config,
    )
    _cache_put(_GDN_RANK_PLAN_CACHE, cache_key, plan)
    return plan


def _materialize_preplanned_gdn_rank_execution(
    *,
    planning_key: str,
    cp_rank: int,
    gdn_planner_config: Any | None,
    device: torch.device,
) -> Any:
    """Materialize a CPU-planned GDN rank without permitting late planning."""
    cpu_key = _gdn_rank_plan_cache_key(
        planning_key=planning_key,
        cp_rank=cp_rank,
        gdn_planner_config=gdn_planner_config,
        device=torch.device("cpu"),
    )
    cpu_plan = _GDN_RANK_PLAN_CACHE.get(cpu_key)
    if cpu_plan is None:
        raise RuntimeError(
            "GDN execution plan was not built at the CPU planning boundary"
        )
    if device.type == "cpu":
        return cpu_plan

    device_key = _gdn_rank_plan_cache_key(
        planning_key=planning_key,
        cp_rank=cp_rank,
        gdn_planner_config=gdn_planner_config,
        device=device,
    )
    device_plan = _GDN_RANK_PLAN_CACHE.get(device_key)
    if device_plan is not None:
        return device_plan

    from art.megatron.gdn.gdn_prefix_tree import (
        move_gdn_rank_execution_plan_to_device,
    )

    device_plan = move_gdn_rank_execution_plan_to_device(cpu_plan, device)
    _cache_put(_GDN_RANK_PLAN_CACHE, device_key, device_plan)
    return device_plan


def context_parallel_rank_model_token_counts(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    topology: ParallelTopology,
    config: ContextParallelConfig,
    original_seq_len: int,
    build_gdn_execution_spec: bool,
    gdn_planner_config: Any | None = None,
) -> tuple[int, ...]:
    """Return each CP rank's maximum model rows across physical layouts."""
    planning_key, bundle, _group_ids_cpu, _parent_ids_cpu = (
        _get_or_build_planning_bundle(
            group_ids=group_ids,
            parent_ids=parent_ids,
            topology=topology,
            config=config,
            original_seq_len=original_seq_len,
            build_gdn_execution_spec=build_gdn_execution_spec,
        )
    )
    attention_counts = bundle.token_layout_index.token_counts_by_rank
    if not build_gdn_execution_spec:
        return attention_counts
    decision = _plan_gdn_global_execution(
        planning_key=planning_key,
        bundle=bundle,
        topology=topology,
        gdn_planner_config=gdn_planner_config,
    )
    gdn_counts = decision.gdn_token_counts_by_rank
    return tuple(
        max(attention_count, gdn_count)
        for attention_count, gdn_count in zip(attention_counts, gdn_counts, strict=True)
    )


def _normalized_chunk_size(
    *,
    valid_tokens: int,
    block_size: int,
    requested_chunk_size: int,
    cp_size: int | None = None,
    config: ContextParallelConfig | None = None,
) -> int:
    chunk_size = max(int(block_size), int(requested_chunk_size))
    if chunk_size % int(block_size) != 0:
        chunk_size = ((chunk_size + int(block_size) - 1) // int(block_size)) * int(
            block_size
        )
    chunk_size = max(1, min(chunk_size, max(valid_tokens, 1)))
    if cp_size is None or config is None:
        return chunk_size

    chunk_budget_base = max(int(config.planner_chunk_budget_base), 0)
    chunk_budget_per_cp_rank = max(int(config.planner_chunk_budget_per_cp_rank), 0)
    if chunk_budget_base <= 0 and chunk_budget_per_cp_rank <= 0:
        return chunk_size

    chunk_budget = max(
        int(cp_size),
        chunk_budget_base + chunk_budget_per_cp_rank * max(int(cp_size), 1),
    )
    if chunk_budget <= 0:
        return chunk_size

    requested_chunk_count = max(
        1,
        (max(int(valid_tokens), 1) + int(chunk_size) - 1) // int(chunk_size),
    )
    if requested_chunk_count <= chunk_budget:
        return chunk_size

    chunk_size = max(
        int(chunk_size),
        (max(int(valid_tokens), 1) + int(chunk_budget) - 1) // int(chunk_budget),
    )
    if chunk_size % int(block_size) != 0:
        chunk_size = ((chunk_size + int(block_size) - 1) // int(block_size)) * int(
            block_size
        )
    return max(1, min(chunk_size, max(valid_tokens, 1)))


def _search_config_for_chunk_count(
    *,
    config: ContextParallelConfig,
    chunk_count: int,
) -> ContextParallelConfig:
    if int(chunk_count) >= 128:
        updates = {
            "planner_max_search_steps": min(int(config.planner_max_search_steps), 2),
            "planner_candidate_chunk_limit": min(
                int(config.planner_candidate_chunk_limit), 4
            ),
            "planner_max_remote_waves": min(int(config.planner_max_remote_waves), 2),
        }
    elif int(chunk_count) >= 64:
        updates = {
            "planner_max_search_steps": min(int(config.planner_max_search_steps), 4),
            "planner_candidate_chunk_limit": min(
                int(config.planner_candidate_chunk_limit), 6
            ),
            "planner_max_remote_waves": min(int(config.planner_max_remote_waves), 3),
        }
    else:
        return config
    if all(int(getattr(config, key)) == int(value) for key, value in updates.items()):
        return config
    return replace(config, **updates)


def _best_improving_move(
    *,
    current_owners: tuple[int, ...],
    current_eval: dict[str, Any],
    wave_assignment: tuple[int, ...],
    cp_size: int,
    q_weights: list[float],
    candidate_limit: int,
    evaluate_candidate: Any,
) -> tuple[tuple[int, ...], dict[str, Any]] | None:
    slow_rank = int(
        max(
            range(cp_size),
            key=lambda rank: cast(tuple[float, ...], current_eval["rank_scores"])[rank],
        )
    )
    candidate_chunks = _candidate_chunk_indices(
        owners=current_owners,
        target_rank=slow_rank,
        q_weights=q_weights,
        limit=int(candidate_limit),
    )
    if not candidate_chunks:
        return None

    best_move: tuple[tuple[int, ...], dict[str, Any]] | None = None
    for chunk_index in candidate_chunks:
        for dst_rank in range(cp_size):
            if dst_rank == slow_rank:
                continue
            candidate = list(current_owners)
            candidate[chunk_index] = dst_rank
            candidate_owners = tuple(candidate)
            if (
                len(candidate_owners) >= cp_size
                and len(set(candidate_owners)) != cp_size
            ):
                continue
            candidate_eval = evaluate_candidate(
                owners=candidate_owners,
                wave_assignment=wave_assignment,
            )
            if float(candidate_eval["score"]) + 1e-9 >= float(current_eval["score"]):
                continue
            if best_move is None or float(candidate_eval["score"]) + 1e-9 < float(
                best_move[1]["score"]
            ):
                best_move = (candidate_owners, candidate_eval)
    return best_move


def _build_chunk_ranges(
    *,
    valid_tokens: int,
    chunk_size: int,
) -> tuple[TokenRange, ...]:
    return tuple(
        TokenRange(start=start, end=min(start + chunk_size, valid_tokens))
        for start in range(0, valid_tokens, chunk_size)
    )


def _indexed_intersections(
    base_range: TokenRange,
    candidate_ranges: tuple[TokenRange, ...],
    *,
    candidate_starts: tuple[int, ...] | None = None,
    candidate_ends: tuple[int, ...] | None = None,
) -> list[tuple[int, TokenRange]]:
    if not candidate_ranges:
        return []
    base_start = int(base_range.start)
    base_end = int(base_range.end)
    if candidate_starts is None:
        candidate_starts = tuple(int(candidate.start) for candidate in candidate_ranges)
    if candidate_ends is None:
        candidate_ends = tuple(int(candidate.end) for candidate in candidate_ranges)
    first_index = bisect_right(candidate_ends, base_start)
    last_index = bisect_left(candidate_starts, base_end, lo=first_index)
    intersections: list[tuple[int, TokenRange]] = []
    for index in range(first_index, last_index):
        candidate = candidate_ranges[index]
        start = max(base_start, int(candidate.start))
        end = min(base_end, int(candidate.end))
        if end > start:
            intersections.append((index, TokenRange(start=start, end=end)))
    return intersections


def _causal_piece_pair_count_from_bounds(
    *,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
) -> int:
    if q_end <= q_start or k_end <= k_start:
        return 0

    k_len = k_end - k_start
    partial_q_start = max(q_start, k_start)
    partial_q_end = min(q_end - 1, k_end - 2)
    partial = 0
    if partial_q_start <= partial_q_end:
        count = partial_q_end - partial_q_start + 1
        partial = count * (partial_q_start + partial_q_end + 2 - 2 * k_start) // 2

    full_q_start = max(q_start, k_end - 1)
    full_q_end = q_end - 1
    full = 0
    if full_q_start <= full_q_end:
        full = (full_q_end - full_q_start + 1) * k_len
    return int(partial + full)


def _build_chunk_pair_program(
    row_spec: PackedRowAttentionSpec,
    *,
    chunk_ranges: tuple[TokenRange, ...],
) -> tuple[torch.Tensor, list[float]]:
    chunk_count = len(chunk_ranges)
    if chunk_count == 0:
        return torch.zeros((0, 0), dtype=torch.int64), []
    chunk_size = int(chunk_ranges[0].size())
    pair_rows = [[0 for _ in range(chunk_count)] for _ in range(chunk_count)]
    q_weights = [0.0 for _ in range(chunk_count)]

    for slice_ in row_spec.slices:
        q_start = int(slice_.q_range.start)
        q_end = int(slice_.q_range.end)
        k_start = int(slice_.k_range.start)
        k_end = int(slice_.k_range.end)
        if q_end <= q_start or k_end <= k_start:
            continue

        q_first = q_start // chunk_size
        q_last = (q_end - 1) // chunk_size
        k_first = k_start // chunk_size
        k_last = (k_end - 1) // chunk_size

        k_piece_lengths: list[int] = []
        k_piece_prefix_lengths: list[int] = []
        running_k_len = 0
        for k_chunk_index in range(k_first, k_last + 1):
            k_piece_start = (
                k_start if k_chunk_index == k_first else k_chunk_index * chunk_size
            )
            k_piece_end = (
                k_end if k_chunk_index == k_last else (k_chunk_index + 1) * chunk_size
            )
            k_piece_len = k_piece_end - k_piece_start
            if k_piece_len <= 0:
                continue
            running_k_len += k_piece_len
            k_piece_lengths.append(k_piece_len)
            k_piece_prefix_lengths.append(running_k_len)
        if not k_piece_lengths:
            continue

        if slice_.mask_kind is AttnMaskKind.FULL:
            total_k_len = running_k_len
            for q_chunk_index in range(q_first, q_last + 1):
                q_piece_start = (
                    q_start if q_chunk_index == q_first else q_chunk_index * chunk_size
                )
                q_piece_end = (
                    q_end
                    if q_chunk_index == q_last
                    else (q_chunk_index + 1) * chunk_size
                )
                q_piece_len = q_piece_end - q_piece_start
                if q_piece_len <= 0:
                    continue
                row = pair_rows[q_chunk_index]
                for k_offset, k_piece_len in enumerate(k_piece_lengths):
                    row[k_first + k_offset] += q_piece_len * k_piece_len
                q_weights[q_chunk_index] += float(q_piece_len * total_k_len)
            continue

        for q_chunk_index in range(q_first, q_last + 1):
            q_piece_start = (
                q_start if q_chunk_index == q_first else q_chunk_index * chunk_size
            )
            q_piece_end = (
                q_end if q_chunk_index == q_last else (q_chunk_index + 1) * chunk_size
            )
            q_piece_len = q_piece_end - q_piece_start
            if q_piece_len <= 0:
                continue

            row = pair_rows[q_chunk_index]
            q_total = 0

            full_k_last = min(k_last, q_chunk_index - 1)
            if full_k_last >= k_first:
                full_k_limit = full_k_last - k_first
                for k_offset in range(full_k_limit + 1):
                    row[k_first + k_offset] += q_piece_len * k_piece_lengths[k_offset]
                q_total += q_piece_len * k_piece_prefix_lengths[full_k_limit]

            if k_first <= q_chunk_index <= k_last:
                k_piece_start = q_chunk_index * chunk_size
                if q_chunk_index == k_first:
                    k_piece_start = max(k_piece_start, k_start)
                k_piece_end = (q_chunk_index + 1) * chunk_size
                if q_chunk_index == k_last:
                    k_piece_end = min(k_piece_end, k_end)
                pair_count = _causal_piece_pair_count_from_bounds(
                    q_start=q_piece_start,
                    q_end=q_piece_end,
                    k_start=k_piece_start,
                    k_end=k_piece_end,
                )
                if pair_count > 0:
                    row[q_chunk_index] += pair_count
                    q_total += pair_count

            if q_total > 0:
                q_weights[q_chunk_index] += float(q_total)
    return torch.tensor(pair_rows, dtype=torch.int64), q_weights


def _collect_rank_stage_pieces(
    row_spec: PackedRowAttentionSpec,
    *,
    chunk_ranges: tuple[TokenRange, ...],
    owners: tuple[int, ...],
    wave_assignment: tuple[int, ...],
    target_rank: int,
    cp_size: int,
) -> tuple[
    list[StagePiece],
    list[list[StagePiece]],
    list[list[list[TokenRange]]],
    list[list[list[TokenRange]]],
]:
    wave_count = max(wave_assignment, default=0) + 1 if wave_assignment else 0
    local_stage_pieces: list[StagePiece] = []
    remote_stage_pieces: list[list[StagePiece]] = [[] for _ in range(wave_count)]
    recv_request_ranges: list[list[list[TokenRange]]] = [
        [[] for _ in range(cp_size)] for _ in range(wave_count)
    ]
    send_request_ranges: list[list[list[TokenRange]]] = [
        [[] for _ in range(cp_size)] for _ in range(wave_count)
    ]
    chunk_starts = tuple(int(range_.start) for range_ in chunk_ranges)
    chunk_ends = tuple(int(range_.end) for range_ in chunk_ranges)

    for slice_ in row_spec.slices:
        q_parts = _indexed_intersections(
            slice_.q_range,
            chunk_ranges,
            candidate_starts=chunk_starts,
            candidate_ends=chunk_ends,
        )
        if not q_parts:
            continue
        k_parts = _indexed_intersections(
            slice_.k_range,
            chunk_ranges,
            candidate_starts=chunk_starts,
            candidate_ends=chunk_ends,
        )
        if not k_parts:
            continue

        target_q_parts = [
            (q_chunk_index, q_piece)
            for q_chunk_index, q_piece in q_parts
            if int(owners[q_chunk_index]) == int(target_rank)
        ]
        target_k_parts = [
            (k_chunk_index, k_piece)
            for k_chunk_index, k_piece in k_parts
            if int(owners[k_chunk_index]) == int(target_rank)
        ]

        if target_q_parts:
            for q_chunk_index, q_piece in target_q_parts:
                del q_chunk_index
                for k_chunk_index, k_piece in k_parts:
                    piece_mask_kind = _resolve_stage_mask_kind(
                        mask_kind=slice_.mask_kind,
                        q_piece=q_piece,
                        k_piece=k_piece,
                    )
                    if piece_mask_kind is None:
                        continue
                    source_rank = int(owners[k_chunk_index])
                    piece = (
                        q_piece,
                        k_piece,
                        piece_mask_kind,
                        slice_.family_index,
                    )
                    if source_rank == int(target_rank):
                        local_stage_pieces.append(piece)
                        continue
                    wave_index = int(wave_assignment[k_chunk_index])
                    remote_stage_pieces[wave_index].append(piece)
                    recv_request_ranges[wave_index][source_rank].append(k_piece)

        if target_k_parts:
            for q_chunk_index, q_piece in q_parts:
                host_rank = int(owners[q_chunk_index])
                if host_rank == int(target_rank):
                    continue
                for k_chunk_index, k_piece in target_k_parts:
                    piece_mask_kind = _resolve_stage_mask_kind(
                        mask_kind=slice_.mask_kind,
                        q_piece=q_piece,
                        k_piece=k_piece,
                    )
                    if piece_mask_kind is None:
                        continue
                    wave_index = int(wave_assignment[k_chunk_index])
                    send_request_ranges[wave_index][host_rank].append(k_piece)

    return (
        local_stage_pieces,
        remote_stage_pieces,
        recv_request_ranges,
        send_request_ranges,
    )


def _contiguous_chunk_assignment(
    *,
    q_weights: list[float],
    cp_size: int,
) -> tuple[int, ...]:
    chunk_count = len(q_weights)
    if chunk_count == 0:
        return tuple()
    if cp_size <= 1:
        return tuple(0 for _ in range(chunk_count))
    prefix = [0.0]
    for weight in q_weights:
        prefix.append(prefix[-1] + weight)
    total = prefix[-1]
    boundaries = [0]
    for split_index in range(1, cp_size):
        remaining_ranks = cp_size - split_index
        min_boundary = boundaries[-1] + 1
        max_boundary = chunk_count - remaining_ranks
        if min_boundary > max_boundary:
            boundaries.append(boundaries[-1])
            continue
        target = (
            total * split_index / cp_size
            if total > 0.0
            else float(chunk_count) * split_index / cp_size
        )
        best_boundary = min_boundary
        best_error = float("inf")
        for boundary in range(min_boundary, max_boundary + 1):
            current = prefix[boundary] if total > 0.0 else float(boundary)
            error = abs(current - target)
            if error < best_error:
                best_error = error
                best_boundary = boundary
        boundaries.append(best_boundary)
    boundaries.append(chunk_count)

    owners = [0 for _ in range(chunk_count)]
    for rank, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        for chunk_index in range(start, end):
            owners[chunk_index] = rank
    return tuple(owners)


def _candidate_chunk_indices(
    *,
    owners: tuple[int, ...],
    target_rank: int,
    q_weights: list[float],
    limit: int,
) -> tuple[int, ...]:
    rank_chunks = [
        chunk_index
        for chunk_index, owner in enumerate(owners)
        if int(owner) == int(target_rank)
    ]
    if not rank_chunks:
        return tuple()
    if limit <= 0 or len(rank_chunks) <= limit:
        return tuple(rank_chunks)

    boundary_chunks = [
        chunk_index
        for chunk_index in rank_chunks
        if chunk_index == 0
        or chunk_index + 1 == len(owners)
        or int(owners[chunk_index - 1]) != int(target_rank)
        or int(owners[chunk_index + 1]) != int(target_rank)
    ]
    weighted_chunks = sorted(
        rank_chunks,
        key=lambda index: (-q_weights[index], index),
    )[:limit]
    ordered_candidates = [*boundary_chunks, *weighted_chunks]
    deduped: list[int] = []
    seen: set[int] = set()
    for chunk_index in ordered_candidates:
        if chunk_index in seen:
            continue
        deduped.append(chunk_index)
        seen.add(chunk_index)
        if len(deduped) >= limit:
            break
    return tuple(deduped)


def _wave_assignment(
    *,
    chunk_count: int,
    wave_count: int,
) -> tuple[int, ...]:
    if chunk_count <= 0:
        return tuple()
    if wave_count <= 1:
        return tuple(0 for _ in range(chunk_count))
    return tuple(
        (chunk_index * wave_count) // chunk_count for chunk_index in range(chunk_count)
    )


def _chunk_ranges_for_owner(
    *,
    chunk_ranges: tuple[TokenRange, ...],
    owners: tuple[int, ...],
    owner_rank: int,
) -> tuple[TokenRange, ...]:
    return _merge_ranges(
        [
            chunk_ranges[chunk_index]
            for chunk_index, rank in enumerate(owners)
            if int(rank) == int(owner_rank)
        ]
    )


def _ranges_size(ranges: tuple[TokenRange, ...]) -> int:
    return int(sum(range_.size() for range_ in ranges))


def _chunk_mask_stats(
    *,
    chunk_lengths: tuple[int, ...],
    chunk_mask: torch.Tensor,
    chunk_lengths_tensor: torch.Tensor | None = None,
) -> tuple[int, int]:
    if (
        chunk_lengths_tensor is not None
        and len(chunk_lengths) >= _CHUNK_MASK_STATS_TORCH_THRESHOLD
    ):
        if int(chunk_mask.numel()) == 0 or not bool(chunk_mask.any().item()):
            return 0, 0
        token_count = int(chunk_lengths_tensor[chunk_mask].sum().item())
        run_starts = chunk_mask.clone()
        run_starts[1:] = torch.logical_and(
            run_starts[1:], torch.logical_not(chunk_mask[:-1])
        )
        range_count = int(run_starts.sum().item())
        return token_count, range_count
    token_count = 0
    range_count = 0
    in_run = False
    for is_set, length in zip(chunk_mask.tolist(), chunk_lengths, strict=True):
        if bool(is_set):
            token_count += int(length)
            if not in_run:
                range_count += 1
                in_run = True
            continue
        in_run = False
    return token_count, range_count


def _stage_cost_ms(
    *,
    pair_count: int,
    q_tokens: int,
    k_tokens: int,
    q_range_count: int,
    k_range_count: int,
    config: ContextParallelConfig,
    backward: bool,
    local: bool,
) -> float:
    pair_ms = (
        config.planner_local_backward_pair_ms
        if backward and local
        else (
            config.planner_remote_backward_pair_ms
            if backward
            else (
                config.planner_local_pair_ms if local else config.planner_remote_pair_ms
            )
        )
    )
    remote_underfill_ms = 0.0
    if not local and (pair_count > 0 or q_tokens > 0 or k_tokens > 0):
        token_shortfall = max(
            int(config.planner_remote_stage_token_floor) - min(q_tokens, k_tokens),
            0,
        )
        pair_shortfall = max(
            int(config.planner_remote_stage_pair_floor) - int(pair_count),
            0,
        )
        token_scale = (
            float(token_shortfall) / float(config.planner_remote_stage_token_floor)
            if int(config.planner_remote_stage_token_floor) > 0
            else 0.0
        )
        pair_scale = (
            float(pair_shortfall) / float(config.planner_remote_stage_pair_floor)
            if int(config.planner_remote_stage_pair_floor) > 0
            else 0.0
        )
        remote_underfill_ms = float(config.planner_remote_stage_underfill_ms) * max(
            token_scale,
            pair_scale,
        )
    return (
        float(config.planner_stage_overhead_ms)
        + float(pair_count) * float(pair_ms)
        + float(q_tokens) * float(config.planner_merge_q_token_ms)
        + float(q_range_count + k_range_count)
        * float(config.planner_interval_overhead_ms)
        + remote_underfill_ms
    )


def _comm_cost_ms(
    *,
    tokens: int,
    range_count: int,
    config: ContextParallelConfig,
    backward: bool,
) -> float:
    per_token = (
        float(config.planner_reduce_token_ms)
        if backward
        else float(config.planner_fetch_token_ms)
    )
    if tokens <= 0 and range_count <= 0:
        return 0.0
    return (
        float(config.planner_comm_stage_overhead_ms)
        + float(tokens) * per_token
        + float(range_count) * float(config.planner_interval_overhead_ms)
    )


def _simulate_forward_time_ms(
    *,
    local_stage_ms: float,
    remote_stage_ms: tuple[float, ...],
    remote_fetch_ms: tuple[float, ...],
) -> float:
    if not remote_stage_ms:
        return local_stage_ms

    fetch_ready = float(remote_fetch_ms[0])
    current_time = float(local_stage_ms)
    for wave_index, stage_ms in enumerate(remote_stage_ms):
        compute_start = max(current_time, fetch_ready)
        if wave_index + 1 < len(remote_stage_ms):
            fetch_ready = compute_start + float(remote_fetch_ms[wave_index + 1])
        current_time = compute_start + float(stage_ms)
    return current_time


def _simulate_backward_time_ms(
    *,
    local_stage_ms: float,
    remote_stage_ms: tuple[float, ...],
    remote_reduce_ms: tuple[float, ...],
) -> float:
    if not remote_stage_ms:
        return local_stage_ms

    current_time = 0.0
    reduce_ready_times: list[float] = []
    for stage_ms, reduce_ms in zip(remote_stage_ms, remote_reduce_ms):
        current_time += float(stage_ms)
        reduce_ready_times.append(current_time + float(reduce_ms))
    current_time += float(local_stage_ms)
    return max(current_time, max(reduce_ready_times, default=0.0))


def _evaluate_plan(
    *,
    chunk_ranges: tuple[TokenRange, ...],
    pair_matrix: list[list[int]] | torch.Tensor,
    owners: tuple[int, ...],
    wave_assignment: tuple[int, ...],
    cp_size: int,
    config: ContextParallelConfig,
    pair_positive: torch.Tensor | None = None,
    chunk_lengths: tuple[int, ...] | None = None,
    chunk_lengths_tensor: torch.Tensor | None = None,
) -> dict[str, Any]:
    rank_scores: list[float] = []
    rank_forward_ms: list[float] = []
    rank_backward_ms: list[float] = []
    chunk_count = len(chunk_ranges)
    wave_count = max(wave_assignment, default=0) + 1 if wave_assignment else 0
    pair_counts = (
        pair_matrix
        if isinstance(pair_matrix, torch.Tensor) and pair_matrix.dtype == torch.int64
        else torch.as_tensor(pair_matrix, dtype=torch.int64)
    )
    if pair_positive is None:
        pair_positive = pair_counts > 0
    if chunk_lengths is None:
        chunk_lengths = tuple(int(range_.size()) for range_ in chunk_ranges)
    if (
        chunk_lengths_tensor is None
        and len(chunk_lengths) >= _CHUNK_MASK_STATS_TORCH_THRESHOLD
    ):
        chunk_lengths_tensor = torch.tensor(chunk_lengths, dtype=torch.int64)
    owners_tensor = torch.tensor(owners, dtype=torch.int64)
    wave_tensor = torch.tensor(
        wave_assignment,
        dtype=torch.int64,
    )
    owner_masks = [owners_tensor == rank for rank in range(cp_size)]
    owner_indices = [
        torch.nonzero(owner_mask, as_tuple=False).flatten()
        for owner_mask in owner_masks
    ]
    empty_pair_counts = pair_counts.new_zeros((0, chunk_count))
    empty_pair_positive = pair_positive.new_zeros((0, chunk_count))
    pair_counts_by_rank_rows = [
        (
            empty_pair_counts
            if int(owner_index.numel()) == 0
            else pair_counts.index_select(0, owner_index)
        )
        for owner_index in owner_indices
    ]
    pair_positive_by_rank_rows = [
        (
            empty_pair_positive
            if int(owner_index.numel()) == 0
            else pair_positive.index_select(0, owner_index)
        )
        for owner_index in owner_indices
    ]
    pair_positive_by_rank_cols = [
        (
            torch.zeros(chunk_count, dtype=torch.bool)
            if int(rank_rows.numel()) == 0
            else rank_rows.any(dim=0)
        )
        for rank_rows in pair_positive_by_rank_rows
    ]
    wave_masks = [wave_tensor == wave_index for wave_index in range(wave_count)]

    for rank in range(cp_size):
        owned_q_mask = owner_masks[rank]
        owned_q_indices = owner_indices[rank]
        owned_pair_counts = pair_counts_by_rank_rows[rank]
        owned_pair_positive = pair_positive_by_rank_rows[rank]
        owned_positive_cols = pair_positive_by_rank_cols[rank]

        local_pairs = (
            0
            if int(owned_q_indices.numel()) == 0
            else int(owned_pair_counts.index_select(1, owned_q_indices).sum().item())
        )
        local_q_mask = torch.zeros(chunk_count, dtype=torch.bool)
        if int(owned_q_indices.numel()) > 0:
            touched_local_q = owned_pair_positive.index_select(1, owned_q_indices).any(
                dim=1
            )
            if bool(touched_local_q.any().item()):
                local_q_mask[owned_q_indices[touched_local_q]] = True
        local_k_mask = owned_q_mask & owned_positive_cols
        local_q_tokens, local_q_range_count = _chunk_mask_stats(
            chunk_lengths=chunk_lengths,
            chunk_mask=local_q_mask,
            chunk_lengths_tensor=chunk_lengths_tensor,
        )
        local_k_tokens, local_k_range_count = _chunk_mask_stats(
            chunk_lengths=chunk_lengths,
            chunk_mask=local_k_mask,
            chunk_lengths_tensor=chunk_lengths_tensor,
        )
        local_stage_ms = _stage_cost_ms(
            pair_count=local_pairs,
            q_tokens=local_q_tokens,
            k_tokens=local_k_tokens,
            q_range_count=local_q_range_count,
            k_range_count=local_k_range_count,
            config=config,
            backward=False,
            local=True,
        )
        local_backward_ms = _stage_cost_ms(
            pair_count=local_pairs,
            q_tokens=local_q_tokens,
            k_tokens=local_k_tokens,
            q_range_count=local_q_range_count,
            k_range_count=local_k_range_count,
            config=config,
            backward=True,
            local=True,
        )

        remote_stage_ms: list[float] = []
        remote_fetch_ms: list[float] = []
        remote_backward_ms: list[float] = []
        remote_reduce_ms: list[float] = []
        for wave_index in range(wave_count):
            request_tokens_by_source = [0 for _ in range(cp_size)]
            request_range_counts_by_source = [0 for _ in range(cp_size)]
            request_pairs = 0
            touched_q_mask = torch.zeros(chunk_count, dtype=torch.bool)
            for source_rank in range(cp_size):
                if source_rank == rank:
                    continue
                touched_source_mask = (
                    owner_masks[source_rank]
                    & wave_masks[wave_index]
                    & owned_positive_cols
                )
                (
                    request_tokens_by_source[source_rank],
                    request_range_counts_by_source[source_rank],
                ) = _chunk_mask_stats(
                    chunk_lengths=chunk_lengths,
                    chunk_mask=touched_source_mask,
                    chunk_lengths_tensor=chunk_lengths_tensor,
                )
                if request_tokens_by_source[source_rank] <= 0:
                    continue
                touched_source_indices = torch.nonzero(
                    touched_source_mask,
                    as_tuple=False,
                ).flatten()
                request_pairs += int(
                    owned_pair_counts.index_select(1, touched_source_indices)
                    .sum()
                    .item()
                )
                touched_remote_q = owned_pair_positive.index_select(
                    1,
                    touched_source_indices,
                ).any(dim=1)
                if bool(touched_remote_q.any().item()):
                    touched_q_mask[owned_q_indices[touched_remote_q]] = True
            recv_tokens = sum(request_tokens_by_source)
            recv_range_count = sum(request_range_counts_by_source)
            if request_pairs <= 0 and recv_tokens <= 0 and recv_range_count <= 0:
                continue

            send_tokens_by_peer = [0 for _ in range(cp_size)]
            send_range_counts_by_peer = [0 for _ in range(cp_size)]
            aggregate_send_mask = torch.zeros(chunk_count, dtype=torch.bool)
            owned_wave_mask = owned_q_mask & wave_masks[wave_index]
            if bool(owned_wave_mask.any().item()):
                for peer_rank in range(cp_size):
                    if peer_rank == rank:
                        continue
                    send_mask = owned_wave_mask & pair_positive_by_rank_cols[peer_rank]
                    (
                        send_tokens_by_peer[peer_rank],
                        send_range_counts_by_peer[peer_rank],
                    ) = _chunk_mask_stats(
                        chunk_lengths=chunk_lengths,
                        chunk_mask=send_mask,
                        chunk_lengths_tensor=chunk_lengths_tensor,
                    )
                    if send_tokens_by_peer[peer_rank] > 0:
                        aggregate_send_mask |= send_mask
            (
                send_tokens_by_peer[rank],
                send_range_counts_by_peer[rank],
            ) = _chunk_mask_stats(
                chunk_lengths=chunk_lengths,
                chunk_mask=aggregate_send_mask,
                chunk_lengths_tensor=chunk_lengths_tensor,
            )

            send_tokens = sum(send_tokens_by_peer)
            q_tokens, q_range_count = _chunk_mask_stats(
                chunk_lengths=chunk_lengths,
                chunk_mask=touched_q_mask,
                chunk_lengths_tensor=chunk_lengths_tensor,
            )
            remote_stage_ms.append(
                _stage_cost_ms(
                    pair_count=request_pairs,
                    q_tokens=q_tokens,
                    k_tokens=recv_tokens,
                    q_range_count=q_range_count,
                    k_range_count=recv_range_count,
                    config=config,
                    backward=False,
                    local=False,
                )
            )
            remote_backward_ms.append(
                _stage_cost_ms(
                    pair_count=request_pairs,
                    q_tokens=q_tokens,
                    k_tokens=recv_tokens,
                    q_range_count=q_range_count,
                    k_range_count=recv_range_count,
                    config=config,
                    backward=True,
                    local=False,
                )
            )
            remote_fetch_ms.append(
                _comm_cost_ms(
                    tokens=max(send_tokens, recv_tokens),
                    range_count=max(sum(send_range_counts_by_peer), recv_range_count),
                    config=config,
                    backward=False,
                )
            )
            remote_reduce_ms.append(
                _comm_cost_ms(
                    tokens=max(send_tokens, recv_tokens),
                    range_count=max(sum(send_range_counts_by_peer), recv_range_count),
                    config=config,
                    backward=True,
                )
            )

        forward_ms = _simulate_forward_time_ms(
            local_stage_ms=local_stage_ms if local_pairs > 0 else 0.0,
            remote_stage_ms=tuple(remote_stage_ms),
            remote_fetch_ms=tuple(remote_fetch_ms),
        )
        backward_ms = _simulate_backward_time_ms(
            local_stage_ms=local_backward_ms if local_pairs > 0 else 0.0,
            remote_stage_ms=tuple(remote_backward_ms),
            remote_reduce_ms=tuple(remote_reduce_ms),
        )
        rank_forward_ms.append(float(forward_ms))
        rank_backward_ms.append(float(backward_ms))
        rank_scores.append(float(forward_ms + backward_ms))
    return {
        "score": max(rank_scores, default=0.0),
        "rank_scores": tuple(rank_scores),
        "rank_forward_ms": tuple(rank_forward_ms),
        "rank_backward_ms": tuple(rank_backward_ms),
    }


def _search_generic_chunk_assignment(
    *,
    chunk_ranges: tuple[TokenRange, ...],
    pair_matrix: list[list[int]] | torch.Tensor,
    q_weights: list[float],
    cp_size: int,
    config: ContextParallelConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    config = _search_config_for_chunk_count(
        config=config,
        chunk_count=len(chunk_ranges),
    )
    wave_count_candidates = range(
        1,
        min(int(config.planner_max_remote_waves), len(chunk_ranges)) + 1,
    )
    best: tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]] | None = None
    eval_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[str, Any]] = {}
    pair_counts = torch.as_tensor(pair_matrix, dtype=torch.int64)
    pair_positive = pair_counts > 0
    chunk_lengths = tuple(int(range_.size()) for range_ in chunk_ranges)
    chunk_lengths_tensor = (
        torch.tensor(chunk_lengths, dtype=torch.int64)
        if len(chunk_lengths) >= _CHUNK_MASK_STATS_TORCH_THRESHOLD
        else None
    )

    def _evaluate_candidate(
        *,
        owners: tuple[int, ...],
        wave_assignment: tuple[int, ...],
    ) -> dict[str, Any]:
        cache_key = (owners, wave_assignment)
        cached = eval_cache.get(cache_key)
        if cached is not None:
            return cached
        cached = _evaluate_plan(
            chunk_ranges=chunk_ranges,
            pair_matrix=pair_counts,
            owners=owners,
            wave_assignment=wave_assignment,
            cp_size=cp_size,
            config=config,
            pair_positive=pair_positive,
            chunk_lengths=chunk_lengths,
            chunk_lengths_tensor=chunk_lengths_tensor,
        )
        eval_cache[cache_key] = cached
        return cached

    contiguous_owners = _contiguous_chunk_assignment(
        q_weights=q_weights,
        cp_size=cp_size,
    )
    if not contiguous_owners:
        wave_assignment = _wave_assignment(chunk_count=len(chunk_ranges), wave_count=1)
        return (
            contiguous_owners,
            wave_assignment,
            _evaluate_candidate(
                owners=contiguous_owners,
                wave_assignment=wave_assignment,
            ),
        )

    for wave_count in wave_count_candidates:
        wave_assignment = _wave_assignment(
            chunk_count=len(chunk_ranges),
            wave_count=wave_count,
        )
        current_owners = contiguous_owners
        current_eval = _evaluate_candidate(
            owners=current_owners,
            wave_assignment=wave_assignment,
        )

        search_steps_remaining = (
            0 if cp_size >= 8 else int(config.planner_max_search_steps)
        )
        if cp_size == 4 and search_steps_remaining > 0:
            probe_move = _best_improving_move(
                current_owners=current_owners,
                current_eval=current_eval,
                wave_assignment=wave_assignment,
                cp_size=cp_size,
                q_weights=q_weights,
                candidate_limit=min(
                    int(config.planner_candidate_chunk_limit),
                    _CP4_SEARCH_PROBE_CANDIDATE_LIMIT,
                ),
                evaluate_candidate=_evaluate_candidate,
            )
            if (
                probe_move is not None
                and float(current_eval["score"]) - float(probe_move[1]["score"])
                >= _CP4_SEARCH_PROBE_IMPROVEMENT_MS
            ):
                current_owners, current_eval = probe_move
                search_steps_remaining -= 1
            else:
                search_steps_remaining = 0

        for _ in range(search_steps_remaining):
            best_move = _best_improving_move(
                current_owners=current_owners,
                current_eval=current_eval,
                wave_assignment=wave_assignment,
                cp_size=cp_size,
                q_weights=q_weights,
                candidate_limit=int(config.planner_candidate_chunk_limit),
                evaluate_candidate=_evaluate_candidate,
            )
            if best_move is None:
                break
            current_owners, current_eval = best_move

        if best is None or float(current_eval["score"]) + 1e-9 < float(
            best[2]["score"]
        ):
            best = (current_owners, wave_assignment, current_eval)
    if best is None:
        raise RuntimeError("Failed to evaluate any CP planner wave assignment.")
    return best


def _folded_chunk_assignment(
    *,
    weights: list[float],
    cp_size: int,
) -> tuple[int, ...]:
    if len(weights) < 2 * cp_size:
        return tuple()
    slab_owners = _contiguous_chunk_assignment(
        q_weights=weights,
        cp_size=2 * cp_size,
    )
    return tuple(min(owner, 2 * cp_size - 1 - owner) for owner in slab_owners)


def _ownership_range_counts(
    owners: tuple[int, ...],
    *,
    cp_size: int,
) -> tuple[int, ...]:
    counts = [0 for _ in range(cp_size)]
    previous = -1
    for owner in owners:
        if owner != previous:
            counts[int(owner)] += 1
        previous = int(owner)
    return tuple(counts)


def _rounded(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _stage_indexer_tile_pairs(
    pieces: list[ProfiledChunkPiece],
    *,
    profile: ContextParallelWorkloadProfile,
) -> int:
    queries: dict[tuple[int, int], int] = {}
    for (
        _q_index,
        _k_index,
        q_start,
        q_end,
        k_start,
        k_end,
        _mask_kind,
        _family,
    ) in pieces:
        query = (q_start, q_end)
        queries[query] = queries.get(query, 0) + k_end - k_start

    tile_pairs = 0
    for (q_start, q_end), k_tokens in queries.items():
        if k_tokens <= 0:
            continue
        k_chunk = min(k_tokens, int(profile.indexer_max_k_tokens))
        q_chunk = max(1, int(profile.indexer_score_workspace_elements) // k_chunk)
        rounded_q = sum(
            _rounded(
                min(q_chunk, q_end - start),
                int(profile.query_tile_size),
            )
            for start in range(q_start, q_end, q_chunk)
        )
        rounded_k = sum(
            _rounded(
                min(k_chunk, k_tokens - start),
                int(profile.key_tile_size),
            )
            for start in range(0, k_tokens, k_chunk)
        )
        tile_pairs += rounded_q * rounded_k
    return tile_pairs


def _intervals_size(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(set(intervals))
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _profiled_chunk_pieces(
    row_spec: PackedRowAttentionSpec,
    *,
    chunk_ranges: tuple[TokenRange, ...],
) -> tuple[ProfiledChunkPiece, ...]:
    pieces = []
    chunk_starts = tuple(int(range_.start) for range_ in chunk_ranges)
    chunk_ends = tuple(int(range_.end) for range_ in chunk_ranges)
    for slice_ in row_spec.slices:
        q_parts = _indexed_intersections(
            slice_.q_range,
            chunk_ranges,
            candidate_starts=chunk_starts,
            candidate_ends=chunk_ends,
        )
        k_parts = _indexed_intersections(
            slice_.k_range,
            chunk_ranges,
            candidate_starts=chunk_starts,
            candidate_ends=chunk_ends,
        )
        for q_index, q_piece in q_parts:
            for k_index, k_piece in k_parts:
                mask_kind = _resolve_stage_mask_kind(
                    mask_kind=slice_.mask_kind,
                    q_piece=q_piece,
                    k_piece=k_piece,
                )
                if mask_kind is not None:
                    pieces.append(
                        (
                            q_index,
                            k_index,
                            int(q_piece.start),
                            int(q_piece.end),
                            int(k_piece.start),
                            int(k_piece.end),
                            mask_kind.value,
                            slice_.family_index,
                        )
                    )
    return tuple(dict.fromkeys(pieces))


def _profiled_rank_statistics(
    *,
    chunk_pieces: tuple[ProfiledChunkPiece, ...],
    chunk_ranges: tuple[TokenRange, ...],
    owners: tuple[int, ...],
    wave_assignment: tuple[int, ...],
    cp_size: int,
    profile: ContextParallelWorkloadProfile,
) -> list[dict[str, int]]:
    wave_count = max(wave_assignment, default=0) + 1 if wave_assignment else 0
    stages: list[list[list[ProfiledChunkPiece]]] = [
        [[] for _ in range(wave_count + 1)] for _ in range(cp_size)
    ]
    recv_ranges: list[list[list[list[tuple[int, int]]]]] = [
        [[[] for _ in range(cp_size)] for _ in range(wave_count)]
        for _ in range(cp_size)
    ]
    send_ranges: list[list[list[list[tuple[int, int]]]]] = [
        [[[] for _ in range(cp_size)] for _ in range(wave_count)]
        for _ in range(cp_size)
    ]
    for piece in chunk_pieces:
        q_index, k_index, _q_start, _q_end, k_start, k_end, _mask, _family = piece
        destination = int(owners[q_index])
        source = int(owners[k_index])
        if source == destination:
            stages[destination][0].append(piece)
            continue
        wave = int(wave_assignment[k_index])
        stages[destination][wave + 1].append(piece)
        interval = (k_start, k_end)
        recv_ranges[destination][wave][source].append(interval)
        send_ranges[source][wave][destination].append(interval)

    query_tokens = [0 for _ in range(cp_size)]
    for range_, owner in zip(chunk_ranges, owners, strict=True):
        query_tokens[int(owner)] += int(range_.size())
    statistics = []
    for rank in range(cp_size):
        combined_k_tokens = _intervals_size(
            [(piece[4], piece[5]) for piece in stages[rank][0]]
        )
        recv_tokens = 0
        send_tokens = 0
        remote_peers: set[int] = set()
        for wave_recv, wave_send in zip(
            recv_ranges[rank],
            send_ranges[rank],
            strict=True,
        ):
            for peer, (peer_recv, peer_send) in enumerate(
                zip(wave_recv, wave_send, strict=True)
            ):
                recv_size = _intervals_size(peer_recv)
                send_size = _intervals_size(peer_send)
                recv_tokens += recv_size
                send_tokens += send_size
                combined_k_tokens += recv_size
                if peer != rank and (recv_size or send_size):
                    remote_peers.add(peer)
        statistics.append(
            {
                "query_tokens": query_tokens[rank],
                "tile_pairs": sum(
                    _stage_indexer_tile_pairs(pieces, profile=profile)
                    for pieces in stages[rank]
                ),
                "combined_k_tokens": combined_k_tokens,
                "fetch_send_tokens": send_tokens,
                "fetch_recv_tokens": recv_tokens,
                "remote_peers": len(remote_peers),
            }
        )
    return statistics


def _evaluate_profiled_assignment(
    *,
    chunk_pieces: tuple[ProfiledChunkPiece, ...],
    chunk_ranges: tuple[TokenRange, ...],
    owners: tuple[int, ...],
    wave_assignment: tuple[int, ...],
    cp_size: int,
    profile: ContextParallelWorkloadProfile,
) -> dict[str, Any]:
    rank_stats = _profiled_rank_statistics(
        chunk_pieces=chunk_pieces,
        chunk_ranges=chunk_ranges,
        owners=owners,
        wave_assignment=wave_assignment,
        cp_size=cp_size,
        profile=profile,
    )
    query_flops = 0
    indexer_flops = 0
    hbm_k_bytes = 0
    peak_memory_bytes = 0
    fetch_send_bytes = 0
    fetch_recv_bytes = 0
    dkv_send_bytes = 0
    dkv_recv_bytes = 0
    for rank in rank_stats:
        for stage in profile.stages:
            query_flops = max(
                query_flops,
                rank["query_tokens"] * int(stage.query_flops_per_token),
            )
            indexer_flops = max(
                indexer_flops,
                rank["tile_pairs"] * int(stage.tile_pair_flops),
            )
            hbm_k_bytes = max(
                hbm_k_bytes,
                rank["combined_k_tokens"] * int(stage.k_hbm_bytes_per_token),
            )
            peak_memory_bytes = max(
                peak_memory_bytes,
                rank["query_tokens"] * int(stage.query_memory_bytes_per_token)
                + rank["combined_k_tokens"] * int(stage.k_memory_bytes_per_token),
            )
            fetch_send_bytes = max(
                fetch_send_bytes,
                rank["fetch_send_tokens"] * int(stage.k_fetch_bytes_per_token),
            )
            fetch_recv_bytes = max(
                fetch_recv_bytes,
                rank["fetch_recv_tokens"] * int(stage.k_fetch_bytes_per_token),
            )
            dkv_send_bytes = max(
                dkv_send_bytes,
                rank["fetch_recv_tokens"] * int(stage.dkv_reduce_bytes_per_token),
            )
            dkv_recv_bytes = max(
                dkv_recv_bytes,
                rank["fetch_send_tokens"] * int(stage.dkv_reduce_bytes_per_token),
            )

    range_counts = _ownership_range_counts(owners, cp_size=cp_size)
    max_network_bytes = max(
        fetch_send_bytes,
        fetch_recv_bytes,
        dkv_send_bytes,
        dkv_recv_bytes,
    )
    return {
        "score": query_flops,
        "query_flops": query_flops,
        "indexer_flops": indexer_flops,
        "hbm_k_bytes": hbm_k_bytes,
        "peak_memory_bytes": peak_memory_bytes,
        "max_network_bytes": max_network_bytes,
        "fetch_send_bytes": fetch_send_bytes,
        "fetch_recv_bytes": fetch_recv_bytes,
        "dkv_send_bytes": dkv_send_bytes,
        "dkv_recv_bytes": dkv_recv_bytes,
        "max_remote_peers": max(
            (rank["remote_peers"] for rank in rank_stats),
            default=0,
        ),
        "max_ownership_ranges": max(range_counts, default=0),
        "rank_query_tokens": tuple(rank["query_tokens"] for rank in rank_stats),
        "rank_tile_pairs": tuple(rank["tile_pairs"] for rank in rank_stats),
        "rank_combined_k_tokens": tuple(
            rank["combined_k_tokens"] for rank in rank_stats
        ),
        "rank_fetch_send_tokens": tuple(
            rank["fetch_send_tokens"] for rank in rank_stats
        ),
        "rank_fetch_recv_tokens": tuple(
            rank["fetch_recv_tokens"] for rank in rank_stats
        ),
        "rank_remote_peers": tuple(rank["remote_peers"] for rank in rank_stats),
        "ownership_range_counts": range_counts,
    }


def _profiled_assignment_key(
    evaluation: dict[str, Any],
    owners: tuple[int, ...],
) -> tuple[Any, ...]:
    # These terms retain their own physical units; they are never added together.
    return (
        int(evaluation["query_flops"]),
        int(evaluation["max_network_bytes"]),
        int(evaluation["hbm_k_bytes"]),
        int(evaluation["indexer_flops"]),
        int(evaluation["max_remote_peers"]),
        int(evaluation["max_ownership_ranges"]),
        owners,
    )


def _search_chunk_assignment(
    *,
    row_spec: PackedRowAttentionSpec | None = None,
    chunk_ranges: tuple[TokenRange, ...],
    pair_matrix: list[list[int]] | torch.Tensor,
    q_weights: list[float],
    cp_size: int,
    config: ContextParallelConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, Any]]:
    generic = _search_generic_chunk_assignment(
        chunk_ranges=chunk_ranges,
        pair_matrix=pair_matrix,
        q_weights=q_weights,
        cp_size=cp_size,
        config=config,
    )
    profile = config.workload_profile
    if profile is None:
        return generic
    if row_spec is None:
        raise RuntimeError("Profile-aware CP planning requires the packed row spec.")

    chunk_pieces = _profiled_chunk_pieces(row_spec, chunk_ranges=chunk_ranges)
    lengths = [float(range_.size()) for range_ in chunk_ranges]
    candidates = [
        generic[0],
        _contiguous_chunk_assignment(q_weights=lengths, cp_size=cp_size),
        _folded_chunk_assignment(weights=lengths, cp_size=cp_size),
    ]
    baseline = _evaluate_profiled_assignment(
        chunk_pieces=chunk_pieces,
        chunk_ranges=chunk_ranges,
        owners=generic[0],
        wave_assignment=generic[1],
        cp_size=cp_size,
        profile=profile,
    )
    best_owners = generic[0]
    best_eval = baseline
    seen = {generic[0]}
    for owners in candidates[1:]:
        if not owners or owners in seen:
            continue
        seen.add(owners)
        if len(set(owners)) != cp_size:
            continue
        range_counts = _ownership_range_counts(owners, cp_size=cp_size)
        if max(range_counts, default=0) > int(profile.max_ownership_ranges_per_rank):
            continue
        evaluation = _evaluate_profiled_assignment(
            chunk_pieces=chunk_pieces,
            chunk_ranges=chunk_ranges,
            owners=owners,
            wave_assignment=generic[1],
            cp_size=cp_size,
            profile=profile,
        )
        if int(evaluation["peak_memory_bytes"]) > int(
            baseline["peak_memory_bytes"]
        ) or _profiled_assignment_key(evaluation, owners) >= _profiled_assignment_key(
            baseline, generic[0]
        ):
            continue
        if _profiled_assignment_key(evaluation, owners) < _profiled_assignment_key(
            best_eval, best_owners
        ):
            best_owners = owners
            best_eval = evaluation
    return best_owners, generic[1], best_eval


def _flatten_ranges_by_peer(
    ranges_by_peer: tuple[tuple[TokenRange, ...], ...],
) -> tuple[TokenRange, ...]:
    return tuple(range_ for peer_ranges in ranges_by_peer for range_ in peer_ranges)


def _stage_local_buffer_ranges(
    global_ranges: tuple[TokenRange, ...],
) -> tuple[TokenRange, ...]:
    cursor = 0
    local_ranges: list[TokenRange] = []
    for range_ in global_ranges:
        size = int(range_.size())
        if size <= 0:
            continue
        local_ranges.append(TokenRange(start=cursor, end=cursor + size))
        cursor += size
    return tuple(local_ranges)


def _build_stage_from_pieces(
    *,
    stage_index: int,
    source_rank: int,
    source_ranks: tuple[int, ...],
    is_local_stage: bool,
    wave_index: int | None,
    pieces: list[StagePiece],
    host_local_ranges: tuple[TokenRange, ...],
    global_k_ranges: tuple[TokenRange, ...],
    local_k_ranges: tuple[TokenRange, ...],
    kv_fetch_plan: KvFetchPlan | None,
    dkv_reduce_plan: DkvReducePlan | None,
    remote_buffer_range: TokenRange | None,
    block_size: int,
) -> StagePlan:
    global_q_ranges = _merge_ranges([piece[0] for piece in pieces])
    logical_q_len = _ranges_size(global_q_ranges)
    logical_k_len = _ranges_size(global_k_ranges)
    q_len = (
        0
        if logical_q_len <= 0
        else ((logical_q_len + int(block_size) - 1) // int(block_size))
        * int(block_size)
    )
    k_len = (
        0
        if logical_k_len <= 0
        else ((logical_k_len + int(block_size) - 1) // int(block_size))
        * int(block_size)
    )
    owner_local_q_ranges: tuple[TokenRange, ...] = tuple()
    localized_slices: list[AttnSlice] = []
    mask_metadata: ExactMaskMetadata | None = None
    q_remap_cache: dict[tuple[int, int], TokenRange] = {}
    k_remap_cache: dict[tuple[int, int], TokenRange] = {}
    source_index_cache: dict[tuple[int, int], torch.Tensor] = {}
    assigned_q_keys: set[tuple[int, int]] = set()
    assigned_k_keys: set[tuple[int, int]] = set()

    if global_q_ranges:
        owner_local_q_ranges = tuple(
            _remap_subrange(range_, host_local_ranges) for range_ in global_q_ranges
        )
    q_token_indices = torch.full((q_len,), -1, dtype=torch.int64)
    k_token_indices = torch.full((k_len,), -1, dtype=torch.int64)
    last_slice_key: StageSliceKey | None = None
    for q_piece, k_piece, piece_mask_kind, family_index in sorted(
        pieces,
        key=lambda piece: (
            int(piece[0].start),
            int(piece[0].end),
            int(piece[1].start),
            int(piece[1].end),
            piece[2].value,
            -1 if piece[3] is None else int(piece[3]),
        ),
    ):
        q_key = _range_key(q_piece)
        k_key = _range_key(k_piece)
        localized_q = q_remap_cache.get(q_key)
        if localized_q is None:
            localized_q = _remap_subrange(q_piece, global_q_ranges)
            q_remap_cache[q_key] = localized_q
        localized_k = k_remap_cache.get(k_key)
        if localized_k is None:
            localized_k = _remap_subrange(k_piece, global_k_ranges)
            k_remap_cache[k_key] = localized_k
        q_source_indices = source_index_cache.get(q_key)
        if q_source_indices is None:
            q_source_indices = torch.arange(
                q_piece.start, q_piece.end, dtype=torch.int64
            )
            source_index_cache[q_key] = q_source_indices
        k_source_indices = source_index_cache.get(k_key)
        if k_source_indices is None:
            k_source_indices = torch.arange(
                k_piece.start, k_piece.end, dtype=torch.int64
            )
            source_index_cache[k_key] = k_source_indices
        slice_key = (
            0,
            int(localized_q.start),
            int(localized_q.end),
            int(localized_k.start),
            int(localized_k.end),
            piece_mask_kind.value,
            -1 if family_index is None else int(family_index),
        )
        if slice_key != last_slice_key:
            localized_slices.append(
                AttnSlice(
                    q_range=localized_q,
                    k_range=localized_k,
                    mask_kind=piece_mask_kind,
                    row_index=0,
                    family_index=family_index,
                )
            )
            last_slice_key = slice_key
        if q_key not in assigned_q_keys:
            _set_stage_token_indices(
                target_indices=q_token_indices,
                stage_range=localized_q,
                source_range=q_piece,
                source_indices=q_source_indices,
            )
            assigned_q_keys.add(q_key)
        if k_key not in assigned_k_keys:
            _set_stage_token_indices(
                target_indices=k_token_indices,
                stage_range=localized_k,
                source_range=k_piece,
                source_indices=k_source_indices,
            )
            assigned_k_keys.add(k_key)
    if localized_slices:
        mask_metadata = ExactMaskMetadata(
            q_token_indices=q_token_indices,
            k_token_indices=k_token_indices,
            cache_key=_exact_mask_metadata_cache_key(
                q_token_indices=q_token_indices,
                k_token_indices=k_token_indices,
            ),
        )
    return StagePlan(
        stage_index=stage_index,
        source_rank=source_rank,
        source_ranks=source_ranks,
        is_local_stage=is_local_stage,
        wave_index=wave_index,
        slices=tuple(localized_slices),
        global_q_ranges=global_q_ranges,
        global_k_ranges=global_k_ranges,
        owner_local_q_ranges=owner_local_q_ranges,
        owner_local_k_ranges=local_k_ranges,
        mask_metadata=mask_metadata,
        remote_buffer_range=remote_buffer_range,
        q_len=q_len,
        k_len=k_len,
        kv_fetch_plan=kv_fetch_plan,
        dkv_reduce_plan=dkv_reduce_plan,
    )


def _stage_plan_pair_count(stage: StagePlan) -> int:
    metadata = stage.mask_metadata
    if metadata is None:
        return 0
    pair_count = 0
    for slice_ in stage.slices:
        q_len = slice_.q_range.size()
        k_len = slice_.k_range.size()
        if q_len <= 0 or k_len <= 0:
            continue
        if slice_.mask_kind == AttnMaskKind.FULL:
            pair_count += q_len * k_len
            continue
        q_start = int(metadata.q_token_indices[slice_.q_range.start].item())
        k_start = int(metadata.k_token_indices[slice_.k_range.start].item())
        pair_count += _causal_piece_pair_count_from_bounds(
            q_start=q_start,
            q_end=q_start + q_len,
            k_start=k_start,
            k_end=k_start + k_len,
        )
    return pair_count


def _fusion_reduces_predicted_runtime(
    local_stage: StagePlan,
    remote_stage: StagePlan,
    *,
    config: ContextParallelConfig,
) -> bool:
    def stage_cost(stage: StagePlan, *, backward: bool, local: bool) -> float:
        return _stage_cost_ms(
            pair_count=_stage_plan_pair_count(stage),
            q_tokens=_ranges_size(stage.owner_local_q_ranges),
            k_tokens=_ranges_size(stage.owner_local_k_ranges),
            q_range_count=len(stage.owner_local_q_ranges),
            k_range_count=len(stage.owner_local_k_ranges),
            config=config,
            backward=backward,
            local=local,
        )

    def comm_cost(*, backward: bool) -> float:
        plan = remote_stage.dkv_reduce_plan if backward else remote_stage.kv_fetch_plan
        assert plan is not None
        peer_ranges = (
            cast(DkvReducePlan, plan).recv_ranges_by_peer
            if backward
            else cast(KvFetchPlan, plan).send_ranges_by_peer
        )
        return _comm_cost_ms(
            tokens=max(sum(plan.send_splits), sum(plan.recv_splits)),
            range_count=max(
                sum(len(ranges) for ranges in peer_ranges),
                len(remote_stage.global_k_ranges),
            ),
            config=config,
            backward=backward,
        )

    local_forward_ms, local_backward_ms = (
        stage_cost(local_stage, backward=backward, local=True)
        for backward in (False, True)
    )
    remote_forward_ms, remote_backward_ms = (
        stage_cost(remote_stage, backward=backward, local=False)
        for backward in (False, True)
    )
    fetch_ms, reduce_ms = (comm_cost(backward=value) for value in (False, True))

    separate_ms = _simulate_forward_time_ms(
        local_stage_ms=local_forward_ms,
        remote_stage_ms=(remote_forward_ms,),
        remote_fetch_ms=(fetch_ms,),
    ) + _simulate_backward_time_ms(
        local_stage_ms=local_backward_ms,
        remote_stage_ms=(remote_backward_ms,),
        remote_reduce_ms=(reduce_ms,),
    )
    combined_pairs = _stage_plan_pair_count(local_stage) + _stage_plan_pair_count(
        remote_stage
    )
    combined_q_tokens = _ranges_size(local_stage.owner_local_q_ranges)
    combined_k_tokens = _ranges_size(local_stage.owner_local_k_ranges) + _ranges_size(
        remote_stage.owner_local_k_ranges
    )
    fused_stage_ms = sum(
        _stage_cost_ms(
            pair_count=combined_pairs,
            q_tokens=combined_q_tokens,
            k_tokens=combined_k_tokens,
            q_range_count=len(local_stage.owner_local_q_ranges),
            k_range_count=len(local_stage.owner_local_k_ranges)
            + len(remote_stage.owner_local_k_ranges),
            config=config,
            backward=backward,
            local=True,
        )
        for backward in (False, True)
    )
    fused_ms = fetch_ms + fused_stage_ms + reduce_ms
    return fused_ms < separate_ms


def _fuse_remote_stage_when_faster(
    local_stage: StagePlan,
    remote_stage: StagePlan,
    *,
    block_size: int,
    config: ContextParallelConfig,
) -> StagePlan | None:
    if (
        not local_stage.is_local_stage
        or local_stage.fused_remote_k_len
        or remote_stage.is_local_stage
        or local_stage.mask_metadata is None
        or remote_stage.mask_metadata is None
        or remote_stage.kv_fetch_plan is None
        or remote_stage.dkv_reduce_plan is None
    ):
        return None

    local_k_len = _ranges_size(local_stage.owner_local_k_ranges)
    remote_k_len = _ranges_size(remote_stage.owner_local_k_ranges)
    logical_q_len = _ranges_size(local_stage.owner_local_q_ranges)
    if (
        logical_q_len <= 0
        or local_k_len <= 0
        or remote_k_len <= 0
        or not _fusion_reduces_predicted_runtime(
            local_stage,
            remote_stage,
            config=config,
        )
    ):
        return None

    shifted_remote_slices: list[AttnSlice] = []
    for slice_ in remote_stage.slices:
        try:
            global_q_range = _unmap_stage_subrange(
                slice_.q_range,
                remote_stage.global_q_ranges,
            )
            local_q_range = _remap_subrange(
                global_q_range,
                local_stage.global_q_ranges,
            )
        except RuntimeError:
            return None
        remote_q_indices = remote_stage.mask_metadata.q_token_indices[
            slice_.q_range.start : slice_.q_range.end
        ]
        local_q_indices = local_stage.mask_metadata.q_token_indices[
            local_q_range.start : local_q_range.end
        ]
        if not torch.equal(local_q_indices, remote_q_indices):
            return None
        shifted_remote_slices.append(
            replace(
                slice_,
                q_range=local_q_range,
                k_range=TokenRange(
                    start=int(slice_.k_range.start) + local_k_len,
                    end=int(slice_.k_range.end) + local_k_len,
                ),
            )
        )

    combined_k_len = local_k_len + remote_k_len
    padded_k_len = ((combined_k_len + int(block_size) - 1) // int(block_size)) * int(
        block_size
    )
    k_token_indices = torch.cat(
        (
            local_stage.mask_metadata.k_token_indices[:local_k_len],
            remote_stage.mask_metadata.k_token_indices[:remote_k_len],
            torch.full(
                (padded_k_len - combined_k_len,),
                -1,
                dtype=local_stage.mask_metadata.k_token_indices.dtype,
                device=local_stage.mask_metadata.k_token_indices.device,
            ),
        )
    )
    q_token_indices = local_stage.mask_metadata.q_token_indices.clone()
    return replace(
        local_stage,
        source_ranks=remote_stage.source_ranks,
        wave_index=remote_stage.wave_index,
        slices=local_stage.slices + tuple(shifted_remote_slices),
        global_k_ranges=local_stage.global_k_ranges + remote_stage.global_k_ranges,
        mask_metadata=ExactMaskMetadata(
            q_token_indices=q_token_indices,
            k_token_indices=k_token_indices,
            cache_key=_exact_mask_metadata_cache_key(
                q_token_indices=q_token_indices,
                k_token_indices=k_token_indices,
            ),
        ),
        remote_buffer_range=remote_stage.remote_buffer_range,
        fused_remote_k_len=remote_k_len,
        k_len=padded_k_len,
        kv_fetch_plan=remote_stage.kv_fetch_plan,
        dkv_reduce_plan=remote_stage.dkv_reduce_plan,
    )


def _empty_remote_stage(stage: StagePlan) -> StagePlan:
    return replace(
        stage,
        slices=(),
        owner_local_q_ranges=(),
        owner_local_k_ranges=(),
        q_len=0,
        k_len=0,
        global_q_ranges=(),
        global_k_ranges=(),
        mask_metadata=None,
        remote_buffer_range=None,
        kv_fetch_plan=None,
        dkv_reduce_plan=None,
    )


def _build_rank_runtime_plan(
    *,
    row_spec: PackedRowAttentionSpec,
    chunk_ranges: tuple[TokenRange, ...],
    owners: tuple[int, ...],
    wave_assignment: tuple[int, ...],
    token_layout_index: TokenLayoutIndex,
    cp_size: int,
    original_seq_len: int,
    target_rank: int,
    block_size: int,
    config: ContextParallelConfig,
) -> RankRuntimePlan:
    host_local_ranges = _chunk_ranges_for_owner(
        chunk_ranges=chunk_ranges,
        owners=owners,
        owner_rank=target_rank,
    )
    local_row_ranges = (
        tuple(host_local_ranges)
        if host_local_ranges
        else cast(tuple[TokenRange | None, ...], (None,))
    )
    local_token_count = _ranges_size(host_local_ranges)
    (
        local_stage_pieces,
        remote_stage_pieces,
        recv_request_ranges,
        send_request_ranges,
    ) = _collect_rank_stage_pieces(
        row_spec,
        chunk_ranges=chunk_ranges,
        owners=owners,
        wave_assignment=wave_assignment,
        target_rank=target_rank,
        cp_size=cp_size,
    )

    stage_plans: list[StagePlan] = []
    local_global_k_ranges = _merge_ranges([piece[1] for piece in local_stage_pieces])
    local_stage = _build_stage_from_pieces(
        stage_index=0,
        source_rank=target_rank,
        source_ranks=(target_rank,),
        is_local_stage=True,
        wave_index=None,
        pieces=local_stage_pieces,
        host_local_ranges=host_local_ranges,
        global_k_ranges=local_global_k_ranges,
        local_k_ranges=tuple(
            _remap_subrange(range_, host_local_ranges)
            for range_ in local_global_k_ranges
        ),
        kv_fetch_plan=None,
        dkv_reduce_plan=None,
        remote_buffer_range=None,
        block_size=block_size,
    )
    stage_plans.append(local_stage)

    wave_count = max(wave_assignment, default=0) + 1 if wave_assignment else 0
    remote_cursor = 0
    aggregate_send_ranges_by_peer: list[list[TokenRange]] = [[] for _ in range(cp_size)]
    aggregate_recv_splits = [0 for _ in range(cp_size)]
    backward_stage_indices: list[int] = []

    for wave_index in range(wave_count):
        request_ranges_by_source = tuple(
            (
                _merge_ranges(recv_request_ranges[wave_index][source_rank])
                if source_rank != target_rank
                else tuple()
            )
            for source_rank in range(cp_size)
        )
        send_global_ranges_by_peer = tuple(
            (
                _merge_ranges(send_request_ranges[wave_index][peer_rank])
                if peer_rank != target_rank
                else tuple()
            )
            for peer_rank in range(cp_size)
        )
        send_ranges_by_peer = tuple(
            (
                tuple(
                    _remap_subrange(range_, host_local_ranges) for range_ in peer_ranges
                )
                if peer_rank != target_rank
                else tuple()
            )
            for peer_rank, peer_ranges in enumerate(send_global_ranges_by_peer)
        )
        recv_splits = tuple(
            (
                _ranges_size(request_ranges_by_source[source_rank])
                if source_rank != target_rank
                else 0
            )
            for source_rank in range(cp_size)
        )
        send_splits = tuple(
            _ranges_size(peer_ranges) for peer_ranges in send_ranges_by_peer
        )
        for peer_rank, peer_ranges in enumerate(send_ranges_by_peer):
            if peer_rank == target_rank:
                continue
            aggregate_send_ranges_by_peer[peer_rank].extend(peer_ranges)
            aggregate_recv_splits[peer_rank] += int(recv_splits[peer_rank])
        global_k_ranges = _flatten_ranges_by_peer(request_ranges_by_source)
        local_k_ranges = _stage_local_buffer_ranges(global_k_ranges)
        stage_k_len = _ranges_size(global_k_ranges)
        remote_buffer_range = None
        if stage_k_len > 0:
            remote_buffer_range = TokenRange(
                start=remote_cursor,
                end=remote_cursor + stage_k_len,
            )
            remote_cursor += stage_k_len
        source_ranks = tuple(
            source_rank
            for source_rank in range(cp_size)
            if source_rank != target_rank and recv_splits[source_rank] > 0
        )
        stage_plan = _build_stage_from_pieces(
            stage_index=wave_index + 1,
            source_rank=-1 if len(source_ranks) != 1 else source_ranks[0],
            source_ranks=source_ranks,
            is_local_stage=False,
            wave_index=wave_index,
            pieces=remote_stage_pieces[wave_index],
            host_local_ranges=host_local_ranges,
            global_k_ranges=global_k_ranges,
            local_k_ranges=local_k_ranges,
            kv_fetch_plan=KvFetchPlan(
                send_splits=send_splits,
                recv_splits=recv_splits,
                send_ranges_by_peer=send_ranges_by_peer,
            ),
            dkv_reduce_plan=DkvReducePlan(
                send_splits=recv_splits,
                recv_splits=send_splits,
                recv_ranges_by_peer=send_ranges_by_peer,
            ),
            remote_buffer_range=remote_buffer_range,
            block_size=block_size,
        )
        fused_local_stage = _fuse_remote_stage_when_faster(
            stage_plans[0],
            stage_plan,
            block_size=block_size,
            config=config,
        )
        if fused_local_stage is not None:
            stage_plans[0] = fused_local_stage
            backward_stage_indices.append(0)
            stage_plan = _empty_remote_stage(stage_plan)
        else:
            backward_stage_indices.append(int(stage_plan.stage_index))
        stage_plans.append(stage_plan)

    aggregate_send_ranges = tuple(
        tuple(peer_ranges) for peer_ranges in aggregate_send_ranges_by_peer
    )
    aggregate_send_splits = tuple(
        _ranges_size(peer_ranges) for peer_ranges in aggregate_send_ranges
    )
    return RankRuntimePlan(
        rank=target_rank,
        original_seq_len=original_seq_len,
        token_layout_index=token_layout_index,
        local_valid_lengths=(local_token_count,),
        local_row_ranges=local_row_ranges,
        stage_plans=tuple(stage_plans),
        backward_stage_indices=tuple(
            backward_stage_indices + ([] if stage_plans[0].fused_remote_k_len else [0])
        ),
        remote_dkv_reduce_plan=DkvReducePlan(
            send_splits=tuple(aggregate_recv_splits),
            recv_splits=aggregate_send_splits,
            recv_ranges_by_peer=aggregate_send_ranges,
        ),
    )


def prepare_cp_micro(
    *,
    micro: PackedTensors,
    topology: ParallelTopology,
    config: ContextParallelConfig,
    cp_group: Any,
    cp_rank: int,
    build_gdn_execution_spec: bool = False,
    gdn_planner_config: Any | None = None,
    trace_token_uids: bool = False,
    prepare_execution_state: bool = True,
    block_mask_variants: tuple[CpBlockMaskVariant, ...] = (),
    target_device: torch.device | None = None,
    ref_logprobs: torch.Tensor | None = None,
    model_support_handler: Any | None = None,
    attention_head_dim: int | None = None,
    attention_value_head_dim: int | None = None,
) -> PreparedMegatronBatch:
    """Prepare one CP microbatch with a CPU-only planning phase.

    The intended overlap contract is: build the prefix-tree/runtime plan from
    CPU metadata, then materialize local tensors and BlockMasks on
    `target_device`. Passing CUDA `group_ids` or `parent_ids` still works for
    older direct callers, but it reintroduces D2H syncs and invalidates the
    host-ahead/device-behind lookahead assumption.

    Model-owned state is built exactly once here, after rank-local dispatch.
    Its handler must only enqueue device work: scalar CUDA reads would expose
    planning on the host and invalidate lookahead overlap.
    """
    state, rank_plan, spec, pad_multiple = prepare_megatron_context_parallel_state(
        micro=micro,
        topology=topology,
        config=config,
        cp_group=cp_group,
        cp_rank=cp_rank,
        build_gdn_execution_spec=build_gdn_execution_spec,
        gdn_planner_config=gdn_planner_config,
        block_mask_variants=block_mask_variants,
        target_device=target_device,
    )
    tensors, workload = dispatch_megatron_context_parallel_training_tensors(
        micro=micro,
        rank_plan=rank_plan,
        spec=spec,
        pad_multiple=pad_multiple,
        trace_token_uids=trace_token_uids,
        target_device=target_device,
        cp_group=cp_group,
        ref_logprobs=ref_logprobs,
    )
    if model_support_handler is not None:
        from art.megatron.model_support.spec import PrefixTreeModelStateContext

        state.model_state = dict(
            model_support_handler.build_prefix_tree_model_state(
                PrefixTreeModelStateContext(
                    input_pos=tensors.input_pos,
                    group_ids=micro["group_ids"],
                    parent_ids=micro["parent_ids"],
                    device=tensors.tokens.device,
                    attention_token_layout_index=rank_plan.token_layout_index,
                    attention_head_dim=attention_head_dim,
                    attention_value_head_dim=attention_value_head_dim,
                    context_parallel_state=state,
                )
            )
        )
    if tensors.token_uids is not None:
        state = replace(state, trace_token_uids=tensors.token_uids)
    if prepare_execution_state:
        from .executor import prepare_context_parallel_execution_state

        prepare_context_parallel_execution_state(
            state=state,
            device=tensors.tokens.device,
        )
    return PreparedMegatronBatch(
        tensors=tensors,
        packed_seq_params=None,
        attention_state=state,
        workload=workload,
        rank_plan=rank_plan,
        pad_multiple=pad_multiple,
    )


def preplan_megatron_context_parallel_state(
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    original_seq_len: int,
    topology: ParallelTopology,
    config: ContextParallelConfig,
    cp_rank: int,
    build_gdn_execution_spec: bool = False,
    gdn_planner_config: Any | None = None,
) -> str:
    """Warm one rank's immutable CPU plans without touching CUDA or collectives."""
    if int(topology.cp) <= 1:
        raise RuntimeError("context-parallel preplanning requires CP > 1")
    planning_key, bundle, _group_ids, _parent_ids = _get_or_build_planning_bundle(
        group_ids=group_ids,
        parent_ids=parent_ids,
        topology=topology,
        config=config,
        original_seq_len=original_seq_len,
        build_gdn_execution_spec=build_gdn_execution_spec,
    )
    _get_or_build_bundle_rank_plan(
        planning_key=planning_key,
        bundle=bundle,
        original_seq_len=original_seq_len,
        target_rank=cp_rank,
        block_size=int(config.block_size),
    )
    if build_gdn_execution_spec:
        _plan_gdn_rank_execution(
            planning_key=planning_key,
            bundle=bundle,
            topology=topology,
            cp_rank=cp_rank,
            gdn_planner_config=gdn_planner_config,
        )
    return planning_key


def prepare_megatron_context_parallel_state(
    *,
    micro: PackedTensors,
    topology: ParallelTopology,
    config: ContextParallelConfig,
    cp_group: Any,
    cp_rank: int,
    build_gdn_execution_spec: bool = False,
    gdn_planner_config: Any | None = None,
    block_mask_variants: tuple[CpBlockMaskVariant, ...] = (),
    target_device: torch.device | None = None,
) -> tuple[ArtContextParallelState, RankRuntimePlan, PackedBatchAttentionSpec, int]:
    """Build CP runtime state from CPU metadata.

    This is the portion of CP prepare that must stay free of CUDA reads so the
    training loop can run it after enqueueing backward for the previous
    microbatch. If device metadata reaches this function, scalar reads,
    cache-key hashing, and prefix-tree parsing can block the host on GPU work.
    """
    if int(topology.cp) <= 1:
        raise RuntimeError(
            "prepare_cp_micro is CP-only. Non-CP runs must bypass the context parallel dispatcher in train.py."
        )
    if int(micro["tokens"].shape[0]) != 1:
        raise RuntimeError(
            "ART context parallel currently supports exactly one packed sequence at a time, "
            f"got token batch={int(micro['tokens'].shape[0])}."
        )
    if int(micro["group_ids"].shape[0]) != 1:
        raise RuntimeError(
            "ART context parallel currently supports exactly one packed sequence at a time, "
            f"got batch={int(micro['group_ids'].shape[0])}."
        )
    input_pos_cpu = _planning_metadata_cpu(micro["input_pos"])
    planning_key, bundle, group_ids_cpu, parent_ids_cpu = _get_or_build_planning_bundle(
        group_ids=micro["group_ids"],
        parent_ids=micro["parent_ids"],
        topology=topology,
        config=config,
        original_seq_len=int(micro["tokens"].shape[1]),
        build_gdn_execution_spec=build_gdn_execution_spec,
    )
    rank_plan = _get_or_build_bundle_rank_plan(
        planning_key=planning_key,
        bundle=bundle,
        original_seq_len=int(micro["tokens"].shape[1]),
        target_rank=cp_rank,
        block_size=int(config.block_size),
    )
    gdn_execution_plan = None
    if build_gdn_execution_spec:
        _plan_gdn_rank_execution(
            planning_key=planning_key,
            bundle=bundle,
            topology=topology,
            cp_rank=cp_rank,
            gdn_planner_config=gdn_planner_config,
        )
        gdn_plan_device = (
            target_device if target_device is not None else micro["tokens"].device
        )
        gdn_execution_plan = _materialize_preplanned_gdn_rank_execution(
            planning_key=planning_key,
            cp_rank=cp_rank,
            gdn_planner_config=gdn_planner_config,
            device=gdn_plan_device,
        )
    pad_multiple = int(topology.tp) if bool(topology.sp) and int(topology.tp) > 1 else 1
    state = ArtContextParallelState(
        rank_plan=rank_plan,
        cp_group=cp_group,
        config=config,
        group_ids=group_ids_cpu[0].contiguous(),
        parent_ids=parent_ids_cpu[0].contiguous(),
        input_pos=input_pos_cpu[0].contiguous(),
        block_mask_variants=block_mask_variants,
        gdn_execution_spec=bundle.gdn_execution_spec,
        gdn_execution_plan=gdn_execution_plan,
        trace_token_uids=None,
    )
    return state, rank_plan, bundle.spec, pad_multiple


def dispatch_megatron_context_parallel_training_tensors(
    *,
    micro: PackedTensors,
    rank_plan: RankRuntimePlan,
    spec: PackedBatchAttentionSpec,
    pad_multiple: int,
    trace_token_uids: bool = False,
    target_device: torch.device | None = None,
    cp_group: Any | None = None,
    ref_logprobs: torch.Tensor | None = None,
) -> tuple[DispatchedPackedTensors, TrainingMicrobatchWorkload]:
    """Gather this rank's training tensors and optionally move them to device.

    Dispatch may enqueue H2D copies when `target_device` is CUDA, but it must
    not read CUDA metadata back to host. Padding control flow is therefore
    derived from rank-plan shape metadata, not from scalar CUDA tensor reads.
    """
    dispatch_meta_cache: dict[
        tuple[tuple[tuple[int, int], ...], int, str, int | None],
        tuple[torch.Tensor, torch.Tensor],
    ] = {}
    tokenized = "target_tokens" in micro
    assistant_mask = (
        micro["assistant_mask"]
        if tokenized
        else shift_tensor(micro["assistant_mask"], False)
    )
    labels = (
        micro["target_tokens"][..., 0]
        if tokenized
        else torch.where(
            assistant_mask,
            shift_tensor(micro["tokens"], -100),
            torch.full_like(micro["tokens"], -100),
        )
    )
    old_logprobs = shift_tensor(micro["logprobs"], float("nan"))
    advantages = shift_tensor(micro["advantages"], 0.0)
    weights = shift_tensor(micro["weights"], 0.0)
    shifted_group_ids = (
        micro["group_ids"] if tokenized else shift_tensor(micro["group_ids"], 0)
    )
    original_logprobs_source = cast(Any, micro).get("original_logprobs")
    original_logprobs = (
        None
        if original_logprobs_source is None
        else shift_tensor(original_logprobs_source, 0.0)
    )
    token_uids = (
        _build_token_uids(spec, seq_len=int(micro["tokens"].shape[1]))
        if trace_token_uids
        else None
    )

    def dispatch(
        tensor: torch.Tensor,
        pad_value: int | float | bool,
        *,
        move_to_target: bool = True,
    ) -> torch.Tensor:
        local = _dispatch_tensor(
            tensor,
            rank_plan=rank_plan,
            pad_value=pad_value,
            pad_multiple=pad_multiple,
            dispatch_meta_cache=dispatch_meta_cache,
        )
        return _to_target_device(local, target_device) if move_to_target else local

    def maybe_dispatch(
        tensor: torch.Tensor | None,
        pad_value: int | float | bool,
    ) -> torch.Tensor | None:
        return None if tensor is None else dispatch(tensor, pad_value)

    local_labels = dispatch(labels, -100, move_to_target=False)
    local_assistant_mask = dispatch(assistant_mask, False, move_to_target=False).to(
        dtype=torch.bool
    )
    lm_head_selection = LmHeadTokenSelection.from_labels(
        local_labels,
        target_device=target_device,
    )
    local_token_uids = (
        None if token_uids is None else dispatch(token_uids, -1, move_to_target=False)
    )
    tensors = DispatchedPackedTensors(
        tokens=dispatch(micro["tokens"], 0),
        labels=_to_target_device(local_labels, target_device),
        input_pos=dispatch(micro["input_pos"], 0),
        assistant_mask=_to_target_device(local_assistant_mask, target_device),
        group_ids=dispatch(shifted_group_ids, 0),
        old_logprobs=dispatch(old_logprobs, float("nan")),
        advantages=dispatch(advantages, 0.0),
        weights=dispatch(weights, 0.0),
        valid_lengths=rank_plan.local_valid_lengths,
        lm_head_selection=lm_head_selection,
        original_logprobs=maybe_dispatch(original_logprobs, 0.0),
        ref_logprobs=maybe_dispatch(ref_logprobs, float("nan")),
        loss_all_reduce_group=cp_group,
        token_uids=None if local_token_uids is None else local_token_uids.contiguous(),
        target_tokens=maybe_dispatch(micro.get("target_tokens"), -100),
        loss_weights=maybe_dispatch(micro.get("loss_weights"), 0.0),
        behavior_logprobs=maybe_dispatch(micro.get("behavior_logprobs"), 0.0),
        token_advantages=maybe_dispatch(micro.get("token_advantages"), 0.0),
    )
    workload = TrainingMicrobatchWorkload(
        logical_nonpadding_tokens=sum(rank_plan.local_valid_lengths),
        loss_bearing_tokens=int(local_assistant_mask.sum().item()),
        executed_token_equivalents=int(local_labels.numel()),
        nominal_schedule_capacity_tokens=rank_plan.original_seq_len,
    )
    return tensors, workload


def get_or_build_runtime_plan(
    spec: PackedBatchAttentionSpec,
    *,
    topology: ParallelTopology,
    config: ContextParallelConfig,
    original_seq_len: int,
) -> tuple[RankRuntimePlan, ...]:
    key = _runtime_plan_cache_key(
        spec,
        topology=topology,
        config=config,
        original_seq_len=original_seq_len,
    )
    cached = _RUNTIME_PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    plan = _build_runtime_plan(
        spec,
        topology=topology,
        config=config,
        original_seq_len=original_seq_len,
    )
    _cache_put(_RUNTIME_PLAN_CACHE, key, plan)
    return plan


def _runtime_plan_assignment(
    spec: PackedBatchAttentionSpec,
    *,
    topology: ParallelTopology,
    config: ContextParallelConfig,
) -> tuple[
    PackedRowAttentionSpec, tuple[TokenRange, ...], tuple[int, ...], tuple[int, ...]
]:
    cp_size = max(int(topology.cp), 1)
    if len(spec.rows) != 1:
        raise RuntimeError(
            "ART context parallel runtime planning expects exactly one packed sequence, "
            f"got {len(spec.rows)} rows."
        )
    row_spec = spec.rows[0]
    chunk_size = _normalized_chunk_size(
        valid_tokens=int(row_spec.valid_tokens),
        block_size=int(config.block_size),
        requested_chunk_size=int(config.planner_chunk_size),
        cp_size=cp_size,
        config=config,
    )
    chunk_ranges = _build_chunk_ranges(
        valid_tokens=int(row_spec.valid_tokens),
        chunk_size=chunk_size,
    )
    if len(chunk_ranges) < cp_size and int(row_spec.valid_tokens) >= cp_size:
        chunk_ranges = _build_chunk_ranges(
            valid_tokens=int(row_spec.valid_tokens),
            chunk_size=max(1, int(row_spec.valid_tokens) // cp_size),
        )
    pair_matrix, q_weights = _build_chunk_pair_program(
        row_spec,
        chunk_ranges=chunk_ranges,
    )
    owners, wave_assignment, _planner_eval = _search_chunk_assignment(
        row_spec=row_spec,
        chunk_ranges=chunk_ranges,
        pair_matrix=pair_matrix,
        q_weights=q_weights,
        cp_size=cp_size,
        config=config,
    )
    return row_spec, chunk_ranges, owners, wave_assignment


def _build_runtime_plan(
    spec: PackedBatchAttentionSpec,
    *,
    topology: ParallelTopology,
    config: ContextParallelConfig,
    original_seq_len: int,
) -> tuple[RankRuntimePlan, ...]:
    row_spec, chunk_ranges, owners, wave_assignment = _runtime_plan_assignment(
        spec,
        topology=topology,
        config=config,
    )
    cp_size = max(int(topology.cp), 1)
    token_layout_index = _build_runtime_token_layout_index(
        chunk_ranges=chunk_ranges,
        owners=owners,
        cp_size=cp_size,
    )
    return tuple(
        _build_rank_runtime_plan(
            row_spec=row_spec,
            chunk_ranges=chunk_ranges,
            owners=owners,
            wave_assignment=wave_assignment,
            token_layout_index=token_layout_index,
            cp_size=cp_size,
            original_seq_len=original_seq_len,
            target_rank=rank,
            block_size=int(config.block_size),
            config=config,
        )
        for rank in range(cp_size)
    )


def _build_runtime_token_layout_index(
    *,
    chunk_ranges: tuple[TokenRange, ...],
    owners: tuple[int, ...],
    cp_size: int,
) -> TokenLayoutIndex:
    ranges_by_rank: list[list[tuple[int, int, int]]] = [[] for _ in range(cp_size)]
    rank_positions = [0 for _ in range(cp_size)]
    for chunk_range, owner in zip(chunk_ranges, owners, strict=True):
        rank = int(owner)
        position = rank_positions[rank]
        ranges_by_rank[rank].append(
            (int(chunk_range.start), int(chunk_range.end), position)
        )
        rank_positions[rank] += int(chunk_range.size())
    return TokenLayoutIndex(
        ownership_ranges_by_rank=tuple(tuple(ranges) for ranges in ranges_by_rank),
        token_counts_by_rank=tuple(rank_positions),
    )


def _row_signature(row_spec: PackedRowAttentionSpec) -> str:
    payload = {
        "valid_tokens": row_spec.valid_tokens,
        "slices": [_attn_slice_payload(slice_) for slice_ in row_spec.slices],
    }
    return json.dumps(payload, sort_keys=True)


def _runtime_plan_cache_key(
    spec: PackedBatchAttentionSpec,
    *,
    topology: ParallelTopology,
    config: ContextParallelConfig,
    original_seq_len: int,
) -> str:
    return _json_cache_key(
        {
            "topology": _dataclass_payload(topology),
            "config": _dataclass_payload(config),
            "row_signatures": tuple(_row_signature(row) for row in spec.rows),
            "original_seq_len": int(original_seq_len),
        }
    )


def _dataclass_payload(value: Any) -> dict[str, Any]:
    return {
        key: (item.model_dump(mode="json") if isinstance(item, BaseModel) else item)
        for key, item in value.__dict__.items()
    }


def _attn_slice_payload(slice_: AttnSlice) -> dict[str, Any]:
    return {
        "q_range": _dataclass_payload(slice_.q_range),
        "k_range": _dataclass_payload(slice_.k_range),
        "mask_kind": slice_.mask_kind.value,
        "row_index": slice_.row_index,
        "family_index": slice_.family_index,
    }


def _range_key(range_: TokenRange) -> tuple[int, int]:
    return (int(range_.start), int(range_.end))


def _set_stage_token_indices(
    *,
    target_indices: torch.Tensor,
    stage_range: TokenRange,
    source_range: TokenRange,
    source_indices: torch.Tensor,
) -> None:
    if stage_range.size() != source_range.size():
        raise RuntimeError(
            "Stage-local and packed-sequence token ranges must have matched sizes, got "
            f"{stage_range} vs {source_range}"
        )

    current_indices = target_indices[stage_range.start : stage_range.end]
    if not bool(
        torch.logical_or(current_indices == -1, current_indices == source_indices)
        .all()
        .item()
    ):
        mismatch = torch.nonzero(
            torch.logical_not(
                torch.logical_or(
                    current_indices == -1, current_indices == source_indices
                )
            ),
            as_tuple=False,
        ).flatten()
        mismatch_offset = int(mismatch[0].item())
        mismatch_index = int(stage_range.start) + mismatch_offset
        raise RuntimeError(
            "Stage mask token index mismatch at stage index "
            f"{mismatch_index}: {int(current_indices[mismatch_offset].item())} vs "
            f"{int(source_indices[mismatch_offset].item())}"
        )
    current_indices.copy_(source_indices)


def _resolve_stage_mask_kind(
    *,
    mask_kind: AttnMaskKind,
    q_piece: TokenRange,
    k_piece: TokenRange,
) -> AttnMaskKind | None:
    if mask_kind is AttnMaskKind.FULL:
        return AttnMaskKind.FULL
    if k_piece.start >= q_piece.end:
        return None
    if k_piece.end <= q_piece.start:
        return AttnMaskKind.FULL
    return AttnMaskKind.CAUSAL


def _merge_ranges(ranges: list[TokenRange]) -> tuple[TokenRange, ...]:
    if not ranges:
        return tuple()
    sorted_ranges = sorted(ranges, key=lambda range_: (range_.start, range_.end))
    merged: list[TokenRange] = [sorted_ranges[0]]
    for range_ in sorted_ranges[1:]:
        last = merged[-1]
        if range_.start <= last.end:
            merged[-1] = TokenRange(start=last.start, end=max(last.end, range_.end))
            continue
        merged.append(range_)
    return tuple(merged)


def _remap_subrange(
    subrange: TokenRange,
    merged_ranges: tuple[TokenRange, ...],
) -> TokenRange:
    stage_offset = 0
    for merged_range in merged_ranges:
        if subrange.start >= merged_range.start and subrange.end <= merged_range.end:
            return TokenRange(
                start=stage_offset + subrange.start - merged_range.start,
                end=stage_offset + subrange.end - merged_range.start,
            )
        stage_offset += merged_range.size()
    raise RuntimeError(
        "Failed to remap subrange into merged ranges: "
        f"subrange={subrange}, merged_ranges={merged_ranges}"
    )


def _unmap_stage_subrange(
    subrange: TokenRange,
    merged_ranges: tuple[TokenRange, ...],
) -> TokenRange:
    stage_offset = 0
    for merged_range in merged_ranges:
        stage_end = stage_offset + merged_range.size()
        if subrange.start >= stage_offset and subrange.end <= stage_end:
            return TokenRange(
                start=merged_range.start + subrange.start - stage_offset,
                end=merged_range.start + subrange.end - stage_offset,
            )
        stage_offset = stage_end
    raise RuntimeError(
        "Failed to map stage subrange into global ranges: "
        f"subrange={subrange}, merged_ranges={merged_ranges}"
    )


def _tensor_sha1(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().contiguous().to(device="cpu", dtype=torch.int64)
    return hashlib.sha1(cpu_tensor.numpy().tobytes()).hexdigest()


def _exact_mask_metadata_cache_key(
    *,
    q_token_indices: torch.Tensor,
    k_token_indices: torch.Tensor,
) -> str:
    return json.dumps(
        {
            "q_token_indices_sha1": _tensor_sha1(q_token_indices),
            "k_token_indices_sha1": _tensor_sha1(k_token_indices),
            "q_len": int(q_token_indices.numel()),
            "k_len": int(k_token_indices.numel()),
        },
        sort_keys=True,
    )


def _build_token_uids(
    spec: PackedBatchAttentionSpec,
    *,
    seq_len: int,
) -> torch.Tensor:
    tensor = torch.full((len(spec.rows), seq_len), fill_value=-1, dtype=torch.int64)
    cursor = 0
    for row_index, row_spec in enumerate(spec.rows):
        if row_spec.valid_tokens <= 0:
            continue
        tensor[row_index, : row_spec.valid_tokens] = torch.arange(
            cursor,
            cursor + row_spec.valid_tokens,
            dtype=torch.int64,
        )
        cursor += row_spec.valid_tokens
    return tensor


def _to_target_device(
    tensor: torch.Tensor,
    target_device: torch.device | None,
) -> torch.Tensor:
    """Move dispatched local tensors without adding host-side device reads."""
    if target_device is None or tensor.device == target_device:
        return tensor
    return tensor.to(device=target_device, non_blocking=True)


def _dispatch_tensor(
    tensor: torch.Tensor,
    *,
    rank_plan: RankRuntimePlan,
    pad_value: int | float | bool,
    pad_multiple: int = 1,
    dispatch_meta_cache: (
        dict[
            tuple[tuple[tuple[int, int], ...], int, str, int | None],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None
    ) = None,
) -> torch.Tensor:
    """Gather local rows without branching on CUDA tensor values.

    The old `bool(valid_mask.all())` branch synchronized whenever dispatch ran
    on CUDA. Use the rank plan's Python length metadata to decide whether a pad
    mask is needed.
    """
    if tensor.ndim not in {2, 3}:
        raise RuntimeError(
            "_dispatch_tensor expected a rank-2 or rank-3 tensor, "
            f"got shape {tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) != 1:
        raise RuntimeError(
            "ART context parallel dispatch expects exactly one packed sequence, "
            f"got tensor batch={int(tensor.shape[0])}."
        )
    if len(rank_plan.local_valid_lengths) != 1:
        raise RuntimeError(
            "ART context parallel dispatch expects exactly one packed local sequence length, "
            f"got local_valid_lengths={len(rank_plan.local_valid_lengths)}."
        )
    max_local_len = max(int(rank_plan.local_valid_lengths[0]), 1)
    if pad_multiple > 1 and max_local_len % pad_multiple != 0:
        max_local_len = (
            (max_local_len + pad_multiple - 1) // pad_multiple
        ) * pad_multiple
    gather_index, valid_mask = _dispatch_meta(
        rank_plan=rank_plan,
        max_local_len=max_local_len,
        device=tensor.device,
        dispatch_meta_cache=dispatch_meta_cache,
    )
    if tensor.ndim == 3:
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, tensor.shape[2])
        valid_mask = valid_mask.unsqueeze(-1)
    output = torch.gather(tensor, dim=1, index=gather_index)
    if int(rank_plan.local_valid_lengths[0]) < max_local_len:
        output = output.masked_fill(~valid_mask, pad_value)
    return output


def _dispatch_meta(
    *,
    rank_plan: RankRuntimePlan,
    max_local_len: int,
    device: torch.device,
    dispatch_meta_cache: (
        dict[
            tuple[tuple[tuple[int, int], ...], int, str, int | None],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None
    ) = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    owner_ranges = tuple(
        range_
        for range_ in rank_plan.local_row_ranges
        if isinstance(range_, TokenRange) and range_.size() > 0
    )
    key = (
        tuple((range_.start, range_.end) for range_ in owner_ranges),
        max_local_len,
        device.type,
        device.index,
    )
    if dispatch_meta_cache is not None:
        cached = dispatch_meta_cache.get(key)
        if cached is not None:
            return cached

    flat_indices_parts = [
        torch.arange(range_.start, range_.end, device=device, dtype=torch.int64)
        for range_ in owner_ranges
    ]
    flat_indices = (
        torch.cat(flat_indices_parts, dim=0)
        if flat_indices_parts
        else torch.empty((0,), device=device, dtype=torch.int64)
    )
    valid_count = int(flat_indices.numel())
    if valid_count < max_local_len:
        gather_index = torch.zeros((max_local_len,), device=device, dtype=torch.int64)
        if valid_count > 0:
            gather_index[:valid_count] = flat_indices
    else:
        gather_index = flat_indices[:max_local_len].contiguous()
    valid_mask = torch.zeros((max_local_len,), device=device, dtype=torch.bool)
    if valid_count > 0:
        valid_mask[: min(valid_count, max_local_len)] = True
    gather_index = gather_index.unsqueeze(0)
    valid_mask = valid_mask.unsqueeze(0)
    cached = (gather_index, valid_mask)
    if dispatch_meta_cache is not None:
        dispatch_meta_cache[key] = cached
    return cached
