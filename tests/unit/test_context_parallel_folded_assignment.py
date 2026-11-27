"""Unit tests for folded causal-attention CP chunk assignment."""

from __future__ import annotations

import dataclasses

import pytest
import torch

pytest.importorskip("megatron.core")
from art.megatron.context_parallel import runtime as rt  # noqa: E402
from art.megatron.context_parallel.builder import (  # noqa: E402
    build_prefix_tree_attention_spec,
)
from art.megatron.context_parallel.types import (  # noqa: E402
    AttnMaskKind,
    ContextParallelConfig,
    ContextParallelStageWorkProfile,
    ContextParallelWorkloadProfile,
    ParallelTopology,
    estimate_owned_token_ms,
)
from art.trainer_rank import _memory as m  # noqa: E402


def _single_seq(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    meta = torch.ones((1, length), dtype=torch.int64)
    return meta, meta


def _packed_seqs(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    groups = [
        torch.full((length,), idx + 1, dtype=torch.int64)
        for idx, length in enumerate(lengths)
    ]
    return torch.cat(groups).unsqueeze(0), torch.cat(groups).unsqueeze(0)


def _radix_tree(
    root_len: int, leaf_lens: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = [torch.full((root_len,), 1, dtype=torch.int64)]
    parents = [torch.full((root_len,), 1, dtype=torch.int64)]
    for idx, length in enumerate(leaf_lens):
        groups.append(torch.full((length,), idx + 2, dtype=torch.int64))
        parents.append(torch.full((length,), 1, dtype=torch.int64))
    return torch.cat(groups).unsqueeze(0), torch.cat(parents).unsqueeze(0)


def _plan_assignment(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    cp_size: int,
    config: ContextParallelConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    spec = build_prefix_tree_attention_spec(group_ids=group_ids, parent_ids=parent_ids)
    _row_spec, chunk_ranges, owners, waves = rt._runtime_plan_assignment(
        spec,
        topology=ParallelTopology(cp=cp_size),
        config=config,
    )
    tokens_by_rank = tuple(
        sum(
            int(r.size())
            for r, owner in zip(chunk_ranges, owners, strict=True)
            if owner == rank
        )
        for rank in range(cp_size)
    )
    range_counts = rt._ownership_range_counts(owners, cp_size=cp_size)
    wave_count = max(waves) + 1 if waves else 0
    return tokens_by_rank, range_counts, wave_count


@pytest.mark.parametrize(
    ("cp_size", "seq_len", "owned_ms"),
    [
        (
            2,
            131_072,
            estimate_owned_token_ms(
                hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=1
            ),
        ),
        (
            4,
            131_072,
            estimate_owned_token_ms(
                hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=2
            ),
        ),
        (
            4,
            196_608,
            estimate_owned_token_ms(
                hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=2
            ),
        ),
        (
            4,
            262_144,
            estimate_owned_token_ms(
                hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=2
            ),
        ),
        (
            8,
            131_072,
            estimate_owned_token_ms(
                hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=1
            ),
        ),
        (
            8,
            196_608,
            estimate_owned_token_ms(
                hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=1
            ),
        ),
        (
            8,
            262_144,
            estimate_owned_token_ms(
                hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=1
            ),
        ),
        (
            4,
            131_072,
            estimate_owned_token_ms(
                hidden_size=2048,
                ffn_hidden_size=0,
                moe_topk=8,
                moe_ffn_hidden_size=512,
                moe_shared_expert_ffn=512,
                tensor_parallel_size=1,
                expert_parallel_size=4,
            ),
        ),
        (
            8,
            262_144,
            estimate_owned_token_ms(
                hidden_size=2048,
                ffn_hidden_size=0,
                moe_topk=8,
                moe_ffn_hidden_size=512,
                moe_shared_expert_ffn=512,
                tensor_parallel_size=1,
                expert_parallel_size=8,
            ),
        ),
    ],
)
def test_folded_assignment_balances_long_causal_sequences(
    cp_size: int, seq_len: int, owned_ms: float
) -> None:
    """Long single causal sequences fold cleanly to ~1.0x token balance and <=2 ranges."""
    group_ids, parent_ids = _single_seq(seq_len)
    config = ContextParallelConfig(planner_owned_token_ms=owned_ms)
    tokens_by_rank, range_counts, wave_count = _plan_assignment(
        group_ids, parent_ids, cp_size=cp_size, config=config
    )
    assert sum(tokens_by_rank) == seq_len
    assert min(tokens_by_rank) > 0
    assert max(tokens_by_rank) / min(tokens_by_rank) <= 1.05
    assert max(range_counts) == 2
    assert range_counts[-1] == 1
    assert wave_count == 1


def test_folded_assignment_handles_packed_multi_sequence_batches() -> None:
    """Packed multi-sequence batches fold each long causal ramp independently."""
    owned_27b_tp1 = estimate_owned_token_ms(
        hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=1
    )
    owned_27b_tp2 = estimate_owned_token_ms(
        hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=2
    )

    # 2x131K on CP=8: each sequence folds independently across its 4 ranks.
    toks_8, ranges_8, waves_8 = _plan_assignment(
        *_packed_seqs([131_072, 131_072]),
        cp_size=8,
        config=ContextParallelConfig(planner_owned_token_ms=owned_27b_tp1),
    )
    assert toks_8 == (32_768,) * 8
    assert ranges_8 == (2, 2, 2, 1, 2, 2, 2, 1)
    assert waves_8 == 1

    # 2x131K on CP=4: each sequence folds independently across its 2 ranks.
    toks_4, ranges_4, waves_4 = _plan_assignment(
        *_packed_seqs([131_072, 131_072]),
        cp_size=4,
        config=ContextParallelConfig(planner_owned_token_ms=owned_27b_tp2),
    )
    assert sum(toks_4) == 262_144
    assert max(toks_4) / min(toks_4) <= 1.05
    assert ranges_4 == (2, 1, 2, 1)
    assert waves_4 == 1

    # Already-balanced 8x32K on CP=8 stays strictly contiguous (zero fragmentation).
    toks_bal, ranges_bal, _ = _plan_assignment(
        *_packed_seqs([32_768] * 8),
        cp_size=8,
        config=ContextParallelConfig(planner_owned_token_ms=owned_27b_tp1),
    )
    assert toks_bal == (32_768,) * 8
    assert ranges_bal == (1,) * 8

    # Heterogeneous 1x196K + 1x65K on CP=8 folds only the 196K ramp ranks.
    g_het, p_het = _packed_seqs([196_608, 65_536])
    spec = build_prefix_tree_attention_spec(group_ids=g_het, parent_ids=p_het)
    cfg = ContextParallelConfig(planner_owned_token_ms=owned_27b_tp1)
    toks_het, ranges_het, _ = _plan_assignment(g_het, p_het, cp_size=8, config=cfg)
    assert sum(toks_het) == 262_144
    assert max(ranges_het) <= 2
    assert ranges_het[-1] == 1
    # Verify peak rank load drops below the contiguous baseline (~60,416).
    assert max(toks_het) < 56_000
    assert spec.rows[0].valid_tokens == 262_144


def test_folded_assignment_handles_radix_trees_without_branch_bleed() -> None:
    """Shared-prefix radix trees fold only the causal ramp ranks and preserve leaf ranks."""
    owned_27b_tp1 = estimate_owned_token_ms(
        hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=1
    )
    cfg = ContextParallelConfig(planner_owned_token_ms=owned_27b_tp1)

    # root 128K + 8x16K leaves on CP=8: root ranks fold, leaf ranks stay contiguous.
    toks, ranges, waves = _plan_assignment(
        *_radix_tree(131_072, [16_384] * 8), cp_size=8, config=cfg
    )
    assert sum(toks) == 262_144
    assert max(ranges) == 2
    assert ranges[-4:] == (1, 1, 1, 1)
    assert max(toks) < 50_000  # contiguous Rank 0 is ~67,584
    assert waves == 1

    # root 128K + 2x64K leaves on CP=8: root+leaf0 ramp and leaf1 ramp fold independently.
    toks_2, ranges_2, _ = _plan_assignment(
        *_radix_tree(131_072, [65_536, 65_536]), cp_size=8, config=cfg
    )
    assert sum(toks_2) == 262_144
    assert max(ranges_2) == 2
    assert max(toks_2) < 42_000  # contiguous Rank 0 is ~71,680


def test_folded_assignment_respects_range_limit_and_short_sequence_floor() -> None:
    """Short sequences and max_ownership_ranges_per_rank=1 profiles stay contiguous."""
    owned_27b_tp1 = estimate_owned_token_ms(
        hidden_size=5120, ffn_hidden_size=17408, tensor_parallel_size=1
    )
    # Short sequence (16K on CP=8): slab < planner_remote_stage_token_floor -> contiguous.
    _toks_s, ranges_s, _ = _plan_assignment(
        *_single_seq(16_384),
        cp_size=8,
        config=ContextParallelConfig(planner_owned_token_ms=owned_27b_tp1),
    )
    assert ranges_s == (1,) * 8

    # Profile with max_ownership_ranges_per_rank=1 disables folding even at 262K.
    profile_one_range = ContextParallelWorkloadProfile(
        stages=(
            ContextParallelStageWorkProfile(
                physical_pipeline_rank=0,
                query_flops_per_token=1,
                tile_pair_flops=1,
                k_hbm_bytes_per_token=1,
                k_fetch_bytes_per_token=1,
                dkv_reduce_bytes_per_token=1,
                query_memory_bytes_per_token=1,
                k_memory_bytes_per_token=1,
            ),
        ),
        query_tile_size=128,
        key_tile_size=128,
        indexer_score_workspace_elements=16_384,
        indexer_max_k_tokens=4096,
        max_ownership_ranges_per_rank=1,
    )
    cfg_cap = ContextParallelConfig(
        planner_owned_token_ms=owned_27b_tp1,
        workload_profile=profile_one_range,
    )
    g_262k, p_262k = _single_seq(262_144)
    spec = build_prefix_tree_attention_spec(group_ids=g_262k, parent_ids=p_262k)
    row_spec = spec.rows[0]
    chunk_size = rt._normalized_chunk_size(
        valid_tokens=row_spec.valid_tokens,
        block_size=cfg_cap.block_size,
        requested_chunk_size=cfg_cap.planner_chunk_size,
        cp_size=8,
        config=cfg_cap,
    )
    chunk_ranges = rt._build_chunk_ranges(
        valid_tokens=row_spec.valid_tokens, chunk_size=chunk_size
    )
    pair_matrix, q_weights = rt._build_chunk_pair_program(
        row_spec, chunk_ranges=chunk_ranges
    )
    owners, _waves, _ev = rt._search_generic_chunk_assignment(
        chunk_ranges=chunk_ranges,
        pair_matrix=pair_matrix,
        q_weights=q_weights,
        cp_size=8,
        config=cfg_cap,
    )
    assert rt._ownership_range_counts(owners, cp_size=8) == (1,) * 8


def _expected_attention_pairs(spec_row) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for sl in spec_row.slices:
        for q in range(sl.q_range.start, sl.q_range.end):
            k_hi = (
                sl.k_range.end
                if sl.mask_kind is AttnMaskKind.FULL
                else min(sl.k_range.end, q + 1)
            )
            for k in range(sl.k_range.start, k_hi):
                pairs.add((q, k))
    return pairs


def _executed_attention_pairs(plans) -> list[tuple[int, int]]:
    executed: list[tuple[int, int]] = []
    for plan in plans:
        for stage in plan.stage_plans:
            if not stage.slices:
                continue
            assert stage.mask_metadata is not None
            q_idx = stage.mask_metadata.q_token_indices.tolist()
            k_idx = stage.mask_metadata.k_token_indices.tolist()
            for sl in stage.slices:
                for q_local in range(sl.q_range.start, sl.q_range.end):
                    q_global = q_idx[q_local]
                    assert q_global >= 0
                    for k_local in range(sl.k_range.start, sl.k_range.end):
                        if sl.mask_kind is AttnMaskKind.CAUSAL and k_local > q_local:
                            continue
                        k_global = k_idx[k_local]
                        assert k_global >= 0
                        executed.append((q_global, k_global))
    return executed


@pytest.mark.parametrize(
    ("name", "group_ids", "parent_ids", "cp_size"),
    [
        ("single_cp4", *_single_seq(128), 4),
        ("single_cp8", *_single_seq(256), 8),
        ("packed_2seq_cp4", *_packed_seqs([64, 64]), 4),
        ("radix_tree_cp4", *_radix_tree(64, [32, 32]), 4),
    ],
)
def test_folded_runtime_plan_exact_mask_and_comm_symmetry(
    name: str,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    cp_size: int,
) -> None:
    """Folded runtime plans execute every valid (q, k) pair exactly once with symmetric comm."""
    del name
    spec = build_prefix_tree_attention_spec(group_ids=group_ids, parent_ids=parent_ids)
    # Use small block/chunk sizes and zero fixed/overhead terms to trigger folding
    # at small sequence lengths where every (q, k) pair can be checked exhaustively.
    default_cfg = ContextParallelConfig()
    cfg = ContextParallelConfig(
        block_size=4,
        planner_chunk_size=4,
        planner_remote_stage_token_floor=0,
        planner_remote_stage_pair_floor=0,
        planner_remote_stage_host_ms=0.0,
        planner_stage_overhead_ms=0.0,
        planner_comm_stage_overhead_ms=0.0,
        planner_interval_overhead_ms=0.0,
        planner_merge_q_token_ms=0.0,
        planner_fetch_token_ms=0.0,
        planner_reduce_token_ms=0.0,
        planner_remote_pair_ms=default_cfg.planner_local_pair_ms,
        planner_remote_backward_pair_ms=default_cfg.planner_local_backward_pair_ms,
        planner_owned_token_ms=0.0,
    )
    plans = rt._build_runtime_plan(
        spec,
        topology=ParallelTopology(cp=cp_size),
        config=cfg,
        original_seq_len=int(group_ids.shape[1]),
    )
    assert any(len(plan.local_row_ranges) == 2 for plan in plans)

    # 1. Exact disjoint partition of all tokens across ranks.
    owned_tokens: list[int] = []
    for plan in plans:
        for rng in plan.local_row_ranges:
            if rng is not None:
                owned_tokens.extend(range(rng.start, rng.end))
    assert sorted(owned_tokens) == list(range(spec.rows[0].valid_tokens))

    # 2. Exact (q, k) attention pair equivalence (no missing, no duplicate, no causal leak).
    expected = _expected_attention_pairs(spec.rows[0])
    executed = _executed_attention_pairs(plans)
    assert len(executed) == len(set(executed)), "Duplicate (q, k) pair executed!"
    assert set(executed) == expected

    # 3. Exact peer-to-peer send/recv symmetry across all remote stages and aggregate dKV reduce.
    stage_count = len(plans[0].stage_plans)
    for stage_idx in range(1, stage_count):
        for src in range(cp_size):
            src_fetch = plans[src].stage_plans[stage_idx].kv_fetch_plan
            assert src_fetch is not None
            for dst in range(cp_size):
                dst_fetch = plans[dst].stage_plans[stage_idx].kv_fetch_plan
                assert dst_fetch is not None
                assert src_fetch.send_splits[dst] == dst_fetch.recv_splits[src]
    for src in range(cp_size):
        for dst in range(cp_size):
            assert (
                plans[src].remote_dkv_reduce_plan.send_splits[dst]
                == plans[dst].remote_dkv_reduce_plan.recv_splits[src]
            )


@pytest.mark.parametrize("cp_size", (4, 8))
def test_folded_cp_preserves_memory_workload_coverage_monotonicity(
    cp_size: int,
) -> None:
    """Warm ladders cover both contiguous (<=32K) and folded (>=64K) plans monotonically."""
    owned_ms = estimate_owned_token_ms(
        hidden_size=5120,
        ffn_hidden_size=17408,
        tensor_parallel_size=2 if cp_size == 4 else 1,
    )
    cfg = ContextParallelConfig(planner_owned_token_ms=owned_ms)
    topo = ParallelTopology(cp=cp_size)

    def workload(length: int):
        g, p = _single_seq(length)
        return rt.context_parallel_memory_workload(
            group_ids=g,
            parent_ids=p,
            topology=topo,
            config=cfg,
            original_seq_len=length,
            build_gdn_execution_spec=False,
        )

    # Folded CP cuts rank model token counts to ~N / cp_size at 131K and 262K.
    g_262k, p_262k = _single_seq(262_144)
    counts_262k = rt.context_parallel_rank_model_token_counts(
        group_ids=g_262k,
        parent_ids=p_262k,
        topology=topo,
        config=cfg,
        original_seq_len=262_144,
        build_gdn_execution_spec=False,
    )
    assert max(counts_262k) <= int((262_144 / cp_size) * 1.05)

    # Standard short warm ladder (128, 8192, 32768) covers all <=32K lengths.
    short_peaks = tuple(
        (length, length * 1024, *workload(length)) for length in (128, 8192, 32_768)
    )
    short_profile = m._CallerMemoryProfile(
        bytes_per_token=1024,
        packed_tokens=32_768,
        caller_cp_peaks=short_peaks,
    )
    for query in (8500, 9216, 12_000, 16_384, 24_576):
        assert short_profile.covering_cp_peak(query, workload(query)) is not None

    # Extended ladder including 131K and 262K covers intermediate folded lengths.
    full_peaks = short_peaks + tuple(
        (length, length * 1024, *workload(length)) for length in (131_072, 262_144)
    )
    full_profile = dataclasses.replace(
        short_profile,
        packed_tokens=262_144,
        caller_cp_peaks=full_peaks,
    )
    for query in (65_536, 98_304, 196_608, 250_000):
        assert full_profile.covering_cp_peak(query, workload(query)) is not None
