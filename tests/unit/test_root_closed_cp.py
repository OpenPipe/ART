from dataclasses import replace
import random

import pytest
import torch

from art.megatron.context_parallel.root_closed import root_owner_spans
from art.megatron.context_parallel.runtime import _get_or_build_planning_bundle
from art.megatron.context_parallel.types import ContextParallelConfig, ParallelTopology
from art.megatron.gdn.gdn_prefix_tree import (
    GdnPlannerConfig,
    build_gdn_global_execution_decision,
    materialize_gdn_rank_execution_plan,
)


def plan(lengths, ranks, cp):
    ids = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))[
        None
    ]
    placements, start = [], 0
    for length, coordinates in zip(lengths, ranks):
        rank, span = coordinates[:2]
        placements.append((start, start + length, rank, span, *coordinates[2:]))
        start += length
    _, bundle, *_ = _get_or_build_planning_bundle(
        group_ids=ids,
        parent_ids=ids,
        topology=ParallelTopology(cp=cp),
        config=ContextParallelConfig(root_placements=tuple(placements)),
        original_seq_len=start,
        build_gdn_execution_spec=True,
    )
    assert bundle.gdn_execution_spec is not None
    decision = build_gdn_global_execution_decision(
        bundle.gdn_execution_spec,
        cp_size=cp,
        attention_token_layout_index=bundle.token_layout_index,
        planner_config=GdnPlannerConfig(root_closed=True),
    )
    return bundle, decision


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_subset_keeps_root_owners_and_gdn_decisions(cp):
    rng = random.Random(314)
    lengths = [1024, 32768, 2048, 65536, 16384, 4096]
    ranks = []
    for span in (1, min(2, cp), 1, cp, 1, 1):
        ranks.append((rng.randrange(cp - span + 1), span))
    full, full_d = plan(lengths, ranks, cp)
    offset = 0
    for i, length in enumerate(lengths):
        one, one_d = plan([length], [ranks[i]], cp)
        assert one_d.chained_nodes == (full_d.chained_nodes[i],)
        assert one_d.owner_by_node == (full_d.owner_by_node[i],)
        for r in range(cp):
            original = [
                (max(a, offset) - offset, min(b, offset + length) - offset)
                for a, b, _ in full_d.gdn_ranges_by_rank_by_position[r]
                if a < offset + length and b > offset
            ]
            isolated = [(a, b) for a, b, _ in one_d.gdn_ranges_by_rank_by_position[r]]
            assert original == isolated
        offset += length
    for r in range(cp):
        p = materialize_gdn_rank_execution_plan(
            full.gdn_execution_spec, full_d, device="cpu", cp_rank=r
        )
        for buckets in p.tree_chain_buckets_by_depth:
            for bucket in buckets:
                assert bucket.lengths_by_rank_cpu is not None
                assert torch.all(
                    torch.cumsum(bucket.lengths_by_rank_cpu, 0)[:-1] % 64 == 0
                )
    assert sum(full_d.gdn_token_counts_by_rank) == sum(lengths)


def test_root_alignment():
    assert root_owner_spans(65536, 4) == (
        (0, 16384, 0),
        (16384, 32768, 1),
        (32768, 49152, 2),
        (49152, 65536, 3),
    )
    with pytest.raises(ValueError):
        root_owner_spans(1001, 2)
    with pytest.raises(ValueError):
        plan([1025], [(0, 1)], 2)


def test_tree_boundary_cannot_be_cut():
    from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
    from art.megatron.context_parallel.root_closed import root_closed_assignment

    ids = torch.tensor([[0] * 128 + [1] * 128])
    parent = torch.tensor([[0] * 256])
    spec = build_prefix_tree_attention_spec(group_ids=ids, parent_ids=parent)
    with pytest.raises(ValueError, match="dependency"):
        root_closed_assignment(
            row_spec=spec.rows[0],
            placements=((0, 128, 0, 1), (128, 256, 1, 1)),
            cp_size=2,
            block_size=128,
        )


def test_fla_alignment_ignores_empty_trailing_ranks():
    from art.megatron.gdn.fla_cp import _fla_chunk_boundaries_aligned_cpu

    for lengths in ([[8192], [8195], [0], [0]], [[0], [8192], [8195], [0]]):
        assert not _fla_chunk_boundaries_aligned_cpu(torch.tensor(lengths))
        assert _fla_chunk_boundaries_aligned_cpu(
            torch.tensor(lengths), allow_empty_shards=True
        )
    assert not _fla_chunk_boundaries_aligned_cpu(
        torch.tensor([[8195], [8192], [0], [0]]), allow_empty_shards=True
    )


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_sparse_root_communication_is_reciprocal(cp):
    from art.megatron.context_parallel.runtime import _build_runtime_plan

    bundle, _ = plan([256, 512, 128], [(0, 1), (0, 2), (1, 1)], cp)
    plans = _build_runtime_plan(
        bundle.spec,
        topology=ParallelTopology(cp=cp),
        config=ContextParallelConfig(
            root_placements=((0, 256, 0, 1), (256, 768, 0, 2), (768, 896, 1, 1))
        ),
        original_seq_len=896,
    )
    counts = bundle.token_layout_index.token_counts_by_rank
    assert sum(counts) == 896
    assert all(n == 0 for n in counts[2:])
    for rank, p in enumerate(plans):
        for stage in p.stage_plans[1:]:
            fetch = stage.kv_fetch_plan
            reverse = stage.dkv_reduce_plan
            assert fetch is not None and reverse is not None
            for peer in range(cp):
                peer_fetch = plans[peer].stage_plans[stage.stage_index].kv_fetch_plan
                assert peer_fetch is not None
                assert fetch.send_splits[peer] == peer_fetch.recv_splits[rank]
                assert reverse.send_splits[peer] == fetch.recv_splits[peer]
                assert reverse.recv_splits[peer] == fetch.send_splits[peer]


def test_reversed_attention_spans_keep_canonical_gdn_chains():
    bundle, decision = plan([32768, 32768], [(0, 2, False), (0, 2, True)], 2)
    assert bundle.owners == (0, 1, 1, 0)
    assert decision.chained_nodes == (True, True)
    assert decision.cross_rank_token_count == 32768
    assert decision.gdn_token_counts_by_rank == (32768, 32768)
    _, alone = plan([32768], [(0, 2, True)], 2)
    for rank in range(2):
        in_wave = [
            (max(a, 32768) - 32768, b - 32768)
            for a, b, _ in decision.gdn_ranges_by_rank_by_position[rank]
            if b > 32768
        ]
        assert in_wave == [
            (a, b) for a, b, _ in alone.gdn_ranges_by_rank_by_position[rank]
        ]


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_reordered_subsets_preserve_each_roots_gdn_ranges(cp):
    rng = random.Random(517)
    lengths = [128, 512, 32768, 65536]
    ranks = [(0, 1, False), (cp - 1, 1, False), (0, 2, True), (0, cp, False)]
    _, full = plan(lengths, ranks, cp)
    full_offsets = [sum(lengths[:i]) for i in range(len(lengths))]
    for _ in range(12):
        order = rng.sample(range(len(lengths)), rng.randrange(1, len(lengths) + 1))
        _, subset = plan([lengths[i] for i in order], [ranks[i] for i in order], cp)
        offset = 0
        for j, i in enumerate(order):
            assert subset.owner_by_node[j] == full.owner_by_node[i]
            assert subset.chained_nodes[j] == full.chained_nodes[i]
            for rank in range(cp):

                def relative(decision, start):
                    return [
                        (max(a, start) - start, min(b, start + lengths[i]) - start)
                        for a, b, _ in decision.gdn_ranges_by_rank_by_position[rank]
                        if a < start + lengths[i] and b > start
                    ]

                assert relative(subset, offset) == relative(full, full_offsets[i])
            offset += lengths[i]


@pytest.mark.parametrize("multi", [False, True])
def test_empty_chain_identity_state_and_gradient(monkeypatch, multi):
    from art.megatron.gdn import fla_cp as f

    n = 2 if multi else 1
    state = torch.randn(n, 2, 4, 3, requires_grad=True)
    before = state.detach().clone()
    summary = f._identity_summary(state, multi=multi)
    torch.testing.assert_close(f._scan_fwd_initial_state(summary, state), state)
    prefix = summary.clone()
    prefix[..., 3:] *= 2
    prefix[..., :3] = 0.25
    monkeypatch.setattr(f, "_prefix_summary_exclusive", lambda *args: prefix)

    def broadcast(tensor, group):
        tensor.add_(5)
        return tensor

    monkeypatch.setattr(f, "_broadcast_chain_final_state", broadcast)
    monkeypatch.setattr(
        f,
        "_external_final_state_grad",
        lambda grad, reference, **kwargs: (
            grad if grad is not None else torch.zeros_like(reference)
        ),
    )
    full = summary.clone()
    full[..., 3:] *= 2
    monkeypatch.setattr(
        f, "_suffix_summary_exclusive_and_full", lambda *args: (None, full)
    )
    q = torch.empty(1, 0, 2, 4, requires_grad=True)
    k = torch.empty_like(q, requires_grad=True)
    v = torch.empty(1, 0, 2, 3, requires_grad=True)
    g = torch.empty(1, 0, 2, requires_grad=True)
    beta = torch.empty_like(g, requires_grad=True)
    cu = torch.zeros(n + 1, dtype=torch.int32) if multi else None
    out, final = f._NativeCpChunkGatedDeltaRule.apply(
        q, k, v, g, beta, state, cu, cu, object(), True, 0.5, True
    )
    assert out.shape == v.shape
    torch.testing.assert_close(state, before)
    torch.testing.assert_close(final, before * 2 + 5.25)
    (out.sum() + final.sum()).backward()
    torch.testing.assert_close(state.grad, torch.full_like(state, 2))
    assert all(t.grad is not None and t.grad.numel() == 0 for t in (q, k, v, g, beta))


def test_native_chunk_indices_preserve_empty_sequence_ids():
    from art.megatron.gdn.fla_cp import _native_chunk_indices

    cu = torch.tensor([0, 0, 64, 64, 192], dtype=torch.int32)
    assert _native_chunk_indices(cu, cu).tolist() == [[1, 0], [3, 0], [3, 1]]


def test_execution_scopes_provider_configs_and_restores_after_exception():
    from types import SimpleNamespace

    from art.megatron.context_parallel.root_closed import (
        current_root_placements,
        execution,
    )
    from art.megatron.training.microbatches import (
        _context_parallel_config_for_provider,
        _gdn_planner_config_for_provider,
    )

    provider = SimpleNamespace(
        hidden_size=16,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    handler = SimpleNamespace(
        context_parallel_workload_profile=lambda provider: None,
        build_gdn_execution_spec=True,
    )
    placements = ((0, 1024, 0, 2),)
    assert current_root_placements() == ()
    with pytest.raises(RuntimeError, match="probe"):
        with execution(placements):
            config = _context_parallel_config_for_provider(
                provider, torch.device("cpu"), handler
            )
            assert config.root_placements == placements
            gdn = _gdn_planner_config_for_provider(provider, handler)
            assert gdn is not None and gdn.root_closed
            with execution(()):
                gdn = _gdn_planner_config_for_provider(provider, handler)
                assert gdn is not None and not gdn.root_closed
            assert current_root_placements() == placements
            raise RuntimeError("probe")
    assert current_root_placements() == ()


def test_planner_positional_constructor_matches_main():
    assert GdnPlannerConfig(8) == replace(GdnPlannerConfig(), cp_chain_beam_width=8)
    assert GdnPlannerConfig(8, 3, root_closed=True) == replace(
        GdnPlannerConfig(root_closed=True),
        cp_chain_beam_width=8,
        cp_chain_beam_branch_factor=3,
    )


@pytest.mark.parametrize("allow_empty_shards", [False, True])
def test_chain_bucket_empty_segments_require_explicit_permission(allow_empty_shards):
    from art.megatron.gdn.gdn_prefix_tree import (
        GdnSegmentSpec,
        _build_tree_bucket_plans,
    )

    segment = GdnSegmentSpec(0, 0, 0, 0, 0, 128, "prefix")

    def build():
        return _build_tree_bucket_plans(
            (segment,),
            (-1,),
            (False,),
            local_token_ranges=(),
            sequence_length=128,
            device="cpu",
            token_ranges_by_rank=(((0, 128, 0),), ()),
            allow_empty_shards=allow_empty_shards,
        )[0]

    if not allow_empty_shards:
        with pytest.raises(ValueError, match="no local tokens"):
            build()
    else:
        bucket = build()
        assert bucket.real_token_count == 0
        assert bucket.allow_empty_shards
        assert bucket.lengths_by_rank_cpu.tolist() == [[128], [0]]


@pytest.mark.parametrize("allow_empty_shards", [False, True])
@pytest.mark.parametrize("lengths", [[[0], [64]], [[65], [0]], [[64], [64]]])
def test_native_cp_empty_validation_is_opt_in(monkeypatch, allow_empty_shards, lengths):
    from art.megatron.gdn import fla_cp as f

    monkeypatch.setattr(f.dist, "is_available", lambda: True)
    monkeypatch.setattr(f.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(f.dist, "get_world_size", lambda group: 2)
    calls = []
    monkeypatch.setattr(
        f._NativeCpChunkGatedDeltaRule, "apply", lambda *args: calls.append(args)
    )
    tokens = lengths[0][0]
    q = torch.zeros(1, tokens, 1, 2)

    def run():
        f.chunk_gated_delta_rule_native_cp(
            q,
            q,
            q,
            g=q[..., 0],
            beta=q[..., 0],
            initial_state=torch.zeros(1, 1, 2, 2),
            group=object(),
            output_final_state=True,
            lengths_by_rank_cpu=torch.tensor(lengths),
            allow_empty_shards=allow_empty_shards,
        )

    if not allow_empty_shards and lengths != [[64], [64]]:
        with pytest.raises(ValueError, match="non-empty|64-token aligned"):
            run()
        assert calls == []
    else:
        run()
        assert len(calls) == 1
        assert calls[0][-1] is allow_empty_shards


def test_speculative_planner_propagates_placements_and_isolates_layout_cache(
    monkeypatch,
):
    from threading import Event

    from art.megatron.context_parallel.root_closed import (
        current_root_placements,
        execution,
    )
    from art.megatron.training.microbatches import _gdn_planner_config_for_provider
    from art.trainer_rank import Unset
    from tests.unit.test_trainer_rank_weird_shapes import (
        _prewarmed_rank,
        _rows,
        _unshared_requests,
    )

    inputs = _unshared_requests(8)
    rank = _prewarmed_rank(monkeypatch, inputs, inputs[:4])
    rank._last_global_micro_batch_size = 4
    rows = _rows(inputs[4:8])
    default_key = rank._layout_cache_key(rows)
    default_layout = rank._compute_group_layout(rows, default_key)
    observed = []
    release = Event()
    original = rank._compute_group_layout

    def compute(snapshots, key):
        assert release.wait(timeout=30)
        config = _gdn_planner_config_for_provider(
            rank.runtime.provider, rank.runtime.model_support_handler
        )
        assert config is not None
        observed.append(
            (
                key,
                current_root_placements(),
                config.root_closed,
            )
        )
        return original(snapshots, key)

    # Config construction below needs the same shape fields used by production.
    provider = rank.runtime.provider
    provider.linear_num_key_heads = provider.linear_num_value_heads = 1
    provider.linear_key_head_dim = provider.linear_value_head_dim = 8
    monkeypatch.setattr(rank, "_compute_group_layout", compute)
    placements = ((0, 128, 0, 1),)
    try:
        with execution(placements):
            root_key = rank._layout_cache_key(rows)
            assert root_key != default_key
            assert rank._cached_group_layout(root_key) is None
            rank._submit_speculative_wave_planning(inputs, 4, checkpoint=Unset)
            future = rank._speculative_planning_future
            assert future is not None
        # The caller exits the scope before the worker builds provider configs.
        release.set()
        future.result(timeout=30)
        assert observed == [(root_key, placements, True)]
        assert current_root_placements() == ()
        assert rank._layout_cache_key(rows) == default_key
        assert rank._cached_group_layout(default_key) is default_layout
        assert rank._cached_group_layout(root_key) is not None
        with execution(((0, 128, 1, 1),)):
            assert rank._layout_cache_key(rows) not in (default_key, root_key)
    finally:
        release.set()
        rank._speculative_planning_executor().shutdown(wait=True)


def test_root_closed_decision_carries_empty_permission_without_planner_context():
    bundle, decision = plan([32768], [(1, 2)], 4)
    assert decision.allow_empty_shards
    assert decision.chained_nodes == (True,)
    for rank in range(4):
        rank_plan = materialize_gdn_rank_execution_plan(
            bundle.gdn_execution_spec,
            decision,
            device="cpu",
            cp_rank=rank,
        )
        buckets = rank_plan.tree_chain_buckets_by_depth[0]
        assert buckets
        assert all(bucket.allow_empty_shards for bucket in buckets)
        assert sum(bucket.real_token_count for bucket in buckets) == (
            16384 if rank in (1, 2) else 0
        )
