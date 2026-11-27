"""Unit tests for synchronized depth-epoch runtime modeling in the GDN CP planner."""

from __future__ import annotations

import pytest

pytest.importorskip("triton")

from art.megatron.context_parallel.layout_index import TokenLayoutIndex  # noqa: E402
from art.megatron.gdn.gdn_prefix_tree import (  # noqa: E402
    GdnPackedExecutionSpec,
    GdnPlannerConfig,
    GdnSegmentSpec,
    _count_parent_state_exchanges_by_depth,
    _estimate_chain_runtime_from_counts,
    _estimate_local_runtime_from_lengths,
    _predict_gdn_plan_runtime_ms,
    build_gdn_global_execution_decision,
)


def _uniform_attention_layout(total_tokens: int, cp_size: int) -> TokenLayoutIndex:
    base = total_tokens // cp_size
    counts = [base] * cp_size
    counts[-1] = total_tokens - base * (cp_size - 1)
    ranges = []
    cursor = 0
    for count in counts:
        ranges.append(((cursor, cursor + count, 0),))
        cursor += count
    return TokenLayoutIndex(
        ownership_ranges_by_rank=tuple(ranges),
        token_counts_by_rank=tuple(counts),
    )


def _alone_spec(prefix_len: int, tail_len: int) -> GdnPackedExecutionSpec:
    total = prefix_len + tail_len
    return GdnPackedExecutionSpec(
        batch_size=1,
        sequence_length=total,
        valid_lengths=(total,),
        tree_segments=(GdnSegmentSpec(0, 0, 1, -1, 0, total, "prefix"),),
        tree_parent_indices=(-1,),
        tree_depths=(0,),
    )


def _grpo_batch_spec(
    prefix_len: int,
    tail_len: int,
    num_rollouts: int,
) -> GdnPackedExecutionSpec:
    total = prefix_len + num_rollouts * tail_len
    segments = [GdnSegmentSpec(0, 0, 1, -1, 0, prefix_len, "prefix")]
    for idx in range(num_rollouts):
        start = prefix_len + idx * tail_len
        segments.append(
            GdnSegmentSpec(
                0,
                idx + 1,
                idx + 2,
                1,
                start,
                start + tail_len,
                "completion",
                idx,
            )
        )
    return GdnPackedExecutionSpec(
        batch_size=1,
        sequence_length=total,
        valid_lengths=(total,),
        tree_segments=tuple(segments),
        tree_parent_indices=(-1, *([0] * num_rollouts)),
        tree_depths=(0, *([1] * num_rollouts)),
    )


@pytest.mark.parametrize(
    ("model_name", "hidden_size", "value_heads"),
    [
        ("qwen35_35b_a3b", 2048, 32),
        ("qwen35_4b", 2560, 32),
        ("qwen38_27b", 5120, 48),
    ],
)
@pytest.mark.parametrize("num_rollouts", [2, 4])
def test_gdn_cp_planner_matches_alone_and_grpo_batch_chain_decisions(
    model_name: str,
    hidden_size: int,
    value_heads: int,
    num_rollouts: int,
) -> None:
    del model_name
    cfg = GdnPlannerConfig.from_model_shape(
        hidden_size=hidden_size,
        tensor_model_parallel_size=1,
        linear_num_key_heads=16,
        linear_num_value_heads=value_heads,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    for prefix_len in (1536, 2048, 2560, 3072, 3584, 4096, 4224, 4608, 5120, 8192):
        alone = _alone_spec(prefix_len, 63)
        batch = _grpo_batch_spec(prefix_len, 63, num_rollouts)
        dec_alone = build_gdn_global_execution_decision(
            alone,
            cp_size=4,
            attention_token_layout_index=_uniform_attention_layout(
                alone.real_token_count, 4
            ),
            planner_config=cfg,
        )
        dec_batch = build_gdn_global_execution_decision(
            batch,
            cp_size=4,
            attention_token_layout_index=_uniform_attention_layout(
                batch.real_token_count, 4
            ),
            planner_config=cfg,
        )
        assert dec_alone.chained_nodes[0] == dec_batch.chained_nodes[0], (
            f"prefix={prefix_len} N={num_rollouts} h={hidden_size} hv={value_heads}: "
            f"alone_chained={dec_alone.chained_nodes[0]} vs "
            f"batch_chained={dec_batch.chained_nodes[0]}"
        )


def test_gdn_cp_planner_chains_35b_4096_grpo_root_consistently() -> None:
    cfg = GdnPlannerConfig.from_model_shape(
        hidden_size=2048,
        tensor_model_parallel_size=1,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    for num_rollouts in (2, 4):
        batch = _grpo_batch_spec(4096, 63, num_rollouts)
        dec = build_gdn_global_execution_decision(
            batch,
            cp_size=4,
            attention_token_layout_index=_uniform_attention_layout(
                batch.real_token_count, 4
            ),
            planner_config=cfg,
        )
        assert dec.chained_nodes[0] is True
        assert all(not c for c in dec.chained_nodes[1:])
        assert max(dec.gdn_token_counts_by_rank) < 1200


def test_depth_barrier_critical_path_charges_cross_rank_child_depth_sequentially() -> (
    None
):
    cfg = GdnPlannerConfig()
    # Rank 0 owns depth-0 root (4096); Rank 3 owns depth-1 children (4 x 63).
    local_lengths = (
        ((4096,), ()),
        ((), ()),
        ((), ()),
        ((), (63, 63, 63, 63)),
    )
    local_runtime = _estimate_local_runtime_from_lengths(local_lengths)
    chain_runtime = _estimate_chain_runtime_from_counts(((), ()), cp_size=4)

    # Without cross-rank state exchange barrier: depths could overlap across ranks.
    overlapped_ms = _predict_gdn_plan_runtime_ms(
        local_runtime,
        chain_runtime,
        cross_rank_token_count=0,
        parent_state_exchange_count=0,
        planner_config=cfg,
        rank_token_counts=(4096, 0, 0, 252),
        parent_state_exchanges_by_depth=(0, 0),
    )
    # With a depth-1 cross-rank parent-state exchange barrier: depth 1 waits for depth 0.
    synchronized_ms = _predict_gdn_plan_runtime_ms(
        local_runtime,
        chain_runtime,
        cross_rank_token_count=0,
        parent_state_exchange_count=0,
        planner_config=cfg,
        rank_token_counts=(4096, 0, 0, 252),
        parent_state_exchanges_by_depth=(0, 1),
    )
    expected_depth1_ms = (
        (4 * 63) / cfg.runtime_local_recurrent_tokens_per_ms
        + cfg.runtime_local_bucket_launch_ms
        + 4 * cfg.runtime_local_segment_launch_ms
    )
    assert synchronized_ms - overlapped_ms == pytest.approx(expected_depth1_ms)


def test_count_parent_state_exchanges_by_depth_tracks_remote_edges() -> None:
    spec = _grpo_batch_spec(4096, 63, 4)
    # Local root on rank 0, children on rank 3 -> 4 remote edges at depth 1.
    assert _count_parent_state_exchanges_by_depth(
        spec,
        owner_by_node=(0, 3, 3, 3, 3),
        chain_nodes=(False, False, False, False, False),
        depth_count=2,
    ) == (0, 4)
    # Chained root broadcasts its final state directly -> 0 local parent-state exchanges.
    assert _count_parent_state_exchanges_by_depth(
        spec,
        owner_by_node=(-1, 3, 0, 1, 2),
        chain_nodes=(True, False, False, False, False),
        depth_count=2,
    ) == (0, 0)
    # Same-rank local parent and children -> 0 cross-rank exchanges.
    assert _count_parent_state_exchanges_by_depth(
        spec,
        owner_by_node=(0, 0, 0, 0, 0),
        chain_nodes=(False, False, False, False, False),
        depth_count=2,
    ) == (0, 0)
