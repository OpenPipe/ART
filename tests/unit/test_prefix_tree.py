from __future__ import annotations

import pytest
import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.prefix_tree import parse_prefix_tree_row
from art.megatron.prefix_tree_packing import prefix_tree_pack


def test_parse_prefix_tree_row_tracks_ancestors_and_depth() -> None:
    pack = prefix_tree_pack(
        (
            torch.tensor([1, 2, 3, 4, 8]),
            torch.tensor([1, 2, 3, 4, 9]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 6]),
        ),
        max_depth=3,
    )

    tree = parse_prefix_tree_row(
        group_ids=pack.group_ids[0],
        parent_ids=pack.parent_ids[0],
    )

    assert tree.valid_tokens == int(pack.tokens.numel())
    assert max(segment.depth for segment in tree.segments) == 3
    assert [(segment.group_id, segment.ancestors) for segment in tree.segments] == [
        (1, ()),
        (2, (1,)),
        (3, (1, 2)),
        (4, (1, 2, 3)),
        (5, (1, 2, 3)),
        (6, (1, 2)),
        (7, (1,)),
    ]


def test_parse_prefix_tree_row_rejects_missing_parent() -> None:
    with pytest.raises(RuntimeError, match="missing parent"):
        parse_prefix_tree_row(
            group_ids=torch.tensor([1, 2]),
            parent_ids=torch.tensor([1, 3]),
        )


def test_parse_prefix_tree_row_rejects_non_contiguous_group() -> None:
    with pytest.raises(RuntimeError, match="contiguous group runs"):
        parse_prefix_tree_row(
            group_ids=torch.tensor([1, 2, 1]),
            parent_ids=torch.tensor([1, 1, 1]),
        )


def test_gdn_tree_parser_accepts_nested_tree() -> None:
    pytest.importorskip("megatron.core.packed_seq_params")
    from art.megatron.gdn.gdn_prefix_tree import (
        GdnPlannerConfig,
        build_gdn_rank_execution_plan,
        parse_gdn_prefix_tree_segments,
    )

    pack = prefix_tree_pack(
        (
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([1, 6]),
        ),
        max_depth=2,
    )

    spec = parse_gdn_prefix_tree_segments(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    plan = build_gdn_rank_execution_plan(spec, device="cpu")

    assert spec.tree_parent_indices == (-1, 0, 1, 1, 0)
    assert spec.tree_depths == (0, 1, 2, 2, 1)
    assert [
        sum(bucket.segment_count for bucket in buckets)
        for buckets in plan.tree_segment_buckets_by_depth
    ] == [0, 2, 2]
    child_bucket = plan.tree_segment_buckets_by_depth[1][0]
    assert child_bucket.parent_indices is not None
    assert child_bucket.parent_indices.tolist() == [-1, -1]
    assert child_bucket.family_indices.tolist() == [1, 4]
    assert child_bucket.position_indices.tolist() == [0, 1, 2, 0, 5]
    assert child_bucket.output_mask is not None
    assert child_bucket.output_mask.tolist() == [True, True, True, False, True]


def test_gdn_tree_parser_accepts_zero_depth_roots() -> None:
    pytest.importorskip("megatron.core.packed_seq_params")
    from art.megatron.gdn.gdn_prefix_tree import (
        build_gdn_rank_execution_plan,
        parse_gdn_prefix_tree_segments,
    )

    pack = prefix_tree_pack(
        (
            torch.tensor([1, 2]),
            torch.tensor([1, 3]),
            torch.tensor([4]),
        ),
        max_depth=0,
    )

    spec = parse_gdn_prefix_tree_segments(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    plan = build_gdn_rank_execution_plan(spec, device="cpu")

    assert spec.tree_parent_indices == (-1, -1, -1)
    assert spec.tree_depths == (0, 0, 0)
    assert [bucket.segment_count for bucket in plan.tree_segment_buckets_by_depth[0]]
    assert not hasattr(plan, "local_prefix_buckets")
    assert not hasattr(plan, "chain_completion_buckets")
    assert not hasattr(plan, "prefix_boundary_buckets")
    assert all(
        not bucket.needs_final_state for bucket in plan.tree_segment_buckets_by_depth[0]
    )


def test_gdn_tree_planner_splits_leaf_and_internal_final_state_buckets() -> None:
    pytest.importorskip("megatron.core.packed_seq_params")
    from art.megatron.gdn.gdn_prefix_tree import (
        GdnPlannerConfig,
        build_gdn_rank_execution_plan,
        parse_gdn_prefix_tree_segments,
    )

    pack = prefix_tree_pack(
        (
            torch.tensor([1, 2, 3, 4, 7]),
            torch.tensor([1, 2, 3, 4, 8]),
            torch.tensor([1, 2, 5, 6]),
        ),
        max_depth=2,
    )

    spec = parse_gdn_prefix_tree_segments(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    plan = build_gdn_rank_execution_plan(
        spec,
        device="cpu",
        planner_config=GdnPlannerConfig(),
    )
    tree_has_children = _tree_has_children(spec)

    depth_one_bucket = plan.tree_segment_buckets_by_depth[1][0]
    bucket_state_flags = {
        tree_has_children[family_index]
        for family_index in depth_one_bucket.family_indices.tolist()
    }
    assert bucket_state_flags == {False, True}
    assert depth_one_bucket.needs_final_state


def test_gdn_tree_cp_plan_chains_long_nodes() -> None:
    pytest.importorskip("megatron.core.packed_seq_params")
    from art.megatron.gdn.gdn_prefix_tree import (
        GdnPlannerConfig,
        build_gdn_rank_execution_plan,
        parse_gdn_prefix_tree_segments,
    )

    root = torch.arange(1, 321)
    mid = torch.arange(1001, 1321)
    other = torch.arange(2001, 2321)
    pack = prefix_tree_pack(
        (
            torch.cat((root, mid, torch.tensor([11]))),
            torch.cat((root, mid, torch.tensor([12]))),
            torch.cat((root, other, torch.tensor([13]))),
        ),
        max_depth=3,
    )
    spec = parse_gdn_prefix_tree_segments(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    config = _chain_every_legal_segment_config()
    plans = tuple(
        build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=rank,
            cp_size=4,
            attention_token_layout_index=_uniform_attention_layout(
                spec.real_token_count, 4
            ),
            planner_config=config,
        )
        for rank in range(4)
    )

    assert _covered_token_indices(plans) == set(range(spec.real_token_count))
    assert any(plans[0].tree_chain_buckets_by_depth[0])
    assert any(
        bucket
        for plan in plans
        for depth_buckets in plan.tree_chain_buckets_by_depth[1:]
        for bucket in depth_buckets
    )
    _assert_remote_parent_state_transfers_cover(spec, plans)
    for plan in plans:
        assert sum(plan.gdn_token_count for plan in plans) == spec.real_token_count
        for depth_buckets in plan.tree_chain_buckets_by_depth:
            for bucket in depth_buckets:
                assert bucket.lengths_by_rank_cpu is not None
                assert tuple(bucket.lengths_by_rank_cpu.shape)[0] == 4
                assert bucket.parent_indices is not None


def test_gdn_tree_cp_plan_exchanges_remote_parent_states() -> None:
    pytest.importorskip("megatron.core.packed_seq_params")
    from art.megatron.gdn.gdn_prefix_tree import (
        build_gdn_rank_execution_plan,
        parse_gdn_prefix_tree_segments,
    )

    root = torch.arange(1, 17)
    mid = torch.arange(1001, 1321)
    pack = prefix_tree_pack(
        (
            torch.cat((root, mid, torch.tensor([11]))),
            torch.cat((root, mid, torch.tensor([12]))),
            torch.cat((root, torch.tensor([99]))),
        ),
        max_depth=2,
    )
    spec = parse_gdn_prefix_tree_segments(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    plans = tuple(
        build_gdn_rank_execution_plan(
            spec,
            device="cpu",
            cp_rank=rank,
            cp_size=4,
            attention_token_layout_index=_uniform_attention_layout(
                spec.real_token_count, 4
            ),
            planner_config=_chain_every_legal_segment_config(),
        )
        for rank in range(4)
    )
    assert _covered_token_indices(plans) == set(range(spec.real_token_count))
    assert any(
        bucket
        for plan in plans
        for depth_buckets in plan.tree_chain_buckets_by_depth[1:]
        for bucket in depth_buckets
    )
    assert _remote_parent_state_transfer_count(plans) > 0
    _assert_remote_parent_state_transfers_cover(spec, plans)


def test_gdn_tree_cp_randomized_plans_cover_each_token_once() -> None:
    pytest.importorskip("megatron.core.packed_seq_params")
    from art.megatron.gdn.gdn_prefix_tree import (
        build_gdn_rank_execution_plan,
        parse_gdn_prefix_tree_segments,
    )

    config = _chain_every_legal_segment_config()
    for seed in range(8):
        pack = prefix_tree_pack(
            _random_tree_sequences(seed),
            max_depth=4,
        )
        spec = parse_gdn_prefix_tree_segments(
            group_ids=pack.group_ids,
            parent_ids=pack.parent_ids,
        )
        plans = tuple(
            build_gdn_rank_execution_plan(
                spec,
                device="cpu",
                cp_rank=rank,
                cp_size=4,
                attention_token_layout_index=_uniform_attention_layout(
                    spec.real_token_count, 4
                ),
                planner_config=config,
            )
            for rank in range(4)
        )

        assert _covered_token_indices(plans) == set(range(spec.real_token_count))
        assert sum(plan.gdn_token_count for plan in plans) == spec.real_token_count
        for plan in plans:
            for depth_buckets in (
                *plan.tree_segment_buckets_by_depth,
                *plan.tree_chain_buckets_by_depth,
            ):
                for bucket in depth_buckets:
                    assert bucket.parent_indices is not None
                    assert int(bucket.real_token_count) > 0


def test_gdn_tree_cp_randomized_plans_pass_health_checks() -> None:
    pytest.importorskip("megatron.core.packed_seq_params")
    from art.megatron.gdn.gdn_prefix_tree import (
        GdnPlannerConfig,
        build_gdn_rank_execution_plan,
        parse_gdn_prefix_tree_segments,
    )

    config = GdnPlannerConfig()
    for seed in range(16):
        pack = prefix_tree_pack(
            _random_tree_sequences(seed + 100, max_depth=5),
            max_depth=5,
        )
        spec = parse_gdn_prefix_tree_segments(
            group_ids=pack.group_ids,
            parent_ids=pack.parent_ids,
        )
        plans = tuple(
            build_gdn_rank_execution_plan(
                spec,
                device="cpu",
                cp_rank=rank,
                cp_size=4,
                attention_token_layout_index=_uniform_attention_layout(
                    spec.real_token_count, 4
                ),
                planner_config=config,
            )
            for rank in range(4)
        )

        _assert_tree_plan_health(spec, plans)


def _chain_every_legal_segment_config():
    from art.megatron.gdn.gdn_prefix_tree import GdnPlannerConfig

    return GdnPlannerConfig(
        cp_chain_min_runtime_delta_ms=0.0,
        runtime_local_recurrent_tokens_per_ms=1.0,
        runtime_chain_recurrent_tokens_per_ms=1_000_000_000.0,
        runtime_cp_summary_bytes_per_segment=0,
        runtime_cp_summary_exchange_count_per_bucket=0,
        runtime_cp_summary_compute_segments_per_ms=1_000_000_000.0,
        runtime_cp_suffix_scan_latency_ms=0.0,
        runtime_parent_state_bytes_per_exchange=0,
    )


def _covered_token_indices(plans) -> set[int]:
    return {
        token
        for plan in plans
        for start, end, _position in plan.gdn_token_ranges
        for token in range(start, end)
    }


def _local_owner_by_family(plans) -> dict[int, int]:
    owner_by_family = {}
    for rank, plan in enumerate(plans):
        for depth_buckets in plan.tree_segment_buckets_by_depth:
            for bucket in depth_buckets:
                for family_index in bucket.family_indices.tolist():
                    previous = owner_by_family.setdefault(int(family_index), rank)
                    assert previous == rank
    return owner_by_family


def _chained_families(plans) -> set[int]:
    return {
        int(family_index)
        for plan in plans
        for depth_buckets in plan.tree_chain_buckets_by_depth
        for bucket in depth_buckets
        for family_index in bucket.family_indices.tolist()
    }


def _assert_remote_parent_state_transfers_cover(spec, plans) -> None:
    owner_by_family = _local_owner_by_family(plans)
    chained_families = _chained_families(plans)

    def assert_transfer(
        parent_index: int, source_rank: int, dest_rank: int, depth: int
    ) -> None:
        source_exchange = plans[source_rank].tree_state_exchanges_by_depth[depth]
        dest_exchange = plans[dest_rank].tree_state_exchanges_by_depth[depth]
        assert source_exchange is not None
        assert dest_exchange is not None
        assert parent_index in source_exchange.source_family_indices
        assert parent_index in dest_exchange.dest_family_indices
        matching = [
            transfer
            for transfer in dest_exchange.exchange.transfers
            if transfer.source_rank == source_rank and transfer.dest_rank == dest_rank
        ]
        assert matching

    for family_index, parent_index in enumerate(spec.tree_parent_indices):
        if parent_index < 0 or parent_index not in owner_by_family:
            continue
        source_rank = owner_by_family[parent_index]
        depth = spec.tree_depths[family_index]
        if family_index in chained_families:
            for dest_rank in range(len(plans)):
                if source_rank != dest_rank:
                    assert_transfer(parent_index, source_rank, dest_rank, depth)
            continue
        assert family_index in owner_by_family
        dest_rank = owner_by_family[family_index]
        if source_rank != dest_rank:
            assert_transfer(parent_index, source_rank, dest_rank, depth)


def _remote_parent_state_transfer_count(plans) -> int:
    return sum(
        exchange.exchange.cross_rank_token_count
        for plan in plans
        for exchange in plan.tree_state_exchanges_by_depth
        if exchange is not None
    ) // len(plans)


def _tree_has_children(spec) -> list[bool]:
    has_children = [False] * spec.family_count
    for parent_index in spec.tree_parent_indices:
        if parent_index >= 0:
            has_children[parent_index] = True
    return has_children


def _assert_tree_plan_health(spec, plans) -> None:
    tree_has_children = _tree_has_children(spec)
    token_counts = [0] * int(spec.real_token_count)
    for plan in plans:
        range_tokens = sum(
            end - start for start, end, _position in plan.gdn_token_ranges
        )
        assert range_tokens == int(plan.gdn_token_count)
        assert len(plan.attention_token_indices) == int(plan.attention_token_count)

        bucket_tokens = 0
        for depth_buckets in plan.tree_segment_buckets_by_depth:
            for bucket in depth_buckets:
                bucket_tokens += int(bucket.real_token_count)
                assert bucket.parent_indices is not None
                assert int(bucket.parent_indices.numel()) == int(bucket.segment_count)
                assert int(bucket.real_token_count) > 0
                bucket_state_flags = {
                    tree_has_children[family_index]
                    for family_index in bucket.family_indices.tolist()
                }
                assert bucket_state_flags == {bucket.needs_final_state}
                for family_index, parent_index in zip(
                    bucket.family_indices.tolist(),
                    bucket.parent_indices.tolist(),
                    strict=True,
                ):
                    assert spec.tree_parent_indices[family_index] == parent_index

        for depth_buckets in plan.tree_chain_buckets_by_depth:
            for bucket in depth_buckets:
                bucket_tokens += int(bucket.real_token_count)
                assert bucket.parent_indices is not None
                assert int(bucket.parent_indices.numel()) == int(bucket.segment_count)
                assert int(bucket.real_token_count) > 0
                bucket_state_flags = {
                    tree_has_children[family_index]
                    for family_index in bucket.family_indices.tolist()
                }
                if bucket.needs_final_state:
                    assert any(bucket_state_flags)
                else:
                    assert bucket_state_flags == {False}
                for family_index, parent_index in zip(
                    bucket.family_indices.tolist(),
                    bucket.parent_indices.tolist(),
                    strict=True,
                ):
                    assert spec.tree_parent_indices[family_index] == parent_index
        assert bucket_tokens == int(plan.gdn_token_count)

        for start, end, _position in plan.gdn_token_ranges:
            for token_index in range(start, end):
                token_counts[token_index] += 1

    _assert_remote_parent_state_transfers_cover(spec, plans)
    assert token_counts == [1] * int(spec.real_token_count)
    rank_tokens = [int(plan.gdn_token_count) for plan in plans]
    assert max(rank_tokens) - min(rank_tokens) <= max(256, spec.real_token_count // 3)


def _uniform_attention_layout(real_token_count: int, cp_size: int) -> TokenLayoutIndex:
    ranges_by_rank: list[tuple[tuple[int, int, int], ...]] = []
    for rank in range(cp_size):
        start = (int(real_token_count) * rank) // cp_size
        end = (int(real_token_count) * (rank + 1)) // cp_size
        ranges_by_rank.append(((start, end, 0),) if end > start else ())
    return TokenLayoutIndex(
        ownership_ranges_by_rank=tuple(ranges_by_rank),
        token_counts_by_rank=tuple(
            sum(end - start for start, end, _position in ranges)
            for ranges in ranges_by_rank
        ),
    )


def _random_tree_sequences(
    seed: int, *, max_depth: int = 4
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    next_token = 1

    def tokens(length: int) -> torch.Tensor:
        nonlocal next_token
        out = torch.arange(next_token, next_token + length)
        next_token += length
        return out

    def randint(low: int, high: int) -> int:
        return int(torch.randint(low, high + 1, (), generator=generator).item())

    def walk(prefix: torch.Tensor, depth: int) -> list[torch.Tensor]:
        segment_length = [1, 3, 17, 64, 129, 257][randint(0, 5)]
        here = torch.cat((prefix, tokens(segment_length)))
        if depth + 1 >= max_depth:
            return [
                torch.cat((here, tokens(randint(1, 9)))) for _ in range(randint(2, 4))
            ]
        leaves: list[torch.Tensor] = []
        for _ in range(randint(2, 3)):
            leaves.extend(walk(here, depth + 1))
        return leaves

    return tuple(walk(torch.empty(0, dtype=torch.long), 0))
