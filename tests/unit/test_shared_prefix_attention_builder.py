from __future__ import annotations

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask as torch_block_mask

pytest.importorskip("megatron.core.packed_seq_params")

from art.megatron.context_parallel.block_mask import (
    build_block_mask_from_context,
    prepare_block_mask_context,
)
from art.megatron.context_parallel.builder import (
    build_dense_reference_mask,
    build_shared_prefix_attention_spec,
)
from art.megatron.context_parallel.runtime import get_or_build_runtime_plan
from art.megatron.context_parallel.types import (
    AttnMaskKind,
    AttnSlice,
    ContextParallelConfig,
    ExactMaskMetadata,
    FlexMaskSpec,
    ParallelTopology,
    TokenRange,
)
from art.megatron.shared_prefix_packing import SharedPrefixPack, pack_shared_prefixes
from art.megatron.shared_prefix_state import create_shared_prefix_state


def build_block_mask(
    spec: FlexMaskSpec,
    *,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    device: torch.device,
) -> BlockMask | None:
    return build_block_mask_from_context(
        spec,
        context=prepare_block_mask_context(
            group_ids=group_ids,
            parent_ids=parent_ids,
        ),
        device=device,
    )


def test_shared_prefix_attention_spec_supports_branching_completions() -> None:
    group_ids, parent_ids = _branching_prefix_inputs()

    spec = build_shared_prefix_attention_spec(
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    dense = build_dense_reference_mask(row_spec=spec.rows[0])

    assert dense.int().tolist() == [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 0, 1, 1],
    ]


def test_shared_prefix_attention_spec_matches_tree_reference() -> None:
    group_ids, parent_ids = _branching_prefix_inputs()

    spec = build_shared_prefix_attention_spec(
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    dense = build_dense_reference_mask(row_spec=spec.rows[0])

    assert dense.equal(_reference_tree_mask(group_ids[0], parent_ids[0]))


def test_shared_prefix_can_build_context_parallel_layout() -> None:
    group_ids, parent_ids = _branching_prefix_inputs()
    spec = build_shared_prefix_attention_spec(
        group_ids=group_ids,
        parent_ids=parent_ids,
    )

    plan = get_or_build_runtime_plan(
        spec,
        topology=ParallelTopology(cp=2),
        config=ContextParallelConfig(planner_chunk_size=2, planner_max_search_steps=1),
        original_seq_len=int(group_ids.numel()),
    )

    assert sum(plan[rank].local_valid_lengths[0] for rank in range(2)) == int(
        group_ids.numel()
    )


def test_sparse_block_mask_exact_predicate_matches_dense_reference() -> None:
    group_ids, parent_ids = _branching_prefix_inputs()
    spec = build_shared_prefix_attention_spec(
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    row = spec.rows[0]
    token_indices = torch.arange(row.valid_tokens, dtype=torch.long)
    block_mask = build_block_mask(
        FlexMaskSpec(
            q_len=row.valid_tokens,
            k_len=row.valid_tokens,
            block_size=(2, 2),
            slices=row.slices,
            exact_mask=ExactMaskMetadata(
                q_token_indices=token_indices,
                k_token_indices=token_indices,
                cache_key="depth-two",
            ),
        ),
        group_ids=group_ids[0],
        parent_ids=parent_ids[0],
        device=torch.device("cpu"),
    )

    assert block_mask is not None
    q_indices = torch.arange(row.valid_tokens)[:, None]
    k_indices = torch.arange(row.valid_tokens)[None, :]
    actual = block_mask.mask_mod(
        torch.zeros_like(q_indices),
        torch.zeros_like(q_indices),
        q_indices,
        k_indices,
    )

    assert actual.equal(build_dense_reference_mask(row_spec=row))


@pytest.mark.parametrize(
    ("name", "pack"),
    (
        (
            "no-sharing",
            pack_shared_prefixes(
                (
                    torch.tensor([1, 2, 3]),
                    torch.tensor([4, 5]),
                    torch.tensor([6, 7, 8, 9]),
                ),
                max_depth=0,
            ),
        ),
        (
            "depth-one",
            pack_shared_prefixes(
                (
                    torch.tensor([1, 2, 3, 4]),
                    torch.tensor([1, 2, 3, 5]),
                    torch.tensor([1, 2, 6]),
                ),
                max_depth=1,
            ),
        ),
        (
            "depth-three",
            pack_shared_prefixes(
                (
                    torch.tensor([1, 2, 3, 4, 8]),
                    torch.tensor([1, 2, 3, 4, 9]),
                    torch.tensor([1, 2, 3, 5]),
                    torch.tensor([1, 6]),
                ),
                max_depth=3,
            ),
        ),
    ),
)
def test_sparse_block_mask_matches_torch_block_metadata(
    name: str,
    pack: SharedPrefixPack,
) -> None:
    del name
    spec = build_shared_prefix_attention_spec(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    row = spec.rows[0]
    token_indices = torch.arange(row.valid_tokens, dtype=torch.long)
    block_mask = build_block_mask(
        FlexMaskSpec(
            q_len=row.valid_tokens,
            k_len=row.valid_tokens,
            block_size=(2, 2),
            slices=row.slices,
            exact_mask=ExactMaskMetadata(
                q_token_indices=token_indices,
                k_token_indices=token_indices,
                cache_key="torch-parity",
            ),
        ),
        group_ids=pack.group_ids[0],
        parent_ids=pack.parent_ids[0],
        device=torch.device("cpu"),
    )

    assert block_mask is not None
    _assert_matches_torch_block_mask(block_mask)


def test_sparse_block_mask_prunes_exact_blocks_rejected_by_group_tree() -> None:
    group_ids = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    parent_ids = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    block_mask = build_block_mask(
        FlexMaskSpec(
            q_len=4,
            k_len=4,
            block_size=(2, 2),
            slices=(
                AttnSlice(
                    q_range=TokenRange(start=0, end=4),
                    k_range=TokenRange(start=0, end=4),
                    mask_kind=AttnMaskKind.CAUSAL,
                    row_index=0,
                ),
            ),
            exact_mask=ExactMaskMetadata(
                q_token_indices=torch.tensor([4, 5, 6, 7], dtype=torch.long),
                k_token_indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
                cache_key="all-false-cross-family",
            ),
        ),
        group_ids=group_ids,
        parent_ids=parent_ids,
        device=torch.device("cpu"),
    )

    assert block_mask is not None
    assert int(block_mask.kv_num_blocks.sum().item()) == 0
    assert int(block_mask.full_kv_num_blocks.sum().item()) == 0
    _assert_matches_torch_block_mask(block_mask)


def test_shared_prefix_state_builds_batched_block_mask() -> None:
    group_ids = torch.tensor(
        [
            [1, 1, 2, 2, -1],
            [10, 11, 11, -1, -1],
        ],
        dtype=torch.long,
    )
    parent_ids = torch.tensor(
        [
            [1, 1, 1, 1, -1],
            [10, 10, 10, -1, -1],
        ],
        dtype=torch.long,
    )

    state = create_shared_prefix_state(
        group_ids=group_ids,
        parent_ids=parent_ids,
        target_device=torch.device("cpu"),
    )
    seq_len = int(group_ids.shape[1])
    batch_idx = torch.arange(2)[:, None, None].expand(2, seq_len, seq_len)
    query_idx = torch.arange(seq_len)[None, :, None].expand(2, seq_len, seq_len)
    kv_idx = torch.arange(seq_len)[None, None, :].expand(2, seq_len, seq_len)
    actual = state.block_mask.mask_mod(
        batch_idx,
        torch.zeros_like(batch_idx),
        query_idx,
        kv_idx,
    )
    spec = build_shared_prefix_attention_spec(
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    assert int(state.block_mask.kv_num_blocks.shape[0]) == 2
    for row_index, row_spec in enumerate(spec.rows):
        valid_tokens = int(row_spec.valid_tokens)
        assert actual[
            row_index,
            :valid_tokens,
            :valid_tokens,
        ].equal(build_dense_reference_mask(row_spec=row_spec))
    _assert_matches_torch_block_mask(state.block_mask, batch_size=2)


def test_context_parallel_stage_masks_match_dense_nested_tree() -> None:
    _assert_context_parallel_stage_masks_match_dense(
        pack_shared_prefixes(
            (
                torch.tensor([1, 2, 3, 4, 8]),
                torch.tensor([1, 2, 3, 4, 9]),
                torch.tensor([1, 2, 3, 5]),
                torch.tensor([1, 6]),
            ),
            max_depth=3,
        ),
        require_remote_stage=True,
    )
    _assert_context_parallel_stage_masks_match_dense(
        pack_shared_prefixes(
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
                torch.tensor([7, 8]),
                torch.tensor([9, 10, 11, 12]),
            ),
            max_depth=3,
        ),
        require_remote_stage=False,
    )


def _assert_context_parallel_stage_masks_match_dense(
    pack: SharedPrefixPack,
    *,
    require_remote_stage: bool,
) -> None:
    spec = build_shared_prefix_attention_spec(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    row = spec.rows[0]
    dense = build_dense_reference_mask(row_spec=row)
    topology = ParallelTopology(cp=2)
    config = ContextParallelConfig(
        block_size=2,
        planner_chunk_size=2,
        planner_max_search_steps=1,
        planner_remote_stage_token_floor=1,
        planner_remote_stage_pair_floor=1,
    )
    plan = get_or_build_runtime_plan(
        spec,
        topology=topology,
        config=config,
        original_seq_len=int(pack.tokens.numel()),
    )

    checked_stages = 0
    checked_remote_stages = 0
    for rank_plan in plan:
        for stage in rank_plan.stage_plans:
            if stage.mask_metadata is None:
                continue
            block_mask = build_block_mask(
                FlexMaskSpec(
                    q_len=stage.q_len,
                    k_len=stage.k_len,
                    block_size=(2, 2),
                    slices=stage.slices,
                    exact_mask=stage.mask_metadata,
                ),
                group_ids=pack.group_ids[0],
                parent_ids=pack.parent_ids[0],
                device=torch.device("cpu"),
            )
            assert block_mask is not None
            q_offsets = torch.arange(stage.q_len)[:, None]
            k_offsets = torch.arange(stage.k_len)[None, :]
            actual = block_mask.mask_mod(
                torch.zeros_like(q_offsets),
                torch.zeros_like(q_offsets),
                q_offsets,
                k_offsets,
            )
            q_tokens = stage.mask_metadata.q_token_indices
            k_tokens = stage.mask_metadata.k_token_indices
            expected = (
                dense[q_tokens.clamp_min(0)[:, None], k_tokens.clamp_min(0)[None, :]]
                & (q_tokens[:, None] >= 0)
                & (k_tokens[None, :] >= 0)
            )

            assert actual.equal(expected)
            assert _effective_block_mask(block_mask).equal(expected)
            _assert_matches_torch_block_mask(block_mask)
            checked_stages += 1
            checked_remote_stages += int(not stage.is_local_stage)

    assert checked_stages
    if require_remote_stage:
        assert checked_remote_stages


def _effective_block_mask(block_mask: BlockMask) -> torch.Tensor:
    q_len, k_len = block_mask.seq_lengths
    q_block, k_block = block_mask.BLOCK_SIZE
    effective = torch.zeros((q_len, k_len), dtype=torch.bool)
    _fill_full_blocks(effective, block_mask, q_block=q_block, k_block=k_block)
    _fill_partial_blocks(effective, block_mask, q_block=q_block, k_block=k_block)
    return effective


def _fill_full_blocks(
    effective: torch.Tensor,
    block_mask: BlockMask,
    *,
    q_block: int,
    k_block: int,
) -> None:
    if block_mask.full_kv_num_blocks is None or block_mask.full_kv_indices is None:
        return
    for q_block_index in range(int(block_mask.full_kv_num_blocks.shape[-1])):
        q_slice = slice(q_block_index * q_block, (q_block_index + 1) * q_block)
        block_count = int(block_mask.full_kv_num_blocks[0, 0, q_block_index])
        for k_block_index in block_mask.full_kv_indices[
            0, 0, q_block_index, :block_count
        ].tolist():
            k_slice = slice(
                int(k_block_index) * k_block,
                (int(k_block_index) + 1) * k_block,
            )
            effective[q_slice, k_slice] = True


def _fill_partial_blocks(
    effective: torch.Tensor,
    block_mask: BlockMask,
    *,
    q_block: int,
    k_block: int,
) -> None:
    for q_block_index in range(int(block_mask.kv_num_blocks.shape[-1])):
        q_offsets = torch.arange(
            q_block_index * q_block,
            min((q_block_index + 1) * q_block, effective.shape[0]),
        )[:, None]
        block_count = int(block_mask.kv_num_blocks[0, 0, q_block_index])
        for k_block_index in block_mask.kv_indices[
            0, 0, q_block_index, :block_count
        ].tolist():
            k_offsets = torch.arange(
                int(k_block_index) * k_block,
                min((int(k_block_index) + 1) * k_block, effective.shape[1]),
            )[None, :]
            effective[q_offsets, k_offsets] |= block_mask.mask_mod(
                torch.zeros_like(q_offsets),
                torch.zeros_like(q_offsets),
                q_offsets,
                k_offsets,
            )


def test_sparse_block_mask_supports_non_monotonic_remote_k_indices() -> None:
    q_token_indices = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    k_token_indices = torch.tensor([0, 1, 6, 2, 3, 4], dtype=torch.long)
    block_mask = build_block_mask(
        FlexMaskSpec(
            q_len=int(q_token_indices.numel()),
            k_len=int(k_token_indices.numel()),
            block_size=(2, 2),
            slices=(
                AttnSlice(
                    q_range=TokenRange(start=0, end=int(q_token_indices.numel())),
                    k_range=TokenRange(start=0, end=int(k_token_indices.numel())),
                    mask_kind=AttnMaskKind.CAUSAL,
                    row_index=0,
                ),
            ),
            exact_mask=ExactMaskMetadata(
                q_token_indices=q_token_indices,
                k_token_indices=k_token_indices,
                cache_key="non-monotonic-k",
            ),
        ),
        group_ids=torch.ones(8, dtype=torch.long),
        parent_ids=torch.ones(8, dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert block_mask is not None
    q_indices = torch.arange(q_token_indices.numel())[:, None]
    k_indices = torch.arange(k_token_indices.numel())[None, :]

    actual = block_mask.mask_mod(
        torch.zeros_like(q_indices),
        torch.zeros_like(q_indices),
        q_indices,
        k_indices,
    )

    assert actual.equal(q_token_indices[:, None] >= k_token_indices[None, :])
    _assert_matches_torch_block_mask(block_mask)


def _assert_matches_torch_block_mask(
    block_mask: BlockMask,
    *,
    batch_size: int = 1,
) -> None:
    q_len, k_len = block_mask.seq_lengths
    reference = torch_block_mask(
        block_mask.mask_mod,
        B=batch_size,
        H=1,
        Q_LEN=q_len,
        KV_LEN=k_len,
        device="cpu",
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
    )
    assert _effective_block_mask(block_mask).equal(_effective_block_mask(reference))
    for counts_name, indices_name in (
        ("kv_num_blocks", "kv_indices"),
        ("full_kv_num_blocks", "full_kv_indices"),
        ("q_num_blocks", "q_indices"),
        ("full_q_num_blocks", "full_q_indices"),
    ):
        assert _block_entries(block_mask, counts_name, indices_name) == _block_entries(
            reference,
            counts_name,
            indices_name,
        )


def _block_entries(
    block_mask: BlockMask,
    counts_name: str,
    indices_name: str,
) -> set[tuple[int, int, int, int]]:
    counts = getattr(block_mask, counts_name)
    indices = getattr(block_mask, indices_name)
    if counts is None or indices is None:
        return set()
    entries = set()
    for batch_index in range(int(counts.shape[0])):
        for head_index in range(int(counts.shape[1])):
            for block_index in range(int(counts.shape[2])):
                block_count = int(counts[batch_index, head_index, block_index])
                for other_block in indices[
                    batch_index,
                    head_index,
                    block_index,
                    :block_count,
                ].tolist():
                    entries.add(
                        (
                            batch_index,
                            head_index,
                            block_index,
                            int(other_block),
                        )
                    )
    return entries


def _branching_prefix_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1, 1, 1, 2, 3, 4, 4]], dtype=torch.long),
        torch.tensor([[1, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
    )


def _reference_tree_mask(
    group_ids: torch.Tensor, parent_ids: torch.Tensor
) -> torch.Tensor:
    group_list = [int(value) for value in group_ids.tolist()]
    parent_by_group: dict[int, int | None] = {}
    for group_id, parent_id in zip(group_list, parent_ids.tolist(), strict=True):
        group_id = int(group_id)
        parent_id = int(parent_id)
        if group_id not in parent_by_group:
            parent_by_group[group_id] = None if parent_id == group_id else parent_id

    ancestors_by_group = {
        group_id: _ancestors(group_id, parent_by_group) for group_id in parent_by_group
    }
    dense = torch.zeros((len(group_list), len(group_list)), dtype=torch.bool)
    for q_pos, q_group in enumerate(group_list):
        allowed_groups = ancestors_by_group[q_group] | {q_group}
        for k_pos, k_group in enumerate(group_list):
            dense[q_pos, k_pos] = k_pos <= q_pos and k_group in allowed_groups
    return dense


def _ancestors(
    group_id: int,
    parent_by_group: dict[int, int | None],
) -> set[int]:
    ancestors: set[int] = set()
    cursor = parent_by_group[group_id]
    while cursor is not None and cursor not in ancestors:
        ancestors.add(cursor)
        cursor = parent_by_group.get(cursor)
    return ancestors
