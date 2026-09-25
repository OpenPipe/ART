"""Size-only retained-set mirror of the CP attention executor's recompute path."""

import pytest

pytest.importorskip("triton")

from art.megatron.context_parallel.executor import (  # noqa: E402
    minimum_retained_bytes_per_row,
    retained_stage_record_bytes,
)
from art.megatron.context_parallel.types import (  # noqa: E402
    RankRuntimePlan,
    StagePlan,
    TokenRange,
)

# Qwen3.6-35B-A3B attention: 16 query heads, 2 KV heads of 256, BF16, and the
# H200 flash block for a 256-wide head (128 query, 64 key rows).
GEOMETRY = dict(
    q_heads=16,
    kv_heads=2,
    head_dim=256,
    value_head_dim=256,
    element_size=2,
    block_size=(128, 64),
)
# Flex's output keeps its own LSE beside the normalized copy it returns;
# logical-length copies keep one.
Q, KV, FLEX, OUT, TAPE = 8192, 2048, 8192 + 2 * 64, 8192 + 64, 16 * 257 * 4


def stage(index, *, local, q, k, q_len=None, k_len=None, own=None, source=0):
    q_ranges = (
        (TokenRange(0, q),) if own is None or q == own else (TokenRange(1, 1 + q),)
    )
    return StagePlan(
        stage_index=index,
        source_rank=source,
        is_local_stage=local,
        slices=("slice",),  # ty: ignore[invalid-argument-type]
        owner_local_q_ranges=q_ranges if q else (),
        owner_local_k_ranges=(TokenRange(0, k),) if k else (),
        q_len=q if q_len is None else q_len,
        k_len=k if k_len is None else k_len,
    )


def plan(own, *stages):
    return RankRuntimePlan(
        rank=0,
        original_seq_len=own,
        token_layout_index=None,  # ty: ignore[invalid-argument-type]
        local_valid_lengths=(own,),
        local_row_ranges=(TokenRange(0, own),),
        stage_plans=stages,
        remote_dkv_reduce_plan=None,  # ty: ignore[invalid-argument-type]
    )


def test_aligned_single_stage_copies_views_without_output_copies():
    # Real-data rank 0: one aligned local stage. Contiguous copies of the
    # permuted Q/K/V views, flex output and LSE; no padding, copies or tape.
    rows = 52480
    retained = retained_stage_record_bytes(
        plan(rows, stage(0, local=True, q=rows, k=rows)), **GEOMETRY
    )
    assert retained == Q * rows + KV * rows + FLEX * rows  # 0.971 GB traced
    geometry = {k: v for k, v in GEOMETRY.items() if k != "block_size"}
    assert retained == rows * minimum_retained_bytes_per_row(**geometry)


def test_unaligned_single_stage_pads_and_copies_the_logical_output():
    # The tail chunk on the single-stage rank: the planner rounds both lengths
    # up to its 128-row block, so both pad.
    rows = 52481
    stage_len = 411 * 128
    retained = retained_stage_record_bytes(
        plan(
            rows, stage(0, local=True, q=rows, k=rows, q_len=stage_len, k_len=stage_len)
        ),
        **GEOMETRY,
    )
    assert retained == Q * stage_len + KV * stage_len + FLEX * stage_len + OUT * rows


def test_tiny_stage_pads_to_two_blocks():
    retained = retained_stage_record_bytes(
        plan(5, stage(0, local=True, q=5, k=5)), **GEOMETRY
    )
    assert retained == Q * 256 + KV * 128 + FLEX * 256 + OUT * 5


def test_single_head_views_are_already_contiguous():
    rows = 1024
    geometry = dict(GEOMETRY, q_heads=1, kv_heads=1)
    retained = retained_stage_record_bytes(
        plan(rows, stage(0, local=True, q=rows, k=rows)), **geometry
    )
    assert retained == (256 * 2 + 2 * 4) * rows
    del geometry["block_size"]
    assert minimum_retained_bytes_per_row(**geometry) == 256 * 2 + 2 * 4


def test_full_query_remote_stage_keeps_fetch_buffers_and_a_merge_tape():
    # Real-data rank 1: a local stage and a remote stage over all of its
    # queries. Planner lengths round up, so both stages pad.
    own, remote_k = 44314, 16504
    local = stage(0, local=True, q=own, k=own, q_len=44352, k_len=44352)
    remote = stage(1, local=False, q=own, k=remote_k, q_len=44352, k_len=16512)
    retained = retained_stage_record_bytes(plan(own, local, remote), **GEOMETRY)
    q_pad = 347 * 128  # 44,416, as flex's traced output size shows
    local_bytes = Q * q_pad + KV * 44352 + FLEX * q_pad + OUT * own
    remote_bytes = (
        Q * q_pad + KV * remote_k + KV * 16512 + FLEX * q_pad + OUT * own + TAPE * own
    )
    assert retained == local_bytes + remote_bytes
    assert 3.05e9 < retained < 3.10e9  # 3.07 GB traced


def test_partial_query_remote_stage_keeps_its_gather_and_a_partial_tape():
    # Random-data rank 1: a large local stage and a 768-row remote stage.
    own = 105153
    local = stage(0, local=True, q=own, k=own, q_len=105216, k_len=105216)
    remote = stage(1, local=False, q=768, k=20608, own=own)
    retained = retained_stage_record_bytes(plan(own, local, remote), **GEOMETRY)
    local_bytes = Q * 105216 + KV * 105216 + FLEX * 105216 + OUT * own
    # Aligned: the query gather and fetch buffers feed flex without copies.
    remote_bytes = Q * 768 + KV * 20608 + FLEX * 768 + TAPE * 768
    assert retained == local_bytes + remote_bytes


def test_empty_remote_stage_and_missing_local_stage():
    rows = 1024
    empty = stage(1, local=False, q=0, k=0)
    alone = retained_stage_record_bytes(
        plan(rows, stage(0, local=True, q=rows, k=rows), empty), **GEOMETRY
    )
    assert alone == Q * rows + KV * rows + FLEX * rows
    # Without a local stage, the first ready remote stage records no tape; the
    # mirror drops the smallest so it never under-counts the order.
    small = stage(1, local=False, q=256, k=256, own=rows)
    full = stage(2, local=False, q=rows, k=512)
    both = retained_stage_record_bytes(plan(rows, small, full), **GEOMETRY)
    without_small_tape = retained_stage_record_bytes(plan(rows, full), **GEOMETRY)
    assert both - without_small_tape == Q * 256 + KV * 256 + FLEX * 256 + TAPE * rows
    assert retained_stage_record_bytes(plan(rows), **GEOMETRY) == 0
