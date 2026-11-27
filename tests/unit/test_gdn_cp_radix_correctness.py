"""Regression tests for GDN + CP attention radix prefix-tree correctness fixes.

Covers:
1. Contiguous single-parent state broadcast in ``_select_state_rows``,
   ``_ConvTailScan.forward``, and ``fla_cp._scan_fwd_initial_state`` (P0).
2. Unified ``use_exp2=False`` + fused ``chunk_gated_delta_rule_fwd_intra``
   across local buckets and native CP GDN chains (P1).
3. ``K_BLOCK``-aligned sibling K/V boundaries in
   ``art.megatron.context_parallel.executor`` with exact block-mask parity
   and differentiable forward/backward gradients (P1).
"""

from __future__ import annotations

import math
import sys
import types as pytypes
from typing import Any

import pytest
import torch

pytest.importorskip("triton")

from art.megatron.gdn import fla_cp  # noqa: E402

# ---------------------------------------------------------------------------
# Fix 1 (P0): Contiguous parent state rows & stride-0 kernel safety
# ---------------------------------------------------------------------------


def test_fla_cp_scan_initial_state_materializes_stride0_tensors() -> None:
    base = torch.randn(1, 2, 4, 4, dtype=torch.float32, requires_grad=True)
    stride0 = base.expand(4, 2, 4, 4)
    assert stride0.stride(0) == 0
    assert not stride0.is_contiguous()

    # CP rank 0 (summary is None): h0.float() alone would be a no-op on FP32!
    rank0_initial = fla_cp._scan_fwd_initial_state(None, stride0)
    assert rank0_initial.is_contiguous()
    assert rank0_initial.stride(0) > 0
    assert torch.equal(rank0_initial, stride0.contiguous())

    # Non-zero CP rank with summary scan.
    summary = torch.zeros(4, 2, 4, 8, dtype=torch.float32)
    summary[..., 4:] = torch.eye(4, dtype=torch.float32)
    rankn_initial = fla_cp._scan_fwd_initial_state(summary, stride0)
    assert rankn_initial.is_contiguous()
    assert rankn_initial.stride(0) > 0
    assert torch.allclose(rankn_initial, stride0.contiguous())

    # Backward gradient helpers also return contiguous tensors.
    bwd_rank_last = fla_cp._scan_bwd_local_final_grad(None, stride0)
    assert bwd_rank_last.is_contiguous()
    assert bwd_rank_last.stride(0) > 0
    bwd_init = fla_cp._scan_bwd_initial_state_grad(summary, stride0)
    assert bwd_init.is_contiguous()
    assert bwd_init.stride(0) > 0


def test_conv_tail_scan_forward_materializes_non_contiguous_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    from art.megatron.gdn import operator

    captured_fwd: list[tuple[torch.Tensor, ...]] = []
    captured_bwd: list[tuple[torch.Tensor, ...]] = []

    class _CaptureKernel:
        def __init__(self, sink: list[tuple[torch.Tensor, ...]]) -> None:
            self._sink = sink

        def __getitem__(self, grid: Any) -> Any:
            del grid

            def _launch(*args: Any, **kwargs: Any) -> None:
                del kwargs
                tensors = tuple(arg for arg in args if isinstance(arg, torch.Tensor))
                self._sink.append(tensors)

            return _launch

    monkeypatch.setattr(
        operator, "_conv_tail_scan_kernel", _CaptureKernel(captured_fwd)
    )
    monkeypatch.setattr(
        operator, "_conv_tail_scan_backward_kernel", _CaptureKernel(captured_bwd)
    )

    parent_base = torch.randn(1, 8, 3, dtype=torch.float32, requires_grad=True)
    parent_stride0 = parent_base.expand(4, 8, 3)
    tails = (
        torch.randn(2, 4, 3, 8, dtype=torch.float32)
        .transpose(2, 3)
        .requires_grad_(True)
    )
    source_ids = torch.zeros(3, 4, dtype=torch.int32).transpose(0, 1)
    source_pos = torch.zeros(3, 4, dtype=torch.int32).transpose(0, 1)

    assert not parent_stride0.is_contiguous()
    assert not tails.is_contiguous()
    assert not source_ids.is_contiguous()
    assert not source_pos.is_contiguous()

    out = operator._ConvTailScan.apply(parent_stride0, tails, source_ids, source_pos)
    assert len(captured_fwd) == 1
    fwd_parent, fwd_tails, fwd_sids, fwd_spos, fwd_out = captured_fwd[0]
    for tensor in (fwd_parent, fwd_tails, fwd_sids, fwd_spos, fwd_out):
        assert tensor.is_contiguous()
    assert fwd_parent.stride(0) > 0

    out.sum().backward()
    assert len(captured_bwd) == 1
    for tensor in captured_bwd[0]:
        assert tensor.is_contiguous()


# ---------------------------------------------------------------------------
# Fix 3 (P1): Unified FLA intra-chunk kernel & use_exp2=False
# ---------------------------------------------------------------------------


def test_local_chunk_gated_delta_rule_uses_use_exp2_false_and_segmented_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    from art.megatron import segmented_gdn
    from art.megatron.gdn import operator

    recorded: list[tuple[str, dict[str, Any]]] = []
    delegated: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _fake_fwd(**kw: Any) -> tuple[Any, ...]:
        recorded.append(
            (
                "fwd",
                {
                    "use_exp2": kw["use_exp2"],
                    "h0_contig": (
                        kw["initial_state"].is_contiguous()
                        if kw["initial_state"] is not None
                        else None
                    ),
                    "active": segmented_gdn._ACTIVE.get(),
                },
            )
        )
        o = (
            kw["q"]
            + kw["k"]
            + kw["v"]
            + kw["g"].unsqueeze(-1)
            + kw["beta"].unsqueeze(-1)
        )
        final_state = (
            kw["initial_state"] * 2.0
            if kw["output_final_state"] and kw["initial_state"] is not None
            else None
        )
        return kw["g"], o, kw["k"], final_state, kw["initial_state"], None

    def _fake_bwd(**kw: Any) -> tuple[Any, ...]:
        recorded.append(
            (
                "bwd",
                {
                    "use_exp2": kw["use_exp2"],
                    "active": segmented_gdn._ACTIVE.get(),
                },
            )
        )
        do = kw["do"]
        dh0 = (
            kw["dht"] * 2.0
            if kw["dht"] is not None
            else (
                torch.zeros_like(kw["initial_state"])
                if kw["initial_state"] is not None
                else None
            )
        )
        return do, do, do, do.sum(dim=-1), do.sum(dim=-1), dh0, None, None

    def _fake_public_gdr(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        delegated.append((args, kwargs))
        return args[0], None

    fake_fla = pytypes.ModuleType("fla")
    fake_fla_ops = pytypes.ModuleType("fla.ops")
    fake_fla_gdr = pytypes.ModuleType("fla.ops.gated_delta_rule")
    setattr(fake_fla_gdr, "chunk_gated_delta_rule", _fake_public_gdr)
    fake_fla_gdr_chunk = pytypes.ModuleType("fla.ops.gated_delta_rule.chunk")
    setattr(fake_fla_gdr_chunk, "chunk_gated_delta_rule_fwd", _fake_fwd)
    setattr(fake_fla_gdr_chunk, "chunk_gated_delta_rule_bwd", _fake_bwd)
    fake_fla_ops_utils = pytypes.ModuleType("fla.ops.utils")
    setattr(fake_fla_ops_utils, "prepare_chunk_indices", lambda *a, **kw: None)

    for name, mod in {
        "fla": fake_fla,
        "fla.ops": fake_fla_ops,
        "fla.ops.gated_delta_rule": fake_fla_gdr,
        "fla.ops.gated_delta_rule.chunk": fake_fla_gdr_chunk,
        "fla.ops.utils": fake_fla_ops_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(operator, "ensure_blackwell_fwd_h_configs", lambda: None)

    q = torch.randn(2, 4, 2, 8, requires_grad=True)
    k = torch.randn(2, 4, 2, 8, requires_grad=True)
    v = torch.randn(2, 4, 2, 8, requires_grad=True)
    g = torch.randn(2, 4, 2, requires_grad=True)
    beta = torch.randn(2, 4, 2, requires_grad=True)
    h0_base = torch.randn(1, 2, 8, 8, requires_grad=True)
    h0_stride0 = h0_base.expand(2, 2, 8, 8)
    assert not h0_stride0.is_contiguous()

    # Forward runs inside native_gdn_reductions scope; backward runs AFTER exit
    # and must still observe `_ACTIVE == True` via ctx._art_native_gdn_reductions.
    token = segmented_gdn._ACTIVE.set(True)
    try:
        out, h_final = operator._chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=h0_stride0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    finally:
        segmented_gdn._ACTIVE.reset(token)

    assert not segmented_gdn._ACTIVE.get()
    assert h_final is not None
    (out.sum() + h_final.sum()).backward()
    assert not segmented_gdn._ACTIVE.get()
    assert recorded == [
        ("fwd", {"use_exp2": False, "h0_contig": True, "active": True}),
        ("bwd", {"use_exp2": False, "active": True}),
    ]
    assert h0_base.grad is not None
    assert torch.allclose(h0_base.grad, torch.full_like(h0_base, 4.0))

    # Non-standard flags still delegate to public `chunk_gated_delta_rule`.
    operator._chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True
    )
    operator._chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_exp2=True)
    assert len(delegated) == 2


def test_native_cp_forward_uses_fused_fwd_intra_and_use_exp2_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, dict[str, Any]]] = []

    def _fake_fwd_intra(**kw: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recorded.append(("fwd_intra", {"use_exp2": kw["use_exp2"]}))
        return kw["k"], kw["v"], kw["k"]

    def _fake_fwd_h(**kw: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recorded.append(
            (
                "fwd_h",
                {
                    "use_exp2": kw["use_exp2"],
                    "h0_contig": kw["initial_state"].is_contiguous(),
                    "h0_stride0": kw["initial_state"].stride(0),
                },
            )
        )
        return kw["initial_state"], kw["u"], kw["initial_state"]

    def _fake_fwd_o(**kw: Any) -> torch.Tensor:
        recorded.append(("fwd_o", {"use_exp2": kw["use_exp2"]}))
        return kw["q"] + kw["v"]

    fake_fla = pytypes.ModuleType("fla")
    fake_fla_ops = pytypes.ModuleType("fla.ops")
    fake_fla_common = pytypes.ModuleType("fla.ops.common")
    fake_fla_delta_h = pytypes.ModuleType("fla.ops.common.chunk_delta_h")
    setattr(fake_fla_delta_h, "chunk_gated_delta_rule_fwd_h", _fake_fwd_h)
    fake_fla_chunk_o = pytypes.ModuleType("fla.ops.common.chunk_o")
    setattr(fake_fla_chunk_o, "chunk_fwd_o", _fake_fwd_o)
    fake_fla_gdr = pytypes.ModuleType("fla.ops.gated_delta_rule")
    fake_fla_gdr_fwd = pytypes.ModuleType("fla.ops.gated_delta_rule.chunk_fwd")
    setattr(fake_fla_gdr_fwd, "chunk_gated_delta_rule_fwd_intra", _fake_fwd_intra)
    fake_fla_ops_utils = pytypes.ModuleType("fla.ops.utils")
    setattr(fake_fla_ops_utils, "chunk_local_cumsum", lambda g, **kw: g)
    setattr(fake_fla_ops_utils, "prepare_chunk_indices", lambda *a, **kw: None)

    for name, mod in {
        "fla": fake_fla,
        "fla.ops": fake_fla_ops,
        "fla.ops.common": fake_fla_common,
        "fla.ops.common.chunk_delta_h": fake_fla_delta_h,
        "fla.ops.common.chunk_o": fake_fla_chunk_o,
        "fla.ops.gated_delta_rule": fake_fla_gdr,
        "fla.ops.gated_delta_rule.chunk_fwd": fake_fla_gdr_fwd,
        "fla.ops.utils": fake_fla_ops_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setattr(fla_cp, "ensure_blackwell_fwd_h_configs", lambda: None)
    monkeypatch.setattr(fla_cp, "_fwd_summary", lambda **kw: kw["k"])
    monkeypatch.setattr(fla_cp, "_prefix_summary_exclusive", lambda s, g: None)
    monkeypatch.setattr(fla_cp, "_broadcast_chain_final_state", lambda s, g: s)

    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 8, 2, 4)
    v = torch.randn(1, 8, 2, 4)
    g = torch.randn(1, 8, 2)
    beta = torch.randn(1, 8, 2)
    h0_stride0 = torch.randn(1, 2, 4, 4, dtype=torch.float32).expand(2, 2, 4, 4)
    assert h0_stride0.stride(0) == 0

    ctx = pytypes.SimpleNamespace(save_for_backward=lambda *tensors: None)
    out, final_state = fla_cp._NativeCpChunkGatedDeltaRule.forward(
        ctx, q, k, v, g, beta, h0_stride0, None, None, None, True, 0.5
    )
    assert out.shape == q.shape
    assert final_state is not None and final_state.shape == h0_stride0.shape
    assert recorded == [
        ("fwd_intra", {"use_exp2": False}),
        ("fwd_h", {"use_exp2": False, "h0_contig": True, "h0_stride0": 32}),
        ("fwd_o", {"use_exp2": False}),
    ]


# ---------------------------------------------------------------------------
# Fix 4 (P1): K_BLOCK-aligned sibling boundaries in CP attention executor
# ---------------------------------------------------------------------------


def _import_cp_executor() -> tuple[Any, Any]:
    pytest.importorskip("triton")
    pytest.importorskip("megatron.core.packed_seq_params")
    from art.megatron.context_parallel import executor, types

    return executor, types


def test_compute_group_aligned_k_map_fast_path_and_s05_layout() -> None:
    executor, _ = _import_cp_executor()

    # Single group or already k_block-aligned -> fast-path None.
    assert (
        executor._compute_group_aligned_k_map(list(range(128)), [0] * 128, k_block=64)
        is None
    )
    assert (
        executor._compute_group_aligned_k_map(
            list(range(128)), [0] * 64 + [1] * 64, k_block=64
        )
        is None
    )
    # Degenerate or out-of-range inputs -> safe None.
    assert executor._compute_group_aligned_k_map([], [], k_block=64) is None
    assert executor._compute_group_aligned_k_map([0], [0], k_block=64) is None
    assert executor._compute_group_aligned_k_map([0, 1], [0, 1], k_block=1) is None
    assert executor._compute_group_aligned_k_map([0, -1], [0, 1], k_block=64) is None
    assert executor._compute_group_aligned_k_map([0, 99], [0, 1], k_block=64) is None

    # Exact S0-5 4x63 sibling completions after 1024-token prefix tail on rank 3.
    group_ids = [0] * 1024 + [1] * 63 + [2] * 63 + [3] * 63 + [4] * 63
    res = executor._compute_group_aligned_k_map(
        list(range(len(group_ids))), group_ids, k_block=64
    )
    assert res is not None
    dst, aligned_len = res
    assert len(dst) == len(group_ids)
    assert dst[0] == 0 and dst[1023] == 1023
    assert dst[1024] == 1024 and dst[1024 + 62] == 1086
    assert dst[1024 + 63] == 1088 and dst[1024 + 125] == 1150
    assert dst[1024 + 126] == 1152 and dst[1024 + 188] == 1214
    assert dst[1024 + 189] == 1216 and dst[1024 + 251] == 1278
    assert aligned_len == 1279


def test_remap_slices_for_k_alignment_splits_across_pad_gaps() -> None:
    executor, cp_types = _import_cp_executor()

    dst = (0, 1, 2, 4, 5, 6)
    slices = (
        cp_types.AttnSlice(
            q_range=cp_types.TokenRange(0, 3),
            k_range=cp_types.TokenRange(0, 3),
            mask_kind=cp_types.AttnMaskKind.CAUSAL,
            row_index=0,
            family_index=0,
        ),
        cp_types.AttnSlice(
            q_range=cp_types.TokenRange(3, 6),
            k_range=cp_types.TokenRange(0, 6),
            mask_kind=cp_types.AttnMaskKind.FULL,
            row_index=0,
            family_index=1,
        ),
        cp_types.AttnSlice(
            q_range=cp_types.TokenRange(3, 6),
            k_range=cp_types.TokenRange(3, 6),
            mask_kind=cp_types.AttnMaskKind.CAUSAL,
            row_index=0,
            family_index=1,
        ),
    )
    remapped = executor._remap_slices_for_k_alignment(slices, dst)
    assert [(s.k_range.start, s.k_range.end, s.mask_kind) for s in remapped] == [
        (0, 3, cp_types.AttnMaskKind.CAUSAL),
        (0, 3, cp_types.AttnMaskKind.FULL),
        (4, 7, cp_types.AttnMaskKind.FULL),
        (4, 7, cp_types.AttnMaskKind.CAUSAL),
    ]
    for sl in remapped:
        assert 3 not in range(sl.k_range.start, sl.k_range.end)


@pytest.mark.parametrize("sliding_window", [None, 12])
def test_kblock_aligned_stage_block_mask_matches_torch_reference(
    sliding_window: int | None,
) -> None:
    from torch.nn.attention.flex_attention import create_block_mask as torch_block_mask

    executor, cp_types = _import_cp_executor()
    from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
    from art.megatron.context_parallel.layout_index import TokenLayoutIndex

    # 16-token prefix (group 1) + two 7-token siblings (groups 2 and 3, parent 1).
    # With k_block=8: group 2 starts at 16 (aligned), group 3 starts at 23 -> shifts to 24!
    group_ids = torch.tensor([1] * 16 + [2] * 7 + [3] * 7, dtype=torch.int64)
    parent_ids = torch.tensor([1] * 16 + [1] * 7 + [1] * 7, dtype=torch.int64)
    input_pos = torch.cat(
        [torch.arange(16), torch.arange(16, 23), torch.arange(16, 23)]
    ).to(dtype=torch.int64)
    total_len = int(group_ids.numel())

    row_spec = build_prefix_tree_attention_spec(
        group_ids=group_ids.unsqueeze(0),
        parent_ids=parent_ids.unsqueeze(0),
    ).rows[0]
    stage_plan = cp_types.StagePlan(
        stage_index=0,
        source_rank=0,
        is_local_stage=True,
        slices=row_spec.slices,
        owner_local_q_ranges=(cp_types.TokenRange(0, total_len),),
        owner_local_k_ranges=(cp_types.TokenRange(0, total_len),),
        q_len=total_len,
        k_len=total_len,
        mask_metadata=cp_types.ExactMaskMetadata(
            q_token_indices=torch.arange(total_len, dtype=torch.int64),
            k_token_indices=torch.arange(total_len, dtype=torch.int64),
            cache_key="stage0",
        ),
    )
    rank_plan = cp_types.RankRuntimePlan(
        rank=0,
        original_seq_len=total_len,
        token_layout_index=TokenLayoutIndex((((0, total_len, 0),),), (total_len,)),
        local_valid_lengths=(total_len,),
        local_row_ranges=(cp_types.TokenRange(0, total_len),),
        stage_plans=(stage_plan,),
        remote_dkv_reduce_plan=cp_types.DkvReducePlan(
            send_splits=(0,), recv_splits=(0,), recv_ranges_by_peer=((),)
        ),
    )
    state = cp_types.ArtContextParallelState(
        rank_plan=rank_plan,
        cp_group=None,
        config=cp_types.ContextParallelConfig(
            block_size=8, attention_sparse_block_size=(8, 8)
        ),
        group_ids=group_ids,
        parent_ids=parent_ids,
        input_pos=input_pos,
        block_mask_variants=(
            cp_types.CpBlockMaskVariant(
                block_size=(8, 8), sliding_window=sliding_window
            ),
        ),
    )

    spec = executor._resolve_stage_execution_spec(
        stage_plan=stage_plan, state=state, block_size=(8, 8)
    )
    assert spec.k_align_dst_indices is not None
    assert spec.mask_metadata is not None
    assert spec.slices is not None
    # Slot 23 is the inserted -1 gap; group 3 starts aligned at slot 24.
    assert int(spec.mask_metadata.k_token_indices[23]) == -1
    assert spec.mask_metadata.k_token_indices[24:31].tolist() == list(range(23, 30))
    for sl in spec.slices:
        assert (
            spec.mask_metadata.k_token_indices[sl.k_range.start : sl.k_range.end] >= 0
        ).all()

    mask = executor._build_stage_block_mask(
        stage_plan=stage_plan,
        state=state,
        device=torch.device("cpu"),
        execution_spec=spec,
        block_size=(8, 8),
        sliding_window=sliding_window,
    )
    assert mask is not None
    reference = torch_block_mask(
        mask.mask_mod,
        B=1,
        H=1,
        Q_LEN=mask.seq_lengths[0],
        KV_LEN=mask.seq_lengths[1],
        device="cpu",
        BLOCK_SIZE=mask.BLOCK_SIZE,
    )
    # Exact partial and full block equivalence with PyTorch's reference builder.
    for attr in (
        "kv_num_blocks",
        "full_kv_num_blocks",
        "q_num_blocks",
        "full_q_num_blocks",
    ):
        assert torch.equal(getattr(mask, attr), getattr(reference, attr))
    for counts_attr, idx_attr in (
        ("kv_num_blocks", "kv_indices"),
        ("full_kv_num_blocks", "full_kv_indices"),
        ("q_num_blocks", "q_indices"),
        ("full_q_num_blocks", "full_q_indices"),
    ):
        c = getattr(mask, counts_attr)[0, 0]
        actual_idx = getattr(mask, idx_attr)[0, 0]
        ref_idx = getattr(reference, idx_attr)[0, 0]
        for row in range(int(c.numel())):
            k = int(c[row])
            assert set(actual_idx[row, :k].tolist()) == set(ref_idx[row, :k].tolist())


def test_kblock_aligned_stage_attention_exact_fwd_and_bwd_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, cp_types = _import_cp_executor()
    from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
    from art.megatron.context_parallel.layout_index import TokenLayoutIndex

    # 16-token prefix + two 7-token siblings -> unaligned transition at 23 -> shifts to 24.
    group_ids = torch.tensor([1] * 16 + [2] * 7 + [3] * 7, dtype=torch.int64)
    parent_ids = torch.tensor([1] * 16 + [1] * 7 + [1] * 7, dtype=torch.int64)
    input_pos = torch.cat(
        [torch.arange(16), torch.arange(16, 23), torch.arange(16, 23)]
    ).to(dtype=torch.int64)
    total_len = int(group_ids.numel())

    row_spec = build_prefix_tree_attention_spec(
        group_ids=group_ids.unsqueeze(0),
        parent_ids=parent_ids.unsqueeze(0),
    ).rows[0]
    stage_plan = cp_types.StagePlan(
        stage_index=0,
        source_rank=0,
        is_local_stage=True,
        slices=row_spec.slices,
        owner_local_q_ranges=(cp_types.TokenRange(0, total_len),),
        owner_local_k_ranges=(cp_types.TokenRange(0, total_len),),
        q_len=total_len,
        k_len=total_len,
        mask_metadata=cp_types.ExactMaskMetadata(
            q_token_indices=torch.arange(total_len, dtype=torch.int64),
            k_token_indices=torch.arange(total_len, dtype=torch.int64),
            cache_key="stage0",
        ),
    )
    rank_plan = cp_types.RankRuntimePlan(
        rank=0,
        original_seq_len=total_len,
        token_layout_index=TokenLayoutIndex((((0, total_len, 0),),), (total_len,)),
        local_valid_lengths=(total_len,),
        local_row_ranges=(cp_types.TokenRange(0, total_len),),
        stage_plans=(stage_plan,),
        remote_dkv_reduce_plan=cp_types.DkvReducePlan(
            send_splits=(0,), recv_splits=(0,), recv_ranges_by_peer=((),)
        ),
    )
    state = cp_types.ArtContextParallelState(
        rank_plan=rank_plan,
        cp_group=None,
        config=cp_types.ContextParallelConfig(
            block_size=8, attention_sparse_block_size=(8, 8)
        ),
        group_ids=group_ids,
        parent_ids=parent_ids,
        input_pos=input_pos,
        block_mask_variants=(cp_types.CpBlockMaskVariant(block_size=(8, 8)),),
    )

    monkeypatch.setattr(executor, "_stage_sparse_block_size", lambda q, v: (8, 8))
    executor.prepare_context_parallel_execution_state(
        state=state, device=torch.device("cpu")
    )
    # Prepare step pre-populates the device dst-index cache for zero-H2D forward.
    assert len(state.execution_cache.k_align_index_tensors) == 1

    spec = executor._resolve_stage_execution_spec(
        stage_plan=stage_plan, state=state, block_size=(8, 8)
    )
    assert spec.mask_metadata is not None
    k_abs = spec.mask_metadata.k_token_indices

    class _ExactDenseKernel:
        def __init__(self) -> None:
            self.seen_k_lens: list[int] = []

        def run(
            self,
            q_flex: torch.Tensor,
            k_flex: torch.Tensor,
            v_flex: torch.Tensor,
            *,
            block_mask: Any,
            scale: float,
            enable_gqa: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del enable_gqa
            q_len, k_len = block_mask.seq_lengths
            self.seen_k_lens.append(int(k_flex.shape[2]))
            pad_slots = k_abs < 0
            assert torch.equal(
                k_flex[:, :, pad_slots, :], torch.zeros_like(k_flex[:, :, pad_slots, :])
            )
            assert torch.equal(
                v_flex[:, :, pad_slots, :], torch.zeros_like(v_flex[:, :, pad_slots, :])
            )
            qi = torch.arange(q_len)[:, None]
            ki = torch.arange(k_len)[None, :]
            dense_mask = block_mask.mask_mod(torch.tensor(0), torch.tensor(0), qi, ki)
            scores = torch.matmul(q_flex, k_flex.transpose(-1, -2)) * scale
            scores = scores.masked_fill(
                ~dense_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
            valid_row = dense_mask.any(dim=-1)
            lse = torch.where(
                valid_row.unsqueeze(0).unsqueeze(0),
                torch.logsumexp(scores, dim=-1),
                torch.full_like(scores[..., 0], float("-inf")),
            )
            probs = torch.where(
                valid_row.unsqueeze(0).unsqueeze(0).unsqueeze(-1),
                torch.softmax(scores, dim=-1),
                torch.zeros_like(scores),
            )
            return torch.matmul(probs, v_flex), lse

    # Build unaligned dense boolean mask for reference attention.
    from art.megatron.context_parallel.builder import build_dense_reference_mask

    ref_dense_mask = build_dense_reference_mask(row_spec=row_spec)
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(8)
    for head_major in (True, False):
        shape = (2, total_len, 8) if head_major else (total_len, 2, 8)
        q_ref = torch.randn(shape, dtype=torch.float32, requires_grad=True)
        k_ref = torch.randn(shape, dtype=torch.float32, requires_grad=True)
        v_ref = torch.randn(shape, dtype=torch.float32, requires_grad=True)
        q_fix = q_ref.detach().clone().requires_grad_(True)
        k_fix = k_ref.detach().clone().requires_grad_(True)
        v_fix = v_ref.detach().clone().requires_grad_(True)

        qf = q_ref.unsqueeze(0) if head_major else q_ref.permute(1, 0, 2).unsqueeze(0)
        kf = k_ref.unsqueeze(0) if head_major else k_ref.permute(1, 0, 2).unsqueeze(0)
        vf = v_ref.unsqueeze(0) if head_major else v_ref.permute(1, 0, 2).unsqueeze(0)
        scores_r = (torch.matmul(qf, kf.transpose(-1, -2)) * scale).masked_fill(
            ~ref_dense_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        lse_r = torch.logsumexp(scores_r, dim=-1)
        out_r = torch.matmul(torch.softmax(scores_r, dim=-1), vf)
        out_r = out_r.squeeze(0) if head_major else out_r.squeeze(0).permute(1, 0, 2)
        lse_r = lse_r.squeeze(0) if head_major else lse_r.squeeze(0).permute(1, 0)

        kernel = _ExactDenseKernel()
        out_f, lse_f = executor._run_stage_attention(
            q_stage=q_fix,
            k_stage=k_fix,
            v_stage=v_fix,
            stage_plan=stage_plan,
            state=state,
            kernel=kernel,  # type: ignore[arg-type]
            scale=scale,
            enable_gqa=False,
            sliding_window=None,
        )
        assert kernel.seen_k_lens == [spec.k_len]
        assert torch.allclose(out_f, out_r, atol=1e-6, rtol=1e-6)
        assert torch.allclose(lse_f, lse_r, atol=1e-6, rtol=1e-6)

        w_out = torch.randn_like(out_r)
        w_lse = torch.randn_like(lse_r)
        ((out_r * w_out).sum() + (lse_r * w_lse).sum()).backward()
        ((out_f * w_out).sum() + (lse_f * w_lse).sum()).backward()
        assert q_fix.grad is not None and q_ref.grad is not None
        assert k_fix.grad is not None and k_ref.grad is not None
        assert v_fix.grad is not None and v_ref.grad is not None
        assert torch.allclose(q_fix.grad, q_ref.grad, atol=1e-5, rtol=1e-5)
        assert torch.allclose(k_fix.grad, k_ref.grad, atol=1e-5, rtol=1e-5)
        assert torch.allclose(v_fix.grad, v_ref.grad, atol=1e-5, rtol=1e-5)

    # Fast path: single-group stage leaves k_align_dst_indices as None.
    single_stage = cp_types.StagePlan(
        stage_index=1,
        source_rank=0,
        is_local_stage=True,
        slices=(
            cp_types.AttnSlice(
                q_range=cp_types.TokenRange(0, 16),
                k_range=cp_types.TokenRange(0, 16),
                mask_kind=cp_types.AttnMaskKind.CAUSAL,
                row_index=0,
            ),
        ),
        owner_local_q_ranges=(cp_types.TokenRange(0, 16),),
        owner_local_k_ranges=(cp_types.TokenRange(0, 16),),
        q_len=16,
        k_len=16,
        mask_metadata=cp_types.ExactMaskMetadata(
            q_token_indices=torch.arange(16, dtype=torch.int64),
            k_token_indices=torch.arange(16, dtype=torch.int64),
            cache_key="single",
        ),
    )
    single_state = cp_types.ArtContextParallelState(
        rank_plan=rank_plan,
        cp_group=None,
        config=cp_types.ContextParallelConfig(
            block_size=8, attention_sparse_block_size=(8, 8)
        ),
        group_ids=torch.ones(16, dtype=torch.int64),
        parent_ids=torch.ones(16, dtype=torch.int64),
        input_pos=torch.arange(16, dtype=torch.int64),
    )
    single_spec = executor._resolve_stage_execution_spec(
        stage_plan=single_stage, state=single_state, block_size=(8, 8)
    )
    assert single_spec.k_align_dst_indices is None
    assert single_spec.slices is None
