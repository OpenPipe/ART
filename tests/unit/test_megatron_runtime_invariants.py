"""Unit tests for upstreamed Megatron trainer runtime invariants.

Covers all five native runtime invariants:
1. Batch-invariant TE RMSNorm / LayerNormLinear mod-4 row padding, TE transpose
   cache disabling, and Dynamo ``assume_static_by_default = False``.
2. Sub-block single-rank CP plan assignment (``valid_tokens < max(block_size, cp)``).
3. ``@torch.compiler.disable`` on ``LoRA._active_lora_storage`` (zero slot-name recompiles).
4. Short-sequence (``<= 256``) grad-enabled deterministic Triton attention backward
   with FLASH forward via a distinct compiled ``__code__`` object.
5. TileLang dense-to-varlen ``chunk_bwd_dqkwg_tilelang`` canonicalization onto
   ``B=1, IS_VARLEN=True``.
"""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters
from torch.nn.attention.flex_attention import create_block_mask

from art.megatron.flex_attn import compiled as flex_compiled
from art.megatron.flex_attn import flash_dlse_patch
from art.megatron.fp32_residual import (
    _call_with_mod4_row_padding,
    disable_te_parameter_transpose_cache,
)
from art.megatron.gdn.fla_tilelang import (
    _TILELANG_VARLEN_INDEX_CACHE_MAX,
    ensure_tilelang_varlen_bwd,
)

# ---------------------------------------------------------------------------
# 1. Batch-invariant TE RMSNorm mod-4 padding, transpose cache & Dynamo config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 7, 8])
def test_mod4_row_padding_pads_and_preserves_exact_fwd_and_bwd(
    ndim: int, rows: int
) -> None:
    torch.manual_seed(100 + ndim * 17 + rows)
    seen_rows: list[int] = []
    weight = torch.randn(8, 8, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(8, dtype=torch.float32, requires_grad=True)

    def _fake_ln_linear_fwd(
        _mod: Any, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seen_rows.append(int(x.shape[0]))
        return x @ weight.T + bias, bias

    mod = SimpleNamespace(sequence_parallel=False)
    shape = (rows, 8) if ndim == 2 else (rows, 1, 8) if ndim == 3 else (rows, 1, 2, 8)
    x = torch.randn(shape, dtype=torch.float32, requires_grad=True)

    out, ret_bias = _call_with_mod4_row_padding(_fake_ln_linear_fwd, mod, x)
    expected_padded = rows if rows % 4 == 0 else rows + (4 - rows % 4)
    assert seen_rows == [expected_padded]
    assert out.shape == shape
    assert ret_bias is bias

    ref_x = x.detach().clone().requires_grad_(True)
    ref_w = weight.detach().clone().requires_grad_(True)
    ref_b = bias.detach().clone().requires_grad_(True)
    ref_out = ref_x @ ref_w.T + ref_b
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-6)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    ref_out.backward(grad_out)
    assert x.grad is not None and ref_x.grad is not None
    assert weight.grad is not None and ref_w.grad is not None
    assert bias.grad is not None and ref_b.grad is not None
    # Padded rows must contribute zero gradients to x, weight, and bias.
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight.grad, ref_w.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bias.grad, ref_b.grad, rtol=1e-5, atol=1e-6)


def test_mod4_row_padding_slices_nested_outputs_and_skips_sp_and_empty() -> None:
    seen_rows: list[int] = []

    def _nested_fwd(_mod: Any, x: torch.Tensor) -> Any:
        seen_rows.append(int(x.shape[0]))
        bias = torch.ones((x.shape[-1],), dtype=x.dtype)
        return ((x + 1.0, x * 3.0), bias)

    x5 = torch.arange(5 * 1 * 8, dtype=torch.float32).reshape(5, 1, 8)
    (out_l, ln_out_l), bias_l = _call_with_mod4_row_padding(
        _nested_fwd, SimpleNamespace(sequence_parallel=False), x5
    )
    assert seen_rows == [8]
    assert tuple(out_l.shape) == (5, 1, 8)
    assert tuple(ln_out_l.shape) == (5, 1, 8)
    assert tuple(bias_l.shape) == (8,)
    assert torch.equal(out_l, x5 + 1.0)
    assert torch.equal(ln_out_l, x5 * 3.0)

    # sequence_parallel=True leaves rows unpadded.
    (out_sp, _), _ = _call_with_mod4_row_padding(
        _nested_fwd, SimpleNamespace(sequence_parallel=True), x5
    )
    assert seen_rows == [8, 5]
    assert tuple(out_sp.shape) == (5, 1, 8)

    # Empty tensor (0 rows) leaves rows unpadded.
    x0 = torch.empty((0, 1, 8), dtype=torch.float32)
    (out_0, _), _ = _call_with_mod4_row_padding(
        _nested_fwd, SimpleNamespace(sequence_parallel=False), x0
    )
    assert seen_rows == [8, 5, 0]
    assert tuple(out_0.shape) == (0, 1, 8)


def test_mod4_row_padding_disabled_for_dynamo_avoids_mod4_recompiles() -> None:
    torch._dynamo.reset()
    backend = CompileCounterWithBackend("eager")

    def _inner(_mod: Any, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0

    mod = SimpleNamespace(sequence_parallel=False)

    def _fn(x: torch.Tensor) -> torch.Tensor:
        y = _call_with_mod4_row_padding(_inner, mod, x)
        return y + 1.0

    compiled_fn = torch.compile(_fn, backend=backend)
    prev_static = torch._dynamo.config.assume_static_by_default
    try:
        torch._dynamo.config.assume_static_by_default = False
        first = compiled_fn(torch.ones(5, 1, 8))
        assert torch.equal(first, torch.full((5, 1, 8), 3.0))
        frames = backend.frame_count
        with torch._dynamo.config.patch(error_on_recompile=True):
            # Test all residues mod 4 (>= 2 to avoid Dynamo's built-in 0/1 specialization).
            for rows in (2, 3, 4, 6, 7, 8, 9, 13):
                out = compiled_fn(torch.ones(rows, 1, 8))
                assert torch.equal(out, torch.full((rows, 1, 8), 3.0))
        assert backend.frame_count == frames
    finally:
        torch._dynamo.config.assume_static_by_default = prev_static
        torch._dynamo.reset()


def test_disable_te_parameter_transpose_cache_marks_all_matching_modules() -> None:
    class _WithCache(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.disable_parameter_transpose_cache = False

    class _WithoutCache(torch.nn.Module):
        pass

    a = _WithCache()
    b = _WithCache()
    c = _WithoutCache()
    chunk = torch.nn.Sequential(a, torch.nn.ModuleList([b, c]))

    disable_te_parameter_transpose_cache([chunk, "non-module"])
    assert a.disable_parameter_transpose_cache is True
    assert b.disable_parameter_transpose_cache is True
    assert not hasattr(c, "disable_parameter_transpose_cache")


def test_patched_te_modules_pad_bf16_rows_mod4_and_skip_sp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("megatron.core")
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
    )
    import transformer_engine.pytorch as te

    from art.megatron.fp32_residual import install_fp32_residual_patches

    install_fp32_residual_patches()
    assert getattr(te.RMSNorm.forward, "__art_fp32_residual__", False)
    assert getattr(te.LayerNormLinear.forward, "__art_fp32_residual__", False)
    assert getattr(
        TELayerNormColumnParallelLinear.forward, "__art_fp32_residual__", False
    )

    seen_rms_rows: list[int] = []
    monkeypatch.setattr(
        "transformer_engine.pytorch.ops.fuser.OperationFuser.__call__",
        lambda self, x, *a, **kw: (seen_rms_rows.append(int(x.shape[0])), x * 2.0)[1],
    )
    norm = te.RMSNorm.__new__(te.RMSNorm)
    torch.nn.Module.__init__(norm)
    norm.eps = 1e-6
    norm.zero_centered_gamma = True
    norm._sm_margins = {"forward": 0, "backward": 0, "inference": 0}
    norm.weight = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    norm.sequence_parallel = False

    x5 = torch.randn(5, 1, 16, dtype=torch.bfloat16)
    y5 = norm(x5)
    assert seen_rms_rows == [8]
    assert y5.shape == (5, 1, 16)
    assert torch.equal(y5, x5 * 2.0)

    norm.sequence_parallel = True
    y5_sp = norm(x5)
    assert seen_rms_rows == [8, 5]
    assert y5_sp.shape == (5, 1, 16)

    seen_lnl_rows: list[int] = []
    monkeypatch.setattr(
        "transformer_engine.pytorch.module.layernorm_linear.LayerNormLinear.forward",
        lambda self, x, *a, **kw: (seen_lnl_rows.append(int(x.shape[0])), x + 1.0)[1],
    )
    ln_lin = TELayerNormColumnParallelLinear.__new__(TELayerNormColumnParallelLinear)
    torch.nn.Module.__init__(ln_lin)
    ln_lin.weight = torch.nn.Parameter(torch.randn(16, 16, dtype=torch.bfloat16))
    ln_lin.layer_norm_weight = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    ln_lin.normalization = "RMSNorm"
    ln_lin.te_return_bias = False
    ln_lin.disable_parameter_transpose_cache = True
    ln_lin.is_first_microbatch = False
    ln_lin.te_quant_params = None
    ln_lin.sequence_parallel = False

    out_l, bias_l = ln_lin(x5)
    assert seen_lnl_rows == [8]
    assert out_l.shape == (5, 1, 16)
    assert bias_l is None
    assert torch.equal(out_l, x5 + 1.0)


def test_configure_dynamo_disables_static_by_default() -> None:
    pytest.importorskip("megatron.core")
    from art.megatron.training.compile import _configure_dynamo

    prev_static = torch._dynamo.config.assume_static_by_default
    prev_param_static = torch._dynamo.config.force_parameter_static_shapes
    try:
        torch._dynamo.config.assume_static_by_default = True
        torch._dynamo.config.force_parameter_static_shapes = True
        _configure_dynamo()
        assert torch._dynamo.config.assume_static_by_default is False
        assert torch._dynamo.config.force_parameter_static_shapes is False
    finally:
        torch._dynamo.config.assume_static_by_default = prev_static
        torch._dynamo.config.force_parameter_static_shapes = prev_param_static


# ---------------------------------------------------------------------------
# 2. Short-sequence (< max(block_size, cp_size)) single-rank CP plan
# ---------------------------------------------------------------------------


def test_short_sequence_cp_plan_keeps_sub_block_single_rank_on_rank_0() -> None:
    pytest.importorskip("megatron.core")
    from art.megatron.context_parallel.builder import build_prefix_tree_attention_spec
    from art.megatron.context_parallel.runtime import (
        _build_runtime_plan,
        _runtime_plan_assignment,
    )
    from art.megatron.context_parallel.types import (
        ContextParallelConfig,
        ParallelTopology,
        TokenRange,
    )

    cfg128 = ContextParallelConfig(block_size=128)
    for cp in (2, 4, 8):
        topo = ParallelTopology(cp=cp)
        for valid_tokens in (1, 3, 4, 5, 8, 9, 64, 127):
            groups = torch.zeros((1, valid_tokens), dtype=torch.long)
            spec = build_prefix_tree_attention_spec(group_ids=groups, parent_ids=groups)
            row_spec, chunk_ranges, owners, waves = _runtime_plan_assignment(
                spec, topology=topo, config=cfg128
            )
            assert row_spec is spec.rows[0]
            assert chunk_ranges == (TokenRange(start=0, end=valid_tokens),)
            assert owners == (0,)
            assert waves == (0,)

            plans = _build_runtime_plan(
                spec, topology=topo, config=cfg128, original_seq_len=valid_tokens
            )
            assert [p.local_valid_lengths[0] for p in plans] == [valid_tokens] + [0] * (
                cp - 1
            )
            # Rank 0 has only a local active stage (zero active remote stages) for adjacent lengths 4 vs 5.
            active_stages = [s for s in plans[0].stage_plans if s.slices]
            assert len(active_stages) == 1 and active_stages[0].is_local_stage
            assert all(
                s.q_len == 0 and s.k_len == 0
                for s in plans[0].stage_plans
                if not s.is_local_stage
            )

    # At >= max(block_size, cp_size), full multi-rank CP planner runs normally.
    groups_128 = torch.zeros((1, 128), dtype=torch.long)
    spec_128 = build_prefix_tree_attention_spec(
        group_ids=groups_128, parent_ids=groups_128
    )
    plans_128 = _build_runtime_plan(
        spec_128,
        topology=ParallelTopology(cp=4),
        config=cfg128,
        original_seq_len=128,
    )
    counts_128 = [p.local_valid_lengths[0] for p in plans_128]
    assert sum(counts_128) == 128
    assert min(counts_128) > 0

    # When block_size < cp_size, threshold is max(block_size, cp_size) = cp_size.
    cfg2 = ContextParallelConfig(block_size=2)
    groups_7 = torch.zeros((1, 7), dtype=torch.long)
    spec_7 = build_prefix_tree_attention_spec(group_ids=groups_7, parent_ids=groups_7)
    _, ranges_7, owners_7, waves_7 = _runtime_plan_assignment(
        spec_7, topology=ParallelTopology(cp=8), config=cfg2
    )
    assert (ranges_7, owners_7, waves_7) == ((TokenRange(0, 7),), (0,), (0,))

    groups_8 = torch.zeros((1, 8), dtype=torch.long)
    spec_8 = build_prefix_tree_attention_spec(group_ids=groups_8, parent_ids=groups_8)
    plans_8 = _build_runtime_plan(
        spec_8, topology=ParallelTopology(cp=8), config=cfg2, original_seq_len=8
    )
    assert [p.local_valid_lengths[0] for p in plans_8] == [1] * 8


# ---------------------------------------------------------------------------
# 3. @torch.compiler.disable on LoRA._active_lora_storage
# ---------------------------------------------------------------------------


def test_lora_active_storage_is_marked_compiler_disabled() -> None:
    pytest.importorskip("megatron.core")
    from art.megatron.lora import LoRA

    raw = LoRA.__dict__["_active_lora_storage"]
    assert getattr(raw, "_torchdynamo_disable", False) is True


# ---------------------------------------------------------------------------
# 4. Short-sequence deterministic Triton flex-attention backward
# ---------------------------------------------------------------------------


def test_short_deterministic_flex_callable_has_distinct_code_and_bool_option() -> None:
    regular_fn = flex_compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"})
    short_fn = flex_compiled._short_deterministic_flash_sparse_flex_attention
    assert short_fn.__code__ is not regular_fn.__code__

    opt_val = flex_compiled._SHORT_DETERMINISTIC_FLASH_SPARSE_OPTIONS[
        flash_dlse_patch.SHORT_DETERMINISTIC_BACKWARD_OPTION_KEY
    ]
    assert opt_val is True
    assert type(opt_val) is bool

    # Verify zero graph breaks when tracing the dedicated short-seq wrapper.
    torch._dynamo.reset()
    q = torch.ones(1, 1, 2, 8, requires_grad=True)
    k = torch.ones(1, 1, 1, 8, requires_grad=True)
    mask = create_block_mask(
        lambda b, h, q_idx, k_idx: q_idx >= 0,
        B=1,
        H=1,
        Q_LEN=2,
        KV_LEN=1,
        device="cpu",
    )
    try:
        explanation = torch._dynamo.explain(short_fn)(
            q, k, k, block_mask=mask, scale=0.125, enable_gqa=False
        )
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0, explanation.break_reasons
    finally:
        torch._dynamo.reset()


def test_short_deterministic_flash_backward_lowering_converts_lse_and_strips_opt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch._inductor.kernel.flex.flex_attention as flex_attention_mod
    from torch._inductor.lowering import lowerings

    bwd_op = torch.ops.higher_order.flex_attention_backward
    mul_op = torch.ops.aten.mul.Tensor
    captured_orig: list[tuple[Any, Any, dict[str, Any]]] = []
    mul_calls: list[tuple[Any, float]] = []

    def _fake_orig(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    ) -> str:
        del (
            query,
            key,
            value,
            out,
            grad_out,
            fw_graph,
            joint_graph,
            block_mask,
            scale,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )
        captured_orig.append((logsumexp, grad_logsumexp, dict(kernel_options)))
        return "ok"

    def _fake_mul(tensor: Any, factor: float) -> tuple[str, Any, float]:
        mul_calls.append((tensor, float(factor)))
        return ("mul", tensor, float(factor))

    fake_query = SimpleNamespace(get_size=lambda: (1, 24, 256, 256))
    fake_sizevars = SimpleNamespace(guard_int=lambda v: int(v))

    monkeypatch.setitem(lowerings, bwd_op, _fake_orig)
    monkeypatch.setitem(lowerings, mul_op, _fake_mul)
    monkeypatch.setattr(
        flex_attention_mod,
        "V",
        SimpleNamespace(graph=SimpleNamespace(sizevars=fake_sizevars)),
    )
    monkeypatch.setattr(flash_dlse_patch, "_DETERMINISTIC_BWD_PATCH_APPLIED", False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    counters["art"]["flex_flash_backward_override"] = 0

    flash_dlse_patch.apply_deterministic_flash_backward_patch()
    patched_bwd = lowerings[bwd_op]

    # 1. Short deterministic option switches FLASH -> TRITON with exact LSE rescaling
    #    even when global deterministic algorithms are disabled and head_dim=256.
    opts_short = {
        "BACKEND": "FLASH",
        flash_dlse_patch.SHORT_DETERMINISTIC_BACKWARD_OPTION_KEY: True,
        "WRITE_DQ": True,
    }
    assert (
        patched_bwd(
            fake_query,
            None,
            None,
            None,
            "lse_nat",
            None,
            "dlse_nat",
            None,
            None,
            None,
            0.0625,
            opts_short,
            (),
            (),
        )
        == "ok"
    )
    assert len(captured_orig) == 1
    lse_arg, dlse_arg, opts_out = captured_orig[0]
    assert opts_out["BACKEND"] == "TRITON"
    assert flash_dlse_patch.SHORT_DETERMINISTIC_BACKWARD_OPTION_KEY not in opts_out
    assert opts_out["WRITE_DQ"] is True
    assert lse_arg == ("mul", "lse_nat", math.log2(math.e))
    assert dlse_arg == ("mul", "dlse_nat", math.log(2.0))
    assert counters["art"]["flex_flash_backward_override"] == 0

    # 2. Long stage (no short option, head_dim=256, deterministic=False) stays on FLASH.
    opts_long = {"BACKEND": "FLASH", "WRITE_DQ": True}
    patched_bwd(
        fake_query,
        None,
        None,
        None,
        "lse_long",
        None,
        None,
        None,
        None,
        None,
        0.0625,
        opts_long,
        (),
        (),
    )
    assert captured_orig[1] == (
        "lse_long",
        None,
        {"BACKEND": "FLASH", "WRITE_DQ": True},
    )


def test_flex_attention_kernel_routes_only_short_grad_flash_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    from art.megatron.context_parallel import executor as cp_exec

    calls: list[tuple[str, int, int]] = []

    def _short_spy(q, k, v, *, block_mask, scale, enable_gqa):
        del v, block_mask, scale, enable_gqa
        calls.append(("short_det", int(q.shape[2]), int(k.shape[2])))
        return q, torch.zeros(q.shape[:3])

    def _regular_spy(q, k, v, *, block_mask, scale, enable_gqa):
        del v, block_mask, scale, enable_gqa
        calls.append(("regular_flash", int(q.shape[2]), int(k.shape[2])))
        return q, torch.zeros(q.shape[:3])

    def _other_spy(q, k, v, *, block_mask, scale, enable_gqa):
        del v, block_mask, scale, enable_gqa
        calls.append(("other_backend", int(q.shape[2]), int(k.shape[2])))
        return q, torch.zeros(q.shape[:3])

    backend_box = ["FLASH"]
    monkeypatch.setattr(
        cp_exec, "short_deterministic_flash_sparse_compiled_flex_attention", _short_spy
    )
    monkeypatch.setattr(cp_exec, "sparse_compiled_flex_attention", _regular_spy)
    monkeypatch.setattr(
        cp_exec, "get_sparse_compiled_flex_attention", lambda **_kw: _other_spy
    )
    monkeypatch.setattr(
        cp_exec, "flex_backend_for_head_dims", lambda **_kw: backend_box[0]
    )
    monkeypatch.setattr(
        cp_exec, "prepare_sparse_flex_attention", lambda *_a, **_kw: None
    )

    kernel = cp_exec.FlexAttentionKernel(compile_enabled=True)

    def _bm(ql: int, kl: int) -> Any:
        return SimpleNamespace(BLOCK_SIZE=(128, 64), shape=(1, 1, ql, kl))

    q256 = torch.randn(1, 24, 256, 256, dtype=torch.bfloat16)
    k128 = torch.randn(1, 4, 128, 256, dtype=torch.bfloat16)
    v128 = torch.randn(1, 4, 128, 256, dtype=torch.bfloat16)
    q512 = torch.randn(1, 24, 512, 256, dtype=torch.bfloat16)
    k512 = torch.randn(1, 4, 512, 256, dtype=torch.bfloat16)

    # Grad-enabled <= 256 FLASH stage -> short deterministic path.
    with torch.enable_grad():
        kernel.run(
            q256, k128, v128, block_mask=_bm(256, 128), scale=0.0625, enable_gqa=True
        )
        # Long q or k (> 256) -> unmodified regular FLASH path.
        kernel.run(
            q512, k128, v128, block_mask=_bm(512, 128), scale=0.0625, enable_gqa=True
        )
        kernel.run(
            q256, k512, k512, block_mask=_bm(256, 512), scale=0.0625, enable_gqa=True
        )
        # TRITON backend -> normal backend selector.
        backend_box[0] = "TRITON"
        kernel.run(
            q256, k128, v128, block_mask=_bm(256, 128), scale=0.0625, enable_gqa=True
        )
        backend_box[0] = "FLASH"

    # no_grad -> regular FLASH path even for <= 256.
    with torch.no_grad():
        kernel.run(
            q256, k128, v128, block_mask=_bm(256, 128), scale=0.0625, enable_gqa=True
        )

    assert calls == [
        ("short_det", 256, 128),
        ("regular_flash", 512, 128),
        ("regular_flash", 256, 512),
        ("other_backend", 256, 128),
        ("regular_flash", 256, 128),
    ]


# ---------------------------------------------------------------------------
# 5. TileLang dense-to-varlen chunk_bwd_dqkwg canonicalization
# ---------------------------------------------------------------------------


def test_tilelang_varlen_bwd_canonicalizes_dense_batches_and_caches_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_calls: list[dict[str, Any]] = []

    def _fake_chunk_bwd_dqkwg_tilelang(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        h: torch.Tensor,
        dh: torch.Tensor,
        w: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        g_gamma: Any = None,
        dv: torch.Tensor | None = None,
        scale: Any = None,
        cu_seqlens: torch.Tensor | None = None,
        chunk_size: int = 64,
        chunk_indices: torch.Tensor | None = None,
        use_exp2: bool = False,
        transpose_state_layout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        recorded_calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
                "v_shape": tuple(v.shape),
                "do_shape": tuple(do.shape),
                "h_shape": tuple(h.shape),
                "dh_shape": tuple(dh.shape),
                "w_shape": None if w is None else tuple(w.shape),
                "g_shape": None if g is None else tuple(g.shape),
                "dv_shape": None if dv is None else tuple(dv.shape),
                "cu_seqlens": None if cu_seqlens is None else cu_seqlens.tolist(),
                "chunk_indices": (
                    None if chunk_indices is None else chunk_indices.tolist()
                ),
                "cu_seqlens_obj": cu_seqlens,
                "chunk_indices_obj": chunk_indices,
                "scale": scale,
                "chunk_size": chunk_size,
                "use_exp2": use_exp2,
                "transpose_state_layout": transpose_state_layout,
            }
        )
        dq = q + 1.0
        dk = k + 2.0
        dw = None if w is None else w + 3.0
        dg = None if g is None else g + 4.0
        return dq, dk, dw, dg

    fla_mod = ModuleType("fla")
    ops_mod = ModuleType("fla.ops")
    common_mod = ModuleType("fla.ops.common")
    backends_mod = ModuleType("fla.ops.common.backends")
    tl_pkg = ModuleType("fla.ops.common.backends.tilelang")
    chunk_bwd_mod: Any = ModuleType("fla.ops.common.backends.tilelang.chunk_bwd")
    chunk_bwd_mod.chunk_bwd_dqkwg_tilelang = _fake_chunk_bwd_dqkwg_tilelang
    for name, mod in (
        ("fla", fla_mod),
        ("fla.ops", ops_mod),
        ("fla.ops.common", common_mod),
        ("fla.ops.common.backends", backends_mod),
        ("fla.ops.common.backends.tilelang", tl_pkg),
        ("fla.ops.common.backends.tilelang.chunk_bwd", chunk_bwd_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    # FLA_TILELANG=0 opt-out skips cleanly.
    monkeypatch.setenv("FLA_TILELANG", "0")
    assert ensure_tilelang_varlen_bwd() is False
    assert chunk_bwd_mod.chunk_bwd_dqkwg_tilelang is _fake_chunk_bwd_dqkwg_tilelang

    # Enable and verify idempotent installation.
    monkeypatch.setenv("FLA_TILELANG", "1")
    assert ensure_tilelang_varlen_bwd() is True
    wrapped = chunk_bwd_mod.chunk_bwd_dqkwg_tilelang
    assert wrapped is not _fake_chunk_bwd_dqkwg_tilelang
    assert ensure_tilelang_varlen_bwd() is True
    assert chunk_bwd_mod.chunk_bwd_dqkwg_tilelang is wrapped

    # 1. Dense uniform batches across B in {1, 2, 3, 5, 8} with T=100 (non-multiple of 64)
    #    all fold onto B=1, IS_VARLEN=True with exact cu_seqlens / chunk_indices.
    T, H, K, V, BT = 100, 4, 8, 12, 64
    NT = (T + BT - 1) // BT
    for B in (1, 2, 3, 5, 8):
        q = (
            torch.arange(B * T * H * K, dtype=torch.float32)
            .reshape(B, H, T, K)
            .transpose(1, 2)
        )
        assert not q.is_contiguous()
        k = torch.ones((B, T, H, K), dtype=torch.float32)
        v = torch.ones((B, T, H, V), dtype=torch.float32)
        do = torch.ones((B, T, H, V), dtype=torch.float32)
        w = torch.ones((B, T, H, K), dtype=torch.float32)
        g = torch.ones((B, T, H), dtype=torch.float32)
        h = torch.zeros((B, NT, H, K, V), dtype=torch.float32)
        dh = torch.zeros((B, NT, H, K, V), dtype=torch.float32)

        dq, dk, dw, dg = wrapped(
            q, k, v, do, h, dh, w=w, g=g, scale=0.25, chunk_size=BT
        )
        call = recorded_calls[-1]
        assert call["q_shape"] == (1, B * T, H, K)
        assert call["v_shape"] == (1, B * T, H, V)
        assert call["h_shape"] == (1, B * NT, H, K, V)
        assert call["g_shape"] == (1, B * T, H)
        assert call["cu_seqlens"] == [i * T for i in range(B + 1)]
        assert call["chunk_indices"] == [[b, c] for b in range(B) for c in range(NT)]
        assert dq.shape == (B, T, H, K)
        assert dk.shape == (B, T, H, K)
        assert dw is not None and dw.shape == (B, T, H, K)
        assert dg is not None and dg.shape == (B, T, H)
        assert torch.equal(dq, q.contiguous() + 1.0)
        assert torch.equal(dk, k + 2.0)
        assert torch.equal(dw, w + 3.0)
        assert torch.equal(dg, g + 4.0)

    # 2. Index cache returns identical tensor objects on repeat (device, B, T, chunk_size).
    prev_cu = recorded_calls[-1]["cu_seqlens_obj"]
    prev_ci = recorded_calls[-1]["chunk_indices_obj"]
    B = 8
    q8 = torch.zeros((B, T, H, K), dtype=torch.float32)
    k8 = torch.zeros((B, T, H, K), dtype=torch.float32)
    v8 = torch.zeros((B, T, H, V), dtype=torch.float32)
    do8 = torch.zeros((B, T, H, V), dtype=torch.float32)
    h8 = torch.zeros((B, NT, H, K, V), dtype=torch.float32)
    dh8 = torch.zeros((B, NT, H, K, V), dtype=torch.float32)
    wrapped(q8, k8, v8, do8, h8, dh8, chunk_size=BT)
    assert recorded_calls[-1]["cu_seqlens_obj"] is prev_cu
    assert recorded_calls[-1]["chunk_indices_obj"] is prev_ci

    # 3. Existing varlen calls pass through untouched.
    cu_existing = torch.tensor([0, 40, 100], dtype=torch.int32)
    ci_existing = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32)
    q_vl = torch.zeros((1, 100, H, K), dtype=torch.float32)
    k_vl = torch.zeros((1, 100, H, K), dtype=torch.float32)
    v_vl = torch.zeros((1, 100, H, V), dtype=torch.float32)
    do_vl = torch.zeros((1, 100, H, V), dtype=torch.float32)
    h_vl = torch.zeros((1, 2, H, K, V), dtype=torch.float32)
    dh_vl = torch.zeros((1, 2, H, K, V), dtype=torch.float32)
    wrapped(
        q_vl,
        k_vl,
        v_vl,
        do_vl,
        h_vl,
        dh_vl,
        cu_seqlens=cu_existing,
        chunk_indices=ci_existing,
        chunk_size=BT,
    )
    assert recorded_calls[-1]["cu_seqlens_obj"] is cu_existing
    assert recorded_calls[-1]["chunk_indices_obj"] is ci_existing

    # 4. Non-canonical g_gamma passes through untouched.
    wrapped(
        q8,
        k8,
        v8,
        do8,
        h8,
        dh8,
        g_gamma=torch.zeros((H,), dtype=torch.float32),
        chunk_size=BT,
    )
    assert recorded_calls[-1]["q_shape"] == (B, T, H, K)
    assert recorded_calls[-1]["cu_seqlens"] is None

    # 5. Cache size is bounded by _TILELANG_VARLEN_INDEX_CACHE_MAX.
    for seq_len in range(1, _TILELANG_VARLEN_INDEX_CACHE_MAX + 12):
        nt = (seq_len + BT - 1) // BT
        t_q = torch.zeros((1, seq_len, 1, 4), dtype=torch.float32)
        t_h = torch.zeros((1, nt, 1, 4, 4), dtype=torch.float32)
        wrapped(t_q, t_q, t_q, t_q, t_h, t_h, chunk_size=BT)
    assert len(wrapped._art_index_cache) == _TILELANG_VARLEN_INDEX_CACHE_MAX
