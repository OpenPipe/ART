"""Unit tests for FP32 residual stream support (Design A)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F

from art.megatron.fp32_residual import (
    _fp32_input_rmsnorm,
    _fp32_layernorm_linear_forward,
    _FP32InputRMSNormFn,
)
from art.megatron.model_support.handlers.qwen3_5 import (
    QWEN3_5_DENSE_HANDLER,
    QWEN3_5_MOE_HANDLER,
)
from art.megatron.model_support.registry import get_model_support_handler


def _fp32_rmsnorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    zero_centered_gamma: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    x_fp32 = x.float()
    rstd = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    scale = weight.float() + 1.0 if zero_centered_gamma else weight.float()
    return (x_fp32 * rstd * scale).to(dtype=out_dtype)


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 7, 17])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_fp32_input_rmsnorm_fn_matches_fp32_reference_forward_and_backward(
    rows: int,
    zero_centered_gamma: bool,
) -> None:
    torch.manual_seed(42 + rows)
    hidden = 64
    eps = 1e-6
    x_data = torch.randn(rows, 1, hidden, dtype=torch.float32) * 30.0
    w_data = torch.randn(hidden, dtype=torch.bfloat16) * 0.25

    x = x_data.clone().requires_grad_(True)
    w = w_data.clone().requires_grad_(True)
    y = _fp32_input_rmsnorm(
        x,
        w,
        eps=eps,
        out_dtype=torch.bfloat16,
        zero_centered_gamma=zero_centered_gamma,
    )

    x_ref = x_data.clone().requires_grad_(True)
    w_ref = w_data.clone().requires_grad_(True)
    y_ref = _fp32_rmsnorm_ref(
        x_ref,
        w_ref,
        eps=eps,
        zero_centered_gamma=zero_centered_gamma,
        out_dtype=torch.bfloat16,
    )

    assert y.dtype == torch.bfloat16
    assert y.shape == x.shape
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=0.0)

    grad_out = torch.randn_like(y)
    y.backward(grad_out)
    y_ref.backward(grad_out)

    assert x.grad is not None and x_ref.grad is not None
    assert w.grad is not None and w_ref.grad is not None
    assert x.grad.dtype == torch.float32
    assert w.grad.dtype == torch.bfloat16
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(w.grad, w_ref.grad, rtol=1e-2, atol=1e-2)


def test_fp32_input_rmsnorm_avoids_bf16_pre_truncation() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 1, 64, dtype=torch.float32) * 30.0
    w = torch.randn(64, dtype=torch.bfloat16)

    y_fp32 = _fp32_input_rmsnorm(
        x, w, eps=1e-6, out_dtype=torch.bfloat16, zero_centered_gamma=True
    )
    y_ref = _fp32_rmsnorm_ref(
        x, w, eps=1e-6, zero_centered_gamma=True, out_dtype=torch.bfloat16
    )
    y_bf16_trunc = _fp32_rmsnorm_ref(
        x.bfloat16(), w, eps=1e-6, zero_centered_gamma=True, out_dtype=torch.bfloat16
    )

    torch.testing.assert_close(y_fp32, y_ref, rtol=0.0, atol=0.0)
    assert not torch.equal(y_fp32, y_bf16_trunc)


def test_fp32_input_rmsnorm_empty_tensor_returns_out_dtype_and_zero_grad() -> None:
    x = torch.empty(0, 1, 32, dtype=torch.float32, requires_grad=True)
    w = torch.ones(32, dtype=torch.bfloat16, requires_grad=True)
    y = _fp32_input_rmsnorm(
        x, w, eps=1e-6, out_dtype=torch.bfloat16, zero_centered_gamma=True
    )
    assert y.shape == (0, 1, 32)
    assert y.dtype == torch.bfloat16
    assert y.requires_grad
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == (0, 1, 32)
    assert x.grad.dtype == torch.float32
    assert w.grad is not None and w.grad.shape == (32,)
    assert w.grad.dtype == torch.bfloat16
    assert w.grad.count_nonzero() == 0


def test_fp32_layernorm_linear_preserves_return_contracts() -> None:
    torch.manual_seed(7)
    x = torch.randn(5, 1, 16, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(24, 16, dtype=torch.bfloat16)
    ln_weight = torch.randn(16, dtype=torch.bfloat16)
    bias = torch.randn(24, dtype=torch.bfloat16)

    for return_ln in (False, True):
        for te_return_bias in (False, True):
            mod = SimpleNamespace(
                normalization="RMSNorm",
                eps=1e-6,
                zero_centered_gamma=True,
                tp_size=1,
                sequence_parallel=False,
                apply_bias=not te_return_bias,
                bias=bias,
                is_first_microbatch=True,
                return_layernorm_output=return_ln,
                return_layernorm_output_gathered=False,
                te_return_bias=te_return_bias,
            )
            result = _fp32_layernorm_linear_forward(mod, x, weight, ln_weight)
            assert mod.is_first_microbatch is False
            ln_ref = _fp32_rmsnorm_ref(
                x,
                ln_weight,
                eps=1e-6,
                zero_centered_gamma=True,
                out_dtype=torch.bfloat16,
            )
            lin_ref = F.linear(ln_ref, weight, None if te_return_bias else bias)
            if return_ln and te_return_bias:
                lin_out, ret_bias, ln_out = result
                assert ret_bias is not None and torch.equal(ret_bias, bias)
                torch.testing.assert_close(ln_out, ln_ref, rtol=0.0, atol=0.0)
            elif return_ln:
                (lin_out, ln_out), ret_bias = result
                assert ret_bias is None
                torch.testing.assert_close(ln_out, ln_ref, rtol=0.0, atol=0.0)
            elif te_return_bias:
                lin_out, ret_bias = result
                assert ret_bias is not None and torch.equal(ret_bias, bias)
            else:
                lin_out, ret_bias = result
                assert ret_bias is None
            assert lin_out.dtype == torch.bfloat16
            torch.testing.assert_close(lin_out, lin_ref, rtol=0.0, atol=0.0)


def test_qwen35_handler_fp32_residual_defaults() -> None:
    moe_handler = get_model_support_handler("Qwen/Qwen3.6-35B-A3B")
    dense_handler = get_model_support_handler("Qwen/Qwen3.8-27B")
    assert moe_handler is QWEN3_5_MOE_HANDLER
    assert dense_handler is QWEN3_5_DENSE_HANDLER

    moe_provider = SimpleNamespace(
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        moe_shared_expert_overlap=True,
        fp32_residual_connection=False,
    )
    moe_handler.configure_provider_for_runtime(moe_provider)
    assert moe_provider.fp32_residual_connection is True
    assert moe_provider.moe_shared_expert_overlap is False

    dense_provider = SimpleNamespace(
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        fp32_residual_connection=False,
    )
    dense_handler.configure_provider_for_runtime(dense_provider)
    assert dense_provider.fp32_residual_connection is False


def test_provider_env_override_controls_fp32_residual_connection() -> None:
    pytest.importorskip("megatron.core")
    from art.megatron.provider import _apply_runtime_env_overrides, _ProviderRuntimeEnv

    provider = SimpleNamespace(
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
        ep_overlap_early_attn_memory_release=False,
        moe_hybridep_num_sms=24,
        moe_apply_probs_on_input=False,
        bias_activation_fusion=False,
        fine_grained_activation_offloading=False,
        offload_modules=[],
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        microbatch_group_size_per_vp_stage=None,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        recompute_modules=[],
        moe_shared_expert_overlap=False,
        moe_router_dtype="fp32",
        fp32_residual_connection=True,
    )

    env_off = _ProviderRuntimeEnv.from_environ(
        {"ART_MEGATRON_FP32_RESIDUAL_CONNECTION": "0"}
    )
    assert env_off.fp32_residual_connection is False
    _apply_runtime_env_overrides(cast(Any, provider), env_off)
    assert provider.fp32_residual_connection is False

    env_on = _ProviderRuntimeEnv.from_environ(
        {"ART_MEGATRON_FP32_RESIDUAL_CONNECTION": "true"}
    )
    assert env_on.fp32_residual_connection is True
    _apply_runtime_env_overrides(cast(Any, provider), env_on)
    assert provider.fp32_residual_connection is True


def test_moe_postprocess_patch_adds_in_fp32_only_when_enabled() -> None:
    pytest.importorskip("megatron.core")
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from art.megatron.fp32_residual import install_fp32_residual_patches

    install_fp32_residual_patches()

    class _FakeDispatcher:
        def combine_postprocess(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

    routed = torch.tensor([[33.5, 1.0]], dtype=torch.bfloat16)
    shared = torch.tensor([[0.08, 0.25]], dtype=torch.bfloat16)

    layer_fp32 = MoELayer.__new__(MoELayer)
    layer_fp32.token_dispatcher = _FakeDispatcher()
    layer_fp32.config = SimpleNamespace(
        moe_latent_size=None, fp32_residual_connection=True
    )
    out_fp32 = MoELayer.postprocess(layer_fp32, routed, shared)
    assert out_fp32.dtype == torch.float32
    torch.testing.assert_close(
        out_fp32, routed.float() + shared.float(), rtol=0.0, atol=0.0
    )

    # With fp32_residual_connection=False, postprocess is an unmodified BF16 no-op.
    layer_bf16 = MoELayer.__new__(MoELayer)
    layer_bf16.token_dispatcher = _FakeDispatcher()
    layer_bf16.config = SimpleNamespace(
        moe_latent_size=None, fp32_residual_connection=False
    )
    out_bf16 = MoELayer.postprocess(layer_bf16, routed, shared)
    assert out_bf16.dtype == torch.bfloat16
    torch.testing.assert_close(out_bf16, routed + shared, rtol=0.0, atol=0.0)


def test_te_norm_and_ln_linear_patches_noop_on_bf16_inputs(monkeypatch) -> None:
    pytest.importorskip("megatron.core")
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
    )
    import transformer_engine.pytorch as te

    from art.megatron.fp32_residual import install_fp32_residual_patches

    install_fp32_residual_patches()
    assert getattr(
        TELayerNormColumnParallelLinear.forward, "__art_fp32_residual__", False
    )
    assert getattr(te.RMSNorm.forward, "__art_fp32_residual__", False)

    # Verify calling patched RMSNorm.forward with FP32 input and BF16 weight
    # dispatches through _FP32InputRMSNormFn without calling TE's BF16 path.
    norm = te.RMSNorm.__new__(te.RMSNorm)
    torch.nn.Module.__init__(norm)
    norm.eps = 1e-6
    norm.zero_centered_gamma = True
    norm._sm_margins = {"forward": 0, "backward": 0, "inference": 0}
    norm.weight = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    x = torch.randn(3, 1, 16, dtype=torch.float32, requires_grad=True)
    y = norm(x)
    assert y.dtype == torch.bfloat16
    y_ref = _fp32_rmsnorm_ref(
        x, norm.weight, eps=1e-6, zero_centered_gamma=True, out_dtype=torch.bfloat16
    )
    torch.testing.assert_close(y, y_ref, rtol=0.0, atol=0.0)


def test_empty_cp_rank_gdn_and_norm_preserve_params_dtype() -> None:
    pytest.importorskip("megatron.core")
    from art.megatron.gdn.operator import (
        _apply_explicit_norm,
        _empty_safe_norm_forward,
        _project_empty_gdn_inputs,
    )

    gdn = SimpleNamespace(
        sp_size=1,
        tp_size=1,
        qk_dim=32,
        v_dim=64,
        num_value_heads=4,
        value_head_dim=16,
        config=SimpleNamespace(params_dtype=torch.bfloat16),
    )
    hidden = torch.empty(0, 1, 64, dtype=torch.float32, requires_grad=True)
    qkv, gate, beta, recurrent_g = _project_empty_gdn_inputs(gdn, hidden)

    assert qkv.dtype == torch.bfloat16
    assert gate.dtype == torch.bfloat16
    assert beta.dtype == torch.bfloat16
    assert recurrent_g.dtype == torch.float32
    assert qkv.shape == (1, 0, 128)
    assert gate.shape == (1, 0, 4, 16)
    assert beta.shape == (1, 0, 4)
    assert recurrent_g.shape == (1, 0, 4)
    (
        qkv.float().sum() + gate.float().sum() + beta.float().sum() + recurrent_g.sum()
    ).backward()
    assert hidden.grad is not None
    assert hidden.grad.dtype == torch.float32
    assert hidden.grad.shape == hidden.shape

    norm = SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(64, dtype=torch.bfloat16)),
        bias=None,
        eps=1e-6,
        normalization="RMSNorm",
        zero_centered_gamma=True,
    )
    norm_out = _apply_explicit_norm(
        norm, hidden, config=None, weight_name="weight", bias_name="bias"
    )
    assert norm_out.dtype == torch.bfloat16
    assert norm_out.shape == hidden.shape

    empty_out = _empty_safe_norm_forward(norm, hidden)
    assert empty_out.dtype == torch.bfloat16
    assert empty_out.shape == hidden.shape


@pytest.mark.parametrize("non_gated", [False, True])
def test_empty_cp_rank_shared_experts_fc1_lora_returns_weight_dtype(
    non_gated: bool,
) -> None:
    pytest.importorskip("megatron.core")
    from art.megatron.lora import LoRA, SharedExpertsLinearFC1LoRA

    fc1 = SharedExpertsLinearFC1LoRA.__new__(SharedExpertsLinearFC1LoRA)
    torch.nn.Module.__init__(fc1)
    fc1.non_gated = non_gated
    fc1.out_features = 32 if non_gated else 64
    linear = torch.nn.Module()
    linear.weight = torch.nn.Parameter(
        torch.zeros(fc1.out_features, 16, dtype=torch.bfloat16)
    )
    linear.layer_norm_weight = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
    fc1.linear_fc1 = cast(Any, linear)
    if non_gated:
        fc1.up_lora = LoRA(
            "layer.0.mlp.up_proj", 16, 32, 4, 8, torch.bfloat16, torch.device("cpu")
        )
    else:
        fc1.gate_lora = LoRA(
            "layer.0.mlp.gate_proj", 16, 32, 4, 8, torch.bfloat16, torch.device("cpu")
        )
        fc1.up_lora = LoRA(
            "layer.0.mlp.up_proj", 16, 32, 4, 8, torch.bfloat16, torch.device("cpu")
        )

    x = torch.empty(0, 1, 16, dtype=torch.float32, requires_grad=True)
    out, bias = fc1(x)
    assert bias is None
    assert out.dtype == torch.bfloat16
    assert out.shape == (0, 1, fc1.out_features)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.dtype == torch.float32
    assert x.grad.shape == (0, 1, 16)


def test_patched_te_wrappers_delegate_untouched_for_bf16_and_full_fp32(
    monkeypatch,
) -> None:
    pytest.importorskip("megatron.core")
    pytest.importorskip("transformer_engine.pytorch")
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
    )
    import transformer_engine.pytorch as te

    from art.megatron.fp32_residual import install_fp32_residual_patches

    install_fp32_residual_patches()

    # Check that patched RMSNorm does not intercept BF16 inputs or FP32-weight inputs.
    norm = te.RMSNorm.__new__(te.RMSNorm)
    torch.nn.Module.__init__(norm)
    norm.eps = 1e-6
    norm.zero_centered_gamma = True
    norm._sm_margins = {"forward": 0, "backward": 0, "inference": 0}
    norm.weight = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))

    intercepted = False

    def _spy(*args, **kwargs):
        nonlocal intercepted
        intercepted = True
        raise AssertionError("should not intercept non-mixed precision inputs")

    monkeypatch.setattr("art.megatron.fp32_residual._fp32_input_rmsnorm", _spy)
    sentinel = torch.zeros(2, 1, 16, dtype=torch.bfloat16)
    monkeypatch.setattr(
        "transformer_engine.pytorch.ops.fuser.OperationFuser.__call__",
        lambda self, *a, **kw: sentinel,
    )
    assert norm(torch.randn(2, 1, 16, dtype=torch.bfloat16)) is sentinel
    assert not intercepted

    norm.weight = torch.nn.Parameter(torch.randn(16, dtype=torch.float32))
    assert norm(torch.randn(2, 1, 16, dtype=torch.float32)) is sentinel
    assert not intercepted

    # Check TELayerNormColumnParallelLinear patch does not intercept BF16 inputs.
    ln_lin = TELayerNormColumnParallelLinear.__new__(TELayerNormColumnParallelLinear)
    torch.nn.Module.__init__(ln_lin)
    ln_lin.weight = torch.nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16))
    ln_lin.layer_norm_weight = torch.nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    ln_lin.normalization = "RMSNorm"
    ln_lin.te_return_bias = False
    ln_lin.disable_parameter_transpose_cache = True
    ln_lin.is_first_microbatch = False
    ln_lin.te_quant_params = None
    monkeypatch.setattr(
        "transformer_engine.pytorch.module.layernorm_linear.LayerNormLinear.forward",
        lambda self, x, **kw: sentinel,
    )
    out, bias = ln_lin(torch.randn(2, 1, 16, dtype=torch.bfloat16))
    assert out is sentinel and bias is None
    assert not intercepted
