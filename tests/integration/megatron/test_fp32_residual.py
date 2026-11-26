"""GPU integration tests for FP32 residual stream TE C++ RMSNorm kernels."""

from __future__ import annotations

import pytest
import torch

from art.megatron.fp32_residual import (
    _FP32InputRMSNormFn,
    install_fp32_residual_patches,
)

pytestmark = pytest.mark.gpu


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


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 17])
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_fp32_input_rmsnorm_fn_cuda_te_parity(
    rows: int,
    zero_centered_gamma: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for TE rmsnorm kernel test")
    pytest.importorskip("transformer_engine.pytorch")

    torch.manual_seed(123 + rows)
    hidden = 2048
    eps = 1e-6
    x_data = torch.randn(rows, 1, hidden, device="cuda", dtype=torch.float32) * 30.0
    w_data = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.25

    x = x_data.clone().requires_grad_(True)
    w = w_data.clone().requires_grad_(True)
    y = _FP32InputRMSNormFn.apply(x, w, eps, torch.bfloat16, zero_centered_gamma, 0, 0)
    assert isinstance(y, torch.Tensor)

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
    assert (y.float() - y_ref.float()).abs().max().item() == 0.0

    dy = torch.randn_like(y)
    y.backward(dy)
    y_ref.backward(dy)

    assert x.grad is not None and x_ref.grad is not None
    assert w.grad is not None and w_ref.grad is not None
    assert x.grad.dtype == torch.float32
    assert w.grad.dtype == torch.bfloat16
    dx_rel = (x.grad - x_ref.grad).abs().max().item() / (
        x_ref.grad.abs().max().item() + 1e-12
    )
    assert dx_rel < 1e-5
    assert (w.grad.float() - w_ref.grad.float()).abs().max().item() == 0.0


def test_patched_te_rmsnorm_matches_fp32_and_is_noop_on_bf16_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for TE RMSNorm test")
    te = pytest.importorskip("transformer_engine.pytorch")
    pytest.importorskip("megatron.core")

    install_fp32_residual_patches()
    torch.manual_seed(7)
    norm = te.RMSNorm(
        normalized_shape=2048,
        eps=1e-6,
        zero_centered_gamma=True,
        dtype=torch.bfloat16,
        device="cuda",
    )
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight) * 0.25)

    # FP32 input: exact FP32 RMSNorm (no silent BF16 pre-truncation).
    x_fp32 = torch.randn(5, 1, 2048, device="cuda", dtype=torch.float32) * 30.0
    y_fp32 = norm(x_fp32)
    y_ref = _fp32_rmsnorm_ref(
        x_fp32,
        norm.weight,
        eps=1e-6,
        zero_centered_gamma=True,
        out_dtype=torch.bfloat16,
    )
    assert y_fp32.dtype == torch.bfloat16
    assert (y_fp32.float() - y_ref.float()).abs().max().item() == 0.0

    # BF16 input: standard TE BF16 path remains untouched.
    x_bf16 = x_fp32.bfloat16()
    y_bf16 = norm(x_bf16)
    y_bf16_ref = _fp32_rmsnorm_ref(
        x_bf16,
        norm.weight,
        eps=1e-6,
        zero_centered_gamma=True,
        out_dtype=torch.bfloat16,
    )
    assert y_bf16.dtype == torch.bfloat16
    assert (y_bf16.float() - y_bf16_ref.float()).abs().max().item() == 0.0
    assert not torch.equal(y_fp32, y_bf16)
