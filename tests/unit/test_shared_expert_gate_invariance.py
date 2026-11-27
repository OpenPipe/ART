"""Unit tests for row-count-invariant SharedExpertMLP sigmoid gating."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from art.megatron.shared_expert_gate import (
    _SHARED_GATE_CHUNK_ROWS,
    _is_patched,
    install_shared_expert_gate_patch,
    row_invariant_shared_expert_gate,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_row_invariant_shared_expert_gate_matches_reference_fwd_and_bwd(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(17)
    rows = _SHARED_GATE_CHUNK_ROWS + 137
    hidden = 64
    x_data = torch.randn(rows, 1, hidden, dtype=dtype)
    w_data = torch.randn(1, hidden, dtype=dtype) * 0.05
    grad_out = torch.randn(rows, 1, 1, dtype=dtype)

    x_ref = x_data.detach().clone().requires_grad_(True)
    w_ref = w_data.detach().clone().requires_grad_(True)
    y_ref = torch.sigmoid(F.linear(x_ref.float(), w_ref.float()).to(dtype=dtype))
    (y_ref.float() * grad_out.float()).sum().backward()

    x_got = x_data.detach().clone().requires_grad_(True)
    w_got = w_data.detach().clone().requires_grad_(True)
    y_got = row_invariant_shared_expert_gate(x_got, w_got)
    (y_got.float() * grad_out.float()).sum().backward()

    assert y_got.dtype == dtype
    assert y_got.shape == (rows, 1, 1)
    torch.testing.assert_close(y_got.float(), y_ref.float(), atol=1e-2, rtol=1e-2)
    assert x_got.grad is not None and x_ref.grad is not None
    assert w_got.grad is not None and w_ref.grad is not None
    torch.testing.assert_close(
        x_got.grad.float(), x_ref.grad.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        w_got.grad.float(), w_ref.grad.float(), atol=5e-2, rtol=5e-2
    )


def test_row_invariant_shared_expert_gate_is_bit_exact_across_row_counts() -> None:
    torch.manual_seed(29)
    max_rows = _SHARED_GATE_CHUNK_ROWS + 257
    hidden = 128
    x_full = torch.randn(max_rows, 1, hidden, dtype=torch.bfloat16)
    w = torch.randn(1, hidden, dtype=torch.bfloat16) * 0.05
    grad_out = torch.randn(max_rows, 1, 1, dtype=torch.bfloat16)

    x_ref = x_full.detach().clone().requires_grad_(True)
    y_ref = row_invariant_shared_expert_gate(x_ref, w)
    (y_ref * grad_out).sum().backward()
    assert x_ref.grad is not None

    for rows in (
        1,
        7,
        63,
        128,
        575,
        1024,
        1087,
        _SHARED_GATE_CHUNK_ROWS,
        _SHARED_GATE_CHUNK_ROWS + 1,
        max_rows,
    ):
        x_sub = x_full[:rows].detach().clone().requires_grad_(True)
        y_sub = row_invariant_shared_expert_gate(x_sub, w)
        (y_sub * grad_out[:rows]).sum().backward()
        assert torch.equal(y_sub, y_ref[:rows]), f"forward drift at M={rows}"
        assert x_sub.grad is not None
        assert torch.equal(x_sub.grad, x_ref.grad[:rows]), f"dx drift at M={rows}"


def test_row_invariant_shared_expert_gate_handles_empty_input() -> None:
    x = torch.empty((0, 1, 64), dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((1, 64), dtype=torch.bfloat16, requires_grad=True)
    y = row_invariant_shared_expert_gate(x, w)
    assert y.shape == (0, 1, 1)
    assert y.dtype == torch.bfloat16
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == (0, 1, 64)
    assert w.grad is not None and torch.equal(w.grad, torch.zeros_like(w))


def test_install_shared_expert_gate_patch_is_class_level_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_calls: list[int] = []
    orig_comm_calls: list[int] = []

    class FakeMLP:
        def forward(self, hidden_states: torch.Tensor) -> Any:
            base_calls.append(int(hidden_states.shape[0]))
            return hidden_states * 2.0, None

    class FakeSharedExpertMLP(FakeMLP):
        def __init__(self, *, use_gate: bool, hidden: int = 32) -> None:
            self.use_shared_expert_gate = use_gate
            self.gate_weight = (
                torch.nn.Parameter(torch.randn(1, hidden, dtype=torch.bfloat16) * 0.1)
                if use_gate
                else None
            )
            self.gate_score: torch.Tensor | None = None
            self.stream = None

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            output, _ = super().forward(hidden_states)
            if self.use_shared_expert_gate:
                assert self.gate_weight is not None
                logits = F.linear(hidden_states, self.gate_weight)
                output = output * torch.sigmoid(logits)
            return output

        def pre_forward_comm(self, input: torch.Tensor) -> str:
            orig_comm_calls.append(int(input.shape[0]))
            if self.use_shared_expert_gate:
                assert self.gate_weight is not None
                self.gate_score = torch.sigmoid(F.linear(input, self.gate_weight))
            return "ok"

    # Simulate compile_workarounds having already disabled compile on pre_forward_comm.
    disabled_comm: Any = torch.compiler.disable(FakeSharedExpertMLP.pre_forward_comm)
    setattr(disabled_comm, "__art_compile_disabled__", True)
    FakeSharedExpertMLP.pre_forward_comm = disabled_comm

    fake_pkg = types.ModuleType("megatron")
    fake_core = types.ModuleType("megatron.core")
    fake_transformer = types.ModuleType("megatron.core.transformer")
    fake_moe = types.ModuleType("megatron.core.transformer.moe")
    fake_shared_mod = types.ModuleType("megatron.core.transformer.moe.shared_experts")
    setattr(fake_shared_mod, "SharedExpertMLP", FakeSharedExpertMLP)
    monkeypatch.setitem(sys.modules, "megatron", fake_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core", fake_core)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", fake_transformer)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe", fake_moe)
    monkeypatch.setitem(
        sys.modules, "megatron.core.transformer.moe.shared_experts", fake_shared_mod
    )

    install_shared_expert_gate_patch()
    first_fwd = FakeSharedExpertMLP.forward
    first_comm = FakeSharedExpertMLP.pre_forward_comm
    install_shared_expert_gate_patch()
    assert FakeSharedExpertMLP.forward is first_fwd
    assert FakeSharedExpertMLP.pre_forward_comm is first_comm
    assert _is_patched(FakeSharedExpertMLP.forward)
    assert _is_patched(FakeSharedExpertMLP.pre_forward_comm)
    assert getattr(
        FakeSharedExpertMLP.pre_forward_comm, "__art_compile_disabled__", False
    )

    # Patch must live on the class so instance vars remain untouched for memory accounting.
    gated = FakeSharedExpertMLP(use_gate=True)
    ungated = FakeSharedExpertMLP(use_gate=False)
    assert "forward" not in vars(gated)
    assert "forward" not in vars(ungated)

    x = torch.randn(37, 1, 32, dtype=torch.bfloat16)
    out_gated = gated.forward(x)
    assert gated.gate_weight is not None
    expected_gated = (x * 2.0) * row_invariant_shared_expert_gate(x, gated.gate_weight)
    assert torch.equal(out_gated, expected_gated)

    out_ungated = ungated.forward(x)
    assert torch.equal(out_ungated, x * 2.0)

    assert gated.pre_forward_comm(x) == "ok"
    assert gated.gate_score is not None
    assert torch.equal(
        gated.gate_score, row_invariant_shared_expert_gate(x, gated.gate_weight)
    )
    assert base_calls == [37, 37]
    assert orig_comm_calls == [37]
