import sys
from types import ModuleType
from typing import Any

import pytest
import torch

from art.megatron.runtime import te_cutlass_grouped_gemm as guard

EXPERTS, IN, OUT = 4, 8, 6


@pytest.fixture
def te_gemm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """A fake TE grouped GEMM that must never see a call without rows."""
    calls: list[dict[str, Any]] = []

    def general_grouped_gemm(A, B, out, quantization_params, out_dtype, **kwargs):
        rows = sum(kwargs["m_splits"])
        assert rows > 0, "an all-empty grouped GEMM reached Transformer Engine"
        calls.append({"layout": kwargs.get("layout", "TN"), "rows": rows})
        return out, [], []

    modules = {
        name: ModuleType(name)
        for name in (
            "transformer_engine",
            "transformer_engine.pytorch",
            "transformer_engine.pytorch.cpp_extensions",
            "transformer_engine.pytorch.cpp_extensions.gemm",
            "transformer_engine.pytorch.module.grouped_linear",
        )
    }
    for name, module in modules.items():
        setattr(module, "general_grouped_gemm", general_grouped_gemm)
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, module)
    monkeypatch.setattr(
        guard, "_te_cutlass_grouped_gemm_fallback_reason", lambda **_: None
    )
    # Restore the TE environment the guard forces once the test ends.
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")
    monkeypatch.setenv("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", "1")
    guard.install_te_cutlass_grouped_gemm_guard()
    return calls


class _GroupedLinear(torch.autograd.Function):
    """TE GroupedLinear's forward, dgrad and wgrad grouped GEMM calls."""

    @staticmethod
    def forward(ctx, x, weight, main_grad, m_splits, accumulate):
        gemm = sys.modules["transformer_engine.pytorch.module.grouped_linear"]
        weights = list(weight.unbind(0))
        out = torch.empty(sum(m_splits), OUT, dtype=x.dtype)
        gemm.general_grouped_gemm(
            weights,
            list(torch.split(x, m_splits)),
            [out],
            [None],
            x.dtype,
            single_output=True,
            m_splits=m_splits,
        )
        ctx.save_for_backward(x, weight, main_grad)
        ctx.m_splits, ctx.accumulate = m_splits, accumulate
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        gemm = sys.modules["transformer_engine.pytorch.module.grouped_linear"]
        (grad_output,) = grad_outputs
        x, weight, main_grad = ctx.saved_tensors
        m_splits = ctx.m_splits
        dgrad = torch.empty(sum(m_splits), IN, dtype=x.dtype)
        gemm.general_grouped_gemm(
            list(weight.unbind(0)),
            list(torch.split(grad_output, m_splits)),
            [dgrad],
            [None],
            x.dtype,
            single_output=True,
            layout="NN",
            m_splits=m_splits,
            grad=True,
        )
        # Uninitialized wgrad buffers, or main_grad when accumulating into it.
        wgrad = main_grad if ctx.accumulate else torch.full_like(weight, float("nan"))
        gemm.general_grouped_gemm(
            list(torch.split(x, m_splits)),
            list(torch.split(grad_output, m_splits)),
            list(wgrad.unbind(0)),
            [None],
            x.dtype,
            layout="NT",
            m_splits=m_splits,
            grad=True,
            accumulate=ctx.accumulate,
        )
        return dgrad, None if ctx.accumulate else wgrad, None, None, None


def _run(m_splits: list[int], *, accumulate: bool = False):
    x = torch.randn(sum(m_splits), IN, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(EXPERTS, OUT, IN, dtype=torch.bfloat16, requires_grad=True)
    main_grad = torch.ones(EXPERTS, OUT, IN, dtype=torch.bfloat16)
    y = _GroupedLinear.apply(x, weight, main_grad, m_splits, accumulate)
    (y.float().sum() * 2.0).backward()
    return x, weight, main_grad, y


def test_all_empty_groups_skip_te_and_keep_autograd_shapes(te_gemm) -> None:
    x, weight, _main_grad, y = _run([0] * EXPERTS)

    assert te_gemm == []
    assert y.shape == (0, OUT) and y.dtype == torch.bfloat16
    assert x.grad is not None and x.grad.shape == (0, IN)
    assert weight.grad is not None and weight.grad.shape == weight.shape
    assert torch.count_nonzero(weight.grad) == 0


def test_all_empty_groups_leave_accumulated_main_grad_unchanged(te_gemm) -> None:
    _x, weight, main_grad, _y = _run([0] * EXPERTS, accumulate=True)

    assert te_gemm == []
    assert weight.grad is None
    assert torch.equal(main_grad, torch.ones_like(main_grad))


def test_calls_with_rows_reach_te(te_gemm) -> None:
    _run([0, 3, 0, 2])

    assert te_gemm == [
        {"layout": "TN", "rows": 5},
        {"layout": "NN", "rows": 5},
        {"layout": "NT", "rows": 5},
    ]


def test_fallback_refusal_is_unchanged(te_gemm, monkeypatch) -> None:
    monkeypatch.setattr(
        guard, "_te_cutlass_grouped_gemm_fallback_reason", lambda **_: "test reason"
    )

    with pytest.raises(RuntimeError, match="fallback path: test reason"):
        _run([0, 3, 0, 2])
    assert te_gemm == []
