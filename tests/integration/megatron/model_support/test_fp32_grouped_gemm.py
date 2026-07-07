from __future__ import annotations

import sys
from types import ModuleType

import pytest

from . import fp32_grouped_gemm


def test_fp32_grouped_gemm_helper_restores_guarded_te_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def original() -> None:
        pass

    def guarded() -> None:
        pass

    setattr(guarded, fp32_grouped_gemm._GUARD_ATTR, True)
    setattr(guarded, fp32_grouped_gemm._ORIGINAL_ATTR, original)
    gemm = ModuleType("transformer_engine.pytorch.cpp_extensions.gemm")
    gemm.general_grouped_gemm = guarded  # type: ignore[attr-defined]
    cpp_extensions = ModuleType("transformer_engine.pytorch.cpp_extensions")
    cpp_extensions.gemm = gemm  # type: ignore[attr-defined]
    cpp_extensions.general_grouped_gemm = guarded  # type: ignore[attr-defined]
    grouped_linear = ModuleType("transformer_engine.pytorch.module.grouped_linear")
    grouped_linear.general_grouped_gemm = guarded  # type: ignore[attr-defined]
    linear = ModuleType("transformer_engine.pytorch.module.linear")
    linear.general_grouped_gemm = guarded  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "transformer_engine", ModuleType("te"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", ModuleType("te.pt"))
    monkeypatch.setitem(
        sys.modules, "transformer_engine.pytorch.cpp_extensions", cpp_extensions
    )
    monkeypatch.setitem(
        sys.modules, "transformer_engine.pytorch.module.grouped_linear", grouped_linear
    )
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module.linear", linear)

    fp32_grouped_gemm.allow_fp32_grouped_gemm_fallback_for_model_support_tests()

    assert gemm.general_grouped_gemm is original
    assert cpp_extensions.general_grouped_gemm is original
    assert grouped_linear.general_grouped_gemm is original
    assert linear.general_grouped_gemm is original
