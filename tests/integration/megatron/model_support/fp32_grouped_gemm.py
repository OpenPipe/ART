from __future__ import annotations

import os
import sys
from typing import Any

_GUARD_ATTR = "__art_te_cutlass_grouped_gemm_guard__"
_ORIGINAL_ATTR = "__art_original_general_grouped_gemm__"


def allow_fp32_grouped_gemm_fallback_for_model_support_tests() -> None:
    """Use TE's fp32 grouped-GEMM fallback in semantic model-support tests."""
    os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "0"
    os.environ["NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"] = "0"
    try:
        from transformer_engine.pytorch.cpp_extensions import gemm
    except Exception:
        return

    current = gemm.general_grouped_gemm
    if not getattr(current, _GUARD_ATTR, False):
        return
    original = getattr(current, _ORIGINAL_ATTR)
    setattr(gemm, "general_grouped_gemm", original)
    _patch_if_guarded("transformer_engine.pytorch.cpp_extensions", current, original)
    _patch_if_guarded(
        "transformer_engine.pytorch.module.grouped_linear", current, original
    )
    _patch_if_guarded("transformer_engine.pytorch.module.linear", current, original)


def _patch_if_guarded(module_name: str, guarded: Any, original: Any) -> None:
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "general_grouped_gemm", None) is guarded:
        setattr(module, "general_grouped_gemm", original)
