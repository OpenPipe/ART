from __future__ import annotations

import functools
import os
import sys
from typing import Any

_GUARD_ATTR = "__art_te_cutlass_grouped_gemm_guard__"
_ORIGINAL_ATTR = "__art_original_general_grouped_gemm__"
_REFERENCE_ATTR = "__art_fp32_grouped_linear_reference__"


def allow_fp32_grouped_gemm_fallback_for_model_support_tests() -> None:
    """Use topology-stable fp32 expert GEMMs in semantic model-support tests."""
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
    _install_fp32_grouped_linear_reference()


def _install_fp32_grouped_linear_reference() -> None:
    from megatron.core.extensions.transformer_engine import TEGroupedLinear
    import torch

    assert TEGroupedLinear is not None
    current = TEGroupedLinear.forward
    if getattr(current, _REFERENCE_ATTR, False):
        return

    @functools.wraps(current)
    def forward(self, x, m_splits):
        if x.dtype is not torch.float32:
            return current(self, x, m_splits)
        counts = [int(count) for count in m_splits]
        weights = self._get_weight_tensors()
        biases = self._get_bias_tensors()
        outputs = [
            torch.nn.functional.linear(
                rows,
                weight,
                bias if self.apply_bias else None,
            )
            for rows, weight, bias in zip(x.split(counts), weights, biases, strict=True)
        ]
        output = torch.cat(outputs)
        self.is_first_microbatch = False
        if self.te_return_bias:
            return output, biases
        return output, None

    setattr(forward, _REFERENCE_ATTR, True)
    setattr(TEGroupedLinear, "forward", forward)


def _patch_if_guarded(module_name: str, guarded: Any, original: Any) -> None:
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "general_grouped_gemm", None) is guarded:
        setattr(module, "general_grouped_gemm", original)
