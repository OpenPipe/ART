from __future__ import annotations

import functools
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

_GUARD_ATTR = "__art_te_cutlass_grouped_gemm_guard__"
_ORIGINAL_ATTR = "__art_original_general_grouped_gemm__"
_DEVICE_CAPABILITIES: dict[int, tuple[int, int]] = {}


def force_te_cutlass_grouped_gemm_env(env: dict[str, str] | None = None) -> None:
    target = os.environ if env is None else env
    target["NVTE_USE_CUTLASS_GROUPED_GEMM"] = "1"
    target["NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK"] = "1"


def install_te_cutlass_grouped_gemm_guard() -> None:
    force_te_cutlass_grouped_gemm_env()
    from transformer_engine.pytorch.cpp_extensions import gemm

    current = gemm.general_grouped_gemm
    if getattr(current, _GUARD_ATTR, False):
        return
    original = getattr(current, _ORIGINAL_ATTR, current)

    @functools.wraps(original)
    def _guarded_general_grouped_gemm(
        A: list[torch.Tensor],
        B: list[torch.Tensor],
        out: list[torch.Tensor],
        quantization_params: list[Any | None],
        out_dtype: torch.dtype,
        layout: str = "TN",
        m_splits: list[int] | None = None,
        gelu: bool = False,
        grad: bool = False,
        accumulate: bool = False,
        bias: list[torch.Tensor] | None = None,
        use_bias: bool = False,
        use_split_accumulator: bool = False,
        D_dtype: Any | None = None,
        single_output: bool = False,
    ):
        _raise_if_te_cutlass_grouped_gemm_would_fallback(
            A=A,
            B=B,
            out=out,
            quantization_params=quantization_params,
            layout=layout,
            gelu=gelu,
            use_bias=use_bias,
        )
        return original(
            A,
            B,
            out,
            quantization_params,
            out_dtype,
            layout=layout,
            m_splits=m_splits,
            gelu=gelu,
            grad=grad,
            accumulate=accumulate,
            bias=bias,
            use_bias=use_bias,
            use_split_accumulator=use_split_accumulator,
            D_dtype=D_dtype,
            single_output=single_output,
        )

    setattr(_guarded_general_grouped_gemm, _GUARD_ATTR, True)
    setattr(_guarded_general_grouped_gemm, _ORIGINAL_ATTR, original)
    setattr(gemm, "general_grouped_gemm", _guarded_general_grouped_gemm)
    _patch_cpp_extensions_export(original, _guarded_general_grouped_gemm)
    _patch_imported_grouped_linear(original, _guarded_general_grouped_gemm)


def _raise_if_te_cutlass_grouped_gemm_would_fallback(
    *,
    A: list[torch.Tensor],
    B: list[torch.Tensor],
    out: list[torch.Tensor],
    quantization_params: list[Any | None],
    layout: str,
    gelu: bool,
    use_bias: bool,
) -> None:
    reason = _te_cutlass_grouped_gemm_fallback_reason(
        A=A,
        B=B,
        out=out,
        quantization_params=quantization_params,
        layout=layout,
        gelu=gelu,
        use_bias=use_bias,
    )
    if reason is None:
        return
    raise RuntimeError(
        "ART requires Transformer Engine CUTLASS grouped GEMM, but this "
        f"grouped GEMM call would use the fallback path: {reason}. "
        "Required shape: Hopper SM90, BF16/FP16 A/B/out tensors with matching "
        "dtypes, no grouped bias/GELU/debug quantizer path, and uniform B K "
        "dimension divisible by 128."
    )


def _te_cutlass_grouped_gemm_fallback_reason(
    *,
    A: list[torch.Tensor],
    B: list[torch.Tensor],
    out: list[torch.Tensor],
    quantization_params: list[Any | None],
    layout: str,
    gelu: bool,
    use_bias: bool,
) -> str | None:
    torch = _torch()
    # Keep this in sync with TE's validated CUTLASS grouped-GEMM selector. ART
    # currently supports the TE 2.11 SM90/Hopper path; SM100/Blackwell support
    # should come from an upgraded Transformer Engine build using this same API.
    if not A or not B or not out:
        return "A, B, and out must all be non-empty"
    if len(layout) < 2:
        return f"invalid layout {layout!r}"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if (device_reason := _sm90_device_reason(A[0])) is not None:
        return device_reason
    if gelu:
        return "grouped GELU pre-activation output is not supported"
    if use_bias:
        return "grouped bias is not supported"
    if (
        quantization_params
        and quantization_params[0] is not None
        and type(quantization_params[0]).__name__ == "DebugQuantizer"
    ):
        return "Transformer Engine debug quantizer bypasses grouped GEMM"
    if (
        dtype_reason := _dtype_reason(
            _tensor_dtype(A[0], "A[0]"),
            _tensor_dtype(B[0], "B[0]"),
            _tensor_dtype(out[0], "out[0]"),
        )
    ) is not None:
        return dtype_reason
    return _uniform_b_k128_reason(B, transb=layout[1] == "T")


def _sm90_device_reason(tensor: torch.Tensor) -> str | None:
    torch = _torch()
    device = getattr(tensor, "device", None)
    device_index = torch.cuda.current_device()
    if isinstance(device, torch.device) and device.type == "cuda":
        device_index = 0 if device.index is None else device.index
    capability = _DEVICE_CAPABILITIES.get(device_index)
    if capability is None:
        capability = torch.cuda.get_device_capability(device_index)
        _DEVICE_CAPABILITIES[device_index] = capability
    if capability != (9, 0):
        return f"CUDA device {device_index} has capability {capability}, not SM90"
    return None


def _tensor_dtype(tensor: torch.Tensor, name: str) -> torch.dtype | str:
    torch = _torch()
    dtype = getattr(tensor, "dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else f"{name} has no torch dtype"


def _dtype_reason(
    a_dtype: torch.dtype | str,
    b_dtype: torch.dtype | str,
    out_dtype: torch.dtype | str,
) -> str | None:
    torch = _torch()
    for dtype in (a_dtype, b_dtype, out_dtype):
        if isinstance(dtype, str):
            return dtype
    if a_dtype != b_dtype or a_dtype != out_dtype:
        return f"dtype mismatch A={a_dtype}, B={b_dtype}, out={out_dtype}"
    if a_dtype not in {torch.bfloat16, torch.float16}:
        return f"dtype {a_dtype} is not BF16 or FP16"
    return None


def _uniform_b_k128_reason(B: list[torch.Tensor], *, transb: bool) -> str | None:
    k_dim = 0 if transb else 1
    expected_k: int | None = None
    for index, tensor in enumerate(B):
        shape = tuple(getattr(tensor, "shape", ()))
        if len(shape) <= k_dim:
            return f"B[{index}] shape {shape} has no K dimension"
        k_value = int(shape[k_dim])
        if expected_k is None:
            expected_k = k_value
            if expected_k % 128 != 0:
                return f"B K dimension {expected_k} is not divisible by 128"
        elif k_value != expected_k:
            return (
                f"B K dimension is not uniform: B[0] has {expected_k}, "
                f"B[{index}] has {k_value}"
            )
    return None


def _patch_imported_grouped_linear(original: Any, guarded: Any) -> None:
    for module_name in (
        "transformer_engine.pytorch.module.grouped_linear",
        "transformer_engine.pytorch.module.linear",
    ):
        module = sys.modules.get(module_name)
        if (
            module is not None
            and getattr(module, "general_grouped_gemm", None) is original
        ):
            setattr(module, "general_grouped_gemm", guarded)


def _patch_cpp_extensions_export(original: Any, guarded: Any) -> None:
    module = sys.modules.get("transformer_engine.pytorch.cpp_extensions")
    if module is not None and getattr(module, "general_grouped_gemm", None) is original:
        setattr(module, "general_grouped_gemm", guarded)


def _torch() -> Any:
    import torch

    return torch
