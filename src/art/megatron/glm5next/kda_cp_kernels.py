"""Use TF32x3 for FLA 0.5's FP32 KDA CP affine chains.

The public CP implementation predates FLA #1546. Launch options supply the
precision fix without copying the upstream kernels or changing GDN launches.
Only the two helper aliases in the KDA forward/backward modules are replaced.
"""

from __future__ import annotations

import importlib
from types import FunctionType
from typing import Any


class _PreciseLaunch:
    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel

    def __getitem__(self, grid: Any) -> Any:
        launch = self.kernel[grid]

        def run(*args: Any, **kwargs: Any) -> Any:
            kwargs["default_dot_input_precision"] = "tf32x3"
            return launch(*args, **kwargs)

        return run


def install_kda_cp_precision() -> None:
    cp = importlib.import_module("fla.ops.cp.chunk_delta_h")
    forward = importlib.import_module("fla.ops.kda.chunk_fwd")
    backward = importlib.import_module("fla.ops.kda.chunk_bwd")
    if getattr(forward, "_art_kda_cp_precision", False):
        return
    kernels = (
        "pre_process_fwd_kernel_merged",
        "pre_process_bwd_kernel_merged",
        "merge_fwd_bwd_kernel",
    )
    if any(not hasattr(cp, name) for name in kernels):
        raise RuntimeError(
            "ART KDA CP precision shim requires the pinned FLA 0.5 kernel API"
        )
    namespace = dict(vars(cp))
    for name in kernels:
        namespace[name] = _PreciseLaunch(getattr(cp, name))
    for module, name in (
        (forward, "chunk_gated_delta_rule_fwd_h_pre_process"),
        (backward, "chunk_gated_delta_rule_bwd_dhu_pre_process"),
    ):
        original = getattr(cp, name)
        clone = FunctionType(
            original.__code__,
            namespace,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        clone.__kwdefaults__ = original.__kwdefaults__
        setattr(module, name, clone)
    setattr(forward, "_art_kda_cp_precision", True)
