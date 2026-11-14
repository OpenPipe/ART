"""Exclude the measured FLA 0.5.0 fwd_h races on datacenter Blackwell."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_MODULE = "fla.ops.common.chunk_delta_h"
_KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
# FLA 0.5.0, on devices satisfying its Ampere/Ada shared-memory checks.
# https://github.com/fla-org/flash-linear-attention/blob/v0.5.0/fla/ops/common/chunk_delta_h.py
_EXPECTED = tuple(
    (warps, stages, bv) for warps in (2, 4) for stages in (2, 3, 4) for bv in (32, 64)
)
_RACY = frozenset({(4, 2, 32), (4, 2, 64), (4, 3, 32)})
_SAFE = tuple(config for config in _EXPECTED if config not in _RACY)


def _config_tuple(config: Any) -> tuple[int, int, int]:
    if (
        set(config.kwargs) != {"BV"}
        or config.num_ctas != 1
        or config.maxnreg is not None
        or config.pre_hook is not None
        or config.ir_override is not None
    ):
        raise ValueError("unexpected Triton config options")
    return config.num_warps, config.num_stages, config.kwargs["BV"]


def filter_fwd_h_configs(
    kernel: Any, *, fla_version: str, capability: tuple[int, int]
) -> bool:
    """Filter one verified Heuristics -> Autotuner -> JITFunction in place.

    Return whether the verified safe list is in effect. Repeated imports/calls
    preserve tuned safe choices. No CUDA calls are made here, allowing CPU tests.
    """
    if capability[0] != 10:
        return False
    try:
        if fla_version != "0.5.0":
            raise ValueError(f"expected FLA 0.5.0, found {fla_version}")
        tuner = kernel.fn
        jit = tuner.fn
        if jit.__name__ != _KERNEL or jit.__module__ != _MODULE:
            raise ValueError("unexpected kernel name/module")
        configs = tuple(_config_tuple(config) for config in tuner.configs)
        # Only accept a previously filtered list if this tuner carries our marker.
        already_filtered = getattr(tuner, "_art_sm100_safe_fwd_h", False)
        if configs != (_SAFE if already_filtered else _EXPECTED):
            raise ValueError(f"unexpected config tuples: {configs}")
        if not isinstance(tuner.cache, dict):
            raise ValueError("unexpected autotune cache")
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("Skipping ART Blackwell FLA fwd_h config filter: %s", exc)
        return False

    if not already_filtered:
        tuner.configs = [
            config for config in tuner.configs if _config_tuple(config) not in _RACY
        ]
        tuner.cache.clear()
        # These are diagnostic state, but must not advertise a racy selection.
        for attr in ("best_config", "configs_timings"):
            if hasattr(tuner, attr):
                delattr(tuner, attr)
        # Triton 3.6's disk autotune key hashes the candidate config list, so the
        # nine-config list cannot reuse results from the original twelve.
        tuner._art_sm100_safe_fwd_h = True
    logger.debug("ART Blackwell FLA %s configs (warps, stages, BV): %s", _KERNEL, _SAFE)
    return True


def ensure_blackwell_fwd_h_configs() -> None:
    """Apply before ART's GDN calls, after the rank selects its CUDA device.

    sm103 is covered conservatively; only sm100 has separate measurements.
    CPU, Hopper and other compute-capability majors retain FLA's defaults.
    """
    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        return
    try:
        # The kernels live in fla-core; the public package can upgrade it
        # independently, so verify both distributions before importing them.
        versions = {
            name: version(name) for name in ("flash-linear-attention", "fla-core")
        }
        if any(value != "0.5.0" for value in versions.values()):
            logger.warning(
                "Skipping ART Blackwell FLA fwd_h config filter: expected "
                "flash-linear-attention/fla-core 0.5.0, found %s",
                versions,
            )
            return
        module = importlib.import_module(_MODULE)
        kernel = getattr(module, _KERNEL)
    except (ImportError, AttributeError, PackageNotFoundError) as exc:
        logger.warning("Skipping ART Blackwell FLA fwd_h config filter: %s", exc)
        return
    filter_fwd_h_configs(kernel, fla_version="0.5.0", capability=capability)
