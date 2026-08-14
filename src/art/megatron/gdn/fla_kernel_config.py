from __future__ import annotations

from typing import Any

import torch

_KEYS = [
    "H",
    "HV",
    "K",
    "V",
    "BT",
    "BV",
    "USE_G",
    "USE_EXP2",
    "TRANSPOSE_STATE",
]
_HOPPER_CONFIGS = frozenset(
    (bv, warps, stages) for warps in (2, 4) for stages in (2, 3, 4) for bv in (32, 64)
)
_HOPPER_CONFIG = (32, 4, 2)


def _signature(config: Any) -> tuple[int, int, int]:
    return int(config.kwargs["BV"]), int(config.num_warps), int(config.num_stages)


def configure_fla_gdn_hopper_backward() -> None:
    if not torch.cuda.is_available():
        return
    if torch.cuda.get_device_capability() != (9, 0):
        return
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 as kernel,
    )

    autotuner = kernel.fn
    configs = tuple(autotuner.configs)
    signatures = frozenset(_signature(config) for config in configs)
    if signatures == {_HOPPER_CONFIG}:
        return
    if list(autotuner.keys) != _KEYS or signatures != _HOPPER_CONFIGS:
        raise RuntimeError("unsupported FLA GDN backward autotuner structure")
    # FLA 0.5.0 otherwise compiles 12 schedules whenever prefix-tree execution
    # first exposes a dense/varlen state mode. H200 captures found this schedule
    # exact or within 8.6% of the tuned kernel and avoid multi-second mid-run tuning.
    autotuner.configs = [
        next(config for config in configs if _signature(config) == _HOPPER_CONFIG)
    ]
