from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

from ..types import MegatronRuntimeConfig, MegatronTopologyConfig

_MEGATRON_RUNTIME_CONFIG: MegatronRuntimeConfig | None = None


def init_megatron_runtime_config(
    config: MegatronRuntimeConfig | Mapping[str, Any] | None = None,
    *,
    topology: MegatronTopologyConfig | Mapping[str, int | None] | None = None,
    packed_sequence_length: int | None = None,
    snapshot_pool_capacity: int = 2,
    compile_cache: bool | None = None,
    streaming_weight_offload: bool = False,
) -> MegatronRuntimeConfig:
    global _MEGATRON_RUNTIME_CONFIG
    if config is None:
        config = {
            "topology": topology,
            "packed_sequence_length": packed_sequence_length,
            "snapshot_pool_capacity": snapshot_pool_capacity,
            "compile_cache": (
                os.environ.get("ART_MEGATRON_COMPILE_CACHE", "0").lower()
                in {"1", "true", "yes", "on"}
                if compile_cache is None
                else compile_cache
            ),
            "streaming_weight_offload": streaming_weight_offload,
        }
    runtime_config = MegatronRuntimeConfig.model_validate(config)
    if _MEGATRON_RUNTIME_CONFIG is None:
        _MEGATRON_RUNTIME_CONFIG = runtime_config
    elif _MEGATRON_RUNTIME_CONFIG != runtime_config:
        raise ValueError(
            "Megatron runtime config is already initialized with "
            f"{_MEGATRON_RUNTIME_CONFIG.model_dump(mode='json')}, got "
            f"{runtime_config.model_dump(mode='json')}."
        )
    return _MEGATRON_RUNTIME_CONFIG


def get_megatron_runtime_config() -> MegatronRuntimeConfig:
    if _MEGATRON_RUNTIME_CONFIG is None:
        raise RuntimeError(
            "Call art.init_megatron_runtime_config(...) before using MegatronBackend."
        )
    return _MEGATRON_RUNTIME_CONFIG
