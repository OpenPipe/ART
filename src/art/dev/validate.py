"""Validation functions for model configuration."""

from collections.abc import Mapping
from typing import cast

from .model import (
    InternalModelConfig,
    RolloutWeightUpdateMode,
    VllmRuntimeMode,
)


def _vllm_runtime_mode(config: InternalModelConfig) -> VllmRuntimeMode:
    runtime_config = config.get("vllm_runtime", {})
    if not isinstance(runtime_config, Mapping):
        raise ValueError("vllm_runtime must be a mapping")
    mode = runtime_config.get("mode", "managed")
    if mode in {"managed", "external"}:
        return cast(VllmRuntimeMode, mode)
    raise ValueError("vllm_runtime.mode must be either 'managed' or 'external'")


def is_external_vllm_mode(config: InternalModelConfig) -> bool:
    return _vllm_runtime_mode(config) == "external"


def is_dedicated_mode(config: InternalModelConfig) -> bool:
    """Return True if the config specifies dedicated mode (separate training and inference GPUs)."""
    return is_external_vllm_mode(config) or (
        "trainer_gpu_ids" in config and "inference_gpu_ids" in config
    )


def _rollout_weight_update_mode(
    config: InternalModelConfig,
) -> RolloutWeightUpdateMode:
    mode = config.get("rollout_weight_update_mode", "step_lora")
    if mode in {"step_lora", "in_flight_lora"}:
        return mode
    raise ValueError(
        "rollout_weight_update_mode must be either 'step_lora' or 'in_flight_lora'"
    )


def validate_dedicated_config(config: InternalModelConfig) -> None:
    """Validate dedicated mode GPU configuration.

    Raises ValueError if the configuration is invalid.
    Does nothing if neither trainer_gpu_ids nor inference_gpu_ids is set (shared mode).
    """
    if "rollout_weights_mode" in config:
        raise ValueError(
            "rollout_weights_mode has been removed; ART always serves native LoRA adapters"
        )
    has_trainer = "trainer_gpu_ids" in config
    has_inference = "inference_gpu_ids" in config
    _rollout_weight_update_mode(config)
    external = is_external_vllm_mode(config)

    if external:
        runtime_config = config.get("vllm_runtime", {})
        assert isinstance(runtime_config, Mapping)
        if not runtime_config.get("server_url"):
            raise ValueError("vllm_runtime.server_url is required for external mode")
        if has_trainer and not config["trainer_gpu_ids"]:
            raise ValueError("trainer_gpu_ids must be non-empty")
        if "fast_inference" in config.get("init_args", {}):
            raise ValueError(
                "fast_inference is no longer supported; ART always uses an external "
                "vLLM runtime"
            )
        return

    if has_trainer != has_inference:
        raise ValueError(
            "trainer_gpu_ids and inference_gpu_ids must both be set or both unset"
        )

    if "fast_inference" in config.get("init_args", {}):
        raise ValueError(
            "fast_inference is no longer supported; ART always uses an external "
            "vLLM runtime"
        )

    if not has_trainer:
        return

    trainer_gpu_ids = config["trainer_gpu_ids"]
    inference_gpu_ids = config["inference_gpu_ids"]

    if not trainer_gpu_ids:
        raise ValueError("trainer_gpu_ids must be non-empty")

    if not inference_gpu_ids:
        raise ValueError("inference_gpu_ids must be non-empty")

    if set(trainer_gpu_ids) & set(inference_gpu_ids):
        raise ValueError("trainer_gpu_ids and inference_gpu_ids must not overlap")

    inference_tp = int(config.get("engine_args", {}).get("tensor_parallel_size", 1))
    if len(inference_gpu_ids) > 1 and inference_tp != len(inference_gpu_ids):
        raise ValueError(
            "Multi-GPU inference requires engine_args.tensor_parallel_size to "
            "match len(inference_gpu_ids)"
        )

    if config.get("engine_args", {}).get("enable_sleep_mode"):
        raise ValueError(
            "enable_sleep_mode is incompatible with dedicated mode "
            "(shared-GPU mode uses runtime sleep/wake; dedicated mode does not)"
        )
