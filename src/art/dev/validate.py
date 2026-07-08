"""Validation functions for model configuration."""

from collections.abc import Mapping
from typing import cast

from .model import InternalModelConfig, RolloutWeightsMode, VllmRuntimeMode


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


def _rollout_weights_mode(config: InternalModelConfig) -> RolloutWeightsMode:
    mode = config.get("rollout_weights_mode", "lora")
    if mode in {"lora", "merged"}:
        return mode
    raise ValueError("rollout_weights_mode must be either 'lora' or 'merged'")


def validate_dedicated_config(config: InternalModelConfig) -> None:
    """Validate dedicated mode GPU configuration.

    Raises ValueError if the configuration is invalid.
    Does nothing if neither trainer_gpu_ids nor inference_gpu_ids is set (shared mode).
    """
    has_trainer = "trainer_gpu_ids" in config
    has_inference = "inference_gpu_ids" in config
    rollout_weights_mode = _rollout_weights_mode(config)
    external = is_external_vllm_mode(config)

    if external:
        runtime_config = config.get("vllm_runtime", {})
        assert isinstance(runtime_config, Mapping)
        if not runtime_config.get("server_url"):
            raise ValueError("vllm_runtime.server_url is required for external mode")
        if rollout_weights_mode != "lora":
            raise ValueError(
                "vllm_runtime.mode='external' requires rollout_weights_mode='lora'"
            )
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

    if rollout_weights_mode == "merged" and not has_trainer:
        raise ValueError(
            "rollout_weights_mode='merged' requires dedicated mode "
            "(set both trainer_gpu_ids and inference_gpu_ids)"
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

    if len(inference_gpu_ids) > 1:
        raise ValueError(
            "Multi-GPU inference not yet supported; inference_gpu_ids must have exactly one GPU"
        )

    if trainer_gpu_ids[0] != 0:
        raise ValueError(
            "trainer_gpu_ids must start at GPU 0 (training runs in-process)"
        )

    expected = list(range(len(trainer_gpu_ids)))
    if trainer_gpu_ids != expected:
        raise ValueError(
            "trainer_gpu_ids must be contiguous starting from 0 (e.g., [0], [0,1])"
        )

    if config.get("engine_args", {}).get("enable_sleep_mode"):
        raise ValueError(
            "enable_sleep_mode is incompatible with dedicated mode "
            "(shared-GPU mode uses runtime sleep/wake; dedicated mode does not)"
        )
