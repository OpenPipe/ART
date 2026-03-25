"""Validation functions for model configuration."""

from .model import InternalModelConfig, RolloutWeightsMode

QWEN3_5_MOE_MODELS = {
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-397B-A17B",
}


def is_dedicated_mode(config: InternalModelConfig) -> bool:
    """Return True if the config specifies dedicated mode (separate training and inference GPUs)."""
    return "trainer_gpu_ids" in config and "inference_gpu_ids" in config


def _rollout_weights_mode(config: InternalModelConfig) -> RolloutWeightsMode:
    mode = config.get("rollout_weights_mode", "lora")
    if mode in {"lora", "merged"}:
        return mode
    raise ValueError("rollout_weights_mode must be either 'lora' or 'merged'")


def _is_qwen3_5_moe_model(config: InternalModelConfig) -> bool:
    model_name = config.get("engine_args", {}).get("model")
    return model_name in QWEN3_5_MOE_MODELS


def validate_dedicated_config(config: InternalModelConfig) -> None:
    """Validate dedicated mode GPU configuration.

    Raises ValueError if the configuration is invalid.
    Does nothing if neither trainer_gpu_ids nor inference_gpu_ids is set (shared mode).
    """
    has_trainer = "trainer_gpu_ids" in config
    has_inference = "inference_gpu_ids" in config
    rollout_weights_mode = _rollout_weights_mode(config)

    if has_trainer != has_inference:
        raise ValueError(
            "trainer_gpu_ids and inference_gpu_ids must both be set or both unset"
        )

    if rollout_weights_mode == "merged" and not has_trainer:
        raise ValueError(
            "rollout_weights_mode='merged' requires dedicated mode "
            "(set both trainer_gpu_ids and inference_gpu_ids)"
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

    # Reject settings that are incompatible with dedicated mode
    if config.get("init_args", {}).get("fast_inference"):
        raise ValueError(
            "fast_inference is incompatible with dedicated mode "
            "(dedicated mode runs vLLM as a subprocess, not in-process)"
        )

    if config.get("engine_args", {}).get("enable_sleep_mode"):
        raise ValueError(
            "enable_sleep_mode is incompatible with dedicated mode "
            "(dedicated mode runs vLLM on a separate GPU, sleep/wake is not needed)"
        )

    if _is_qwen3_5_moe_model(config) and rollout_weights_mode == "lora":
        raise ValueError(
            "Qwen3.5-MoE models require rollout_weights_mode='merged' with the "
            "current vLLM version because direct LoRA inference is currently broken"
        )
