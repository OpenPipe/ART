"""Validation functions for model configuration."""

from .model import InternalModelConfig, RolloutWeightsMode


def is_dedicated_mode(config: InternalModelConfig) -> bool:
    """Return True if the config specifies dedicated mode (separate training and inference GPUs)."""
    return "trainer_gpu_ids" in config and "inference_gpu_ids" in config


def _rollout_weights_mode(config: InternalModelConfig) -> RolloutWeightsMode:
    mode = config.get("rollout_weights_mode", "lora")
    if mode in {"lora", "merged"}:
        return mode
    raise ValueError("rollout_weights_mode must be either 'lora' or 'merged'")


def _engine_parallel_size(config: InternalModelConfig) -> int:
    engine_args = config.get("engine_args", {})
    tensor_parallel_size = engine_args.get("tensor_parallel_size", 1)
    pipeline_parallel_size = engine_args.get("pipeline_parallel_size", 1)
    tp = 1 if tensor_parallel_size is None else int(tensor_parallel_size)
    pp = 1 if pipeline_parallel_size is None else int(pipeline_parallel_size)
    if tp < 1 or pp < 1:
        raise ValueError(
            "engine_args tensor_parallel_size and pipeline_parallel_size must be positive"
        )
    return tp * pp


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

    engine_args = config.get("engine_args", {})
    if "tensor_parallel_size" in engine_args or "pipeline_parallel_size" in engine_args:
        inference_parallel_size = _engine_parallel_size(config)
        if inference_parallel_size != len(inference_gpu_ids):
            raise ValueError(
                "Dedicated inference GPU count must match engine_args "
                "tensor_parallel_size * pipeline_parallel_size"
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
