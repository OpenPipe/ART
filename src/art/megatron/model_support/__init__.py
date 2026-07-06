from art.megatron.model_support.registry import (
    DEFAULT_DENSE_SPEC,
    GEMMA4_DENSE_MODELS,
    GEMMA4_DENSE_SPEC,
    GEMMA4_MOE_MODELS,
    GEMMA4_MOE_SPEC,
    PROBE_ONLY_MODEL_SUPPORT_SPECS,
    QWEN3_5_DENSE_MODELS,
    QWEN3_5_DENSE_SPEC,
    QWEN3_5_MODELS,
    QWEN3_5_MOE_MODELS,
    QWEN3_5_MOE_SPEC,
    QWEN3_DENSE_MODELS,
    QWEN3_DENSE_SPEC,
    QWEN3_MOE_MODELS,
    QWEN3_MOE_SPEC,
    VALIDATED_MODEL_SUPPORT_SPECS,
    UnsupportedModelArchitectureError,
    default_target_modules_for_model,
    ensure_model_support_bridge_registered_for_spec,
    get_model_support_handler,
    get_model_support_handler_for_spec,
    get_model_support_spec,
    is_model_support_registered,
    list_model_support_specs,
    model_requires_merged_rollout,
    model_uses_expert_parallel,
    native_vllm_lora_status_for_model,
)
from art.megatron.model_support.spec import (
    ArchitectureReport,
    DependencyFloor,
    LayerFamilyInstance,
    ModelSupportHandler,
    ModelSupportSpec,
    NativeVllmLoraStatus,
    RolloutWeightsMode,
)

_LAZY_EXPORT_MODULES = {
    "inspect_architecture": "art.megatron.model_support.discovery",
    "summarize_layer_families": "art.megatron.model_support.discovery",
}


def __getattr__(name: str):
    import importlib

    try:
        module_name = _LAZY_EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "ArchitectureReport",
    "DEFAULT_DENSE_SPEC",
    "DependencyFloor",
    "GEMMA4_DENSE_MODELS",
    "GEMMA4_DENSE_SPEC",
    "GEMMA4_MOE_MODELS",
    "GEMMA4_MOE_SPEC",
    "LayerFamilyInstance",
    "ModelSupportHandler",
    "ModelSupportSpec",
    "NativeVllmLoraStatus",
    "QWEN3_5_DENSE_MODELS",
    "QWEN3_5_DENSE_SPEC",
    "QWEN3_5_MODELS",
    "QWEN3_5_MOE_MODELS",
    "QWEN3_DENSE_MODELS",
    "QWEN3_DENSE_SPEC",
    "QWEN3_MOE_MODELS",
    "QWEN3_MOE_SPEC",
    "QWEN3_5_MOE_SPEC",
    "PROBE_ONLY_MODEL_SUPPORT_SPECS",
    "RolloutWeightsMode",
    "UnsupportedModelArchitectureError",
    "VALIDATED_MODEL_SUPPORT_SPECS",
    "default_target_modules_for_model",
    "ensure_model_support_bridge_registered_for_spec",
    "get_model_support_handler",
    "get_model_support_handler_for_spec",
    "get_model_support_spec",
    "inspect_architecture",
    "is_model_support_registered",
    "list_model_support_specs",
    "model_uses_expert_parallel",
    "model_requires_merged_rollout",
    "native_vllm_lora_status_for_model",
    "summarize_layer_families",
]
