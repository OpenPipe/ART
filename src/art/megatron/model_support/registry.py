import importlib

from art.megatron.model_support.spec import (
    ALL_HANDLER_METAS,
    DEFAULT_DENSE_META,
    QWEN3_5_DENSE_META,
    QWEN3_5_MOE_META,
    QWEN3_DENSE_META,
    QWEN3_MOE_META,
    DependencyFloor,
    HandlerMeta,
    ModelSupportHandler,
    ModelSupportSpec,
)

# Handler modules top-level import megatron-core, so we keep the registry
# importable without megatron-core by deferring those imports until something
# actually needs a handler instance. The handler `key` / `native_vllm_lora_status`
# values shared between spec construction (here) and the handler classes live
# in `handler_meta.py`, which has no megatron dependency.
_HANDLER_METAS_BY_KEY: dict[str, HandlerMeta] = {
    meta.key: meta for meta in ALL_HANDLER_METAS
}
_HANDLERS_BY_KEY: dict[str, ModelSupportHandler] = {}


def _load_handler(handler_key: str) -> ModelSupportHandler:
    cached = _HANDLERS_BY_KEY.get(handler_key)
    if cached is not None:
        return cached
    meta = _HANDLER_METAS_BY_KEY[handler_key]
    handler = getattr(importlib.import_module(meta.module), meta.attr)
    _HANDLERS_BY_KEY[handler_key] = handler
    return handler


_DENSE_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_QWEN3_MOE_TARGET_MODULES = (*_DENSE_TARGET_MODULES, "experts")

_QWEN3_5_DENSE_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

_QWEN3_5_MOE_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "experts",
)

DEFAULT_DENSE_SPEC = ModelSupportSpec(
    key=DEFAULT_DENSE_META.key,
    handler_key=DEFAULT_DENSE_META.key,
    default_target_modules=_DENSE_TARGET_MODULES,
    native_vllm_lora_status=DEFAULT_DENSE_META.native_vllm_lora_status,
)

QWEN3_MOE_SPEC = ModelSupportSpec(
    key=QWEN3_MOE_META.key,
    handler_key=QWEN3_MOE_META.key,
    model_names=(
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B-Base",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ),
    default_target_modules=_QWEN3_MOE_TARGET_MODULES,
    native_vllm_lora_status=QWEN3_MOE_META.native_vllm_lora_status,
)

QWEN3_DENSE_SPEC = ModelSupportSpec(
    key=QWEN3_DENSE_META.key,
    handler_key=QWEN3_DENSE_META.key,
    model_names=(
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-14B-Base",
        "OpenPipe/Qwen3-14B-Instruct",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-32B-Base",
    ),
    default_target_modules=_DENSE_TARGET_MODULES,
    native_vllm_lora_status=QWEN3_DENSE_META.native_vllm_lora_status,
)

QWEN3_5_DENSE_SPEC = ModelSupportSpec(
    key=QWEN3_5_DENSE_META.key,
    handler_key=QWEN3_5_DENSE_META.key,
    model_names=(
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.6-27B",
    ),
    default_target_modules=_QWEN3_5_DENSE_TARGET_MODULES,
    native_vllm_lora_status=QWEN3_5_DENSE_META.native_vllm_lora_status,
    dependency_floor=DependencyFloor(
        megatron_bridge="e049cc00c24d03e2ae45d2608c7a44e2d2364e3d",
    ),
)

QWEN3_5_MOE_SPEC = ModelSupportSpec(
    key=QWEN3_5_MOE_META.key,
    handler_key=QWEN3_5_MOE_META.key,
    model_names=(
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.6-35B-A3B",
    ),
    default_target_modules=_QWEN3_5_MOE_TARGET_MODULES,
    native_vllm_lora_status=QWEN3_5_MOE_META.native_vllm_lora_status,
    dependency_floor=DependencyFloor(
        megatron_bridge="e049cc00c24d03e2ae45d2608c7a44e2d2364e3d",
    ),
)

VALIDATED_MODEL_SUPPORT_SPECS = (
    QWEN3_MOE_SPEC,
    QWEN3_DENSE_SPEC,
    QWEN3_5_MOE_SPEC,
    QWEN3_5_DENSE_SPEC,
)
PROBE_ONLY_MODEL_SUPPORT_SPECS = ()
_ALL_MODEL_SUPPORT_SPECS = (
    DEFAULT_DENSE_SPEC,
    *VALIDATED_MODEL_SUPPORT_SPECS,
    *PROBE_ONLY_MODEL_SUPPORT_SPECS,
)
_SPECS_BY_KEY = {spec.key: spec for spec in _ALL_MODEL_SUPPORT_SPECS}
_SPECS_BY_MODEL = {
    model_name: spec
    for spec in VALIDATED_MODEL_SUPPORT_SPECS
    for model_name in spec.model_names
}
_UNVALIDATED_ARCH_SPECS_BY_MODEL = {
    model_name: spec
    for spec in PROBE_ONLY_MODEL_SUPPORT_SPECS
    for model_name in spec.model_names
}
QWEN3_DENSE_MODELS = frozenset(QWEN3_DENSE_SPEC.model_names)
QWEN3_MOE_MODELS = frozenset(QWEN3_MOE_SPEC.model_names)
QWEN3_5_DENSE_MODELS = frozenset(QWEN3_5_DENSE_SPEC.model_names)
QWEN3_5_MOE_MODELS = frozenset(QWEN3_5_MOE_SPEC.model_names)
QWEN3_5_MODELS = QWEN3_5_DENSE_MODELS | QWEN3_5_MOE_MODELS


class UnsupportedModelArchitectureError(ValueError):
    """Raised when a model has not passed the Megatron support workflow."""


def get_model_support_spec(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> ModelSupportSpec:
    if spec := _SPECS_BY_MODEL.get(base_model):
        return spec
    if allow_unvalidated_arch:
        return _UNVALIDATED_ARCH_SPECS_BY_MODEL.get(base_model, DEFAULT_DENSE_SPEC)
    supported = ", ".join(sorted(_SPECS_BY_MODEL))
    raise UnsupportedModelArchitectureError(
        f"{base_model!r} has not passed the Megatron model-support workflow. "
        "Pass allow_unvalidated_arch=True only for explicit validation/probing. "
        f"Supported models: {supported}."
    )


def get_model_support_handler(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> ModelSupportHandler:
    return get_model_support_handler_for_spec(
        get_model_support_spec(
            base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
    )


def get_model_support_handler_for_spec(
    spec: ModelSupportSpec,
) -> ModelSupportHandler:
    return _load_handler(spec.handler_key)


def default_target_modules_for_model(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> list[str]:
    return list(
        get_model_support_spec(
            base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        ).default_target_modules
    )


def native_vllm_lora_status_for_model(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> str:
    return get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    ).native_vllm_lora_status


def model_requires_merged_rollout(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> bool:
    return (
        get_model_support_spec(
            base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        ).default_rollout_weights_mode
        == "merged"
    )


def model_uses_expert_parallel(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> bool:
    return bool(
        get_model_support_handler(
            base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        ).is_moe
    )


def is_model_support_registered(base_model: str) -> bool:
    return base_model in _SPECS_BY_MODEL


def list_model_support_specs() -> list[ModelSupportSpec]:
    return list(VALIDATED_MODEL_SUPPORT_SPECS)
