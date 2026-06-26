from importlib import import_module

from art.megatron.model_support.spec import (
    DependencyFloor,
    ModelSupportHandler,
    ModelSupportSpec,
    NativeVllmLoraStatus,
)

_DEFAULT_DENSE_HANDLER_KEY = "default_dense"
_QWEN3_DENSE_HANDLER_KEY = "qwen3_dense"
_QWEN3_MOE_HANDLER_KEY = "qwen3_moe"
_QWEN3_5_DENSE_HANDLER_KEY = "qwen3_5_dense"
_QWEN3_5_MOE_HANDLER_KEY = "qwen3_5_moe"
_GEMMA4_DENSE_HANDLER_KEY = "gemma4_dense"
_GEMMA4_MOE_HANDLER_KEY = "gemma4_moe"
_GPT_OSS_MOE_HANDLER_KEY = "gpt_oss_moe"
_VALIDATED_NATIVE_VLLM_LORA_STATUS: NativeVllmLoraStatus = "validated"
_WIP_NATIVE_VLLM_LORA_STATUS: NativeVllmLoraStatus = "wip"
_DISABLED_NATIVE_VLLM_LORA_STATUS: NativeVllmLoraStatus = "disabled"

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
_GEMMA4_MOE_TARGET_MODULES = (*_DENSE_TARGET_MODULES, "experts")
_GPT_OSS_MOE_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "experts")

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
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_DENSE_SPEC = ModelSupportSpec(
    key="default_dense",
    handler_key=_DEFAULT_DENSE_HANDLER_KEY,
    default_target_modules=_DENSE_TARGET_MODULES,
    native_vllm_lora_status=_DISABLED_NATIVE_VLLM_LORA_STATUS,
)

QWEN3_MOE_SPEC = ModelSupportSpec(
    key="qwen3_moe",
    handler_key=_QWEN3_MOE_HANDLER_KEY,
    is_moe=True,
    model_names=(
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B-Base",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
    ),
    default_target_modules=_QWEN3_MOE_TARGET_MODULES,
    native_vllm_lora_status=_VALIDATED_NATIVE_VLLM_LORA_STATUS,
)

QWEN3_DENSE_SPEC = ModelSupportSpec(
    key="qwen3_dense",
    handler_key=_QWEN3_DENSE_HANDLER_KEY,
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
    native_vllm_lora_status=_VALIDATED_NATIVE_VLLM_LORA_STATUS,
)

QWEN3_5_DENSE_SPEC = ModelSupportSpec(
    key="qwen3_5_dense",
    handler_key=_QWEN3_5_DENSE_HANDLER_KEY,
    model_names=(
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-27B",
        "Qwen/Qwen3.6-27B",
    ),
    default_target_modules=_QWEN3_5_DENSE_TARGET_MODULES,
    native_vllm_lora_status=_VALIDATED_NATIVE_VLLM_LORA_STATUS,
    dependency_floor=DependencyFloor(
        megatron_bridge="e049cc00c24d03e2ae45d2608c7a44e2d2364e3d",
    ),
)

QWEN3_5_MOE_SPEC = ModelSupportSpec(
    key="qwen3_5_moe",
    handler_key=_QWEN3_5_MOE_HANDLER_KEY,
    is_moe=True,
    model_names=(
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.6-35B-A3B",
    ),
    default_target_modules=_QWEN3_5_MOE_TARGET_MODULES,
    native_vllm_lora_status=_VALIDATED_NATIVE_VLLM_LORA_STATUS,
    dependency_floor=DependencyFloor(
        megatron_bridge="e049cc00c24d03e2ae45d2608c7a44e2d2364e3d",
    ),
)

GEMMA4_MOE_SPEC = ModelSupportSpec(
    key="gemma4_moe",
    handler_key=_GEMMA4_MOE_HANDLER_KEY,
    is_moe=True,
    model_names=(
        "google/gemma-4-26B-A4B",
        "google/gemma-4-26B-A4B-it",
    ),
    default_target_modules=_GEMMA4_MOE_TARGET_MODULES,
    native_vllm_lora_status=_VALIDATED_NATIVE_VLLM_LORA_STATUS,
    dependency_floor=DependencyFloor(
        transformers="5.6.2",
        megatron_bridge="e1a207ac757e5d0ed94d8ffbe1cbd28e81d8c084",
    ),
)

GEMMA4_DENSE_SPEC = ModelSupportSpec(
    key="gemma4_dense",
    handler_key=_GEMMA4_DENSE_HANDLER_KEY,
    model_names=(
        "google/gemma-4-31B",
        "google/gemma-4-31B-it",
    ),
    default_target_modules=_DENSE_TARGET_MODULES,
    native_vllm_lora_status=_VALIDATED_NATIVE_VLLM_LORA_STATUS,
    dependency_floor=DependencyFloor(
        transformers="5.6.2",
        megatron_bridge="e1a207ac757e5d0ed94d8ffbe1cbd28e81d8c084",
    ),
)

GPT_OSS_MOE_SPEC = ModelSupportSpec(
    key="gpt_oss_moe",
    handler_key=_GPT_OSS_MOE_HANDLER_KEY,
    is_moe=True,
    model_names=(
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ),
    default_target_modules=_GPT_OSS_MOE_TARGET_MODULES,
    native_vllm_lora_status=_WIP_NATIVE_VLLM_LORA_STATUS,
    dependency_floor=DependencyFloor(
        transformers="5.6.2",
        megatron_bridge="e1a207ac757e5d0ed94d8ffbe1cbd28e81d8c084",
    ),
)

VALIDATED_MODEL_SUPPORT_SPECS = (
    QWEN3_MOE_SPEC,
    QWEN3_DENSE_SPEC,
    QWEN3_5_MOE_SPEC,
    QWEN3_5_DENSE_SPEC,
    GEMMA4_MOE_SPEC,
    GEMMA4_DENSE_SPEC,
    GPT_OSS_MOE_SPEC,
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
_HANDLER_IMPORTS: dict[str, tuple[str, str]] = {
    _DEFAULT_DENSE_HANDLER_KEY: (
        "art.megatron.model_support.handlers.default_dense",
        "DEFAULT_DENSE_HANDLER",
    ),
    _QWEN3_DENSE_HANDLER_KEY: (
        "art.megatron.model_support.handlers.qwen3_dense",
        "QWEN3_DENSE_HANDLER",
    ),
    _QWEN3_MOE_HANDLER_KEY: (
        "art.megatron.model_support.handlers.qwen3_moe",
        "QWEN3_MOE_HANDLER",
    ),
    _QWEN3_5_DENSE_HANDLER_KEY: (
        "art.megatron.model_support.handlers.qwen3_5",
        "QWEN3_5_DENSE_HANDLER",
    ),
    _QWEN3_5_MOE_HANDLER_KEY: (
        "art.megatron.model_support.handlers.qwen3_5",
        "QWEN3_5_MOE_HANDLER",
    ),
    _GEMMA4_MOE_HANDLER_KEY: (
        "art.megatron.model_support.handlers.gemma4",
        "GEMMA4_MOE_HANDLER",
    ),
    _GEMMA4_DENSE_HANDLER_KEY: (
        "art.megatron.model_support.handlers.gemma4",
        "GEMMA4_DENSE_HANDLER",
    ),
    _GPT_OSS_MOE_HANDLER_KEY: (
        "art.megatron.model_support.handlers.gpt_oss",
        "GPT_OSS_MOE_HANDLER",
    ),
}
_BRIDGE_REGISTRATION_IMPORTS: dict[str, tuple[str, str]] = {
    "qwen3_5_dense": (
        "art.megatron.model_support.handlers.qwen3_5",
        "ensure_qwen35_text_only_bridge_registered",
    ),
    "qwen3_5_moe": (
        "art.megatron.model_support.handlers.qwen3_5",
        "ensure_qwen35_text_only_bridge_registered",
    ),
    "gemma4_moe": (
        "art.megatron.model_support.handlers.gemma4",
        "ensure_gemma4_text_only_bridge_registered",
    ),
    "gemma4_dense": (
        "art.megatron.model_support.handlers.gemma4",
        "ensure_gemma4_text_only_bridge_registered",
    ),
}
_HANDLERS_BY_KEY: dict[str, ModelSupportHandler] = {}
_REGISTERED_BRIDGE_KEYS: set[str] = set()

QWEN3_DENSE_MODELS = frozenset(QWEN3_DENSE_SPEC.model_names)
QWEN3_MOE_MODELS = frozenset(QWEN3_MOE_SPEC.model_names)
QWEN3_5_DENSE_MODELS = frozenset(QWEN3_5_DENSE_SPEC.model_names)
QWEN3_5_MOE_MODELS = frozenset(QWEN3_5_MOE_SPEC.model_names)
QWEN3_5_MODELS = QWEN3_5_DENSE_MODELS | QWEN3_5_MOE_MODELS
GEMMA4_MOE_MODELS = frozenset(GEMMA4_MOE_SPEC.model_names)
GEMMA4_DENSE_MODELS = frozenset(GEMMA4_DENSE_SPEC.model_names)
GPT_OSS_MOE_MODELS = frozenset(GPT_OSS_MOE_SPEC.model_names)


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
    if handler := _HANDLERS_BY_KEY.get(spec.handler_key):
        return handler
    try:
        module_name, attribute_name = _HANDLER_IMPORTS[spec.handler_key]
    except KeyError as exc:
        raise KeyError(
            f"No model support handler registered for {spec.handler_key}"
        ) from exc
    handler = getattr(import_module(module_name), attribute_name)
    if handler.key != spec.handler_key:
        raise RuntimeError(
            f"Model support handler {module_name}.{attribute_name} has key "
            f"{handler.key!r}; expected {spec.handler_key!r}."
        )
    _HANDLERS_BY_KEY[spec.handler_key] = handler
    return handler


def ensure_model_support_bridge_registered_for_spec(
    spec: ModelSupportSpec,
) -> None:
    if spec.key in _REGISTERED_BRIDGE_KEYS:
        return
    bridge_registration = _BRIDGE_REGISTRATION_IMPORTS.get(spec.key)
    if bridge_registration is not None:
        module_name, attribute_name = bridge_registration
        ensure_registered = getattr(import_module(module_name), attribute_name)
        ensure_registered()
    _REGISTERED_BRIDGE_KEYS.add(spec.key)


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
    return get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    ).is_moe


def is_model_support_registered(base_model: str) -> bool:
    return base_model in _SPECS_BY_MODEL


def list_model_support_specs() -> list[ModelSupportSpec]:
    return list(VALIDATED_MODEL_SUPPORT_SPECS)
