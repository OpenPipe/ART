from collections.abc import Mapping
import copy
import inspect
import logging
import os
from typing import Any, Literal, cast

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.flex_dispatcher_backend import (
    apply_flex_dispatcher_backend,
)
from megatron.core.transformer.enums import AttnBackend
from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.model_support.registry import (
    ensure_model_support_bridge_registered_for_spec,
    get_model_support_handler_for_spec,
    get_model_support_spec,
)
from art.megatron.model_support.spec import ModelSupportSpec
from art.megatron.runtime.bridge_runtime import install_art_bridge_runtime_patches

install_art_bridge_runtime_patches()


_NONE_ENV_VALUES = {"", "none", "null", "off", "disable", "disabled"}
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}
_DEEPEP_ROUTER_PROB_WARNING = (
    "DeepEP only supports float32 probs, please set --moe-router-dtype=fp32"
)
_DEEPEP_TOKEN_DISPATCHER_LOGGER = "megatron.core.transformer.moe.token_dispatcher"
_RECOMPUTE_GRANULARITIES = {"full", "selective"}
_RECOMPUTE_METHODS = {"uniform", "block"}
_FLEX_DISPATCHER_BACKENDS = {"deepep", "hybridep"}
_MOE_ROUTER_DTYPES = {"fp32", "fp64", "none"}
_BOOL_ENV_FIELDS = (
    (
        "overlap_moe_expert_parallel_comm",
        "ART_MEGATRON_OVERLAP_MOE_EXPERT_PARALLEL_COMM",
    ),
    ("delay_wgrad_compute", "ART_MEGATRON_DELAY_WGRAD_COMPUTE"),
    (
        "ep_overlap_early_attn_memory_release",
        "ART_MEGATRON_EP_OVERLAP_EARLY_ATTN_MEMORY_RELEASE",
    ),
    ("moe_apply_probs_on_input", "ART_MEGATRON_MOE_APPLY_PROBS_ON_INPUT"),
    ("bias_activation_fusion", "ART_MEGATRON_BIAS_ACTIVATION_FUSION"),
    (
        "fine_grained_activation_offloading",
        "ART_MEGATRON_FINE_GRAINED_ACTIVATION_OFFLOADING",
    ),
    ("moe_shared_expert_overlap", "ART_MEGATRON_MOE_SHARED_EXPERT_OVERLAP"),
)
_INT_ENV_FIELDS = (
    ("tensor_model_parallel_size", "ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE"),
    ("context_parallel_size", "ART_MEGATRON_CONTEXT_PARALLEL_SIZE"),
    ("pipeline_model_parallel_size", "ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE"),
    (
        "virtual_pipeline_model_parallel_size",
        "ART_MEGATRON_VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE",
    ),
    ("expert_model_parallel_size", "ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE"),
    ("recompute_num_layers", "ART_MEGATRON_RECOMPUTE_NUM_LAYERS"),
)
_STR_LIST_ENV_FIELDS = (
    ("offload_modules", "ART_MEGATRON_OFFLOAD_MODULES"),
    ("recompute_modules", "ART_MEGATRON_RECOMPUTE_MODULES"),
)
_CHOICE_ENV_FIELDS = (
    (
        "recompute_granularity",
        "ART_MEGATRON_RECOMPUTE_GRANULARITY",
        _RECOMPUTE_GRANULARITIES,
    ),
    ("recompute_method", "ART_MEGATRON_RECOMPUTE_METHOD", _RECOMPUTE_METHODS),
    (
        "moe_flex_dispatcher_backend",
        "ART_MEGATRON_MOE_FLEX_DISPATCHER_BACKEND",
        _FLEX_DISPATCHER_BACKENDS,
    ),
    ("moe_router_dtype", "ART_MEGATRON_MOE_ROUTER_DTYPE", _MOE_ROUTER_DTYPES),
)


class ProviderBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Any
    bridge: Any
    handler: Any
    spec: ModelSupportSpec


class _DeepEpRouterProbWarningFilter(logging.Filter):
    _art_deepep_router_prob_warning_filter = True

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != _DEEPEP_ROUTER_PROB_WARNING


def _install_deepep_router_prob_warning_filter() -> None:
    logger = logging.getLogger(_DEEPEP_TOKEN_DISPATCHER_LOGGER)
    for existing in logger.filters:
        if getattr(existing, "_art_deepep_router_prob_warning_filter", False):
            return
    logger.addFilter(_DeepEpRouterProbWarningFilter())


def resolve_layer_spec(
    base_layer_spec: Any,
    config: Any,
    vp_stage: int | None = None,
) -> Any:
    module_spec_type = _optional_module_spec_type()
    if module_spec_type is not None and isinstance(base_layer_spec, module_spec_type):
        return copy.deepcopy(base_layer_spec)
    kwargs = (
        {"vp_stage": vp_stage}
        if vp_stage in inspect.signature(base_layer_spec).parameters
        else {}
    )
    return base_layer_spec(config, **kwargs)


def patch_core_attention(layer_spec: object, core_attention: object) -> None:
    submodules = getattr(layer_spec, "submodules", None)
    self_attention = getattr(submodules, "self_attention", None)
    attention_submodules = getattr(self_attention, "submodules", None)
    if attention_submodules is None or not hasattr(
        attention_submodules,
        "core_attention",
    ):
        return
    attention_submodules.core_attention = core_attention


def patch_layer_spec_tree(layer_spec: object, core_attention: object) -> None:
    layer_specs = getattr(layer_spec, "layer_specs", None)
    if layer_specs is None:
        patch_core_attention(layer_spec, core_attention)
        return
    for block_layer_spec in layer_specs:
        patch_core_attention(block_layer_spec, core_attention)


def art_context_parallel_size(config: object) -> int:
    configured = int(getattr(config, "context_parallel_size", 1) or 1)
    return max(configured, _runtime_context_parallel_size())


def patch_art_flex_attention(layer_spec: object, config: object) -> None:
    patch_layer_spec_tree(layer_spec, _art_flex_core_attention(config))


def _art_flex_core_attention(config: object) -> object:
    if art_context_parallel_size(config) > 1:
        from art.megatron.context_parallel.core_attention import (
            ArtContextParallelCoreAttention,
        )

        base_core_attention = ArtContextParallelCoreAttention
    else:
        from art.megatron.flex_attn.attention import FlexDotProductAttention

        base_core_attention = FlexDotProductAttention
    wrapper = getattr(config, "art_flex_core_attention_wrapper", None)
    if wrapper is None:
        return base_core_attention
    return wrapper(config, base_core_attention)


def _runtime_context_parallel_size() -> int:
    try:
        from megatron.core import parallel_state

        if not parallel_state.model_parallel_is_initialized():
            return 1
        return int(parallel_state.get_context_parallel_world_size())
    except (AssertionError, ImportError, RuntimeError, ValueError):
        return 1


def _optional_module_spec_type() -> type[Any] | None:
    try:
        from megatron.core.transformer.spec_utils import ModuleSpec
    except ImportError:
        return None
    return ModuleSpec


class _ProviderRuntimeEnv(BaseModel):
    model_config = ConfigDict(frozen=True)

    overlap_moe_expert_parallel_comm: bool | None = None
    delay_wgrad_compute: bool | None = None
    ep_overlap_early_attn_memory_release: bool | None = None
    moe_deepep_num_sms: int | None = None
    moe_apply_probs_on_input: bool | None = None
    bias_activation_fusion: bool | None = None
    fine_grained_activation_offloading: bool | None = None
    offload_modules: list[str] | None = None
    tensor_model_parallel_size: int | None = None
    context_parallel_size: int | None = None
    pipeline_model_parallel_size: int | None = None
    virtual_pipeline_model_parallel_size: int | None = None
    expert_model_parallel_size: int | None = None
    expert_tensor_parallel_size: int | None = None
    recompute_granularity: Literal["full", "selective"] | None = None
    recompute_method: Literal["uniform", "block"] | None = None
    recompute_num_layers: int | None = None
    recompute_modules: list[str] | None = None
    moe_shared_expert_overlap: bool | None = None
    moe_flex_dispatcher_backend: Literal["deepep", "hybridep"] | None = None
    moe_router_dtype: Literal["fp32", "fp64"] | None = None

    @classmethod
    def from_environ(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "_ProviderRuntimeEnv":
        env = os.environ if env is None else env
        values: dict[str, Any] = {}
        for field_name, env_name in _BOOL_ENV_FIELDS:
            _set_if_found(values, field_name, _env_bool(env, env_name))
        for field_name, env_name in _INT_ENV_FIELDS:
            _set_if_found(values, field_name, _env_optional_int(env, env_name))
        for field_name, env_name in _STR_LIST_ENV_FIELDS:
            _set_if_found(values, field_name, _env_optional_str_list(env, env_name))
        for field_name, env_name, choices in _CHOICE_ENV_FIELDS:
            _set_if_found(
                values,
                field_name,
                _env_optional_choice(env, env_name, choices),
            )
        _set_if_found(
            values,
            "moe_deepep_num_sms",
            _env_default_or_even_positive_int(
                env,
                "ART_MEGATRON_MOE_DEEPEP_NUM_SMS",
            ),
        )
        _set_if_found(
            values, "expert_tensor_parallel_size", _env_expert_tensor_parallel_size(env)
        )
        return cls(**values)

    def is_set(self, field_name: str) -> bool:
        return field_name in self.model_fields_set


def _set_if_found(
    values: dict[str, Any],
    field_name: str,
    parsed: tuple[bool, Any],
) -> None:
    found, value = parsed
    if found:
        values[field_name] = value


def _env_bool(env: Mapping[str, str], name: str) -> tuple[bool, bool | None]:
    raw = env.get(name)
    if raw is None:
        return False, None
    value = raw.strip().lower()
    if value in _TRUE_ENV_VALUES:
        return True, True
    if value in _FALSE_ENV_VALUES:
        return True, False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}")


def _env_optional_str(
    env: Mapping[str, str],
    name: str,
) -> tuple[bool, str | None]:
    raw = env.get(name)
    if raw is None:
        return False, None
    value = raw.strip()
    if value.lower() in _NONE_ENV_VALUES:
        return True, None
    return True, value


def _env_optional_int(
    env: Mapping[str, str],
    name: str,
) -> tuple[bool, int | None]:
    found, value = _env_optional_str(env, name)
    if not found or value is None:
        return found, None
    return True, int(value)


def _env_default_or_even_positive_int(
    env: Mapping[str, str],
    name: str,
) -> tuple[bool, int | None]:
    raw = env.get(name)
    if raw is None:
        return False, None
    value = raw.strip().lower()
    if value == "default":
        return True, None
    try:
        parsed = int(raw.strip())
    except ValueError as exc:
        raise ValueError(
            f"{name} must be 'default' or a positive, even integer, got {raw!r}"
        ) from exc
    if parsed <= 0 or parsed % 2 != 0:
        raise ValueError(
            f"{name} must be 'default' or a positive, even integer, got {raw!r}"
        )
    return True, parsed


def _env_optional_str_list(
    env: Mapping[str, str],
    name: str,
) -> tuple[bool, list[str] | None]:
    found, value = _env_optional_str(env, name)
    if not found or value is None:
        return found, None
    parts = [part.strip() for part in value.split(",")]
    return True, [part for part in parts if part]


def _env_optional_choice(
    env: Mapping[str, str],
    name: str,
    choices: set[str],
) -> tuple[bool, str | None]:
    found, value = _env_optional_str(env, name)
    if not found or value is None:
        return found, None
    if value not in choices:
        expected = ", ".join(repr(choice) for choice in sorted(choices))
        raise ValueError(f"{name} must be one of {expected}, got {value!r}")
    return True, value


def _env_expert_tensor_parallel_size(
    env: Mapping[str, str],
) -> tuple[bool, int | None]:
    found, value = _env_optional_int(env, "ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE")
    if found:
        return found, value
    return _env_optional_int(env, "ART_MEGATRON_EXPERT_TENSOR_MODEL_PARALLEL_SIZE")


def _resolve_default_deepep_num_sms(provider: GPTModelProvider) -> int:
    if provider.overlap_moe_expert_parallel_comm:
        return 20
    if not torch.cuda.is_available():
        return 20
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    sm_count -= sm_count % 2
    return sm_count if sm_count >= 2 else 20


def _handler_cp_supported(handler: Any) -> bool:
    return bool(getattr(handler, "cp_supported", True))


def _apply_default_parallel_topology(
    provider: GPTModelProvider,
    handler: Any,
) -> None:
    visible_gpu_count = max(torch.cuda.device_count(), 1)
    cp_supported = _handler_cp_supported(handler)
    provider.tensor_model_parallel_size = 1 if cp_supported else visible_gpu_count
    provider.context_parallel_size = visible_gpu_count if cp_supported else 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = (
        visible_gpu_count
        if int(getattr(provider, "num_moe_experts", 0) or 0) > 0
        else 1
    )
    provider.expert_tensor_parallel_size = 1


def _apply_art_training_runtime_prepare_defaults(
    provider: GPTModelProvider,
    handler: Any,
) -> None:
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    provider.moe_shared_expert_overlap = True
    _apply_default_parallel_topology(provider, handler)


def _validate_context_parallel_support(
    handler: Any,
    runtime_env: _ProviderRuntimeEnv,
) -> None:
    if _handler_cp_supported(handler):
        return
    if (
        runtime_env.is_set("context_parallel_size")
        and runtime_env.context_parallel_size is not None
        and runtime_env.context_parallel_size > 1
    ):
        raise RuntimeError(
            f"{handler.key} model support does not implement context parallelism; "
            "set ART_MEGATRON_CONTEXT_PARALLEL_SIZE=1."
        )


def _apply_art_training_runtime_finalize_defaults(
    provider: GPTModelProvider,
    runtime_env: _ProviderRuntimeEnv | None = None,
) -> None:
    if int(provider.expert_model_parallel_size or 1) <= 1:
        return
    runtime_env = (
        _ProviderRuntimeEnv.from_environ() if runtime_env is None else runtime_env
    )
    backend = (
        runtime_env.moe_flex_dispatcher_backend
        if runtime_env.is_set("moe_flex_dispatcher_backend")
        else "deepep"
    )
    if backend is None:
        return
    # Expert communication is comparable to expert MLP compute, so the ART
    # runtime uses Megatron's optimized flex dispatcher instead of all-to-all.
    apply_flex_dispatcher_backend(provider, moe_flex_dispatcher_backend=backend)


def _normalize_recompute_settings(provider: GPTModelProvider) -> None:
    if provider.recompute_granularity is None:
        provider.recompute_method = None
        provider.recompute_num_layers = None
        provider.recompute_modules = []


def _apply_runtime_env_overrides(
    provider: GPTModelProvider,
    runtime_env: _ProviderRuntimeEnv | None = None,
) -> None:
    runtime_env = (
        _ProviderRuntimeEnv.from_environ() if runtime_env is None else runtime_env
    )
    _apply_provider_attr_if_value(
        provider,
        runtime_env,
        "overlap_moe_expert_parallel_comm",
    )
    if runtime_env.delay_wgrad_compute is not None:
        provider.delay_wgrad_compute = runtime_env.delay_wgrad_compute
        if runtime_env.delay_wgrad_compute:
            provider.overlap_moe_expert_parallel_comm = True
    _apply_provider_attr_if_value(
        provider,
        runtime_env,
        "ep_overlap_early_attn_memory_release",
    )

    if runtime_env.is_set("moe_deepep_num_sms"):
        provider.moe_deepep_num_sms = (
            _resolve_default_deepep_num_sms(provider)
            if runtime_env.moe_deepep_num_sms is None
            else runtime_env.moe_deepep_num_sms
        )
    else:
        provider.moe_deepep_num_sms = _resolve_default_deepep_num_sms(provider)

    _apply_provider_attr_if_value(provider, runtime_env, "moe_apply_probs_on_input")
    _apply_provider_attr_if_value(provider, runtime_env, "bias_activation_fusion")
    _apply_provider_attr_if_value(
        provider,
        runtime_env,
        "fine_grained_activation_offloading",
    )
    if runtime_env.is_set("offload_modules"):
        provider.offload_modules = (
            [] if runtime_env.offload_modules is None else runtime_env.offload_modules
        )

    _apply_provider_attr_if_value(provider, runtime_env, "tensor_model_parallel_size")
    _apply_provider_attr_if_value(provider, runtime_env, "context_parallel_size")
    _apply_provider_attr_if_value(provider, runtime_env, "pipeline_model_parallel_size")
    _apply_provider_attr_if_set(
        provider,
        runtime_env,
        "virtual_pipeline_model_parallel_size",
    )
    _apply_provider_attr_if_value(provider, runtime_env, "expert_model_parallel_size")
    _apply_provider_attr_if_value(provider, runtime_env, "expert_tensor_parallel_size")
    _apply_provider_attr_if_set(provider, runtime_env, "recompute_granularity")
    _apply_provider_attr_if_set(provider, runtime_env, "recompute_method")
    _apply_provider_attr_if_set(provider, runtime_env, "recompute_num_layers")
    _apply_provider_attr_if_set(provider, runtime_env, "recompute_modules")
    _apply_provider_attr_if_value(provider, runtime_env, "moe_shared_expert_overlap")
    _apply_provider_attr_if_set(provider, runtime_env, "moe_router_dtype")
    _enforce_ep_overlap_recompute_contract(provider)
    _normalize_recompute_settings(provider)


def _apply_provider_attr_if_value(
    provider: GPTModelProvider,
    runtime_env: _ProviderRuntimeEnv,
    field_name: str,
) -> None:
    value = getattr(runtime_env, field_name)
    if value is not None:
        setattr(provider, field_name, value)


def _apply_provider_attr_if_set(
    provider: GPTModelProvider,
    runtime_env: _ProviderRuntimeEnv,
    field_name: str,
) -> None:
    if runtime_env.is_set(field_name):
        setattr(provider, field_name, getattr(runtime_env, field_name))


def _enforce_ep_overlap_recompute_contract(provider: GPTModelProvider) -> None:
    if not provider.overlap_moe_expert_parallel_comm:
        return
    # EP overlap is incompatible with full recompute in Megatron, so treat
    # overlap as the authoritative request even if a launcher exported the
    # usual recompute defaults. Selective recompute is still allowed.
    provider.moe_shared_expert_overlap = False
    provider.recompute_method = None
    provider.recompute_num_layers = None
    if provider.recompute_granularity != "selective":
        provider.recompute_granularity = None


def _install_art_training_flex_attention(provider: GPTModelProvider) -> None:
    _register_art_flex_attention_mapping_types()
    base_layer_spec = provider.transformer_layer_spec

    def _flex_attention_layer_spec(
        config: GPTModelProvider, vp_stage: int | None = None
    ) -> object:
        layer_spec = resolve_layer_spec(base_layer_spec, config, vp_stage)
        patch_art_flex_attention(layer_spec, config)
        return layer_spec

    provider.transformer_layer_spec = cast(Any, _flex_attention_layer_spec)


def _register_art_flex_attention_mapping_types() -> None:
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    AutoMapping.register_module_type("FlexDotProductAttention", "column")
    AutoMapping.register_module_type("ArtContextParallelCoreAttention", "column")


def _build_provider_bundle(
    model: str,
    *,
    torch_dtype: torch.dtype,
    allow_unvalidated_arch: bool = False,
) -> ProviderBundle:
    spec = get_model_support_spec(
        model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    ensure_model_support_bridge_registered_for_spec(spec)
    handler = get_model_support_handler_for_spec(spec)
    bridge = AutoBridge.from_hf_pretrained(
        model,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    provider = bridge.to_megatron_provider()
    handler.patch_bridge(bridge)
    return ProviderBundle(
        provider=provider,
        bridge=bridge,
        handler=handler,
        spec=spec,
    )


def prepare_provider_bundle(
    model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    allow_unvalidated_arch: bool = False,
) -> ProviderBundle:
    _install_deepep_router_prob_warning_filter()
    runtime_env = _ProviderRuntimeEnv.from_environ()
    bundle = _build_provider_bundle(
        model,
        torch_dtype=torch_dtype,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    provider = bundle.provider
    setattr(provider, "_art_model_support_handler", bundle.handler)
    setattr(provider, "_art_model_support_spec", bundle.spec)
    provider.attention_backend = AttnBackend.auto
    provider.moe_permute_fusion = True
    provider.moe_router_dtype = "fp32"
    # params are disabled anyways, but should know about this if we switch to full FT
    # because DP 'dummy' microbatches will unintentionally have loss for this
    provider.moe_aux_loss_coeff = 0.0
    # effectively just a flag modifying finalize_model_grads behavior for DPxCP
    provider.calculate_per_token_loss = True
    provider.cross_entropy_loss_fusion = True
    provider.cross_entropy_fusion_impl = "te"
    _apply_art_training_runtime_prepare_defaults(provider, bundle.handler)
    bundle.handler.configure_provider_for_runtime(provider)
    _validate_context_parallel_support(bundle.handler, runtime_env)
    _apply_runtime_env_overrides(provider, runtime_env)
    provider.art_flex_compile_crash_config = (
        bundle.handler.flex_attention_compile_crash_config(provider)
    )
    provider.sequence_parallel = provider.tensor_model_parallel_size > 1
    _install_art_training_flex_attention(provider)
    bundle.handler.patch_provider(provider, bundle.bridge)
    return bundle


def finalize_provider_bundle(provider_bundle: ProviderBundle) -> ProviderBundle:
    runtime_env = _ProviderRuntimeEnv.from_environ()
    provider = cast(GPTModelProvider, provider_bundle.provider)
    _apply_art_training_runtime_finalize_defaults(provider, runtime_env)
    _finalize_provider_with_art_overrides(provider)
    _normalize_recompute_settings(provider)
    return provider_bundle


def _finalize_provider_with_art_overrides(provider: GPTModelProvider) -> None:
    if not _is_art_gdn_context_parallel_provider(provider):
        provider.finalize()
        return
    _validate_art_gdn_context_parallel_provider(provider)
    variant = provider.experimental_attention_variant
    provider.experimental_attention_variant = None
    try:
        provider.finalize()
    finally:
        provider.experimental_attention_variant = variant


def _is_art_gdn_context_parallel_provider(provider: GPTModelProvider) -> bool:
    return (
        getattr(provider, "experimental_attention_variant", None) == "gated_delta_net"
        and int(getattr(provider, "context_parallel_size", 1) or 1) > 1
    )


def _validate_art_gdn_context_parallel_provider(provider: GPTModelProvider) -> None:
    required = (
        "linear_attention_freq",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
    )
    missing = [name for name in required if getattr(provider, name, None) is None]
    if missing:
        raise ValueError(
            "GatedDeltaNet context parallel provider is missing required fields: "
            + ", ".join(missing)
        )
    raw_linear_num_key_heads = provider.linear_num_key_heads
    raw_linear_num_value_heads = provider.linear_num_value_heads
    assert raw_linear_num_key_heads is not None
    assert raw_linear_num_value_heads is not None
    linear_num_key_heads = int(raw_linear_num_key_heads)
    linear_num_value_heads = int(raw_linear_num_value_heads)
    tensor_model_parallel_size = int(provider.tensor_model_parallel_size)
    if linear_num_value_heads % linear_num_key_heads != 0:
        raise ValueError(
            "linear_num_value_heads must be a multiple of linear_num_key_heads."
        )
    if linear_num_key_heads % tensor_model_parallel_size != 0:
        raise ValueError(
            "linear_num_key_heads must be a multiple of tensor_model_parallel_size."
        )
    if linear_num_value_heads % tensor_model_parallel_size != 0:
        raise ValueError(
            "linear_num_value_heads must be a multiple of tensor_model_parallel_size."
        )


def get_provider_bundle(
    model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    allow_unvalidated_arch: bool = False,
) -> ProviderBundle:
    return finalize_provider_bundle(
        prepare_provider_bundle(
            model,
            torch_dtype=torch_dtype,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
    )


def get_provider(
    model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    allow_unvalidated_arch: bool = False,
) -> GPTModelProvider:
    return get_provider_bundle(
        model,
        torch_dtype=torch_dtype,
        allow_unvalidated_arch=allow_unvalidated_arch,
    ).provider
