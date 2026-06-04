import os
from typing import Any, Literal, cast

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.flex_dispatcher_backend import (
    apply_flex_dispatcher_backend,
)
from megatron.core.transformer.enums import AttnBackend
import torch

from art.megatron.flex_attention import FlexDotProductAttention
from art.megatron.model_support.registry import (
    get_model_support_handler_for_spec,
    get_model_support_spec,
)
from art.megatron.provider_common import (
    ProviderBundle,
    patch_layer_spec_tree,
    resolve_layer_spec,
)


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}")


def _env_override_str(name: str) -> tuple[bool, str | None]:
    raw = os.environ.get(name)
    if raw is None:
        return False, None
    value = raw.strip()
    if not value or value.lower() in {"none", "null", "off", "disable", "disabled"}:
        return True, None
    return True, value


def _env_override_int(name: str) -> tuple[bool, int | None]:
    found, value = _env_override_str(name)
    if not found or value is None:
        return found, None
    return True, int(value)


def _env_override_str_list(name: str) -> tuple[bool, list[str] | None]:
    found, value = _env_override_str(name)
    if not found or value is None:
        return found, None
    parts = [part.strip() for part in value.split(",")]
    return True, [part for part in parts if part]


def _env_override_recompute_granularity(
    name: str,
) -> tuple[bool, Literal["full", "selective"] | None]:
    found, value = _env_override_str(name)
    if not found or value is None:
        return found, None
    if value not in {"full", "selective"}:
        raise ValueError(f"{name} must be one of 'full' or 'selective', got {value!r}")
    return True, cast(Literal["full", "selective"], value)


def _env_override_recompute_method(
    name: str,
) -> tuple[bool, Literal["uniform", "block"] | None]:
    found, value = _env_override_str(name)
    if not found or value is None:
        return found, None
    if value not in {"uniform", "block"}:
        raise ValueError(f"{name} must be one of 'uniform' or 'block', got {value!r}")
    return True, cast(Literal["uniform", "block"], value)


def _resolve_default_deepep_num_sms(provider: GPTModelProvider) -> int:
    if provider.overlap_moe_expert_parallel_comm:
        return 20
    if not torch.cuda.is_available():
        return 20
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    sm_count -= sm_count % 2
    return sm_count if sm_count >= 2 else 20


def _apply_default_parallel_topology(provider: GPTModelProvider) -> None:
    visible_gpu_count = max(torch.cuda.device_count(), 1)
    provider.tensor_model_parallel_size = visible_gpu_count
    provider.context_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = (
        visible_gpu_count
        if int(getattr(provider, "num_moe_experts", 0) or 0) > 0
        else 1
    )
    provider.expert_tensor_parallel_size = 1


def _apply_art_training_runtime_prepare_defaults(provider: GPTModelProvider) -> None:
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    provider.moe_shared_expert_overlap = True
    _apply_default_parallel_topology(provider)


def _apply_art_training_runtime_finalize_defaults(provider: GPTModelProvider) -> None:
    if provider.expert_model_parallel_size <= 1:
        return
    # use DeepEP for MoE expert comm. comm can be the same amount of time as actual MLP
    # compute, so these are very beneficial
    apply_flex_dispatcher_backend(provider, moe_flex_dispatcher_backend="deepep")


def _apply_runtime_env_overrides(provider: GPTModelProvider) -> None:
    overlap = _env_flag("ART_MEGATRON_OVERLAP_MOE_EXPERT_PARALLEL_COMM")
    if overlap is not None:
        provider.overlap_moe_expert_parallel_comm = overlap

    delay_wgrad = _env_flag("ART_MEGATRON_DELAY_WGRAD_COMPUTE")
    if delay_wgrad is not None:
        provider.delay_wgrad_compute = delay_wgrad
        if delay_wgrad:
            provider.overlap_moe_expert_parallel_comm = True

    early_attn_release = _env_flag("ART_MEGATRON_EP_OVERLAP_EARLY_ATTN_MEMORY_RELEASE")
    if early_attn_release is not None:
        provider.ep_overlap_early_attn_memory_release = early_attn_release

    found, deepep_num_sms = _env_override_int("ART_MEGATRON_MOE_DEEPEP_NUM_SMS")
    if found and deepep_num_sms is not None:
        provider.moe_deepep_num_sms = deepep_num_sms
    if "ART_MEGATRON_MOE_DEEPEP_NUM_SMS" not in os.environ:
        provider.moe_deepep_num_sms = _resolve_default_deepep_num_sms(provider)

    moe_apply_probs_on_input = _env_flag("ART_MEGATRON_MOE_APPLY_PROBS_ON_INPUT")
    if moe_apply_probs_on_input is not None:
        provider.moe_apply_probs_on_input = moe_apply_probs_on_input

    bias_activation_fusion = _env_flag("ART_MEGATRON_BIAS_ACTIVATION_FUSION")
    if bias_activation_fusion is not None:
        provider.bias_activation_fusion = bias_activation_fusion

    fine_grained_activation_offloading = _env_flag(
        "ART_MEGATRON_FINE_GRAINED_ACTIVATION_OFFLOADING"
    )
    if fine_grained_activation_offloading is not None:
        provider.fine_grained_activation_offloading = fine_grained_activation_offloading

    offload_modules_found, offload_modules = _env_override_str_list(
        "ART_MEGATRON_OFFLOAD_MODULES"
    )
    if offload_modules_found:
        provider.offload_modules = [] if offload_modules is None else offload_modules

    found, tensor_model_parallel_size = _env_override_int(
        "ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE"
    )
    if found and tensor_model_parallel_size is not None:
        provider.tensor_model_parallel_size = tensor_model_parallel_size

    found, expert_model_parallel_size = _env_override_int(
        "ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE"
    )
    if found and expert_model_parallel_size is not None:
        provider.expert_model_parallel_size = expert_model_parallel_size

    found, expert_tensor_parallel_size = _env_override_int(
        "ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE"
    )
    if not found:
        found, expert_tensor_parallel_size = _env_override_int(
            "ART_MEGATRON_EXPERT_TENSOR_MODEL_PARALLEL_SIZE"
        )
    if found and expert_tensor_parallel_size is not None:
        provider.expert_tensor_parallel_size = expert_tensor_parallel_size

    recompute_granularity_found, recompute_granularity = (
        _env_override_recompute_granularity("ART_MEGATRON_RECOMPUTE_GRANULARITY")
    )
    if recompute_granularity_found:
        provider.recompute_granularity = recompute_granularity

    recompute_method_found, recompute_method = _env_override_recompute_method(
        "ART_MEGATRON_RECOMPUTE_METHOD"
    )
    if recompute_method_found:
        provider.recompute_method = recompute_method

    recompute_num_layers_found, recompute_num_layers = _env_override_int(
        "ART_MEGATRON_RECOMPUTE_NUM_LAYERS"
    )
    if recompute_num_layers_found:
        provider.recompute_num_layers = recompute_num_layers

    recompute_modules_found, recompute_modules = _env_override_str_list(
        "ART_MEGATRON_RECOMPUTE_MODULES"
    )
    if recompute_modules_found:
        provider.recompute_modules = recompute_modules

    shared_expert_overlap = _env_flag("ART_MEGATRON_MOE_SHARED_EXPERT_OVERLAP")
    if shared_expert_overlap is not None:
        provider.moe_shared_expert_overlap = shared_expert_overlap

    if provider.overlap_moe_expert_parallel_comm:
        # EP overlap is incompatible with full recompute in Megatron, so treat
        # overlap as the authoritative request even if a launcher exported the
        # usual recompute defaults. Selective recompute is still allowed.
        provider.moe_shared_expert_overlap = False
        provider.recompute_method = None
        provider.recompute_num_layers = None
        if provider.recompute_granularity != "selective":
            provider.recompute_granularity = None


def _install_art_training_flex_attention(provider: GPTModelProvider) -> None:
    base_layer_spec = provider.transformer_layer_spec

    def _flex_attention_layer_spec(
        config: GPTModelProvider, vp_stage: int | None = None
    ) -> object:
        layer_spec = resolve_layer_spec(base_layer_spec, config, vp_stage)
        patch_layer_spec_tree(layer_spec, FlexDotProductAttention)
        return layer_spec

    provider.transformer_layer_spec = cast(Any, _flex_attention_layer_spec)


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
    handler = get_model_support_handler_for_spec(spec)
    bridge = AutoBridge.from_hf_pretrained(
        model,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    handler.patch_bridge(bridge)
    return ProviderBundle(
        provider=bridge.to_megatron_provider(),
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
    _apply_art_training_runtime_prepare_defaults(provider)
    bundle.handler.configure_provider_for_runtime(provider)
    _apply_runtime_env_overrides(provider)
    provider.sequence_parallel = provider.tensor_model_parallel_size > 1
    _install_art_training_flex_attention(provider)
    bundle.handler.patch_provider(provider, bundle.bridge)
    return bundle


def finalize_provider_bundle(provider_bundle: ProviderBundle) -> ProviderBundle:
    provider = provider_bundle.provider
    _apply_art_training_runtime_finalize_defaults(provider)
    provider.finalize()
    return provider_bundle


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
