import copy
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Literal, cast
import warnings

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.state import (
    SafeTensorsStateSource,
    StateDict,
    StateSource,
)
from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import (
    Qwen3VLSelfAttention,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    Qwen35VLMoEModelProvider,
    _patch_standard_attention_specs,
)
from megatron.bridge.training.flex_dispatcher_backend import (
    apply_flex_dispatcher_backend,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
import torch

from art.megatron.flex_attention import FlexDotProductAttention

_finalized_env_settings_printed = False


@dataclass(frozen=True)
class ProviderBundle:
    provider: GPTModelProvider
    bridge: Any


def _resolve_layer_spec(
    base_layer_spec: ModuleSpec | Callable[[GPTModelProvider], ModuleSpec],
    config: GPTModelProvider,
    vp_stage: int | None = None,
) -> ModuleSpec:
    if isinstance(base_layer_spec, ModuleSpec):
        return copy.deepcopy(base_layer_spec)
    kwargs = (
        {"vp_stage": vp_stage}
        if vp_stage in inspect.signature(base_layer_spec).parameters
        else {}
    )
    return base_layer_spec(config, **kwargs)


def _patch_core_attention(layer_spec: object) -> None:
    submodules = getattr(layer_spec, "submodules", None)
    self_attention = getattr(submodules, "self_attention", None)
    attention_submodules = getattr(self_attention, "submodules", None)
    if attention_submodules is None or not hasattr(
        attention_submodules, "core_attention"
    ):
        return
    attention_submodules.core_attention = FlexDotProductAttention


class _CastingStateSource(StateSource):
    def __init__(self, source: StateSource, *, dtype: torch.dtype):
        self._source = source
        self._dtype = dtype

    def get_all_keys(self) -> list[str]:
        return self._source.get_all_keys()

    def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        loaded = self._source.load_tensors(keys)
        return {
            key: (
                value.to(dtype=self._dtype)
                if torch.is_floating_point(value) and value.dtype != self._dtype
                else value
            )
            for key, value in loaded.items()
        }

    def has_glob(self, pattern: str) -> bool:
        return self._source.has_glob(pattern)


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


def _env_optional_str(name: str) -> tuple[bool, str | None]:
    raw = os.environ.get(name)
    if raw is None:
        return False, None
    value = raw.strip()
    if not value or value.lower() in {"none", "null", "off", "disable", "disabled"}:
        return True, None
    return True, value


def _env_optional_int(name: str) -> tuple[bool, int | None]:
    found, value = _env_optional_str(name)
    if not found or value is None:
        return found, None
    return True, int(value)


def _env_default_or_even_positive_int(name: str) -> tuple[bool, int | None]:
    raw = os.environ.get(name)
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


def _env_optional_str_list(name: str) -> tuple[bool, list[str] | None]:
    found, value = _env_optional_str(name)
    if not found or value is None:
        return found, None
    parts = [part.strip() for part in value.split(",")]
    return True, [part for part in parts if part]


def _env_optional_recompute_granularity(
    name: str,
) -> tuple[bool, Literal["full", "selective"] | None]:
    found, value = _env_optional_str(name)
    if not found or value is None:
        return found, None
    if value not in {"full", "selective"}:
        raise ValueError(f"{name} must be one of 'full' or 'selective', got {value!r}")
    return True, cast(Literal["full", "selective"], value)


def _env_optional_recompute_method(
    name: str,
) -> tuple[bool, Literal["uniform", "block"] | None]:
    found, value = _env_optional_str(name)
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
    provider.expert_model_parallel_size = visible_gpu_count
    provider.expert_tensor_parallel_size = 1


def _tp_ep_parallel_domain_size(provider: GPTModelProvider) -> int:
    return int(provider.tensor_model_parallel_size) * int(
        provider.expert_model_parallel_size
    )


def _maybe_print_finalized_env_settings(provider: GPTModelProvider) -> None:
    global _finalized_env_settings_printed
    if _finalized_env_settings_printed:
        return
    if torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        if torch.distributed.get_rank() != 0:  # ty: ignore[possibly-missing-attribute]
            _finalized_env_settings_printed = True
            return
    _finalized_env_settings_printed = True
    print(
        "Finalized Megatron env settings:",
        json.dumps(
            {
                "tensor_model_parallel_size": provider.tensor_model_parallel_size,
                "expert_model_parallel_size": provider.expert_model_parallel_size,
                "overlap_moe_expert_parallel_comm": provider.overlap_moe_expert_parallel_comm,
                "delay_wgrad_compute": provider.delay_wgrad_compute,
                "ep_overlap_early_attn_memory_release": provider.ep_overlap_early_attn_memory_release,
                "moe_deepep_num_sms": provider.moe_deepep_num_sms,
                "moe_apply_probs_on_input": provider.moe_apply_probs_on_input,
                "bias_activation_fusion": provider.bias_activation_fusion,
                "fine_grained_activation_offloading": provider.fine_grained_activation_offloading,
                "offload_modules": provider.offload_modules,
                "recompute_granularity": provider.recompute_granularity,
                "recompute_method": provider.recompute_method,
                "recompute_num_layers": provider.recompute_num_layers,
                "recompute_modules": provider.recompute_modules,
                "moe_shared_expert_overlap": provider.moe_shared_expert_overlap,
                "moe_flex_dispatcher_backend": (
                    "deepep" if _tp_ep_parallel_domain_size(provider) > 1 else None
                ),
                "sequence_parallel": provider.sequence_parallel,
            },
            indent=2,
        ),
    )


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

    found, deepep_num_sms = _env_default_or_even_positive_int(
        "ART_MEGATRON_MOE_DEEPEP_NUM_SMS"
    )
    if found:
        provider.moe_deepep_num_sms = (
            _resolve_default_deepep_num_sms(provider)
            if deepep_num_sms is None
            else deepep_num_sms
        )
    else:
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

    offload_modules_found, offload_modules = _env_optional_str_list(
        "ART_MEGATRON_OFFLOAD_MODULES"
    )
    if offload_modules_found:
        provider.offload_modules = [] if offload_modules is None else offload_modules

    found, tensor_model_parallel_size = _env_optional_int(
        "ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE"
    )
    if found and tensor_model_parallel_size is not None:
        provider.tensor_model_parallel_size = tensor_model_parallel_size

    recompute_granularity_found, recompute_granularity = (
        _env_optional_recompute_granularity("ART_MEGATRON_RECOMPUTE_GRANULARITY")
    )
    if recompute_granularity_found:
        provider.recompute_granularity = recompute_granularity

    recompute_method_found, recompute_method = _env_optional_recompute_method(
        "ART_MEGATRON_RECOMPUTE_METHOD"
    )
    if recompute_method_found:
        provider.recompute_method = recompute_method

    recompute_num_layers_found, recompute_num_layers = _env_optional_int(
        "ART_MEGATRON_RECOMPUTE_NUM_LAYERS"
    )
    if recompute_num_layers_found:
        provider.recompute_num_layers = recompute_num_layers

    recompute_modules_found, recompute_modules = _env_optional_str_list(
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
        if shared_expert_overlap:
            warnings.warn(
                "ART_MEGATRON_MOE_SHARED_EXPERT_OVERLAP=true is incompatible with "
                "ART_MEGATRON_OVERLAP_MOE_EXPERT_PARALLEL_COMM; forcing "
                "moe_shared_expert_overlap=False",
                stacklevel=2,
            )
        provider.moe_shared_expert_overlap = False
        provider.recompute_method = None
        provider.recompute_num_layers = None
        if provider.recompute_granularity != "selective":
            provider.recompute_granularity = None


def get_provider_bundle(
    model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> ProviderBundle:
    bridge = AutoBridge.from_hf_pretrained(
        model,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    assert isinstance(bridge._model_bridge, (Qwen3MoEBridge, Qwen35VLMoEBridge)), (
        "Only Qwen3 and Qwen3.5 MoE models are supported"
    )
    if torch_dtype != torch.bfloat16:
        model_name_or_path = bridge.hf_pretrained.model_name_or_path
        assert model_name_or_path is not None
        bridge.hf_pretrained._state_dict_accessor = StateDict(
            _CastingStateSource(
                SafeTensorsStateSource(cast(str | Path, model_name_or_path)),
                dtype=torch_dtype,
            )
        )
    provider = bridge.to_megatron_provider()
    if isinstance(provider, Qwen35VLMoEModelProvider):
        from megatron.bridge.models.gpt_provider import mtp_block_spec

        def _patch_qwen35_block_spec(block_spec: TransformerBlockSubmodules) -> None:
            _patch_standard_attention_specs(block_spec, Qwen3VLSelfAttention)
            layer_specs = block_spec.layer_specs
            if layer_specs is None:
                return
            for layer_spec in layer_specs:
                _patch_core_attention(layer_spec)

        def _qwen35_layer_spec(
            config: GPTModelProvider, vp_stage: int | None = None
        ) -> ModuleSpec:
            block_spec = get_transformer_block_with_experimental_attention_variant_spec(
                config,
                vp_stage=vp_stage,
            )
            _patch_qwen35_block_spec(block_spec)
            return cast(ModuleSpec, block_spec)

        provider.transformer_layer_spec = _qwen35_layer_spec

        def _provide_qwen35_with_flex_attention(
            self: Qwen35VLMoEModelProvider,
            pre_process: bool | None = None,
            post_process: bool | None = None,
            vp_stage: int | None = None,
        ) -> Qwen3VLModel:
            language_transformer_config = cast(Any, self)
            hf_vision_config = self.vision_config
            hf_vision_config.torch_dtype = self.params_dtype
            block_spec = cast(
                ModuleSpec,
                get_transformer_block_with_experimental_attention_variant_spec(
                    language_transformer_config,
                    vp_stage=vp_stage,
                ),
            )
            _patch_qwen35_block_spec(cast(TransformerBlockSubmodules, block_spec))
            pre_process_value = True if pre_process is None else pre_process
            post_process_value = True if post_process is None else post_process
            model = Qwen3VLModel(
                language_transformer_config=language_transformer_config,
                language_transformer_layer_spec=block_spec,
                vision_transformer_config=hf_vision_config,
                pre_process=pre_process_value,
                post_process=post_process_value,
                pg_collection=cast(Any, self._pg_collection),
                mtp_block_spec=mtp_block_spec(self, vp_stage=vp_stage),
                vp_stage=vp_stage,
            )
            if (
                self.freeze_language_model
                or self.freeze_vision_model
                or self.freeze_vision_projection
            ):
                model.freeze(
                    freeze_language_model=self.freeze_language_model,
                    freeze_vision_model=self.freeze_vision_model,
                    freeze_vision_projection=self.freeze_vision_projection,
                )
            return model

        provider.provide = cast(
            Any, MethodType(_provide_qwen35_with_flex_attention, provider)
        )
    base_layer_spec = provider.transformer_layer_spec

    def _flex_attention_layer_spec(
        config: GPTModelProvider, vp_stage: int | None = None
    ) -> ModuleSpec:
        layer_spec = _resolve_layer_spec(base_layer_spec, config, vp_stage)
        layer_specs = getattr(layer_spec, "layer_specs", None)
        if layer_specs is None:
            _patch_core_attention(layer_spec)
        else:
            for block_layer_spec in layer_specs:
                _patch_core_attention(block_layer_spec)
        return layer_spec

    provider.transformer_layer_spec = _flex_attention_layer_spec
    provider.attention_backend = AttnBackend.auto
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
    provider.moe_shared_expert_overlap = True
    _apply_default_parallel_topology(provider)
    _apply_runtime_env_overrides(provider)
    if _tp_ep_parallel_domain_size(provider) > 1:
        # use DeepEP for MoE expert comm. comm can be the same amount of time as actual MLP
        # compute, so these are very beneficial
        apply_flex_dispatcher_backend(provider, moe_flex_dispatcher_backend="deepep")
    provider.moe_permute_fusion = True
    provider.moe_router_dtype = "fp32"
    # params are disabled anyways, but should know about this if we switch to full FT
    # because DP 'dummy' microbatches will unintentionally have loss for this
    provider.moe_aux_loss_coeff = 0.0
    # effectively just a flag modifying finalize_model_grads behavior for DPxCP
    provider.calculate_per_token_loss = True
    provider.sequence_parallel = provider.tensor_model_parallel_size > 1
    # ART computes its own RL loss, so MTP only adds incompatible postprocess work.
    provider.mtp_enabled = False
    provider.mtp_num_layers = 0
    _maybe_print_finalized_env_settings(provider)
    provider.finalize()
    return ProviderBundle(provider=provider, bridge=bridge)


def get_provider(
    model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> GPTModelProvider:
    return get_provider_bundle(model, torch_dtype=torch_dtype).provider
