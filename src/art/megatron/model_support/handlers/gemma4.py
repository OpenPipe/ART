from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from functools import lru_cache
import json
from pathlib import Path
import re
from types import MethodType
from typing import Any, Sequence, cast

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import (
    TERowParallelLinear,
    te_checkpoint,
)
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
import torch

from art.megatron import lora as art_lora
from art.megatron.lora import SelfAttentionLinearProjLoRA
from art.megatron.model_support.handlers.default_dense import (
    DefaultDenseHandler,
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
    _require_dense_mlp,
    _require_moe_experts,
)
from art.megatron.model_support.handlers.qwen3_common import (
    _context_parallel_world_size,
)
from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    ExpertPackedLoraGroup,
    ExpertPackedLoraSlot,
    FlexAttentionCompileCrashConfig,
    LayerFamilyInstance,
)

_GEMMA4_MOE_COMPILE_WORKAROUND_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "flex_token_dispatch_combine",
    "te_triton_permute_with_mask_map",
)
_GEMMA4_TRITON_NUM_STAGES_2_SIGNATURES = {
    # google/gemma-4-31B-it: Triton flex attention raises "No valid triton
    # configs" for global attention head_dim=512 with backend-only options.
    ("dense", 60, 5376, 32, 256, 512, 4),
    # google/gemma-4-26B-A4B-it hits the same Triton resource limit on global
    # attention head_dim=512 with backend-only options.
    ("moe", 30, 2816, 16, 256, 512, 2),
}
_ART_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<module>gate_up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)
_VLLM_MOE_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.moe\.experts)\."
    r"(?:(?P<base_layer>base_layer)\.)?(?P<lora>lora_[AB])\.weight$"
)
_VLLM_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.moe\.experts)\.(?P<expert>\d+)\."
    r"(?P<module>gate_proj|up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)
_DENSE_MLP_LORA_KEY_RE = re.compile(
    r"(?P<prefix>\.mlp)\.(?P<module>gate_proj|up_proj|down_proj)\."
    r"(?P<lora>lora_[AB])\.weight$"
)
_SHARED_EXPERT_FC1_LORA_A_KEY_RE = re.compile(
    r"^.*\.layers\.(?P<layer>\d+)\.mlp\.(?:shared_expert\.)?"
    r"(?:gate_proj|up_proj)\.lora_A\.weight$"
)
_SELF_ATTN_V_LORA_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.self_attn\.)v_proj\."
    r"(?P<suffix>lora_[AB]\.weight)$"
)
_SELF_ATTN_K_LORA_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.(?P<layer>\d+)\.self_attn\.)k_proj\."
    r"(?P<suffix>lora_[AB]\.weight)$"
)
_MEGATRON_LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.")
_HF_TEXT_EXPERT_KEY_RE = re.compile(r"(?P<layer>\.layers\.\d+)\.experts")


def _gemma4_forward_kwargs(model: Any, **kwargs: Any) -> dict[str, Any]:
    attention_bias = kwargs.get("attention_bias")
    from art.megatron.context_parallel.types import ArtContextParallelState

    module = model
    while hasattr(module, "module"):
        module = module.module
    gpt_module = getattr(module, "language_model", module)
    if isinstance(attention_bias, ArtContextParallelState):
        setattr(
            gpt_module,
            "_art_gemma4_rotary_seq_len",
            int(attention_bias.rank_plan.original_seq_len),
        )
    else:
        setattr(gpt_module, "_art_gemma4_rotary_seq_len", None)
    return {"extra_block_kwargs": kwargs}


class Gemma4MoeHandler(DefaultMoeHandler):
    key = "gemma4_moe"
    is_moe = True
    native_vllm_lora_status = "validated"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return getattr(base_config, "text_config", base_config)

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        return _gemma4_forward_kwargs(model, **kwargs)

    def _identity_lora_parameter_suffixes(
        self,
        target_modules: list[str],
    ) -> tuple[str, ...]:
        suffixes = list(super()._identity_lora_parameter_suffixes(target_modules))
        target_set = set(target_modules)
        if {"experts", "gate_proj", "up_proj"} & target_set:
            suffixes.append("experts.gate_up_proj")
        if {"experts", "down_proj"} & target_set:
            suffixes.append("experts.down_proj")
        return tuple(dict.fromkeys(suffixes))

    def configure_provider_for_runtime(self, provider: Any) -> None:
        _patch_gemma4_router_for_mcore()
        _patch_gemma4_rotary_for_hf_proportional()
        _patch_gemma4_qkv_for_hf_tied_value()
        window_size = int(getattr(provider, "window_size", 1024))
        _install_gemma4_flex_core_attention_wrapper(provider)
        provider.art_flex_sliding_windows = (window_size,)
        provider.art_flex_head_dims_by_window = {
            None: int(getattr(provider, "global_head_dim", provider.kv_channels)),
            window_size: int(provider.kv_channels),
        }
        provider.moe_shared_expert_overlap = False

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        _install_gemma4_preprocess_patch(model_chunks)
        _install_gemma4_full_recompute_patch(model_chunks)

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        if int(getattr(provider, "num_moe_experts", 0) or 0) <= 0:
            raise TypeError("Gemma 4 MoE handler received a dense provider")
        sliding_count, global_count = _gemma4_attention_pattern(provider)
        families = [
            LayerFamilyInstance(key="gemma4_sliding_attention", layer_index=0),
            LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
        ]
        if global_count > 0:
            families.append(
                LayerFamilyInstance(
                    key="gemma4_global_attention",
                    layer_index=sliding_count,
                )
            )
        if int(getattr(provider, "moe_shared_expert_intermediate_size", 0) or 0) > 0:
            families.append(
                LayerFamilyInstance(key="shared_experts_mlp", layer_index=0)
            )
        return families

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None:
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import (
            _adapter_model_prefix,
            _is_language_transformer_layer_name,
            wrap_grouped_moe_experts_3d,
            wrap_shared_experts_mlp,
            wrap_standard_self_attention,
        )

        target_set = set(target_modules)
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                adapter_model_prefix = _adapter_model_prefix(module)
                if not isinstance(module.self_attention, SelfAttention):
                    raise TypeError(
                        "Gemma 4 expected a SelfAttention module, got "
                        f"{type(module.self_attention)}"
                    )
                attention_provider = _attention_provider_for_layer(provider, module)
                qkv_targets = (
                    {"q_proj", "k_proj", "v_proj"}
                    if not target_set
                    else target_set - {"o_proj"}
                )
                if qkv_targets:
                    wrap_standard_self_attention(
                        module.self_attention,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=attention_provider,
                        target_modules=qkv_targets,
                        rank=rank,
                        alpha=alpha,
                    )
                if (
                    not target_set or {"q_proj", "k_proj", "v_proj"} & target_set
                ) and _is_gemma4_global_layer(int(module.layer_number), provider):
                    _tie_global_value_lora_to_key(module.self_attention)
                _wrap_gemma4_attention_output_lora(
                    module.self_attention,
                    adapter_model_prefix=adapter_model_prefix,
                    provider=attention_provider,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )
                wrap_grouped_moe_experts_3d(
                    _require_moe_experts(module),
                    adapter_model_prefix=adapter_model_prefix,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )
                shared_experts = getattr(module.mlp, "shared_experts", None)
                if shared_experts is not None:
                    wrap_shared_experts_mlp(
                        shared_experts,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=provider,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )

    def build_adapter_weights_by_base(
        self,
        model_chunks: Sequence[Any],
    ) -> dict[str, list[Any]]:
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import _is_language_transformer_layer_name
        from art.megatron.weights.adapter_export import (
            add_grouped_moe_adapter_weights,
            add_shared_experts_adapter_weights,
            add_standard_self_attention_adapter_weights,
            layer_base_prefix,
        )

        adapter_weights_by_base: dict[str, list[Any]] = {}
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                layer_prefix = layer_base_prefix(module, module_name=module_name)
                add_standard_self_attention_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    self_attention=module.self_attention,
                )
                add_grouped_moe_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    experts=_require_moe_experts(module),
                )
                shared_experts = getattr(module.mlp, "shared_experts", None)
                if shared_experts is not None:
                    add_shared_experts_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        shared_experts=shared_experts,
                    )
        return adapter_weights_by_base

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return _to_vllm_lora_tensors(tensors, adapter_config=adapter_config)

    def from_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return _from_vllm_lora_tensors(tensors, adapter_config=adapter_config)

    def expert_packed_lora_groups(self) -> tuple[ExpertPackedLoraGroup, ...]:
        return (
            ExpertPackedLoraGroup(
                art_group_suffix=".mlp.experts",
                slots=(
                    ExpertPackedLoraSlot(
                        source_projection="gate_up_proj",
                        source_lora="lora_A",
                        output_suffix="base_layer.lora_A.weight",
                        pack_layout="expert_rows",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="gate_up_proj",
                        source_lora="lora_B",
                        output_suffix="base_layer.lora_B.weight",
                        pack_layout="rank_major_expert_cols",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="down_proj",
                        source_lora="lora_A",
                        output_suffix="lora_A.weight",
                        pack_layout="expert_rows",
                    ),
                    ExpertPackedLoraSlot(
                        source_projection="down_proj",
                        source_lora="lora_B",
                        output_suffix="lora_B.weight",
                        pack_layout="rank_major_expert_cols",
                    ),
                ),
            ),
        )

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        if bool(getattr(provider, "moe_shared_expert_overlap", False)):
            return CompileWorkaroundConfig(
                shared_expert_state="shared_expert_overlap",
                disable_compile=True,
            )
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(
                provider,
                _GEMMA4_MOE_COMPILE_WORKAROUND_FLAGS,
            ),
            shared_expert_state="shared_experts",
            disable_compile=False,
        )

    def flex_attention_compile_crash_config(
        self,
        provider: Any,
    ) -> FlexAttentionCompileCrashConfig:
        if (
            _gemma4_compile_crash_signature(provider)
            in _GEMMA4_TRITON_NUM_STAGES_2_SIGNATURES
        ):
            return FlexAttentionCompileCrashConfig(
                triton_num_stages_2_head_dims=(int(provider.global_head_dim),)
            )
        return FlexAttentionCompileCrashConfig()


GEMMA4_MOE_HANDLER = Gemma4MoeHandler()


class Gemma4DenseHandler(DefaultDenseHandler):
    key = "gemma4_dense"
    native_vllm_lora_status = "validated"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return getattr(base_config, "text_config", base_config)

    def configure_provider_for_runtime(self, provider: Any) -> None:
        _patch_gemma4_rotary_for_hf_proportional()
        _patch_gemma4_qkv_for_hf_tied_value()
        window_size = int(getattr(provider, "window_size", 1024))
        _install_gemma4_flex_core_attention_wrapper(provider)
        provider.art_flex_sliding_windows = (window_size,)
        provider.art_flex_head_dims_by_window = {
            None: int(getattr(provider, "global_head_dim", provider.kv_channels)),
            window_size: int(provider.kv_channels),
        }

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        _install_gemma4_preprocess_patch(model_chunks)
        _install_gemma4_full_recompute_patch(model_chunks)

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        if int(getattr(provider, "num_moe_experts", 0) or 0) > 0:
            raise TypeError("Gemma 4 dense handler received a MoE provider")
        sliding_count, global_count = _gemma4_attention_pattern(provider)
        families = [
            LayerFamilyInstance(key="gemma4_sliding_attention", layer_index=0),
            LayerFamilyInstance(key="dense_mlp", layer_index=0),
        ]
        if global_count > 0:
            families.append(
                LayerFamilyInstance(
                    key="gemma4_global_attention",
                    layer_index=sliding_count,
                )
            )
        return families

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None:
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import (
            _adapter_model_prefix,
            _is_language_transformer_layer_name,
            wrap_dense_mlp,
            wrap_standard_self_attention,
        )

        target_set = set(target_modules)
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                adapter_model_prefix = _adapter_model_prefix(module)
                if not isinstance(module.self_attention, SelfAttention):
                    raise TypeError(
                        "Gemma 4 expected a SelfAttention module, got "
                        f"{type(module.self_attention)}"
                    )
                attention_provider = _attention_provider_for_layer(provider, module)
                qkv_targets = (
                    {"q_proj", "k_proj", "v_proj"}
                    if not target_set
                    else target_set - {"o_proj"}
                )
                if qkv_targets:
                    wrap_standard_self_attention(
                        module.self_attention,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=attention_provider,
                        target_modules=qkv_targets,
                        rank=rank,
                        alpha=alpha,
                    )
                if (
                    not target_set or {"q_proj", "k_proj", "v_proj"} & target_set
                ) and _is_gemma4_global_layer(int(module.layer_number), provider):
                    _tie_global_value_lora_to_key(module.self_attention)
                _wrap_gemma4_attention_output_lora(
                    module.self_attention,
                    adapter_model_prefix=adapter_model_prefix,
                    provider=attention_provider,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )
                _require_dense_mlp(module)
                wrap_dense_mlp(
                    module.mlp,
                    adapter_model_prefix=adapter_model_prefix,
                    provider=provider,
                    target_modules=target_set,
                    rank=rank,
                    alpha=alpha,
                )

    def build_adapter_weights_by_base(
        self,
        model_chunks: Sequence[Any],
    ) -> dict[str, list[Any]]:
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import _is_language_transformer_layer_name
        from art.megatron.weights.adapter_export import (
            add_dense_mlp_adapter_weights,
            add_standard_self_attention_adapter_weights,
            layer_base_prefix,
        )

        adapter_weights_by_base: dict[str, list[Any]] = {}
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                layer_prefix = layer_base_prefix(module, module_name=module_name)
                _require_dense_mlp(module)
                add_standard_self_attention_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    self_attention=module.self_attention,
                )
                add_dense_mlp_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    mlp=module.mlp,
                )
        return adapter_weights_by_base

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        transformed = {_to_vllm_key(key): tensor for key, tensor in tensors.items()}
        if len(transformed) != len(tensors):
            raise RuntimeError("Duplicate Gemma 4 LoRA tensor after vLLM conversion")
        return (
            _add_gemma4_k_eq_v_v_lora_tensors(
                transformed,
                adapter_config=adapter_config,
            ),
            adapter_config,
        )

    def from_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return _drop_gemma4_k_eq_v_v_lora_tensors(
            dict(tensors),
            adapter_config=adapter_config,
        )

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(provider),
            shared_expert_state="none",
            disable_compile=False,
        )

    def flex_attention_compile_crash_config(
        self,
        provider: Any,
    ) -> FlexAttentionCompileCrashConfig:
        if (
            _gemma4_compile_crash_signature(provider)
            in _GEMMA4_TRITON_NUM_STAGES_2_SIGNATURES
        ):
            return FlexAttentionCompileCrashConfig(
                triton_num_stages_2_head_dims=(int(provider.global_head_dim),)
            )
        return FlexAttentionCompileCrashConfig()

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        return _gemma4_forward_kwargs(model, **kwargs)


GEMMA4_DENSE_HANDLER = Gemma4DenseHandler()

_GEMMA4_ROUTER_PATCHED = False
_GEMMA4_ROTARY_PATCHED = False
_GEMMA4_QKV_PATCHED = False


def _patch_gemma4_router_for_mcore() -> None:
    global _GEMMA4_ROUTER_PATCHED
    if _GEMMA4_ROUTER_PATCHED:
        return
    from megatron.bridge.models.gemma import gemma4_provider
    from megatron.core.transformer.moe.router import TopKRouter

    def _art_gemma4_router_routing(
        self: Any,
        logits: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del input_ids
        routing_probs, routing_map = TopKRouter.routing(
            self,
            logits,
            padding_mask=padding_mask,
        )
        if routing_map is not None:
            prob_sums = routing_probs.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            routing_probs = routing_probs / prob_sums
            routing_probs = routing_probs * self.per_expert_scale.unsqueeze(0)
        return routing_probs, routing_map

    setattr(gemma4_provider.Gemma4TopKRouter, "routing", _art_gemma4_router_routing)
    _GEMMA4_ROUTER_PATCHED = True


def _gemma4_hf_proportional_inv_freq(
    *,
    global_kv_channels: int,
    global_rotary_percent: float,
    rotary_base: int,
    device: torch.device,
) -> torch.Tensor:
    """HF proportional RoPE pads non-rotary pairs with zero-frequency angles."""
    rope_angles = int(global_rotary_percent * global_kv_channels // 2)
    inv_freq_rotated = 1.0 / (
        rotary_base
        ** (
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device)
            / global_kv_channels
        )
    )
    nope_angles = global_kv_channels // 2 - rope_angles
    if nope_angles <= 0:
        return inv_freq_rotated
    return torch.cat(
        (
            inv_freq_rotated,
            torch.zeros(nope_angles, dtype=torch.float32, device=device),
        ),
        dim=0,
    )


def _patch_gemma4_rotary_for_hf_proportional() -> None:
    global _GEMMA4_ROTARY_PATCHED
    if _GEMMA4_ROTARY_PATCHED:
        return
    from megatron.bridge.models.gemma import gemma4_provider

    original_init = cast(Any, gemma4_provider.Gemma4RotaryEmbedding.__init__)

    def _art_gemma4_rotary_init(
        self: Any,
        *,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float | None = None,
        rotary_base: int = 1_000_000,
        rope_scaling: bool = False,
        use_cpu_initialization: bool = False,
        rotary_base_local: int = 10_000,
        global_kv_channels: int = 512,
        global_rotary_percent: float = 0.25,
    ) -> None:
        original_init(
            self,
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            use_cpu_initialization=use_cpu_initialization,
            rotary_base_local=rotary_base_local,
            global_kv_channels=global_kv_channels,
            global_rotary_percent=global_rotary_percent,
        )
        self.inv_freq = _gemma4_hf_proportional_inv_freq(
            global_kv_channels=global_kv_channels,
            global_rotary_percent=global_rotary_percent,
            rotary_base=rotary_base,
            device=self.inv_freq.device,
        )

    setattr(gemma4_provider.Gemma4RotaryEmbedding, "__init__", _art_gemma4_rotary_init)
    _GEMMA4_ROTARY_PATCHED = True


def _patch_gemma4_qkv_for_hf_tied_value() -> None:
    global _GEMMA4_QKV_PATCHED
    if _GEMMA4_QKV_PATCHED:
        return
    from megatron.bridge.models.gemma import gemma4_provider
    from megatron.core.transformer.attention import SelfAttention

    def _art_gemma4_get_query_key_value_tensors(
        self: Any,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        result = cast(
            tuple[Any, ...],
            SelfAttention.get_query_key_value_tensors(
                self,
                hidden_states,
                key_value_states,
                **kwargs,
            ),
        )
        if len(result) < 3:
            return result
        query, key, value = result[0], result[1], result[2]
        # HF global K=V uses the raw K projection for V before k_norm; the
        # synthesized V rows are loaded from K, so V-norm should consume them here.
        v_float = value.float()
        rms = v_float.pow(2).mean(-1, keepdim=True).add(self._v_norm_eps).sqrt()
        value = (v_float / rms).to(value.dtype)
        return (query, key, value) + result[3:]

    setattr(
        gemma4_provider.Gemma4SelfAttention,
        "get_query_key_value_tensors",
        _art_gemma4_get_query_key_value_tensors,
    )
    _GEMMA4_QKV_PATCHED = True


def _gather_absolute_rotary_pos_emb(
    table_source: torch.Tensor,
    *,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    embedding_dim = int(table_source.shape[-1])
    batch_size, sequence_length = position_ids.shape
    gathered = table_source.view(table_source.shape[0], embedding_dim).index_select(
        0,
        position_ids.reshape(-1),
    )
    return (
        gathered.view(batch_size, sequence_length, embedding_dim)
        .permute(1, 0, 2)
        .contiguous()
        .unsqueeze(2)
    )


def _install_gemma4_preprocess_patch(model_chunks: Sequence[Any]) -> None:
    from megatron.core.models.gpt.gpt_model import GPTModel

    for chunk in model_chunks:
        module: Any = chunk
        while hasattr(module, "module"):
            module = module.module
        gpt_module = (
            module
            if isinstance(module, GPTModel)
            else cast(GPTModel, getattr(module, "language_model"))
        )
        preprocess = gpt_module._preprocess

        def preprocess_hook(
            *args: Any,
            _gpt_module: Any = gpt_module,
            _preprocess: Any = preprocess,
            **kwargs: Any,
        ) -> tuple[Any, ...]:
            position_ids = kwargs.get("position_ids")
            gemma4_rotary = getattr(_gpt_module, "rotary_pos_emb")
            local_rotary = getattr(gemma4_rotary, "rope_local", None)
            rotary_cp_group = getattr(gemma4_rotary, "cp_group", None)
            local_rotary_cp_group = getattr(local_rotary, "cp_group", None)
            uses_dispatched_local_cp_positions = (
                isinstance(position_ids, torch.Tensor)
                and position_ids.ndim == 2
                and _context_parallel_world_size(getattr(_gpt_module, "config", None))
                > 1
                and (rotary_cp_group is not None or local_rotary_cp_group is not None)
            )
            if uses_dispatched_local_cp_positions:
                setattr(gemma4_rotary, "cp_group", None)
                if local_rotary is not None:
                    setattr(local_rotary, "cp_group", None)
                rotary_seq_len = getattr(
                    _gpt_module, "_art_gemma4_rotary_seq_len", None
                )
                if rotary_seq_len is not None:
                    from megatron.core.packed_seq_params import PackedSeqParams

                    kwargs = dict(kwargs)
                    kwargs["packed_seq_params"] = PackedSeqParams(
                        max_seqlen_q=int(rotary_seq_len),
                        max_seqlen_kv=int(rotary_seq_len),
                    )
            try:
                preproc_output = list(_preprocess(*args, **kwargs))
            finally:
                if uses_dispatched_local_cp_positions:
                    setattr(gemma4_rotary, "cp_group", rotary_cp_group)
                    if local_rotary is not None:
                        setattr(local_rotary, "cp_group", local_rotary_cp_group)
            decoder_input = cast(torch.Tensor, preproc_output[0])
            if not decoder_input.requires_grad and decoder_input.is_leaf:
                decoder_input.requires_grad_(True)
            rotary_pos_emb = preproc_output[1]
            if not isinstance(position_ids, torch.Tensor) or not isinstance(
                rotary_pos_emb,
                (tuple, list),
            ):
                return tuple(preproc_output)
            local_table, global_table = rotary_pos_emb
            if not torch.is_tensor(local_table) or not torch.is_tensor(global_table):
                return tuple(preproc_output)
            preproc_output[1] = (
                _gather_absolute_rotary_pos_emb(
                    local_table,
                    position_ids=position_ids,
                ),
                _gather_absolute_rotary_pos_emb(
                    global_table,
                    position_ids=position_ids,
                ),
            )
            return tuple(preproc_output)

        gpt_module._preprocess = preprocess_hook  # type: ignore[attr-defined]


def _install_gemma4_full_recompute_patch(model_chunks: Sequence[Any]) -> None:
    for chunk in model_chunks:
        module: Any = chunk
        while hasattr(module, "module"):
            module = module.module
        gpt_module = getattr(module, "language_model", module)
        decoder = getattr(gpt_module, "decoder", None)
        if decoder is None or getattr(
            decoder, "_art_gemma4_full_recompute_patch", False
        ):
            continue
        original_checkpointed_forward = decoder._checkpointed_forward

        def checkpointed_forward(
            self: Any,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            context: torch.Tensor,
            context_mask: torch.Tensor,
            rotary_pos_emb: Any,
            attention_bias: Any,
            packed_seq_params: Any,
            use_inner_quantization_context: bool,
            padding_mask: torch.Tensor | None = None,
            extract_layer_indices: set[int] | None = None,
            layer_offset: int = 0,
            *,
            _original_checkpointed_forward: Any = original_checkpointed_forward,
        ) -> Any:
            if not isinstance(rotary_pos_emb, (tuple, list)):
                return _original_checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                    padding_mask=padding_mask,
                    extract_layer_indices=extract_layer_indices,
                    layer_offset=layer_offset,
                )
            rotary_pos_emb_local, rotary_pos_emb_global = rotary_pos_emb
            if extract_layer_indices is None:
                extract_layer_indices = set()
            intermediate_hidden_states: list[torch.Tensor] = []

            def custom(start: int, end: int) -> Any:
                def custom_forward(
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor,
                    context: torch.Tensor,
                    context_mask: torch.Tensor,
                    rotary_pos_emb_local: torch.Tensor,
                    rotary_pos_emb_global: torch.Tensor,
                    padding_mask: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    rotary_pair = (rotary_pos_emb_local, rotary_pos_emb_global)
                    for index in range(start, end):
                        layer = self._get_layer(index)
                        if use_inner_quantization_context:
                            if self.config.fp8:
                                inner_quantization_context = get_fp8_context(
                                    self.config, layer.layer_number - 1
                                )
                            elif self.config.fp4:
                                inner_quantization_context = get_fp4_context(
                                    self.config, layer.layer_number - 1
                                )
                            else:
                                inner_quantization_context = nullcontext()
                        else:
                            inner_quantization_context = nullcontext()
                        with inner_quantization_context:
                            hidden_states, context = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                context=context,
                                context_mask=context_mask,
                                rotary_pos_emb=rotary_pair,
                                attention_bias=attention_bias,
                                inference_context=None,
                                packed_seq_params=packed_seq_params,
                                padding_mask=padding_mask,
                            )
                    return hidden_states, context

                return custom_forward

            def checkpoint_handler(forward_func: Any) -> Any:
                args = (
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb_local,
                    rotary_pos_emb_global,
                    padding_mask,
                )
                if self.config.fp8 or self.config.fp4:
                    return te_checkpoint(
                        forward_func,
                        self.config.distribute_saved_activations,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        self.pg_collection.tp,
                        *args,
                    )
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    *args,
                )

            if self.config.recompute_method == "uniform":
                layer_idx = 0
                while layer_idx < self.num_layers_per_pipeline_rank:
                    chunk_end = min(
                        layer_idx + self.config.recompute_num_layers,
                        self.num_layers_per_pipeline_rank,
                    )
                    hidden_states, context = checkpoint_handler(
                        custom(layer_idx, chunk_end)
                    )
                    for idx in range(layer_idx, chunk_end):
                        if (idx + layer_offset) in extract_layer_indices:
                            if idx == chunk_end - 1:
                                intermediate_hidden_states.append(hidden_states)
                    layer_idx += self.config.recompute_num_layers
            elif self.config.recompute_method == "block":
                recompute_skip_num_layers = 0
                for layer_idx in range(self.num_layers_per_pipeline_rank):
                    if (
                        self.config.fp8 or self.config.fp4
                    ) and not hidden_states.requires_grad:
                        recompute_skip_num_layers += 1
                    if (
                        layer_idx >= recompute_skip_num_layers
                        and layer_idx
                        < self.config.recompute_num_layers + recompute_skip_num_layers
                    ):
                        hidden_states, context = checkpoint_handler(
                            custom(layer_idx, layer_idx + 1)
                        )
                    else:
                        hidden_states, context = custom(layer_idx, layer_idx + 1)(
                            hidden_states,
                            attention_mask,
                            context,
                            context_mask,
                            rotary_pos_emb_local,
                            rotary_pos_emb_global,
                            padding_mask,
                        )
                    if (layer_idx + layer_offset) in extract_layer_indices:
                        intermediate_hidden_states.append(hidden_states)
            else:
                raise ValueError("Invalid activation recompute method.")
            if len(extract_layer_indices) > 0:
                return hidden_states, intermediate_hidden_states
            return hidden_states

        decoder._checkpointed_forward = MethodType(checkpointed_forward, decoder)
        decoder._art_gemma4_full_recompute_patch = True


def _gemma4_attention_pattern(provider: Any) -> tuple[int, int]:
    pattern = getattr(provider, "interleaved_attn_pattern", (0, 1))
    if not pattern:
        return (0, 1)
    if len(pattern) == 1:
        return (int(pattern[0]), 0)
    return (int(pattern[0]), int(pattern[1]))


def _gemma4_compile_crash_signature(provider: Any) -> tuple[Any, ...]:
    return (
        "moe" if int(getattr(provider, "num_moe_experts", 0) or 0) > 0 else "dense",
        int(provider.num_layers),
        int(provider.hidden_size),
        int(provider.num_attention_heads),
        int(provider.kv_channels),
        int(getattr(provider, "global_head_dim", 0) or 0),
        int(getattr(provider, "num_global_key_value_heads", 0) or 0),
    )


def _is_gemma4_global_layer(layer_number: int, provider: Any) -> bool:
    layer_types = getattr(provider, "art_gemma4_layer_types", None)
    if layer_types is not None:
        return layer_types[int(layer_number) - 1] == "full_attention"
    sliding_count, global_count = _gemma4_attention_pattern(provider)
    if global_count <= 0:
        return False
    cycle = sliding_count + global_count
    if cycle <= 0:
        return False
    return (layer_number - 1) % cycle >= sliding_count


def _gemma4_sliding_window_for_layer(provider: Any, layer_number: int) -> int | None:
    if _is_gemma4_global_layer(int(layer_number), provider):
        return None
    return int(provider.window_size)


def _install_gemma4_flex_core_attention_wrapper(provider: Any) -> None:
    def _wrapper(_config: Any, base_cls: type[Any]) -> type[Any]:
        return _gemma4_flex_core_attention_wrapper(provider, base_cls)

    provider.art_flex_core_attention_wrapper = _wrapper


def _gemma4_flex_core_attention_wrapper(
    provider: Any, base_cls: type[Any]
) -> type[Any]:
    class Gemma4ArtFlexCoreAttention(base_cls):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            config: Any,
            layer_number: int,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            compile_crash_config = getattr(
                provider, "art_flex_compile_crash_config", None
            )
            if compile_crash_config is not None:
                config.art_flex_compile_crash_config = compile_crash_config
            super().__init__(config, layer_number, *args, **kwargs)
            self.art_sliding_window = _gemma4_sliding_window_for_layer(
                provider,
                layer_number,
            )

    return Gemma4ArtFlexCoreAttention


def _attention_provider_for_layer(provider: Any, module: Any) -> Any:
    if not _is_gemma4_global_layer(int(module.layer_number), provider):
        return provider
    global_provider = copy(provider)
    global_provider.kv_channels = getattr(provider, "global_head_dim")
    global_provider.num_query_groups = getattr(provider, "num_global_key_value_heads")
    return global_provider


def _tie_global_value_lora_to_key(self_attention: Any) -> None:
    linear_qkv = self_attention.linear_qkv
    linear_qkv.v_proj_lora = linear_qkv.k_proj_lora


class _Gemma4SelfAttentionLinearProjLoRA(SelfAttentionLinearProjLoRA):
    def __init__(
        self,
        *,
        adapter_model_prefix: str,
        linear_proj: TERowParallelLinear,
        rank: int,
        alpha: int,
        provider: Any,
    ) -> None:
        super().__init__(
            adapter_model_prefix=adapter_model_prefix,
            linear_proj=linear_proj,
            rank=rank,
            alpha=alpha,
            provider=provider,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        linear_proj = self.linear_proj
        base_output, bias_output = TERowParallelLinear.forward(linear_proj, x)
        lora_output = self.lora(x)
        if self.reduce_output and self.provider.tensor_model_parallel_size > 1:
            if self.provider.sequence_parallel:
                lora_output = art_lora.reduce_scatter_to_sequence_parallel_region(
                    lora_output
                )
            else:
                lora_output = art_lora.reduce_from_tensor_model_parallel_region(
                    lora_output
                )
        output = base_output + lora_output
        post_layernorm = getattr(linear_proj, "post_layernorm", None)
        if post_layernorm is not None:
            output = post_layernorm(output)
            if isinstance(output, tuple):
                output = output[0]
        return output, bias_output


def _wrap_gemma4_attention_output_lora(
    self_attention: Any,
    *,
    adapter_model_prefix: str,
    provider: Any,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    from art.megatron.lora import _targets_include, _unwrap_attr

    if not _targets_include(target_modules, "o_proj"):
        return
    linear_proj = _unwrap_attr(
        self_attention.linear_proj,
        "linear_proj",
        TERowParallelLinear,
    )
    self_attention.linear_proj = _Gemma4SelfAttentionLinearProjLoRA(
        adapter_model_prefix=f"{adapter_model_prefix}.self_attn.o_proj",
        linear_proj=linear_proj,
        rank=rank,
        alpha=alpha,
        provider=provider,
    )


def _to_vllm_key(key: str) -> str:
    key = key.replace(".mlp.shared_expert.", ".mlp.").replace(
        ".mlp.experts",
        ".moe.experts",
    )
    return _HF_TEXT_EXPERT_KEY_RE.sub(r"\g<layer>.moe.experts", key)


def _from_vllm_key(key: str) -> str:
    key = key.replace(".moe.experts", ".mlp.experts")
    return _DENSE_MLP_LORA_KEY_RE.sub(
        r"\g<prefix>.shared_expert.\g<module>.\g<lora>.weight",
        key,
    )


def _pack_vllm_3d_lora_b(blocks: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(blocks, dim=0)
    return stacked.permute(1, 2, 0).reshape(stacked.shape[1], -1).contiguous()


def _unpack_vllm_3d_lora_b(
    tensor: torch.Tensor,
    *,
    num_experts: int,
    rank: int,
) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], rank, num_experts).permute(2, 0, 1)


def _clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clone().contiguous()


@lru_cache(maxsize=8)
def _gemma4_text_config_dict(base_model_name_or_path: str) -> dict[str, Any]:
    config_path = Path(base_model_name_or_path) / "config.json"
    if not config_path.exists():
        from huggingface_hub import hf_hub_download

        config_path = Path(
            hf_hub_download(
                base_model_name_or_path,
                "config.json",
                local_files_only=True,
            )
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return dict(config.get("text_config") or config)


def _gemma4_k_eq_v_layers(adapter_config: dict[str, Any]) -> set[int]:
    base_model = str(adapter_config["base_model_name_or_path"])
    config = _gemma4_text_config_dict(base_model)
    if not bool(config.get("attention_k_eq_v", False)):
        return set()
    return {
        layer_idx
        for layer_idx, layer_type in enumerate(config["layer_types"])
        if layer_type == "full_attention"
    }


def _gemma4_hf_file(base_model_name_or_path: str, filename: str) -> Path:
    base_path = Path(base_model_name_or_path)
    if base_path.exists():
        return base_path / filename
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            base_model_name_or_path,
            filename,
            local_files_only=True,
        )
    )


@lru_cache(maxsize=8)
def _gemma4_shared_expert_prenorm_corrections(
    base_model_name_or_path: str,
) -> tuple[torch.Tensor, ...]:
    from safetensors import safe_open

    index = json.loads(
        _gemma4_hf_file(
            base_model_name_or_path,
            "model.safetensors.index.json",
        ).read_text(encoding="utf-8")
    )
    weight_map = dict(index["weight_map"])
    text_config = _gemma4_text_config_dict(base_model_name_or_path)
    num_layers = int(text_config["num_hidden_layers"])
    norm_keys_by_file: dict[str, list[tuple[int, str, str]]] = {}

    for layer in range(num_layers):
        for suffix in (
            "pre_feedforward_layernorm",
            "pre_feedforward_layernorm_2",
        ):
            candidates = (
                f"model.language_model.layers.{layer}.{suffix}.weight",
                f"model.layers.{layer}.{suffix}.weight",
            )
            key = next(candidate for candidate in candidates if candidate in weight_map)
            norm_keys_by_file.setdefault(weight_map[key], []).append(
                (layer, suffix, key)
            )
    norm_weights: dict[tuple[int, str], torch.Tensor] = {}
    for filename, entries in norm_keys_by_file.items():
        with safe_open(
            _gemma4_hf_file(base_model_name_or_path, filename),
            framework="pt",
            device="cpu",
        ) as handle:
            for layer, suffix, key in entries:
                norm_weights[(layer, suffix)] = handle.get_tensor(key).float()

    return tuple(
        norm_weights[(layer, "pre_feedforward_layernorm")]
        / norm_weights[(layer, "pre_feedforward_layernorm_2")]
        for layer in range(num_layers)
    )


def _shared_expert_fc1_prenorm_correction(
    *,
    adapter_config: dict[str, Any],
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    # Megatron Bridge folds pffl/pffl2 into shared-expert FC1 base weights because
    # MCore feeds pffl2-normalized activations while HF/vLLM feeds pffl-normalized
    # activations. LoRA-A needs the same basis change at the HF/vLLM boundary.
    return _gemma4_shared_expert_prenorm_corrections(
        str(adapter_config["base_model_name_or_path"])
    )[layer].to(device=device)


def _rescale_shared_expert_fc1_lora_a(
    key: str,
    tensor: torch.Tensor,
    *,
    adapter_config: dict[str, Any],
    to_vllm: bool,
) -> torch.Tensor:
    match = _SHARED_EXPERT_FC1_LORA_A_KEY_RE.match(key)
    if match is None:
        return tensor
    correction = _shared_expert_fc1_prenorm_correction(
        adapter_config=adapter_config,
        layer=int(match.group("layer")),
        device=tensor.device,
    )
    factor = correction.reciprocal() if to_vllm else correction
    return (tensor.float() * factor.unsqueeze(0)).to(tensor.dtype).contiguous()


def _drop_gemma4_k_eq_v_v_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    k_eq_v_layers = _gemma4_k_eq_v_layers(adapter_config)
    if not k_eq_v_layers:
        return tensors
    return {
        key: tensor
        for key, tensor in tensors.items()
        if not (
            (match := _SELF_ATTN_V_LORA_KEY_RE.match(key)) is not None
            and int(match.group("layer")) in k_eq_v_layers
        )
    }


def _add_gemma4_k_eq_v_v_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    k_eq_v_layers = _gemma4_k_eq_v_layers(adapter_config)
    if not k_eq_v_layers:
        return tensors
    transformed = dict(tensors)
    for key, tensor in tensors.items():
        match = _SELF_ATTN_K_LORA_KEY_RE.match(key)
        if match is None or int(match.group("layer")) not in k_eq_v_layers:
            continue
        v_key = f"{match.group('prefix')}v_proj.{match.group('suffix')}"
        if v_key not in transformed:
            transformed[v_key] = tensor.clone().contiguous()
    return transformed


def _vllm_moe_config(adapter_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(adapter_config)
    target_modules = list(config.get("target_modules") or [])
    if "experts" not in target_modules:
        target_modules.append("experts")
    config["target_modules"] = target_modules
    return config


def _group_art_moe_tensors(
    tensors: dict[str, torch.Tensor],
) -> dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]]:
    grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    for key, tensor in tensors.items():
        match = _ART_MOE_EXPERT_KEY_RE.match(key)
        if match is None:
            continue
        grouped.setdefault(match.group("prefix"), {}).setdefault(
            int(match.group("expert")),
            {},
        ).setdefault(match.group("module"), {})[match.group("lora")] = tensor
    return grouped


def _to_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    grouped = _group_art_moe_tensors(tensors)
    if not grouped:
        transformed = {
            vllm_key: _rescale_shared_expert_fc1_lora_a(
                vllm_key,
                tensor,
                adapter_config=adapter_config,
                to_vllm=True,
            )
            for key, tensor in tensors.items()
            for vllm_key in (_to_vllm_key(key),)
        }
        if len(transformed) != len(tensors):
            raise RuntimeError("Duplicate Gemma 4 LoRA tensor after vLLM conversion")
        transformed = _add_gemma4_k_eq_v_v_lora_tensors(
            transformed,
            adapter_config=adapter_config,
        )
        has_fused_experts = any(_VLLM_MOE_KEY_RE.match(key) for key in transformed)
        return (
            transformed,
            _vllm_moe_config(adapter_config) if has_fused_experts else adapter_config,
        )

    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    for prefix, experts in grouped.items():
        vllm_prefix = _to_vllm_key(prefix)
        gate_up_a: list[torch.Tensor] = []
        gate_up_b: list[torch.Tensor] = []
        down_a: list[torch.Tensor] = []
        down_b: list[torch.Tensor] = []
        for expert in sorted(experts):
            modules = experts[expert]
            try:
                gate_up_a_tensor = modules["gate_up_proj"]["lora_A"]
                gate_up_b_tensor = modules["gate_up_proj"]["lora_B"]
                down_a_tensor = modules["down_proj"]["lora_A"]
                down_b_tensor = modules["down_proj"]["lora_B"]
            except KeyError as exc:
                raise RuntimeError(
                    f"Incomplete Gemma 4 MoE LoRA block for {prefix}.{expert}"
                ) from exc
            gate_up_a.append(gate_up_a_tensor.contiguous())
            gate_up_b.append(gate_up_b_tensor.contiguous())
            down_a.append(down_a_tensor.contiguous())
            down_b.append(down_b_tensor.contiguous())
            for module_name in ("gate_up_proj", "down_proj"):
                for lora_name in ("lora_A", "lora_B"):
                    used_keys.add(f"{prefix}.{expert}.{module_name}.{lora_name}.weight")
        transformed[f"{vllm_prefix}.base_layer.lora_A.weight"] = torch.cat(
            gate_up_a,
            dim=0,
        ).contiguous()
        transformed[f"{vllm_prefix}.base_layer.lora_B.weight"] = _pack_vllm_3d_lora_b(
            gate_up_b
        )
        transformed[f"{vllm_prefix}.lora_A.weight"] = torch.cat(
            down_a,
            dim=0,
        ).contiguous()
        transformed[f"{vllm_prefix}.lora_B.weight"] = _pack_vllm_3d_lora_b(down_b)

    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        vllm_key = _to_vllm_key(key)
        if vllm_key in transformed:
            raise RuntimeError(
                f"Duplicate Gemma 4 LoRA tensor after conversion: {vllm_key}"
            )
        transformed[vllm_key] = _rescale_shared_expert_fc1_lora_a(
            vllm_key,
            tensor,
            adapter_config=adapter_config,
            to_vllm=True,
        )
    transformed = _add_gemma4_k_eq_v_v_lora_tensors(
        transformed,
        adapter_config=adapter_config,
    )
    return transformed, _vllm_moe_config(adapter_config)


def _from_vllm_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    adapter_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    expert_grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]] = {}
    for key, tensor in tensors.items():
        match = _VLLM_MOE_EXPERT_KEY_RE.match(key)
        if match is None:
            continue
        expert_grouped.setdefault(match.group("prefix"), {}).setdefault(
            int(match.group("expert")),
            {},
        ).setdefault(match.group("module"), {})[match.group("lora")] = tensor
    if expert_grouped:
        return _drop_gemma4_k_eq_v_v_lora_tensors(
            _from_vllm_per_expert_lora_tensors(
                tensors,
                expert_grouped=expert_grouped,
                adapter_config=adapter_config,
            ),
            adapter_config=adapter_config,
        )

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        match = _VLLM_MOE_KEY_RE.match(key)
        if match is None:
            continue
        slot = (
            f"{'base_layer.' if match.group('base_layer') else ''}{match.group('lora')}"
        )
        grouped.setdefault(match.group("prefix"), {})[slot] = tensor
    if not grouped:
        return _drop_gemma4_k_eq_v_v_lora_tensors(
            {
                art_key: _rescale_shared_expert_fc1_lora_a(
                    art_key,
                    tensor,
                    adapter_config=adapter_config,
                    to_vllm=False,
                )
                for key, tensor in tensors.items()
                for art_key in (_from_vllm_key(key),)
            },
            adapter_config=adapter_config,
        )

    rank = int(adapter_config["r"])
    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    for prefix, slots in grouped.items():
        try:
            gate_up_a = slots["base_layer.lora_A"]
            gate_up_b = slots["base_layer.lora_B"]
            down_a = slots["lora_A"]
            down_b = slots["lora_B"]
        except KeyError as exc:
            raise RuntimeError(
                f"Incomplete Gemma 4 vLLM MoE LoRA block for {prefix}"
            ) from exc
        if gate_up_a.shape[0] % rank != 0:
            raise RuntimeError(
                f"{prefix}: gate/up lora_A shape {tuple(gate_up_a.shape)} "
                f"is not divisible by rank {rank}"
            )
        num_experts = gate_up_a.shape[0] // rank
        art_prefix = _from_vllm_key(prefix)
        gate_up_b_by_expert = _unpack_vllm_3d_lora_b(
            gate_up_b,
            num_experts=num_experts,
            rank=rank,
        )
        down_b_by_expert = _unpack_vllm_3d_lora_b(
            down_b,
            num_experts=num_experts,
            rank=rank,
        )
        for expert in range(num_experts):
            row = expert * rank
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_A.weight"] = (
                gate_up_a[row : row + rank].contiguous()
            )
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_B.weight"] = (
                gate_up_b_by_expert[expert].contiguous()
            )
            transformed[f"{art_prefix}.{expert}.down_proj.lora_A.weight"] = down_a[
                row : row + rank
            ].contiguous()
            transformed[f"{art_prefix}.{expert}.down_proj.lora_B.weight"] = (
                down_b_by_expert[expert].contiguous()
            )
        used_keys.update(
            {
                f"{prefix}.base_layer.lora_A.weight",
                f"{prefix}.base_layer.lora_B.weight",
                f"{prefix}.lora_A.weight",
                f"{prefix}.lora_B.weight",
            }
        )
    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        art_key = _from_vllm_key(key)
        if art_key in transformed:
            raise RuntimeError(
                f"Duplicate Gemma 4 LoRA tensor after conversion: {art_key}"
            )
        transformed[art_key] = _rescale_shared_expert_fc1_lora_a(
            art_key,
            tensor,
            adapter_config=adapter_config,
            to_vllm=False,
        )
    return _drop_gemma4_k_eq_v_v_lora_tensors(
        transformed,
        adapter_config=adapter_config,
    )


def _from_vllm_per_expert_lora_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    expert_grouped: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]],
    adapter_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    transformed: dict[str, torch.Tensor] = {}
    used_keys: set[str] = set()
    for prefix, experts in expert_grouped.items():
        art_prefix = _from_vllm_key(prefix)
        for expert, modules in experts.items():
            try:
                gate_a = modules["gate_proj"]["lora_A"]
                gate_b = modules["gate_proj"]["lora_B"]
                up_a = modules["up_proj"]["lora_A"]
                up_b = modules["up_proj"]["lora_B"]
                down_a = modules["down_proj"]["lora_A"]
                down_b = modules["down_proj"]["lora_B"]
            except KeyError as exc:
                raise RuntimeError(
                    f"Incomplete Gemma 4 vLLM MoE LoRA block for {prefix}.{expert}"
                ) from exc
            if not torch.equal(gate_a, up_a):
                raise RuntimeError(
                    "Gemma 4 Megatron gate_up_proj requires gate/up LoRA-A "
                    f"tensors to match for {prefix}.{expert}"
                )
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_A.weight"] = _clone(
                gate_a
            )
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_B.weight"] = (
                torch.cat([gate_b, up_b], dim=0).contiguous()
            )
            transformed[f"{art_prefix}.{expert}.down_proj.lora_A.weight"] = _clone(
                down_a
            )
            transformed[f"{art_prefix}.{expert}.down_proj.lora_B.weight"] = _clone(
                down_b
            )
            for module_name in ("gate_proj", "up_proj", "down_proj"):
                for lora_name in ("lora_A", "lora_B"):
                    used_keys.add(f"{prefix}.{expert}.{module_name}.{lora_name}.weight")
    for key, tensor in tensors.items():
        if key in used_keys:
            continue
        if _VLLM_MOE_KEY_RE.match(key) is not None:
            raise RuntimeError(
                "Mixed fused and per-expert Gemma 4 vLLM MoE LoRA tensors"
            )
        art_key = _from_vllm_key(key)
        transformed[art_key] = _rescale_shared_expert_fc1_lora_a(
            art_key,
            tensor,
            adapter_config=adapter_config,
            to_vllm=False,
        )
    return transformed


def _gemma4_text_only_mapping_registry(hf_config: Any | None = None) -> Any:
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.gemma.gemma4_bridge import _Gemma4QKVMapping
    from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge

    upstream_registry = Gemma4VLBridge().mapping_registry()
    global_layer_indices = _gemma4_global_layer_indices(hf_config)
    (
        bridge_gate_up_mapping,
        bridge_down_mapping,
        art_gate_up_mapping,
        art_down_mapping,
    ) = _art_gemma4_expert_mapping_types()

    class _ArtGemma4TextOnlyQKVMapping(_Gemma4QKVMapping):
        def __init__(
            self,
            megatron_param: str,
            q: str,
            k: str,
            v: str,
            *,
            global_layer_indices: tuple[int, ...],
        ) -> None:
            super().__init__(megatron_param, q, k, v)
            self._global_layer_indices = global_layer_indices
            self._export_hf_param = dict(cast(dict[str, str], self.hf_param))

        def resolve(self, captures: tuple[str, ...]) -> Any:
            megatron_param, hf_param = self._resolve_names(captures)
            hf_param = cast(dict[str, str], hf_param)
            resolved = type(self)(
                megatron_param,
                hf_param["q"],
                hf_param["k"],
                hf_param["v"],
                global_layer_indices=self._global_layer_indices,
            )
            layer_index = _megatron_layer_index(megatron_param)
            if layer_index in self._global_layer_indices:
                resolved_hf_param = dict(cast(dict[str, str], resolved.hf_param))
                resolved_hf_param["v"] = resolved_hf_param["k"]
                resolved.hf_param = resolved_hf_param
            return resolved

        def megatron_to_hf(
            self,
            megatron_weights: torch.Tensor | None,
            megatron_module: Any | None,
        ) -> dict[str, torch.Tensor]:
            import_hf_param = self.hf_param
            self.hf_param = self._export_hf_param
            try:
                return super().megatron_to_hf(megatron_weights, megatron_module)
            finally:
                self.hf_param = import_hf_param

    text_config = getattr(hf_config, "text_config", hf_config)
    is_moe = bool(getattr(text_config, "enable_moe_block", False))
    language_mappings = []
    for mapping in upstream_registry.mappings:
        if not mapping.megatron_param.startswith("language_model."):
            continue
        text_mapping = _text_only_gemma4_mapping(
            mapping,
            qkv_mapping_type=_ArtGemma4TextOnlyQKVMapping,
            bridge_gate_up_mapping=bridge_gate_up_mapping,
            bridge_down_mapping=bridge_down_mapping,
            art_gate_up_mapping=art_gate_up_mapping,
            art_down_mapping=art_down_mapping,
            global_layer_indices=global_layer_indices,
        )
        if not is_moe:
            if _is_gemma4_moe_mapping(text_mapping):
                continue
            text_mapping = _gemma4_dense_mapping(text_mapping)
        language_mappings.append(text_mapping)
    return MegatronMappingRegistry(*language_mappings)


class _ImportOnlyMapping:
    def __init__(self, mapping: Any) -> None:
        self._mapping = mapping
        self.megatron_param = mapping.megatron_param
        self.hf_param = mapping.hf_param

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mapping, name)

    def resolve(self, captures: tuple[str, ...]) -> Any:
        return type(self)(self._mapping.resolve(captures))

    def hf_to_megatron(self, hf_weights: Any, megatron_module: Any) -> Any:
        return self._mapping.hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: Any | None,
    ) -> dict[str, torch.Tensor]:
        return {}


def _gemma4_dense_mapping(mapping: Any) -> Any:
    megatron_param = str(getattr(mapping, "megatron_param", ""))
    hf_param = getattr(mapping, "hf_param", None)
    if megatron_param.endswith(".mlp.linear_fc1.layer_norm_weight") and isinstance(
        hf_param, str
    ):
        cloned = copy(mapping)
        cloned.hf_param = hf_param.replace(
            ".post_attention_layernorm.weight",
            ".pre_feedforward_layernorm.weight",
        )
        return cloned
    if megatron_param.endswith(".pffl_weight"):
        return _ImportOnlyMapping(mapping)
    return mapping


def _is_gemma4_moe_mapping(mapping: Any) -> bool:
    megatron_param = str(getattr(mapping, "megatron_param", ""))
    return any(
        part in megatron_param
        for part in (
            ".mlp.shared_experts.",
            ".mlp.router.",
            ".mlp.experts.",
            ".mlp.post_moe_layernorm.",
        )
    )


def _text_only_gemma4_mapping(
    mapping: Any,
    *,
    qkv_mapping_type: type[Any],
    bridge_gate_up_mapping: type[Any],
    bridge_down_mapping: type[Any],
    art_gate_up_mapping: type[Any],
    art_down_mapping: type[Any],
    global_layer_indices: tuple[int, ...],
) -> Any:
    megatron_param = mapping.megatron_param.removeprefix("language_model.")
    hf_param = getattr(mapping, "hf_param", None)
    if isinstance(mapping, bridge_gate_up_mapping):
        return art_gate_up_mapping(megatron_param, hf_param)
    if isinstance(mapping, bridge_down_mapping):
        return art_down_mapping(megatron_param, hf_param)
    if (
        megatron_param.endswith(".self_attention.linear_qkv.weight")
        and isinstance(hf_param, dict)
        and set(hf_param) == {"q", "k", "v"}
    ):
        return qkv_mapping_type(
            megatron_param,
            hf_param["q"],
            hf_param["k"],
            hf_param["v"],
            global_layer_indices=global_layer_indices,
        )
    cloned = copy(mapping)
    cloned.megatron_param = megatron_param
    return cloned


def _art_gemma4_expert_mapping_types() -> tuple[
    type[Any], type[Any], type[Any], type[Any]
]:
    from megatron.bridge.models.conversion.param_mapping import (
        ColumnParallelMapping,
        FusedExpertMapping,
        FusedGatedExpertMapping,
        RowParallelMapping,
        _align_expert_weight_to_shape,
    )
    from megatron.bridge.models.conversion.utils import (
        get_module_and_param_from_name,
    )
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param

    class _ArtGemma4ExpertGateUpProjMapping(FusedGatedExpertMapping):
        def hf_to_megatron(
            self,
            hf_weights: Any,
            megatron_module: Any,
        ) -> torch.Tensor:
            global_expert_number = extract_expert_number_from_param(self.megatron_param)
            expert_weight = _select_gemma4_expert_weight(
                hf_weights,
                global_expert_number=global_expert_number,
                ep_size=int(self.ep_size),
            )
            normalized_param = self._normalize_expert_param_name(self.megatron_param)
            target_param = get_module_and_param_from_name(
                megatron_module, normalized_param
            )[1]
            full_target_shape = (
                target_param.shape[0] * self.tp_size,
                target_param.shape[1],
            )
            gate_target_shape = (
                full_target_shape[0] // 2,
                full_target_shape[1],
            )
            if full_target_shape[0] % 2 != 0:
                raise ValueError(
                    f"Expected even fused dim for {self.megatron_param}, got {full_target_shape}."
                )
            if (
                isinstance(expert_weight, torch.Tensor)
                and expert_weight.ndim == 3
                and expert_weight.shape[0] == 2
            ):
                gate = _align_expert_weight_to_shape(
                    expert_weight[0], torch.Size(gate_target_shape), "gate"
                )
                up = _align_expert_weight_to_shape(
                    expert_weight[1], torch.Size(gate_target_shape), "up"
                )
            else:
                fused = _align_expert_weight_to_shape(
                    cast(torch.Tensor, expert_weight),
                    torch.Size(full_target_shape),
                    "gate_up",
                )
                gate, up = torch.chunk(fused, 2, dim=0)
            return self._gated_mapping.hf_to_megatron(
                {"gate": gate, "up": up},
                megatron_module,
            )

    class _ArtGemma4ExpertDownProjMapping(FusedExpertMapping):
        def hf_to_megatron(
            self,
            hf_weights: Any,
            megatron_module: Any,
        ) -> torch.Tensor:
            global_expert_number = extract_expert_number_from_param(self.megatron_param)
            expert_weight = _select_gemma4_expert_weight(
                hf_weights,
                global_expert_number=global_expert_number,
                ep_size=int(self.ep_size),
            )
            normalized_param = self._normalize_expert_param_name(self.megatron_param)
            target_param = get_module_and_param_from_name(
                megatron_module, normalized_param
            )[1]
            if self._mapping is None:
                self._detected_type = self._detect_parallelism_type(megatron_module)
                self._mapping = self._get_or_create_mapping(self._detected_type)
            if isinstance(self._mapping, ColumnParallelMapping):
                full_target_shape = (
                    target_param.shape[0] * self.tp_size,
                    target_param.shape[1],
                )
            elif isinstance(self._mapping, RowParallelMapping):
                full_target_shape = (
                    target_param.shape[0],
                    target_param.shape[1] * self.tp_size,
                )
            else:
                full_target_shape = tuple(target_param.shape)
            aligned = _align_expert_weight_to_shape(
                expert_weight,
                torch.Size(full_target_shape),
                "down_proj",
            )
            return self._mapping.hf_to_megatron(aligned, megatron_module)

    return (
        FusedGatedExpertMapping,
        FusedExpertMapping,
        _ArtGemma4ExpertGateUpProjMapping,
        _ArtGemma4ExpertDownProjMapping,
    )


def _select_gemma4_expert_weight(
    hf_weights: Any,
    *,
    global_expert_number: int,
    ep_size: int,
) -> Any:
    from art.megatron.runtime.bridge_runtime import ExpertTensorSlice

    if isinstance(hf_weights, ExpertTensorSlice):
        return hf_weights.get(global_expert_number)
    if isinstance(hf_weights, torch.Tensor) and hf_weights.ndim >= 3:
        if ep_size > 1:
            raise RuntimeError(
                "Gemma 4 EP expert loading expected a sliced fused-expert "
                "HF tensor, but received the full all-expert tensor for "
                f"global expert {global_expert_number}."
            )
        return hf_weights[global_expert_number]
    return hf_weights


def _gemma4_global_layer_indices(hf_config: Any | None) -> tuple[int, ...]:
    text_config = getattr(hf_config, "text_config", hf_config)
    layer_types = getattr(text_config, "layer_types", None)
    if not layer_types:
        return ()
    return tuple(
        layer_index
        for layer_index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    )


def _megatron_layer_index(megatron_param: str) -> int | None:
    match = _MEGATRON_LAYER_RE.search(megatron_param)
    return None if match is None else int(match.group("layer"))


_GEMMA4_TEXT_ONLY_BRIDGE_REGISTERED = False


def ensure_gemma4_text_only_bridge_registered() -> None:
    global _GEMMA4_TEXT_ONLY_BRIDGE_REGISTERED
    if _GEMMA4_TEXT_ONLY_BRIDGE_REGISTERED:
        return

    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.transformers_compat import (
        rope_local_base_freq_from_hf,
        rope_theta_from_hf,
    )
    from megatron.bridge.models.gemma.gemma4_bridge import (
        Gemma4Bridge,
        _infer_attn_pattern,
    )
    from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
    from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge
    from megatron.core.models.gpt.gpt_model import GPTModel

    @MegatronModelBridge.register_bridge(
        source="Gemma4ForConditionalGeneration",
        target=GPTModel,
        provider=Gemma4ModelProvider,
        model_type="gemma4",
    )
    class _ArtGemma4TextOnlyBridge(Gemma4Bridge):
        def maybe_modify_converted_hf_weight(
            self,
            task: Any,
            converted_weights_dict: Any,
            hf_state_dict: Any,
        ) -> Any:
            return cast(Any, Gemma4VLBridge).maybe_modify_converted_hf_weight(
                self,
                task,
                converted_weights_dict,
                hf_state_dict,
            )

        def maybe_modify_loaded_hf_weight(
            self,
            hf_param: str | dict[str, str],
            hf_state_dict: Any,
        ) -> Any:
            if isinstance(hf_param, dict) and "v" in hf_param:
                v_name = hf_param["v"]
                if v_name not in hf_state_dict:
                    k_name = hf_param["k"]
                    return {
                        role: (
                            hf_state_dict[k_name].clone()
                            if role == "v"
                            else hf_state_dict[name]
                        )
                        for role, name in hf_param.items()
                    }
            if isinstance(hf_param, dict) and "gate" in hf_param:
                gate_name = hf_param["gate"]
                if "mlp.gate_proj" in gate_name:
                    return cast(Any, Gemma4VLBridge)._fuse_shared_expert_prenorm(
                        self,
                        hf_param,
                        hf_state_dict,
                    )
            if isinstance(hf_param, str) and hf_param.endswith("router.proj.weight"):
                return cast(Any, Gemma4VLBridge)._fuse_router_weight(
                    self,
                    hf_param,
                    hf_state_dict,
                )
            return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

        def provider_bridge(self, hf_pretrained: Any) -> Any:
            text_config = getattr(
                hf_pretrained.config,
                "text_config",
                hf_pretrained.config,
            )
            if int(getattr(text_config, "hidden_size_per_layer_input", 0) or 0) > 0:
                raise ValueError(
                    "ART Gemma 4 support currently targets text backbones without "
                    "per-layer embeddings."
                )
            is_moe = bool(getattr(text_config, "enable_moe_block", False))

            provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
            provider = Gemma4ModelProvider(**provider_kwargs)
            provider.window_size = getattr(text_config, "sliding_window", 1024)
            provider.rotary_base = (
                rope_local_base_freq_from_hf(text_config),
                rope_theta_from_hf(text_config),
            )
            provider.softmax_scale = 1.0
            provider.kv_channels = getattr(text_config, "head_dim", 256)
            provider.qk_layernorm = True
            provider.global_head_dim = getattr(text_config, "global_head_dim", 512)
            provider.num_global_key_value_heads = getattr(
                text_config,
                "num_global_key_value_heads",
                2,
            )
            provider.attention_k_eq_v = getattr(text_config, "attention_k_eq_v", False)
            rope_params = getattr(text_config, "rope_parameters", {})
            if isinstance(rope_params, dict):
                full_attn_rope = rope_params.get("full_attention", {})
                provider.global_rotary_percent = full_attn_rope.get(
                    "partial_rotary_factor",
                    0.25,
                )
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types:
                setattr(provider, "art_gemma4_layer_types", tuple(layer_types))
                provider.interleaved_attn_pattern = _infer_attn_pattern(layer_types)

            if is_moe:
                provider.num_moe_experts = getattr(text_config, "num_experts", 128)
                provider.moe_router_topk = getattr(text_config, "top_k_experts", 8)
                provider.moe_ffn_hidden_size = getattr(
                    text_config,
                    "moe_intermediate_size",
                    704,
                )
                provider.moe_shared_expert_intermediate_size = getattr(
                    text_config,
                    "intermediate_size",
                    2112,
                )
                provider.moe_shared_expert_overlap = False
                provider.moe_shared_expert_gate = False
                provider.moe_layer_freq = 1
            else:
                setattr(provider, "num_moe_experts", None)
                setattr(provider, "moe_ffn_hidden_size", None)
                setattr(provider, "moe_shared_expert_intermediate_size", None)
                provider.ffn_hidden_size = getattr(text_config, "intermediate_size")
            provider.final_logit_softcapping = getattr(
                text_config,
                "final_logit_softcapping",
                30.0,
            )
            provider.bf16 = True
            provider.params_dtype = torch.bfloat16
            provider.autocast_dtype = torch.bfloat16
            provider.make_vocab_size_divisible_by = 128
            return provider

        def mapping_registry(self) -> Any:
            return _gemma4_text_only_mapping_registry(getattr(self, "hf_config", None))

    _GEMMA4_TEXT_ONLY_BRIDGE_REGISTERED = True
