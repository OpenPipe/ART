from __future__ import annotations

from copy import copy
from functools import lru_cache
import re
from types import MethodType
from typing import Any, Sequence, cast

import torch

from art.megatron.model_support.handlers.default_dense import (
    DefaultDenseHandler,
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
    LayerFamilyInstance,
)

_QWEN35_MOE_COMPILE_WORKAROUND_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "flex_token_dispatch_combine",
    "moe_postprocess",
    "te_triton_permute_with_mask_map",
    # Torch 2.11.0 compiles Megatron's weighted SwiGLU custom autograd
    # function with zero cotangents when its forward casts internally.
    "weighted_bias_swiglu_no_inner_forward_cast",
)
_QWEN35_MOE_UNCONDITIONAL_COMPILE_WORKAROUND_FLAGS: tuple[str, ...] = ()
_ART_LAYER_PREFIX = "base_model.model.model.layers."
_VLLM_LAYER_PREFIX = "base_model.model.model.language_model.layers."
_ART_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<module>gate_up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)
_VLLM_MOE_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\."
    r"(?:(?P<base_layer>base_layer)\.)?(?P<lora>lora_[AB])\.weight$"
)
_VLLM_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<module>gate_proj|up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight$"
)


class Qwen35BaseHandler(DefaultDenseHandler):
    key = "qwen3_5_base"
    build_gdn_execution_spec = True
    native_vllm_lora_status = "validated"

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return getattr(base_config, "text_config", base_config)

    def _identity_lora_parameter_suffixes(
        self,
        target_modules: list[str],
    ) -> tuple[str, ...]:
        suffixes = list(super()._identity_lora_parameter_suffixes(target_modules))
        target_set = set(target_modules)
        if "in_proj_qkv" in target_set:
            suffixes.append("linear_attn.in_proj_qkv.weight")
        if "in_proj_z" in target_set:
            suffixes.append("linear_attn.in_proj_z.weight")
        if "out_proj" in target_set:
            suffixes.append("linear_attn.out_proj.weight")
        return tuple(dict.fromkeys(suffixes))

    def to_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if _group_art_moe_tensors(tensors):
            raise TypeError("Dense Qwen3.5 handler received MoE LoRA tensors")
        transformed: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            vllm_key, tensor = _to_vllm_lora_tensor(
                key,
                tensor,
                adapter_config=adapter_config,
            )
            transformed[vllm_key] = tensor
        return (
            transformed,
            adapter_config,
        )

    def from_vllm_lora_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        adapter_config: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if any(_VLLM_MOE_KEY_RE.match(key) for key in tensors):
            raise TypeError("Dense Qwen3.5 handler received MoE vLLM LoRA tensors")
        transformed: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            art_key, tensor = _from_vllm_lora_tensor(
                key,
                tensor,
                adapter_config=adapter_config,
            )
            transformed[art_key] = tensor
        return transformed

    def install_preprocess_patch(self, model_chunks: Sequence[Any]) -> None:
        from megatron.core.models.gpt.gpt_model import GPTModel

        from art.megatron.gdn.operator import (
            install_gdn_island_hooks,
            install_shared_prefix_gdn_hooks,
        )

        install_shared_prefix_gdn_hooks(model_chunks)
        install_gdn_island_hooks(model_chunks)
        for chunk in list(model_chunks):
            module: Any = chunk
            while hasattr(module, "module"):
                module = module.module
            gpt_module = (
                module
                if isinstance(module, GPTModel)
                else cast(GPTModel, getattr(module, "language_model"))
            )
            if getattr(gpt_module, "mtp_process", False) or hasattr(gpt_module, "mtp"):
                raise RuntimeError("ART Qwen3.5 Megatron training does not use MTP.")
            preprocess = gpt_module._preprocess

            def preprocess_hook(*args, _preprocess=preprocess, **kwargs):
                position_ids = kwargs.get("position_ids")
                if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 2:
                    kwargs = dict(kwargs)
                    kwargs["position_ids"] = position_ids.unsqueeze(0).expand(
                        3,
                        position_ids.shape[0],
                        position_ids.shape[1],
                    )
                rotary_pos_emb = getattr(gpt_module, "rotary_pos_emb", None)
                rotary_cp_group = getattr(rotary_pos_emb, "cp_group", None)
                dispatched_local_cp_positions = (
                    isinstance(position_ids, torch.Tensor)
                    and position_ids.ndim == 2
                    and _context_parallel_world_size(
                        getattr(gpt_module, "config", None)
                    )
                    > 1
                    and rotary_cp_group is not None
                )
                if dispatched_local_cp_positions:
                    setattr(rotary_pos_emb, "cp_group", None)
                try:
                    preproc_output = list(_preprocess(*args, **kwargs))
                finally:
                    if dispatched_local_cp_positions:
                        setattr(rotary_pos_emb, "cp_group", rotary_cp_group)
                decoder_input = cast(torch.Tensor, preproc_output[0])
                if not decoder_input.requires_grad and decoder_input.is_leaf:
                    decoder_input.requires_grad_(True)
                return tuple(preproc_output)

            gpt_module._preprocess = preprocess_hook  # type: ignore[attr-defined]

    def _attention_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        linear_attention_pattern = _linear_attention_pattern(provider)
        gated_delta_net_layer_index = (
            linear_attention_pattern.index(1) if 1 in linear_attention_pattern else 0
        )
        standard_attention_layer_index = (
            linear_attention_pattern.index(0) if 0 in linear_attention_pattern else 0
        )
        layer_families = [
            LayerFamilyInstance(
                key="standard_attention",
                layer_index=standard_attention_layer_index,
            ),
            LayerFamilyInstance(
                key="gated_delta_net_attention",
                layer_index=gated_delta_net_layer_index,
            ),
        ]
        return layer_families

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        if int(getattr(provider, "num_moe_experts", 0) or 0) > 0:
            raise TypeError("Dense Qwen3.5 handler received a MoE provider")
        return [
            *self._attention_layer_families(provider),
            LayerFamilyInstance(key="dense_mlp", layer_index=0),
        ]

    def patch_bridge(self, bridge: Any) -> None:
        del bridge

    def configure_provider_for_runtime(self, provider: Any) -> None:
        provider.mtp_num_layers = None
        provider.mtp_loss_scaling_factor = None

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        del bridge
        (
            qwen3_vl_self_attention,
            qwen35_provider_types,
            patch_standard_attention_specs,
            transformer_block_spec_factory,
        ) = _require_qwen35_provider_symbols()
        from art.megatron.provider import patch_art_flex_attention

        matched_provider_type = next(
            provider_type
            for provider_type in qwen35_provider_types
            if isinstance(provider, provider_type)
        )

        def _patch_qwen35_block_spec(block_spec: object, config: Any) -> None:
            patch_standard_attention_specs(block_spec, qwen3_vl_self_attention)
            for layer_spec in getattr(block_spec, "layer_specs", ()):
                patch_art_flex_attention(layer_spec, config)

        def _qwen35_layer_spec(config: Any, vp_stage: int | None = None) -> object:
            block_spec = transformer_block_spec_factory(config, vp_stage=vp_stage)
            _patch_qwen35_block_spec(block_spec, config)
            return block_spec

        def _provide_qwen35_with_flex_attention(
            self: Any,
            pre_process: bool | None = None,
            post_process: bool | None = None,
            vp_stage: int | None = None,
        ) -> Any:
            return matched_provider_type.provide_language_model(
                self,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )

        provider.scatter_embedding_sequence_parallel = True
        provider.transformer_layer_spec = _qwen35_layer_spec
        provider.provide = MethodType(_provide_qwen35_with_flex_attention, provider)
        setattr(provider, "_art_text_only_language_model", True)

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
    ) -> None:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import (
            _adapter_model_prefix,
            _is_language_transformer_layer_name,
            wrap_gated_delta_net_attention,
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
                if isinstance(module.self_attention, SelfAttention):
                    wrap_standard_self_attention(
                        module.self_attention,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=provider,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )
                elif isinstance(module.self_attention, GatedDeltaNet):
                    wrap_gated_delta_net_attention(
                        module.self_attention,
                        adapter_model_prefix=adapter_model_prefix,
                        provider=provider,
                        target_modules=target_set,
                        rank=rank,
                        alpha=alpha,
                    )
                else:
                    raise TypeError(
                        "Unsupported self_attention module type for Megatron LoRA: "
                        f"{type(module.self_attention)}"
                    )
                self._wrap_mlp_lora(
                    module,
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
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
        from megatron.core.transformer.attention import SelfAttention
        from megatron.core.transformer.transformer_layer import TransformerLayer

        from art.megatron.lora import _is_language_transformer_layer_name
        from art.megatron.weights.adapter_export import (
            add_gated_delta_net_adapter_weights,
            add_standard_self_attention_adapter_weights,
            layer_base_prefix,
        )

        _ensure_bridge_qwen35_adapter_name_map()
        adapter_weights_by_base: dict[str, list[Any]] = {}
        for chunk in model_chunks:
            for module_name, module in chunk.named_modules():
                if not isinstance(module, TransformerLayer):
                    continue
                if not _is_language_transformer_layer_name(module_name):
                    continue
                layer_prefix = layer_base_prefix(module, module_name=module_name)
                if isinstance(module.self_attention, SelfAttention):
                    add_standard_self_attention_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        self_attention=module.self_attention,
                    )
                elif isinstance(module.self_attention, GatedDeltaNet):
                    add_gated_delta_net_adapter_weights(
                        adapter_weights_by_base,
                        layer_prefix=layer_prefix,
                        self_attention=module.self_attention,
                    )
                self._add_mlp_adapter_weights(
                    adapter_weights_by_base,
                    layer_prefix=layer_prefix,
                    module=module,
                )
        return adapter_weights_by_base

    def _wrap_mlp_lora(
        self,
        module: Any,
        *,
        adapter_model_prefix: str,
        provider: Any,
        target_modules: set[str],
        rank: int,
        alpha: int,
    ) -> None:
        from art.megatron.lora import wrap_dense_mlp

        _require_dense_mlp(module)
        wrap_dense_mlp(
            module.mlp,
            adapter_model_prefix=adapter_model_prefix,
            provider=provider,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
        )

    def _add_mlp_adapter_weights(
        self,
        adapter_weights_by_base: dict[str, list[Any]],
        *,
        layer_prefix: str,
        module: Any,
    ) -> None:
        from art.megatron.weights.adapter_export import add_dense_mlp_adapter_weights

        _require_dense_mlp(module)
        add_dense_mlp_adapter_weights(
            adapter_weights_by_base,
            layer_prefix=layer_prefix,
            mlp=module.mlp,
        )

    def get_forward_kwargs(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        if type(unwrapped).__name__ == "Qwen3VLModel":
            return {"extra_block_kwargs": {"extra_block_kwargs": kwargs}}
        return {"extra_block_kwargs": kwargs}


class Qwen35DenseHandler(Qwen35BaseHandler):
    key = "qwen3_5_dense"


class Qwen35MoeHandler(Qwen35BaseHandler):
    key = "qwen3_5_moe"
    is_moe = True

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

    def configure_provider_for_runtime(self, provider: Any) -> None:
        super().configure_provider_for_runtime(provider)
        provider.moe_shared_expert_overlap = False

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        if int(getattr(provider, "num_moe_experts", 0) or 0) <= 0:
            raise TypeError("MoE Qwen3.5 handler received a dense provider")
        layer_families = [
            *self._attention_layer_families(provider),
            LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
        ]
        if int(getattr(provider, "moe_shared_expert_intermediate_size", 0) or 0) > 0:
            layer_families.append(
                LayerFamilyInstance(key="shared_experts_mlp", layer_index=0)
            )
        return layer_families

    def _wrap_mlp_lora(
        self,
        module: Any,
        *,
        adapter_model_prefix: str,
        provider: Any,
        target_modules: set[str],
        rank: int,
        alpha: int,
    ) -> None:
        from art.megatron.lora import (
            wrap_grouped_moe_experts_3d,
            wrap_shared_experts_mlp,
        )

        wrap_grouped_moe_experts_3d(
            _require_moe_experts(module),
            adapter_model_prefix=adapter_model_prefix,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
        )
        shared_experts = getattr(module.mlp, "shared_experts", None)
        if shared_experts is not None:
            wrap_shared_experts_mlp(
                shared_experts,
                adapter_model_prefix=adapter_model_prefix,
                provider=provider,
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
            )

    def _add_mlp_adapter_weights(
        self,
        adapter_weights_by_base: dict[str, list[Any]],
        *,
        layer_prefix: str,
        module: Any,
    ) -> None:
        from art.megatron.weights.adapter_export import (
            add_grouped_moe_adapter_weights,
            add_shared_experts_adapter_weights,
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

    def compile_workaround_config(
        self,
        provider: Any,
    ) -> CompileWorkaroundConfig:
        if bool(getattr(provider, "moe_shared_expert_overlap", False)):
            return CompileWorkaroundConfig(
                flags=("moe_forward",),
                unconditional_flags=_QWEN35_MOE_UNCONDITIONAL_COMPILE_WORKAROUND_FLAGS,
                shared_expert_state="shared_expert_overlap",
                disable_compile=True,
            )
        return CompileWorkaroundConfig(
            flags=_compile_workaround_flags_for_provider(
                provider,
                _QWEN35_MOE_COMPILE_WORKAROUND_FLAGS,
            ),
            unconditional_flags=_QWEN35_MOE_UNCONDITIONAL_COMPILE_WORKAROUND_FLAGS,
            shared_expert_state="shared_experts",
            disable_compile=False,
        )


QWEN3_5_DENSE_HANDLER = Qwen35DenseHandler()
QWEN3_5_MOE_HANDLER = Qwen35MoeHandler()


def _to_vllm_key(key: str) -> str:
    return (
        key.replace(_ART_LAYER_PREFIX, _VLLM_LAYER_PREFIX, 1)
        if key.startswith(_ART_LAYER_PREFIX)
        else key
    )


def _from_vllm_key(key: str) -> str:
    return (
        key.replace(_VLLM_LAYER_PREFIX, _ART_LAYER_PREFIX, 1)
        if key.startswith(_VLLM_LAYER_PREFIX)
        else key
    )


def _is_lora_weight_key(key: str) -> bool:
    return key.endswith((".lora_A.weight", ".lora_B.weight"))


def _is_self_attn_q_proj_lora_b(key: str) -> bool:
    return key.endswith(".self_attn.q_proj.lora_B.weight")


@lru_cache(maxsize=8)
def _qwen35_text_config(base_model_name_or_path: str) -> Any:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        base_model_name_or_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    return getattr(config, "text_config", config)


def _qwen35_attention_dims(adapter_config: dict[str, Any]) -> tuple[int, int, int]:
    num_heads = adapter_config.get("num_attention_heads")
    num_groups = adapter_config.get("num_key_value_heads")
    head_dim = adapter_config.get("head_dim")
    hidden_size = adapter_config.get("hidden_size")
    if num_heads is None:
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise RuntimeError("Qwen3.5 LoRA adapter config is missing base model path")
        config = _qwen35_text_config(str(base_model))
        num_heads = getattr(config, "num_attention_heads")
        num_groups = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", None)
        hidden_size = getattr(config, "hidden_size", None)
    num_heads = int(num_heads)
    num_groups = int(num_groups if num_groups is not None else num_heads)
    if head_dim is None:
        if hidden_size is None:
            raise RuntimeError("Qwen3.5 config is missing head_dim and hidden_size")
        head_dim = int(hidden_size) // num_heads
    head_dim = int(head_dim)
    if num_heads % num_groups != 0:
        raise RuntimeError(
            f"Qwen3.5 attention heads {num_heads} are not divisible by "
            f"query groups {num_groups}"
        )
    return num_heads, num_groups, head_dim


def _qwen35_q_proj_lora_b_to_vllm(
    tensor: torch.Tensor,
    adapter_config: dict[str, Any],
) -> torch.Tensor:
    num_heads, num_groups, head_dim = _qwen35_attention_dims(adapter_config)
    heads_per_group = num_heads // num_groups
    expected_rows = num_groups * 2 * heads_per_group * head_dim
    if tensor.shape[0] != expected_rows:
        raise RuntimeError(
            f"Qwen3.5 q_proj LoRA-B rows {tensor.shape[0]} do not match "
            f"attention output rows {expected_rows}"
        )
    rank = tensor.shape[1]
    grouped = tensor.reshape(num_groups, 2 * heads_per_group, head_dim, rank)
    query = grouped[:, :heads_per_group]
    gate = grouped[:, heads_per_group:]
    return torch.cat((query, gate), dim=2).reshape(tensor.shape).contiguous()


def _qwen35_q_proj_lora_b_from_vllm(
    tensor: torch.Tensor,
    adapter_config: dict[str, Any],
) -> torch.Tensor:
    num_heads, num_groups, head_dim = _qwen35_attention_dims(adapter_config)
    heads_per_group = num_heads // num_groups
    expected_rows = num_groups * heads_per_group * 2 * head_dim
    if tensor.shape[0] != expected_rows:
        raise RuntimeError(
            f"Qwen3.5 q_proj LoRA-B rows {tensor.shape[0]} do not match "
            f"attention output rows {expected_rows}"
        )
    rank = tensor.shape[1]
    per_head = tensor.reshape(num_groups, heads_per_group, 2 * head_dim, rank)
    query, gate = per_head.split(head_dim, dim=2)
    return torch.cat((query, gate), dim=1).reshape(tensor.shape).contiguous()


def _to_vllm_lora_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    adapter_config: dict[str, Any],
) -> tuple[str, torch.Tensor]:
    vllm_key = _to_vllm_key(key)
    if _is_self_attn_q_proj_lora_b(vllm_key):
        tensor = _qwen35_q_proj_lora_b_to_vllm(tensor, adapter_config)
    return vllm_key, tensor


def _from_vllm_lora_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    adapter_config: dict[str, Any],
) -> tuple[str, torch.Tensor]:
    art_key = _from_vllm_key(key)
    if _is_self_attn_q_proj_lora_b(art_key):
        tensor = _qwen35_q_proj_lora_b_from_vllm(tensor, adapter_config)
    return art_key, tensor


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


def _has_shared_expert_lora_tensors(tensors: dict[str, torch.Tensor]) -> bool:
    return any(".mlp.shared_expert." in key for key in tensors)


def _vllm_moe_config(
    adapter_config: dict[str, Any],
    *,
    has_shared_experts: bool = False,
) -> dict[str, Any]:
    config = dict(adapter_config)
    stripped_modules = {"gate_up_proj"}
    if not has_shared_experts:
        stripped_modules.update({"gate_proj", "up_proj", "down_proj"})
    target_modules = [
        module
        for module in list(config.get("target_modules") or [])
        if module not in stripped_modules
    ]
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
    has_shared_experts = _has_shared_expert_lora_tensors(tensors)
    transformed: dict[str, torch.Tensor] = {}
    if not grouped:
        has_fused_experts = any(_VLLM_MOE_KEY_RE.match(key) for key in tensors)
        for key, tensor in tensors.items():
            vllm_key, tensor = _to_vllm_lora_tensor(
                key,
                tensor,
                adapter_config=adapter_config,
            )
            if vllm_key in transformed:
                raise RuntimeError(
                    f"Duplicate Qwen3.5 LoRA tensor after conversion: {vllm_key}"
                )
            transformed[vllm_key] = tensor
        return transformed, (
            _vllm_moe_config(
                adapter_config,
                has_shared_experts=has_shared_experts,
            )
            if has_fused_experts
            else adapter_config
        )
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
                d_a = modules["down_proj"]["lora_A"]
                d_b = modules["down_proj"]["lora_B"]
            except KeyError as exc:
                raise RuntimeError(
                    f"Incomplete Qwen3.5 MoE LoRA block for {prefix}.{expert}"
                ) from exc
            if gate_up_b_tensor.shape[0] % 2 != 0:
                raise RuntimeError(
                    f"{prefix}.{expert}: gate/up lora_B rows "
                    f"{gate_up_b_tensor.shape[0]} are not even"
                )
            gate_up_a.append(gate_up_a_tensor.contiguous())
            gate_up_b.append(gate_up_b_tensor.contiguous())
            down_a.append(d_a.contiguous())
            down_b.append(d_b.contiguous())
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
        vllm_key, tensor = _to_vllm_lora_tensor(
            key,
            tensor,
            adapter_config=adapter_config,
        )
        transformed[vllm_key] = tensor
    return transformed, _vllm_moe_config(
        adapter_config,
        has_shared_experts=has_shared_experts,
    )


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
                        f"Incomplete Qwen3.5 vLLM MoE LoRA block for {prefix}.{expert}"
                    ) from exc
                if not torch.equal(gate_a, up_a):
                    raise RuntimeError(
                        "Qwen3.5 Megatron gate_up_proj requires gate/up "
                        f"LoRA-A tensors to match for {prefix}.{expert}"
                    )
                transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_A.weight"] = (
                    _clone(gate_a)
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
                        used_keys.add(
                            f"{prefix}.{expert}.{module_name}.{lora_name}.weight"
                        )
        for key, tensor in tensors.items():
            if key in used_keys:
                continue
            if _VLLM_MOE_KEY_RE.match(key) is not None:
                raise RuntimeError(
                    "Mixed fused and per-expert Qwen3.5 vLLM MoE LoRA tensors"
                )
            art_key, tensor = _from_vllm_lora_tensor(
                key,
                tensor,
                adapter_config=adapter_config,
            )
            transformed[art_key] = tensor
        return transformed

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
        transformed: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            art_key, tensor = _from_vllm_lora_tensor(
                key,
                tensor,
                adapter_config=adapter_config,
            )
            transformed[art_key] = tensor
        return transformed

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
                f"Incomplete Qwen3.5 vLLM MoE LoRA block for {prefix}"
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
            gate_up_a_block = gate_up_a[row : row + rank]
            down_a_block = down_a[row : row + rank]
            gate_up_b_block = gate_up_b_by_expert[expert]
            down_b_block = down_b_by_expert[expert]
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_A.weight"] = (
                gate_up_a_block.contiguous()
            )
            transformed[f"{art_prefix}.{expert}.gate_up_proj.lora_B.weight"] = (
                gate_up_b_block.contiguous()
            )
            transformed[f"{art_prefix}.{expert}.down_proj.lora_A.weight"] = (
                down_a_block.contiguous()
            )
            transformed[f"{art_prefix}.{expert}.down_proj.lora_B.weight"] = (
                down_b_block.contiguous()
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
        art_key, tensor = _from_vllm_lora_tensor(
            key,
            tensor,
            adapter_config=adapter_config,
        )
        transformed[art_key] = tensor
    return transformed


def _ensure_bridge_qwen35_adapter_name_map() -> None:
    from megatron.bridge.models.conversion import peft_bridge

    extra_entries = {
        ".in_proj_qkv.weight": "adapter_qkv",
        ".in_proj_z.weight": "adapter_z",
        ".in_proj_b.weight": "adapter_b",
        ".in_proj_a.weight": "adapter_a",
    }
    for suffix, adapter_key in extra_entries.items():
        peft_bridge.ADAPTER_NAME_MAP.setdefault(suffix, adapter_key)
        peft_bridge.ADAPTER_KEY_TO_SUFFIX.setdefault(adapter_key, suffix)


def _qwen35_provider_types() -> tuple[type[Any], ...]:
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
        Qwen35VLModelProvider,
        Qwen35VLMoEModelProvider,
    )

    return (Qwen35VLModelProvider, Qwen35VLMoEModelProvider)


def _require_qwen35_provider_symbols() -> tuple[Any, ...]:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import (
        Qwen3VLSelfAttention,
    )
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
        Qwen35VLModelProvider,
        Qwen35VLMoEModelProvider,
        _patch_standard_attention_specs,
    )
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )

    return (
        Qwen3VLSelfAttention,
        (Qwen35VLModelProvider, Qwen35VLMoEModelProvider),
        _patch_standard_attention_specs,
        get_transformer_block_with_experimental_attention_variant_spec,
    )


def _register_qwen35_text_only_module_types() -> None:
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    AutoMapping.register_module_type("SharedExpertMLP", "column")
    AutoMapping.register_module_type("GatedDeltaNet", "column")


def _qwen35_text_only_mapping_registry(
    bridge_type: type[Any] | None = None,
) -> Any:
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import (
        Qwen35VLBridge,
        Qwen35VLMoEBridge,
    )

    _register_qwen35_text_only_module_types()
    upstream_bridge_type = bridge_type or Qwen35VLMoEBridge
    assert upstream_bridge_type in {Qwen35VLBridge, Qwen35VLMoEBridge}
    upstream_registry = upstream_bridge_type().mapping_registry()
    language_mappings = [
        _text_only_qwen35_mapping(mapping)
        for mapping in upstream_registry.mappings
        if mapping.megatron_param.startswith("language_model.")
        and not mapping.megatron_param.startswith("language_model.mtp.")
    ]
    return MegatronMappingRegistry(*language_mappings)


def _text_only_qwen35_mapping(mapping: Any) -> Any:
    (
        bridge_gate_up_mapping,
        bridge_down_mapping,
        art_gate_up_mapping,
        art_down_mapping,
    ) = _art_qwen35_expert_mapping_types()
    megatron_param = mapping.megatron_param.removeprefix("language_model.")
    if isinstance(mapping, bridge_gate_up_mapping):
        return art_gate_up_mapping(megatron_param, mapping.hf_param)
    if isinstance(mapping, bridge_down_mapping):
        return art_down_mapping(megatron_param, mapping.hf_param)
    cloned = copy(mapping)
    cloned.megatron_param = megatron_param
    return cloned


@lru_cache(maxsize=1)
def _art_qwen35_expert_mapping_types() -> tuple[
    type[Any], type[Any], type[Any], type[Any]
]:
    from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import (
        FusedExpertMapping,
        FusedGatedExpertMapping,
    )

    class _ArtExpertMLPGateUpProjMapping(FusedGatedExpertMapping):
        def hf_to_megatron(
            self,
            hf_weights: Any,
            megatron_module: Any,
        ) -> torch.Tensor:
            from megatron.bridge.models.conversion.param_mapping import (
                _align_expert_weight_to_shape,
            )
            from megatron.bridge.models.conversion.utils import (
                get_module_and_param_from_name,
            )
            from megatron.bridge.utils.common_utils import (
                extract_expert_number_from_param,
            )

            global_expert_number = extract_expert_number_from_param(self.megatron_param)
            expert_weight = _select_qwen35_expert_weight(
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

    class _ArtExpertMLPDownProjMapping(FusedExpertMapping):
        def hf_to_megatron(
            self,
            hf_weights: Any,
            megatron_module: Any,
        ) -> torch.Tensor:
            from megatron.bridge.models.conversion.param_mapping import (
                ColumnParallelMapping,
                RowParallelMapping,
                _align_expert_weight_to_shape,
            )
            from megatron.bridge.models.conversion.utils import (
                get_module_and_param_from_name,
            )
            from megatron.bridge.utils.common_utils import (
                extract_expert_number_from_param,
            )

            global_expert_number = extract_expert_number_from_param(self.megatron_param)
            expert_weight = _select_qwen35_expert_weight(
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
        _ArtExpertMLPGateUpProjMapping,
        _ArtExpertMLPDownProjMapping,
    )


def _select_qwen35_expert_weight(
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
                "Qwen3.5 EP expert loading expected a sliced fused-expert "
                "HF tensor, but received the full all-expert tensor for "
                f"global expert {global_expert_number}."
            )
        return hf_weights[global_expert_number]
    return hf_weights


_QWEN35_TEXT_ONLY_BRIDGE_REGISTERED = False


def ensure_qwen35_text_only_bridge_registered() -> None:
    global _QWEN35_TEXT_ONLY_BRIDGE_REGISTERED
    if _QWEN35_TEXT_ONLY_BRIDGE_REGISTERED:
        return

    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import (
        _QWEN3_5_DENSE_HF_CLASS_NAME,
        _QWEN3_5_MOE_HF_CLASS_NAME,
        Qwen35VLBridge,
        Qwen35VLMoEBridge,
    )
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
        Qwen35VLModelProvider,
        Qwen35VLMoEModelProvider,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    @MegatronModelBridge.register_bridge(
        source=_QWEN3_5_DENSE_HF_CLASS_NAME,
        target=GPTModel,
        provider=Qwen35VLModelProvider,
        model_type="qwen3_5",
    )
    class _ArtQwen35DenseTextOnlyBridge(Qwen35VLBridge):
        def mapping_registry(self) -> Any:
            return _qwen35_text_only_mapping_registry(Qwen35VLBridge)

    @MegatronModelBridge.register_bridge(
        source=_QWEN3_5_MOE_HF_CLASS_NAME,
        target=GPTModel,
        provider=Qwen35VLMoEModelProvider,
        model_type="qwen3_5_moe",
    )
    class _ArtQwen35TextOnlyBridge(Qwen35VLMoEBridge):
        def mapping_registry(self) -> Any:
            return _qwen35_text_only_mapping_registry(Qwen35VLMoEBridge)

    _QWEN35_TEXT_ONLY_BRIDGE_REGISTERED = True


def _linear_attention_pattern(provider: Any) -> list[int]:
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_linear_attention_pattern,
    )

    return list(get_linear_attention_pattern(provider))
