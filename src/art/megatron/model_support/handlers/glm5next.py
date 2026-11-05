from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch

from art.megatron.glm5next.config import (
    SERVING_MODEL,
    SERVING_REVISION,
    configure_glm5next_provider,
    text_config,
)
from art.megatron.model_support.handlers.glm52 import Glm52Handler
from art.megatron.model_support.spec import HfWeightSource, LayerFamilyInstance


class Glm5NextHandler(Glm52Handler):
    key = "glm5next"
    build_gdn_execution_spec = True
    has_recurrent_layers = True
    native_vllm_lora_status = "wip"
    # M2/M3 implement CP; this handler remains probe-only until GPU qualification.
    cp_supported = True

    def identity_lora_model_config(self, base_config: Any) -> Any:
        return text_config(base_config)

    def patch_bridge(self, bridge: Any) -> None:
        from art.megatron.glm5next.bridge import ArtGlm5NextBridge

        if not isinstance(bridge._model_bridge, ArtGlm5NextBridge):
            raise TypeError("GLM-5.3 Flash requires ArtGlm5NextBridge.")

    def patch_provider(self, provider: Any, bridge: Any) -> None:
        from art.megatron.glm5next.spec import get_glm5next_decoder_block_spec

        pretrained = bridge.hf_pretrained
        configure_glm5next_provider(provider, getattr(pretrained, "config", pretrained))
        provider.transformer_layer_spec = get_glm5next_decoder_block_spec
        if int(provider.pipeline_model_parallel_size or 1) != 1:
            raise ValueError("GLM-5.3 Flash integration currently supports PP1 only.")

    def hf_weight_source(
        self, bridge: Any, hf_param: str, *, task: Any | None = None
    ) -> HfWeightSource:
        del bridge, task
        from art.megatron.glm5next.source import hf_weight_source

        return hf_weight_source(hf_param)

    def context_parallel_workload_profile(self, provider: Any) -> Any:
        del provider
        return None

    def build_prefix_tree_model_state(self, context: Any) -> dict[str, Any]:
        from art.megatron.glm5next.dsa_cp import build_glm5next_dsa_state

        if context.input_pos is None:
            raise ValueError("GLM-5.3 Flash packed attention requires position IDs.")
        return {
            "glm5next_dsa": build_glm5next_dsa_state(
                position_ids=context.input_pos,
                group_ids=context.group_ids,
                parent_ids=context.parent_ids,
                device=context.device,
                context_parallel_state=context.context_parallel_state,
            )
        }

    def collect_layer_families(self, provider: Any) -> list[LayerFamilyInstance]:
        families = [LayerFamilyInstance(key="mhc", layer_index=0)]
        for key, pattern, expected in (
            ("kda_attention", provider.linear_attention_freq, 1),
            ("nope_dsa_attention", provider.linear_attention_freq, 0),
            ("dense_mlp", provider.moe_layer_freq, 0),
            ("grouped_moe_mlp", provider.moe_layer_freq, 1),
            ("shared_experts_mlp", provider.moe_layer_freq, 1),
        ):
            if expected in pattern:
                families.append(
                    LayerFamilyInstance(key=key, layer_index=pattern.index(expected))
                )
        return families

    def prepare_hf_reference_model(self, model: Any) -> Any:
        for module in model.modules():
            if type(module).__name__ == "Glm5NextTextIndexer":
                module.requires_grad_(False)
        return model

    def apply_lora_adapters(
        self,
        model_chunks: Sequence[Any],
        provider: Any,
        *,
        target_modules: list[str],
        rank: int,
        alpha: int,
        lora_layout: Any = None,
    ) -> None:
        from art.megatron.glm5next.lora import (
            LoraDimensions,
            LoraFlags,
            apply_glm5next_lora,
        )
        from art.megatron.glm5next.lora_export import adapter_config
        from art.megatron.lora import LoraLayout

        if lora_layout is not None and lora_layout != LoraLayout(expert="shared_outer"):
            raise ValueError("GLM-5.3 Flash requires the shared_outer LoRA layout.")
        targets = set(target_modules)
        groups = (
            {"q_proj", "k_proj", "v_proj", "o_proj", "q_a_proj", "kv_a_proj_with_mqa"},
            {"gate_proj", "up_proj", "down_proj", "experts"},
            {"lm_head"},
        )
        if targets - set.union(*groups):
            raise ValueError(
                f"Unsupported GLM adapter targets: {sorted(targets - set.union(*groups))}"
            )
        if any(targets & group and not group <= targets for group in groups):
            raise ValueError(
                "GLM adapter flags select complete attention/MLP/unembed families."
            )
        flags = LoraFlags(*(bool(targets & group) for group in groups))
        apply_glm5next_lora(model_chunks, provider, rank=rank, alpha=alpha, flags=flags)
        config = adapter_config(
            rank,
            alpha=alpha,
            dimensions=LoraDimensions.from_config(provider.glm5next_hf_config),
            flags=flags,
        )
        for chunk in model_chunks:
            chunk._art_glm5next_adapter_config = config

    def prepare_adapter_config(
        self, adapter_config: dict[str, Any], model_chunks: Sequence[Any]
    ) -> dict[str, Any]:
        from art.megatron.glm5next.lora_export import META

        configs = [
            module._art_glm5next_adapter_config
            for chunk in model_chunks
            for module in chunk.modules()
            if hasattr(module, "_art_glm5next_adapter_config")
        ]
        if not configs or any(config != configs[0] for config in configs):
            raise ValueError(
                "GLM adapter publication requires one consistent installed LoRA contract."
            )
        contract = deepcopy(configs[0])
        for key in ("r", "lora_alpha"):
            if key in adapter_config and adapter_config[key] != contract[key]:
                raise ValueError(
                    f"GLM adapter publication changed {key} after installation."
                )
        return {**adapter_config, **contract, META: contract[META]}

    def build_adapter_weights_by_base(
        self, model_chunks: Sequence[Any]
    ) -> dict[str, list[Any]]:
        from art.megatron.glm5next.lora_layers import (
            build_glm5next_adapter_weights_by_base,
        )

        return build_glm5next_adapter_weights_by_base(model_chunks)

    def to_vllm_lora_tensors(
        self, tensors: dict[str, torch.Tensor], *, adapter_config: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        from art.megatron.glm5next.lora_export import to_vllm_lora_tensors

        tensors, config = to_vllm_lora_tensors(tensors, adapter_config=adapter_config)
        return tensors, self.to_vllm_lora_config(config)

    def from_vllm_lora_tensors(
        self, tensors: dict[str, torch.Tensor], *, adapter_config: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        from art.megatron.glm5next.lora_export import from_vllm_lora_tensors

        return from_vllm_lora_tensors(tensors, adapter_config=adapter_config)

    def expert_packed_lora_groups(self) -> tuple:
        # M5's codec needs native global expert factors before stacking w1/w2/w3.
        return ()

    def to_vllm_lora_config(self, adapter_config: dict[str, Any]) -> dict[str, Any]:
        config = dict(adapter_config)
        config["base_model_name_or_path"] = SERVING_MODEL
        config["revision"] = SERVING_REVISION
        return config


GLM5NEXT_HANDLER = Glm5NextHandler()


def ensure_glm5next_bridge_registered() -> None:
    from art.megatron.glm5next.bridge import (
        ensure_glm5next_bridge_registered as register,
    )

    register()
