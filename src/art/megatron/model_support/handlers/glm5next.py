from __future__ import annotations

from typing import Any, Sequence

import torch

from art.megatron.glm5next.config import (
    SERVING_MODEL,
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
    # The fallback operators refuse CP>1. M2/M3 CP qualification precedes
    # enabling CP for this handler; an explicit override alone cannot enable it.
    cp_supported = False

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
        if context.context_parallel_state is not None:
            raise NotImplementedError(
                "GLM CP state awaits M2/M3 native attention integration."
            )
        # The reference modules consume group/parent metadata directly from
        # attention_bias. M3 adds its native cached DSA state on integration.
        return {}

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
    ) -> None:
        del model_chunks, provider, target_modules, rank, alpha
        raise NotImplementedError(
            "GLM adapter wrapping awaits the M5 LoRA contract handoff."
        )

    def build_adapter_weights_by_base(
        self, model_chunks: Sequence[Any]
    ) -> dict[str, list[Any]]:
        del model_chunks
        raise NotImplementedError(
            "GLM adapter export awaits the M5 LoRA contract handoff."
        )

    def to_vllm_lora_tensors(
        self, tensors: dict[str, torch.Tensor], *, adapter_config: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        del tensors, adapter_config
        raise NotImplementedError(
            "GLM adapter export awaits the M5 LoRA contract handoff."
        )

    def to_vllm_lora_config(self, adapter_config: dict[str, Any]) -> dict[str, Any]:
        config = dict(adapter_config)
        config["base_model_name_or_path"] = SERVING_MODEL
        return config


GLM5NEXT_HANDLER = Glm5NextHandler()


def ensure_glm5next_bridge_registered() -> None:
    from art.megatron.glm5next.bridge import (
        ensure_glm5next_bridge_registered as register,
    )

    register()
