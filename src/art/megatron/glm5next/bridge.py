# Text mapping port from NVIDIA Megatron-Bridge #6044 (58382ddf), Apache-2.0,
# and SkyRL's glm5_next bridge at 3ba6f33d. ART uses one layer per HF block.
from __future__ import annotations

from typing import Any

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.models.gpt.gpt_model import GPTModel
import torch

from art.megatron.glm5next.config import (
    SERVING_MODEL,
    TRAINING_MODEL,
    TRAINING_REVISION,
    configure_glm5next_provider,
    text_config,
)

_BRIDGE_REGISTERED = False


def training_checkpoint_path(model_name_or_path: str) -> str:
    """Resolve the pinned sibling before Bridge's revision-unaware StateSource."""
    if model_name_or_path not in {SERVING_MODEL, TRAINING_MODEL}:
        return model_name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=TRAINING_MODEL,
        revision=TRAINING_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer.model"],
    )


def glm5next_mapping_registry() -> MegatronMappingRegistry:
    for module in ("Glm5NextRMSNorm", "Glm5NextFinalNorm", "Glm5NextRouter"):
        AutoMapping.register_module_type(module, "replicated")
    layer = "decoder.layers.*"
    hf = "model.language_model.layers.*"
    attn, hf_attn = f"{layer}.self_attention", f"{hf}.self_attn"
    direct = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "output_layer.weight": "lm_head.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        f"{layer}.input_layernorm.weight": f"{hf}.input_layernorm.weight",
        f"{layer}.pre_mlp_layernorm.weight": f"{hf}.post_attention_layernorm.weight",
        f"{layer}.mlp.router.weight": f"{hf}.mlp.gate.weight",
        f"{layer}.mlp.router.expert_bias": f"{hf}.mlp.gate.e_score_correction_bias",
        f"{layer}.mlp.linear_fc2.weight": f"{hf}.mlp.down_proj.weight",
        f"{layer}.mlp.shared_experts.linear_fc2.weight": f"{hf}.mlp.shared_experts.down_proj.weight",
        f"{layer}.mlp.experts.linear_fc2.weight*": f"{hf}.mlp.experts.*.down_proj.weight",
        f"{layer}.mlp.experts.local_experts.*.linear_fc2.weight": f"{hf}.mlp.experts.*.down_proj.weight",
    }
    # glm52-compatible MLA names for M3; the indexer can live directly on the
    # attention or under core_attention, depending on the handed-off spec.
    for module, source in (
        ("linear_q_down_proj", "q_a_proj"),
        ("linear_q_up_proj", "q_b_proj"),
        ("q_layernorm", "q_a_layernorm"),
        ("linear_kv_down_proj", "kv_a_proj_with_mqa"),
        ("linear_kv_up_proj", "kv_b_proj"),
        ("kv_layernorm", "kv_a_layernorm"),
        ("linear_proj", "o_proj"),
    ):
        direct[f"{attn}.{module}.weight"] = f"{hf_attn}.{source}.weight"
    mappings: list[Any] = [AutoMapping(m, h) for m, h in direct.items()]
    for site in ("attn", "ffn"):
        for suffix in ("fn", "base", "scale"):
            mappings.append(
                ReplicatedMapping(
                    f"{layer}.hc_{site}_{suffix}", f"{hf}.hc_{site}_{suffix}"
                )
            )
    for name in (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_conv1d.weight",
        "k_conv1d.weight",
        "v_conv1d.weight",
        "f_b_proj.weight",
        "g_b_proj.weight",
        "b_proj.weight",
        "A_log",
        "dt_bias",
    ):
        mappings.append(ColumnParallelMapping(f"{attn}.{name}", f"{hf_attn}.{name}"))
    for name in ("f_a_proj.weight", "g_a_proj.weight", "o_norm.weight"):
        mappings.append(ReplicatedMapping(f"{attn}.{name}", f"{hf_attn}.{name}"))
    mappings.append(
        RowParallelMapping(f"{attn}.o_proj.weight", f"{hf_attn}.o_proj.weight")
    )
    for prefix in ("indexer", "core_attention.indexer"):
        for name, source in (
            ("linear_wq_b.weight", "wq_b.weight"),
            ("linear_wk.weight", "wk.weight"),
            ("linear_weights_proj.weight", "weights_proj.weight"),
            ("k_norm.weight", "k_norm.weight"),
            ("k_norm.bias", "k_norm.bias"),
            ("index_kpool_compress_ape", "index_kpool_compress_ape"),
            ("index_kpool_compress_gate", "index_kpool_compress_gate"),
        ):
            mappings.append(
                ReplicatedMapping(
                    f"{attn}.{prefix}.{name}", f"{hf_attn}.indexer.{source}"
                )
            )
    for projection, hf_projection in (
        ("mlp.linear_fc1.weight", "mlp"),
        ("mlp.shared_experts.linear_fc1.weight", "mlp.shared_experts"),
        ("mlp.experts.linear_fc1.weight*", "mlp.experts.*"),
        ("mlp.experts.local_experts.*.linear_fc1.weight", "mlp.experts.*"),
    ):
        mappings.append(
            GatedMLPMapping(
                f"{layer}.{projection}",
                gate=f"{hf}.{hf_projection}.gate_proj.weight",
                up=f"{hf}.{hf_projection}.up_proj.weight",
            )
        )
    # Fallback reference modules retain HF's canonical child names. The source
    # hook materializes fused conv weights and newer HF's nested mHC/gate names.
    mappings.append(ReplicatedMapping(f"{attn}.hf_reference.**", f"{hf_attn}.**"))
    return MegatronMappingRegistry(*mappings)


class ArtGlm5NextBridge(MegatronModelBridge[Any, GPTModelProvider, GPTModel]):
    art_stream_host_import = True

    def hf_config_to_provider_kwargs(self, hf_config: Any) -> dict[str, Any]:
        cfg = text_config(hf_config)
        values = super().hf_config_to_provider_kwargs(cfg)
        # HF head_dim=0 is a NoPE sentinel, not the attention channel width.
        values["kv_channels"] = int(cfg.v_head_dim)
        return values

    def provider_bridge(self, hf_pretrained: Any) -> Any:
        from art.megatron.glm5next.spec import get_glm5next_decoder_block_spec

        provider = super().provider_bridge(hf_pretrained)
        configure_glm5next_provider(provider, hf_pretrained.config)
        provider.transformer_layer_spec = get_glm5next_decoder_block_spec
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        return glm5next_mapping_registry()

    def build_conversion_tasks(self, hf_pretrained: Any, megatron_model: Any) -> Any:
        from art.megatron.glm5next.source import Glm5NextTaskSource

        state = getattr(hf_pretrained, "state", None)
        source = getattr(state, "source", None)
        if state is None or source is None:
            return super().build_conversion_tasks(hf_pretrained, megatron_model)
        # Bridge checks key existence before calling our materializer. F8's HF
        # layout has nested mHC/forget-gate names and fused conv/expert tensors.
        # Advertise only aliases backed by real keys, during planning only.
        cfg = text_config(hf_pretrained.config)
        state.source = Glm5NextTaskSource(source, int(cfg.n_routed_experts))
        try:
            return super().build_conversion_tasks(hf_pretrained, megatron_model)
        finally:
            state.source = source

    def _art_hf_weight_source(self, key: str, *, task: Any = None) -> Any:
        from art.megatron.glm5next.source import hf_weight_source

        return hf_weight_source(key)

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained: Any,
        megatron_model: Any,
        allowed_mismatched_params: list[str] | None = None,
    ) -> Any:
        source = str(hf_pretrained.model_name_or_path)
        path = training_checkpoint_path(source)
        if path != source:
            hf_pretrained = PreTrainedCausalLM.from_pretrained(path)
        return super().load_weights_hf_to_megatron(
            hf_pretrained, megatron_model, allowed_mismatched_params
        )

    def maybe_modify_loaded_hf_weight(self, hf_param: Any, hf_state_dict: Any) -> Any:
        if isinstance(hf_param, dict):
            return {
                key: self.maybe_modify_loaded_hf_weight(name, hf_state_dict)
                for key, name in hf_param.items()
            }
        from art.megatron.glm5next.source import materialize_hf_weight

        result = materialize_hf_weight(hf_param, hf_state_dict)
        if result.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError(
                "GLM training requires BF16 weights; FP8 import is not supported."
            )
        return result


def ensure_glm5next_bridge_registered() -> None:
    global _BRIDGE_REGISTERED
    if _BRIDGE_REGISTERED:
        return
    from art.megatron.glm5next.hf_config import ensure_glm5next_config_registered

    ensure_glm5next_config_registered()
    MegatronModelBridge.register_bridge(
        source="Glm5NextForConditionalGeneration",
        target=GPTModel,
        provider=GPTModelProvider,
        model_type="glm5_next",
    )(ArtGlm5NextBridge)  # ty: ignore[invalid-argument-type]
    MegatronModelBridge.register_bridge(
        source="Glm5NextTextOracle",
        target=GPTModel,
        provider=GPTModelProvider,
        model_type="glm5_next_text",
    )(ArtGlm5NextBridge)  # ty: ignore[invalid-argument-type]
    _BRIDGE_REGISTERED = True
