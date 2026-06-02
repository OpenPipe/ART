from functools import partial
from typing import Any, cast

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel

from art.megatron.dsv4.spec import get_dsv4_decoder_block_spec


def _dsv4_mapping_registry() -> MegatronMappingRegistry:
    mappings: list[Any] = [
        AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        AutoMapping(
            "decoder.layers.*.input_layernorm.weight",
            "model.layers.*.input_layernorm.weight",
        ),
        AutoMapping(
            "decoder.layers.*.pre_mlp_layernorm.weight",
            "model.layers.*.post_attention_layernorm.weight",
        ),
        AutoMapping(
            "decoder.layers.*.mlp.router.weight", "model.layers.*.mlp.gate.weight"
        ),
        AutoMapping(
            "decoder.layers.*.mlp.router.tid2eid", "model.layers.*.mlp.gate.tid2eid"
        ),
        AutoMapping(
            "decoder.layers.*.mlp.router.e_score_correction_bias",
            "model.layers.*.mlp.gate.e_score_correction_bias",
        ),
        AutoMapping(
            "decoder.layers.*.mlp.experts.linear_fc2.weight*",
            "model.layers.*.mlp.experts.*.down_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
            "model.layers.*.mlp.shared_experts.down_proj.weight",
        ),
        AutoMapping("decoder.final_layernorm.weight", "model.norm.weight"),
        AutoMapping(
            "decoder.final_layernorm.hc_head_params.hc_head_fn", "model.hc_head.hc_fn"
        ),
        AutoMapping(
            "decoder.final_layernorm.hc_head_params.hc_head_base",
            "model.hc_head.hc_base",
        ),
        AutoMapping(
            "decoder.final_layernorm.hc_head_params.hc_head_scale",
            "model.hc_head.hc_scale",
        ),
        AutoMapping("output_layer.weight", "lm_head.weight"),
        AutoMapping("decoder.layers.*.hc_attn_fn", "model.layers.*.attn_hc.fn"),
        AutoMapping("decoder.layers.*.hc_attn_base", "model.layers.*.attn_hc.base"),
        AutoMapping("decoder.layers.*.hc_attn_scale", "model.layers.*.attn_hc.scale"),
        AutoMapping("decoder.layers.*.hc_ffn_fn", "model.layers.*.ffn_hc.fn"),
        AutoMapping("decoder.layers.*.hc_ffn_base", "model.layers.*.ffn_hc.base"),
        AutoMapping("decoder.layers.*.hc_ffn_scale", "model.layers.*.ffn_hc.scale"),
        AutoMapping(
            "decoder.layers.*.self_attention.wq_a.weight",
            "model.layers.*.self_attn.q_a_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.q_norm.weight",
            "model.layers.*.self_attn.q_a_norm.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.wq_b.weight",
            "model.layers.*.self_attn.q_b_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.wkv.weight",
            "model.layers.*.self_attn.kv_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.kv_norm.weight",
            "model.layers.*.self_attn.kv_norm.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.wo_a.weight",
            "model.layers.*.self_attn.o_a_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.wo_b.weight",
            "model.layers.*.self_attn.o_b_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.attn_sink",
            "model.layers.*.self_attn.sinks",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.compressor.ape",
            "model.layers.*.self_attn.compressor.position_bias",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.compressor.wkv.weight",
            "model.layers.*.self_attn.compressor.kv_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.compressor.wgate.weight",
            "model.layers.*.self_attn.compressor.gate_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.compressor.norm.weight",
            "model.layers.*.self_attn.compressor.kv_norm.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.indexer.linear_wq_b.weight",
            "model.layers.*.self_attn.compressor.indexer.q_b_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.indexer.linear_weights_proj.weight",
            "model.layers.*.self_attn.compressor.indexer.weights_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.indexer.compressor.ape",
            "model.layers.*.self_attn.compressor.indexer.position_bias",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.indexer.compressor.wkv.weight",
            "model.layers.*.self_attn.compressor.indexer.kv_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.indexer.compressor.wgate.weight",
            "model.layers.*.self_attn.compressor.indexer.gate_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.self_attention.indexer.compressor.norm.weight",
            "model.layers.*.self_attn.compressor.indexer.kv_norm.weight",
        ),
        GatedMLPMapping(
            megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
            gate="model.layers.*.mlp.experts.*.gate_proj.weight",
            up="model.layers.*.mlp.experts.*.up_proj.weight",
        ),
        GatedMLPMapping(
            megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
            gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
            up="model.layers.*.mlp.shared_experts.up_proj.weight",
        ),
    ]
    return MegatronMappingRegistry(*mappings)


class ArtDeepSeekV4Bridge(DeepSeekV3Bridge):
    def provider_bridge(self, hf_pretrained: Any):
        hf_config = hf_pretrained.config
        if not hasattr(hf_config, "first_k_dense_replace"):
            hf_config.first_k_dense_replace = 0
        provider = cast(Any, super().provider_bridge(hf_pretrained))
        provider.transformer_layer_spec = partial(get_dsv4_decoder_block_spec)
        provider.num_layers = hf_config.num_hidden_layers
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.multi_latent_attention = False
        provider.q_lora_rank = hf_config.q_lora_rank
        provider.kv_lora_rank = hf_config.head_dim
        provider.qk_pos_emb_head_dim = hf_config.qk_rope_head_dim
        provider.num_attention_heads = hf_config.num_attention_heads
        provider.num_query_groups = 1
        provider.kv_channels = hf_config.head_dim
        provider.num_moe_experts = hf_config.n_routed_experts
        provider.moe_router_topk = hf_config.num_experts_per_tok
        provider.moe_router_score_function = hf_config.scoring_func
        provider.moe_router_topk_scaling_factor = hf_config.routed_scaling_factor
        provider.moe_router_enable_expert_bias = False
        provider.moe_router_fusion = False
        provider.moe_layer_freq = [1] * hf_config.num_hidden_layers
        provider.moe_ffn_hidden_size = hf_config.moe_intermediate_size
        provider.ffn_hidden_size = hf_config.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = (
            hf_config.moe_intermediate_size * hf_config.n_shared_experts
        )
        provider.dsv4_hc_mult = getattr(hf_config, "hc_mult", 4)
        provider.dsv4_hc_sinkhorn_iters = getattr(hf_config, "hc_sinkhorn_iters", 20)
        provider.dsv4_hc_eps = getattr(hf_config, "hc_eps", 1e-6)
        provider.dsv4_compress_ratios = getattr(hf_config, "compress_ratios", None)
        provider.dsv4_compress_rope_theta = getattr(
            hf_config, "compress_rope_theta", 160000
        )
        provider.dsv4_swiglu_limit = getattr(hf_config, "swiglu_limit", 0.0)
        provider.dsv4_o_groups = getattr(hf_config, "o_groups", 16)
        provider.dsv4_o_lora_rank = getattr(hf_config, "o_lora_rank", 1024)
        provider.dsv4_n_hash_layers = getattr(hf_config, "n_hash_layers", 3)
        provider.dsv4_window_size = getattr(hf_config, "sliding_window", 128)
        provider.dsa_indexer_n_heads = getattr(hf_config, "index_n_heads", 64)
        provider.dsa_indexer_head_dim = getattr(hf_config, "index_head_dim", 128)
        provider.dsa_indexer_topk = getattr(hf_config, "index_topk", 1024)
        if provider.dsv4_swiglu_limit > 0:
            provider.bias_activation_fusion = False
            provider.activation_func_clamp_value = provider.dsv4_swiglu_limit
        provider.mtp_num_layers = None
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        return _dsv4_mapping_registry()


_DSV4_BRIDGE_REGISTERED = False


def ensure_dsv4_bridge_registered() -> None:
    global _DSV4_BRIDGE_REGISTERED
    if _DSV4_BRIDGE_REGISTERED:
        return
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

    from art.megatron.dsv4.hf_config import ensure_dsv4_hf_config_registered

    ensure_dsv4_hf_config_registered()
    MegatronModelBridge.register_bridge(
        source="DeepseekV4ForCausalLM",
        target=GPTModel,
        provider=MLAModelProvider,
        model_type="deepseek_v4",
    )(ArtDeepSeekV4Bridge)
    _DSV4_BRIDGE_REGISTERED = True
