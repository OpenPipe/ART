from __future__ import annotations

from typing import Any

SERVING_MODEL = "zai-org/GLM-5.3-Flash"
SERVING_REVISION = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
TRAINING_MODEL = "zai-org/GLM-5.3-Flash-BF16"
TRAINING_REVISION = "f9fed3bbc82291073373d67452e9f25317ef7577"


def text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def configure_glm5next_provider(provider: Any, config: Any) -> None:
    """Port #6044's text geometry without its MCore 0.21 HybridModel dependency.

    One ART TransformerLayer is one HF layer, with two mHC sites. Upstream's
    90 attention/FFN physical layers therefore become 45 ART decoder layers.
    """
    cfg = text_config(config)
    count = int(cfg.num_hidden_layers)
    attention = tuple(cfg.layer_types)
    mlp = tuple(cfg.mlp_layer_types)
    if len(attention) != count or len(mlp) != count:
        raise ValueError("GLM-5.3 Flash layer patterns must cover num_hidden_layers.")
    dsa_types = {"deepseek_sparse_attention", "indexed_attention", "full_attention"}
    if set(attention) - {"linear_attention", *dsa_types}:
        raise ValueError(f"Unsupported GLM-5.3 Flash attention pattern: {attention}.")
    if set(mlp) - {"dense", "sparse"}:
        raise ValueError(f"Unsupported GLM-5.3 Flash MLP pattern: {mlp}.")
    if int(cfg.qk_rope_head_dim) != 0 or not getattr(cfg, "mla_use_nope", True):
        raise ValueError("GLM-5.3 Flash requires NoPE MLA (qk_rope_head_dim=0).")
    if getattr(cfg, "scoring_func", "sigmoid") != "sigmoid":
        raise ValueError("GLM-5.3 Flash requires sigmoid routing.")
    if getattr(cfg, "topk_method", "noaux_tc") != "noaux_tc":
        raise ValueError("GLM-5.3 Flash requires noaux_tc routing.")
    if not getattr(cfg, "norm_topk_prob", True):
        raise ValueError("GLM-5.3 Flash requires normalized top-k probabilities.")
    if int(getattr(cfg, "n_group", 1) or 1) != 1:
        raise ValueError("GLM-5.3 Flash currently requires one routing group.")
    if any(
        kind != "full"
        for kind, layer in zip(cfg.indexer_types, attention, strict=True)
        if layer in dsa_types
    ):
        raise ValueError("GLM-5.3 Flash does not support cross-layer index sharing.")

    linear = getattr(cfg, "linear_attn_config", {}) or {}

    def linear_value(name: str, alias: str) -> Any:
        value = getattr(cfg, name, None)
        return linear[alias] if value is None else value

    values = {
        "num_layers": count,
        "hidden_size": int(cfg.hidden_size),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_query_groups": int(cfg.num_attention_heads),
        "kv_channels": int(cfg.v_head_dim),
        "ffn_hidden_size": int(cfg.intermediate_size),
        "vocab_size": int(cfg.vocab_size),
        "make_vocab_size_divisible_by": 128,
        "share_embeddings_and_output_weights": False,
        "normalization": "RMSNorm",
        "layernorm_epsilon": float(cfg.rms_norm_eps),
        "add_bias_linear": False,
        "add_qkv_bias": False,
        "gated_linear_unit": True,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "multi_latent_attention": False,
        "qk_layernorm": True,
        "q_lora_rank": int(cfg.q_lora_rank),
        "kv_lora_rank": int(cfg.kv_lora_rank),
        "qk_head_dim": int(cfg.qk_nope_head_dim),
        "qk_pos_emb_head_dim": 0,
        "v_head_dim": int(cfg.v_head_dim),
        "position_embedding_type": "none",
        "apply_rope_fusion": False,
        "experimental_attention_variant": None,
        "linear_attention_freq": [int(t == "linear_attention") for t in attention],
        "linear_num_key_heads": int(linear_value("linear_num_heads", "num_heads")),
        "linear_num_value_heads": int(linear_value("linear_num_heads", "num_heads")),
        "linear_key_head_dim": int(linear_value("linear_head_dim", "head_dim")),
        "linear_value_head_dim": int(linear_value("linear_head_dim", "head_dim")),
        "linear_conv_kernel_dim": int(
            linear_value("linear_conv_kernel_dim", "short_conv_kernel_size")
        ),
        "kda_gate_lower_bound": linear_value("linear_lower_bound", "gate_lower_bound"),
        "kda_lower_bound": linear_value("linear_lower_bound", "gate_lower_bound"),
        "dsa_indexer_n_heads": int(cfg.index_n_heads),
        "dsa_indexer_head_dim": int(cfg.index_head_dim),
        "dsa_indexer_topk": int(cfg.index_topk),
        "dsa_indexer_kpool": int(cfg.index_kpool),
        "dsa_indexer_kpool_always_select_tail": bool(
            cfg.index_kpool_always_select_tail
        ),
        "dsa_indexer_loss_coeff": 0.0,
        "dsa_indexer_use_sparse_loss": False,
        "num_moe_experts": int(cfg.n_routed_experts),
        "moe_layer_freq": [int(t == "sparse") for t in mlp],
        "moe_ffn_hidden_size": int(cfg.moe_intermediate_size),
        "moe_shared_expert_intermediate_size": int(cfg.moe_intermediate_size)
        * int(cfg.n_shared_experts),
        "moe_shared_expert_gate": False,
        "moe_shared_expert_overlap": False,
        "moe_grouped_gemm": True,
        "moe_router_topk": int(cfg.num_experts_per_tok),
        "moe_router_score_function": "sigmoid",
        "moe_router_pre_softmax": True,
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 0.0,
        "moe_router_topk_scaling_factor": float(cfg.routed_scaling_factor),
        "moe_router_dtype": "fp32",
        "moe_router_num_groups": None,
        "moe_router_group_topk": None,
        "moe_router_fusion": False,
        "moe_aux_loss_coeff": 0.0,
        "moe_apply_probs_on_input": False,
        "activation_func_clamp_value": float(cfg.swiglu_limit),
        "bias_activation_fusion": False,
        "use_te_activation_func": False,
        "mtp_num_layers": None,
        "mtp_loss_scaling_factor": None,
        "glm5next_hf_config": cfg,
        # DSV4's implementation supplies the identical two-site Sinkhorn math.
        "dsv4_hc_mult": int(cfg.hc_mult),
        "dsv4_hc_sinkhorn_iters": int(cfg.hc_sinkhorn_iters),
        "dsv4_hc_eps": float(cfg.hc_eps),
        "art_pipeline_activation_multiplier": int(cfg.hc_mult),
    }
    for name, value in values.items():
        setattr(provider, name, value)
