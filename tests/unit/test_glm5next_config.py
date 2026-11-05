from types import SimpleNamespace

import pytest
import torch

from art.megatron.glm5next.config import (
    SERVING_MODEL,
    TRAINING_MODEL,
    configure_glm5next_provider,
)
from art.megatron.glm5next.source import hf_weight_source, materialize_hf_weight
from art.megatron.model_support.registry import (
    UnsupportedModelArchitectureError,
    get_model_support_spec,
)


def production_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=45,
        layer_types=[
            "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
            for i in range(45)
        ],
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 42,
        indexer_types=["full"] * 45,
        hidden_size=4096,
        num_attention_heads=64,
        v_head_dim=256,
        intermediate_size=12288,
        vocab_size=154880,
        rms_norm_eps=1e-5,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        qk_nope_head_dim=256,
        mla_use_nope=True,
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        n_routed_experts=288,
        moe_intermediate_size=2048,
        n_shared_experts=1,
        num_experts_per_tok=8,
        routed_scaling_factor=2.5,
        swiglu_limit=10.0,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


def test_production_topology_and_frozen_router_contract() -> None:
    provider = SimpleNamespace()
    configure_glm5next_provider(
        provider, SimpleNamespace(text_config=production_config())
    )
    assert [
        i for i, linear in enumerate(provider.linear_attention_freq) if not linear
    ] == list(range(3, 45, 4))
    assert provider.moe_layer_freq == [0] * 3 + [1] * 42
    assert provider.num_moe_experts // 8 == 36
    assert provider.vocab_size == 154880
    assert provider.kv_channels == 256
    assert provider.share_embeddings_and_output_weights is False
    assert provider.mtp_num_layers is None
    assert (
        provider.moe_router_bias_update_rate
        == provider.moe_aux_loss_coeff
        == provider.dsa_indexer_loss_coeff
        == 0
    )
    assert provider.moe_router_dtype == "fp32"
    assert provider.moe_router_topk == 8
    assert provider.moe_router_topk_scaling_factor == 2.5
    assert provider.activation_func_clamp_value == 10
    assert not provider.bias_activation_fusion
    assert provider.dsv4_hc_mult == 4
    assert provider.dsv4_hc_sinkhorn_iters == 20


@pytest.mark.parametrize(
    "field,value", [("qk_rope_head_dim", 64), ("topk_method", "greedy"), ("n_group", 2)]
)
def test_reject_unsupported_architecture(field: str, value: object) -> None:
    config = production_config()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        configure_glm5next_provider(SimpleNamespace(), config)


@pytest.mark.parametrize("model", [SERVING_MODEL, TRAINING_MODEL])
def test_unqualified_model_requires_explicit_probe(model: str) -> None:
    with pytest.raises(UnsupportedModelArchitectureError):
        get_model_support_spec(model)
    spec = get_model_support_spec(model, allow_unvalidated_arch=True)
    assert spec.key == "glm5next"
    assert "lm_head" in spec.default_target_modules
    assert "q_b_proj" not in spec.default_target_modules
    assert "kv_b_proj" not in spec.default_target_modules


def test_reference_conv_and_forget_gate_load_released_checkpoint_names() -> None:
    prefix = "model.language_model.layers.0.self_attn."
    state = {
        f"{prefix}{name}_conv1d.weight": torch.full((2, 1, 4), float(i))
        for i, name in enumerate(("q", "k", "v"))
    }
    result = materialize_hf_weight(f"{prefix}conv1d.weight", state)
    assert result.shape == (6, 1, 4)
    assert result[:, 0, 0].tolist() == [0, 0, 1, 1, 2, 2]
    for index, part in enumerate(("q", "k", "v")):
        torch.testing.assert_close(
            materialize_hf_weight(
                f"{prefix}{part}_conv1d.weight", {f"{prefix}conv1d.weight": result}
            ),
            result[index * 2 : (index + 1) * 2],
        )
    gate = torch.randn(2, 8)
    assert (
        materialize_hf_weight(
            f"{prefix}f_a_proj.weight", {f"{prefix}forget_gate.f_a_proj.weight": gate}
        )
        is gate
    )
    assert (
        materialize_hf_weight(
            f"{prefix}forget_gate.f_a_proj.weight", {f"{prefix}f_a_proj.weight": gate}
        )
        is gate
    )


def test_routed_experts_materialize_from_hf_packed_layout() -> None:
    prefix = "model.language_model.layers.3.mlp.experts"
    packed = torch.arange(4 * 6 * 8).reshape(4, 6, 8)
    state = {f"{prefix}.gate_up_proj": packed}
    for expert in range(4):
        torch.testing.assert_close(
            materialize_hf_weight(f"{prefix}.{expert}.gate_proj.weight", state),
            packed[expert, :3],
        )
        torch.testing.assert_close(
            materialize_hf_weight(f"{prefix}.{expert}.up_proj.weight", state),
            packed[expert, 3:],
        )


def test_missing_source_is_an_error_and_aliases_are_explicit() -> None:
    name = "model.language_model.layers.0.hc_attn_fn"
    source = hf_weight_source(name)
    assert source.physical_key_options == (
        (name,),
        ("model.language_model.layers.0.attn_hc.fn",),
    )
    with pytest.raises(KeyError, match="No BF16 source"):
        materialize_hf_weight(name, {})
