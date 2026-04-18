from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("megatron.bridge")

import torch

from art.megatron.lora import SelfAttentionLinearQKVLoRA
from art.megatron.train import _canonical_art_param_name


def test_canonical_art_param_name_strips_art_wrapper_segments() -> None:
    assert (
        _canonical_art_param_name(
            "module.language_model.decoder.layers.0.self_attention.out_proj.linear_proj.weight"
        )
        == "language_model.decoder.layers.0.self_attention.out_proj.weight"
    )
    assert (
        _canonical_art_param_name(
            "module.language_model.decoder.layers.0.mlp.linear_fc2.row_parallel_lora.linear_proj.weight"
        )
        == "language_model.decoder.layers.0.mlp.linear_fc2.weight"
    )


def test_self_attention_linear_qkv_lora_accepts_nongated_qwen3_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "art.megatron.lora.ps.get_tensor_model_parallel_world_size", lambda: 1
    )
    provider: Any = SimpleNamespace(
        kv_channels=128,
        num_query_groups=4,
        num_attention_heads=32,
        attention_output_gate=False,
    )
    q_out_features = provider.kv_channels * provider.num_attention_heads
    kv_out_features = provider.kv_channels * provider.num_query_groups
    linear_qkv: Any = SimpleNamespace(
        weight=torch.empty(q_out_features + 2 * kv_out_features, 16),
        in_features=16,
        return_layernorm_output=False,
        return_layernorm_output_gathered=False,
    )

    wrapped = SelfAttentionLinearQKVLoRA(
        adapter_model_prefix="base_model.model.model.layers.0.self_attn",
        linear_qkv=cast(Any, linear_qkv),
        rank=4,
        alpha=8.0,
        provider=cast(Any, provider),
    )

    assert wrapped.attention_output_gate is False
    assert wrapped.q_proj_lora.B_T.shape[-1] == q_out_features
