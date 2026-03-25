import pytest

pytest.importorskip("trl")
pytest.importorskip("vllm")

from art.unsloth.service import _normalize_merged_checkpoint_name


def test_normalize_merged_checkpoint_name_strips_peft_wrapper_segments():
    assert (
        _normalize_merged_checkpoint_name(
            "model.language_model.layers.3.self_attn.q_proj.base_layer.weight"
        )
        == "model.language_model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        _normalize_merged_checkpoint_name(
            "model.language_model.layers.3.mlp.shared_expert.gate_proj.base_layer.weight"
        )
        == "model.language_model.layers.3.mlp.shared_expert.gate_proj.weight"
    )
    assert (
        _normalize_merged_checkpoint_name(
            "model.language_model.layers.3.mlp.experts.base_layer.base_layer.gate_up_proj"
        )
        == "model.language_model.layers.3.mlp.experts.gate_up_proj"
    )
    assert (
        _normalize_merged_checkpoint_name(
            "model.language_model.layers.3.mlp.experts.base_layer.base_layer.down_proj"
        )
        == "model.language_model.layers.3.mlp.experts.down_proj"
    )


def test_normalize_merged_checkpoint_name_strips_peft_prefix():
    assert (
        _normalize_merged_checkpoint_name(
            "base_model.model.model.language_model.layers.7.self_attn.o_proj.base_layer.weight"
        )
        == "model.language_model.layers.7.self_attn.o_proj.weight"
    )
    assert (
        _normalize_merged_checkpoint_name("base_model.model.lm_head.weight")
        == "lm_head.weight"
    )


def test_normalize_merged_checkpoint_name_leaves_regular_names_unchanged():
    assert (
        _normalize_merged_checkpoint_name(
            "model.language_model.layers.3.self_attn.q_norm.weight"
        )
        == "model.language_model.layers.3.self_attn.q_norm.weight"
    )
