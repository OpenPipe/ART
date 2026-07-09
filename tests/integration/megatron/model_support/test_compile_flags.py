from art.megatron.model_support.handlers.gemma4 import (
    GEMMA4_DENSE_HANDLER,
    GEMMA4_MOE_HANDLER,
)
from art.megatron.model_support.handlers.qwen3_5 import QWEN3_5_MOE_HANDLER
from art.megatron.model_support.handlers.qwen3_moe import QWEN3_MOE_HANDLER

_GEMMA4_MOE_COMPILE_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "flex_token_dispatch_combine",
    "gemma4_moe_postprocess",
    "moe_postprocess",
    "te_triton_permute_with_mask_map",
)
_QWEN3_MOE_COMPILE_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "moe_postprocess",
    "te_triton_permute_with_mask_map",
)
_QWEN35_MOE_COMPILE_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "flex_token_dispatch_combine",
    "moe_postprocess",
    "te_triton_permute_with_mask_map",
    "weighted_bias_swiglu_no_inner_forward_cast",
)


def test_qwen3_moe_compile_workarounds_cover_deepep_permute_restore() -> None:
    provider = type("Provider", (), {"context_parallel_size": 1})()
    config = QWEN3_MOE_HANDLER.compile_workaround_config(provider)
    assert config.flags == _QWEN3_MOE_COMPILE_FLAGS
    assert config.unconditional_flags == ()


def test_qwen35_moe_compile_workarounds_cover_deepep_permute_restore() -> None:
    provider = type("Provider", (), {"moe_shared_expert_overlap": False})()
    config = QWEN3_5_MOE_HANDLER.compile_workaround_config(provider)
    assert config.flags == _QWEN35_MOE_COMPILE_FLAGS
    assert config.unconditional_flags == ()


def test_gemma4_wide_global_attention_uses_lower_triton_stage_count() -> None:
    provider = type("Provider", (), {"global_head_dim": 512})()

    assert GEMMA4_DENSE_HANDLER.flex_attention_compile_crash_config(
        provider
    ).triton_num_stages_2_head_dims == (512,)
    assert GEMMA4_MOE_HANDLER.flex_attention_compile_crash_config(
        provider
    ).triton_num_stages_2_head_dims == (512,)


def test_gemma4_standard_global_attention_keeps_default_triton_stage_count() -> None:
    provider = type("Provider", (), {"global_head_dim": 256})()

    assert (
        GEMMA4_DENSE_HANDLER.flex_attention_compile_crash_config(
            provider
        ).triton_num_stages_2_head_dims
        == ()
    )
    assert (
        GEMMA4_MOE_HANDLER.flex_attention_compile_crash_config(
            provider
        ).triton_num_stages_2_head_dims
        == ()
    )


def test_gemma4_moe_compile_workarounds_cover_moe_postprocess() -> None:
    provider = type("Provider", (), {"moe_shared_expert_overlap": False})()
    config = GEMMA4_MOE_HANDLER.compile_workaround_config(provider)
    assert config.flags == _GEMMA4_MOE_COMPILE_FLAGS
    assert config.unconditional_flags == ()


def test_gemma4_moe_postprocess_workaround_disables_bridge_override(
    monkeypatch,
) -> None:
    from megatron.bridge.models.gemma import gemma4_provider

    from art.megatron.compile_workarounds import (
        _install_gemma4_moe_postprocess_workaround,
    )

    original = gemma4_provider.Gemma4MoELayer.postprocess
    try:
        _install_gemma4_moe_postprocess_workaround()
        assert getattr(
            gemma4_provider.Gemma4MoELayer.postprocess,
            "__art_compile_disabled__",
            False,
        )
    finally:
        monkeypatch.setattr(gemma4_provider.Gemma4MoELayer, "postprocess", original)


def test_gemma4_attention_forward_patch_uses_precomputed_rotary_index(
    monkeypatch,
) -> None:
    from megatron.bridge.models.gemma import gemma4_provider

    import art.megatron.model_support.handlers.gemma4 as gemma4_handler

    original_init = gemma4_provider.Gemma4SelfAttention.__init__
    original_forward = gemma4_provider.Gemma4SelfAttention.forward
    monkeypatch.setattr(gemma4_handler, "_GEMMA4_ATTENTION_FORWARD_PATCHED", False)
    try:
        gemma4_handler._patch_gemma4_attention_forward_rotary_selection()
        patched_forward = gemma4_provider.Gemma4SelfAttention.forward
        assert patched_forward is not original_forward
        assert "layer_number" not in patched_forward.__code__.co_names
        assert "_art_gemma4_rotary_pos_emb_index" in patched_forward.__code__.co_names
    finally:
        monkeypatch.setattr(
            gemma4_provider.Gemma4SelfAttention, "__init__", original_init
        )
        monkeypatch.setattr(
            gemma4_provider.Gemma4SelfAttention,
            "forward",
            original_forward,
        )
