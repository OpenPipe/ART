from typing import Any

from art.megatron.flex_attn import compiled as compiled_flex_attention
from art.megatron.model_support.handlers.gemma4 import (
    GEMMA4_DENSE_HANDLER,
    GEMMA4_MOE_HANDLER,
    _install_gemma4_flex_core_attention_wrapper,
)
from art.megatron.model_support.handlers.qwen3_5 import QWEN3_5_MOE_HANDLER
from art.megatron.model_support.handlers.qwen3_moe import QWEN3_MOE_HANDLER

_QWEN3_MOE_COMPILE_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "te_triton_permute_with_mask_map",
)
_QWEN35_MOE_COMPILE_FLAGS = (
    "alltoall_dtoh",
    "alltoall_dispatch_preprocess",
    "deepep_dispatch_combine",
    "deepep_permute_restore",
    "flex_token_dispatch_combine",
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


def _gemma4_provider(**overrides: int) -> Any:
    attrs = {
        "num_moe_experts": 128,
        "num_layers": 30,
        "hidden_size": 2816,
        "num_attention_heads": 16,
        "kv_channels": 256,
        "global_head_dim": 512,
        "num_global_key_value_heads": 2,
    }
    attrs.update(overrides)
    return type("Provider", (), attrs)()


def test_gemma4_known_wide_global_attention_signatures_use_lower_triton_stage_count() -> (
    None
):
    dense_provider = _gemma4_provider(
        num_moe_experts=0,
        num_layers=60,
        hidden_size=5376,
        num_attention_heads=32,
        num_global_key_value_heads=4,
    )
    moe_provider = _gemma4_provider()

    assert GEMMA4_DENSE_HANDLER.flex_attention_compile_crash_config(
        dense_provider
    ).triton_num_stages_2_head_dims == (512,)
    assert GEMMA4_MOE_HANDLER.flex_attention_compile_crash_config(
        moe_provider
    ).triton_num_stages_2_head_dims == (512,)


def test_gemma4_unlisted_wide_global_attention_signature_keeps_default_stage_count() -> (
    None
):
    provider = _gemma4_provider(hidden_size=2817)

    assert (
        GEMMA4_MOE_HANDLER.flex_attention_compile_crash_config(
            provider
        ).triton_num_stages_2_head_dims
        == ()
    )


def test_gemma4_flex_attention_wrapper_carries_provider_compile_crash_config() -> None:
    provider = _gemma4_provider()
    provider.art_flex_compile_crash_config = (
        GEMMA4_MOE_HANDLER.flex_attention_compile_crash_config(provider)
    )
    _install_gemma4_flex_core_attention_wrapper(provider)

    class BaseAttention:
        def __init__(self, config: Any, layer_number: int) -> None:
            del layer_number
            self.head_dims = (
                config.art_flex_compile_crash_config.triton_num_stages_2_head_dims
            )

    copied_config = type("CopiedConfig", (), {})()
    wrapped_cls = provider.art_flex_core_attention_wrapper(copied_config, BaseAttention)
    wrapped = wrapped_cls(type("LayerConfig", (), {})(), 1)

    assert wrapped.head_dims == (512,)


def test_triton_num_stages_2_selection_overrides_forced_triton_backend(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(compiled_flex_attention, "_FORCED_FLEX_BACKEND", "TRITON")

    assert (
        compiled_flex_attention.get_dense_compiled_flex_attention(
            backend="TRITON",
            head_dim=512,
            head_dim_v=512,
            triton_num_stages_2_head_dims=(512,),
        )
        is compiled_flex_attention.triton_num_stages_2_dense_compiled_flex_attention
    )
    assert (
        compiled_flex_attention.get_sparse_compiled_flex_attention(
            family_key="test",
            backend="TRITON",
            head_dim=512,
            head_dim_v=512,
            triton_num_stages_2_head_dims=(512,),
        )
        is compiled_flex_attention.triton_num_stages_2_sparse_compiled_flex_attention
    )
