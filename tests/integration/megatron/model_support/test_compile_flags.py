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
