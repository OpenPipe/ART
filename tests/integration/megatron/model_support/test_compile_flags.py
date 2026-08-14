import torch

from art.megatron.flex_attn.compiled import _needs_blackwell_wide_head_tile
from art.megatron.model_support.handlers.gemma4 import (
    GEMMA4_DENSE_HANDLER,
    GEMMA4_MOE_HANDLER,
)


def test_wide_head_tile_workaround_is_blackwell_only(monkeypatch) -> None:
    def selected(major: int) -> bool:
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda _device: (major, 0)
        )
        return _needs_blackwell_wide_head_tile(
            backend="TRITON",
            head_dim=512,
            head_dim_v=512,
            triton_num_stages_2_head_dims=(512,),
            device=torch.device("cuda"),
        )

    assert selected(10)
    assert not selected(11)


def test_gemma4_wide_global_attention_uses_lower_triton_stage_count() -> None:
    provider = type(
        "Provider",
        (),
        {
            "global_head_dim": 512,
            "hidden_size": 5376,
            "kv_channels": 256,
            "num_attention_heads": 32,
            "num_layers": 12,
        },
    )()

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
