from art_vllm_runtime.qwen35_patches import (
    _select_blackwell_gdn_prefill_backend,
)


def test_blackwell_qwen35_auto_uses_triton_gdn_prefill() -> None:
    assert (
        _select_blackwell_gdn_prefill_backend(
            model_type="qwen3_5_moe_text",
            requested="auto",
            active="flashinfer",
            is_sm10x=True,
        )
        == "triton"
    )


def test_gdn_prefill_override_is_narrow() -> None:
    cases = (
        ("qwen3_5_moe_text", "cutedsl", "cutedsl", True, "cutedsl"),
        ("qwen3_5_moe_text", "auto", "flashinfer", False, "flashinfer"),
        ("qwen3_moe", "auto", "flashinfer", True, "flashinfer"),
        ("qwen3_5_moe_text", "triton", "triton", True, "triton"),
    )

    for model_type, requested, active, is_sm10x, expected in cases:
        assert (
            _select_blackwell_gdn_prefill_backend(
                model_type=model_type,
                requested=requested,
                active=active,
                is_sm10x=is_sm10x,
            )
            == expected
        )
