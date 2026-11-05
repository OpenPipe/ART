"""Bounded HF workspace must preserve the original within-head equation."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bounded_kda_is_bitwise_equal_to_hf(dtype: torch.dtype) -> None:
    modeling = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.integration.megatron.glm5next_noise_floor import (
        bounded_cpu_kernels,
        cpu_reference_kernels,
    )

    torch.manual_seed(562)
    q, k, v = [torch.randn(1, 1100, 16, 8).to(dtype) for _ in range(3)]
    g, beta = -torch.rand_like(q), torch.rand(1, 1100, 16).to(dtype)
    with cpu_reference_kernels():
        expected, expected_state = modeling.chunk_kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
            output_final_state=True,
        )
    with bounded_cpu_kernels():
        actual, state = modeling.chunk_kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
            output_final_state=True,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()
