"""Regressions for BF16 conversions found by the F4 CPU localization."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig

from art.megatron.glm5next.layer import Glm5NextRouter
from art.megatron.runtime.bridge_runtime import _wrap_with_mp_wrapper


def test_router_correction_bias_survives_actual_mixed_precision_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only redirect MCore's eager counter/buffer allocation; no router or
    # mixed-precision constructor/math is replaced, and CUDA stays uninitialized.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.0,
        moe_router_topk_scaling_factor=2.5,
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.0,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        perform_initialization=False,
    )
    router = Glm5NextRouter(
        config,
        pg_collection=cast(
            Any, SimpleNamespace(tp=None, cp=None, tp_cp=None, tp_dp_cp=None)
        ),
    )
    source_bias = torch.tensor([7.95, 7.865, 7.879, 7.0], dtype=torch.float32)
    assert isinstance(router.expert_bias, torch.Tensor)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[1, 0] = 0.04
        router.expert_bias.copy_(source_bias)
        _, before = router(torch.ones(1, 8, dtype=torch.bfloat16))
    assert not torch.equal(source_bias, source_bias.bfloat16().float())
    old_router = deepcopy(router)
    setattr(old_router, "_keep_fp32_buffers", ())
    _wrap_with_mp_wrapper([old_router], config, Float16Module)
    with torch.no_grad():
        _, old_routes = old_router(torch.ones(1, 8, dtype=torch.bfloat16))
    # The old path has the right dtype after forward and the wrong values/IDs.
    assert isinstance(old_router.expert_bias, torch.Tensor)
    assert old_router.expert_bias.dtype == torch.float32
    assert not torch.equal(old_router.expert_bias, source_bias)
    assert not torch.equal(old_routes, before)
    wrapped = _wrap_with_mp_wrapper([router], config, Float16Module)
    assert wrapped[0].module is router
    # Checking dtype alone misses the original failure: MCore's forward
    # re-promotes already-rounded BF16 bias values to FP32.
    assert isinstance(router.expert_bias, torch.Tensor)
    assert router.expert_bias.dtype == torch.float32
    assert torch.equal(router.expert_bias, source_bias)
    assert router.weight.dtype == torch.float32
    with torch.no_grad():
        probs, after = router(torch.ones(1, 8, dtype=torch.bfloat16))
    assert torch.equal(after, before)
    expected = torch.tensor([[True, False, True, False]])
    assert torch.equal(after, expected)
    torch.testing.assert_close(probs[expected], torch.full((2,), 1.25))
    assert not torch.cuda.is_initialized()
