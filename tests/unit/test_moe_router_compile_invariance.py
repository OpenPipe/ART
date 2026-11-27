"""Tests for MoE TopKRouter.routing compile workaround and cross-rank invariance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from art.megatron import compile_workarounds
from art.megatron.model_support.handlers.default_dense import (
    DefaultDenseHandler,
    DefaultMoeHandler,
    _compile_workaround_flags_for_provider,
)
from art.megatron.model_support.handlers.dsv4 import DSV4_HANDLER
from art.megatron.model_support.handlers.gemma4 import (
    GEMMA4_DENSE_HANDLER,
    GEMMA4_MOE_HANDLER,
)
from art.megatron.model_support.handlers.glm52 import GLM52_HANDLER
from art.megatron.model_support.handlers.gpt_oss import GPT_OSS_MOE_HANDLER
from art.megatron.model_support.handlers.nemotron_h import NEMOTRON_H_HANDLER
from art.megatron.model_support.handlers.qwen3_5 import (
    QWEN3_5_DENSE_HANDLER,
    QWEN3_5_MOE_HANDLER,
)
from art.megatron.model_support.handlers.qwen3_moe import QWEN3_MOE_HANDLER
from art.megatron.model_support.spec import CompileWorkaroundConfig


def _moe_provider(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {
        "num_moe_experts": 256,
        "context_parallel_size": 4,
        "expert_model_parallel_size": 4,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "moe_shared_expert_overlap": False,
        "moe_token_dispatcher_type": "flex",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _dense_provider(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {
        "num_moe_experts": None,
        "context_parallel_size": 4,
        "expert_model_parallel_size": 1,
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "moe_shared_expert_overlap": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_moe_handlers_enable_moe_router_routing_compile_workaround() -> None:
    moe_provider = _moe_provider()
    for handler in (
        DefaultMoeHandler(),
        QWEN3_5_MOE_HANDLER,
        QWEN3_MOE_HANDLER,
        GPT_OSS_MOE_HANDLER,
        NEMOTRON_H_HANDLER,
        GEMMA4_MOE_HANDLER,
        DSV4_HANDLER,
        GLM52_HANDLER,
    ):
        cfg = handler.compile_workaround_config(moe_provider)
        assert "moe_router_routing" in cfg.flags, (
            f"{handler.key} must include 'moe_router_routing' compile workaround"
        )

    assert (
        "gemma4_moe_router_routing"
        in GEMMA4_MOE_HANDLER.compile_workaround_config(moe_provider).flags
    )
    assert (
        "dsv4_moe_router_routing"
        in DSV4_HANDLER.compile_workaround_config(moe_provider).flags
    )

    dense_provider = _dense_provider()
    for dense_handler in (
        DefaultDenseHandler(),
        QWEN3_5_DENSE_HANDLER,
        GEMMA4_DENSE_HANDLER,
    ):
        cfg = dense_handler.compile_workaround_config(dense_provider)
        assert "moe_router_routing" not in cfg.flags

    assert "moe_router_routing" not in _compile_workaround_flags_for_provider(
        _dense_provider(num_moe_experts=0)
    )


def test_install_moe_router_routing_workaround_disables_compile_and_matches_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe import router as moe_router
    from megatron.core.transformer.transformer_config import TransformerConfig

    orig_routing = moe_router.TopKRouter.routing
    monkeypatch.setattr(compile_workarounds, "_INSTALLED_CONFIG", None)
    monkeypatch.setattr(moe_router.TopKRouter, "routing", orig_routing)

    install_cfg = CompileWorkaroundConfig(flags=("moe_router_routing",))
    compile_workarounds.install_torch_compile_workarounds(install_cfg)
    patched_routing = moe_router.TopKRouter.routing
    assert getattr(patched_routing, "__art_compile_disabled__", False) is True

    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        num_moe_experts=16,
        moe_router_topk=4,
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_dtype="fp32",
        moe_router_fusion=False,
        moe_aux_loss_coeff=0.0,
        add_bias_linear=False,
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.tp_cp = groups.tp_dp_cp = None
    router = moe_router.TopKRouter(config, pg_collection=groups)
    router.set_layer_number(1)

    torch.manual_seed(7)
    logits = torch.randn((3, 1, 16), dtype=torch.float32) * 3.0
    expected_probs, expected_map = orig_routing(router, logits)

    # Both bound (`router.routing(...)`) and unbound (`TopKRouter.routing(router, ...)`)
    # calls must match eager bit-for-bit even inside torch.compile.
    torch._dynamo.reset()
    try:

        def _compiled_bound(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return router.routing(x)

        def _compiled_unbound(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return moe_router.TopKRouter.routing(router, x)

        compiled_bound = torch.compile(_compiled_bound)
        compiled_unbound = torch.compile(_compiled_unbound)
        bound_probs, bound_map = compiled_bound(logits)
        unbound_probs, unbound_map = compiled_unbound(logits)
    finally:
        torch._dynamo.reset()

    assert torch.equal(bound_probs, expected_probs)
    assert torch.equal(bound_map, expected_map)
    assert torch.equal(unbound_probs, expected_probs)
    assert torch.equal(unbound_map, expected_map)


def test_install_dsv4_moe_router_routing_workaround_disables_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.dsv4 import layer as dsv4_layer

    orig_dsv4_routing = dsv4_layer.Dsv4Router.routing
    monkeypatch.setattr(compile_workarounds, "_INSTALLED_CONFIG", None)
    monkeypatch.setattr(dsv4_layer.Dsv4Router, "routing", orig_dsv4_routing)

    compile_workarounds.install_torch_compile_workarounds(
        CompileWorkaroundConfig(flags=("dsv4_moe_router_routing",))
    )
    assert (
        getattr(dsv4_layer.Dsv4Router.routing, "__art_compile_disabled__", False)
        is True
    )


@pytest.mark.gpu
def test_moe_router_routing_workaround_eliminates_cross_rank_size_hint_drift_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Inductor persistent-reduction invariance test")

    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe import router as moe_router
    from megatron.core.transformer.transformer_config import TransformerConfig

    orig_routing = moe_router.TopKRouter.routing
    monkeypatch.setattr(compile_workarounds, "_INSTALLED_CONFIG", None)
    monkeypatch.setattr(moe_router.TopKRouter, "routing", orig_routing)

    compile_workarounds.install_torch_compile_workarounds(
        CompileWorkaroundConfig(flags=("moe_router_routing",))
    )

    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        num_moe_experts=256,
        moe_router_topk=8,
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_dtype="fp32",
        moe_router_fusion=False,
        moe_aux_loss_coeff=0.0,
        add_bias_linear=False,
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.tp_cp = groups.tp_dp_cp = None
    router = moe_router.TopKRouter(config, pg_collection=groups).cuda()
    router.set_layer_number(1)

    torch.manual_seed(2026)
    logits_3 = torch.randn((3, 1, 256), device="cuda", dtype=torch.float32) * 3.0
    logits_1 = torch.randn((1, 1, 256), device="cuda", dtype=torch.float32) * 3.0
    logits_0 = torch.empty((0, 1, 256), device="cuda", dtype=torch.float32)
    logits_256 = torch.randn((256, 1, 256), device="cuda", dtype=torch.float32) * 3.0

    eager_probs, eager_map = orig_routing(router, logits_3)

    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(assume_static_by_default=False):
            # Simulate rank 0 (sees S=256 first, then S=3) vs rank 1 (sees S=0, S=1, then S=3).
            rank0_fn = torch.compile(lambda x: router.routing(x))
            rank1_fn = torch.compile(lambda x: router.routing(x))

            rank0_fn(logits_256)
            r0_probs, r0_map = rank0_fn(logits_3)

            rank1_fn(logits_0)
            rank1_fn(logits_1)
            r1_probs, r1_map = rank1_fn(logits_3)
    finally:
        torch._dynamo.reset()

    assert torch.equal(r0_probs, eager_probs)
    assert torch.equal(r1_probs, eager_probs)
    assert torch.equal(r0_probs, r1_probs)
    assert torch.equal(r0_map, eager_map)
    assert torch.equal(r1_map, eager_map)
