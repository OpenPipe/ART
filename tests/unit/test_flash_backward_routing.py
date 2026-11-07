"""Compiled backward routing and LSE units without CUDA or kernel execution."""

from collections import Counter, defaultdict
import importlib
import json
import logging
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.nn.attention.flex_attention import AuxRequest, create_block_mask

from art.megatron.flex_attn import compiled
from art.megatron.flex_attn import flash_dlse_patch as patch


@pytest.fixture
def lowering(monkeypatch):
    import torch._inductor.kernel.flex.flex_attention as kernel
    from torch._inductor.lowering import lowerings

    operation = torch.ops.higher_order.flex_attention_backward
    monkeypatch.setitem(lowerings, operation, lambda *args: args)
    monkeypatch.setitem(
        lowerings, torch.ops.aten.mul.Tensor, lambda x, scale: x * scale
    )
    monkeypatch.setattr(kernel, "flex_attention_backward", lowerings[operation])
    monkeypatch.setattr(patch, "_DETERMINISTIC_BWD_PATCH_APPLIED", False)
    monkeypatch.setattr(patch, "_FLASH_BACKWARD_OVERRIDE_LOGGED", set())
    monkeypatch.setattr(patch, "counters", defaultdict(Counter))
    patch.apply_deterministic_flash_backward_patch()
    # Fake only IR shape guards and the destination lowering. Conversion
    # arithmetic and the installed ART callback are the real implementations.
    graph = SimpleNamespace(sizevars=SimpleNamespace(guard_int=int))
    with kernel.V.set_graph_handler(cast(Any, graph)):
        yield lowerings[operation]


def _arguments(dim, backend, has_dlse):
    return dict(
        query=SimpleNamespace(get_size=lambda: (1, 4, 128, dim)),
        key=None,
        value=None,
        out=None,
        logsumexp=torch.tensor([1.0, 2.0]),
        grad_out=None,
        grad_logsumexp=torch.tensor([0.3, -0.2]) if has_dlse else None,
        fw_graph=None,
        joint_graph=None,
        block_mask=None,
        scale=0.125,
        kernel_options={"BACKEND": backend, "PRESCALE_QK": False},
        score_mod_other_buffers=(),
        mask_mod_other_buffers=(),
    )


@pytest.mark.parametrize("dim", [32, 64, 96, 128, 256])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("backend", ["FLASH", "TRITON"])
@pytest.mark.parametrize("has_dlse", [False, True])
def test_backward_decision_table_and_unit_conversion(
    lowering, monkeypatch, dim, deterministic, backend, has_dlse
):
    monkeypatch.setattr(
        torch, "are_deterministic_algorithms_enabled", lambda: deterministic
    )
    incoming = _arguments(dim, backend, has_dlse)
    expected_route = backend == "FLASH" and (deterministic or dim == 64)
    # Both call forms must route identically, preserving the caller's options.
    for keyword in [False, True]:
        actual = lowering(**incoming) if keyword else lowering(*incoming.values())
        assert actual[11]["BACKEND"] == ("TRITON" if expected_route else backend)
        assert actual[11]["PRESCALE_QK"] is False
        assert incoming["kernel_options"]["BACKEND"] == backend
        if expected_route:
            torch.testing.assert_close(
                actual[4], incoming["logsumexp"] * math.log2(math.e)
            )
            if has_dlse:
                torch.testing.assert_close(
                    actual[6], incoming["grad_logsumexp"] * math.log(2)
                )
            else:
                assert actual[6] is None
        else:
            assert actual[4] is incoming["logsumexp"]
            assert actual[6] is incoming["grad_logsumexp"]
            assert actual[11] is incoming["kernel_options"]
    override = backend == "FLASH" and not deterministic and dim == 64
    assert patch.counters["art"]["flex_flash_backward_override"] == (
        2 if override else 0
    )


def test_override_logs_once_with_bounded_state_and_keeps_policy(
    lowering, monkeypatch, caplog
):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)

    def reject_global_policy_change(*args, **kwargs):
        pytest.fail("Routing must not change the process-wide deterministic policy")

    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", reject_global_policy_change
    )
    with caplog.at_level(logging.WARNING):
        for _ in range(1000):
            actual = lowering(**_arguments(64, "FLASH", True))
            assert actual[11]["BACKEND"] == "TRITON"
        for dim in [32, 96, 128, 256]:
            lowering(**_arguments(dim, "FLASH", True))
    assert len(caplog.records) == 1
    event = json.loads(
        caplog.records[0].message.split("ART_FLEX_ATTENTION_EVENT ", 1)[1]
    )
    assert event["event"] == "flex_flash_backward_override"
    assert event["rank"] == "7" and event["head_dim"] == 64
    assert event["to_backend"] == "TRITON" and event["count"] == 1
    assert patch.counters["art"]["flex_flash_backward_override"] == 1000
    assert patch._FLASH_BACKWARD_OVERRIDE_LOGGED == {64}


def test_triton_inputs_are_not_inspected_or_converted(lowering):
    incoming = _arguments(64, "TRITON", True)
    incoming["query"] = object()
    actual = lowering(**incoming)
    assert actual[4] is incoming["logsumexp"]
    assert actual[6] is incoming["grad_logsumexp"]


def test_install_is_idempotent(lowering):
    from torch._inductor.lowering import lowerings

    patch.apply_deterministic_flash_backward_patch()
    assert lowerings[torch.ops.higher_order.flex_attention_backward] is lowering


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_forward_stays_flash_and_override_set_is_explicit(monkeypatch, dim):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert (
        compiled.flex_backend_for_head_dims(
            head_dim=dim, head_dim_v=dim, device=torch.device("cuda")
        )
        == "FLASH"
    )
    assert patch._FLASH_TRITON_BACKWARD_HEAD_DIMS == {64}


@pytest.mark.parametrize("dim", [32, 64, 96, 128, 256])
@pytest.mark.parametrize("backend", ["FLASH", "TRITON"])
def test_only_affected_hop_cache_inputs_are_versioned(dim, backend):
    options = {"BACKEND": backend, "PRESCALE_QK": False}
    actual = patch.flash_backward_compatibility_options(options, dim)
    if dim == 64 and backend == "FLASH":
        assert actual == {**options, patch._FLASH_BACKWARD_COMPATIBILITY_KEY: 1}
    else:
        assert actual is options
    assert options == {"BACKEND": backend, "PRESCALE_QK": False}


@pytest.mark.parametrize("deterministic", [False, True])
def test_cache_version_is_removed_before_kernel_selection(
    lowering, monkeypatch, deterministic
):
    monkeypatch.setattr(
        torch, "are_deterministic_algorithms_enabled", lambda: deterministic
    )
    incoming = _arguments(64, "FLASH", True)
    options = dict(
        patch.flash_backward_compatibility_options(incoming["kernel_options"], 64)
    )
    incoming["kernel_options"] = options
    actual = lowering(**incoming)
    assert actual[11] == {"BACKEND": "TRITON", "PRESCALE_QK": False}
    assert options[patch._FLASH_BACKWARD_COMPATIBILITY_KEY] == 1


@pytest.mark.parametrize("entry", ["forced_dense", "sparse", "dense"])
@pytest.mark.parametrize("dim", [64, 128, 256])
def test_capture_versions_affected_cache_without_graph_breaks(monkeypatch, entry, dim):
    # Explain captures real HOP inputs without running CUDA or the lowering.
    flex = importlib.import_module("torch.nn.attention.flex_attention")
    monkeypatch.setattr(flex, "_validate_device", lambda *args: None)
    torch._dynamo.reset()
    q = torch.ones(1, 1, 2, dim, requires_grad=True)
    k = torch.ones(1, 1, 1, dim, requires_grad=True)
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=2, KV_LEN=1, device="cpu"
    )
    if entry == "sparse":
        fn = compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"})
        extra = {}
    else:
        fn = (
            compiled._forced_flex_attention_dense
            if entry == "forced_dense"
            else compiled._flex_attention_with_options({"BACKEND": "FLASH"})
        )
        extra = {"return_aux": AuxRequest(lse=True)}
    try:
        explanation = torch._dynamo.explain(fn)(
            q, k, k, block_mask=mask, scale=dim**-0.5, enable_gqa=False, **extra
        )
        assert explanation.graph_count == 1
        assert explanation.graph_break_count == 0, explanation.break_reasons
        hops = [
            node
            for node in explanation.graphs[0].graph.nodes
            if node.target is torch.ops.higher_order.flex_attention
        ]
        assert len(hops) == 1
        options = hops[0].args[6]
        assert options["BACKEND"] == "FLASH"
        assert options.get(patch._FLASH_BACKWARD_COMPATIBILITY_KEY) == (
            1 if dim == 64 else None
        )
    finally:
        torch._dynamo.reset()
