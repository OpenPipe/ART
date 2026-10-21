"""Independent per-token oracle for shared-outer and per-expert waves."""

from typing import Any, cast

import pytest
import torch

from art.megatron.segmented_expert_lora import (
    ExpertAdapterPlan,
    segmented_expert_lora,
)


def test_native_row_geometry_nested_scopes_and_gradients():
    from art.megatron.segmented_expert_lora import _row_forward
    from art.megatron.segmented_lora import (
        _CURRENT_LORA_SEGMENTS,
        LoRASegments,
        use_lora_segments,
    )

    counts = (3, 0, 5)
    segments = LoRASegments(cast(Any, ("a", "idle", "b")), counts)
    calls = []

    def inner(value):
        calls.append((len(value), _CURRENT_LORA_SEGMENTS.get()))
        return value * len(value), torch.ones(7), value + 1

    def outer(value):
        return _row_forward(inner, value)

    x = torch.randn(sum(counts), 1, 7, requires_grad=True)
    with use_lora_segments(segments):
        output, bias, normalized = _row_forward(outer, x)
        assert _CURRENT_LORA_SEGMENTS.get() is segments
    assert [n for n, _ in calls] == [3, 5]
    assert [scope.counts for _, scope in calls] == [(3, 0, 0), (0, 0, 5)]
    expected = torch.cat((x[:3] * 3, x[3:] * 5))
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(normalized, x + 1, atol=0, rtol=0)
    assert bias.shape == (7,)
    output.sum().backward()
    torch.testing.assert_close(
        x.grad, torch.cat((torch.full_like(x[:3], 3), torch.full_like(x[3:], 5)))
    )
    calls.clear()
    _row_forward(outer, x)
    assert calls == [(8, None)]


@pytest.mark.parametrize("shared", ["A", "B", "neither"])
@pytest.mark.parametrize("empty", [False, True])
def test_original_factor_gradients_against_per_token_dense_oracle(shared, empty):
    torch.manual_seed(17)
    counts = [0, 0, 0] if empty else [7, 0, 5]
    ids = torch.tensor(
        [] if empty else [2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 2, 1], dtype=torch.int64
    )
    refs = cast(Any, ("first", "second", "third", "idle", "missing"))
    plan = ExpertAdapterPlan.build(refs, ids, counts)
    x = torch.randn(sum(counts), 5, requires_grad=True)
    experts = torch.repeat_interleave(torch.arange(3), torch.tensor(counts))
    factors = [
        (
            torch.randn(1 if shared == "A" else 3, 5, rank, requires_grad=True),
            torch.randn(1 if shared == "B" else 3, rank, 7, requires_grad=True),
            scale,
        )
        for rank, scale in zip((1, 3, 8, 2), (0.75, 1.5, 0.25, 2.0), strict=True)
    ]
    cotangent = torch.randn(sum(counts), 7)
    # Fully dense per-row factors are an independent oracle, not the production
    # strategy. The implementation must preserve compact shared leaf gradients.
    outputs = []
    for row, (adapter, expert) in enumerate(
        zip(ids.tolist(), experts.tolist(), strict=True)
    ):
        a, b, scale = factors[adapter]
        outputs.append(
            x[row]
            @ a[0 if shared == "A" else expert]
            @ b[0 if shared == "B" else expert]
            * scale
        )
    if not empty:
        expected = torch.stack(outputs)
        expected.backward(cotangent)
        dx = cast(torch.Tensor, x.grad).clone()
        grads = [
            (cast(torch.Tensor, a.grad).clone(), cast(torch.Tensor, b.grad).clone())
            for a, b, _ in factors[:3]
        ]
        for a, b, _ in factors:
            a.grad = b.grad = None
        x.grad = None
    actual = segmented_expert_lora(x, factors + [None], plan, out_features=7)
    actual.backward(cotangent)
    assert x.grad is not None
    if not empty:
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(x.grad, dx, atol=2e-5, rtol=1e-5)
        for (a, b, _), (ga, gb) in zip(factors[:3], grads, strict=True):
            torch.testing.assert_close(a.grad, ga, atol=2e-5, rtol=1e-5)
            torch.testing.assert_close(b.grad, gb, atol=2e-5, rtol=1e-5)
    for a, b, _ in factors:
        assert a.grad is not None and a.grad.shape == a.shape
        assert b.grad is not None and b.grad.shape == b.shape
    for a, b, _ in factors[3:] if not empty else factors:
        assert not cast(torch.Tensor, a.grad).any()
        assert not cast(torch.Tensor, b.grad).any()


def test_plan_validates_dispatch_metadata_and_preserves_row_order():
    refs = cast(Any, ("a", "b"))
    ids = torch.tensor([1, 0, 1, 0])
    plan = ExpertAdapterPlan.build(refs, ids, [1, 3])
    values = torch.arange(4)
    assert torch.equal(values[plan.permutation][plan.inverse], values)
    assert plan.counts == ((0, 2), (1, 1))
    # External ID mutation cannot change a plan retained by autograd.
    ids.fill_(0)
    assert plan.counts == ((0, 2), (1, 1))
    for bad_ids, counts in (
        (torch.tensor([2]), [1]),
        (torch.tensor([0]), [-1]),
        (torch.tensor([0]), [2]),
        (torch.tensor([0.0]), [1]),
    ):
        with pytest.raises(ValueError):
            ExpertAdapterPlan.build(refs, bad_ids, counts)
    with pytest.raises(ValueError, match="rows"):
        segmented_expert_lora(torch.empty(5, 3), [None, None], plan, out_features=2)


def test_recompute_captures_original_tokens_and_dispatched_plan():
    from art.megatron.segmented_expert_lora import (
        _CURRENT_ADAPTER_TOKENS,
        _CURRENT_EXPERT_ADAPTER_PLAN,
        AdapterTokens,
        capture_expert_context,
        multi_adapter_tokens,
        use_expert_adapter_plan,
    )

    refs = cast(Any, ("a", "b"))
    tokens = AdapterTokens(refs, torch.tensor([0, 1, 1]))
    plan = ExpertAdapterPlan.build(refs, tokens.ids, [1, 2])
    with multi_adapter_tokens(tokens), use_expert_adapter_plan(plan):
        captured = capture_expert_context(
            lambda: (_CURRENT_ADAPTER_TOKENS.get(), _CURRENT_EXPERT_ADAPTER_PLAN.get())
        )
    other = AdapterTokens(refs, torch.tensor([1]))
    with multi_adapter_tokens(other):
        assert captured() == (tokens, plan)
        assert _CURRENT_ADAPTER_TOKENS.get() is other
    assert _CURRENT_ADAPTER_TOKENS.get() is None


def test_native_ep1_dispatch_replays_routing_without_modifying_base_result(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from art.megatron.segmented_expert_lora import (
        _CURRENT_EXPERT_ADAPTER_PLAN,
        AdapterTokens,
        _ep1_dispatch_postprocess,
        multi_adapter_tokens,
    )

    routing = torch.tensor(
        [[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=torch.bool
    )
    source_rows = routing.T.nonzero()[:, 1]
    calls = []

    def native_permute(
        values, actual_routing, *, probs, num_out_tokens, fused, drop_and_pad
    ):
        assert actual_routing is routing and fused and not drop_and_pad
        assert num_out_tokens == 7 and not values.requires_grad
        calls.append("permute IDs")
        selected = probs.T[routing.T]
        # TE masks payloads at zero-probability slots even with a true route.
        payload = values[source_rows].masked_fill(selected[:, None] == 0, 0)
        return payload, selected, source_rows, None, None

    module = ModuleType("megatron.core.transformer.moe.moe_utils")
    setattr(module, "permute", native_permute)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    dispatcher = SimpleNamespace(
        routing_map=routing,
        num_out_tokens=7,
        config=SimpleNamespace(moe_permute_fusion=True),
    )
    hidden = torch.randn(7, 5, requires_grad=True)
    probabilities = torch.randn(7, requires_grad=True)
    result = (hidden, torch.tensor([2, 2, 3]), probabilities)

    def original(x, p):
        assert x is hidden and p is probabilities
        calls.append("native postprocess")
        return result

    tokens = AdapterTokens(cast(Any, ("a", "b")), torch.tensor([0, 1, 0, 1]))
    with multi_adapter_tokens(tokens):
        assert (
            _ep1_dispatch_postprocess(dispatcher, original, hidden, probabilities)
            is result
        )
        plan = _CURRENT_EXPERT_ADAPTER_PLAN.get()
        assert plan is not None and plan.counts == ((1, 1, 2), (1, 1, 1))
    assert calls == ["native postprocess", "permute IDs"]
    assert _CURRENT_EXPERT_ADAPTER_PLAN.get() is None
    # Installation must preserve ordinary non-wave dispatch on the same model.
    calls.clear()
    assert (
        _ep1_dispatch_postprocess(dispatcher, original, hidden, probabilities) is result
    )
    assert calls == ["native postprocess"]


def test_installer_rejects_unsupported_transport_and_parallelism(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from art.megatron.segmented_expert_lora import install_segmented_dispatcher

    module = ModuleType("megatron.core.transformer.moe.token_dispatcher")
    setattr(
        module, "MoEAlltoAllTokenDispatcher", type("MoEAlltoAllTokenDispatcher", (), {})
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    dispatcher = SimpleNamespace()
    layer = SimpleNamespace(config=SimpleNamespace(), token_dispatcher=dispatcher)
    with pytest.raises(ValueError, match="unsupported native"):
        install_segmented_dispatcher(layer)
    assert layer.token_dispatcher is dispatcher
    layer.config.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="TP1/CP1/EP1"):
        install_segmented_dispatcher(layer)
