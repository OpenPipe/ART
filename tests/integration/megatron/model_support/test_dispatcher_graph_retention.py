from __future__ import annotations

import gc
import inspect
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

pytest.importorskip("megatron.bridge")

from megatron.core.tensor_parallel import random as mcore_random
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

from art.megatron.runtime.bridge_runtime import _patch_moe_dispatcher_graph_retention

_UPSTREAM_DISPATCH = inspect.unwrap(MoEAlltoAllTokenDispatcher.dispatch_preprocess)


class _CpuDispatcher(MoEAlltoAllTokenDispatcher):
    def __init__(self):
        # Exercise the real dispatcher/permute without CUDA streams or collectives.
        self.config = SimpleNamespace(
            moe_router_padding_for_quantization=False, moe_permute_fusion=False
        )
        self.shared_experts = None
        self.drop_and_pad = False
        self.num_out_tokens = 22  # Eleven tokens, two selected experts each.

    def preprocess(self, routing_map):
        return routing_map.sum(0)

    def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
        assert tokens_per_expert is not None
        return tokens_per_expert


def _run_checkpointed_router(backend):
    torch.manual_seed(47)
    initial = torch.randn(1, 11, 8, dtype=torch.float64, requires_grad=True)
    weights = [
        torch.randn(8, 4, dtype=torch.float64, requires_grad=True) for _ in range(4)
    ]
    dispatchers = [_CpuDispatcher() for _ in weights]
    inputs = []

    def layer(index):
        def compute(value):
            probs = (value.reshape(-1, 8) @ weights[index]).softmax(-1)
            routing = torch.zeros_like(probs, dtype=torch.bool)
            routing.scatter_(1, probs.topk(2, dim=-1).indices, True)
            dispatcher = dispatchers[index]
            routed, routed_probs = dispatcher.dispatch_preprocess(value, routing, probs)
            transformed = routed.tanh() * routed_probs[:, None]
            merged = torch.zeros_like(value.reshape(-1, 8)).index_add(
                0, dispatcher.reversed_local_input_permutation_mapping, transformed
            )
            return value + 0.2 * merged.reshape_as(value)

        execute = (
            compute if backend is None else torch.compile(compute, backend=backend)
        )

        def forward(value):
            if torch.is_grad_enabled():
                assert value.is_leaf
                inputs.append(weakref.ref(value))
            return execute(value)

        return forward

    hidden = initial
    for index in range(len(weights)):
        hidden = mcore_random.CheckpointFunction.apply(layer(index), False, hidden)
    loss = hidden.square().sum()
    loss.backward()
    gc.collect()
    assert len(inputs) == len(weights)
    alive = [reference() is not None for reference in inputs]
    for reference in inputs:
        if (value := reference()) is not None:
            assert value.grad is not None
    del value
    gradients = []
    for value in [initial, *weights]:
        assert value.grad is not None
        gradients.append(value.grad.clone())
    # Keep the loss/output alive: clearing only the cache must release the leaves.
    cache_sizes = [dispatcher.probs.numel() for dispatcher in dispatchers]
    for dispatcher in dispatchers:
        assert dispatcher.probs.dtype == initial.dtype
        assert dispatcher.probs.device == initial.device
        dispatcher.probs = None
    gc.collect()
    assert all(reference() is None for reference in inputs)
    return loss.detach(), gradients, alive, cache_sizes


@pytest.mark.parametrize("compiled", [False, True])
def test_dispatcher_cache_does_not_retain_checkpoint_inputs(monkeypatch, compiled):
    # The imported MCore checkpoint implementation runs unchanged except CPU RNG.
    monkeypatch.setattr(
        mcore_random, "_get_all_rng_states", lambda: (torch.get_rng_state(),)
    )
    monkeypatch.setattr(
        mcore_random, "_set_all_rng_states", lambda state: torch.set_rng_state(state)
    )
    monkeypatch.setattr(
        MoEAlltoAllTokenDispatcher, "dispatch_preprocess", _UPSTREAM_DISPATCH
    )
    backend = CompileCounterWithBackend("aot_eager") if compiled else None
    try:
        reference_loss, reference_grads, retained, sizes = _run_checkpointed_router(
            backend
        )
        assert all(retained)
        assert sizes == [44] * 4
        # Production installs runtime patches before compiling any model graph.
        torch.compiler.reset()
        _patch_moe_dispatcher_graph_retention()
        patched = MoEAlltoAllTokenDispatcher.dispatch_preprocess
        _patch_moe_dispatcher_graph_retention()
        assert MoEAlltoAllTokenDispatcher.dispatch_preprocess is patched
        loss, gradients, retained, sizes = _run_checkpointed_router(backend)
        assert not any(retained)
        assert sizes == [0] * 4
        assert torch.equal(loss, reference_loss)
        for actual, expected in zip(gradients, reference_grads, strict=True):
            assert torch.equal(actual, expected)
        if backend is not None:
            assert backend.frame_count > 0
    finally:
        torch.compiler.reset()
