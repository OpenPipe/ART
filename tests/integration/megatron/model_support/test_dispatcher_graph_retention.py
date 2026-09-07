from __future__ import annotations

from copy import deepcopy
from functools import partial
import gc
import pickle
from types import SimpleNamespace
from typing import Any, cast
import weakref

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

pytest.importorskip("megatron.bridge")

from megatron.core.tensor_parallel import random as mcore_random
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)

from art.trainer_rank import TrainerRank
from art.trainer_rank._impl import _configure_moe_dispatcher_caches


def _preprocess(self, routing_map):
    return routing_map.sum(0)


def _synchronize(self, point, tokens_per_expert=None):
    assert tokens_per_expert is not None
    return tokens_per_expert


def _dispatcher() -> Any:
    # Keep the exact upstream type and dispatch/permute implementation, replacing
    # only CUDA/distributed initialization and metadata transfers for this CPU test.
    dispatcher: Any = object.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = SimpleNamespace(
        moe_router_padding_for_quantization=False, moe_permute_fusion=False
    )
    dispatcher.shared_experts = None
    dispatcher.drop_and_pad = False
    dispatcher.num_out_tokens = 22
    dispatcher.preprocess = partial(_preprocess, dispatcher)
    dispatcher._maybe_dtoh_and_synchronize = partial(_synchronize, dispatcher)
    return dispatcher


class _RouterLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float64))
        self.token_dispatcher = _dispatcher()

    def forward(self, value):
        probs = (value.reshape(-1, 8) @ self.weight).softmax(-1)
        routing = torch.zeros_like(probs, dtype=torch.bool)
        routing.scatter_(1, probs.topk(2, dim=-1).indices, True)
        dispatcher = self.token_dispatcher
        routed, routed_probs = dispatcher.dispatch_preprocess(value, routing, probs)
        transformed = routed.tanh() * routed_probs[:, None]
        merged = torch.zeros_like(value.reshape(-1, 8)).index_add(
            0, dispatcher.reversed_local_input_permutation_mapping, transformed
        )
        return value + 0.2 * merged.reshape_as(value)


def _run_checkpointed_router(model, *, install_before_backward=False):
    model.zero_grad(set_to_none=True)
    initial = torch.linspace(-1, 1, 88, dtype=torch.float64).reshape(1, 11, 8)
    initial.requires_grad_()
    inputs = []

    def checkpointed(layer):
        def compute(value):
            if torch.is_grad_enabled():
                assert value.is_leaf
                inputs.append(weakref.ref(value))
            return layer(value)

        return compute

    hidden = initial
    for layer in model:
        hidden = mcore_random.CheckpointFunction.apply(
            checkpointed(layer), False, hidden
        )
    loss = hidden.square().sum()
    if install_before_backward:
        _configure_moe_dispatcher_caches([model])
    loss.backward()
    gc.collect()
    assert len(inputs) == len(model)
    alive = [reference() is not None for reference in inputs]
    gradients = []
    for value in [initial, *model.parameters()]:
        assert value.grad is not None
        gradients.append(value.grad.clone())
    # Keep the loss/output alive: clearing only the cache must release the leaves.
    cache_sizes = []
    for layer in model:
        dispatcher = layer.token_dispatcher
        cache_sizes.append(dispatcher.probs.numel())
        assert dispatcher.probs.dtype == initial.dtype
        assert dispatcher.probs.device == initial.device
        dispatcher.probs = None
    gc.collect()
    assert all(reference() is None for reference in inputs)
    return loss.detach(), gradients, alive, cache_sizes


@pytest.fixture
def cpu_checkpoint_rng(monkeypatch):
    # The imported MCore checkpoint implementation otherwise runs unchanged.
    monkeypatch.setattr(
        mcore_random, "_get_all_rng_states", lambda: (torch.get_rng_state(),)
    )
    monkeypatch.setattr(
        mcore_random, "_set_all_rng_states", lambda state: torch.set_rng_state(state)
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("pending_graph", [False, True])
def test_dispatcher_cache_releases_checkpoint_inputs(
    cpu_checkpoint_rng, compiled, pending_graph
):
    torch.manual_seed(47)
    original = MoEAlltoAllTokenDispatcher.dispatch_preprocess
    backend = CompileCounterWithBackend("aot_eager") if compiled else None
    model = torch.nn.ModuleList([_RouterLayer() for _ in range(4)])
    if backend is not None:
        model = torch.nn.ModuleList(
            [
                cast(torch.nn.Module, torch.compile(layer, backend=backend))
                for layer in model
            ]
        )
    try:
        reference_loss, reference_grads, retained, sizes = _run_checkpointed_router(
            model
        )
        assert all(retained)
        assert sizes == [44] * 4
        # This same model has already compiled/executed both forward and backward.
        # Do not reset the compiler between warming and installing the adaptation.
        if not pending_graph:
            _configure_moe_dispatcher_caches([model])
        loss, gradients, retained, sizes = _run_checkpointed_router(
            model, install_before_backward=pending_graph
        )
        assert not any(retained)
        assert sizes == [0] * 4
        assert torch.equal(loss, reference_loss)
        for actual, expected in zip(gradients, reference_grads, strict=True):
            assert torch.equal(actual, expected)
        assert MoEAlltoAllTokenDispatcher.dispatch_preprocess is original
        if backend is not None:
            assert backend.frame_count > 0
    finally:
        torch.compiler.reset()


def test_dispatcher_adaptation_is_instance_scoped_and_collectable(cpu_checkpoint_rng):
    class CustomDispatcher(MoEAlltoAllTokenDispatcher):
        pass

    target = _RouterLayer()
    untouched = _RouterLayer()
    subclass = object.__new__(CustomDispatcher)
    overridden = _dispatcher()
    override = overridden.dispatch_preprocess
    overridden.dispatch_preprocess = override
    excluded = [
        subclass,
        overridden,
        object.__new__(MoEAllGatherTokenDispatcher),
        object.__new__(MoEFlexTokenDispatcher),
    ]
    owners = [target, target]  # Shared module and dispatcher discovery is harmless.
    for dispatcher in [target.token_dispatcher, *excluded]:
        owner = torch.nn.Module()
        setattr(owner, "token_dispatcher", dispatcher)
        owners.append(owner)
    model = torch.nn.ModuleList(owners)
    original = MoEAlltoAllTokenDispatcher.dispatch_preprocess
    target.token_dispatcher.probs = target.weight.softmax(-1)
    cached = weakref.ref(target.token_dispatcher.probs)
    runtime = SimpleNamespace(
        model=[model],
        provider=SimpleNamespace(hidden_size=8, num_layers=1),
        model_support_handler=SimpleNamespace(),
        optimizer=None,
    )
    trainer = TrainerRank(cast(Any, runtime))
    assert cached() is None
    assert target.token_dispatcher.probs.numel() == 0
    adapted = target.token_dispatcher.dispatch_preprocess
    _configure_moe_dispatcher_caches([model, model])
    assert target.token_dispatcher.dispatch_preprocess is adapted
    assert MoEAlltoAllTokenDispatcher.dispatch_preprocess is original
    assert untouched.token_dispatcher.dispatch_preprocess.__func__ is original
    assert subclass.dispatch_preprocess.__func__ is original
    assert overridden.dispatch_preprocess is override
    for dispatcher in excluded:
        assert "probs" not in vars(dispatcher)
    _, _, retained, sizes = _run_checkpointed_router(torch.nn.ModuleList([untouched]))
    assert retained == [True]
    assert sizes == [44]
    # Persistent callables must not keep the dispatcher alive once its owner
    # and any pending graphs have gone away.
    reference = weakref.ref(target.token_dispatcher)
    del adapted, target, owners, model, owner, trainer, runtime
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("round_trip", ["pickle", "deepcopy"])
@pytest.mark.parametrize("compiled", [False, True])
def test_dispatcher_adaptation_survives_serialization(
    cpu_checkpoint_rng, round_trip, compiled
):
    torch.manual_seed(47)
    model = torch.nn.ModuleList([_RouterLayer() for _ in range(4)])
    expected_loss, expected_grads, retained, _ = _run_checkpointed_router(model)
    assert all(retained)
    _configure_moe_dispatcher_caches([model])
    clone = (
        pickle.loads(pickle.dumps(model)) if round_trip == "pickle" else deepcopy(model)
    )
    for layer in clone:
        dispatcher: Any = layer.token_dispatcher
        wrapper = dispatcher.dispatch_preprocess
        assert isinstance(wrapper, partial)
        assert wrapper.args[0] is dispatcher
        assert all(dispatcher is not original.token_dispatcher for original in model)
        _configure_moe_dispatcher_caches([clone])
        assert dispatcher.dispatch_preprocess is wrapper
    backend = CompileCounterWithBackend("aot_eager") if compiled else None
    if backend is not None:
        clone = torch.nn.ModuleList(
            [
                cast(torch.nn.Module, torch.compile(layer, backend=backend))
                for layer in clone
            ]
        )
    try:
        loss, gradients, retained, sizes = _run_checkpointed_router(clone)
        assert not any(retained)
        assert sizes == [0] * 4
        assert torch.equal(loss, expected_loss)
        for actual, expected in zip(gradients, expected_grads, strict=True):
            assert torch.equal(actual, expected)
        if backend is not None:
            assert backend.frame_count > 0
    finally:
        torch.compiler.reset()
