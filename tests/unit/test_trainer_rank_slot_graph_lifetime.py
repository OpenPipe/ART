from __future__ import annotations

from contextlib import nullcontext
import gc
from typing import Literal

import pytest
from test_trainer_rank_custom_tensors import _trainer
import torch

from art.megatron.context_parallel.types import ParallelTopology
from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import ForwardInput, ForwardOptions, ForwardOutput
from art.trainer_rank._impl import (
    TrainerRankSlotStateError,
    _ForwardGroupPlan,
    _ForwardItem,
)
from art.trainer_rank._tensors import ManagedTensor


@pytest.fixture(params=["model", "cpu"])
def output_device(request):
    return request.param


@pytest.fixture(params=["none", "detach", "cpu"], autouse=True)
def ambient_hooks(request):
    saved = []

    def pack(tensor):
        saved.append(tensor.dtype)
        return tensor.detach()

    context = (
        torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor)
        if request.param == "detach"
        else torch.autograd.graph.save_on_cpu()
        if request.param == "cpu"
        else nullcontext()
    )
    with context:
        yield saved if request.param == "detach" else None


def _forward(monkeypatch, retention, output_device, *, grad_enabled=True):
    trainer, _ = _trainer("student")
    ref = trainer._slot_ref("student")
    weight = torch.nn.Parameter(torch.tensor(2.0))
    tokens = torch.tensor([1, 2])
    request = ForwardInput(
        input_tokens=tokens,
        hidden_states=True,
        logits=True,
        options=ForwardOptions(backward_state=retention, output_device=output_device),
    )
    group = _ForwardGroupPlan(
        ref,
        grad_enabled,
        (0,),
        (_ForwardItem(request, tokens, None),),
        prefix_tree_pack((tokens,), max_depth=0),
    )
    monkeypatch.setattr(trainer, "_topology", lambda: ParallelTopology())
    monkeypatch.setattr(trainer, "_configure_hybridep", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_prepare_packed_forward", lambda packed: None)
    monkeypatch.setattr(trainer, "_hybridep_graph_tracking", True, raising=False)
    monkeypatch.setattr(
        trainer,
        "_forward_packed",
        lambda items, prepared: [
            ForwardOutput(
                target_logprobs=None,
                top_k=None,
                hidden_states=weight.square(),
                logits=weight.pow(3),
            )
        ],
    )
    output = trainer._execute_graph_group(group)[0]
    return trainer, ref, weight, output


@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay"])
def test_native_cached_slot_guard_tracks_retained_and_consumed_graph(
    monkeypatch, retention: Literal["gpu", "cpu", "replay"], output_device
):
    trainer, ref, weight, output = _forward(monkeypatch, retention, output_device)
    loss = output.hidden_states
    assert loss is not None
    assert isinstance(loss, ManagedTensor) == (output_device == "cpu")
    assert trainer._has_live_slot_graph(ref)
    assert trainer._has_live_hybridep_graphs()
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)
    handle = trainer._forward_graph_cache().handles()[0]
    trainer._forward_graph_cache().evict(handle)
    assert trainer._has_live_slot_graph(ref)
    for _ in range(2):
        trainer.backward(loss, retain_graph=True)
        assert trainer._has_live_slot_graph(ref)
        assert trainer._has_live_hybridep_graphs()
    trainer.backward(loss)
    torch.testing.assert_close(weight.grad, torch.tensor(12.0))
    # Keep both returned outputs alive: the unused logits cannot be consumed
    # after final backward releases their shared physical graph.
    assert output.logits is not None
    assert not trainer._forward_graph_cache().handles()
    assert not trainer._has_live_slot_graph(ref)
    assert not trainer._has_live_hybridep_graphs()
    trainer._guard_slot_can_load(ref)


@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay"])
def test_native_cached_slot_guard_follows_dependent_loss(
    monkeypatch, retention, output_device
):
    trainer, ref, _, output = _forward(monkeypatch, retention, output_device)
    assert output.hidden_states is not None
    loss = output.hidden_states.square()
    del output
    gc.collect()
    assert trainer._has_live_slot_graph(ref)
    del loss
    gc.collect()
    assert not trainer._has_live_slot_graph(ref)
    assert not trainer._has_live_hybridep_graphs()
    assert not trainer._forward_graph_cache().handles()
    trainer._guard_slot_can_load(ref)


@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay"])
def test_native_no_grad_forward_has_no_slot_graph_guard(
    monkeypatch, retention, output_device
):
    trainer, ref, _, output = _forward(
        monkeypatch, retention, output_device, grad_enabled=False
    )
    assert output.hidden_states is not None and not output.hidden_states.requires_grad
    assert not trainer._has_live_slot_graph(ref)
    assert not trainer._has_live_hybridep_graphs()
    assert not trainer._forward_graph_cache().handles()
    trainer._guard_slot_can_load(ref)


@pytest.mark.parametrize("kind", ["parameter", "module"])
def test_custom_slot_guard_preserves_markers_and_ambient_activation_hooks(
    kind, ambient_hooks
):
    trainer, rank = _trainer("student")
    ref = trainer._slot_ref("student")
    if kind == "parameter":
        parameter = rank.parameter("p", lambda: torch.tensor(2.0), checkpoint="student")
        loss = parameter.square()
    else:
        head = rank.module("head", lambda: torch.nn.Linear(1, 1), checkpoint="student")
        loss = head(torch.ones(1, 1, requires_grad=True)).square().sum()
    with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
        trainer._guard_slot_can_load(ref)
    if ambient_hooks is not None:
        assert torch.float32 in ambient_hooks
        assert torch.bool not in ambient_hooks
    trainer.backward(loss, retain_graph=True)
    assert trainer._has_live_slot_graph(ref)
    trainer.backward(loss)
    trainer.zero_grad()
    assert not trainer._has_live_slot_graph(ref)
    trainer._guard_slot_can_load(ref)
