from __future__ import annotations

import asyncio
from contextlib import nullcontext
import gc
from typing import Literal
import weakref

import pytest
from test_trainer_rank_custom_tensors import _real_lora_trainer, _trainer
import torch

from art.megatron.context_parallel.types import ParallelTopology
from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import ForwardInput, ForwardOptions, ForwardOutput, TopK
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


def _prepare_forward(monkeypatch, retention, output_device, *, grad_enabled=True):
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
    return trainer, ref, weight, group


def _forward(monkeypatch, retention, output_device, *, grad_enabled=True):
    trainer, ref, weight, group = _prepare_forward(
        monkeypatch, retention, output_device, grad_enabled=grad_enabled
    )
    output = trainer._execute_graph_group(group)[0]
    return trainer, ref, weight, output


@pytest.mark.parametrize("failure_type", [MemoryError, asyncio.CancelledError])
def test_failed_correction_capture_releases_unattached_graph(monkeypatch, failure_type):
    trainer, ref, weight, group = _prepare_forward(monkeypatch, "cpu", "cpu")
    monkeypatch.setattr(
        trainer,
        "_forward_packed",
        lambda items, prepared: [
            ForwardOutput(
                weight.square(),
                TopK(weight.pow(3).reshape(1), torch.tensor([0])),
                None,
                None,
            )
        ],
    )
    cache = trainer._forward_graph_cache()
    older = torch.nn.Parameter(torch.tensor(3.0))
    old_handle, (old_output,) = cache.run(lambda _: (older.square(),), None)
    old_record = weakref.ref(cache._records[old_handle])
    primary = failure_type("correction CPU snapshot failed")
    primary.__cause__ = cause = RuntimeError("original cause")
    saved, outputs, versions, detached, copies = [], [], [], [], []
    to = torch.Tensor.to

    def fail_snapshot(value, *args, **kwargs):
        if args == ("cpu",) and kwargs.get("copy") and len(cache.handles()) == 2:
            if len(copies) < 2:
                copied = to(value, *args, **kwargs)
                copies.append(weakref.ref(copied))
                return copied
            handle = next(handle for handle in cache.handles() if handle != old_handle)
            record = cache._records[handle]
            saved.extend(record.saved or ())
            outputs.extend(weakref.ref(output) for output in record.outputs or ())
            versions.extend(
                weakref.ref(v) for v in trainer._version_state().lora.values()
            )
            assert saved and outputs and len(versions) == 1
            assert record.checkpoint_versions == (
                trainer._capture_checkpoint_version("student"),
            )
            del value
            raise primary
        copied = to(value, *args, **kwargs)
        if kwargs.get("copy") and "device" in kwargs:
            detached.append(weakref.ref(copied))
        return copied

    with monkeypatch.context() as allocation:
        allocation.setattr(torch.Tensor, "to", fail_snapshot)
        with pytest.raises(failure_type) as caught:
            trainer._execute_graph_group(group)
    assert caught.value is primary and primary.__cause__ is cause
    assert primary.__traceback__ is not None
    assert cache.handles() == (old_handle,)
    assert cache._records[old_handle] is old_record()
    assert len(detached) == 3 and len(copies) == 2
    assert all(reference() is None for reference in (*saved, *outputs))
    assert all(reference() is None for reference in (*detached, *copies))
    assert not trainer._has_live_slot_graph(ref)
    assert all(reference() is None for reference in versions)
    assert not trainer._version_state().lora
    trainer._guard_slot_can_load(ref)
    assert weight.grad is None
    assert old_output.item() == 9
    cache.backward(old_handle, (torch.tensor(1.0),))
    torch.testing.assert_close(older.grad, torch.tensor(6.0))
    assert not cache.handles()
    output = trainer._execute_graph_group(group)[0].target_logprobs
    assert output is not None and output.item() == 4
    assert len(cache.handles()) == 1
    trainer.backward(output)
    torch.testing.assert_close(weight.grad, torch.tensor(4.0))
    assert not cache.handles()


@pytest.mark.parametrize(
    "phase,shared",
    [
        (phase, shared)
        for phase in ("copy", "correction", "handoff")
        for shared in (False, True)
    ]
    + [("cancel", False)],
)
def test_native_failed_handoff_drops_only_its_capture_owners(
    monkeypatch, phase, shared
):
    trainer, ref, _, group = _prepare_forward(monkeypatch, "cpu", "cpu")
    real, _ = _real_lora_trainer()
    from art.megatron.lora import LoRA

    trainer.runtime, trainer._checkpoint_slots = real.runtime, real._checkpoint_slots
    lora = trainer.runtime.model[0]
    assert isinstance(lora, LoRA)
    current = lora.lora_slot_params(ref)
    with torch.no_grad():
        current[0].fill_(1)
        current[1].fill_(2)
    versions, snapshots, delivered = [], [], []

    def forward(items, prepared):
        active = lora.active_lora_tensors()
        assert active is not None
        a, b, _ = active
        assert a is not current[0] and b is not current[1]
        versions.extend(weakref.ref(v) for v in trainer._version_state().lora.values())
        snapshots.extend((weakref.ref(a), weakref.ref(b)))
        return [ForwardOutput(a.square().sum() + b.square().sum(), None, None, None)]

    monkeypatch.setattr(trainer, "_forward_packed", forward)
    cache = trainer._forward_graph_cache()
    older = torch.nn.Parameter(torch.tensor(3.0))
    if shared:
        old_output = trainer._execute_graph_group(group)[0].target_logprobs
        (old_handle,) = cache.handles()
    else:
        old_handle, (old_output,) = cache.run(lambda _: (older.square(),), None)
    old_record = weakref.ref(cache._records[old_handle])
    versions.clear()
    snapshots.clear()
    error_type = asyncio.CancelledError if phase == "cancel" else MemoryError
    primary = error_type("native handoff failed")
    primary.__cause__ = cause = RuntimeError("original cause")
    to = torch.Tensor.to

    def fail_copy(value, *args, **kwargs):
        if kwargs.get("copy") and (
            (phase in ("copy", "cancel") and "device" in kwargs)
            or (
                phase == "correction" and args == ("cpu",) and len(cache.handles()) == 2
            )
        ):
            del value
            raise primary
        return to(value, *args, **kwargs)

    def fail_handoff(_ref, outputs):
        delivered.extend(weakref.ref(output.target_logprobs) for output in outputs)
        del outputs
        raise primary

    with monkeypatch.context() as failure:
        failure.setattr(torch.Tensor, "to", fail_copy)
        if phase == "handoff":
            failure.setattr(trainer, "_track_slot_graph_outputs", fail_handoff)
        with pytest.raises(error_type) as caught:
            trainer._execute_graph_group(group)
    assert caught.value is primary and primary.__cause__ is cause
    assert primary.__traceback__ is not None
    assert (
        cache.handles() == (old_handle,) and cache._records[old_handle] is old_record()
    )
    assert len(versions) == 1 and len(snapshots) == 2
    assert all(
        (reference() is not None) == shared for reference in (*versions, *snapshots)
    ), (
        "retained version and snapshots",
        tuple(reference() is not None for reference in (*versions, *snapshots)),
    )
    assert all(reference() is None for reference in delivered)
    assert all(parameter.grad is None for parameter in current)
    if shared:
        assert old_output is not None and old_output.item() == 38
        cache.evict(old_handle)
        trainer.backward(old_output)
        torch.testing.assert_close(current[0].grad, torch.full_like(current[0], 2))
        torch.testing.assert_close(current[1].grad, torch.full_like(current[1], 4))
        for parameter in current:
            parameter.grad = None
    else:
        assert old_output is not None and old_output.item() == 9
        cache.backward(old_handle, (torch.tensor(1.0),))
        torch.testing.assert_close(older.grad, torch.tensor(6.0))
    assert all(reference() is None for reference in (*versions, *snapshots))
    assert not trainer._version_state().lora
    trainer._guard_slot_can_load(ref)

    output = trainer._execute_graph_group(group)[0].target_logprobs
    assert output is not None and output.item() == 38
    (handle,) = cache.handles()
    cache.evict(handle)
    trainer.backward(output)
    torch.testing.assert_close(current[0].grad, torch.full_like(current[0], 2))
    torch.testing.assert_close(current[1].grad, torch.full_like(current[1], 4))
    assert not cache.handles()


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
