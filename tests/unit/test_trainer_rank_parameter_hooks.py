from __future__ import annotations

import gc
import json

import pytest
from test_trainer_rank_custom_tensors import _trainer
import torch

from art.trainer_rank import ModuleHandle
from art.trainer_rank._heads import LiveHead, export_head, head_gradient_targets
from art.trainer_rank._tensors import CotangentCollector


def setup(client, values=None):
    trainer, rank = _trainer("student")
    factory = lambda: torch.tensor(2.0) if values is None else values.clone()
    native = rank.parameter("p", factory, checkpoint="student")
    collector = CotangentCollector()
    live = LiveHead(
        export_head(trainer, "student", "p"),
        factory() if values is None else values,
        collector,
    )
    parameter = live.value if client else native
    assert isinstance(parameter, torch.Tensor)

    def backward(loss, **kwargs):
        if client:
            packets = collector.backward(loss, **kwargs)
            with trainer._gradient_transaction():
                for packet in packets:
                    trainer._commit_versioned_gradients(
                        head_gradient_targets(trainer, packet)
                    )
            return packets
        trainer.backward(loss, **kwargs)

    return trainer, native, parameter, live, collector, backward


@pytest.mark.parametrize("client", (False, True))
def test_live_parameter_hook_masks_real_gradient(client):
    _, native, parameter, _, _, backward = setup(client)
    called = []
    handle = parameter.register_hook(
        lambda gradient: called.append(gradient.item()) or gradient * 0
    )
    backward(parameter.square())
    assert called == [4]
    assert native.grad.item() == 0
    handle.remove()
    backward(parameter.square())
    assert native.grad.item() == 4


@pytest.mark.parametrize("client", (False, True))
def test_hooks_aggregate_uses_and_old_versions_before_existing_grad(client):
    trainer, native, parameter, live, _, backward = setup(client)
    old = parameter.square() + parameter * 3
    native.data.fill_(4)
    trainer._checkpoint_slots["student"].revision += 1
    if client:
        live.refresh(export_head(trainer, "student", "p"))
        live.max_gradient_staleness = 0
    new = parameter.square()
    native.grad = torch.tensor(5.0)
    seen = []
    first = parameter.register_hook(
        lambda gradient: seen.append(gradient.item()) or gradient.square()
    )
    second = parameter.register_hook(lambda gradient: gradient + 1)
    packets = backward(old + new)
    assert seen == [15]
    assert native.grad.item() == 231
    if client:
        combined = [
            json.loads(packet.handle[5:])
            for packet in packets
            if any(g is not None for g in packet.gradients)
        ]
        assert len(combined) == 1
        assert combined[0]["revision"] == 0
        assert combined[0]["max_gradient_staleness"] == 1
    first.remove()
    second.remove()
    trainer._checkpoint_slots["student"].revision += 1 if client else 2
    with pytest.raises(RuntimeError, match="staleness"):
        trainer._version_state().validate_accumulated(["student"])


@pytest.mark.parametrize("client", (False, True))
def test_hook_removal_applies_to_retained_graph(client):
    _, native, parameter, _, _, backward = setup(client)
    seen = []
    loss = parameter.square()
    handle = parameter.register_hook(
        lambda gradient: seen.append(gradient.item()) or gradient * 2
    )
    backward(loss, retain_graph=True)
    handle.remove()
    backward(loss)
    assert seen == [4]
    assert native.grad.item() == 12


@pytest.mark.parametrize("client", (False, True))
@pytest.mark.parametrize("bad_result", (False, True))
def test_hook_failure_leaves_all_authoritative_gradients_unchanged(client, bad_result):
    trainer, native, parameter, _, collector, backward = setup(client)
    rank = trainer
    other = rank.parameter("q", lambda: torch.tensor(3.0), checkpoint="student")
    qlive = LiveHead(export_head(trainer, "student", "q"), torch.tensor(3.0), collector)
    q = qlive.value if client else other
    native.grad, other.grad = torch.tensor(5.0), torch.tensor(6.0)
    parameter.register_hook(lambda gradient: gradient * 0)

    def fail(gradient):
        if bad_result:
            return torch.ones(2)
        raise RuntimeError("user hook failed")

    q.register_hook(fail)
    with pytest.raises(RuntimeError, match="hook"):
        backward(parameter.square() + q.square())
    assert native.grad.item() == 5
    assert other.grad.item() == 6


@pytest.mark.parametrize("client", (False, True))
def test_post_accumulate_hook_rejected_at_registration(client):
    _, _, parameter, _, _, _ = setup(client)
    with pytest.raises(RuntimeError, match="do not support register_post_accumulate"):
        parameter.register_post_accumulate_grad_hook(lambda parameter: None)


def test_head_hook_registries_follow_graph_lifetime():
    _, _, parameter, _, collector, _ = setup(True)
    loss = parameter.square()
    assert len(collector._head_hooks) == 1
    del loss
    gc.collect()
    assert not collector._head_hooks


@pytest.mark.parametrize("client", (False, True))
@pytest.mark.parametrize("sparse_first", (False, True))
@pytest.mark.parametrize("mixed", (False, True))
def test_sparse_hook_gradients_preserve_layout_and_mix_with_dense(
    client, sparse_first, mixed
):
    values = torch.arange(1.0, 7).reshape(3, 2)
    _, native, parameter, _, _, backward = setup(client, values)
    seen = []
    parameter.register_hook(lambda gradient: seen.append(gradient.layout) or gradient)
    indices = torch.tensor([0, 2, 0])
    sparse = lambda: torch.nn.functional.embedding(
        indices, parameter, sparse=True
    ).sum()
    dense = lambda: parameter.square().sum()
    loss = (
        (sparse() + dense() if sparse_first else dense() + sparse())
        if mixed
        else sparse()
    )
    native.grad = torch.ones_like(values)

    if not mixed:
        with pytest.raises(ValueError, match="layout"):
            backward(loss)
        torch.testing.assert_close(native.grad, torch.ones_like(values))
        if client:
            assert seen == [torch.sparse_coo]
    else:
        backward(loss)
        assert seen == [torch.strided]
        expected = 1 + 2 * values + torch.tensor([[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]])
        torch.testing.assert_close(native.grad, expected)


@pytest.mark.parametrize("client", (False, True))
def test_tied_module_parameter_hook_sums_all_calls(client):
    from test_trainer_rank_live_heads import TiedHead

    trainer, rank = _trainer("student")
    native = rank.module("head", TiedHead, checkpoint="student")
    collector = CotangentCollector()
    live = LiveHead(export_head(trainer, "student", "head"), TiedHead(), collector)
    head = live.value if client else native
    assert isinstance(head, ModuleHandle)
    seen = []
    head.left.register_hook(
        lambda gradient: seen.append(gradient.item()) or gradient.clamp(max=10)
    )
    loss = head(torch.tensor(3.0)) + head(torch.tensor(1.0))
    if client:
        with trainer._gradient_transaction():
            for packet in collector.backward(loss):
                trainer._commit_versioned_gradients(
                    head_gradient_targets(trainer, packet)
                )
    else:
        trainer.backward(loss)
    assert seen == [18]
    assert native.left.grad.item() == 10


@pytest.mark.parametrize("client", (False, True))
def test_hook_on_direct_root_and_removed_before_first_backward(client):
    _, native, parameter, _, _, backward = setup(client)
    called = []
    handle = parameter.register_hook(
        lambda gradient: called.append(gradient.item()) or gradient * 3
    )
    backward(parameter)
    assert native.grad.item() == 3
    assert called == [1]
    loss = parameter.square()
    handle.remove()
    backward(loss)
    assert native.grad.item() == 7
    assert called == [1]


def test_hook_registry_survives_release_during_inline_backward(monkeypatch):
    _, native, parameter, _, collector, backward = setup(True)
    parameter.register_hook(lambda gradient: gradient * 0)
    record = collector._record

    def release_while_recording(*args):
        record(*args)
        collector._head_hooks.clear()

    monkeypatch.setattr(collector, "_record", release_while_recording)
    backward(parameter.square())
    assert native.grad.item() == 0


@pytest.mark.parametrize("client", (False, True))
def test_hook_cannot_mutate_authoritative_parameter(client):
    trainer, native, parameter, _, _, backward = setup(client)

    def mutate(gradient):
        parameter.add_(1)
        return gradient

    parameter.register_hook(mutate)
    with pytest.raises(
        RuntimeError, match="mutate checkpoint parameters|only be changed"
    ):
        backward(parameter.square())
    assert native.item() == 2
    assert native.grad is None
    assert trainer._checkpoint_slots["student"].revision == 0


def test_later_hook_failure_discards_already_prepared_gradient():
    trainer, native, parameter, _, _, _ = setup(False)
    other = trainer.parameter("q", lambda: torch.tensor(3.0), checkpoint="student")
    native.grad, other.grad = torch.tensor(5.0), torch.tensor(6.0)
    called = []
    parameter.register_hook(lambda gradient: called.append("p") or gradient * 0)

    def fail(gradient):
        called.append("q")
        raise RuntimeError("second hook failed")

    other.register_hook(fail)
    version = trainer._capture_checkpoint_version("student")
    with pytest.raises(RuntimeError, match="second hook failed"):
        trainer._commit_versioned_gradients(
            [
                (version, 2, native, torch.tensor(4.0)),
                (version, 2, other, torch.tensor(6.0)),
            ]
        )
    assert called == ["p", "q"]
    assert native.grad.item() == 5
    assert other.grad.item() == 6
