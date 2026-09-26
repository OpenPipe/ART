from __future__ import annotations

import asyncio

import pytest
from test_trainer_rank_custom_tensors import _trainer
from test_trainer_rank_live_heads import _live_head, _native_head, _step
import torch

from art.trainer_rank._commands import run_rank_callback
from art.trainer_rank._heads import export_head
from art.trainer_rank._tensors import CotangentCollector


def _read(parameter, operation):
    if operation == "detach":
        return parameter.detach()
    if operation == "view":
        return parameter.view(2, 2)
    if operation == "slice":
        return parameter[:, :1]
    if operation == "transpose":
        return parameter.T
    return parameter.to(device=parameter.device, dtype=parameter.dtype)


@pytest.mark.parametrize("surface", ("native", "client", "zero", "rank"))
@pytest.mark.parametrize("operation", ("detach", "view", "slice", "transpose", "to"))
def test_no_grad_parameter_reads_preserve_saved_loss_after_optimizer_step(
    monkeypatch, surface, operation
):
    initial = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    trainer, native = _native_head("parameter", "p", lambda: initial.clone())
    collector = CotangentCollector()
    live = _live_head(trainer, "p", initial, collector)

    def capture(parameter):
        with torch.no_grad():
            saved = _read(parameter, operation)
        assert not saved.requires_grad
        value = torch.ones_like(saved, requires_grad=True)
        return saved, value, (value * saved).sum()

    def logical_capture(view):
        parameter = view.parameter("p", lambda: initial.clone(), checkpoint="student")
        return capture(parameter)

    if surface in {"zero", "rank"}:
        saved, value, loss = asyncio.run(
            run_rank_callback(trainer, logical_capture, mode=surface)
        ).value
    else:
        saved, value, loss = capture(native if surface == "native" else live.value)

    trainer.backward(native.square().sum())
    _step(trainer, monkeypatch)
    assert trainer._checkpoint_slots["student"].revision == 1
    assert not torch.equal(native, initial)
    live.refresh(export_head(trainer, "student", "p"))
    if surface in {"zero", "rank"}:
        asyncio.run(
            run_rank_callback(trainer, lambda view: view.backward(loss), mode=surface)
        )
    elif surface == "client":
        assert collector.backward(loss) == ()
    else:
        trainer.backward(loss)
    torch.testing.assert_close(saved, _read(initial, operation))
    torch.testing.assert_close(value.grad, _read(initial, operation))
    assert native.grad is None


@pytest.mark.parametrize("requires_grad", (False, True))
def test_native_no_grad_detached_read_is_private_and_does_not_capture_gradients(
    monkeypatch, requires_grad
):
    trainer, rank = _trainer("student")

    def factory():
        module = torch.nn.Module()
        module.register_parameter(
            "p", torch.nn.Parameter(torch.tensor(2.0), requires_grad=requires_grad)
        )
        return module

    parameter = rank.module("head", factory, checkpoint="student").p
    assert parameter.requires_grad is requires_grad

    def forbidden_snapshot(*args, **kwargs):
        raise AssertionError("no-grad reads must not create backward snapshots")

    monkeypatch.setattr(trainer, "_snapshot_parameter", forbidden_snapshot)
    with torch.no_grad():
        saved = parameter.detach()
        saved.add_(5)
        assert saved.item() == 7
        assert parameter.item() == 2
    assert parameter.grad is None
    assert trainer._checkpoint_slots["student"].revision == 0
