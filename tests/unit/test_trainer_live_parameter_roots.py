"""Direct live parameter roots use the same snapshots as parameter arithmetic."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from test_trainer_rank_custom_tensors import _trainer
import torch

from art.trainer_rank import TrainerRank
from art.trainer_rank._heads import LiveHead, export_head, head_gradient_targets
from art.trainer_rank._tensors import CotangentCollector


def _live_parameter(
    factory=lambda: torch.tensor(2.0),
) -> tuple[TrainerRank, torch.nn.Parameter, CotangentCollector, LiveHead]:
    trainer, rank = _trainer("student")
    parameter = rank.parameter("weight", factory, checkpoint="student")
    collector = CotangentCollector()
    live = LiveHead(
        export_head(trainer, "student", "weight"), parameter.detach(), collector
    )
    return trainer, parameter, collector, live


class _FailBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, *gradients):
        raise RuntimeError("local backward failed after a direct parameter root")


@pytest.mark.parametrize("existing", [False, True])
def test_native_direct_root_does_not_publish_when_another_root_fails(existing):
    trainer, rank = _trainer("student")
    parameter = rank.parameter(
        "weight", lambda: torch.tensor(2.0), checkpoint="student"
    )
    if existing:
        parameter.grad = torch.tensor(7.0)
    original = parameter.grad
    bad = _FailBackward.apply(torch.tensor(1.0, requires_grad=True))
    with pytest.raises(RuntimeError, match="local backward failed"):
        trainer.backward([bad, parameter])
    assert parameter.grad is original
    if existing:
        torch.testing.assert_close(parameter.grad, torch.tensor(7.0))
    # Failed local collection is reusable and a later direct root commits once.
    trainer.backward(parameter)
    torch.testing.assert_close(parameter.grad, torch.tensor(8.0 if existing else 1.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64])
@pytest.mark.parametrize("surface", ["native", "client"])
def test_direct_roots_preserve_aliases_explicit_gradients_and_dtype(dtype, surface):
    trainer, parameter, collector, live = _live_parameter(
        lambda: torch.tensor([2.0, 3.0], dtype=dtype)
    )
    root: Any = parameter if surface == "native" else live.value
    gradients = (
        torch.tensor([1.0, 2.0], dtype=dtype),
        torch.tensor([3.0, 5.0], dtype=dtype),
    )
    with torch.no_grad():
        if surface == "native":
            trainer.backward((root, root), gradients, retain_graph=True)
        else:
            packets = collector.backward((root, root), gradients, retain_graph=True)
            assert len(packets) == 1
            assert len(packets[0].gradients) == 1
            torch.testing.assert_close(packets[0].gradients[0], sum(gradients))
            trainer._commit_versioned_gradients(
                head_gradient_targets(trainer, packets[0])
            )
    assert parameter.grad is not None and parameter.grad.dtype == dtype
    torch.testing.assert_close(parameter.grad, sum(gradients))
    # Every direct use captures the current version independently of retain_graph.
    if surface == "native":
        trainer.backward(root, gradients[0])
    else:
        (packet,) = collector.backward(root, gradients[0])
        trainer._commit_versioned_gradients(head_gradient_targets(trainer, packet))
        assert root.grad is None
    torch.testing.assert_close(parameter.grad, sum(gradients) + gradients[0])


def test_direct_live_root_and_old_arithmetic_keep_their_own_versions():
    trainer, parameter, collector, live = _live_parameter()
    root: Any = live.value
    old = root.square()
    parameter.data.fill_(5)
    trainer._checkpoint_slots["student"].revision += 1
    live.refresh(export_head(trainer, "student", "weight"))
    packets = collector.backward((old, root))
    assert len(packets) == 2
    targets = [
        target
        for packet in packets
        for target in head_gradient_targets(trainer, packet)
    ]
    assert {target[0].revision for target in targets} == {0, 1}
    trainer._commit_versioned_gradients(targets)
    torch.testing.assert_close(parameter.grad, torch.tensor(5.0))


def test_client_direct_root_failure_discards_cotangents_and_revalidates_handle():
    trainer, parameter, collector, live = _live_parameter()
    root: Any = live.value
    bad = _FailBackward.apply(torch.tensor(1.0, requires_grad=True))
    with pytest.raises(RuntimeError, match="local backward failed"):
        collector.backward([bad, root])
    assert root.grad is None and parameter.grad is None
    (packet,) = collector.backward(root)
    torch.testing.assert_close(packet.gradients[0], torch.tensor(1.0))
    live.refresh(replace(live.state, version=replace(live.state.version, generation=1)))
    live.invalidate("checkpoint replaced")
    with pytest.raises(RuntimeError, match="stale"):
        collector.backward(root)


def test_ordinary_local_parameter_root_keeps_pytorch_accumulation():
    parameter = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    collector = CotangentCollector()
    assert collector.backward(parameter) == ()
    torch.testing.assert_close(parameter.grad, torch.tensor(1.0, dtype=torch.float64))
