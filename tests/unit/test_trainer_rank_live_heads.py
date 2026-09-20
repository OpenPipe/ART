from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace

import pytest
from test_trainer_rank_custom_tensors import _trainer
import torch
from torch.utils.checkpoint import checkpoint

from art.trainer_rank import AdamParams, ModuleHandle, run_rank_callback
from art.trainer_rank._heads import (
    HeadBufferUpdate,
    HeadRegistration,
    LiveHead,
    execute_head_operation,
    export_head,
    head_gradient_targets,
)
from art.trainer_rank._tensors import CotangentCollector


class TiedHead(torch.nn.Module):
    offset: torch.Tensor
    other_offset: torch.Tensor

    def __init__(self, checkpointed: bool | None = None):
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor(2.0))
        self.right = self.left
        self.register_buffer("offset", torch.tensor(1.0))
        self.register_buffer("other_offset", self.offset)
        self.checkpointed = checkpointed

    def forward(self, value):
        def compute(x):
            assert self.left is self.right
            assert self.offset is self.other_offset
            return x * self.left.square() + self.right + self.offset

        return (
            compute(value)
            if self.checkpointed is None
            else checkpoint(compute, value, use_reentrant=self.checkpointed)
        )


def _live_head(
    trainer, name, source, collector=None, *, checkpoint="student"
) -> LiveHead:
    return LiveHead(
        export_head(trainer, checkpoint, name),
        source,
        CotangentCollector() if collector is None else collector,
    )


def _module(live: LiveHead) -> ModuleHandle:
    assert isinstance(live.value, ModuleHandle)
    return live.value


def _tensor(live: LiveHead) -> torch.Tensor:
    assert isinstance(live.value, torch.Tensor)
    return live.value


def _step(trainer, monkeypatch):
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, **kwargs: tuple(
            torch.zeros_like(p, dtype=torch.float32)
            if p.grad is None
            else p.grad.float()
            for p in params
        ),
    )
    trainer.optim_step(
        params=AdamParams(learning_rate=0.1, weight_decay=0.0), checkpoints=["student"]
    )


@pytest.mark.parametrize("checkpointed", (None, False, True))
def test_native_old_head_graph_uses_original_tied_weights_after_step(
    monkeypatch, checkpointed
):
    trainer, rank = _trainer("student")
    head = rank.module("head", lambda: TiedHead(checkpointed), checkpoint="student")
    old_input = torch.tensor(3.0, requires_grad=True)
    old_loss = head(old_input)
    with trainer._gradient_transaction():
        head(torch.tensor(1.0, requires_grad=True)).backward()
    _step(trainer, monkeypatch)
    assert head.left.item() != 2
    latest = head(torch.tensor(3.0))
    torch.testing.assert_close(
        latest, 3 * head.left.detach().square() + head.left.detach() + 1
    )
    with trainer._gradient_transaction():
        old_loss.backward()
    torch.testing.assert_close(head.left.grad, torch.tensor(13.0))
    torch.testing.assert_close(old_input.grad, torch.tensor(4.0))
    assert head.left is head.right


def test_batchnorm_buffers_publish_once_and_old_graph_keeps_original_buffers():
    trainer, rank = _trainer("student")
    head = rank.module(
        "bn", lambda: torch.nn.BatchNorm1d(2, dtype=torch.float64), checkpoint="student"
    )
    initial = deepcopy(head).eval()
    head.eval()
    x = torch.tensor([[1.0, 3.0], [2.0, 5.0]], dtype=torch.float64, requires_grad=True)
    old = head(x).square().sum()
    expected_input = x.detach().clone().requires_grad_()
    expected = initial(expected_input).square().sum()
    head.train()
    head(torch.tensor([[4.0, 2.0], [8.0, 10.0]], dtype=torch.float64))
    assert head.num_batches_tracked.item() == 1
    assert not torch.equal(head.running_mean, initial.running_mean)
    with trainer._gradient_transaction():
        old.backward()
    with trainer._gradient_transaction():
        expected.backward()
    torch.testing.assert_close(x.grad, expected_input.grad)
    torch.testing.assert_close(head.weight.grad, initial.weight.grad)


def test_failed_module_call_does_not_publish_buffers():
    class Failing(torch.nn.BatchNorm1d):
        def forward(self, input):
            super().forward(input)
            raise RuntimeError("failed after buffer mutation")

    trainer, rank = _trainer("student")
    head = rank.module("bn", lambda: Failing(2), checkpoint="student")
    with pytest.raises(RuntimeError, match="failed after"):
        head(torch.ones(4, 2))
    assert head.num_batches_tracked.item() == 0
    torch.testing.assert_close(head.running_mean, torch.zeros(2))


def test_client_tied_head_old_backward_after_native_refresh(monkeypatch):
    trainer, rank = _trainer("student")
    native = rank.module("head", TiedHead, checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "head", TiedHead(), collector)
    old_input = torch.tensor(3.0, requires_grad=True)
    old = _module(live)(old_input)
    with trainer._gradient_transaction():
        native(torch.tensor(1.0)).backward()
    _step(trainer, monkeypatch)
    live.refresh(export_head(trainer, "student", "head"))
    assert _module(live).left is _module(live).right
    assert _module(live)(torch.tensor(3.0)).item() != old.item()
    packets = collector.backward(old)
    assert len(packets) == 1
    trainer._commit_versioned_gradients(head_gradient_targets(trainer, packets[0]))
    torch.testing.assert_close(native.left.grad, torch.tensor(13.0))
    torch.testing.assert_close(old_input.grad, torch.tensor(4.0))


def test_live_parameter_reuses_handle_and_earlier_capture_retains_version():
    trainer, rank = _trainer("student")
    parameter = rank.parameter("gain", lambda: torch.tensor(2.0), checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "gain", torch.tensor(2.0), collector)
    handle = _tensor(live)
    old = handle.square() * 3
    parameter.data.fill_(4)
    trainer._checkpoint_slots["student"].revision += 1
    live.refresh(export_head(trainer, "student", "gain"))
    assert _tensor(live) is handle
    assert (handle * 2).item() == 8
    packets = collector.backward(old)
    trainer._commit_versioned_gradients(head_gradient_targets(trainer, packets[0]))
    torch.testing.assert_close(parameter.grad, torch.tensor(12.0))


def test_client_buffer_publication_conflict_is_atomic():
    trainer, rank = _trainer("student")
    native = rank.module("bn", lambda: torch.nn.BatchNorm1d(2), checkpoint="student")
    live = _live_head(trainer, "bn", torch.nn.BatchNorm1d(2))
    _module(live)(torch.ones(4, 2))
    update = live.take_publication()
    assert update is not None
    execute_head_operation(trainer, "head_publish", (update,))
    live.refresh(export_head(trainer, "student", "bn"))
    assert native.num_batches_tracked.item() == 1
    native(torch.zeros(4, 2))
    _module(live)(torch.ones(4, 2) * 5)
    stale = live.take_publication()
    assert stale is not None
    before = native.running_mean.detach().clone()
    with pytest.raises(RuntimeError, match="buffers changed"):
        execute_head_operation(trainer, "head_publish", (stale,))
    torch.testing.assert_close(native.running_mean, before)


def test_replacement_invalidates_live_handle_and_old_gradient():
    trainer, rank = _trainer("student")
    rank.parameter("gain", lambda: torch.tensor(2.0), checkpoint="student")
    collector = CotangentCollector()
    state = export_head(trainer, "student", "gain")
    live = LiveHead(state, torch.tensor(2.0), collector)
    old = _tensor(live).square()
    trainer._checkpoint_slots["student"].generation += 1
    live.refresh(export_head(trainer, "student", "gain"))
    with pytest.raises(RuntimeError, match="stale"):
        _tensor(live) * 2
    with pytest.raises(RuntimeError, match="replaced"):
        head_gradient_targets(trainer, collector.backward(old)[0])


def test_registration_materializes_factory_state_and_persistent_buffers():
    trainer, _ = _trainer("student")
    source = TiedHead()
    state = execute_head_operation(
        trainer, "head_register", HeadRegistration("student", "head", "module", source)
    ).state
    assert list(state.parameters) == ["left"]
    assert list(state.buffers) == ["offset"]
    registered = trainer._checkpoint_slots["student"].custom["head"].value
    assert isinstance(registered, torch.nn.Module)
    assert registered.left is not source.left
    torch.testing.assert_close(state.parameters["left"], source.left)


@pytest.mark.parametrize("reentrant", (False, True))
def test_external_checkpoint_requires_explicit_snapshot(reentrant):
    trainer, rank = _trainer("student")
    head = rank.module("head", TiedHead, checkpoint="student")
    x = torch.tensor(3.0, requires_grad=True)
    old = checkpoint(head, x, use_reentrant=reentrant)
    head.left.data.fill_(4)
    with pytest.raises(RuntimeError, match="head.snapshot"):
        with trainer._gradient_transaction():
            old.backward()
    assert head.left.grad is None


@pytest.mark.parametrize("reentrant", (False, True))
def test_explicit_snapshot_supports_external_checkpoint(reentrant):
    trainer, rank = _trainer("student")
    head = rank.module("head", TiedHead, checkpoint="student")
    captured = head.snapshot()
    x = torch.tensor(3.0, requires_grad=True)
    old = checkpoint(captured, x, use_reentrant=reentrant)
    head.left.data.fill_(4)
    with trainer._gradient_transaction():
        old.backward()
    torch.testing.assert_close(head.left.grad, torch.tensor(13.0))
    torch.testing.assert_close(x.grad, torch.tensor(4.0))


def test_forward_hooks_see_captured_parameters_and_ties():
    trainer, rank = _trainer("student")
    observed = []

    def factory():
        source = TiedHead()
        source.register_forward_pre_hook(
            lambda module, inputs: observed.append(module.left.item())
        )
        return source

    head = rank.module("head", factory, checkpoint="student")
    head(torch.tensor(1.0))
    head.left.data.fill_(4)
    head(torch.tensor(1.0))
    assert observed == [2, 4]


def test_constructor_staleness_applies_to_heads_before_mutating_gradients():
    from art.trainer_rank._options import ForwardOptions

    trainer, rank = _trainer("student")
    setattr(trainer, "_forward_options", ForwardOptions(max_gradient_staleness=0))
    head = rank.module("head", TiedHead, checkpoint="student")
    old = head(torch.tensor(3.0))
    trainer._checkpoint_slots["student"].revision += 1
    with pytest.raises(RuntimeError, match="staleness"):
        with trainer._gradient_transaction():
            old.backward()
    assert head.left.grad is None


def _buffer_authority_worker(process_rank, init_method):
    from datetime import timedelta

    import torch.distributed as dist

    from art.trainer_rank._heads import synchronize_head_buffers

    dist.init_process_group(
        "gloo",
        rank=process_rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer, rank = _trainer("student")
        head = rank.module("bn", lambda: torch.nn.BatchNorm1d(2), checkpoint="student")
        for _ in range(process_rank + 1):
            head(torch.full((4, 2), float(process_rank + 1)))
        synchronize_head_buffers(trainer)
        torch.testing.assert_close(head.running_mean, torch.full((2,), 0.1))
        assert head.num_batches_tracked.item() == 1
    finally:
        dist.destroy_process_group()


def test_distributed_persistent_buffers_use_dp_zero_authority(tmp_path):
    torch.multiprocessing.spawn(
        _buffer_authority_worker,
        args=(f"file://{tmp_path / 'heads'}",),
        nprocs=2,
        join=True,
    )


def test_remote_reregistration_rejects_changed_ties():
    trainer, rank = _trainer("student")
    rank.module("head", TiedHead, checkpoint="student")
    untied = TiedHead()
    untied.right = torch.nn.Parameter(torch.tensor(2.0))
    with pytest.raises(ValueError, match="schema differs"):
        execute_head_operation(
            trainer,
            "head_register",
            HeadRegistration("student", "head", "module", untied),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("client", (False, True))
def test_cuda_checkpoint_head_across_completed_optimizer_update(monkeypatch, client):
    from art.trainer_rank._tensors import managed_tensor

    trainer, rank = _trainer("student")
    trainer.device = torch.device("cuda", 0)
    native = rank.module("head", lambda: TiedHead(False), checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "head", TiedHead(False), collector) if client else None
    head = native if live is None else _module(live)
    original_input = torch.tensor(3.0, device="cuda", requires_grad=True)
    old = head(managed_tensor(original_input) if client else original_input)
    with trainer._gradient_transaction():
        native(torch.tensor(1.0, device="cuda", requires_grad=True)).backward()
    _step(trainer, monkeypatch)
    if live is not None:
        live.refresh(export_head(trainer, "student", "head"))
        packets = collector.backward(old)
        trainer._commit_versioned_gradients(head_gradient_targets(trainer, packets[0]))
    else:
        with trainer._gradient_transaction():
            old.backward()
    torch.testing.assert_close(native.left.grad, torch.tensor(13.0, device="cuda"))
    torch.testing.assert_close(original_input.grad, torch.tensor(4.0, device="cuda"))


def test_client_buffer_reads_keep_old_graph_and_replacement_invalidates_handle():
    trainer, rank = _trainer("student")
    native = rank.buffer("scale", lambda: torch.tensor(2.0), checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "scale", torch.tensor(2.0), collector)
    x = torch.tensor(3.0, requires_grad=True)
    old = x * _tensor(live)
    native.fill_(4)
    live.refresh(export_head(trainer, "student", "scale"))
    collector.backward(old)
    torch.testing.assert_close(x.grad, torch.tensor(2.0))
    _tensor(live).add_(1)
    update = live.take_publication()
    assert update is not None
    assert update.buffers[""].item() == 5
    trainer._checkpoint_slots["student"].generation += 1
    live.refresh(export_head(trainer, "student", "scale"))
    with pytest.raises(RuntimeError, match="stale"):
        _tensor(live) + 1


def test_client_module_explicit_dtype_move_retains_ties_and_live_parameters():
    trainer, rank = _trainer("student")
    native = rank.module("head", TiedHead, checkpoint="student")
    live = _live_head(trainer, "head", TiedHead())
    head = _module(live).to("cpu").to(dtype=torch.float64)
    assert head.left.dtype == torch.float64
    assert head.offset.dtype == torch.float64
    assert head.left is head.right
    assert head.offset is head.other_offset
    native.left.data.fill_(4)
    trainer._checkpoint_slots["student"].revision += 1
    live.refresh(export_head(trainer, "student", "head"))
    output = head(torch.tensor(3.0, dtype=torch.float64))
    assert output.dtype == torch.float64
    assert output.item() == 53
    head.offset.add_(1)
    update = live.take_publication()
    assert update is not None
    assert update.buffers["offset"].dtype == torch.float32


def test_client_explicit_snapshot_supports_nonreentrant_checkpoint():
    trainer, rank = _trainer("student")
    native = rank.module("head", TiedHead, checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "head", TiedHead(), collector)
    captured = _module(live).snapshot()
    x = torch.tensor(3.0, requires_grad=True)
    old = checkpoint(captured, x, use_reentrant=False)
    native.left.data.fill_(4)
    trainer._checkpoint_slots["student"].revision += 1
    live.refresh(export_head(trainer, "student", "head"))
    packets = collector.backward(old)
    trainer._commit_versioned_gradients(head_gradient_targets(trainer, packets[0]))
    torch.testing.assert_close(native.left.grad, torch.tensor(13.0))
    torch.testing.assert_close(x.grad, torch.tensor(4.0))


@pytest.mark.parametrize("client", (False, True))
def test_buffer_item_and_bitwise_mutations_publish_without_losing_handle(client):
    trainer, rank = _trainer("student")
    native = rank.buffer("mask", lambda: torch.tensor([1, 2]), checkpoint="student")
    live = _live_head(trainer, "mask", torch.tensor([1, 2])) if client else None
    value = native if live is None else _tensor(live)
    value[0] = 4
    original = value
    value |= 1
    assert value is original
    torch.testing.assert_close(value, torch.tensor([5, 3]))
    if live is not None:
        update = live.take_publication()
        assert update is not None
        execute_head_operation(trainer, "head_publish", (update,))
        torch.testing.assert_close(native, torch.tensor([5, 3]))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda parameter: parameter.__setitem__(0, 9),
        lambda parameter: parameter.__iadd__(1),
        lambda parameter: parameter.requires_grad_(False),
        lambda parameter: parameter.data.fill_(9),
    ),
)
def test_client_parameter_mutations_fail_before_changing_owned_values(mutation):
    trainer, rank = _trainer("student")
    rank.parameter("gain", lambda: torch.tensor([2.0]), checkpoint="student")
    live = _live_head(trainer, "gain", torch.tensor([2.0]))
    with pytest.raises(RuntimeError, match="checkpoint parameters"):
        mutation(_tensor(live))
    torch.testing.assert_close(_tensor(live).detach(), torch.tensor([2.0]))


def test_head_export_preserves_strict_constructor_policy_for_client():
    from art.trainer_rank._options import ForwardOptions

    trainer, rank = _trainer("student")
    setattr(trainer, "_forward_options", ForwardOptions(max_gradient_staleness=0))
    rank.parameter("gain", lambda: torch.tensor(2.0), checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "gain", torch.tensor(2.0), collector)
    old = _tensor(live).square()
    trainer._checkpoint_slots["student"].revision += 1
    with pytest.raises(RuntimeError, match="staleness"):
        head_gradient_targets(trainer, collector.backward(old)[0])


def test_reusing_module_factory_value_does_not_share_checkpoint_storage():
    trainer, rank = _trainer("A", "B")
    source = TiedHead()
    first = rank.module("head", lambda: source, checkpoint="A")
    second = rank.module("head", lambda: source, checkpoint="B")
    first.left.data.fill_(4)
    assert first(torch.tensor(3.0)).item() == 53
    assert second(torch.tensor(3.0)).item() == 15
    assert source.left.item() == 2
    with trainer._gradient_transaction():
        second(torch.tensor(3.0)).backward()
    assert first.left.grad is None
    torch.testing.assert_close(second.left.grad, torch.tensor(13.0))


def test_registration_under_no_grad_preserves_authoritative_trainability():
    trainer, rank = _trainer("student")
    native = rank.module("head", TiedHead, checkpoint="student")
    collector = CotangentCollector()
    with torch.no_grad():
        live = _live_head(trainer, "head", TiedHead(), collector)
    assert _module(live).left.requires_grad
    packets = collector.backward(_module(live)(torch.tensor(3.0)))
    trainer._commit_versioned_gradients(head_gradient_targets(trainer, packets[0]))
    torch.testing.assert_close(native.left.grad, torch.tensor(13.0))
    with torch.no_grad():
        assert not _module(live)(torch.tensor(3.0)).requires_grad


@pytest.mark.parametrize("external", (False, True))
def test_client_reentrant_checkpoint_rejects_nested_remote_bridges(external):
    trainer, rank = _trainer("student")
    native = rank.module("head", TiedHead, checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "head", TiedHead(None if external else True), collector)
    x = torch.tensor(3.0, requires_grad=True)
    loss = (
        checkpoint(_module(live).snapshot(), x, use_reentrant=True)
        if external
        else _module(live)(x)
    )
    with pytest.raises(RuntimeError, match="nested remote backward is unsupported"):
        collector.backward(loss)
    assert native.left.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_batchnorm_explicit_placement_publishes_buffers_and_preserves_old_graph():
    from art.trainer_rank._tensors import managed_tensor

    trainer, rank = _trainer("student")
    trainer.device = torch.device("cuda", 0)
    native = rank.module("bn", lambda: torch.nn.BatchNorm1d(2), checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "bn", torch.nn.BatchNorm1d(2), collector)
    head = _module(live).to("cuda")
    head.eval()
    x = torch.tensor([[1.0, 3.0], [2.0, 5.0]], device="cuda", requires_grad=True)
    old = head(managed_tensor(x)).sum()
    head.train()
    head(torch.tensor([[4.0, 2.0], [8.0, 10.0]], device="cuda"))
    update = live.take_publication()
    assert update is not None
    execute_head_operation(trainer, "head_publish", (update,))
    live.refresh(export_head(trainer, "student", "bn"))
    assert native.num_batches_tracked.item() == 1
    torch.testing.assert_close(
        native.running_mean, torch.tensor([0.6, 0.6], device="cuda")
    )
    packets = collector.backward(old)
    trainer._commit_versioned_gradients(head_gradient_targets(trainer, packets[0]))
    torch.testing.assert_close(x.grad, torch.full_like(x, (1.0 + 1e-5) ** -0.5))
    torch.testing.assert_close(
        native.bias.grad, torch.tensor([2.0, 2.0], device="cuda")
    )


def test_remote_registration_of_existing_frozen_snapshot_keeps_frozen_parameters():
    trainer, rank = _trainer("snapshot")
    trainer._checkpoint_slots["snapshot"].snapshot = True
    rank.module("head", TiedHead, checkpoint="snapshot")
    source = TiedHead()
    state = execute_head_operation(
        trainer, "head_register", HeadRegistration("snapshot", "head", "module", source)
    ).state
    assert source.left.requires_grad
    assert not state.parameters["left"].requires_grad
    live = LiveHead(state, source, CotangentCollector())
    assert not _module(live).left.requires_grad
    x = torch.tensor(3.0, requires_grad=True)
    with trainer._gradient_transaction():
        _module(live)(x).backward()
    torch.testing.assert_close(x.grad, torch.tensor(4.0))


def test_client_reused_module_factory_does_not_share_checkpoint_handles():
    trainer, rank = _trainer("A", "B")
    rank.module("head", TiedHead, checkpoint="A")
    rank.module("head", TiedHead, checkpoint="B")
    source = TiedHead()
    first = _live_head(trainer, "head", source, checkpoint="A")
    second = _live_head(trainer, "head", source, checkpoint="B")
    _module(first).offset.add_(2)
    assert _module(first)(torch.tensor(3.0)).item() == 17
    assert _module(second)(torch.tensor(3.0)).item() == 15
    assert source.offset.item() == 1
    assert _module(first).left is not _module(second).left
    assert first.take_publication() is not None
    assert second.take_publication() is None


@pytest.mark.parametrize("operation", ("model_first", "parameter_first", "linear"))
def test_managed_model_operand_captures_live_parameter_and_keeps_old_version(operation):
    from art.trainer_rank._tensors import detach_tree

    trainer, rank = _trainer("student")
    native = rank.parameter(
        "weight", lambda: torch.tensor([2.0, 4.0]), checkpoint="student"
    )
    collector = CotangentCollector()
    live = _live_head(trainer, "weight", torch.zeros(2), collector)
    hidden = collector.attach(
        detach_tree("model", torch.tensor([3.0, 5.0], requires_grad=True)), managed=True
    )
    weight = _tensor(live)
    loss = (
        hidden @ weight
        if operation == "model_first"
        else weight @ hidden
        if operation == "parameter_first"
        else torch.nn.functional.linear(hidden, weight)
    )
    native.data.copy_(torch.tensor([7.0, 11.0]))
    trainer._checkpoint_slots["student"].revision += 1
    live.refresh(export_head(trainer, "student", "weight"))
    assert (hidden @ weight).item() == 76
    packets = collector.backward(loss)
    assert len(packets) == 2
    model = next(packet for packet in packets if packet.handle == "model")
    head = next(packet for packet in packets if packet.handle.startswith("head:"))
    torch.testing.assert_close(model.gradients[0], torch.tensor([2.0, 4.0]))
    trainer._commit_versioned_gradients(head_gradient_targets(trainer, head))
    torch.testing.assert_close(native.grad, torch.tensor([3.0, 5.0]))


@pytest.mark.parametrize("client", (False, True))
def test_module_buffer_reassignment_publishes(client):
    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.tensor(0))

        def forward(self, value):
            self.count = self.count + 1
            return value + self.count

    trainer, rank = _trainer("student")
    native = rank.module("head", Counter, checkpoint="student")
    live = _live_head(trainer, "head", Counter()) if client else None
    head = native if live is None else _module(live)
    assert head(torch.tensor(0)).item() == 1
    assert head(torch.tensor(0)).item() == 2
    assert head.count.item() == 2
    if live is not None:
        execute_head_operation(trainer, "head_publish", (live.take_publication(),))
    assert native.count.item() == 2


@pytest.mark.parametrize("client", (False, True))
def test_failed_buffer_shape_change_rolls_back_every_buffer(client):
    class Resize(torch.nn.Module):
        a: torch.Tensor
        b: torch.Tensor

        def __init__(self):
            super().__init__()
            self.register_buffer("a", torch.zeros(1))
            self.register_buffer("b", torch.zeros(1))

        def forward(self, value):
            self.a.add_(1)
            self.b.resize_(2)
            return value

    trainer, rank = _trainer("student")
    native = rank.module("head", Resize, checkpoint="student")
    live = _live_head(trainer, "head", Resize()) if client else None
    head = native if live is None else _module(live)
    before = export_head(trainer, "student", "head").buffer_revision
    with pytest.raises(ValueError, match="preserve buffer shape"):
        head(torch.tensor(0))
    assert head.a.item() == 0
    assert head.b.shape == (1,)
    assert export_head(trainer, "student", "head").buffer_revision == before
    if live is not None:
        assert live.take_publication() is None


@pytest.mark.parametrize("client", (False, True))
def test_failing_handle_forward_hook_does_not_publish_buffers(client):
    trainer, rank = _trainer("student")
    native = rank.module("head", lambda: torch.nn.BatchNorm1d(2), checkpoint="student")
    live = _live_head(trainer, "head", torch.nn.BatchNorm1d(2)) if client else None
    head = native if live is None else _module(live)

    def fail(*args):
        raise RuntimeError("forward hook failed")

    hook = head.register_forward_hook(fail)
    with pytest.raises(RuntimeError, match="forward hook failed"):
        head(torch.ones(4, 2))
    assert head.num_batches_tracked.item() == 0
    torch.testing.assert_close(head.running_mean, torch.zeros(2))
    if live is not None:
        assert live.take_publication() is None
    hook.remove()
    head(torch.ones(4, 2))
    assert head.num_batches_tracked.item() == 1


@pytest.mark.parametrize("client", (False, True))
@pytest.mark.parametrize("failure", (None, "pre", "forward", "post", "always"))
@pytest.mark.parametrize("pending_before", (False, True))
@pytest.mark.parametrize("use_saved_alias", (False, True))
def test_handle_hooks_share_one_buffer_publication(
    client, failure, pending_before, use_saved_alias
):
    calls = []

    class Counter(torch.nn.Module):
        count: torch.Tensor
        alias: torch.Tensor

        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.tensor(0.0))
            self.register_buffer("alias", self.count)

        def forward(self, value):
            assert self.count is self.alias
            self.count.add_(10)
            calls.append("forward")
            if failure == "forward":
                raise RuntimeError("forward failed")
            return value + self.count

    trainer, rank = _trainer("student")
    native = rank.module("head", Counter, checkpoint="student")
    live = _live_head(trainer, "head", Counter()) if client else None
    head = native if live is None else _module(live)
    initial = 7 if pending_before else 0
    if pending_before:
        head.count.add_(initial)
    revision = export_head(trainer, "student", "head").buffer_revision
    saved_alias = head.count

    def mutate(module, amount, phase):
        assert module.count is module.alias
        (saved_alias if use_saved_alias else module.count).add_(amount)
        calls.append(phase)
        if failure == phase:
            raise RuntimeError(f"{phase} failed")

    head.register_forward_pre_hook(lambda module, args: mutate(module, 1, "pre"))
    head.register_forward_hook(lambda module, args, out: mutate(module, 100, "post"))
    head.register_forward_hook(
        lambda module, args, out: mutate(module, 1000, "always"), always_call=True
    )
    if failure is not None:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            head(torch.tensor(2.0))
        assert calls[-1] == "always"
        expected = initial
    else:
        result = head(torch.tensor(2.0))
        assert calls == ["pre", "forward", "post", "always"]
        assert result.item() == initial + 13
        expected = initial + 1111
    assert head.count.item() == expected
    assert head.count is head.alias
    if live is not None:
        update = live.take_publication()
        assert (update is not None) == (pending_before or failure is None)
        if update is not None:
            execute_head_operation(trainer, "head_publish", (update,))
    assert native.count.item() == expected
    assert export_head(trainer, "student", "head").buffer_revision == revision + int(
        failure is None or (client and pending_before)
    )


@pytest.mark.parametrize("client", (False, True))
def test_handle_hook_parameter_gradients_keep_original_version(client):
    trainer, rank = _trainer("student")
    native = rank.module("head", TiedHead, checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "head", TiedHead(), collector) if client else None
    head = native if live is None else _module(live)
    parameter = head.left
    head.register_forward_hook(lambda module, args, out: out + module.left.square())
    value = torch.tensor(3.0, requires_grad=True)
    old = head(value)
    native.left.data.fill_(4)
    trainer._checkpoint_slots["student"].revision += 1
    if live is not None:
        live.refresh(export_head(trainer, "student", "head"))
        packets = collector.backward(old)
        assert len(packets) == 1
        trainer._commit_versioned_gradients(head_gradient_targets(trainer, packets[0]))
    else:
        with trainer._gradient_transaction():
            old.backward()
    assert head.left is parameter
    torch.testing.assert_close(native.left.grad, torch.tensor(17.0))
    torch.testing.assert_close(value.grad, torch.tensor(4.0))


@pytest.mark.parametrize("client", (False, True))
def test_recursive_handle_hook_does_not_publish_before_outer_failure(client):
    trainer, rank = _trainer("student")
    native = rank.module("head", TiedHead, checkpoint="student")
    live = _live_head(trainer, "head", TiedHead()) if client else None
    head = native if live is None else _module(live)

    def recurse(module, args):
        module.offset.add_(1)
        if args[0].item() == 1:
            module(torch.tensor(2.0))
            raise RuntimeError("outer call failed")

    head.register_forward_pre_hook(recurse)
    with pytest.raises(RuntimeError, match="outer call failed"):
        head(torch.tensor(1.0))
    assert head.offset.item() == 1
    assert export_head(trainer, "student", "head").buffer_revision == 0
    if live is not None:
        assert live.take_publication() is None


@pytest.mark.parametrize("client", (False, True))
@pytest.mark.parametrize(
    "scenario",
    ("alias_failure", "reentry_failure", "reentry_success", "successful_inner"),
)
def test_nested_handle_calls_preserve_active_captures(client, scenario):
    trainer, rank = _trainer("student")
    native = {
        name: rank.module(name, TiedHead, checkpoint="student")
        for name in ("outer", "inner")
    }
    collector = CotangentCollector()
    live = (
        {name: _live_head(trainer, name, TiedHead(), collector) for name in native}
        if client
        else {}
    )
    heads = {name: _module(value) for name, value in live.items()} if client else native
    outer, inner = heads["outer"], heads["inner"]
    outer_alias = outer.offset
    seen = []

    def outer_hook(module, args):
        seen.append(module.offset.item())
        module.offset.add_(10)
        if args[0].item() == 0:
            inner(torch.tensor(1.0))
            if scenario == "successful_inner":
                raise RuntimeError("outer failed")

    def inner_hook(module, args):
        module.offset.add_(100)
        outer_alias.add_(1)
        if scenario.startswith("reentry"):
            outer(torch.tensor(2.0))
        if scenario.endswith("failure"):
            raise RuntimeError("inner failed")

    outer.register_forward_pre_hook(outer_hook)
    inner.register_forward_pre_hook(inner_hook)
    if scenario == "reentry_success":
        assert outer(torch.tensor(0.0)).item() == 24
    else:
        with pytest.raises(RuntimeError, match="failed"):
            outer(torch.tensor(0.0))
    assert seen == ([1, 12] if scenario.startswith("reentry") else [1])
    expected = {
        "outer": 22 if scenario == "reentry_success" else 1,
        "inner": 101 if scenario in ("reentry_success", "successful_inner") else 1,
    }
    for name, value in heads.items():
        assert value.offset.item() == expected[name]
        if client:
            update = live[name].take_publication()
            assert (update is not None) == (expected[name] != 1)
            if update is not None:
                execute_head_operation(trainer, "head_publish", (update,))
        assert native[name].offset.item() == expected[name]
        assert export_head(trainer, "student", name).buffer_revision == int(
            expected[name] != 1
        )


def test_inplace_operation_snapshots_readonly_checkpoint_parameter():
    trainer, rank = _trainer("student")
    parameter = rank.parameter(
        "weight", lambda: torch.tensor(2.0), checkpoint="student"
    )
    input_value = torch.tensor(3.0, requires_grad=True)
    original = (input_value * 1).mul_(parameter)
    parameter.data.fill_(7)
    with trainer._gradient_transaction():
        original.backward()
    torch.testing.assert_close(parameter.grad, torch.tensor(3.0))
    torch.testing.assert_close(input_value.grad, torch.tensor(2.0))
    parameter.grad = None
    stale = torch.tensor(3.0).mul_(parameter)
    trainer._checkpoint_slots["student"].revision += 3
    with (
        pytest.raises(RuntimeError, match="staleness"),
        trainer._gradient_transaction(),
    ):
        stale.backward()
    assert parameter.grad is None


def _live_buffer_authority_worker(process_rank, init_method, asymmetric=False):
    from datetime import timedelta

    import torch.distributed as dist

    from art.trainer_rank._heads import synchronize_head_buffers

    dist.init_process_group(
        "gloo",
        rank=process_rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=30),
    )
    try:
        trainer, rank = _trainer("student")
        native = rank.module(
            "head", lambda: torch.nn.BatchNorm1d(2), checkpoint="student"
        )
        if asymmetric:
            if process_rank == 1:
                if asymmetric == "buffers":
                    del native._buffers["running_mean"]
                else:
                    del trainer._checkpoint_slots["student"].custom["head"]
            with pytest.raises(
                (RuntimeError, ValueError),
                match="registrations differ|preserve buffer names",
            ):
                synchronize_head_buffers(trainer)
            dist.barrier()
            return
        live = _live_head(trainer, "head", torch.nn.BatchNorm1d(2))
        if process_rank == 1:
            _module(live)(torch.ones(4, 2))
        update = live.take_publication()
        execute_head_operation(
            trainer, "head_publish", () if update is None else (update,)
        )
        synchronize_head_buffers(trainer)
        live.refresh(export_head(trainer, "student", "head"))
        assert native.num_batches_tracked.item() == 0
        assert _module(live).num_batches_tracked.item() == 0
        synchronized_revision = live.state.buffer_revision
        synchronize_head_buffers(trainer)
        assert (
            export_head(trainer, "student", "head").buffer_revision
            == synchronized_revision
        )
        if process_rank == 1:
            _module(live)(torch.ones(4, 2))
        update = live.take_publication()
        execute_head_operation(
            trainer, "head_publish", () if update is None else (update,)
        )
        assert native.num_batches_tracked.item() == process_rank
    finally:
        dist.destroy_process_group()


def test_distributed_live_buffer_refresh_accepts_dp_zero_authority(tmp_path):
    torch.multiprocessing.spawn(
        _live_buffer_authority_worker,
        args=(f"file://{tmp_path / 'live_heads'}",),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("kind", ("parameter", "buffer"))
def test_inplace_operation_snapshots_readonly_client_tensor(kind):
    trainer, rank = _trainer("student")
    factory = lambda: torch.tensor(2.0)
    native = getattr(rank, kind)("scale", factory, checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "scale", factory(), collector)
    x = torch.tensor(3.0, requires_grad=True)
    loss = (x * 1).mul_(_tensor(live))
    with torch.no_grad():
        native.fill_(7)
    trainer._checkpoint_slots["student"].revision += 1
    live.refresh(export_head(trainer, "student", "scale"))
    packets = collector.backward(loss)
    torch.testing.assert_close(x.grad, torch.tensor(2.0))
    assert live.take_publication() is None
    if kind == "parameter":
        trainer._commit_versioned_gradients(head_gradient_targets(trainer, packets[0]))
        torch.testing.assert_close(native.grad, torch.tensor(3.0))


def test_logical_callback_reentrant_head_rejects_before_gradient_publication():
    from types import SimpleNamespace

    from art.trainer_rank._heads import logical_register_head

    trainer, _ = _trainer("student")
    collector = CotangentCollector()

    def invoke(operation, kind, payload):
        assert operation == "head"
        return execute_head_operation(trainer, kind, payload)

    view = SimpleNamespace(
        _rank=trainer,
        _invoke=invoke,
        _executor=SimpleNamespace(
            state=SimpleNamespace(collector=collector), invoke=invoke
        ),
        device=torch.device("cpu"),
    )

    def callback(rank):
        head = logical_register_head(
            rank, "module", "head", lambda: TiedHead(True), checkpoint="student"
        )
        loss = head(torch.tensor(3.0, requires_grad=True))
        # The logical executor submits packets only after local collection succeeds.
        packets = collector.backward(loss)
        trainer._commit_versioned_gradients(
            [
                target
                for packet in packets
                for target in head_gradient_targets(trainer, packet)
            ]
        )

    with pytest.raises(RuntimeError, match="nested remote backward is unsupported"):
        callback(view)
    custom = trainer._checkpoint_slots["student"].custom["head"].value
    assert isinstance(custom, torch.nn.Module)
    assert all(parameter.grad is None for parameter in custom.parameters())
    assert (
        getattr(trainer, "_logical_head_handles")[
            ("student", "head")
        ].take_publication()
        is None
    )


@pytest.mark.parametrize("mode", ("rank", "zero"))
@pytest.mark.parametrize("kind", ("buffer", "module"))
def test_logical_registration_recovers_after_rejected_buffer_publication(mode, kind):
    async def run():
        trainer, rank = _trainer("student")
        factory = TiedHead if kind == "module" else lambda: torch.tensor(1.0)
        native = getattr(rank, kind)("head", factory, checkpoint="student")
        retained = []

        def buffer(head):
            return head.offset if kind == "module" else head

        def register(view):
            return getattr(view, kind)("head", factory, checkpoint="student")

        def conflict(view):
            old = register(view)
            retained.append(old)
            buffer(old).add_(1)
            # The logical copy now has a publication against the old revision.
            buffer(native).add_(5)

        with pytest.raises(RuntimeError, match="changed before publication"):
            await run_rank_callback(trainer, conflict, mode=mode)
        (old,) = retained
        with pytest.raises(RuntimeError, match="publication failed"):
            buffer(old).item()
        assert buffer(native).item() == 6

        def recover(view):
            fresh = register(view)
            assert fresh is not old
            assert register(view) is fresh
            assert buffer(fresh).item() == 6
            if kind == "module":
                assert fresh.left is fresh.right
                assert fresh.offset is fresh.other_offset
                assert fresh(torch.tensor(3.0)).item() == 20
            buffer(fresh).add_(2)
            return fresh

        fresh = (await run_rank_callback(trainer, recover, mode=mode)).value
        assert buffer(native).item() == 8
        assert (await run_rank_callback(trainer, register, mode=mode)).value is fresh
        assert buffer(fresh).item() == 8
        with pytest.raises(RuntimeError, match="publication failed"):
            buffer(old).item()

    asyncio.run(run())


@pytest.mark.parametrize("client", (False, True))
@pytest.mark.parametrize("grad_enabled", (False, True))
@pytest.mark.parametrize(
    "view",
    (
        lambda value: value[1:],
        lambda value: value.view(2, 2),
        lambda value: value.narrow(0, 1, 2),
    ),
)
def test_live_buffer_view_mutation_rejects_without_silent_write(
    client, grad_enabled, view
):
    trainer, rank = _trainer("student")
    native = rank.buffer("stats", lambda: torch.zeros(4), checkpoint="student")
    live = _live_head(trainer, "stats", torch.zeros(4)) if client else None
    buffer = native if live is None else _tensor(live)
    revision = export_head(trainer, "student", "stats").buffer_revision
    with torch.set_grad_enabled(grad_enabled):
        snapshot = view(buffer)
        with pytest.raises(
            RuntimeError, match="Views of live checkpoint buffers are read-only"
        ):
            snapshot.fill_(7)
        copy = snapshot.clone()
        copy.fill_(7)
        buffer[2:] = 7
    torch.testing.assert_close(buffer, torch.tensor([0.0, 0.0, 7.0, 7.0]))
    if live is not None:
        update = live.take_publication()
        assert update is not None
        execute_head_operation(trainer, "head_publish", (update,))
    assert export_head(trainer, "student", "stats").buffer_revision == revision + 1


@pytest.mark.parametrize("client", (False, True))
def test_functional_batchnorm_no_grad_publishes_buffer_changes(client):
    trainer, rank = _trainer("student")
    native = rank.buffer("mean", lambda: torch.zeros(2), checkpoint="student")
    live = _live_head(trainer, "mean", torch.zeros(2)) if client else None
    mean = native if live is None else _tensor(live)
    before = export_head(trainer, "student", "mean").buffer_revision
    with torch.no_grad():
        torch.nn.functional.batch_norm(
            torch.ones(4, 2), mean, torch.ones(2), training=True
        )
    torch.testing.assert_close(mean, torch.full((2,), 0.1))
    if live is not None:
        update = live.take_publication()
        assert update is not None
        execute_head_operation(trainer, "head_publish", (update,))
    assert export_head(trainer, "student", "mean").buffer_revision == before + 1


@pytest.mark.parametrize("client", (False, True))
def test_live_parameter_metadata_does_not_capture_weights(monkeypatch, client):
    trainer, rank = _trainer("student")
    native = rank.parameter("weight", lambda: torch.zeros(3, 4), checkpoint="student")
    live = _live_head(trainer, "weight", torch.zeros(3, 4)) if client else None
    value = native if live is None else _tensor(live)

    def unexpected(*args, **kwargs):
        raise AssertionError("metadata inspection must not capture parameter values")

    monkeypatch.setattr(trainer, "_snapshot_parameter", unexpected)
    if live is not None:
        monkeypatch.setattr(live, "capture", unexpected)
    assert value.size() == (3, 4)
    assert value.numel() == 12
    assert value.dim() == 2
    assert value.shape == (3, 4)
    assert value.dtype == torch.float32
    assert value.device == torch.device("cpu")
    assert value.requires_grad
    assert value.is_leaf
    assert value.grad_fn is None
    assert value.grad is None


@pytest.mark.parametrize("client", (False, True))
@pytest.mark.parametrize("property_name", ("T", "mT", "H", "mH", "real", "imag"))
@pytest.mark.parametrize("complex_dtype", (False, True))
def test_parameter_tensor_properties_capture_immutable_versions(
    client, property_name, complex_dtype
):
    initial = torch.tensor([[1 + 2j, 3 - 4j], [5 + 6j, 7 - 8j]])
    if not complex_dtype:
        if property_name == "imag":
            pytest.skip("imag requires a complex tensor")
        initial = initial.real.clone()
    trainer, rank = _trainer("student")
    native = rank.parameter("weight", lambda: initial.clone(), checkpoint="student")
    collector = CotangentCollector()
    live = _live_head(trainer, "weight", initial, collector) if client else None
    value = native if live is None else _tensor(live)
    expected = initial.clone().requires_grad_()
    getattr(expected, property_name).abs().square().sum().backward()
    old = getattr(value, property_name).abs().square().sum()
    stale = getattr(value, property_name).abs().square().sum()
    native.data.copy_(initial * 3)
    trainer._checkpoint_slots["student"].revision += 1
    if live is not None:
        live.refresh(export_head(trainer, "student", "weight"))
    torch.testing.assert_close(
        getattr(value, property_name), getattr(initial * 3, property_name)
    )
    if live is None:
        with trainer._gradient_transaction():
            old.backward()
    else:
        packets = collector.backward(old)
        assert len(packets) == 1
        trainer._commit_versioned_gradients(head_gradient_targets(trainer, packets[0]))
        assert value.grad is None
    torch.testing.assert_close(native.grad, expected.grad)
    trainer._checkpoint_slots["student"].revision += 2
    with pytest.raises(RuntimeError, match="staleness"):
        if live is None:
            with trainer._gradient_transaction():
                stale.backward()
        else:
            head_gradient_targets(trainer, collector.backward(stale)[0])
    torch.testing.assert_close(native.grad, expected.grad)


@pytest.mark.parametrize("client", (False, True))
@pytest.mark.parametrize("property_name", ("T", "mT", "H", "mH", "real", "imag"))
def test_buffer_tensor_properties_reject_unpublished_mutation(client, property_name):
    initial = torch.ones(2, 2, dtype=torch.complex64)
    trainer, rank = _trainer("student")
    native = rank.buffer("stats", lambda: initial.clone(), checkpoint="student")
    live = _live_head(trainer, "stats", initial) if client else None
    value = native if live is None else _tensor(live)
    with pytest.raises(RuntimeError, match="Views of live checkpoint buffers"):
        getattr(value, property_name).fill_(7)
    torch.testing.assert_close(native, initial)
    assert export_head(trainer, "student", "stats").buffer_revision == 0
    if live is not None:
        assert live.take_publication() is None


@pytest.mark.parametrize("asymmetric", (True, "buffers"))
def test_distributed_buffer_registration_mismatch_fails_on_every_rank(
    tmp_path, asymmetric
):
    torch.multiprocessing.spawn(
        _live_buffer_authority_worker,
        args=(f"file://{tmp_path / 'mismatched_heads'}", asymmetric),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("client", (False, True))
def test_module_buffer_view_mutation_rejects_without_publishing(client):
    trainer, rank = _trainer("student")
    native = rank.module("head", lambda: torch.nn.BatchNorm1d(2), checkpoint="student")
    live = _live_head(trainer, "head", torch.nn.BatchNorm1d(2)) if client else None
    head = native if live is None else _module(live)
    with pytest.raises(
        RuntimeError, match="Views of live checkpoint buffers are read-only"
    ):
        head.running_mean[:1].fill_(7)
    torch.testing.assert_close(head.running_mean, torch.zeros(2))
    if live is not None:
        assert live.take_publication() is None


@pytest.mark.parametrize("client", (False, True))
def test_stateful_function_on_buffer_snapshot_view_rejects(client):
    trainer, rank = _trainer("student")
    native = rank.buffer("mean", lambda: torch.zeros(2), checkpoint="student")
    live = _live_head(trainer, "mean", torch.zeros(2)) if client else None
    mean = native if live is None else _tensor(live)
    with (
        torch.no_grad(),
        pytest.raises(
            RuntimeError, match="Stateful operations on buffer snapshot views"
        ),
    ):
        torch.nn.functional.batch_norm(
            torch.ones(4, 2), mean[:], torch.ones(2), training=True
        )
    torch.testing.assert_close(mean, torch.zeros(2))
    if live is not None:
        assert live.take_publication() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("client", (False, True))
def test_cuda_live_buffer_views_and_functional_publication(client):
    trainer, rank = _trainer("student")
    trainer.device = torch.device("cuda", 0)
    native = rank.buffer("mean", lambda: torch.zeros(2), checkpoint="student")
    live = (
        _live_head(trainer, "mean", torch.zeros(2, device="cuda")) if client else None
    )
    mean = native if live is None else _tensor(live)
    with pytest.raises(
        RuntimeError, match="Views of live checkpoint buffers are read-only"
    ):
        mean[:].fill_(7)
    with torch.no_grad():
        torch.nn.functional.batch_norm(
            torch.ones(4, 2, device="cuda"),
            mean,
            torch.ones(2, device="cuda"),
            training=True,
        )
    if live is not None:
        update = live.take_publication()
        assert update is not None
        execute_head_operation(trainer, "head_publish", (update,))
    torch.testing.assert_close(native, torch.full((2,), 0.1, device="cuda"))
    assert export_head(trainer, "student", "mean").buffer_revision == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_buffer_sync_stages_cpu_authority_before_comparison(tmp_path):
    import torch.distributed as dist

    from art.trainer_rank._heads import synchronize_head_buffers

    dist.init_process_group(
        "gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'cuda_sync'}"
    )
    try:
        trainer, rank = _trainer("student")
        trainer.device = torch.device("cuda", 0)
        buffer = rank.buffer("mean", lambda: torch.ones(2), checkpoint="student")
        synchronize_head_buffers(trainer)
        torch.testing.assert_close(buffer, torch.ones(2, device="cuda"))
        assert export_head(trainer, "student", "mean").buffer_revision == 0
    finally:
        dist.destroy_process_group()
