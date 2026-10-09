"""Driver CPU transport stays separate from native output placement."""

import asyncio
from dataclasses import replace
import gc
import sys
from typing import Any, cast
import weakref

import pytest
from test_trainer_rank_commands import _input, _Rank
import torch

from art.trainer_rank import ForwardInput, ForwardOptions, ForwardOutput, _tensors
from art.trainer_rank._commands import _Executor, _view
from art.trainer_rank._operations import TrainerOperation, execute_operation
from art.trainer_rank._options import resolve_forward_options
from art.trainer_rank._tensors import CotangentCollector


class _TransportRank(_Rank):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.weight = torch.nn.Parameter(self.weight.detach().to(device))
        self.device = self.weight.device
        self.policies = []
        self.worker_reject = False

    def forward(self, tree, **kwargs):
        if not isinstance(tree, ForwardInput):
            return [self.forward(child, **kwargs) for child in tree]
        policy = resolve_forward_options(
            method=kwargs.get("options"), input=tree.options
        )
        self.policies.append(policy)
        if self.worker_reject:
            raise MemoryError("worker admission rejected")
        with torch.set_grad_enabled(not kwargs.get("no_grad", False)):
            value = (
                tree.input_tokens.to(self.device).float() * self.weight.clone().square()
            )
            if policy.output_device == "cpu":
                value = value.cpu()
        return ForwardOutput(None, None, None, value)


def _operation(view, kind, payload, identity=None):
    return execute_operation(
        view, TrainerOperation.capture((identity or str(id(payload)), 1), kind, payload)
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batches", [False, True])
async def test_driver_cpu_exports_preserve_old_gradients_and_worker_policy(
    device, batches
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    rank: Any = _TransportRank(device)
    view = _view(_Executor(rank, "zero"))
    client = CotangentCollector()
    request = replace(
        _input(3),
        input_tokens=torch.tensor([3], device=device),
        options=ForwardOptions(output_device="model"),
    )
    # Physical worker succeeds, but no aggregate GPU copy can be admitted.
    rank._available_memory_bytes = lambda: 0
    if device == "cuda":
        with pytest.raises(MemoryError, match="Gathered model-device outputs"):
            view.forward(request)
    seen = []
    attach = view._attach

    def check_cpu(packet):
        assert all(packet.cpu)
        assert all(tensor.device.type == "cpu" for tensor in packet.packet.tensors)
        seen.append(packet.packet.handle)
        # Earlier cycles holding CUDA tensors must not be collected inside attach.
        gc.collect()
        before = torch.cuda.memory_allocated() if device == "cuda" else 0
        result = attach(packet)
        if device == "cuda":
            assert torch.cuda.memory_allocated() == before
        return result

    # A shallow operation view preserves this bound observer; its delegate
    # inspects the actual placement before collector.attach can copy anything.
    setattr(view, "_attach", check_cpu)

    async def forward(identity):
        if batches:
            handle = await _operation(
                view, "batches_open", {"inputs": [request]}, identity + ":open"
            )
            packet = await _operation(
                view, "batches_next", {"handle": handle}, identity
            )
            await _operation(
                view, "batches_close", {"handle": handle}, identity + ":close"
            )
            return client.attach(packet, managed=True).outputs[0].hidden_states
        packet = await _operation(view, "forward", {"inputs": request}, identity)
        return client.attach(packet, managed=True).hidden_states

    old = await forward("old")
    with torch.no_grad():
        rank.weight.add_(1)
    fresh = await forward("fresh")
    assert view._transport_handles is None
    assert old.device.type == fresh.device.type == "cpu"
    assert len(seen) == 2
    assert all(policy.output_device == "model" for policy in rank.policies)
    state = rank._rank_command_state
    assert len(state.exports) == 2
    assert all(
        tensor.device.type == "cpu"
        for tensors in state.exports.values()
        for tensor in tensors
    )
    assert all(
        tensor.device == rank.device
        for tensors in state.graphs.values()
        for tensor in tensors
    )
    head = torch.nn.Parameter(torch.tensor(5.0, device=device))
    loss = (old * fresh * head).sum()
    for retained in (True, False):
        packets = client.backward(loss, retain_graph=retained)
        await _operation(
            view,
            "backward",
            {"packets": packets, "retain_graph": retained},
            "backward:" + str(retained),
        )
        assert len(state.exports) == (2 if retained else 0)
    assert rank.weight.grad is not None and head.grad is not None
    # 3*w_old^2 * 3*w_new^2 * head, differentiated into the same parameter.
    torch.testing.assert_close(rank.weight.grad, torch.tensor(5400.0, device=device))
    torch.testing.assert_close(head.grad, torch.tensor(648.0, device=device))
    assert not state.graphs
    setattr(view, "_attach", attach)
    rank._available_memory_bytes = lambda: 1 << 60
    native = view.forward(request)
    assert native.hidden_states.device == rank.device
    view.backward(native.hidden_states.sum())
    assert not state.graphs


@pytest.mark.parametrize("policy", ["model", "cpu", "auto"])
async def test_transport_preserves_worker_admission_and_native_view(policy):
    rank: Any = _TransportRank()
    view = _view(_Executor(rank, "zero"))
    rank.worker_reject = True
    request = replace(_input(3), options=ForwardOptions(output_device=policy))
    with pytest.raises(MemoryError, match="worker admission rejected"):
        await _operation(view, "forward", {"inputs": request})
    assert rank.policies[0].output_device == policy
    assert view._transport_handles is None
    assert not rank._rank_command_state.exports
    assert not rank._rank_command_state.graphs


@pytest.mark.parametrize(
    "failure_kind", ["budget", "allocation", "release", "cancel", "attach"]
)
@pytest.mark.parametrize("batches", [False, True])
async def test_export_failure_releases_only_failed_operation_and_counts_live_exports(
    monkeypatch, failure_kind, batches
):
    rank: Any = _TransportRank()
    view = _view(_Executor(rank, "zero"))
    state = rank._rank_command_state
    # Additional headroom excludes the CPU payloads held by earlier exports,
    # mirroring fresh host/cgroup accounting used by the real rank.
    total = 64
    observed = []

    def available():
        used = sum(
            t.untyped_storage().nbytes()
            for values in state.exports.values()
            for t in values
        )
        observed.append(used)
        return total - used

    rank._available_cpu_memory_bytes = available
    request = _input(3)
    good = await _operation(view, "forward", {"inputs": request}, "good")
    assert sum(t.numel() * t.element_size() for t in state.exports[good.handle]) == 4
    original_graphs = set(state.graphs)
    handle = (
        await _operation(view, "batches_open", {"inputs": [request, _input(7)]}, "open")
        if batches
        else None
    )
    operation = TrainerOperation.capture(
        ("failure", 1),
        "batches_next" if batches else "forward",
        {"handle": handle} if batches else {"inputs": request},
    )
    error = (asyncio.CancelledError if failure_kind == "cancel" else MemoryError)(
        "injected transport delivery failure"
    )
    release_fails = failure_kind in ("release", "cancel", "attach")
    invoke = view._executor.invoke
    release_calls, flushes = [], []

    def release(operation, *args, **kwargs):
        if operation == "release":
            release_calls.append(args[0])
            if release_fails and len(release_calls) <= 2:
                raise RuntimeError("injected release failure")
        return invoke(operation, *args, **kwargs)

    monkeypatch.setattr(view._executor, "invoke", release)
    monkeypatch.setattr(view, "_flush_heads", lambda: flushes.append(sys.exc_info()[1]))
    # The worker snapshot fits. A budget drop at export must fail before
    # its clone and clean only the newly created physical graphs.
    calls = 0

    def constrained():
        nonlocal calls
        calls += 1
        return available() if calls == 1 else 0

    detach, managed = _tensors.detach_tree, _tensors.managed_tensor
    if failure_kind == "budget":
        rank._available_cpu_memory_bytes = constrained
    elif failure_kind == "attach":

        def reject_attach(tensor):
            raise error

        monkeypatch.setattr(_tensors, "managed_tensor", reject_attach)
    else:

        def reject_clone(handle, *args, **kwargs):
            if handle.startswith("client:"):
                raise error
            return detach(handle, *args, **kwargs)

        monkeypatch.setattr(_tensors, "detach_tree", reject_clone)
    with pytest.raises(
        type(error), match="export snapshot|transport delivery"
    ) as failure:
        await execute_operation(view, operation)
    assert failure.value.__traceback__ is not None
    if failure_kind != "budget":
        assert failure.value is error
    assert len(release_calls) == 1 and flushes and not any(flushes)
    assert (set(state.graphs) != original_graphs) is release_fails
    assert state.released == set(state.graphs) - original_graphs
    with pytest.raises(type(error), match=str(failure.value)) as replay:
        await execute_operation(view, operation)
    assert replay.value is not failure.value and len(release_calls) == 1
    assert set(state.exports) == {good.handle}
    assert 4 in observed
    rank._available_cpu_memory_bytes = available
    monkeypatch.setattr(_tensors, "detach_tree", detach)
    monkeypatch.setattr(_tensors, "managed_tensor", managed)
    if release_fails:
        before = len(flushes)
        with pytest.raises(RuntimeError, match="injected release failure"):
            view.optim_step()
        assert len(release_calls) == 2 and len(flushes) == before and rank.steps == 0
        assert state.released == set(state.graphs) - original_graphs
    assert view.optim_step() == {"steps": 1}
    assert not state.released and set(state.graphs) == original_graphs
    await _operation(view, "release", {"handles": [good.handle]}, "release")
    assert not state.exports
    assert not state.graphs
    assert available() == total
    if batches and failure_kind == "attach":
        assert not state.iterators and not state.batch_inputs
        handle = await _operation(
            view, "batches_open", {"inputs": [_input(7)]}, "reopen"
        )
    packet = await _operation(
        view,
        "batches_next" if batches else "forward",
        {"handle": handle} if batches else {"inputs": _input(7)},
        "recovered",
    )
    client = CotangentCollector()
    output = client.attach(packet)
    value = (output.outputs[0] if batches else output).hidden_states
    await _operation(view, "backward", {"packets": client.backward(value.sum())})
    assert rank.weight.grad.item() == 28
    if batches:
        await _operation(view, "batches_close", {"handle": handle}, "close")
    assert not state.graphs and not state.exports
    assert not state.iterators and not state.batch_inputs


def test_transport_release_drops_cpu_payload_without_collecting_cycles():
    async def run():
        rank: Any = _TransportRank()
        view = _view(_Executor(rank, "zero"))
        packet = await _operation(view, "forward", {"inputs": _input(3)}, "forward")
        state = rank._rank_command_state
        references = [weakref.ref(t) for t in state.exports[packet.handle]]
        await _operation(view, "release", {"handles": [packet.handle]}, "release")
        assert all(ref() is None for ref in references)
        assert not state.exports and not state.graphs

    enabled = gc.isenabled()
    gc.disable()
    try:
        asyncio.run(run())
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("kind", ["forward", "batches_open"])
async def test_abandoned_reply_releases_native_graphs_and_iterators(kind):
    rank = cast(Any, _TransportRank())
    view = _view(_Executor(rank, "zero"))
    state = rank._rank_command_state
    operation = TrainerOperation.capture(
        ("client", 1),
        kind,
        {"inputs": _input(3) if kind == "forward" else [_input(3)]},
    )
    await execute_operation(view, operation)
    if kind == "forward":
        assert state.graphs and state.exports
    else:
        assert state.iterators and state.batch_inputs
    acknowledgement = TrainerOperation.capture(operation.id, "acknowledge", ((), (1,)))
    await execute_operation(view, acknowledgement)
    await execute_operation(view, acknowledgement)
    assert not state.graphs and not state.exports
    assert not state.iterators and not state.batch_inputs
    assert not rank._operation_outcomes.outcomes
