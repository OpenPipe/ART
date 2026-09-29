"""Undelivered logical outputs must not retain physical forward graphs."""

from collections.abc import Generator
from dataclasses import replace
from functools import partial
import gc
from typing import Any
import weakref

import pytest
from test_trainer_rank_commands import _input, _Rank
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from art.trainer_rank import (
    ForwardInput,
    ForwardOutput,
    MicroBatch,
    MicroBatchStats,
    _memory_policy,
    _tensors,
)
from art.trainer_rank._commands import _Executor, _view
from art.trainer_rank._graphs import GraphCache
from art.trainer_rank._tensors import CotangentCollector, detach_tree


class _CachedRank(_Rank):
    def __init__(self, *args):
        super().__init__(*args)
        self.cache = GraphCache()
        self.collector = CotangentCollector()
        self.records = []

    def forward(self, tree, **kwargs):
        if not isinstance(tree, ForwardInput):
            return super().forward(tree, **kwargs)
        handle, (value,) = self.cache.run(
            lambda tokens: (tokens.float() * self.weight,), tree.input_tokens
        )
        self.records.append(weakref.ref(self.cache._records[handle]))
        return self.collector.attach(
            detach_tree(handle, ForwardOutput(None, None, None, value)),
            on_release=partial(self.cache.release, handle),
        )

    def _forward_graph_cache(self):
        return self.cache

    def _forward_cotangent_collector(self):
        return self.collector

    def _forward_memory_group(self):
        return None


def _fail_delivery(monkeypatch, executor, kind, *, packet_number=1):
    """Fail placement, copy, or collection after physical graph registration."""
    error = MemoryError("injected logical output delivery failure")
    packet = executor._packet
    targets, partial_outputs = set(), []
    count = 0

    def capture(*args):
        nonlocal count
        output = packet(*args)
        count += 1
        if count == packet_number:
            targets.update(id(tensor) for tensor in output.packet.tensors)
        return replace(output, cpu=(False,) * len(output.cpu), managed=kind == "attach")

    monkeypatch.setattr(executor, "_packet", capture)
    if kind == "copy":
        to = torch.Tensor.to

        def fail_copy(tensor, *args, **kwargs):
            if id(tensor) in targets:
                raise error
            return to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", fail_copy)
    elif kind == "placement":

        def fail_placement(*args, **kwargs):
            raise error

        monkeypatch.setattr(_memory_policy, "choose_output_placements", fail_placement)
    else:
        managed = _tensors.managed_tensor
        calls = 0

        def fail_attach(tensor):
            nonlocal calls
            calls += 1
            if calls == packet_number:
                # The bridge and its finalizer already exist. Retain its proxy
                # even after the error, so collection cannot repair the leak.
                partial_outputs.append(tensor)
                raise error
            return managed(tensor)

        monkeypatch.setattr(_tensors, "managed_tensor", fail_attach)
    return error, partial_outputs


@pytest.mark.parametrize("mode", ["rank", "zero"])
@pytest.mark.parametrize("kind", ["copy", "attach", "placement"])
def test_failed_delivery_releases_registered_graph_and_native_cache(
    monkeypatch, mode, kind
):
    rank: Any = _CachedRank()
    executor = _Executor(rank, mode)
    view = _view(executor)
    with monkeypatch.context() as patch:
        error, partial_outputs = _fail_delivery(patch, executor, kind)
        with pytest.raises(MemoryError) as failure:
            view.forward(_input(3))
        assert failure.value is error and error.__traceback__ is not None
        assert bool(partial_outputs) is (kind == "attach")
        assert not executor.state.graphs
        assert not rank.cache.handles()
        assert all(record() is None for record in rank.records)
        assert not executor.iterators
    view.backward(view.forward(_input(7)).hidden_states.sum())
    assert rank.weight.grad.item() == 7
    assert not executor.state.graphs and not rank.cache.handles()


@pytest.mark.parametrize(
    "phase", ["peer", "snapshot", "copy", "serialization", "transfer", "exchange"]
)
@pytest.mark.parametrize("operation", ["forward", "next", "batches_next"])
def test_peer_rejection_releases_undelivered_packet_storage(
    monkeypatch, request, operation, phase
):
    if gc.isenabled():
        request.addfinalizer(gc.enable)
    gc.disable()
    rank: Any = _CachedRank()
    executor = _Executor(rank, "zero")
    view = _view(executor)
    previous = view.forward(_input(7))
    graphs, handles = set(executor.state.graphs), rank.cache.handles()
    borrowed = StorageWeakRef(rank.weight.untyped_storage())
    peer_rank: Any = _Rank(1, 2)
    peer = _Executor(peer_rank, "zero")
    monkeypatch.setattr(peer, "_available_host_memory", lambda: 0)
    with pytest.raises(MemoryError, match="output snapshot") as rejected:
        peer.invoke("forward", _input(5))
    assert not peer.state.graphs
    inputs = [_input(3), _input(5)] if phase == "copy" else _input(3)
    argument = (
        inputs
        if operation == "forward"
        else executor.invoke(
            "batches" if operation == "next" else "batches_open", [inputs]
        )
    )
    storages, packets, exchanges = [], [], []
    copy_targets, copy_calls, copy_error = set(), 0, None
    packet = executor._packet

    def observe(*args):
        try:
            if phase in ("snapshot", "copy"):
                for tensor in _tensors.flatten_tensors(args[0])[0]:
                    storages.append(StorageWeakRef(tensor.untyped_storage()))
                    copy_targets.add(tensor.untyped_storage().data_ptr())
                tensor = None
            value = packet(*args)
            packets.append(weakref.ref(value))
            storages.extend(
                StorageWeakRef(t.untyped_storage()) for t in value.packet.tensors
            )
            return value
        finally:
            args = ()  # Observation must not retain the failed call's input tree.

    to = torch.Tensor.to

    def copy(tensor, *args, **kwargs):
        nonlocal copy_calls, copy_error
        try:
            targeted = tensor.untyped_storage().data_ptr() in copy_targets
            if targeted:
                copy_calls += 1
                if copy_calls == 2:
                    # Real CPU Tensor.to failure after one successful owned copy.
                    kwargs["memory_format"] = torch.channels_last
            copied = to(tensor, *args, **kwargs)
            if targeted:
                storages.append(StorageWeakRef(copied.untyped_storage()))
            return copied
        except RuntimeError as error:
            copy_error = error
            raise
        finally:
            tensor = None  # The fault injector must not own the failing tensor.

    exchange_error = RuntimeError("status exchange failed")

    def gather(value):
        exchanges.append(value)
        if phase == "exchange":
            raise exchange_error
        return [value, f"MemoryError: {rejected.value}" if phase == "peer" else value]

    monkeypatch.setattr(executor, "_packet", observe)
    with monkeypatch.context() as patch:
        patch.setattr(executor, "_gather", gather)
        if phase == "copy":
            patch.setattr(torch.Tensor, "to", copy)
        if phase == "snapshot":
            patch.setattr(executor, "_available_host_memory", lambda: 0)
        if phase in ("serialization", "transfer"):
            budgets = iter(
                [2**20, 0] if phase == "serialization" else [2**20, 2**20, 0]
            )
            patch.setattr(executor, "_available_host_memory", lambda: next(budgets))
            patch.setattr(executor, "distributed", True)
            patch.setattr(executor, "members", [0, 1])
            patch.setattr(executor, "_broadcast", lambda command: command)
        message = (
            "required rank 4"
            if phase == "copy"
            else "output snapshot"
            if phase == "peer"
            else "status exchange"
            if phase == "exchange"
            else f"output {phase}"
        )
        with pytest.raises((MemoryError, RuntimeError), match=message) as failure:
            executor.invoke(operation, argument)
    if phase == "copy":
        assert failure.value is copy_error and copy_calls == 2 and len(storages) == 3
    if phase == "exchange":
        assert failure.value is exchange_error
        # A failed status collective has no recovery contract; release only this
        # test's successful graph through the existing explicit release operation.
        pending = set(executor.state.graphs) - graphs
        assert len(pending) == 1
        executor.invoke("release", tuple(pending))
    if operation != "forward":
        executor.invoke("close" if operation == "next" else "batches_close", argument)
    assert (exchanges[0] is None) is (phase not in ("snapshot", "copy"))
    assert len(exchanges) == (2 if phase == "transfer" else 1)
    assert failure.value.__traceback__ is not None
    assert rejected.value.__traceback__ is not None
    assert set(executor.state.graphs) == graphs
    assert storages and all(storage.expired() for storage in storages)
    assert rank.cache.handles() == handles
    assert all(packet() is None for packet in packets)
    assert not borrowed.expired() and rank.weight.item() == 2
    view.backward(previous.hidden_states.sum())
    assert rank.weight.grad.item() == 7
    result = executor.invoke("forward", _input(11))
    assert result[0] is packets[-1]()
    executor.invoke("release", (result[0].packet.handle,))
    assert not executor.state.graphs and not rank.cache.handles()
    assert not storages[-1].expired() and result[0].packet.tensors[0].item() == 22


@pytest.mark.parametrize("delivery", ["forward", "iterator", "persistent"])
def test_later_wave_failure_keeps_only_previously_delivered_outputs(
    monkeypatch, delivery
):
    rank: Any = _CachedRank()
    executor = _Executor(rank, "zero")
    view = _view(executor)
    requests = [_input(3), _input(5)]
    delivered = None
    with monkeypatch.context() as patch:
        error, _ = _fail_delivery(patch, executor, "copy", packet_number=2)
        if delivery == "iterator":
            iterator = view.forward_batches(requests)
            delivered = next(iterator).outputs[0]
            advance = lambda: next(iterator)
        elif delivery == "persistent":
            handle = view.open_forward_batches(requests)
            batch = view.next_forward_batch(handle)
            assert batch is not None
            delivered = batch.outputs[0]
            advance = lambda: view.next_forward_batch(handle)
        else:
            advance = lambda: view.forward(requests)
        previous = set(executor.state.graphs)
        with pytest.raises(MemoryError) as failure:
            advance()
        assert failure.value is error and error.__traceback__ is not None
        assert set(executor.state.graphs) == previous
        assert len(rank.cache.handles()) == int(delivered is not None)
        assert not executor.iterators and not executor.state.iterators
        assert not executor.state.batch_inputs
        assert rank.closed == 1
    if delivered is not None:
        assert delivered.hidden_states.item() == 6
        view.backward(delivered.hidden_states.sum())
        assert rank.weight.grad.item() == 3
    view.zero_grad()
    view.backward(view.forward(_input(7)).hidden_states.sum())
    assert rank.weight.grad.item() == 7
    assert not executor.state.graphs and not rank.cache.handles()


@pytest.mark.parametrize("kind", ["copy", "attach", "assembly"])
def test_later_packet_failure_releases_every_physical_owner(monkeypatch, kind):
    ranks: list[Any] = [_CachedRank(dp, 2) for dp in range(2)]
    executors = [_Executor(rank, "zero") for rank in ranks]
    view = _view(executors[0])
    invoke = executors[0].invoke
    releases = []

    def dispatch(operation, *args, **kwargs):
        result = invoke(operation, *args, **kwargs)
        if operation == "release":
            releases.append(args[0])
            executors[1].invoke(operation, *args, **kwargs)
        return result

    monkeypatch.setattr(executors[0], "invoke", dispatch)
    requests = [_input(3), _input(5)]
    attached = []
    attach = executors[0].state.collector.attach

    def remember(*args, **kwargs):
        result = attach(*args, **kwargs)
        attached.append(result)
        return result

    monkeypatch.setattr(executors[0].state.collector, "attach", remember)
    with monkeypatch.context() as patch:
        if kind != "assembly":
            error, partial_outputs = _fail_delivery(patch, executors[1], kind)
        wave = []
        for index, executor in enumerate(executors):
            packet = executor._packet([ranks[index].forward(requests[index])], 1)
            batch = MicroBatch(
                [], [], [index], MicroBatchStats(0, 1, 2, 1, 0, 0, 0, 0, 0, False)
            )
            wave.append((batch, packet))
        if kind == "assembly":
            wave[0] = (replace(wave[0][0], stats=None), wave[0][1])
        with pytest.raises((MemoryError, TypeError)) as failure:
            view._combine_wave(wave, requests)
        if kind != "assembly":
            assert failure.value is error
            assert bool(partial_outputs) is (kind == "attach")
        assert failure.value.__traceback__ is not None
        assert attached  # A prior packet's proxy remains strongly referenced.
        assert releases and set(releases[-1]) == {"zero:1:dp:0", "zero:1:dp:1"}
        assert all(not executor.state.graphs for executor in executors)
        assert all(not rank.cache.handles() for rank in ranks)
        assert all(record() is None for rank in ranks for record in rank.records)
    for rank, executor in zip(ranks, executors, strict=True):
        local = _view(_Executor(rank, "rank"))
        local.backward(local.forward(_input(7)).hidden_states.sum())
        assert rank.weight.grad.item() == 7


@pytest.mark.parametrize("kind", ["copy", "placement"])
@pytest.mark.parametrize(
    "delivery,close_error",
    [
        ("rank", False),
        ("iterator", False),
        ("persistent", False),
        ("iterator", True),
        ("persistent", True),
    ],
)
def test_failed_release_preserves_delivery_error_and_retries_without_head_flush(
    monkeypatch, delivery, close_error, kind
):
    rank: Any = _CachedRank()
    executor = _Executor(rank, "rank" if delivery == "rank" else "zero")
    view = _view(executor)
    invoke = executor.invoke
    release_calls = 0
    flushes = []

    def fail_release(operation, *args, **kwargs):
        nonlocal release_calls
        if operation == "release":
            release_calls += 1
            if release_calls <= (1 if delivery == "rank" else 2):
                raise RuntimeError("injected release failure")
        return invoke(operation, *args, **kwargs)

    monkeypatch.setattr(executor, "invoke", fail_release)
    monkeypatch.setattr(view, "_flush_heads", lambda: flushes.append(True))
    with monkeypatch.context() as patch:
        if close_error:
            batches = rank.forward_batches

            def fail_close(*args, **kwargs):
                try:
                    yield from batches(*args, **kwargs)
                finally:
                    raise RuntimeError("injected iterator close failure")

            patch.setattr(rank, "forward_batches", fail_close)
        error, _ = _fail_delivery(patch, executor, kind)
        if delivery == "iterator":
            iterator = view.forward_batches([_input(3)])
            advance = lambda: next(iterator)
        elif delivery == "persistent":
            handle = view.open_forward_batches([_input(3)])
            advance = lambda: view.next_forward_batch(handle)
        else:
            advance = lambda: view.forward(_input(3))
        with pytest.raises(MemoryError) as failure:
            advance()
        assert failure.value is error and error.__traceback__ is not None
        assert release_calls == 1
        assert len(flushes) == (1 if delivery == "rank" else 2)
        assert not executor.iterators and not executor.state.iterators
        assert not executor.state.batch_inputs
        assert rank.closed == int(delivery != "rank")
        assert executor.state.released == set(executor.state.graphs)
        assert executor.state.graphs and rank.cache.handles()
    if delivery != "rank":
        with pytest.raises(RuntimeError, match="injected release failure"):
            view.optim_step()
        assert release_calls == 2 and rank.steps == 0 and len(flushes) == 2
        assert executor.state.released == set(executor.state.graphs)
        assert executor.state.graphs and rank.cache.handles()
    assert view.optim_step() == {"steps": 1}
    assert not executor.state.released and not executor.state.graphs
    assert not rank.cache.handles()
    assert all(record() is None for record in rank.records)
    view.backward(view.forward(_input(7)).hidden_states.sum())
    assert rank.weight.grad.item() == 7
    assert not executor.state.graphs and not rank.cache.handles()


@pytest.mark.parametrize("ending", ["exhaust", "close", "throw", "close_error"])
def test_non_delivery_iterator_closure_keeps_head_publication(monkeypatch, ending):
    rank: Any = _CachedRank()
    executor = _Executor(rank, "zero")
    view = _view(executor)
    flushes = []
    fail = False

    def flush():
        flushes.append(True)
        if fail:
            raise RuntimeError("injected close head publication failure")

    monkeypatch.setattr(view, "_flush_heads", flush)
    iterator = view.forward_batches([_input(3)])
    output = next(iterator).outputs[0]
    assert isinstance(iterator, Generator)
    assert len(flushes) == 2
    if ending == "exhaust":
        with pytest.raises(StopIteration):
            next(iterator)
    elif ending == "throw":
        with pytest.raises(ValueError, match="consumer failure"):
            iterator.throw(ValueError("consumer failure"))
    elif ending == "close_error":
        fail = True
        with pytest.raises(RuntimeError, match="close head publication failure"):
            iterator.close()
        assert rank.closed == 0 and executor.iterators
    else:
        iterator.close()
    assert len(flushes) == (4 if ending == "exhaust" else 3)
    executor.stop()
    assert rank.closed == 1 and not executor.iterators
    _view(_Executor(rank, "zero")).backward(output.hidden_states.sum())
    assert rank.weight.grad.item() == 3
    assert not executor.state.graphs and not rank.cache.handles()
