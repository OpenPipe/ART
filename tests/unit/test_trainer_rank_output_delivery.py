"""Undelivered logical outputs must not retain physical forward graphs."""

from dataclasses import replace
from functools import partial
import weakref

import pytest
from test_trainer_rank_commands import _input, _Rank
import torch

from art.trainer_rank import (
    ForwardInput,
    ForwardOutput,
    MicroBatch,
    MicroBatchStats,
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
    """Fail inside the real copy or collector after physical graph registration."""
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
@pytest.mark.parametrize("kind", ["copy", "attach"])
def test_failed_delivery_releases_registered_graph_and_native_cache(
    monkeypatch, mode, kind
):
    rank = _CachedRank()
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


@pytest.mark.parametrize("delivery", ["forward", "iterator", "persistent"])
def test_later_wave_failure_keeps_only_previously_delivered_outputs(
    monkeypatch, delivery
):
    rank = _CachedRank()
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
            delivered = view.next_forward_batch(handle).outputs[0]
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
    ranks = [_CachedRank(dp, 2) for dp in range(2)]
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


def test_failed_release_preserves_delivery_error_and_retries_without_head_flush(
    monkeypatch,
):
    rank = _CachedRank()
    executor = _Executor(rank, "rank")
    view = _view(executor)
    invoke = executor.invoke
    failed = False
    flushes = []

    def release_once(operation, *args, **kwargs):
        nonlocal failed
        if operation == "release" and not failed:
            failed = True
            raise RuntimeError("injected release failure")
        return invoke(operation, *args, **kwargs)

    monkeypatch.setattr(executor, "invoke", release_once)
    monkeypatch.setattr(view, "_flush_heads", lambda: flushes.append(True))
    with monkeypatch.context() as patch:
        error, _ = _fail_delivery(patch, executor, "copy")
        with pytest.raises(MemoryError) as failure:
            view.forward(_input(3))
        assert failure.value is error and error.__traceback__ is not None
        assert failed and len(flushes) == 1
        assert executor.state.released == set(executor.state.graphs)
        assert executor.state.graphs and rank.cache.handles()
    assert view.optim_step() == {"steps": 1}
    assert not executor.state.released and not executor.state.graphs
    assert not rank.cache.handles()
