"""Shared admission/handoff budget and original iterator ownership, CPU only."""

from contextlib import nullcontext
from types import SimpleNamespace
import weakref

import pytest
import torch

from art.trainer_rank import ForwardOutput, TrainerRank, _backward_work, _impl
from tests.unit import test_trainer_rank_cache_recovery as recovery
from tests.unit.test_trainer_rank_physical_reserve import allocator
from tests.unit.test_trainer_rank_validation import (
    _runtime,
    _stub_forward,
    _target_request,
)


@pytest.fixture
def rig():
    fixture = recovery.TestRecovery()
    rank, cuda, clock, ns = fixture.make()
    cuda.memory_reserved = lambda device: cuda.allocated + 100
    cuda.device = lambda device: nullcontext()
    try:
        yield rank, cuda, clock, ns
    finally:
        fixture.doCleanups()


def plan(*modes):
    return SimpleNamespace(groups=[SimpleNamespace(grad_enabled=m) for m in modes])


def test_admission_consumes_the_only_first_release(rig):
    rank, cuda, _, ns = rig
    _, error, _ = recovery.run(rank, [recovery.fail(ns), recovery.success(ns)])
    assert error is None and cuda.events.count("release") == 1
    cuda.free = 1
    before = rank._recovery_state().cost
    rank._release_cached_memory_for_backward(plan(True))
    assert cuda.events.count("release") == 1
    assert rank._recovery_state().cost > before
    rank._record_recovery_work("forward_micro_batches", 100.0)
    rank._release_cached_memory_for_backward(plan(True))
    assert cuda.events.count("release") == 2


def test_handoff_consumes_the_only_first_release(rig):
    rank, cuda, _, ns = rig
    cuda.free = 1
    rank._release_cached_memory_for_backward(plan(True))
    assert cuda.events.count("release") == 1
    cuda.free = 40
    _, error, calls = recovery.run(rank, [recovery.fail(ns), recovery.success(ns)])
    assert isinstance(error, _impl.TrainerRankMemoryError)
    assert calls == 1 and cuda.events.count("release") == 1


@pytest.mark.parametrize("cost, releases", [(40.0, 1), (40.5, 0)])
def test_repeat_budget_has_the_same_five_percent_boundary(rig, cost, releases):
    rank, cuda, _, _ = rig
    state = rank._recovery_state()
    state.first_consumed, state.work, state.cost, state.high = True, 1000.0, cost, 10.0
    ticks = iter((1.0, 1.0, 2.0))
    rank._recovery_clock = lambda: next(ticks)
    cuda.free = 1
    rank._release_cached_memory_for_backward(plan(True))
    assert cuda.events.count("release") == releases
    assert state.cost == cost + 1.0 and state.work == 1000.0 and state.owner is None


@pytest.mark.parametrize(
    "observer_step_ns,cost,releases",
    [(0, 40.0, 1), (1_000_000, 40.0, 0), (1_000_000, 39.0, 1)],
)
def test_imported_observer_cost_moves_repeat_boundary(
    rig, observer_step_ns, cost, releases
):
    rank, cuda, clock, _ = rig
    assert _impl.BackwardWork is _backward_work.BackwardWork
    clock.observer_step_ns = observer_step_ns
    state = rank._recovery_state()
    state.first_consumed, state.work, state.cost, state.high = True, 1000.0, cost, 10.0
    ticks = iter((1.0, 1.0, 2.0))
    rank._recovery_clock = lambda: next(ticks)
    cuda.free = 1
    original_reduce = rank._recovery_reduce
    observations = []

    def reduce(values, *, op, sync_across_dp):
        if op == "SUM":
            observer = state.backward
            assert isinstance(observer, _backward_work.BackwardWork)
            observations.append((list(values), observer.cost_ns / 1e9))
        elif op == "MAX" and len(values) == 4:
            assert values[1:] == [1000.0, 1.0, 0.0]
        return original_reduce(values, op=op, sync_across_dp=sync_across_dp)

    rank._recovery_reduce = reduce
    rank._release_cached_memory_for_backward(plan(True))
    assert len(observations) == 1
    operands, observer_seconds = observations[0]
    assert operands == [cost + observer_seconds, 10.0]
    assert (observer_seconds > 0) == (observer_step_ns > 0)
    assert (sum(operands) <= 0.05 * 1000.0) == bool(releases)
    assert cuda.events.count("release") == releases
    assert state.cost == cost + 1.0 and state.work == 1000.0
    assert state.owner is None and not state.invalid
    assert state.backward.work_ns == 0


@pytest.mark.parametrize("modes", [(), (False,)])
def test_empty_and_local_no_grad_peer_participate_without_cuda_queries(rig, modes):
    rank, cuda, _, _ = rig
    calls = []

    def reduce(values, *, op, sync_across_dp):
        assert sync_across_dp
        calls.append((op, len(values)))
        if op == "MAX" and len(values) == 2:
            values[1] = 1.0  # Only the other peer has local gradient work.
        elif op == "MIN":
            values[1] = -1.0  # The other peer needs/attempts a release.
        return values

    rank._recovery_reduce = reduce
    rank._release_cached_memory_for_backward(plan(*modes))
    assert calls == [("MAX", 2), ("SUM", 2), ("MAX", 4), ("MIN", 4), ("MIN", 2)]
    assert cuda.events == [] and rank._recovery_state().first_consumed


@pytest.mark.parametrize("cancel", [False, True])
def test_forward_error_survives_secondary_exchange_failure(rig, cancel):
    rank, cuda, _, _ = rig
    primary = KeyboardInterrupt("cancel") if cancel else RuntimeError("forward")
    cause, context = ValueError("cause"), LookupError("context")
    primary.__cause__ = cause
    primary.__context__ = context
    primary.__suppress_context__ = True

    def broken(values, **kwargs):
        assert values[0] == 1.0
        raise OSError("secondary exchange")

    rank._recovery_reduce = broken
    with pytest.raises(type(primary)) as caught:
        rank._release_cached_memory_for_backward(plan(True), error=primary)
    assert caught.value is primary and primary.__cause__ is cause
    assert primary.__context__ is context and primary.__suppress_context__
    assert cuda.events == [] and rank._recovery_state().owner is None
    assert "secondary exchange" in "\n".join(primary.__notes__)


def test_peer_forward_failure_prevents_release(rig):
    rank, cuda, _, _ = rig
    rank._recovery_reduce = lambda values, **kwargs: [1.0, 1.0]
    with pytest.raises(RuntimeError, match="another rank"):
        rank._release_cached_memory_for_backward(plan(True))
    assert cuda.events == [] and rank._recovery_state().owner is None


def test_globally_no_grad_stops_after_the_shared_status_vote(rig):
    rank, cuda, _, _ = rig
    calls = []

    def reduce(values, *, op, sync_across_dp):
        calls.append((op, len(values), sync_across_dp))
        return values

    rank._recovery_reduce = reduce
    rank._release_cached_memory_for_backward(plan(False))
    assert calls == [("MAX", 2, True)] and cuda.events == []
    assert not rank._recovery_state().first_consumed
    assert rank._recovery_state().cost > 0


def test_busy_owner_is_not_overwritten_or_cleared(rig):
    rank, cuda, _, _ = rig
    state = rank._recovery_state()
    owner = state.owner = object()
    cuda.free = 1
    rank._release_cached_memory_for_backward(plan(True))
    assert state.owner is owner and state.invalid
    assert "release" not in cuda.events


def test_new_handoff_failure_retires_only_owned_output_aliases(monkeypatch):
    rank = TrainerRank(_runtime())
    refs = []

    def forward(plan, **kwargs):
        rank.device = torch.device("cuda:1")
        tensor = torch.ones(2, requires_grad=True)
        refs.append(weakref.ref(tensor))
        return [ForwardOutput(tensor, None, None, None)]

    _stub_forward(monkeypatch, rank, forward)
    state = allocator(monkeypatch)
    primary = RuntimeError("post-release sample")
    state["failure"] = primary
    with torch.no_grad():
        iterator = rank.forward_micro_batches([_target_request(1)], no_grad=False)
        with pytest.raises(RuntimeError) as caught:
            next(iterator)
        assert caught.value is primary and not torch.is_grad_enabled()
    assert len(refs) == 1 and refs[0]() is None
    assert getattr(iterator, "gi_frame") is None
    assert rank._recovery_state().owner is None
