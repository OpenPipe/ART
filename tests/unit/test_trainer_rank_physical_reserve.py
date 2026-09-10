"""CPU allocator/iterator contracts; no external-library reserve calibration."""

from contextlib import contextmanager
from dataclasses import replace
import inspect
from typing import Any

import pytest
import torch

from art.trainer_rank import ForwardInput, ForwardOutput, TrainerRank, _impl
from tests.unit.test_trainer_rank_validation import (
    _runtime,
    _stub_forward,
    _target_request,
)


def allocator(
    monkeypatch,
    *,
    free=1,
    total=1000,
    allocated=400,
    reserved=900,
    after=100,
    after_reserved=None,
    backend="native",
):
    state: dict[str, Any] = dict(
        free=free,
        total=total,
        allocated=allocated,
        reserved=reserved,
        reads=0,
        releases=0,
        current=torch.device("cuda:7"),
        failure=None,
    )

    def info(device):
        assert device == torch.device("cuda:1")
        state["reads"] += 1
        if state["failure"] is not None and state["releases"]:
            raise state["failure"]
        return state["free"], state["total"]

    @contextmanager
    def device(target):
        previous, state["current"] = state["current"], target
        try:
            yield
        finally:
            state["current"] = previous

    def release():
        assert state["current"] == torch.device("cuda:1")
        state["releases"] += 1
        state.update(
            free=after,
            reserved=allocated if after_reserved is None else after_reserved,
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: backend)
    monkeypatch.setattr(torch.cuda, "mem_get_info", info)
    monkeypatch.setattr(
        torch.cuda, "memory_allocated", lambda _device: state["allocated"]
    )
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda _device: state["reserved"]
    )
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "empty_cache", release)
    return state


@pytest.mark.parametrize(
    ("free", "allocated", "reserved", "after", "releases"),
    [
        (29, 400, 900, 100, 1),
        (30, 400, 900, 100, 0),
        (31, 400, 900, 100, 0),
        (1, 400, 400, 100, 0),
        (1, 400, 399, 100, 0),
        (1, 400, 900, 2, 1),
    ],
)
def test_physical_reserve_is_only_a_soft_release_trigger(
    monkeypatch, free, allocated, reserved, after, releases
):
    rank = TrainerRank(_runtime())
    plan = rank._plan_flat_forward([_target_request(1)])
    rank.device = torch.device("cuda:1")
    state = allocator(
        monkeypatch, free=free, allocated=allocated, reserved=reserved, after=after
    )
    # The existing admission budget includes cache that external CUDA cannot use.
    check = rank._memory_check_required(1)
    assert check.available_bytes == max(0, free + max(0, reserved - allocated) - 30)
    before = state["reads"]
    rank._release_cached_memory_for_backward(plan)
    assert state["releases"] == releases
    assert state["reads"] - before == 1 + releases
    assert state["current"] == torch.device("cuda:7")
    # An unsuccessful attempt to meet the soft trigger adds no refusal/retry.
    if releases:
        assert state["free"] == after


@pytest.mark.parametrize("case", ["cpu", "all_no_grad", "inactive", "unavailable"])
def test_non_gradient_or_non_cuda_work_does_not_query_physical_memory(
    monkeypatch, case
):
    rank = TrainerRank(_runtime())
    requests = (
        [ForwardInput(input_tokens=torch.tensor([1]))]
        if case == "inactive"
        else [_target_request(1)]
    )
    with torch.set_grad_enabled(case != "all_no_grad"):
        plan = rank._plan_flat_forward(requests)
    rank.device = torch.device("cpu" if case == "cpu" else "cuda:1")
    state = allocator(monkeypatch)
    if case == "unavailable":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rank._release_cached_memory_for_backward(plan)
    assert state["reads"] == state["releases"] == 0


def test_mixed_gradient_handoff_preserves_actual_check_profile_and_context(monkeypatch):
    rank = TrainerRank(_runtime())
    checks, profiles, executed, phases = [], [], [], []
    select = rank._select_next_micro_batch

    def selected(*args, **kwargs):
        candidate = select(*args, **kwargs)
        checks.append(candidate.check)
        return candidate

    def forward(plan, **kwargs):
        assert kwargs["check"] is checks[-1]
        executed.append(plan)
        rank.device = torch.device("cuda:1")
        return [
            ForwardOutput(torch.ones(2, requires_grad=g.grad_enabled), None, None, None)
            for g in plan.groups
        ]

    _stub_forward(monkeypatch, rank, forward, profiled=True)
    monkeypatch.setattr(rank, "_select_next_micro_batch", selected)
    monkeypatch.setattr(
        rank,
        "_update_peak_memory_profile",
        lambda plan, baseline: profiles.append((plan, baseline)),
    )
    state = allocator(monkeypatch)
    original_phase = _impl._telemetry_phase

    @contextmanager
    def phase(name, evidence, **kwargs):
        with original_phase(name, evidence, **kwargs):
            yield
        if name == "gradient_handoff_cache_release":
            phases.append(evidence.copy())

    monkeypatch.setattr(_impl, "_telemetry_phase", phase)
    inputs = [
        [
            replace(_target_request(1), no_grad=True),
            replace(_target_request(3), no_grad=False),
        ]
    ]
    with torch.no_grad():
        iterator = rank.forward_micro_batches(inputs, no_grad=True)
        batch = next(iterator)
        assert not torch.is_grad_enabled()
        assert [g.grad_enabled for g in executed[0].groups] == [False, True]
        assert state["releases"] == 1 and state["current"] == torch.device("cuda:7")
        assert batch.inputs == inputs and batch.indices == (0,)
        assert (
            batch.stats.estimated_required_bytes == checks[0].estimated_required_bytes
        )
        assert batch.stats.available_bytes == checks[0].available_bytes
        assert (
            rank.last_forward_telemetry()["usable_limit_bytes"]
            == checks[0].available_bytes
        )
        target = batch.outputs[0][1].target_logprobs
        target.backward(torch.ones_like(target))
        assert profiles == []
        with pytest.raises(StopIteration):
            next(iterator)
        assert not torch.is_grad_enabled()
    assert len(checks) == len(executed) == len(profiles) == 1
    assert profiles == [(executed[0], None)]
    assert phases[0]["physical_free_before_bytes"] == 1
    assert phases[0]["physical_free_after_bytes"] == 100
    assert phases[0]["reserve_trigger_bytes"] == 30


@pytest.mark.parametrize("fault", ["release", "after_snapshot"])
def test_handoff_failure_preserves_error_and_restores_device_and_grad_context(
    monkeypatch, fault
):
    rank = TrainerRank(_runtime())
    original = RuntimeError("original CUDA failure")
    state = allocator(monkeypatch)

    def forward(plan, **kwargs):
        rank.device = torch.device("cuda:1")
        return [ForwardOutput(torch.ones(2, requires_grad=True), None, None, None)]

    _stub_forward(monkeypatch, rank, forward)
    if fault == "release":

        def fail():
            assert state["current"] == rank.device
            raise original

        monkeypatch.setattr(torch.cuda, "empty_cache", fail)
    else:
        state["failure"] = original
    with torch.no_grad():
        iterator = rank.forward_micro_batches([_target_request(1)], no_grad=False)
        with pytest.raises(RuntimeError) as caught:
            next(iterator)
        assert caught.value is original
        assert not torch.is_grad_enabled()
    assert state["current"] == torch.device("cuda:7")
    assert inspect.getgeneratorstate(iterator) == inspect.GEN_CLOSED


@pytest.mark.parametrize("backend", ["cudaMallocAsync", "unknown"])
def test_unqualified_allocator_backend_skips_physical_queries(monkeypatch, backend):
    rank = TrainerRank(_runtime())
    plan = rank._plan_flat_forward([_target_request(1)])
    rank.device = torch.device("cuda:1")
    state = allocator(monkeypatch, backend=backend)
    rank._release_cached_memory_for_backward(plan)
    assert state["reads"] == state["releases"] == 0


def test_matched_h200_snapshot_releases_cache_without_repricing_admission(monkeypatch):
    rank = TrainerRank(_runtime())
    plan = rank._plan_flat_forward([_target_request(1)])
    rank.device = torch.device("cuda:1")
    state = allocator(
        monkeypatch,
        free=14_352_384,
        total=150_121_021_440,
        allocated=92_197_523_968,
        reserved=147_394_134_016,
        after=42_768_990_208,
        after_reserved=104_639_496_192,
    )
    # Replays measured allocator counters, not a calibrated cuBLAS requirement.
    check = rank._memory_check_required(7_976_316_880)
    assert check.fits
    original_check = (check.estimated_required_bytes, check.available_bytes, check.fits)
    rank._release_cached_memory_for_backward(plan)
    assert state["releases"] == 1
    assert state["free"] - 14_352_384 == 42_754_637_824
    assert 147_394_134_016 - state["reserved"] == 42_754_637_824
    assert state["allocated"] == 92_197_523_968
    assert (
        check.estimated_required_bytes,
        check.available_bytes,
        check.fits,
    ) == original_check


@pytest.mark.parametrize("delta, releases", [(-1, 1), (0, 0), (1, 0)])
def test_existing_reserve_boundary_on_h200(monkeypatch, delta, releases):
    rank = TrainerRank(_runtime())
    plan = rank._plan_flat_forward([_target_request(1)])
    rank.device = torch.device("cuda:1")
    total = 150_121_021_440
    reserve = int(total * _impl._MEMORY_RESERVE_FRACTION)
    assert reserve == 4_503_630_643
    state = allocator(monkeypatch, total=total, free=reserve + delta)
    rank._release_cached_memory_for_backward(plan)
    assert state["releases"] == releases


def test_split_plan_releases_once_for_the_whole_gradient_handoff(monkeypatch):
    rank = TrainerRank(_runtime())
    parts = tuple(
        rank._plan_flat_forward([replace(_target_request(i), no_grad=no_grad)])
        for i, no_grad in [(1, True), (2, False), (3, False)]
    )
    plan = _impl._SplitForwardPlan(parts, ((0,), (1,), (2,)), 3)
    assert [g.grad_enabled for g in plan.groups] == [False, True, True]
    rank.device = torch.device("cuda:1")
    state = allocator(monkeypatch)
    rank._release_cached_memory_for_backward(plan)
    assert state["releases"] == 1


def test_direct_forward_has_no_new_handoff_policy(monkeypatch):
    rank = TrainerRank(_runtime())
    executed = []

    def forward(plan, **kwargs):
        executed.append(plan)
        return [ForwardOutput(torch.ones(2, requires_grad=True), None, None, None)]

    _stub_forward(monkeypatch, rank, forward)
    monkeypatch.setattr(
        rank,
        "_release_cached_memory_for_backward",
        lambda plan: pytest.fail("direct forward is outside the iterator handoff"),
    )
    output = rank.dp_rank_forward(_target_request(1))
    output.target_logprobs.sum().backward()
    assert len(executed) == 1
