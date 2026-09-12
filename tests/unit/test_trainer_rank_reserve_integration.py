"""Accepted-source native iterator/profile controls with mocked CUDA counters."""

from dataclasses import asdict

import pytest
import torch

from art.trainer_rank import ForwardOutput, TrainerRank
from tests.unit.test_trainer_rank_physical_reserve import allocator
from tests.unit.test_trainer_rank_validation import (
    _runtime,
    _stub_forward,
    _target_request,
)


@pytest.mark.parametrize(
    "free,after,releases", [(1, 79_601, 1), (3000, 3000, 0), (1, 2, 1)]
)
def test_native_admission_forward_retention_and_backward_profile_survive_handoff(
    monkeypatch, free, after, releases
):
    rank = TrainerRank(_runtime())
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    request = _target_request(1)
    observed = rank._plan_flat_forward([request])
    rank._update_memory_profile(observed, 800, retained_bytes=200)
    original_cost = rank._plan_cost(observed)
    initial_profile = rank._memory_profiles[observed.signature]
    rank.device = torch.device("cuda:1")
    state = allocator(
        monkeypatch,
        free=free,
        total=100_000,
        allocated=10_000,
        reserved=90_000,
        after=after,
        after_reserved=10_400,
    )
    peak = {"bytes": 11_000}
    syncs, resets, selected, tracked, profiles, handoffs = [], [], [], [], [], []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: syncs.append(device))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: resets.append(device)
    )
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda device: peak["bytes"]
    )
    native_select = rank._select_next_micro_batch
    native_tracking = rank._run_flat_plan_with_memory_tracking
    native_profile = rank._update_peak_memory_profile
    native_release = rank._release_cached_memory_for_backward

    def select(*args, **kwargs):
        candidate = native_select(*args, **kwargs)
        selected.append(candidate)
        assert rank._memory_profiles[candidate.plan.signature] is initial_profile
        return candidate

    def track(plan, *, check, context):
        assert check is selected[-1].check
        tracked.append((plan, check))
        return native_tracking(plan, check=check, context=context)

    def execute(plan):
        assert plan is selected[-1].plan
        state["allocated"] = 10_400
        return [ForwardOutput(torch.ones(2, requires_grad=True), None, None, None)]

    def profile(plan, baseline, retained_after=None):
        native_profile(plan, baseline, retained_after)
        profiles.append(
            (plan, baseline, retained_after, rank._memory_profiles[plan.signature])
        )

    def release(plan):
        assert plan is selected[-1].plan
        assert len(profiles) == 1 and profiles[0][2] == 10_400
        learned = rank._memory_profiles[plan.signature]
        assert learned is not initial_profile
        before_cost = rank._plan_cost(plan)
        before_check = asdict(selected[-1].check)
        native_release(plan)
        # Cache release changes availability, not the admitted object or profile.
        assert rank._memory_profiles[plan.signature] is learned
        assert rank._plan_cost(plan) == before_cost
        assert asdict(selected[-1].check) == before_check
        handoffs.append((learned, before_cost))

    monkeypatch.setattr(rank, "_select_next_micro_batch", select)
    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", track)
    monkeypatch.setattr(rank, "_execute_flat_plan", execute)
    monkeypatch.setattr(rank, "_update_peak_memory_profile", profile)
    monkeypatch.setattr(rank, "_release_cached_memory_for_backward", release)
    iterator = rank.forward_micro_batches([request])
    batch = next(iterator)
    assert len(selected) == len(tracked) == len(handoffs) == 1
    check = selected[0].check
    assert check.estimated_required_bytes == original_cost.required
    assert batch.stats.estimated_required_bytes == check.estimated_required_bytes
    assert batch.stats.available_bytes == check.available_bytes
    assert rank.last_forward_telemetry()["usable_limit_bytes"] == check.available_bytes
    assert handoffs[0][1].required > original_cost.required  # real forward observation
    assert state["releases"] == releases
    assert state["current"] == torch.device("cuda:7")
    output = batch.outputs[0].target_logprobs
    output.sum().backward()
    assert torch.equal(output.grad, torch.ones_like(output))
    peak["bytes"] = 13_000
    with pytest.raises(StopIteration):
        next(iterator)
    assert len(profiles) == 2 and profiles[1][:3] == (selected[0].plan, 10_000, None)
    forward_profile, backward_profile = profiles[0][3], profiles[1][3]
    assert backward_profile.bytes_per_token > forward_profile.bytes_per_token
    assert (
        backward_profile.retained_compute_bytes_per_token
        == forward_profile.retained_compute_bytes_per_token
    )
    assert backward_profile.retained_fraction == forward_profile.retained_fraction
    assert len(syncs) == 2 and len(resets) == 1  # existing forward tracking only


@pytest.mark.parametrize(
    "query", ["mem_get_info", "memory_allocated", "memory_reserved"]
)
def test_initial_query_failure_preserves_original_cause_without_release(
    monkeypatch, query
):
    rank = TrainerRank(_runtime())
    plan = rank._plan_flat_forward([_target_request(1)])
    rank.device = torch.device("cuda:1")
    state = allocator(monkeypatch)
    original = RuntimeError("original allocator query failure")
    cause = ValueError("original query cause")
    original.__cause__ = cause

    def fail(*args):
        raise original

    monkeypatch.setattr(torch.cuda, query, fail)
    with pytest.raises(RuntimeError) as caught:
        rank._release_cached_memory_for_backward(plan)
    assert caught.value is original and caught.value.__cause__ is cause
    assert state["releases"] == 0
    assert state["current"] == torch.device("cuda:7")


@pytest.mark.parametrize("yield_empty", [False, True])
def test_public_empty_local_rank_never_queries_or_trims(monkeypatch, yield_empty):
    rank = TrainerRank(_runtime())
    executed = []

    def empty_forward(plan, **kwargs):
        assert not plan.groups and not plan.request_count
        executed.append(plan)
        rank.device = torch.device("cuda:1")
        return []

    _stub_forward(monkeypatch, rank, empty_forward, dp=(1, 2))
    state = allocator(monkeypatch)
    batches = list(
        rank.forward_micro_batches([_target_request(1)], yield_empty=yield_empty)
    )
    assert len(executed) == 1
    assert len(batches) == int(yield_empty)
    assert state["reads"] == state["releases"] == 0
    if yield_empty:
        assert batches[0].indices == () and batches[0].outputs == []
