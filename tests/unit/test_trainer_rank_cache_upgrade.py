"""One minimum-wave cache upgrade, using actual search/recovery and CPU facades."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from test_trainer_rank_handoff_budget import plan as handoff_plan
from test_trainer_rank_handoff_budget import rig
from test_trainer_rank_split import _packed_budget, _rank, _request

from art.trainer_rank import _impl


def candidate(required=5, *, probe=True):
    return _impl._CandidateMicroBatch(
        inputs=[],
        indices=(0,),
        plan=cast(
            _impl._AnyForwardPlan,
            SimpleNamespace(packed_tokens=required, logical_tokens=required),
        ),
        check=_impl._MemoryCheck(required, 10, required <= 10),
        stats_global_count=1,
        rejected_candidates=1,
        cold_start=True,
        recovery_probe=_impl._MemoryCheck(80, 10, False) if probe else None,
    )


def admit(rank, values, calls):
    def search():
        value = values[min(len(calls), len(values) - 1)]
        calls.append(value)
        if isinstance(value, BaseException):
            raise value
        return value

    return rank._recover_admission(
        search,
        lambda value: (value.plan, value.check),
        lambda value, check: replace(value, check=check),
        context="forward_micro_batches",
        sync_across_dp=True,
    )


def test_upgrade_first_attempt_rebuilds_search_and_shares_handoff_budget(rig):
    rank, cuda, _, _ = rig
    calls = []
    flat = candidate(80, probe=False)
    result = admit(rank, [candidate(), flat], calls)
    assert result.plan is flat.plan and result.check.available_bytes == 170
    assert len(calls) == 2 and cuda.events.count("release") == 1
    state = rank._recovery_state()
    assert state.first_consumed and state.cost > 0 and state.owner is None
    cuda.free = 1
    rank._release_cached_memory_for_backward(handoff_plan(True))
    assert cuda.events.count("release") == 1
    rank._record_recovery_work("forward_micro_batches", 100.0)
    rank._release_cached_memory_for_backward(handoff_plan(True))
    assert cuda.events.count("release") == 2


@pytest.mark.parametrize("cost,releases", [(40.0, 1), (40.5, 0)])
def test_upgrade_uses_existing_five_percent_boundary(rig, cost, releases):
    rank, cuda, _, _ = rig
    state = rank._recovery_state()
    state.first_consumed, state.work, state.cost, state.high = True, 1000.0, cost, 10.0
    ticks = iter((1.0, 1.0, 2.0))
    rank._recovery_clock = lambda: next(ticks)
    calls = []
    result = admit(rank, [candidate(), candidate(80, probe=False)], calls)
    assert result.check.estimated_required_bytes == (80 if releases else 5)
    assert cuda.events.count("release") == releases and len(calls) == 1 + releases
    assert state.cost == cost + 1.0 and state.owner is None


@pytest.mark.parametrize("searched", ["split", "refused"])
def test_ineffective_upgrade_returns_fresh_split_without_second_attempt(rig, searched):
    rank, cuda, _, ns = rig
    cuda.empty_cache = lambda: cuda.events.append("release")
    incumbent = candidate()
    retry = (
        incumbent
        if searched == "split"
        else ns["_ForwardRefusal"](
            incumbent.plan, _impl._MemoryCheck(80, 10, False), "still refused"
        )
    )
    calls = []
    result = admit(rank, [incumbent, retry], calls)
    assert result.plan is incumbent.plan and result.check.available_bytes == 10
    assert len(calls) == 2 and cuda.events.count("release") == 1
    assert rank._recovery_state().owner is None


@pytest.mark.parametrize("release", [False, True])
def test_stale_incumbent_refuses_without_second_release(rig, release):
    rank, cuda, _, _ = rig
    if release:

        def empty_cache():
            cuda.events.append("release")
            cuda.free = 31

        cuda.empty_cache = empty_cache
    else:

        def denied(*args, **kwargs):
            cuda.free = 31
            return False

        rank._try_cache_recovery = denied
    calls = []
    with pytest.raises(_impl.TrainerRankMemoryError):
        admit(rank, [candidate()], calls)
    assert cuda.events.count("release") == int(release)
    assert len(calls) == 1 + int(release) and rank._recovery_state().owner is None


@pytest.mark.parametrize("stage", ["release", "search"])
def test_upgrade_preserves_original_failure(rig, stage):
    rank, cuda, _, _ = rig
    error = RuntimeError(stage)
    if stage == "release":
        cuda.failure = error
    with pytest.raises(RuntimeError) as caught:
        admit(rank, [candidate(), error], [])
    assert caught.value is error
    assert cuda.events.count("release") == 1 and rank._recovery_state().owner is None


def test_normal_fit_has_no_upgrade_episode(rig):
    rank, _, _, _ = rig
    rank._cache_recovery_episode = lambda: pytest.fail("no recovery metadata")
    assert admit(rank, [candidate(probe=False)], []).check.fits


def test_actual_minimum_wave_search_upgrades_only_after_physical_release(monkeypatch):
    rank = _rank(monkeypatch)
    requests = [_request(i) for i in range(4)]
    monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *a, **k: 0)
    free, total = [50], 1000
    _packed_budget(monkeypatch, rank, lambda: free[0] - 30)
    monkeypatch.setattr(rank, "device", _impl.torch.device("cuda"))
    monkeypatch.setattr(_impl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_impl.torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        _impl.torch.cuda, "mem_get_info", lambda device: (free[0], total)
    )
    monkeypatch.setattr(_impl.torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.delenv(_impl._TEST_HOOKS_ENV, raising=False)
    releases, searches = [], []
    search = rank._search_next_micro_batch

    def observed(*args, **kwargs):
        result = search(*args, **kwargs)
        searches.append(result)
        return result

    def release():
        releases.append(free[0])
        free[0] = 90

    monkeypatch.setattr(rank, "_search_next_micro_batch", observed)
    monkeypatch.setattr(_impl.torch.cuda, "empty_cache", release)
    selected = rank._select_next_micro_batch([requests], 0)
    assert len(searches) == 2 and searches[0].plan.subforward_count == 2
    assert searches[0].recovery_probe.estimated_required_bytes == 40
    assert selected.plan.subforward_count == 1 and selected.plan.packed_tokens == 40
    assert selected.check.available_bytes == 60 and releases == [50]
    assert not _impl.torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "local,outcome",
    [("split", 2), ("flat", 2), ("empty", 2), ("flat", 3), ("empty", 3)],
)
def test_fallback_world_status_sets_uniform_probe_before_fit_return(
    monkeypatch, local, outcome
):
    rank = _rank(monkeypatch)
    requests = [_request(i) for i in range(4)]
    monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *a, **k: 0)
    _packed_budget(monkeypatch, rank, 20)
    if local == "split":
        found = rank._find_admissible_forward(
            requests, checkpoint=_impl.Unset, refusal_prefix="test"
        )
    else:
        plan = rank._plan_flat_forward([] if local == "empty" else requests[:1])
        found = plan, rank._memory_check(plan)
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (int(local == "empty"), 2))
    monkeypatch.setattr(rank, "_estimate_flat_forward", lambda *a, **k: None)
    monkeypatch.setattr(
        rank, "_memory_check", lambda *a, **k: _impl._MemoryCheck(40, 20, False)
    )
    monkeypatch.setattr(rank, "_find_admissible_forward", lambda *a, **k: found)
    events = []

    def vote(status):
        events.append(("status", status))
        return outcome

    def refresh(check, **kw):
        events.append(("refresh",))
        return replace(
            check, available_bytes=20, fits=check.estimated_required_bytes <= 20
        )

    def reduce(values, *, op, sync_across_dp):
        assert sync_across_dp
        events.append((op,))
        return values

    monkeypatch.setattr(rank, "_admission_outcome", vote)
    monkeypatch.setattr(rank, "_refresh_memory_check", refresh)
    monkeypatch.setattr(rank, "_recovery_reduce", reduce)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 20)
    selected = rank._select_next_micro_batch([requests], 0)
    assert (selected.recovery_probe is not None) == (outcome == 2)
    assert events == [("status", 2 if local == "split" else 3), ("refresh",)] + (
        [("SUM",), ("MAX",), ("MIN",), ("refresh",)] if outcome == 2 else []
    )
    assert selected.inputs == ([] if local == "empty" else [requests])


def test_handoff_first_attempt_is_not_renewed_for_upgrade(rig):
    rank, cuda, _, _ = rig
    cuda.free = 1
    rank._release_cached_memory_for_backward(handoff_plan(True))
    cuda.free = 40
    calls = []
    result = admit(rank, [candidate()], calls)
    assert result.check.fits and len(calls) == 1
    assert cuda.events.count("release") == 1


@pytest.mark.parametrize("reason", ["backend", "cap", "foreign_owner"])
def test_upgrade_preserves_native_cap_and_owner_stops(rig, monkeypatch, reason):
    rank, cuda, _, _ = rig
    if reason == "backend":
        cuda.backend = "cudaMallocAsync"
    elif reason == "cap":
        monkeypatch.setenv("CONTROL_ART_HOOK", "1")
        monkeypatch.setenv("CONTROL_ART_LIMIT", "40")
    else:
        rank._recovery_state().owner = object()
    calls = []
    assert admit(rank, [candidate()], calls).check.fits
    # Existing non-native availability includes cached bytes, so it can cause
    # a fresh search without any native release. This policy is unchanged.
    assert len(calls) == (2 if reason == "backend" else 1)
    assert "release" not in cuda.events


@pytest.mark.parametrize("outcome", [0, 1])
def test_failed_fallback_vote_never_creates_upgrade_probe(monkeypatch, outcome):
    rank = _rank(monkeypatch)
    monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *a, **k: 0)
    _packed_budget(monkeypatch, rank, 20)
    statuses = []
    monkeypatch.setattr(
        rank, "_admission_outcome", lambda status: statuses.append(status) or outcome
    )
    if outcome == 0:
        with pytest.raises(RuntimeError, match="another DP rank"):
            rank._search_next_micro_batch([[_request(i) for i in range(4)]], 0)
    else:
        result = rank._search_next_micro_batch([[_request(i) for i in range(4)]], 0)
        assert isinstance(result, _impl._ForwardRefusal)
    assert statuses == [2]


def test_local_fallback_error_precedes_vote_error(monkeypatch):
    rank = _rank(monkeypatch)
    _packed_budget(monkeypatch, rank, 20)
    primary = RuntimeError("local planner")

    def failed(*args, **kwargs):
        raise primary

    def vote(status):
        assert status == 0
        raise ValueError("secondary exchange")

    monkeypatch.setattr(rank, "_find_admissible_forward", failed)
    monkeypatch.setattr(rank, "_admission_outcome", vote)
    with pytest.raises(RuntimeError) as caught:
        rank._search_next_micro_batch([[_request(i) for i in range(4)]], 0)
    assert caught.value is primary


def test_trust_only_cold_plan_does_not_offer_cache_upgrade(monkeypatch):
    rank = _rank(monkeypatch)
    _packed_budget(monkeypatch, rank, 100)
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **kw: False)
    monkeypatch.setattr(
        rank,
        "_find_admissible_forward",
        lambda *a, **k: pytest.fail("no memory rejection"),
    )
    monkeypatch.setattr(
        rank,
        "_cache_recovery_episode",
        lambda: pytest.fail("trust does not enable release"),
    )
    selected = rank._select_next_micro_batch([[_request(i) for i in range(4)]], 0)
    assert selected.cold_start and selected.recovery_probe is None
