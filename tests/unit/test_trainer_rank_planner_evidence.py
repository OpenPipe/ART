"""Scalar admission evidence uses existing samples and never changes outcomes."""

from dataclasses import asdict
import gc
import json
import math
import sys
from types import FunctionType, SimpleNamespace, TracebackType
from typing import Any
import weakref

import pytest
import test_trainer_rank_cache_recovery as recovery
from test_trainer_rank_cache_recovery import fail, run
from test_trainer_rank_planner_options import _oversized, _reporting_rank
from test_trainer_rank_split import _request

from art.trainer_rank import _impl as tr
from art.trainer_rank import _planner_evidence as evidence
from art.trainer_rank import _planner_misses as reports


def sample(decision, required=100, available=90):
    return decision.observe(
        scope="local",
        local_required_bytes=required,
        local_available_bytes=available,
        reduced_required_bytes=required,
        reduced_available_bytes=available,
        safety_factor=1.1,
        reserve_fraction=0.01,
    )


@pytest.fixture
def scalar(request):
    fixture = recovery.TestRecovery()
    request.addfinalizer(fixture.doCleanups)
    return fixture.make()


def test_first_and_selected_are_not_latest_or_evicted():
    decision = evidence.Decision("dp_rank_forward", sync_across_dp=False)
    first = sample(decision, 100)
    selected = sample(decision, 80)
    for i in range(100):
        sample(decision, i)
    decision.selected, decision.outcome = selected, "refused"
    record = decision.snapshot()
    evidence.validate(record, None)
    assert record["first_sample"] == asdict(first)
    assert record["selected_sample"] == asdict(selected)
    assert record["trace"][-1]["values"]["local_required_bytes"] == 99
    assert len(record["trace"]) == 64
    assert record["omitted_trace_entries"] == 38


def test_nested_rank_context_isolated_and_restored():
    a, b = object(), object()
    outer = evidence.Decision("dp_rank_forward", sync_across_dp=False, owner=a)
    inner = evidence.Decision("dp_rank_forward", sync_across_dp=False, owner=b)
    with evidence.scope(outer):
        assert evidence.current(a) is outer and evidence.current(b) is None
        with evidence.scope(inner):
            assert evidence.current(a) is None and evidence.current(b) is inner
        assert evidence.current(a) is outer
    assert evidence.current() is None


@pytest.mark.parametrize("backend", ["native", "cudaMallocAsync"])
def test_memory_sampling_count_and_policy_unchanged(scalar, backend):
    rank, cuda, _, _ = scalar
    cuda.backend = backend
    calls = []

    def stats(device):
        calls.append("stats")
        return {
            "allocated_bytes.all.current": 10,
            "reserved_bytes.all.current": 70,
            "inactive_split_bytes.all.current": 20,
        }

    cuda.memory_stats = stats
    cuda.memory_allocated = lambda device: stats(device).get(
        "allocated_bytes.all.current", 0
    )
    cuda.memory_reserved = lambda device: stats(device).get(
        "reserved_bytes.all.current", 0
    )
    ordinary = rank._memory_check_required(80)
    ordinary_calls, ordinary_events = calls[:], cuda.events[:]
    calls.clear()
    cuda.events.clear()
    decision = evidence.Decision("dp_rank_forward", sync_across_dp=False, owner=rank)
    with evidence.scope(decision):
        observed = rank._memory_check_required(80)
    assert observed == ordinary
    assert calls == ordinary_calls
    assert cuda.events == ordinary_events == ["sample"]
    assert observed.sample.local_required_bytes == 80
    assert observed.sample.physical_free_bytes == 40
    assert observed.sample.reserved_bytes == 70
    assert observed.sample.inactive_split_bytes == (20 if backend == "native" else None)


def test_local_and_reduced_samples_remain_distinct(scalar, monkeypatch):
    rank, cuda, _, _ = scalar
    cuda.memory_stats = lambda device: {}
    dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        ReduceOp=SimpleNamespace(MAX="MAX", MIN="MIN"),
    )
    calls = []

    def reduce(value, *, op, group):
        calls.append((op, group))
        value.t.data[value.i] = 150 if op == "MAX" else 7

    dist.all_reduce = reduce
    monkeypatch.setattr(tr, "dist", dist)
    decision = evidence.Decision(
        "forward_micro_batches", sync_across_dp=True, owner=rank
    )
    with evidence.scope(decision):
        check = rank._memory_check_required(80, sync_across_dp=True)
    assert calls == [("MAX", None), ("MIN", None)]
    assert check.sample.local_required_bytes == 80
    assert check.sample.reduced_required_bytes == 150
    assert (
        check.sample.local_available_bytes != check.sample.reduced_available_bytes == 7
    )
    assert check.sample.scope == "world"


def test_final_refusal_emits_without_forward_or_memory_window(monkeypatch, tmp_path):
    rank = _oversized(monkeypatch)
    rank._allow_oversized_batches = False
    rank._planner_reporter = reports.Reporter(1e9, spool_dir=tmp_path / "reports")
    rank._planner_device_identity = {}
    with pytest.raises(tr.TrainerRankMemoryError):
        rank.dp_rank_forward([_request(i) for i in range(4)])
    raw = next(rank._planner_reporter.spool_dir.glob("*.json")).read_bytes()
    record = reports.validate_report(raw)
    assert record["event"] == "admission_refused"
    assert record["decision"]["outcome"] == "refused"
    assert (
        record["observed_peak_bytes"]
        is record["partial_peak_bytes"]
        is record["error_pct"]
        is None
    )
    assert record["failure"]["type"] == "TrainerRankMemoryError"
    assert record["phase"] == "planning"
    assert not getattr(rank, "_planner_active_observations", {})


def test_planning_error_original_identity_and_no_prediction(scalar, tmp_path):
    rank, _, _, _ = scalar
    rank._planner_reporter = reports.Reporter(5, spool_dir=tmp_path / "reports")
    original = ValueError("private text")
    _, caught, _ = run(rank, [original])
    assert caught is original
    raw = next(rank._planner_reporter.spool_dir.glob("*.json")).read_bytes()
    record = reports.validate_report(raw)
    assert record["event"] == "planning_error"
    assert record["predicted_peak_bytes"] is None and not record["replay_complete"]
    assert b"private text" not in raw


def test_cancellation_and_reporting_failure_never_replace_original(
    scalar, tmp_path, monkeypatch
):
    rank, _, _, _ = scalar
    rank._planner_reporter = reports.Reporter(5, spool_dir=tmp_path / "reports")

    def search():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        rank._recover_admission(
            search,
            lambda x: x,
            lambda x, c: x,
            context="dp_rank_forward",
            sync_across_dp=False,
        )
    assert not rank._planner_reporter.spool_dir.exists()
    monkeypatch.setattr(
        evidence, "failure", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    original = ValueError("original")
    _, caught, _ = run(rank, [original])
    assert caught is original


def test_actual_cache_release_trace_and_nonfinite_budget(scalar, tmp_path):
    rank, cuda, _, names = scalar
    rank._planner_reporter = reports.Reporter(5, spool_dir=tmp_path / "reports")
    cuda.memory_stats = lambda device: {}
    _, caught, _ = run(rank, [fail(names)])
    assert isinstance(caught, tr.TrainerRankMemoryError)
    record = reports.validate_report(
        next(rank._planner_reporter.spool_dir.glob("*.json")).read_bytes()
    )
    trace = record["decision"]["trace"]
    assert (
        sum(
            item["kind"] == "cache_release" and item["status"] == "attempted"
            for item in trace
        )
        == cuda.events.count("release")
        == 1
    )
    completed = next(
        item
        for item in trace
        if item["kind"] == "recovery" and item["status"] == "completed"
    )
    assert completed["values"]["observed_available_delta_bytes"] == 160
    decision = evidence.Decision("dp_rank_forward", sync_across_dp=False)
    decision.outcome = "planning_error"
    decision.record("recovery", "budget_observed", local_high_seconds=math.inf)
    value = decision.snapshot()
    evidence.validate(value, None)
    assert value["trace"][0]["values"]["local_high_seconds"] is None
    assert value["trace"][0]["unavailable_fields"] == ["local_high_seconds"]


def test_failure_stack_is_bounded_and_retains_no_frames():
    class Payload:
        pass

    obj = Payload()
    reference = weakref.ref(obj)

    def recurse(depth, private):
        if depth:
            return recurse(depth - 1, private)
        raise ValueError("do not include this text")

    try:
        recurse(50, obj)
    except ValueError as error:
        summary = evidence.failure(error, phase="planning")
    del obj
    gc.collect()
    evidence.validate(None, summary)
    assert reference() is None
    assert len(summary["frames"]) <= 32 and summary["omitted_frames"] > 0
    assert summary["frames"][-1]["function"] == "recurse"
    raw = json.dumps(summary).encode()
    assert len(raw) <= 8192 and b"do not include" not in raw
    assert b"filename" not in raw and b"locals" not in raw


@pytest.mark.parametrize("line", [-1, 0])
def test_locationless_failure_still_persists_oom(tmp_path, line):
    def frame():
        return sys._getframe()

    unlocated = FunctionType(frame.__code__.replace(co_linetable=b""), globals())()
    error = RuntimeError().with_traceback(TracebackType(None, unlocated, 0, line))
    failed = evidence.failure(error, phase="forward")
    assert failed["frames"][0]["line"] is None
    reporter = reports.Reporter(5, spool_dir=tmp_path / "reports")
    assert (
        reporter.report(
            predicted_peak_bytes=100,
            observed_peak_bytes=None,
            phase="forward",
            oom=True,
            failure=failed,
            replay_factory=lambda: {},
        )
        is not None
    )


def test_planning_flood_leaves_space_for_oom_and_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "MAX_PLANNING_REPORTS", 2)
    reporter = reports.Reporter(5, spool_dir=tmp_path / "rank")
    args: dict[str, Any] = dict(
        predicted_peak_bytes=None,
        observed_peak_bytes=None,
        phase="planning",
        event="planning_error",
        replay_factory=lambda: {},
    )
    retained = [reporter.report(**args) for _ in range(3)]
    assert all(retained[:2]) and retained[2] is None
    assert reporter.failures == 1
    # Reconstructing the reporter cannot reset the spool budget.
    assert reports.Reporter(5, spool_dir=reporter.spool_dir).report(**args) is None
    # Neither execution OOMs nor planning OOMs use the low-priority allowance.
    assert (
        reporter.report(
            **args,
            failure={
                "type": "OutOfMemoryError",
                "phase": "planning",
                "frames": [],
                "omitted_frames": 0,
            },
        )
        is not None
    )
    assert (
        reporter.report(
            predicted_peak_bytes=100,
            observed_peak_bytes=None,
            phase="forward",
            oom=True,
            replay_factory=lambda: {},
        )
        is not None
    )
    # Delivery reconsumes original bytes under the existing driver budget.
    for path in reporter.spool_dir.glob("*.json"):
        assert reports.persist_report(path.read_bytes(), tmp_path / "driver")


def test_large_planning_replay_keeps_bounded_scalar_event(tmp_path):
    reporter = reports.Reporter(5, spool_dir=tmp_path / "rank")
    path = reporter.report(
        predicted_peak_bytes=None,
        observed_peak_bytes=None,
        phase="planning",
        event="planning_error",
        replay_factory=lambda: {"payload": "x" * 300_000},
    )
    assert path is not None and path.stat().st_size <= 256 * 1024
    record = reports.validate_report(path.read_bytes())
    assert record["event"] == "planning_error" and record["replay"] is None
    assert record["incomplete_reasons"] == ["planning replay exceeds report limit"]


def test_summary_trimming_preserves_first_selected(monkeypatch):
    decision = evidence.Decision("dp_rank_forward", sync_across_dp=False)
    decision.first = decision.selected = sample(decision)
    decision.outcome = "refused"
    for i in range(60):
        decision.record("recovery", "skipped", reason="x" * 128)
    monkeypatch.setattr(evidence, "MAX_SUMMARY_BYTES", 2500)
    value = evidence.bounded(decision.snapshot())
    evidence.validate(value, None)
    assert value is not None
    assert value["first_sample"] == value["selected_sample"] == asdict(decision.first)
    assert value["omitted_trace_entries"] > 0


def test_legacy_report_reader_remains_supported(tmp_path):
    reporter = reports.Reporter(5, spool_dir=tmp_path / "reports")
    path = reporter.report(
        predicted_peak_bytes=100,
        observed_peak_bytes=120,
        phase="forward",
        replay_factory=lambda: {},
    )
    assert path is not None
    record = json.loads(path.read_bytes())
    record["format"] = 1
    for key in ("event", "decision", "failure"):
        record.pop(key)
    assert reports.validate_report(reports._encode(record))["format"] == 1


def test_planning_oom_does_not_borrow_previous_forward(monkeypatch, tmp_path):
    rank, plan, _ = _reporting_rank(monkeypatch, tmp_path)
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
    )
    original = tr.torch.cuda.OutOfMemoryError("new planning allocation")
    _, caught, _ = run(rank, [original])
    assert caught is original
    rank.report_planner_oom(caught)
    rank.finish_planner_observation()
    records = [
        reports.validate_report(path.read_bytes()) for path in tmp_path.glob("*.json")
    ]
    assert [record["event"] for record in records] == ["planning_error"]
    assert records[0]["partial_peak_bytes"] is None


def test_refresh_preserves_local_estimate_and_separate_reduced_operand(
    scalar, monkeypatch
):
    rank, cuda, _, _ = scalar
    cuda.memory_stats = lambda device: {}
    reductions = []
    required = iter((150, 170))
    dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        ReduceOp=SimpleNamespace(MAX="MAX", MIN="MIN"),
    )

    def reduce(value, *, op, group):
        reductions.append((op, group))
        value.t.data[value.i] = next(required) if op == "MAX" else 7

    dist.all_reduce = reduce
    monkeypatch.setattr(tr, "dist", dist)
    decision = evidence.Decision(
        "forward_micro_batches", sync_across_dp=True, owner=rank
    )
    with evidence.scope(decision):
        first = rank._memory_check_required(80, sync_across_dp=True)
        selected = rank._refresh_memory_check(first, sync_across_dp=True)
    assert reductions == [("MAX", None), ("MIN", None)] * 2
    assert (
        first.sample.local_required_bytes == selected.sample.local_required_bytes == 80
    )
    assert (
        selected.sample.required_operand_bytes == first.estimated_required_bytes == 150
    )
    assert (
        selected.sample.reduced_required_bytes
        == selected.estimated_required_bytes
        == 170
    )
    assert selected.sample.required_source == "previous_check"
    assert selected.sample.required_from_ordinal == first.sample.ordinal
    decision.selected, decision.outcome = selected.sample, "admitted"
    evidence.validate(decision.snapshot(), None)


def test_refresh_without_original_sample_leaves_local_requirement_unknown():
    decision = evidence.Decision("dp_rank_forward", sync_across_dp=False)
    with decision.refresh_of(None):
        value = sample(decision, 150)
    assert value.local_required_bytes is value.required_from_ordinal is None
    assert (
        value.required_operand_bytes == 150
        and value.required_source == "previous_check"
    )
    assert sample(decision, 80).local_required_bytes == 80
