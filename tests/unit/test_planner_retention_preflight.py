"""Do not construct unretainable reports or mistake a large miss for exhaustion."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import threading
from types import SimpleNamespace

import pytest
from test_planner_retention_budget import emit, ledger, limits

from art.trainer_rank import _planner_misses as reports


@pytest.mark.parametrize("capacity", [{"max_reports": 0}, {"max_bytes": 0}])
@pytest.mark.parametrize("oom", [False, True])
def test_definitive_exhaustion_skips_factory_source_and_encoding(
    tmp_path, monkeypatch, capacity, oom
):
    bound = limits(tmp_path, **capacity)
    reporter = reports.Reporter(5)
    calls = []

    def unexpected(*args, **kwargs):
        calls.append(True)
        raise AssertionError("unretainable replay was constructed")

    monkeypatch.setattr(reports, "_source_files", unexpected)
    monkeypatch.setattr(reports, "_encode", unexpected)
    with reports.report_retention_scope(bound):
        assert emit(reporter, replay_factory=unexpected, oom=oom) is None
    assert calls == []
    assert reporter.failures == 1
    value = ledger(bound)
    summary = value.pop("event_summaries")["oom" if oom else "estimate_miss"]
    assert summary["omitted"] == 1 and summary["retained"] == 0
    assert value == {
        "allowance": bound.identity(),
        "charges": {},
        "omitted": 1,
        "omitted_bytes": 0,
        "omitted_unmeasured": 1,
    }


def test_exact_byte_exhaustion_remains_exhausted_after_reclaim(tmp_path, monkeypatch):
    raw = emit(reports.Reporter(5, spool_dir=tmp_path / "seed")).read_bytes()
    bound = limits(tmp_path, max_bytes=len(raw))
    reports.persist_report(raw, bound.spool_dir, retention=bound).unlink()
    calls = []
    with reports.report_retention_scope(bound):
        assert emit(reports.Reporter(5), replay_factory=lambda: calls.append(1)) is None
    assert calls == []
    assert ledger(bound)["omitted_unmeasured"] == 1
    assert ledger(bound)["charges"] == {
        json.loads(raw)["id"]: [reports.hashlib.sha256(raw).hexdigest(), len(raw)]
    }


def test_healthy_bytes_and_accounting_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(reports.uuid, "uuid4", lambda: SimpleNamespace(hex="d" * 32))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(reports, "datetime", Clock)
    monkeypatch.setattr(reports, "_source_files", lambda: {})
    raw = emit(reports.Reporter(5, spool_dir=tmp_path / "seed")).read_bytes()
    bound = limits(tmp_path)
    with reports.report_retention_scope(bound):
        assert emit(reports.Reporter(5)).read_bytes() == raw
    assert ledger(bound)["charges"] == {
        "d" * 32: [reports.hashlib.sha256(raw).hexdigest(), len(raw)]
    }
    assert ledger(bound)["omitted"] == ledger(bound)["omitted_bytes"] == 0
    assert "omitted_unmeasured" not in ledger(bound)


def test_nearly_full_budget_keeps_original_refusal_and_small_oom(tmp_path, monkeypatch):
    bound = limits(tmp_path, max_bytes=4096)
    reporter = reports.Reporter(5)
    visited = []

    class Rows(list):
        def __iter__(self):
            for row in super().__iter__():
                visited.append(None)
                yield row

    monkeypatch.setattr(reports, "_source_files", lambda: {})
    with reports.report_retention_scope(bound):
        path = emit(reporter, replay_factory=lambda: {"rows": Rows([12345] * 100000)})
        assert path is None
        assert len(visited) == 100000  # Positive allowance keeps the original path.
        assert ledger(bound)["charges"] == {}
        small = emit(reporter)
        oom = emit(reporter, oom=True, observed_peak_bytes=None, partial_peak_bytes=99)
        assert small is not None and oom is not None
        saved = reports.validate_report(oom.read_bytes())
        assert saved["oom"] and saved["observed_peak_bytes"] is None
        assert saved["partial_peak_bytes"] == 99
    sizes = sum(p.stat().st_size for p in (small, oom))
    assert (
        sum(x[1] for x in ledger(bound)["charges"].values()) == sizes <= bound.max_bytes
    )
    assert ledger(bound)["omitted"] == reporter.failures == 1
    assert ledger(bound)["omitted_bytes"] > bound.max_bytes
    assert "omitted_unmeasured" not in ledger(bound)


def test_concurrent_preflights_do_not_reserve_or_overspend(tmp_path):
    bound = limits(tmp_path, max_reports=1)
    gate = threading.Barrier(2, timeout=3)

    def capture():
        gate.wait()
        return {}

    def run():
        with reports.report_retention_scope(bound):
            return emit(reports.Reporter(5), replay_factory=capture)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(run), pool.submit(run)
        paths = [a.result(timeout=5), b.result(timeout=5)]
    assert sum(path is not None for path in paths) == 1
    state = ledger(bound)
    assert len(state["charges"]) == 1 and state["omitted"] == 1
    assert state["omitted_bytes"] > 0 and "omitted_unmeasured" not in state


def test_planning_oom_bypasses_only_the_lower_planning_allowance(tmp_path, monkeypatch):
    bound = limits(tmp_path)
    monkeypatch.setattr(reports, "MAX_PLANNING_REPORTS", 0)
    reporter = reports.Reporter(5)
    args = dict(event="planning_error", phase="planning", observed_peak_bytes=None)
    with reports.report_retention_scope(bound):
        assert emit(reporter, **args) is None
        path = emit(
            reporter,
            **args,
            failure={
                "type": "OutOfMemoryError",
                "phase": "planning",
                "frames": [],
                "omitted_frames": 0,
            },
        )
        assert path is not None
    record = reports.validate_report(path.read_bytes())
    assert record["failure"]["type"] == "OutOfMemoryError"
    assert ledger(bound)["omitted_unmeasured"] == 1
    assert len(ledger(bound)["charges"]) == 1


@pytest.mark.parametrize("bad", [True, -1, 2])
def test_unknown_byte_counter_is_validated(tmp_path, bad):
    bound = limits(tmp_path, max_reports=0)
    with reports.report_retention_scope(bound):
        assert emit(reports.Reporter(5)) is None
    value = ledger(bound)
    value["omitted_unmeasured"] = bad
    reports._planner_retention._write(bound.spool_dir / ".retention.json", value)
    original = (bound.spool_dir / ".retention.json").read_bytes()
    with reports.report_retention_scope(bound):
        assert emit(reports.Reporter(5)) is None
    assert (bound.spool_dir / ".retention.json").read_bytes() == original


@pytest.mark.parametrize("capacity", [{"max_reports": 1}, {"max_bytes": 1200}])
def test_oversized_miss_does_not_displace_later_oom(tmp_path, monkeypatch, capacity):
    bound = limits(tmp_path, **capacity)
    reporter = reports.Reporter(5)
    monkeypatch.setattr(reports, "_source_files", lambda: {})
    with reports.report_retention_scope(bound):
        assert emit(reporter, replay_factory=lambda: {"rows": [12345] * 200000}) is None
        assert ledger(bound)["charges"] == {}
        path = emit(reporter, oom=True, observed_peak_bytes=None, partial_peak_bytes=99)
    assert path is not None
    saved = reports.validate_report(path.read_bytes())
    assert saved["oom"] and saved["partial_peak_bytes"] == 99
    assert len(ledger(bound)["charges"]) == 1
    assert ledger(bound)["omitted_bytes"] > bound.max_bytes


@pytest.mark.parametrize("kind", ["miss", "oom", "planning_oom"])
def test_near_full_allowance_preserves_original_static_fallback(
    tmp_path, monkeypatch, kind
):
    bound = limits(tmp_path, max_bytes=1500)
    monkeypatch.setattr(reports, "_source_files", lambda: {})
    monkeypatch.setattr(reports, "MAX_REPORT_BYTES", 2048)
    monkeypatch.setattr(reports, "MAX_PLANNING_REPORT_BYTES", 2048)
    values = {"replay_factory": lambda: {"bulk": "x" * 5000}}
    if kind == "oom":
        values.update(oom=True, observed_peak_bytes=None, partial_peak_bytes=99)
    elif kind == "planning_oom":
        values.update(
            event="planning_error",
            phase="planning",
            observed_peak_bytes=None,
            failure={
                "type": "OutOfMemoryError",
                "phase": "planning",
                "frames": [],
                "omitted_frames": 0,
            },
        )
    monkeypatch.setattr(reports.uuid, "uuid4", lambda: SimpleNamespace(hex="d" * 32))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(reports, "datetime", Clock)
    original = emit(reports.Reporter(5, spool_dir=tmp_path / "unassigned"), **values)
    assert original is not None
    with reports.report_retention_scope(bound):
        path = emit(reports.Reporter(5), **values)
    assert path is not None
    assert path.read_bytes() == original.read_bytes()
    record = reports.validate_report(path.read_bytes())
    assert not record["replay_complete"]
    assert record["incomplete_reasons"]
    assert len(ledger(bound)["charges"]) == 1
    assert ledger(bound)["omitted"] == 0
    if kind == "oom":
        assert record["oom"] and record["partial_peak_bytes"] == 99
    elif kind == "planning_oom":
        assert record["failure"]["type"] == "OutOfMemoryError"
