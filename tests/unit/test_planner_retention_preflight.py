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
    assert ledger(bound) == {
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


def test_nearly_full_budget_bounds_encoding_and_preserves_small_oom(
    tmp_path, monkeypatch
):
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
        assert path is not None
        record = reports.validate_report(path.read_bytes())
        assert record["replay"] is None and not record["replay_complete"]
        assert record["incomplete_reasons"] == [
            "replay exceeds remaining retention allowance"
        ]
        assert len(visited) < 1000  # Parent traverses all 100,000 rows.
        small = emit(reporter)
        oom = emit(reporter, oom=True, observed_peak_bytes=None, partial_peak_bytes=99)
        assert small is not None and oom is not None
        saved = reports.validate_report(oom.read_bytes())
        assert saved["oom"] and saved["observed_peak_bytes"] is None
        assert saved["partial_peak_bytes"] == 99
    sizes = sum(p.stat().st_size for p in (path, small, oom))
    assert (
        sum(x[1] for x in ledger(bound)["charges"].values()) == sizes <= bound.max_bytes
    )
    assert ledger(bound)["omitted"] == reporter.failures == 0


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
