"""Assigned quotas count captured bytes once, including after source reclamation."""

from dataclasses import replace
import json

import pytest

from art.trainer_rank import _planner_misses as reports


def limits(tmp_path, **values):
    return reports.RetentionLimits(
        spool_dir=tmp_path / "assigned",
        max_bytes=values.pop("max_bytes", 1024 * 1024),
        max_reports=values.pop("max_reports", 4),
        allowance_id="a" * 32,
        execution_id="b" * 32,
        producer_id="c" * 32,
        **values,
    )


def emit(reporter, **values):
    return reporter.report(
        **{
            "predicted_peak_bytes": 100,
            "observed_peak_bytes": 200,
            "phase": "forward",
            "replay_factory": lambda: {},
            **values,
        }
    )


def ledger(bound):
    return json.loads((bound.spool_dir / ".retention.json").read_bytes())


def test_reclamation_never_refunds_cumulative_budget(tmp_path):
    bound = limits(tmp_path, max_reports=1)
    reporter = reports.Reporter(5, spool_dir=tmp_path / "standalone")
    with reports.report_retention_scope(bound):
        path = emit(reporter)
        assert path is not None and path.parent == bound.spool_dir
        raw = path.read_bytes()
        original = ledger(bound)
        path.unlink()
        assert emit(reporter) is None
    assert not reporter.spool_dir.exists()
    assert ledger(bound)["charges"] == original["charges"]
    assert ledger(bound)["omitted"] == 1
    assert ledger(bound)["omitted_bytes"] > 0
    # Retransmission of the same captured bytes spends nothing new, even if
    # recreating a previously reclaimed source copy for the original report.
    assert (
        reports.persist_report(raw, bound.spool_dir, retention=bound).read_bytes()
        == raw
    )
    assert ledger(bound)["charges"] == original["charges"]


def test_origin_scope_is_nested_and_exception_safe(tmp_path):
    a = limits(tmp_path)
    b = replace(a, spool_dir=tmp_path / "other", allowance_id="d" * 32)
    reporter = reports.Reporter(5, spool_dir=tmp_path / "standalone")
    with reports.report_retention_scope(a):
        with pytest.raises(RuntimeError):
            with reports.report_retention_scope(b):
                assert emit(reporter).parent == b.spool_dir
                raise RuntimeError("science")
        assert emit(reporter).parent == a.spool_dir
    assert emit(reporter).parent == reporter.spool_dir


@pytest.mark.parametrize(
    "field,value",
    [
        ("producer_id", "d" * 32),
        ("execution_id", "e" * 32),
        ("allowance_id", "f" * 32),
        ("max_bytes", 2 * 1024 * 1024),
        ("max_reports", 5),
    ],
)
def test_enrollment_cannot_be_replaced_or_replenished(tmp_path, field, value):
    bound = limits(tmp_path)
    reporter = reports.Reporter(5)
    with reports.report_retention_scope(bound):
        assert emit(reporter) is not None
    original = (bound.spool_dir / ".retention.json").read_bytes()
    with reports.report_retention_scope(replace(bound, **{field: value})):
        assert emit(reporter) is None
    assert (bound.spool_dir / ".retention.json").read_bytes() == original


def test_ambiguous_payload_write_keeps_charge(tmp_path, monkeypatch):
    bound = limits(tmp_path, max_reports=1)
    reporter = reports.Reporter(5)

    def fail(*args):
        raise OSError("synthetic failure after charge")

    with reports.report_retention_scope(bound):
        with monkeypatch.context() as patch:
            patch.setattr(reports.os, "link", fail)
            assert emit(reporter) is None
        assert len(ledger(bound)["charges"]) == 1
        assert emit(reporter) is None
    assert not list(bound.spool_dir.glob("[0-9a-f]*.json"))

    assert ledger(bound)["omitted"] == 2
    assert ledger(bound)["omitted_bytes"] > 0
    assert reporter.failures == 2


def test_planning_flood_keeps_oom_headroom_after_reclaim(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "MAX_PLANNING_REPORTS", 1)
    bound = limits(tmp_path)
    reporter = reports.Reporter(5)
    planning = {
        "event": "planning_error",
        "phase": "planning",
        "observed_peak_bytes": None,
    }
    with reports.report_retention_scope(bound):
        first = emit(reporter, **planning)
        assert first is not None
        first.unlink()
        assert emit(reporter, **planning) is None
        assert (
            emit(reporter, oom=True, observed_peak_bytes=None, partial_peak_bytes=10)
            is not None
        )
    assert len(ledger(bound)["charges"]) == 2


@pytest.mark.parametrize(
    "bound_values", [{"max_bytes": 0}, {"max_reports": 0}, {"max_bytes": 1}]
)
def test_exhausted_allowance_has_no_fallback(tmp_path, bound_values):
    bound = limits(tmp_path, **bound_values)
    reporter = reports.Reporter(5, spool_dir=tmp_path / "standalone")
    with reports.report_retention_scope(bound):
        assert emit(reporter) is None
    assert not reporter.spool_dir.exists()
    assert not list(bound.spool_dir.glob("[0-9a-f]*.json"))
    assert ledger(bound)["omitted"] == 1
    assert ledger(bound)["omitted_bytes"] > 0


def test_unaccounted_spool_and_corrupt_ledger_refuse(tmp_path):
    bound = limits(tmp_path)
    bound.spool_dir.mkdir(mode=0o700)
    foreign = bound.spool_dir / "foreign.json"
    foreign.write_text("retained")
    reporter = reports.Reporter(5)
    with reports.report_retention_scope(bound):
        assert emit(reporter) is None
        assert foreign.read_text() == "retained"
        foreign.unlink()
        assert emit(reporter) is not None
        (bound.spool_dir / ".retention.json").write_text("partial")
        assert emit(reporter) is None


def test_disabled_reporting_never_creates_assigned_spool(tmp_path):
    bound = limits(tmp_path)
    with reports.report_retention_scope(bound):
        assert emit(reports.Reporter()) is None
    assert not bound.spool_dir.exists()


@pytest.mark.parametrize("bounded_by", ["bytes", "reports"])
def test_interrupted_payload_is_not_duplicated_by_retry(tmp_path, bounded_by):
    seed = emit(reports.Reporter(5, spool_dir=tmp_path / "seed"))
    assert seed is not None
    raw = seed.read_bytes()
    bound = limits(
        tmp_path,
        **({"max_bytes": len(raw)} if bounded_by == "bytes" else {"max_reports": 1}),
    )
    committed = reports.persist_report(raw, bound.spool_dir, retention=bound)
    charge = ledger(bound)
    # Same durable state as death after payload fsync but before final link.
    orphan = bound.spool_dir / ".pending-interrupted"
    committed.rename(orphan)
    with pytest.raises(ValueError, match="spool is full"):
        reports.persist_report(raw, bound.spool_dir, retention=bound)
    assert orphan.read_bytes() == raw
    assert not committed.exists()
    assert ledger(bound)["charges"] == charge["charges"]
    assert ledger(bound)["omitted"] == charge["omitted"] + 1
    assert ledger(bound)["omitted_bytes"] == charge["omitted_bytes"] + len(raw)


def test_post_charge_refusal_counts_omission_without_refund(tmp_path):
    bound = limits(tmp_path)
    reporter = reports.Reporter(5)
    with reports.report_retention_scope(bound):
        assert emit(reporter) is not None
        original = ledger(bound)
        orphan = bound.spool_dir / ".pending-orphan"
        orphan.write_bytes(b"x" * bound.max_bytes)
        assert emit(reporter) is None
    checked = ledger(bound)
    assert len(checked["charges"]) == len(original["charges"]) + 1
    assert checked["omitted"] == 1
    assert checked["omitted_bytes"] > 0
    assert orphan.stat().st_size == bound.max_bytes


def test_omission_write_failure_preserves_original_failure(tmp_path, monkeypatch):
    bound = limits(tmp_path)
    seed = emit(reports.Reporter(5, spool_dir=tmp_path / "seed"))
    assert seed is not None
    raw = seed.read_bytes()
    original_write = reports._planner_retention._write
    calls = []

    def write(path, value):
        calls.append(None)
        if len(calls) == 2:
            raise OSError("omission ledger unavailable")
        original_write(path, value)

    def fail(*args):
        raise ValueError("payload failed")

    monkeypatch.setattr(reports._planner_retention, "_write", write)
    monkeypatch.setattr(reports.os, "link", fail)
    with pytest.raises(ValueError, match="payload failed"):
        reports.persist_report(raw, bound.spool_dir, retention=bound)
    assert len(calls) == 2
    assert len(ledger(bound)["charges"]) == 1
