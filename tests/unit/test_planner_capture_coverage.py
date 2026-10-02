"""Scalar miss/OOM coverage survives cumulative full-report exhaustion."""

import json

import pytest
from test_planner_retention_budget import emit, ledger, limits

from art.trainer_rank import _planner_misses as reports
from art.trainer_rank import _planner_retention as retention


def test_luna_sized_omission_preserves_summary_and_smaller_report(
    tmp_path, monkeypatch
):
    # Actual retained Luna budget: ten captures, 15,659,200 of 16 MiB.
    bound = limits(tmp_path, max_bytes=16 * 1024**2, max_reports=64)
    charges = {
        f"{i:032x}": ["0" * 64, n]
        for i, n in enumerate(
            [
                1492598,
                1406242,
                1812159,
                1627974,
                1785980,
                139798,
                2178624,
                1517299,
                1489178,
                2209348,
            ]
        )
    }
    bound.spool_dir.mkdir(mode=0o700)
    retention._write(
        bound.spool_dir / ".retention.json",
        {
            "allowance": bound.identity(),
            "charges": charges,
            "omitted": 32,
            "omitted_bytes": 56624007,
        },
    )
    monkeypatch.setattr(reports, "_source_files", lambda: {})
    reporter = reports.Reporter(10)
    with reports.report_retention_scope(bound):
        assert (
            emit(
                reporter,
                predicted_peak_bytes=5111328762,
                observed_peak_bytes=4239022080,
                phase="forward_and_caller",
                replay_factory=lambda: {"bulk": "x" * 1400000},
            )
            is None
        )
        state = retention.coverage(bound)
        assert state["remaining_bytes"] == 1118016
        assert state["remaining_reports"] == 54
        assert state["omitted"] == 33
        event = state["event_summaries"]["estimate_miss"]
        assert (
            event["omitted"] == 1
        )  # No invented summaries for 32 historical omissions.
        assert event["retained"] == 0
        assert event["latest"]["error_pct"] == pytest.approx(17.0661431228)
        assert event["latest"]["observed_peak_bytes"] == 4239022080
        assert "bulk" not in json.dumps(state)
        assert ledger(bound)["charges"] == charges  # No refund or premature exhaustion.
        assert emit(reporter, oom=True, observed_peak_bytes=None, partial_peak_bytes=99)
    assert retention.coverage(bound)["event_summaries"]["oom"]["retained"] == 1


@pytest.mark.parametrize("capacity", [{"max_bytes": 0}, {"max_reports": 0}])
def test_exhaustion_keeps_fixed_size_event_coverage_without_replay(
    tmp_path, monkeypatch, capacity
):
    bound = limits(tmp_path, **capacity)
    reporter = reports.Reporter(10)

    def unexpected():
        raise AssertionError("must not construct replay")

    with reports.report_retention_scope(bound):
        for index in range(200):
            assert (
                emit(reporter, phase="phase" * 100, replay_factory=unexpected) is None
            )
        assert (
            emit(
                reporter,
                oom=True,
                observed_peak_bytes=None,
                partial_peak_bytes=1024,
                replay_factory=unexpected,
            )
            is None
        )
        assert (
            emit(
                reporter,
                event="planning_error",
                phase="planning",
                observed_peak_bytes=None,
                replay_factory=unexpected,
                failure={
                    "type": "OutOfMemoryError",
                    "phase": "planning",
                    "frames": [],
                    "omitted_frames": 0,
                },
            )
            is None
        )
    state = retention.coverage(bound)
    retention.validate_coverage(state, allowance=bound.identity())
    assert state["charged_bytes"] == state["charged_reports"] == 0
    assert state["omitted_unmeasured"] == state["omitted"] == 202
    events = state["event_summaries"]
    assert events["estimate_miss"]["omitted"] == 200
    assert len(events["estimate_miss"]["latest"]["phase"]) == 80
    assert events["oom"]["latest"]["observed_peak_bytes"] is None
    assert events["oom"]["latest"]["partial_peak_bytes"] == 1024
    assert events["planning_error"]["latest"]["failure_type"] == "OutOfMemoryError"
    assert len(json.dumps(state)) < 4096
    assert {p.name for p in bound.spool_dir.iterdir()} == {
        ".retention.json",
        ".retention.lock",
    }


def test_summary_failure_does_not_replace_training_error_or_capture(
    tmp_path, monkeypatch
):
    bound = limits(tmp_path)
    monkeypatch.setattr(
        retention, "summarize", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    error = RuntimeError("original training failure")
    with pytest.raises(RuntimeError) as caught:
        with reports.report_retention_scope(bound):
            assert emit(reports.Reporter(10)) is not None
            raise error
    assert caught.value is error
    assert "event_summaries" not in ledger(
        bound
    )  # Missing is unknown, not zero events.


def test_delivery_persistence_does_not_fabricate_producer_summary(tmp_path):
    path = emit(reports.Reporter(10, spool_dir=tmp_path / "producer"))
    bound = limits(tmp_path)
    reports.persist_report(path.read_bytes(), bound.spool_dir, retention=bound)
    assert retention.coverage(bound)["event_summaries"] == {}
    assert retention.coverage(bound)["charged_reports"] == 1
