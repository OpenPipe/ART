from __future__ import annotations

from types import SimpleNamespace

import pytest

from art.trainer_rank import _telemetry


def test_guard_failure_delta_accepts_runtime_string_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = object()
    failures = {code: ["plain reason", SimpleNamespace(reason="named reason")]}
    monkeypatch.setattr(_telemetry.torch._dynamo, "guard_failures", failures)
    _telemetry._guard_counts.clear()

    assert _telemetry._new_guard_failures() == ("plain reason", "named reason")
    assert _telemetry._new_guard_failures() == ()


def test_phase_reports_compile_attribution_and_recompiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[dict[str, object], bool]] = []
    clock = iter((0.0, 1.0, 3.0, 5.0, 6.0, 8.0, 11.0, 12.0))
    monkeypatch.setattr(_telemetry, "_install", lambda: None)
    monkeypatch.setattr(_telemetry.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        _telemetry,
        "_emit",
        lambda event, warning=False: events.append((dict(event), warning)),
    )
    monkeypatch.setattr(_telemetry, "_new_guard_failures", lambda: ("size mismatch",))
    _telemetry._compiled_signatures.clear()
    args = SimpleNamespace(
        callback_trigger=SimpleNamespace(name="DYNAMO"), compile_id="0/0"
    )

    with _telemetry.phase("forward", {"packed_tokens": 128}, synchronized=True):
        _telemetry._compile_start(args)
        _telemetry._compile_end(args)
    with _telemetry.phase("forward", {"packed_tokens": 128}, synchronized=True):
        _telemetry._compile_start(args)
        _telemetry._compile_end(args)

    first, second = events
    assert first[0] == {
        "event": "phase",
        "phase": "forward",
        "seconds": 5.0,
        "synchronized": True,
        "signature": {"packed_tokens": 128},
        "compile_status": "new_signature",
        "compile_seconds": 2.0,
        "compile_ids": ["0/0"],
        "compile_triggers": ["DYNAMO"],
        "guard_failures": ["size mismatch"],
        "unique_compile_signatures": 1,
        "outcome": "ok",
        "error_type": None,
    }
    assert first[1]
    assert second[0]["compile_status"] == "recompile"
    assert second[0]["unique_compile_signatures"] == 1
    assert second[1]


def test_phase_without_compilation_reports_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    clock = iter((2.0, 5.0))
    monkeypatch.setattr(_telemetry, "_install", lambda: None)
    monkeypatch.setattr(_telemetry.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        _telemetry,
        "_emit",
        lambda event, warning=False: events.append(dict(event)),
    )
    _telemetry._compiled_signatures.clear()

    with pytest.raises(ValueError, match="failed"):
        with _telemetry.phase("optim", {"checkpoint_count": 1}):
            raise ValueError("failed")

    assert events == [
        {
            "event": "phase",
            "phase": "optim",
            "seconds": 3.0,
            "synchronized": False,
            "signature": {"checkpoint_count": 1},
            "compile_status": "none",
            "compile_seconds": 0.0,
            "compile_ids": [],
            "compile_triggers": [],
            "guard_failures": [],
            "unique_compile_signatures": 0,
            "outcome": "error",
            "error_type": "ValueError",
        }
    ]
