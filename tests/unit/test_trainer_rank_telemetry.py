from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import cast

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
    _telemetry._compiled_plan_signatures.clear()
    first_args = SimpleNamespace(
        callback_trigger=SimpleNamespace(name="DYNAMO"), compile_id="0/0"
    )
    second_args = SimpleNamespace(
        callback_trigger=SimpleNamespace(name="DYNAMO"), compile_id="0/1"
    )

    with _telemetry.phase("forward", {"packed_tokens": 128}, synchronized=True):
        _telemetry._compile_start(first_args)
        _telemetry._compile_end(first_args)
    with _telemetry.phase("forward", {"packed_tokens": 128}, synchronized=True):
        _telemetry._compile_start(second_args)
        _telemetry._compile_end(second_args)

    first, second = events
    assert first[0] == {
        "event": "phase",
        "phase": "forward",
        "seconds": 5.0,
        "synchronized": True,
        "signature": {"packed_tokens": 128},
        "compile_status": "new_graph",
        "compile_seconds": 2.0,
        "compiles": [
            {
                "compile_id": "0/0",
                "trigger": "DYNAMO",
                "seconds": 2.0,
                "guard_failures": ("size mismatch",),
                "frame_id": 0,
                "frame_compile_id": 0,
                "graph_status": "new_graph",
            }
        ],
        "plan_signature_status": "new",
        "unique_compile_plan_signatures": 1,
        "outcome": "ok",
        "error_type": None,
    }
    assert first[1]
    assert second[0]["compile_status"] == "recompile"
    second_compiles = cast(list[dict[str, object]], second[0]["compiles"])
    assert second_compiles[0]["compile_id"] == "0/1"
    assert second[0]["plan_signature_status"] == "repeated"
    assert second[0]["unique_compile_plan_signatures"] == 1
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
    _telemetry._compiled_plan_signatures.clear()

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
            "compiles": [],
            "plan_signature_status": "none",
            "unique_compile_plan_signatures": 0,
            "outcome": "error",
            "error_type": "ValueError",
        }
    ]


def test_compile_end_on_another_thread_keeps_phase_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    clock = iter((0.0, 1.0, 3.0, 4.0))
    monkeypatch.setattr(_telemetry, "_install", lambda: None)
    monkeypatch.setattr(_telemetry.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        _telemetry,
        "_emit",
        lambda event, warning=False: events.append(dict(event)),
    )
    monkeypatch.setattr(_telemetry, "_new_guard_failures", lambda: ())
    _telemetry._compiled_plan_signatures.clear()
    args = SimpleNamespace(
        callback_trigger=SimpleNamespace(name="DYNAMO"), compile_id="4/0"
    )

    with _telemetry.phase("forward", {"packed_tokens": 64}):
        _telemetry._compile_start(args)
        thread = threading.Thread(target=_telemetry._compile_end, args=(args,))
        thread.start()
        thread.join()

    assert events[0]["compiles"] == [
        {
            "compile_id": "4/0",
            "trigger": "DYNAMO",
            "seconds": 2.0,
            "guard_failures": (),
            "frame_id": 4,
            "frame_compile_id": 0,
            "graph_status": "new_graph",
        }
    ]


def test_compile_finishing_after_phase_close_emits_attributed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []
    clock = iter((0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 9.0))
    monkeypatch.setattr(_telemetry, "_install", lambda: None)
    monkeypatch.setattr(_telemetry.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        _telemetry,
        "_emit",
        lambda event, warning=False: events.append(dict(event)),
    )
    monkeypatch.setattr(_telemetry, "_new_guard_failures", lambda: ())
    _telemetry._compiled_plan_signatures.clear()
    args = SimpleNamespace(
        callback_trigger=SimpleNamespace(name="DYNAMO"), compile_id="5/0"
    )

    with _telemetry.phase("forward", {"packed_tokens": 96}):
        _telemetry._compile_start(args)
    _telemetry._compile_end(args)

    assert events[0]["compiles"] == []
    assert events[1] == {
        "event": "compile",
        "phase": "forward",
        "signature": {"packed_tokens": 96},
        "phase_closed": True,
        "plan_signature_status": "new",
        "unique_compile_plan_signatures": 1,
        "compile_id": "5/0",
        "trigger": "DYNAMO",
        "seconds": 3.0,
        "guard_failures": (),
        "frame_id": 5,
        "frame_compile_id": 0,
        "graph_status": "new_graph",
    }

    repeated_args = SimpleNamespace(
        callback_trigger=SimpleNamespace(name="DYNAMO"), compile_id="5/1"
    )
    with _telemetry.phase("forward", {"packed_tokens": 96}):
        _telemetry._compile_start(repeated_args)
    _telemetry._compile_end(repeated_args)

    assert events[3]["plan_signature_status"] == "repeated"
    assert events[3]["unique_compile_plan_signatures"] == 1


def test_install_restores_callbacks_after_handler_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Handler:
        def __init__(self) -> None:
            self.start_callbacks: list[object] = []
            self.end_callbacks: list[object] = []

        def register_start_callback(self, callback: object) -> None:
            self.start_callbacks.append(callback)

        def register_end_callback(self, callback: object) -> None:
            self.end_callbacks.append(callback)

    handler = Handler()
    monkeypatch.setattr(_telemetry.torch._dynamo, "callback_handler", handler)
    code = object()
    failures = {code: ["old failure", "old failure 2"]}
    monkeypatch.setattr(_telemetry.torch._dynamo, "guard_failures", failures)

    _telemetry._install()
    assert handler.start_callbacks == [_telemetry._compile_start]
    assert handler.end_callbacks == [_telemetry._compile_end]
    handler.start_callbacks.clear()
    handler.end_callbacks.clear()
    failures[code] = ["fresh failure"]
    _telemetry._install()
    assert handler.start_callbacks == [_telemetry._compile_start]
    assert handler.end_callbacks == [_telemetry._compile_end]
    failures[code].append("next failure")
    assert _telemetry._new_guard_failures() == ("next failure",)
