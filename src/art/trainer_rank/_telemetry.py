"""Structured host-phase and torch.compile telemetry for trainer-rank processes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
import threading
import time
from typing import Any, TypedDict

import torch

logger = logging.getLogger("art.trainer_rank.telemetry")

_install_lock = threading.Lock()
_state_lock = threading.Lock()
_installed = False
_guard_counts: dict[object, int] = {}
_compiled_plan_signatures: set[str] = set()
_active_compile: _Compile | None = None
_thread_state = threading.local()
_SLOW_PHASE_SECONDS = 10.0


@dataclass
class _Phase:
    name: str
    signature: Mapping[str, object]
    signature_key: str
    start: float
    compiles: list[_CompileRecord] = field(default_factory=list)
    closed: bool = False


class _CompileRecord(TypedDict):
    compile_id: str
    trigger: str
    seconds: float
    guard_failures: tuple[str, ...]
    frame_id: int | None
    frame_compile_id: int | None
    graph_status: str


@dataclass(frozen=True)
class _Compile:
    start: float
    compile_id: str
    trigger: str
    guard_failures: tuple[str, ...]
    phase: _Phase | None


def _compile_identity(compile_id: str) -> tuple[int | None, int | None, str]:
    """Extract Dynamo's frame and per-frame compilation numbers."""

    parts = compile_id.removeprefix("!").split("/")
    if compile_id.startswith("!") and len(parts) == 1:
        return None, None, "compiled_autograd"
    try:
        frame_id, frame_compile_id = int(parts[-2]), int(parts[-1])
    except (IndexError, ValueError):
        return None, None, "unknown"
    return (
        frame_id,
        frame_compile_id,
        "new_graph" if frame_compile_id == 0 else "recompile",
    )


def _emit(event: Mapping[str, object], *, warning: bool = False) -> None:
    log = logger.warning if warning else logger.info
    log("ART_TRAINER_EVENT %s", json.dumps(event, sort_keys=True, default=str))


def _phase_stack() -> list[_Phase]:
    stack = getattr(_thread_state, "phases", None)
    if stack is None:
        stack = []
        _thread_state.phases = stack
    return stack


def _new_guard_failures() -> tuple[str, ...]:
    reasons: list[str] = []
    failures = torch._dynamo.guard_failures
    with _state_lock:
        for code, values in failures.items():
            start = _guard_counts.get(code, 0)
            reasons.extend(
                str(getattr(value, "reason", value))[:1000] for value in values[start:]
            )
            _guard_counts[code] = len(values)
    return tuple(reasons)


def _compile_start(args: Any) -> None:
    global _active_compile
    try:
        trigger = getattr(getattr(args, "callback_trigger", None), "name", None)
        stack = _phase_stack()
        active = _Compile(
            start=time.perf_counter(),
            compile_id=str(getattr(args, "compile_id", "unknown")),
            trigger=str(trigger or getattr(args, "callback_trigger", "unknown")),
            guard_failures=_new_guard_failures(),
            phase=stack[-1] if stack else None,
        )
        with _state_lock:
            _active_compile = active
    except Exception:
        logger.debug("Failed to begin ART compile telemetry", exc_info=True)


def _compile_end(_args: Any) -> None:
    global _active_compile
    try:
        with _state_lock:
            completed = _active_compile
            _active_compile = None
        if not isinstance(completed, _Compile):
            return
        seconds = max(0.0, time.perf_counter() - completed.start)
        frame_id, frame_compile_id, graph_status = _compile_identity(
            completed.compile_id
        )
        record: _CompileRecord = {
            "compile_id": completed.compile_id,
            "trigger": completed.trigger,
            "seconds": seconds,
            "guard_failures": completed.guard_failures,
            "frame_id": frame_id,
            "frame_compile_id": frame_compile_id,
            "graph_status": graph_status,
        }
        closed_phase: _Phase | None = None
        repeated_plan = False
        unique_signatures = 0
        if completed.phase is not None:
            with _state_lock:
                if completed.phase.closed:
                    closed_phase = completed.phase
                    repeated_plan = (
                        closed_phase.signature_key in _compiled_plan_signatures
                    )
                    _compiled_plan_signatures.add(closed_phase.signature_key)
                    unique_signatures = len(_compiled_plan_signatures)
                else:
                    completed.phase.compiles.append(record)
            if closed_phase is None:
                return
        event: dict[str, object] = {
            "event": "compile",
            "phase": closed_phase.name if closed_phase is not None else "unscoped",
            **record,
        }
        if closed_phase is not None:
            event.update(
                {
                    "signature": closed_phase.signature,
                    "phase_closed": True,
                    "plan_signature_status": ("repeated" if repeated_plan else "new"),
                    "unique_compile_plan_signatures": unique_signatures,
                }
            )
        _emit(event, warning=True)
    except Exception:
        logger.debug("Failed to finish ART compile telemetry", exc_info=True)


def _install() -> None:
    global _installed
    with _install_lock:
        handler = torch._dynamo.callback_handler
        if (
            _compile_start in handler.start_callbacks
            and _compile_end in handler.end_callbacks
        ):
            _installed = True
            return
        with _state_lock:
            _guard_counts.clear()
            _guard_counts.update(
                (code, len(values))
                for code, values in torch._dynamo.guard_failures.items()
            )
        if _compile_start not in handler.start_callbacks:
            handler.register_start_callback(_compile_start)
        if _compile_end not in handler.end_callbacks:
            handler.register_end_callback(_compile_end)
        _installed = True


@contextmanager
def phase(
    name: str,
    signature: Mapping[str, object],
    *,
    dedup_signature: Mapping[str, object] | None = None,
    synchronized: bool = False,
) -> Iterator[None]:
    """Emit one structured phase event, including compile work observed within it."""

    _install()
    signature_key = json.dumps(
        {
            "phase": name,
            "signature": signature if dedup_signature is None else dedup_signature,
        },
        sort_keys=True,
        default=str,
    )
    record = _Phase(name, signature, signature_key, time.perf_counter())
    stack = _phase_stack()
    stack.append(record)
    error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        if not stack or stack.pop() is not record:
            logger.debug("Trainer telemetry phase stack changed unexpectedly")
        with _state_lock:
            record.closed = True
            compiles = list(record.compiles)
            repeated_plan = False
            if compiles:
                repeated_plan = record.signature_key in _compiled_plan_signatures
                _compiled_plan_signatures.add(record.signature_key)
            unique_signatures = len(_compiled_plan_signatures)
        seconds = max(0.0, time.perf_counter() - record.start)
        compile_seconds = sum(item["seconds"] for item in compiles)
        graph_statuses = {str(item["graph_status"]) for item in compiles}
        _emit(
            {
                "event": "phase",
                "phase": name,
                "seconds": seconds,
                "synchronized": synchronized,
                "signature": signature,
                "compile_status": (
                    "recompile"
                    if "recompile" in graph_statuses
                    else "new_graph"
                    if "new_graph" in graph_statuses
                    else "compiled_autograd"
                    if "compiled_autograd" in graph_statuses
                    else "unknown"
                    if compiles
                    else "none"
                ),
                "compile_seconds": compile_seconds,
                "compiles": compiles,
                "plan_signature_status": (
                    "repeated" if repeated_plan else "new" if compiles else "none"
                ),
                "unique_compile_plan_signatures": unique_signatures,
                "outcome": "error" if error is not None else "ok",
                "error_type": type(error).__name__ if error is not None else None,
            },
            warning=bool(compiles) or seconds >= _SLOW_PHASE_SECONDS,
        )
