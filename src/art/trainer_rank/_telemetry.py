"""Structured host-phase and torch.compile telemetry for trainer-rank processes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
import threading
import time
from typing import Any

import torch

logger = logging.getLogger("art.trainer_rank.telemetry")

_install_lock = threading.Lock()
_state_lock = threading.Lock()
_installed = False
_guard_counts: dict[object, int] = {}
_compiled_signatures: set[str] = set()
_thread_state = threading.local()
_SLOW_PHASE_SECONDS = 10.0


@dataclass
class _Phase:
    name: str
    signature: Mapping[str, object]
    signature_key: str
    start: float
    compile_seconds: float = 0.0
    compile_ids: list[str] = field(default_factory=list)
    compile_triggers: list[str] = field(default_factory=list)
    guard_failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Compile:
    start: float
    compile_id: str
    trigger: str
    guard_failures: tuple[str, ...]


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
    try:
        trigger = getattr(getattr(args, "callback_trigger", None), "name", None)
        _thread_state.compile = _Compile(
            start=time.perf_counter(),
            compile_id=str(getattr(args, "compile_id", "unknown")),
            trigger=str(trigger or getattr(args, "callback_trigger", "unknown")),
            guard_failures=_new_guard_failures(),
        )
    except Exception:
        logger.debug("Failed to begin ART compile telemetry", exc_info=True)


def _compile_end(_args: Any) -> None:
    try:
        completed = getattr(_thread_state, "compile", None)
        if not isinstance(completed, _Compile):
            return
        del _thread_state.compile
        seconds = max(0.0, time.perf_counter() - completed.start)
        stack = _phase_stack()
        if stack:
            phase = stack[-1]
            phase.compile_seconds += seconds
            phase.compile_ids.append(completed.compile_id)
            phase.compile_triggers.append(completed.trigger)
            phase.guard_failures.extend(completed.guard_failures)
            return
        _emit(
            {
                "event": "compile",
                "phase": "unscoped",
                "compile_id": completed.compile_id,
                "trigger": completed.trigger,
                "seconds": seconds,
                "guard_failures": completed.guard_failures,
            },
            warning=True,
        )
    except Exception:
        logger.debug("Failed to finish ART compile telemetry", exc_info=True)


def _install() -> None:
    global _installed
    if _installed:
        return
    with _install_lock:
        if _installed:
            return
        with _state_lock:
            _guard_counts.update(
                (code, len(values))
                for code, values in torch._dynamo.guard_failures.items()
            )
        torch._dynamo.callback_handler.register_start_callback(_compile_start)
        torch._dynamo.callback_handler.register_end_callback(_compile_end)
        _installed = True


@contextmanager
def phase(
    name: str,
    signature: Mapping[str, object],
    *,
    synchronized: bool = False,
) -> Iterator[None]:
    """Emit one structured phase event, including compile work observed within it."""

    _install()
    signature_key = json.dumps(
        {"phase": name, "signature": signature}, sort_keys=True, default=str
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
        seconds = max(0.0, time.perf_counter() - record.start)
        repeated = False
        with _state_lock:
            if record.compile_ids:
                repeated = record.signature_key in _compiled_signatures
                _compiled_signatures.add(record.signature_key)
            unique_signatures = len(_compiled_signatures)
        _emit(
            {
                "event": "phase",
                "phase": name,
                "seconds": seconds,
                "synchronized": synchronized,
                "signature": signature,
                "compile_status": (
                    "recompile"
                    if repeated
                    else "new_signature"
                    if record.compile_ids
                    else "none"
                ),
                "compile_seconds": record.compile_seconds,
                "compile_ids": record.compile_ids,
                "compile_triggers": record.compile_triggers,
                "guard_failures": record.guard_failures,
                "unique_compile_signatures": unique_signatures,
                "outcome": "error" if error is not None else "ok",
                "error_type": type(error).__name__ if error is not None else None,
            },
            warning=bool(record.compile_ids) or seconds >= _SLOW_PHASE_SECONDS,
        )
