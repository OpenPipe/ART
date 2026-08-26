from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import faulthandler
import json
import os
import sys
from threading import Event, Thread, current_thread
import time
from typing import Any, Iterator

_TRACE_ENV = "ART_MEGATRON_PRESCHEDULE_TRACE"
_WATCHDOG_ENV = "ART_MEGATRON_PRESCHEDULE_WATCHDOG_S"
_TRACE_CONTEXT: ContextVar[tuple[int, str] | None] = ContextVar(
    "art_megatron_preschedule_trace_context", default=None
)


def preschedule_trace_enabled() -> bool:
    return os.environ.get(_TRACE_ENV) == "1"


def trace_preschedule(rank: int, operation_id: str, stage: str, **fields: Any) -> None:
    if not preschedule_trace_enabled():
        return
    print(
        "ART_PRESCHEDULE "
        + json.dumps(
            {
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "thread": current_thread().name,
                "rank": int(rank),
                "operation_id": operation_id,
                "stage": stage,
                **fields,
            },
            default=str,
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


@contextmanager
def preschedule_trace_scope(rank: int, operation_id: str) -> Iterator[None]:
    token = _TRACE_CONTEXT.set((int(rank), operation_id))
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)


def trace_current_preschedule(stage: str, **fields: Any) -> None:
    context = _TRACE_CONTEXT.get()
    if context is not None:
        trace_preschedule(*context, stage, **fields)


def cuda_stream_fields(device: int | None = None) -> dict[str, int]:
    if not preschedule_trace_enabled():
        return {}
    import torch

    if not torch.cuda.is_available():
        return {}
    resolved_device = torch.cuda.current_device() if device is None else int(device)
    stream = torch.cuda.current_stream(resolved_device)
    return {
        "cuda_device": resolved_device,
        "cuda_stream_object_id": id(stream),
        "cuda_stream_handle": int(stream.cuda_stream),
    }


def start_preschedule_watchdog(rank: int, operation_id: str) -> Event | None:
    if os.environ.get(_TRACE_ENV) != "1":
        return None
    stop = Event()
    timeout_s = float(os.environ.get(_WATCHDOG_ENV, "20"))

    def watch() -> None:
        if stop.wait(timeout_s):
            return
        trace_preschedule(rank, operation_id, "watchdog_timeout", timeout_s=timeout_s)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    Thread(
        target=watch,
        name=f"art-preschedule-watchdog-rank-{rank}",
        daemon=True,
    ).start()
    return stop


def stop_preschedule_watchdog(stop: Event | None) -> None:
    if stop is not None:
        stop.set()
