from __future__ import annotations

import faulthandler
import json
import os
import sys
from threading import Event, Thread, current_thread
import time
from typing import Any

_TRACE_ENV = "ART_MEGATRON_PRESCHEDULE_TRACE"
_WATCHDOG_ENV = "ART_MEGATRON_PRESCHEDULE_WATCHDOG_S"


def trace_preschedule(rank: int, operation_id: str, stage: str, **fields: Any) -> None:
    if os.environ.get(_TRACE_ENV) != "1":
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
