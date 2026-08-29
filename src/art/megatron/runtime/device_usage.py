from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import torch

_T = TypeVar("_T")


class DeviceTimedCommandError(RuntimeError):
    """Command failure with exact rank-local GPU service when recoverable."""

    def __init__(self, source: BaseException, gpu_service_ns: int | None) -> None:
        super().__init__(str(source))
        self.source = source
        self.gpu_service_ns = gpu_service_ns


def measure_cuda_call(call: Callable[[], _T]) -> tuple[_T, int]:
    """Measure one rank's device interval without using host-wall time."""

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        result = call()
    except BaseException as source:
        try:
            gpu_service_ns = _finish(start, end)
        except BaseException as timing_error:
            source.add_note(
                "CUDA command timing also failed: "
                f"{type(timing_error).__name__}: {timing_error}"
            )
            gpu_service_ns = None
        raise DeviceTimedCommandError(source, gpu_service_ns) from source
    try:
        gpu_service_ns = _finish(start, end)
    except BaseException as timing_error:
        raise DeviceTimedCommandError(timing_error, None) from timing_error
    return result, gpu_service_ns


def _finish(start: torch.cuda.Event, end: torch.cuda.Event) -> int:
    end.record()
    end.synchronize()
    return max(0, round(start.elapsed_time(end) * 1_000_000))
