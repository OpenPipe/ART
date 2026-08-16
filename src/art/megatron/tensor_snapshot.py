from __future__ import annotations

from threading import Lock
import time
from typing import Any, Generic, NamedTuple, TypeVar

from pydantic import BaseModel, ConfigDict
import torch

_T = TypeVar("_T")


class _CudaFence(NamedTuple):
    device: int
    event: torch.cuda.Event


class SnapshotStageTimings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_contiguous_s: float = 0.0
    pinned_allocate_s: float = 0.0
    copy_launch_s: float = 0.0
    fence_launch_s: float = 0.0
    tensor_count: int = 0
    tensor_bytes: int = 0


class PendingCpuSnapshot(Generic[_T]):
    def __init__(
        self,
        payload: _T,
        fences: tuple[_CudaFence, ...],
        sources: tuple[torch.Tensor, ...],
    ) -> None:
        self.payload = payload
        self.fences = fences
        self._sources = sources

    def resolve(self) -> _T:
        for fence in self.fences:
            fence.event.synchronize()
        self._sources = ()
        return self.payload


class PinnedCpuSnapshotBuilder:
    def __init__(self, stager: "PinnedCpuSnapshotStager") -> None:
        self._stager = stager
        self._devices: set[int] = set()
        self._sources: list[torch.Tensor] = []
        self.timings = SnapshotStageTimings()

    def stage(self, tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach()
        self.timings.tensor_count += 1
        self.timings.tensor_bytes += source.numel() * source.element_size()
        if not source.is_cuda:
            return source.to(device="cpu", copy=True)
        contiguous_started = time.perf_counter()
        source = source.contiguous()
        self.timings.source_contiguous_s += time.perf_counter() - contiguous_started
        device = source.device.index
        if device is None:
            raise RuntimeError("CUDA snapshot tensor has no device index")
        stream = self._stager.stream(device)
        allocate_started = time.perf_counter()
        target = torch.empty_like(source, device="cpu", pin_memory=True)
        self.timings.pinned_allocate_s += time.perf_counter() - allocate_started
        copy_started = time.perf_counter()
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            target.copy_(source, non_blocking=True)
            source.record_stream(stream)
        self.timings.copy_launch_s += time.perf_counter() - copy_started
        self._devices.add(device)
        self._sources.append(source)
        return target

    def finish(self, payload: _T) -> PendingCpuSnapshot[_T]:
        fence_started = time.perf_counter()
        fences: list[_CudaFence] = []
        for device in sorted(self._devices):
            stream = self._stager.stream(device)
            with torch.cuda.device(device), torch.cuda.stream(stream):
                event = torch.cuda.Event(blocking=True)
                event.record(stream)
            fences.append(_CudaFence(device, event))
        self.timings.fence_launch_s += time.perf_counter() - fence_started
        return PendingCpuSnapshot(payload, tuple(fences), tuple(self._sources))


class PinnedCpuSnapshotStager:
    def __init__(self) -> None:
        self._streams: dict[int, torch.cuda.Stream] = {}

    def stream(self, device: int) -> torch.cuda.Stream:
        stream = self._streams.get(device)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
            self._streams[device] = stream
        return stream

    def begin(self) -> PinnedCpuSnapshotBuilder:
        return PinnedCpuSnapshotBuilder(self)


class SnapshotReadBarrier:
    """Lets forward/backward overlap snapshots while fencing optimizer mutation."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._fences: list[_CudaFence] = []

    def register(self, snapshot: PendingCpuSnapshot[Any]) -> None:
        with self._lock:
            self._fences.extend(snapshot.fences)

    def wait_before_mutation(self) -> None:
        for fence in self._take():
            torch.cuda.current_stream(fence.device).wait_event(fence.event)

    def synchronize(self) -> None:
        for fence in self._take():
            fence.event.synchronize()

    def _take(self) -> tuple[_CudaFence, ...]:
        with self._lock:
            fences = tuple(self._fences)
            self._fences.clear()
        return fences
