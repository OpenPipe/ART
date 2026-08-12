from __future__ import annotations

from threading import Lock
from typing import Any, Generic, NamedTuple, TypeVar

import torch

_T = TypeVar("_T")


class _CudaFence(NamedTuple):
    device: int
    event: torch.cuda.Event


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

    def stage(self, tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach()
        if not source.is_cuda:
            return source.to(device="cpu", copy=True)
        source = source.contiguous()
        device = source.device.index
        if device is None:
            raise RuntimeError("CUDA snapshot tensor has no device index")
        stream = self._stager.stream(device)
        target = self._stager.target_like(source)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            target.copy_(source, non_blocking=True)
            source.record_stream(stream)
        self._devices.add(device)
        self._sources.append(source)
        return target

    def finish(self, payload: _T) -> PendingCpuSnapshot[_T]:
        fences: list[_CudaFence] = []
        for device in sorted(self._devices):
            stream = self._stager.stream(device)
            with torch.cuda.device(device), torch.cuda.stream(stream):
                event = torch.cuda.Event(blocking=True)
                event.record(stream)
            fences.append(_CudaFence(device, event))
        return PendingCpuSnapshot(payload, tuple(fences), tuple(self._sources))


class PinnedCpuSnapshotStager:
    def __init__(self, *, reusable: bool = False) -> None:
        self._streams: dict[int, torch.cuda.Stream] = {}
        self._buffers: list[torch.Tensor] | None = [] if reusable else None
        self._next_buffer = 0

    def stream(self, device: int) -> torch.cuda.Stream:
        stream = self._streams.get(device)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
            self._streams[device] = stream
        return stream

    def reset(self) -> None:
        self._next_buffer = 0

    def target_like(self, source: torch.Tensor) -> torch.Tensor:
        if self._buffers is None:
            return torch.empty_like(source, device="cpu", pin_memory=True)
        index = self._next_buffer
        self._next_buffer += 1
        required = source.nbytes
        if index == len(self._buffers):
            self._buffers.append(
                torch.empty(required, dtype=torch.uint8, device="cpu", pin_memory=True)
            )
        elif self._buffers[index].numel() < required:
            self._buffers[index] = torch.empty(
                required, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        return self._buffers[index][:required].view(source.dtype).view(source.shape)

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
