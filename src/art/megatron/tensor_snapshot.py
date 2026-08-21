from __future__ import annotations

from collections.abc import Sequence
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
        if not source.is_cuda:
            self.timings.tensor_count += 1
            self.timings.tensor_bytes += source.nbytes
            return source.to(device="cpu", copy=True)
        return self.stage_group((tensor,))[0]

    def stage_group(self, tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if not tensors:
            return ()
        sources = tuple(tensor.detach() for tensor in tensors)
        device = sources[0].device
        dtype = sources[0].dtype
        if any(source.device != device or source.dtype != dtype for source in sources):
            raise ValueError("snapshot staging group must share one device and dtype")
        self.timings.tensor_count += len(sources)
        self.timings.tensor_bytes += sum(source.nbytes for source in sources)
        contiguous_started = time.perf_counter()
        sources = tuple(source.contiguous() for source in sources)
        self.timings.source_contiguous_s += time.perf_counter() - contiguous_started
        total_numel = sum(source.numel() for source in sources)
        if not total_numel:
            return tuple(
                torch.empty(source.shape, dtype=dtype, device="cpu")
                for source in sources
            )
        if device.type != "cuda":
            target = torch.empty(total_numel, dtype=dtype, device="cpu")
            outputs: list[torch.Tensor] = []
            offset = 0
            for source in sources:
                output = target.narrow(0, offset, source.numel()).view(source.shape)
                output.copy_(source)
                outputs.append(output)
                offset += source.numel()
            return tuple(outputs)
        device_index = device.index
        if device_index is None:
            raise RuntimeError("CUDA snapshot tensor has no device index")
        stream = self._stager.stream(device_index)
        allocate_started = time.perf_counter()
        target = (
            self._stager.target_bytes(total_numel * sources[0].element_size())
            .view(dtype)
            .view(-1)
        )
        self.timings.pinned_allocate_s += time.perf_counter() - allocate_started
        copy_started = time.perf_counter()
        stream.wait_stream(torch.cuda.current_stream(device_index))
        outputs = []
        offset = 0
        with torch.cuda.stream(stream):
            for source in sources:
                output = target.narrow(0, offset, source.numel()).view(source.shape)
                output.copy_(source, non_blocking=True)
                source.record_stream(stream)
                outputs.append(output)
                offset += source.numel()
        self.timings.copy_launch_s += time.perf_counter() - copy_started
        self._devices.add(device_index)
        self._sources.extend(sources)
        return tuple(outputs)

    def fence_current_stream(self, device: int) -> None:
        """Include caller-stream work in the pending snapshot mutation fence."""
        stream = self._stager.stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        self._devices.add(device)

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
        return self.target_bytes(source.nbytes).view(source.dtype).view(source.shape)

    def target_bytes(self, required: int) -> torch.Tensor:
        if required < 1:
            raise ValueError("pinned staging target must be non-empty")
        if self._buffers is None:
            return torch.empty(
                required, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        index = self._next_buffer
        self._next_buffer += 1
        if index == len(self._buffers):
            self._buffers.append(
                torch.empty(required, dtype=torch.uint8, device="cpu", pin_memory=True)
            )
        elif self._buffers[index].numel() < required:
            self._buffers[index] = torch.empty(
                required, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        return self._buffers[index][:required]

    def begin(self) -> PinnedCpuSnapshotBuilder:
        return PinnedCpuSnapshotBuilder(self)


class SnapshotReadBarrier:
    """Lets forward/backward overlap snapshots while fencing optimizer mutation."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._fences: dict[str | None, list[_CudaFence]] = {}

    def register(
        self, snapshot: PendingCpuSnapshot[Any], *, key: str | None = None
    ) -> None:
        with self._lock:
            self._fences.setdefault(key, []).extend(snapshot.fences)

    def wait_before_mutation(self, *, key: str | None = None) -> None:
        for fence in self._take(key):
            torch.cuda.current_stream(fence.device).wait_event(fence.event)

    def synchronize(self, *, key: str | None = None) -> None:
        for fence in self._take(key):
            fence.event.synchronize()

    def _take(self, key: str | None) -> tuple[_CudaFence, ...]:
        with self._lock:
            if key is not None:
                fences = tuple(self._fences.pop(key, ()))
            else:
                fences = tuple(
                    fence for pending in self._fences.values() for fence in pending
                )
                self._fences.clear()
        return fences
