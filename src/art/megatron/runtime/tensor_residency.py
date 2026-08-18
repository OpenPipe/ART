from __future__ import annotations

from collections.abc import Iterable
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import torch

from ..tensor_snapshot import PendingCpuSnapshot, PinnedCpuSnapshotStager


class TensorResidencyStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    storage_count: int = Field(ge=0)
    tensor_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)


class _TensorView(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    tensor: torch.Tensor
    dtype: torch.dtype
    storage_offset: int = Field(ge=0)
    shape: tuple[int, ...]
    stride: tuple[int, ...]


class _StorageGroup(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source: torch.Tensor
    views: tuple[_TensorView, ...]


class _CudaFence(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    device: int = Field(ge=0)
    event: torch.cuda.Event


class TensorResidencyTransition:
    def __init__(
        self,
        *,
        stats: TensorResidencyStats,
        fences: tuple[_CudaFence, ...],
        sources: tuple[torch.Tensor, ...],
    ) -> None:
        self.stats = stats
        self._fences = fences
        self._sources = sources
        self._lock = Lock()
        self._resolved = False

    def wait_on_current_stream(self) -> None:
        with self._lock:
            if self._resolved:
                return
            for fence in self._fences:
                torch.cuda.current_stream(fence.device).wait_event(fence.event)
            self._sources = ()
            self._resolved = True

    def synchronize(self) -> None:
        with self._lock:
            if self._resolved:
                return
            for fence in self._fences:
                fence.event.synchronize()
            self._sources = ()
            self._resolved = True

    @property
    def ready(self) -> bool:
        return self._resolved or all(fence.event.query() for fence in self._fences)


class TensorResidencySnapshot:
    """A lossless pinned L2 image that can replace its live tensor bindings."""

    def __init__(
        self,
        *,
        groups: tuple[_StorageGroup, ...],
        targets: tuple[torch.Tensor, ...],
        pending: PendingCpuSnapshot[None],
    ) -> None:
        self._groups = groups
        self._targets = targets
        self._pending = pending
        self._lock = Lock()
        self._resolved = False

    @property
    def stats(self) -> TensorResidencyStats:
        return TensorResidencyStats(
            storage_count=len(self._groups),
            tensor_count=sum(len(group.views) for group in self._groups),
            byte_count=sum(target.numel() for target in self._targets),
        )

    @property
    def pending(self) -> PendingCpuSnapshot[None]:
        return self._pending

    @property
    def ready(self) -> bool:
        return self._resolved or all(
            fence.event.query() for fence in self._pending.fences
        )

    def resolve(self) -> None:
        with self._lock:
            if self._resolved:
                return
            self._pending.resolve()
            self._resolved = True

    def activate(self) -> None:
        self.resolve()
        with torch.no_grad():
            for group, target in zip(self._groups, self._targets, strict=True):
                storage = target.untyped_storage()
                for view in group.views:
                    replacement = torch.empty(0, dtype=view.dtype, device="cpu")
                    replacement.set_(
                        storage,
                        view.storage_offset,
                        view.shape,
                        view.stride,
                    )
                    view.tensor.data = replacement


class TensorResidencyMover:
    """Move aliased tensor storages without changing owning Python objects."""

    def __init__(self) -> None:
        self._streams: dict[int, torch.cuda.Stream] = {}

    def byte_count(self, tensors: Iterable[torch.Tensor], device_type: str) -> int:
        return sum(
            group.source.numel()
            for group in self._groups(tensors)
            if group.source.device.type == device_type
        )

    def snapshot(
        self,
        tensors: Iterable[torch.Tensor],
    ) -> TensorResidencySnapshot:
        groups = self._groups(tensors)
        if not groups or any(group.source.device.type != "cuda" for group in groups):
            raise RuntimeError("L2 snapshots require non-empty GPU residency")
        stager = PinnedCpuSnapshotStager()
        builder = stager.begin()
        targets = tuple(builder.stage(group.source) for group in groups)
        return TensorResidencySnapshot(
            groups=groups,
            targets=targets,
            pending=builder.finish(None),
        )

    def move(
        self,
        tensors: Iterable[torch.Tensor],
        target: torch.device | str,
    ) -> TensorResidencyTransition:
        target = self._normalized_device(torch.device(target))
        groups = tuple(
            group
            for group in self._groups(tensors)
            if self._normalized_device(group.source.device) != target
        )
        if not groups:
            return TensorResidencyTransition(
                stats=TensorResidencyStats(
                    storage_count=0, tensor_count=0, byte_count=0
                ),
                fences=(),
                sources=(),
            )
        cuda_devices = {
            int(device.index)
            for group in groups
            for device in (group.source.device, target)
            if device.type == "cuda" and device.index is not None
        }
        if len(cuda_devices) != 1:
            raise RuntimeError(
                "one tensor residency move must use exactly one CUDA device"
            )
        if target.type == "cuda" and any(
            group.source.device.type == "cpu" and not group.source.is_pinned()
            for group in groups
        ):
            raise RuntimeError("L2 CPU residency must use pinned memory")
        device = cuda_devices.pop()
        stream = self._stream(device)
        sources: list[torch.Tensor] = []
        with torch.cuda.device(device), torch.cuda.stream(stream), torch.no_grad():
            if any(group.source.device.type == "cuda" for group in groups):
                stream.wait_stream(torch.cuda.current_stream(device))
            for group in groups:
                destination = torch.empty(
                    group.source.numel(),
                    dtype=torch.uint8,
                    device=target,
                    pin_memory=target.type == "cpu",
                )
                destination.copy_(group.source, non_blocking=True)
                if group.source.device.type == "cuda":
                    group.source.record_stream(stream)
                sources.append(group.source)
                storage = destination.untyped_storage()
                for view in group.views:
                    replacement = torch.empty(0, dtype=view.dtype, device=target)
                    replacement.set_(
                        storage,
                        view.storage_offset,
                        view.shape,
                        view.stride,
                    )
                    view.tensor.data = replacement
            event = torch.cuda.Event(blocking=True)
            event.record(stream)
        return TensorResidencyTransition(
            stats=TensorResidencyStats(
                storage_count=len(groups),
                tensor_count=sum(len(group.views) for group in groups),
                byte_count=sum(group.source.numel() for group in groups),
            ),
            fences=(_CudaFence(device=device, event=event),),
            sources=tuple(sources),
        )

    def _stream(self, device: int) -> torch.cuda.Stream:
        stream = self._streams.get(device)
        if stream is None:
            with torch.cuda.device(device):
                stream = torch.cuda.Stream()
            self._streams[device] = stream
        return stream

    @staticmethod
    def _normalized_device(device: torch.device) -> torch.device:
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    @staticmethod
    def _groups(tensors: Iterable[torch.Tensor]) -> tuple[_StorageGroup, ...]:
        grouped: dict[tuple[str, int | None, int, int], list[_TensorView]] = {}
        sources: dict[tuple[str, int | None, int, int], torch.Tensor] = {}
        seen: set[int] = set()
        for tensor in tensors:
            if id(tensor) in seen or tensor.numel() == 0:
                continue
            seen.add(id(tensor))
            storage = tensor.untyped_storage()
            byte_count = storage.nbytes()
            key = (
                tensor.device.type,
                tensor.device.index,
                storage.data_ptr(),
                byte_count,
            )
            source = sources.get(key)
            if source is None:
                source = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                source.set_(storage, 0, (byte_count,), (1,))
                sources[key] = source
                grouped[key] = []
            grouped[key].append(
                _TensorView(
                    tensor=tensor,
                    dtype=tensor.dtype,
                    storage_offset=int(tensor.storage_offset()),
                    shape=tuple(tensor.shape),
                    stride=tuple(tensor.stride()),
                )
            )
        return tuple(
            _StorageGroup(source=sources[key], views=tuple(views))
            for key, views in grouped.items()
        )


def nested_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, dict):
        return tuple(
            tensor
            for key, item in value.items()
            for tensor in (*nested_tensors(key), *nested_tensors(item))
        )
    if isinstance(value, (list, tuple)):
        return tuple(tensor for item in value for tensor in nested_tensors(item))
    return ()
