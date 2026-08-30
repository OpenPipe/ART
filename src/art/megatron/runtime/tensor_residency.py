from __future__ import annotations

from collections.abc import Iterable
from threading import Condition, Lock
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
    tensor_index: int = Field(ge=0)
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
        staging: "_PinnedStagerLease | None" = None,
    ) -> None:
        self.stats = stats
        self._fences = fences
        self._sources = sources
        self._staging = staging
        self._lock = Lock()
        self._resolved = False

    def wait_on_current_stream(self) -> None:
        with self._lock:
            if self._resolved:
                return
            for fence in self._fences:
                torch.cuda.current_stream(fence.device).wait_event(fence.event)
            if self._staging is not None:
                self._staging.release_after(self._fences)
                self._staging = None
            self._sources = ()
            self._resolved = True

    def synchronize(self) -> None:
        with self._lock:
            if self._resolved:
                return
            for fence in self._fences:
                fence.event.synchronize()
            if self._staging is not None:
                self._staging.release()
                self._staging = None
            self._sources = ()
            self._resolved = True

    @property
    def ready(self) -> bool:
        return self._resolved or all(fence.event.query() for fence in self._fences)


class HostTensorImage:
    def __init__(
        self,
        *,
        groups: tuple[_StorageGroup, ...],
        targets: tuple[torch.Tensor, ...],
        input_tensor_count: int,
    ) -> None:
        if len(groups) != len(targets):
            raise RuntimeError("tensor residency groups and targets differ")
        self._groups = tuple(
            group.model_copy(update={"source": target})
            for group, target in zip(groups, targets, strict=True)
        )
        self._targets = targets
        self.input_tensor_count = input_tensor_count

    @property
    def stats(self) -> TensorResidencyStats:
        return TensorResidencyStats(
            storage_count=len(self._groups),
            tensor_count=sum(len(group.views) for group in self._groups),
            byte_count=sum(target.numel() for target in self._targets),
        )

    def activate(self) -> None:
        with torch.no_grad():
            for view, replacement in self.tensor_views():
                view.tensor.data = replacement

    def tensor_views(self) -> tuple[tuple[_TensorView, torch.Tensor], ...]:
        replacements: list[tuple[_TensorView, torch.Tensor]] = []
        for group, target in zip(self._groups, self._targets, strict=True):
            storage = target.untyped_storage()
            for view in group.views:
                base_bytes = int(target.storage_offset())
                element_size = view.tensor.element_size()
                if base_bytes % element_size:
                    raise RuntimeError("host storage offset is not dtype aligned")
                replacement = torch.empty(0, dtype=view.dtype, device="cpu")
                replacement.set_(
                    storage,
                    base_bytes // element_size + view.storage_offset,
                    view.shape,
                    view.stride,
                )
                replacements.append((view, replacement))
        return tuple(replacements)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        tensors: list[torch.Tensor | None] = [None] * self.input_tensor_count
        for view, replacement in self.tensor_views():
            tensors[view.tensor_index] = replacement
        if any(tensor is None for tensor in tensors):
            raise RuntimeError("host image does not cover every input tensor")
        return tuple(tensor for tensor in tensors if tensor is not None)

    def copy(self, *, pin_memory: bool = False) -> "HostTensorImage":
        targets = tuple(
            torch.empty_like(target, device="cpu", pin_memory=pin_memory).copy_(target)
            for target in self._targets
        )
        return HostTensorImage(
            groups=self._groups,
            targets=targets,
            input_tensor_count=self.input_tensor_count,
        )

    def storage_bytes(self) -> tuple[torch.Tensor, ...]:
        return self._targets

    def groups(self) -> tuple[_StorageGroup, ...]:
        return self._groups


class TensorResidencySnapshot(HostTensorImage):
    """A lossless pageable L2 image that can replace its live tensor bindings."""

    def __init__(
        self,
        *,
        groups: tuple[_StorageGroup, ...],
        targets: tuple[torch.Tensor, ...],
        pending: PendingCpuSnapshot[None],
        staging: "_PinnedStagerLease",
        input_tensor_count: int,
    ) -> None:
        super().__init__(
            groups=groups,
            targets=targets,
            input_tensor_count=input_tensor_count,
        )
        self._pending = pending
        self._staging = staging
        self._lock = Lock()
        self._resolved = False

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

    def materialize_l2(self) -> HostTensorImage:
        self.resolve()
        try:
            return self.copy()
        finally:
            self._staging.release()

    def activate(self) -> None:
        self.resolve()
        super().activate()


class _PinnedStagerLease:
    def __init__(
        self,
        pool: "_ReusablePinnedStagerPool",
        stager: PinnedCpuSnapshotStager,
    ) -> None:
        self._pool = pool
        self.stager = stager
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool.release(self.stager)

    def release_after(self, fences: tuple[_CudaFence, ...]) -> None:
        if self._released:
            return
        self._released = True
        self._pool.release(self.stager, fences=fences)


class _ReusablePinnedStagerPool:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("pinned staging capacity must be positive")
        self._available = [
            PinnedCpuSnapshotStager(reusable=True) for _ in range(capacity)
        ]
        self._pending: list[tuple[PinnedCpuSnapshotStager, tuple[_CudaFence, ...]]] = []
        self._condition = Condition()

    def acquire(self) -> _PinnedStagerLease:
        with self._condition:
            while not self._available:
                self._reclaim()
                if not self._available:
                    self._condition.wait(timeout=0.001)
            stager = self._available.pop()
            stager.reset()
            return _PinnedStagerLease(self, stager)

    def release(
        self,
        stager: PinnedCpuSnapshotStager,
        *,
        fences: tuple[_CudaFence, ...] = (),
    ) -> None:
        with self._condition:
            if fences and not all(fence.event.query() for fence in fences):
                self._pending.append((stager, fences))
            else:
                self._available.append(stager)
            self._condition.notify()

    def _reclaim(self) -> None:
        pending: list[tuple[PinnedCpuSnapshotStager, tuple[_CudaFence, ...]]] = []
        for stager, fences in self._pending:
            if all(fence.event.query() for fence in fences):
                self._available.append(stager)
            else:
                pending.append((stager, fences))
        self._pending = pending


class TensorResidencyMover:
    """Move aliased tensor storages without changing owning Python objects."""

    def __init__(self, *, staging_capacity: int = 2) -> None:
        self._streams: dict[int, torch.cuda.Stream] = {}
        self._stagers = _ReusablePinnedStagerPool(staging_capacity)

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
        tensors = tuple(tensors)
        groups = self._groups(tensors)
        if not groups or any(group.source.device.type != "cuda" for group in groups):
            raise RuntimeError("L2 snapshots require non-empty GPU residency")
        staging = self._stagers.acquire()
        try:
            builder = staging.stager.begin()
            targets = tuple(builder.stage(group.source) for group in groups)
            return TensorResidencySnapshot(
                groups=groups,
                targets=targets,
                pending=builder.finish(None),
                staging=staging,
                input_tensor_count=len(tensors),
            )
        except BaseException:
            staging.release()
            raise

    def host_image(self, tensors: Iterable[torch.Tensor]) -> HostTensorImage:
        tensors = tuple(tensors)
        groups = self._groups(tensors)
        if not groups or any(group.source.device.type != "cpu" for group in groups):
            raise RuntimeError("L2 admission requires non-empty CPU residency")
        targets = tuple(
            torch.empty_like(group.source, device="cpu", pin_memory=False).copy_(
                group.source
            )
            for group in groups
        )
        image = HostTensorImage(
            groups=groups,
            targets=targets,
            input_tensor_count=len(tensors),
        )
        image.activate()
        return image

    def adopt_host_image(self, tensors: Iterable[torch.Tensor]) -> HostTensorImage:
        """Adopt immutable CPU tensor storage without copying or rebinding it.

        The caller transfers exclusive ownership of the prepared tensors to the
        returned image; only residency machinery may mutate or rebind them afterward.
        """
        tensors = tuple(tensors)
        groups = self._groups(tensors)
        if not groups or any(group.source.device.type != "cpu" for group in groups):
            raise RuntimeError("L2 adoption requires non-empty CPU residency")
        return HostTensorImage(
            groups=groups,
            targets=tuple(group.source for group in groups),
            input_tensor_count=len(tensors),
        )

    def move(
        self,
        tensors: Iterable[torch.Tensor],
        target: torch.device | str,
        *,
        staging: "_PinnedStagerLease | None" = None,
    ) -> TensorResidencyTransition:
        """Move tensor storage and consume any caller-owned staging lease."""
        return self._move_groups(self._groups(tensors), target, staging=staging)

    def move_host_images(
        self,
        images: Iterable[HostTensorImage],
        target: torch.device | str,
        *,
        staging: "_PinnedStagerLease | None" = None,
    ) -> TensorResidencyTransition:
        """Move the exact authenticated storage groups in host images."""
        return self._move_groups(
            tuple(group for image in images for group in image.groups()),
            target,
            staging=staging,
        )

    def _move_groups(
        self,
        groups: Iterable[_StorageGroup],
        target: torch.device | str,
        *,
        staging: "_PinnedStagerLease | None" = None,
    ) -> TensorResidencyTransition:
        owned_staging = staging
        try:
            target = self._normalized_device(torch.device(target))
            groups = tuple(
                group
                for group in groups
                if self._normalized_device(group.source.device) != target
            )
        except BaseException:
            if owned_staging is not None:
                owned_staging.release()
            raise
        if not groups:
            return TensorResidencyTransition(
                stats=TensorResidencyStats(
                    storage_count=0, tensor_count=0, byte_count=0
                ),
                fences=(),
                sources=(),
                staging=owned_staging,
            )
        cuda_devices = {
            int(device.index)
            for group in groups
            for device in (group.source.device, target)
            if device.type == "cuda" and device.index is not None
        }
        if len(cuda_devices) != 1:
            if owned_staging is not None:
                owned_staging.release()
            raise RuntimeError(
                "one tensor residency move must use exactly one CUDA device"
            )
        if target.type == "cuda" and any(
            group.source.device.type == "cpu" and not group.source.is_pinned()
            for group in groups
        ):
            if owned_staging is not None:
                owned_staging.release()
                raise RuntimeError("caller staging does not own the CPU sources")
            owned_staging = self._stagers.acquire()
            try:
                groups = tuple(
                    group.model_copy(
                        update={
                            "source": owned_staging.stager.target_like(
                                group.source
                            ).copy_(group.source)
                        }
                    )
                    for group in groups
                )
            except BaseException:
                owned_staging.release()
                raise
        device = cuda_devices.pop()
        stream: torch.cuda.Stream | None = None
        sources: list[torch.Tensor] = []
        transfer_started = False
        try:
            stream = self._stream(device)
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
                    transfer_started = True
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
        except BaseException as error:
            if owned_staging is not None:
                try:
                    if transfer_started and stream is not None:
                        with torch.cuda.device(device), torch.cuda.stream(stream):
                            fence = _CudaFence(
                                device=device, event=torch.cuda.Event(blocking=True)
                            )
                            fence.event.record(stream)
                        owned_staging.release_after((fence,))
                    else:
                        owned_staging.release()
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "tensor residency transfer and staging cleanup failed",
                        (error, cleanup_error),
                    ) from None
            raise
        return TensorResidencyTransition(
            stats=TensorResidencyStats(
                storage_count=len(groups),
                tensor_count=sum(len(group.views) for group in groups),
                byte_count=sum(group.source.numel() for group in groups),
            ),
            fences=(_CudaFence(device=device, event=event),),
            sources=tuple(sources),
            staging=owned_staging,
        )

    def staging(self) -> "_PinnedStagerLease":
        return self._stagers.acquire()

    @staticmethod
    def stage_host_image(
        image: HostTensorImage,
        staging: "_PinnedStagerLease",
    ) -> HostTensorImage:
        """Place pageable image storage in one caller-owned transfer lease."""
        targets = tuple(
            source
            if source.is_pinned()
            else staging.stager.target_like(source).copy_(source)
            for source in image.storage_bytes()
        )
        return HostTensorImage(
            groups=image.groups(),
            targets=targets,
            input_tensor_count=image.input_tensor_count,
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
        for tensor_index, tensor in enumerate(tensors):
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
                    tensor_index=tensor_index,
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
