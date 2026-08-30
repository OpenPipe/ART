from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from threading import Lock
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from .residency import ResidencyKey
from .tensor_residency import HostTensorImage, _StorageGroup, _TensorView

_FORMAT = "art_tensor_residency_v3"
_DATA_FILE = "state.bin"
_MANIFEST_FILE = "manifest.json"


class NvmeResidencyStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str = Field(min_length=1)
    shared_storage_mount: str | None = Field(default=None, min_length=1)
    runtime_free_floor_bytes: int = Field(default=8 << 30, ge=0)
    alignment_bytes: int = Field(default=4096, ge=1)
    io_chunk_bytes: int = Field(default=16 << 20, ge=1 << 20)


class NvmeStorageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    offset: int = Field(ge=0)
    byte_count: int = Field(ge=1)


class NvmeTensorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tensor_index: int = Field(ge=0)
    storage_index: int = Field(ge=0)
    dtype: str = Field(min_length=1)
    storage_offset: int = Field(ge=0)
    shape: tuple[int, ...]
    stride: tuple[int, ...]


class NvmeResidencyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["art_tensor_residency_v3"] = _FORMAT
    key: ResidencyKey
    input_tensor_count: int = Field(ge=1)
    payload_bytes: int = Field(ge=1)
    data_bytes: int = Field(ge=1)
    layout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    storages: tuple[NvmeStorageRecord, ...]
    tensors: tuple[NvmeTensorRecord, ...]

    @model_validator(mode="after")
    def _validate_layout(self) -> "NvmeResidencyManifest":
        if not self.storages or not self.tensors:
            raise ValueError("L3 residency manifest cannot be empty")
        if sum(item.byte_count for item in self.storages) != self.payload_bytes:
            raise ValueError("L3 residency payload byte count is inconsistent")
        cursor = 0
        for item in self.storages:
            if item.offset < cursor:
                raise ValueError("L3 residency storages overlap")
            cursor = item.offset + item.byte_count
        if cursor != self.data_bytes:
            raise ValueError("L3 residency data byte count is inconsistent")
        indices = {item.tensor_index for item in self.tensors}
        if max(indices) >= self.input_tensor_count or len(indices) != len(self.tensors):
            raise ValueError("L3 residency tensor indices are invalid")
        if any(item.storage_index >= len(self.storages) for item in self.tensors):
            raise ValueError("L3 residency storage index is invalid")
        return self

    @property
    def physical_bytes(self) -> int:
        return self.data_bytes + len(self.model_dump_json().encode())


class NvmeResidencyWritePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: NvmeResidencyManifest
    manifest_bytes: bytes = Field(min_length=1)

    @property
    def physical_bytes(self) -> int:
        return self.manifest.data_bytes + len(self.manifest_bytes)


class NvmeResidencyStore:
    """Atomic, layout-verified L3 storage for one rank's live tensor images."""

    def __init__(self, config: NvmeResidencyStoreConfig) -> None:
        self.config = config
        self.root = Path(config.root).resolve()
        mount = Path(config.shared_storage_mount or self.root.parent).resolve()
        mount.mkdir(parents=True, exist_ok=True)
        self._mount = mount
        if not self.root.is_relative_to(mount):
            raise ValueError("L3 residency root leaves shared disk admission")
        self.root.mkdir(parents=True, exist_ok=True)
        self._capacity_lock_path = mount / ".art_residency_capacity.lock"
        self._lock = Lock()

    def write(
        self,
        key: ResidencyKey,
        image: HostTensorImage,
    ) -> NvmeResidencyManifest:
        return self.write_prepared(self.prepare_write(key, image), image)

    def prepare_write(
        self,
        key: ResidencyKey,
        image: HostTensorImage,
    ) -> NvmeResidencyWritePlan:
        resolve = getattr(image, "resolve", None)
        if callable(resolve):
            resolve()
        manifest = self._manifest(key, image)
        return NvmeResidencyWritePlan(
            manifest=manifest,
            manifest_bytes=manifest.model_dump_json().encode(),
        )

    def write_prepared(
        self,
        plan: NvmeResidencyWritePlan,
        image: HostTensorImage,
    ) -> NvmeResidencyManifest:
        manifest = plan.manifest
        key = manifest.key
        destination = self.path(key)
        with self._lock:
            if destination.exists():
                existing = self._read_manifest(destination)
                if (
                    existing.key != key
                    or existing.layout_sha256 != manifest.layout_sha256
                ):
                    raise RuntimeError(
                        "L3 residency key changed immutable tensor layout"
                    )
                self._verify_data_size(destination, existing)
                return existing
            temporary = self.root / f".{destination.name}.tmp-{uuid.uuid4().hex}"
            try:
                self._reserve(temporary, plan)
                committed = self._write_image(temporary, plan, image)
                os.rename(temporary, destination)
                self._fsync_dir(self.root)
                return committed
            except BaseException as error:
                failures = [error]
                try:
                    if temporary.exists():
                        shutil.rmtree(temporary)
                except BaseException as cleanup:
                    failures.append(cleanup)
                if len(failures) > 1:
                    raise BaseExceptionGroup(
                        "L3 residency write cleanup failed", failures
                    ) from None
                raise

    def physical_bytes(self, key: ResidencyKey, image: HostTensorImage) -> int:
        return self.prepare_write(key, image).physical_bytes

    def load(
        self,
        key: ResidencyKey,
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[HostTensorImage, NvmeResidencyManifest]:
        destination = self.path(key)
        manifest = self._read_manifest(destination)
        if manifest.key != key:
            raise RuntimeError("L3 residency manifest changed identity")
        self._verify_data_size(destination, manifest)
        return self._map_image(destination, manifest, tensors), manifest

    def map_committed(
        self,
        key: ResidencyKey,
        manifest: NvmeResidencyManifest,
        tensors: tuple[torch.Tensor, ...],
    ) -> HostTensorImage:
        destination = self.path(key)
        if self._read_manifest(destination) != manifest:
            raise RuntimeError("committed L3 residency manifest changed identity")
        self._verify_data_size(destination, manifest)
        return self._map_image(destination, manifest, tensors)

    def map_newly_committed(
        self,
        plan: NvmeResidencyWritePlan,
        tensors: tuple[torch.Tensor, ...],
    ) -> HostTensorImage:
        destination = self.path(plan.manifest.key)
        self._verify_data_size(destination, plan.manifest)
        if (destination / _MANIFEST_FILE).stat().st_size != len(plan.manifest_bytes):
            raise RuntimeError("newly committed L3 manifest changed size")
        return self._map_image(destination, plan.manifest, tensors)

    def read_committed(
        self,
        key: ResidencyKey,
        manifest: NvmeResidencyManifest,
        tensors: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, ...],
    ) -> HostTensorImage:
        """Read L3 into caller-owned L1 transfer staging or durable L2 buffers."""
        destination = self.path(key)
        if self._read_manifest(destination) != manifest:
            raise RuntimeError("committed L3 residency manifest changed identity")
        self._verify_data_size(destination, manifest)
        self._read_targets(destination, manifest, targets)
        return self._image_from_targets(manifest, tensors, targets)

    def _map_image(
        self,
        destination: Path,
        manifest: NvmeResidencyManifest,
        tensors: tuple[torch.Tensor, ...],
    ) -> HostTensorImage:
        if len(tensors) != manifest.input_tensor_count:
            raise RuntimeError("L3 residency tensor count changed")
        mapped = torch.from_file(
            str(destination / _DATA_FILE),
            shared=False,
            size=manifest.data_bytes,
            dtype=torch.uint8,
        )
        targets = tuple(
            mapped[item.offset : item.offset + item.byte_count]
            for item in manifest.storages
        )
        return self._image_from_targets(manifest, tensors, targets)

    @staticmethod
    def _image_from_targets(
        manifest: NvmeResidencyManifest,
        tensors: tuple[torch.Tensor, ...],
        targets: tuple[torch.Tensor, ...],
    ) -> HostTensorImage:
        if len(targets) != len(manifest.storages) or any(
            target.device.type != "cpu"
            or target.dtype != torch.uint8
            or target.numel() != record.byte_count
            for target, record in zip(targets, manifest.storages, strict=True)
        ):
            raise RuntimeError("L3 restore targets do not match committed storages")
        grouped: list[list[_TensorView]] = [[] for _ in manifest.storages]
        for item in manifest.tensors:
            tensor = tensors[item.tensor_index]
            if (
                str(tensor.dtype) != item.dtype
                or tuple(tensor.shape) != item.shape
                or tuple(tensor.stride()) != item.stride
            ):
                raise RuntimeError("L3 residency tensor layout changed")
            grouped[item.storage_index].append(
                _TensorView(
                    tensor=tensor,
                    tensor_index=item.tensor_index,
                    dtype=tensor.dtype,
                    storage_offset=item.storage_offset,
                    shape=item.shape,
                    stride=item.stride,
                )
            )
        groups = tuple(
            _StorageGroup(source=target, views=tuple(views))
            for target, views in zip(targets, grouped, strict=True)
        )
        return HostTensorImage(
            groups=groups,
            targets=targets,
            input_tensor_count=manifest.input_tensor_count,
        )

    def delete(self, key: ResidencyKey) -> int:
        destination = self.path(key)
        manifest = self._read_manifest(destination)
        if manifest.key != key:
            raise RuntimeError("refusing to delete a changed L3 residency identity")
        reclaimed = sum(
            item.stat().st_size for item in destination.iterdir() if item.is_file()
        )
        for name in (_MANIFEST_FILE, _DATA_FILE):
            (destination / name).unlink()
        destination.rmdir()
        self._fsync_dir(self.root)
        return reclaimed

    def path(self, key: ResidencyKey) -> Path:
        digest = hashlib.sha256(key.model_dump_json().encode()).hexdigest()
        return self.root / digest

    def _manifest(
        self, key: ResidencyKey, image: HostTensorImage
    ) -> NvmeResidencyManifest:
        groups = image.groups()
        targets = image.storage_bytes()
        if not groups or len(groups) != len(targets):
            raise RuntimeError("cannot persist an empty L3 tensor image")
        offsets: list[int] = []
        cursor = 0
        for target in targets:
            if target.device.type != "cpu" or target.dtype != torch.uint8:
                raise RuntimeError("L3 tensor images must expose CPU byte storages")
            cursor = self._aligned(cursor)
            offsets.append(cursor)
            cursor += target.numel()
        tensors = tuple(
            NvmeTensorRecord(
                tensor_index=view.tensor_index,
                storage_index=storage_index,
                dtype=str(view.dtype),
                storage_offset=view.storage_offset,
                shape=view.shape,
                stride=view.stride,
            )
            for storage_index, group in enumerate(groups)
            for view in group.views
        )
        input_count = image.input_tensor_count
        layout = {
            "input_tensor_count": input_count,
            "storages": [
                {"offset": offset, "byte_count": target.numel()}
                for offset, target in zip(offsets, targets, strict=True)
            ],
            "tensors": [item.model_dump(mode="json") for item in tensors],
        }
        return NvmeResidencyManifest(
            key=key,
            input_tensor_count=input_count,
            payload_bytes=sum(target.numel() for target in targets),
            data_bytes=cursor,
            layout_sha256=self._digest_json(layout),
            storages=tuple(
                NvmeStorageRecord(offset=offset, byte_count=target.numel())
                for offset, target in zip(offsets, targets, strict=True)
            ),
            tensors=tensors,
        )

    def _write_image(
        self,
        directory: Path,
        plan: NvmeResidencyWritePlan,
        image: HostTensorImage,
    ) -> NvmeResidencyManifest:
        manifest = plan.manifest
        with (directory / _DATA_FILE).open("r+b", buffering=0) as handle:
            cursor = 0
            for record, target in zip(
                manifest.storages, image.storage_bytes(), strict=True
            ):
                if padding := record.offset - cursor:
                    handle.write(bytes(padding))
                    cursor += padding
                view = memoryview(target.numpy()).cast("B")
                for start in range(0, len(view), self.config.io_chunk_bytes):
                    chunk = view[start : start + self.config.io_chunk_bytes]
                    handle.write(chunk)
                    cursor += len(chunk)
                if cursor != record.offset + record.byte_count:
                    raise RuntimeError("L3 residency write changed physical layout")
            handle.flush()
            os.fsync(handle.fileno())
        payload = plan.manifest_bytes
        with (directory / _MANIFEST_FILE).open("wb", buffering=0) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_dir(directory)
        return manifest

    def _reserve(self, temporary: Path, plan: NvmeResidencyWritePlan) -> None:
        """Reserve physical blocks under one mount-wide capacity lock."""

        descriptor = os.open(self._capacity_lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            free = shutil.disk_usage(self._mount).free
            if free - plan.physical_bytes < self.config.runtime_free_floor_bytes:
                raise RuntimeError(
                    "L3 residency would cross the runtime free-space floor: "
                    f"free={free}, incoming={plan.physical_bytes}, "
                    f"floor={self.config.runtime_free_floor_bytes}"
                )
            temporary.mkdir()
            data = os.open(temporary / _DATA_FILE, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                os.posix_fallocate(data, 0, plan.manifest.data_bytes)
            finally:
                os.close(data)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _read_targets(
        self,
        directory: Path,
        manifest: NvmeResidencyManifest,
        targets: tuple[torch.Tensor, ...],
    ) -> None:
        with (directory / _DATA_FILE).open("rb", buffering=0) as handle:
            for record, target in zip(manifest.storages, targets, strict=True):
                handle.seek(record.offset)
                view = memoryview(target.numpy()).cast("B")
                cursor = 0
                while cursor < len(view):
                    chunk = view[cursor : cursor + self.config.io_chunk_bytes]
                    count = handle.readinto(chunk)
                    if count != len(chunk):
                        raise RuntimeError("L3 residency data ended during restore")
                    cursor += count

    @staticmethod
    def _verify_data_size(directory: Path, manifest: NvmeResidencyManifest) -> None:
        if (directory / _DATA_FILE).stat().st_size != manifest.data_bytes:
            raise RuntimeError("L3 residency data size changed")

    def _read_manifest(self, directory: Path) -> NvmeResidencyManifest:
        try:
            return NvmeResidencyManifest.model_validate_json(
                (directory / _MANIFEST_FILE).read_bytes()
            )
        except FileNotFoundError as exc:
            raise RuntimeError("L3 residency image is not committed") from exc

    def _aligned(self, value: int) -> int:
        alignment = self.config.alignment_bytes
        return (value + alignment - 1) // alignment * alignment

    @staticmethod
    def _digest_json(value: object) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
