from __future__ import annotations

from collections.abc import AsyncIterator
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import struct
from typing import BinaryIO, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.training import OperationRef
from art.vllm_route_transport import (
    RetainedRouteBundleRef,
    RouteBundleObjectRef,
)

_FRAME = struct.Struct("!I")
_MAX_RECEIPT_BYTES = 1 << 20
_SHA256 = r"^[0-9a-f]{64}$"


class _Digest(Protocol):
    def update(self, value: bytes | bytearray | memoryview, /) -> None: ...


class RouteArtifactExportReceipt(BaseModel):
    """Private source identity retained with one exported route payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    attempt_id: str = Field(pattern=_SHA256)
    tenant_id: str = Field(min_length=1, max_length=255)
    run_id: str = Field(min_length=1, max_length=255)
    operation_id: str = Field(min_length=1, max_length=255)
    source: RetainedRouteBundleRef
    access_receipt_sha256: str = Field(pattern=_SHA256)

    @classmethod
    def create(
        cls,
        *,
        attempt_id: str,
        tenant_id: str,
        run_id: str,
        operation_id: str,
        source: RetainedRouteBundleRef,
    ) -> Self:
        values = {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "tenant_id": tenant_id,
            "run_id": run_id,
            "operation_id": operation_id,
            "source": source.model_dump(mode="json"),
        }
        return cls(**values, access_receipt_sha256=_digest(values))

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        values = self.model_dump(mode="json", exclude={"access_receipt_sha256"})
        if self.access_receipt_sha256 != _digest(values):
            raise ValueError("route artifact access receipt changed identity")
        return self


class MaterializedRouteArtifact(BaseModel):
    """Attempt-owned local object accepted by ART's ordinary replay path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    export: RouteArtifactExportReceipt
    local: RetainedRouteBundleRef
    manifest_sha256: str = Field(pattern=_SHA256)

    @classmethod
    def create(
        cls,
        export: RouteArtifactExportReceipt,
        local: RetainedRouteBundleRef,
    ) -> Self:
        values = {
            "schema_version": 1,
            "export": export.model_dump(mode="json"),
            "local": local.model_dump(mode="json"),
        }
        return cls(**values, manifest_sha256=_digest(values))

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if (
            self.local.layout != self.export.source.layout
            or self.local.object.store != "holder_local"
            or self.local.object.size_bytes != self.export.source.object.size_bytes
            or self.local.object.sha256 != self.export.source.object.sha256
        ):
            raise ValueError("materialized route artifact changed source identity")
        values = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != _digest(values):
            raise ValueError("route artifact manifest changed identity")
        return self


class RouteArtifactOwnershipHandle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    handle_id: str = Field(pattern=_SHA256)
    operation_id: str = Field(min_length=1, max_length=255)
    bundle_ids: tuple[str, ...] = Field(min_length=1, max_length=4096)


async def materialize_route_artifact(
    chunks: AsyncIterator[bytes | bytearray | memoryview],
    *,
    root: str | Path,
    required_free_bytes: int = 0,
) -> MaterializedRouteArtifact:
    """Consume one framed private export without buffering its route payload."""

    if required_free_bytes < 0:
        raise ValueError("route artifact free-space floor must be nonnegative")
    iterator = aiter(chunks)
    buffered = bytearray()
    while len(buffered) < _FRAME.size:
        try:
            buffered.extend(memoryview(await anext(iterator)).cast("B"))
        except StopAsyncIteration:
            raise RuntimeError("route artifact frame is truncated") from None
    (receipt_size,) = _FRAME.unpack_from(buffered)
    if not 1 <= receipt_size <= _MAX_RECEIPT_BYTES:
        raise RuntimeError("route artifact receipt is outside bounds")
    payload_offset = _FRAME.size + receipt_size
    while len(buffered) < payload_offset:
        try:
            buffered.extend(memoryview(await anext(iterator)).cast("B"))
        except StopAsyncIteration:
            raise RuntimeError("route artifact receipt is truncated") from None
    export = RouteArtifactExportReceipt.model_validate_json(
        buffered[_FRAME.size : payload_offset]
    )
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root_path).free - export.source.object.size_bytes < (
        required_free_bytes
    ):
        raise RuntimeError("route artifact free-space floor reached")
    route_path = root_path / f"{export.access_receipt_sha256}.routes"
    temporary = root_path / (
        f".{export.access_receipt_sha256}.{secrets.token_hex(8)}.partial"
    )
    digest = hashlib.sha256()
    written = 0
    try:
        with temporary.open("xb") as output:
            initial = memoryview(buffered)[payload_offset:]
            if initial:
                written = _write_bounded(
                    output,
                    initial,
                    digest,
                    written=written,
                    maximum=export.source.object.size_bytes,
                )
            async for chunk in iterator:
                written = _write_bounded(
                    output,
                    memoryview(chunk).cast("B"),
                    digest,
                    written=written,
                    maximum=export.source.object.size_bytes,
                )
            output.flush()
            os.fsync(output.fileno())
        if (
            written != export.source.object.size_bytes
            or digest.hexdigest() != export.source.object.sha256
        ):
            raise RuntimeError("route artifact payload changed identity")
        os.chmod(temporary, 0o600)
        if route_path.exists():
            _validate_local_file(route_path, export.source.object)
        else:
            os.replace(temporary, route_path)
    finally:
        temporary.unlink(missing_ok=True)

    local = RetainedRouteBundleRef(
        object=RouteBundleObjectRef(
            store="holder_local",
            locator=str(route_path),
            size_bytes=export.source.object.size_bytes,
            sha256=export.source.object.sha256,
        ),
        layout=export.source.layout,
        lease_id=f"route-artifact-{export.access_receipt_sha256}",
    )
    manifest = MaterializedRouteArtifact.create(export, local)
    _write_exact_json(
        root_path / f"{export.access_receipt_sha256}.json",
        manifest.model_dump_json(indent=2).encode() + b"\n",
    )
    return manifest


class MaterializedRouteArtifactProvider:
    """Reader and bounded ownership adapter for deterministic offline replay."""

    retained_route_transport: Literal["holder_local"] = "holder_local"

    def __init__(self, artifacts: tuple[MaterializedRouteArtifact, ...]) -> None:
        if not artifacts or len(artifacts) > 4096:
            raise ValueError("route artifact provider requires 1-4096 artifacts")
        by_bundle = {item.local.layout.bundle_id: item for item in artifacts}
        if len(by_bundle) != len(artifacts):
            raise ValueError("route artifact provider repeats a bundle")
        self._artifacts = by_bundle
        self._active: dict[str, RouteArtifactOwnershipHandle] = {}

    async def acquire(
        self,
        *,
        operation: OperationRef,
        bundles: tuple[RetainedRouteBundleRef, ...],
    ) -> RouteArtifactOwnershipHandle:
        if not bundles:
            raise ValueError("route artifact ownership requires a bundle")
        for bundle in bundles:
            artifact = self._artifacts.get(bundle.layout.bundle_id)
            if (
                artifact is None
                or artifact.local != bundle
                or artifact.export.run_id != operation.run_id
                or artifact.export.operation_id != operation.operation_id
            ):
                raise RuntimeError("route artifact belongs to another operation")
            _validate_local_file(Path(bundle.object.locator), bundle.object)
        bundle_ids = tuple(bundle.layout.bundle_id for bundle in bundles)
        handle_id = _digest(
            {
                "kind": "materialized-route-artifact-ownership-v1",
                "operation": operation.model_dump(mode="json"),
                "bundle_ids": bundle_ids,
            }
        )
        handle = RouteArtifactOwnershipHandle(
            handle_id=handle_id,
            operation_id=operation.operation_id,
            bundle_ids=bundle_ids,
        )
        prior = self._active.setdefault(handle_id, handle)
        if prior != handle:
            raise RuntimeError("route artifact ownership changed on replay")
        return handle

    async def transfer(
        self,
        handle: object,
        *,
        transfer_id: str,
        target_owner_id: str,
    ) -> object:
        del handle, transfer_id, target_owner_id
        raise RuntimeError("materialized route artifacts cannot migrate")

    async def release(self, handle: object) -> None:
        owned = RouteArtifactOwnershipHandle.model_validate(handle)
        current = self._active.get(owned.handle_id)
        if current is None:
            return
        if current != owned:
            raise RuntimeError("route artifact ownership identity changed")
        self._active.pop(owned.handle_id)

    async def read_stream(
        self, ref: RouteBundleObjectRef, *, lease_id: str
    ) -> AsyncIterator[bytes]:
        artifact = next(
            (
                item
                for item in self._artifacts.values()
                if item.local.object == ref and item.local.lease_id == lease_id
            ),
            None,
        )
        if artifact is None:
            raise RuntimeError("route artifact reader does not own this object")
        bundle_id = artifact.local.layout.bundle_id
        if not any(bundle_id in handle.bundle_ids for handle in self._active.values()):
            raise RuntimeError("route artifact read has no active ownership")
        path = Path(ref.locator)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        digest = hashlib.sha256()
        total = 0
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != ref.size_bytes:
                raise RuntimeError("route artifact changed size or type")
            while chunk := await _read_descriptor(descriptor):
                digest.update(chunk)
                total += len(chunk)
                yield chunk
        finally:
            os.close(descriptor)
        if total != ref.size_bytes or digest.hexdigest() != ref.sha256:
            raise RuntimeError("route artifact changed digest")


async def _read_descriptor(descriptor: int) -> bytes:
    import asyncio

    return await asyncio.to_thread(os.read, descriptor, 1 << 20)


def _write_bounded(
    output: BinaryIO,
    chunk: memoryview,
    digest: _Digest,
    *,
    written: int,
    maximum: int,
) -> int:
    if written + len(chunk) > maximum:
        raise RuntimeError("route artifact payload exceeds its declared size")
    output.write(chunk)
    digest.update(chunk)
    return written + len(chunk)


def _validate_local_file(path: Path, ref: RouteBundleObjectRef) -> None:
    if not path.is_absolute():
        raise RuntimeError("route artifact path must be absolute")
    resolved = path.resolve(strict=True)
    if resolved != path or resolved.suffix != ".routes":
        raise RuntimeError("route artifact path changed identity")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    total = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != ref.size_bytes:
            raise RuntimeError("route artifact changed size or type")
        while chunk := os.read(descriptor, 1 << 20):
            total += len(chunk)
            digest.update(chunk)
    finally:
        os.close(descriptor)
    if total != ref.size_bytes or digest.hexdigest() != ref.sha256:
        raise RuntimeError("route artifact changed digest")


def _write_exact_json(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("route artifact manifest changed on replay")
        return
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(
                    "route artifact manifest changed on replay"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
