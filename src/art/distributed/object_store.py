from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
import hashlib
from io import BytesIO, RawIOBase
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from threading import Lock
from typing import Any, Iterable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

VLLM_LORA_OBJECT_FORMAT = "art_vllm_lora_v1"
MOE_ROUTE_OBJECT_FORMAT = "art_moe_route_bundle_v2"
_S3_MIN_MULTIPART_PART_BYTES = 5 << 20
_S3_MAX_MULTIPART_PART_BYTES = 5 << 30
_S3_MAX_MULTIPART_PARTS = 10_000


class _ObjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class S3ObjectStoreConfig(_ObjectModel):
    endpoint_url: str = Field(pattern=r"^https://")
    region: str = Field(min_length=1)
    bucket: str = Field(min_length=1)
    prefix: str = Field(min_length=1)
    access_key_env: str = "CAIOS_KEY_ID"
    secret_key_env: str = "CAIOS_KEY_SECRET"
    session_token_env: str | None = None
    addressing_style: Literal["virtual", "path"] = "virtual"
    tls_verify: bool = True
    multipart_chunk_bytes: int = Field(
        default=64 << 20,
        ge=_S3_MIN_MULTIPART_PART_BYTES,
        le=_S3_MAX_MULTIPART_PART_BYTES,
    )
    multipart_concurrency: int = Field(default=16, ge=1)
    connect_timeout_s: float = Field(default=2.0, gt=0, le=60)
    read_timeout_s: float = Field(default=2.0, gt=0, le=300)
    max_attempts: int = Field(default=2, ge=1, le=5)

    @field_validator("bucket")
    @classmethod
    def _safe_bucket(cls, value: str) -> str:
        if "/" in value or value in {".", ".."}:
            raise ValueError("object-store bucket must be a bucket name")
        return value

    @field_validator("prefix")
    @classmethod
    def _safe_prefix(cls, value: str) -> str:
        normalized = value.strip("/")
        path = PurePosixPath(normalized)
        if not normalized or path.is_absolute() or ".." in path.parts:
            raise ValueError("object-store prefix must be safe and relative")
        return normalized


class BinaryObjectTarget(_ObjectModel):
    store: S3ObjectStoreConfig
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class BinaryObjectFile(_ObjectModel):
    relative_path: str
    byte_count: int = Field(ge=1)

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class BinaryObjectRef(_ObjectModel):
    transport: Literal["s3"] = "s3"
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: str = Field(min_length=1)
    manifest_uri: str = Field(pattern=r"^s3://")
    byte_count: int = Field(ge=1)
    files: tuple[BinaryObjectFile, ...] = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class BinaryObjectMaterialization(_ObjectModel):
    ref: BinaryObjectRef
    path: str = Field(min_length=1)


class BinaryObjectContents(_ObjectModel):
    ref: BinaryObjectRef
    file: BinaryObjectFile
    data: bytes = Field(min_length=1)


class S3BinaryObjectStore:
    """Publish immutable binary trees by committing their manifest last."""

    def __init__(self, config: S3ObjectStoreConfig) -> None:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config

        self.config = config
        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region,
            aws_access_key_id=_required_env(config.access_key_env),
            aws_secret_access_key=_required_env(config.secret_key_env),
            aws_session_token=(
                os.environ.get(config.session_token_env)
                if config.session_token_env is not None
                else None
            ),
            verify=config.tls_verify,
            config=Config(
                connect_timeout=config.connect_timeout_s,
                max_pool_connections=config.multipart_concurrency + 2,
                read_timeout=config.read_timeout_s,
                request_checksum_calculation="when_supported",
                response_checksum_validation="when_supported",
                retries={
                    "mode": "standard",
                    "total_max_attempts": config.max_attempts,
                },
                s3={"addressing_style": config.addressing_style},
                tcp_keepalive=True,
            ),
        )
        self._parts = ThreadPoolExecutor(
            max_workers=config.multipart_concurrency,
            thread_name_prefix="art-s3-part",
        )
        self._transfer = TransferConfig(
            multipart_threshold=config.multipart_chunk_bytes,
            multipart_chunksize=config.multipart_chunk_bytes,
            max_concurrency=config.multipart_concurrency,
        )
        self._lock = Lock()
        self._closed = False

    def publish(
        self,
        target: BinaryObjectTarget,
        files: dict[str, tuple[memoryview, ...]],
    ) -> BinaryObjectRef:
        self._require_target(target)
        existing = self.resolve(self._manifest_uri(target), missing_ok=True)
        if existing is not None:
            if (
                existing.object_id != target.object_id
                or existing.format != target.format
                or existing.metadata != target.metadata
            ):
                raise RuntimeError("committed object identity differs from its target")
            return existing
        prefix = self._prefix(target)
        published: list[BinaryObjectFile] = []
        # The prefix is the immutable object identity. A concurrent publisher can
        # commit the same prefix while this publisher is unwinding, so failed
        # publishes intentionally leave any partial keys for later lifecycle cleanup.
        for relative_path, chunks in sorted(files.items()):
            relative_path = _safe_relative_path(relative_path)
            byte_count = self._upload(f"{prefix}/{relative_path}", chunks)
            published.append(
                BinaryObjectFile(
                    relative_path=relative_path,
                    byte_count=byte_count,
                )
            )
        result = BinaryObjectRef(
            object_id=target.object_id,
            format=target.format,
            manifest_uri=self._manifest_uri(target),
            byte_count=sum(file.byte_count for file in published),
            files=tuple(published),
            metadata=target.metadata,
        )
        body = _manifest_bytes(result)
        self._client.put_object(
            Bucket=target.store.bucket,
            Key=f"{prefix}/_COMMITTED.json",
            Body=body,
            ContentType="application/json",
        )
        if self._read(f"{prefix}/_COMMITTED.json") != body:
            raise RuntimeError("object commit manifest changed after publication")
        return result

    def resolve(
        self, manifest_uri: str, *, missing_ok: bool = False
    ) -> BinaryObjectRef | None:
        from botocore.exceptions import ClientError

        bucket, key = self._require_manifest_uri(manifest_uri)
        try:
            body = self._read(key, bucket=bucket)
        except ClientError as error:
            if missing_ok and error.response.get("Error", {}).get("Code") in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return None
            raise
        return BinaryObjectRef.model_validate_json(body)

    def materialize(self, ref: BinaryObjectRef, destination: str | Path) -> Path:
        resolved = self.resolve(ref.manifest_uri)
        if resolved != ref:
            raise RuntimeError("binary object manifest changed before materialization")
        return self._materialize(ref, destination)

    def _materialize(self, ref: BinaryObjectRef, destination: str | Path) -> Path:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=False)
        bucket, manifest_key = _parse_s3_uri(ref.manifest_uri)
        prefix = str(PurePosixPath(manifest_key).parent)
        try:
            for file in ref.files:
                target = root / file.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                self._client.download_file(
                    bucket,
                    f"{prefix}/{file.relative_path}",
                    str(target),
                    Config=self._transfer,
                )
                if target.stat().st_size != file.byte_count:
                    raise RuntimeError(
                        f"binary object file size changed: {file.relative_path}"
                    )
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return root

    def _read_file(self, ref: BinaryObjectRef, file: BinaryObjectFile) -> bytes:
        bucket, manifest_key = _parse_s3_uri(ref.manifest_uri)
        prefix = str(PurePosixPath(manifest_key).parent)
        buffer = BytesIO()
        self._client.download_fileobj(
            bucket,
            f"{prefix}/{file.relative_path}",
            buffer,
            Config=self._transfer,
        )
        data = buffer.getvalue()
        if len(data) != file.byte_count:
            raise RuntimeError(f"binary object file size changed: {file.relative_path}")
        return data

    def read_file_into(
        self,
        ref: BinaryObjectRef,
        relative_path: str,
        target: memoryview,
    ) -> BinaryObjectFile:
        """Download one object directly into caller-owned host staging."""
        resolved = self.resolve(ref.manifest_uri)
        if resolved != ref:
            raise RuntimeError("binary object manifest changed before download")
        return self._read_file_into(ref, relative_path, target)

    def _read_file_into(
        self,
        ref: BinaryObjectRef,
        relative_path: str,
        target: memoryview,
    ) -> BinaryObjectFile:
        relative_path = _safe_relative_path(relative_path)
        matches = tuple(
            file for file in ref.files if file.relative_path == relative_path
        )
        if len(matches) != 1:
            raise RuntimeError("binary object file name must match exactly one file")
        file = matches[0]
        writer = _MemoryviewWriter(target, expected_bytes=file.byte_count)
        bucket, manifest_key = _parse_s3_uri(ref.manifest_uri)
        prefix = str(PurePosixPath(manifest_key).parent)
        self._client.download_fileobj(
            bucket,
            f"{prefix}/{file.relative_path}",
            writer,
            Config=self._transfer,
        )
        writer.verify_complete()
        return file

    def delete(self, ref: BinaryObjectRef) -> None:
        bucket, key = self._require_manifest_uri(ref.manifest_uri)
        self._delete_prefix(str(PurePosixPath(key).parent), bucket)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._parts.shutdown(wait=True)
        self._client.close()

    def _upload(self, key: str, chunks: tuple[memoryview, ...]) -> int:
        total = sum(chunk.nbytes for chunk in chunks)
        if total < 1:
            raise ValueError("binary object files must not be empty")
        if total <= self.config.multipart_chunk_bytes:
            self._client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=_ChunkRangeReader(chunks),
            )
            return total
        upload = self._client.create_multipart_upload(
            Bucket=self.config.bucket, Key=key
        )
        upload_id = upload["UploadId"]
        futures: dict[Future[dict[str, object]], int] = {}
        completed: list[dict[str, object]] = []
        part_bytes = _multipart_part_bytes(
            total,
            max_part_bytes=self.config.multipart_chunk_bytes,
            concurrency=self.config.multipart_concurrency,
        )
        try:
            for part_number, (offset, byte_count) in enumerate(
                _multipart_ranges(total, part_bytes), 1
            ):
                future = self._parts.submit(
                    self._client.upload_part,
                    Bucket=self.config.bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=_ChunkRangeReader(
                        chunks,
                        offset=offset,
                        byte_count=byte_count,
                    ),
                )
                futures[future] = part_number
                if len(futures) >= self.config.multipart_concurrency:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    completed.extend(
                        _complete_part(futures.pop(future), future) for future in done
                    )
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                completed.extend(
                    _complete_part(futures.pop(future), future) for future in done
                )
            completed.sort(key=lambda value: int(value["PartNumber"]))
            self._client.complete_multipart_upload(
                Bucket=self.config.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": completed},
            )
        except BaseException:
            pending = tuple(futures)
            for future in pending:
                future.cancel()
            wait(pending)
            for future in pending:
                with suppress(BaseException):
                    future.result()
            self._client.abort_multipart_upload(
                Bucket=self.config.bucket, Key=key, UploadId=upload_id
            )
            raise
        return total

    def _read(self, key: str, *, bucket: str | None = None) -> bytes:
        response = self._client.get_object(Bucket=bucket or self.config.bucket, Key=key)
        return response["Body"].read()

    def _delete_prefix(self, prefix: str, bucket: str) -> None:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
            objects = [{"Key": value["Key"]} for value in page.get("Contents", ())]
            if objects:
                self._client.delete_objects(Bucket=bucket, Delete={"Objects": objects})

    def _require_target(self, target: BinaryObjectTarget) -> None:
        if target.store != self.config:
            raise ValueError("binary object target uses another object store")
        with self._lock:
            if self._closed:
                raise RuntimeError("binary object store is closed")

    def _require_manifest_uri(self, value: str) -> tuple[str, str]:
        bucket, key = _parse_s3_uri(value)
        prefix = PurePosixPath(self.config.prefix)
        try:
            relative = PurePosixPath(key).relative_to(prefix)
        except ValueError as error:
            raise ValueError(
                "binary object URI leaves its configured prefix"
            ) from error
        if (
            bucket != self.config.bucket
            or len(relative.parts) != 2
            or re.fullmatch(r"[0-9a-f]{64}", relative.parts[0]) is None
            or relative.parts[1] != "_COMMITTED.json"
        ):
            raise ValueError("binary object URI does not identify a configured object")
        return bucket, key

    def _prefix(self, target: BinaryObjectTarget) -> str:
        return "/".join(
            value.strip("/")
            for value in (target.store.prefix, target.object_id)
            if value.strip("/")
        )

    def _manifest_uri(self, target: BinaryObjectTarget) -> str:
        return binary_object_manifest_uri(target)


class S3BinaryObjectReceiver:
    """Materialize committed objects into bounded, explicitly released paths."""

    def __init__(self, config: S3ObjectStoreConfig, receive_root: str | Path) -> None:
        self.config = config
        self.receive_root = Path(receive_root).resolve()
        self.receive_root.mkdir(parents=True, exist_ok=True)
        self._store = S3BinaryObjectStore(config)
        self._active: set[Path] = set()
        self._lock = Lock()
        self._closed = False

    def materialize(
        self,
        manifest_uri: str,
        *,
        expected_format: str,
        expected_metadata: dict[str, str],
    ) -> BinaryObjectMaterialization:
        with self._lock:
            if self._closed:
                raise RuntimeError("binary object receiver is closed")
        ref = self._resolve(manifest_uri, expected_format, expected_metadata)
        destination = self.receive_root / f".receive-{uuid4().hex}"
        self._store._materialize(ref, destination)
        with self._lock:
            if self._closed:
                shutil.rmtree(destination, ignore_errors=True)
                raise RuntimeError(
                    "binary object receiver closed during materialization"
                )
            self._active.add(destination)
        return BinaryObjectMaterialization(ref=ref, path=str(destination))

    def read_file(
        self,
        manifest_uri: str,
        *,
        expected_format: str,
        expected_metadata: dict[str, str],
        relative_path: str,
    ) -> BinaryObjectContents:
        with self._lock:
            if self._closed:
                raise RuntimeError("binary object receiver is closed")
        ref = self._resolve(manifest_uri, expected_format, expected_metadata)
        matches = tuple(
            file for file in ref.files if file.relative_path == relative_path
        )
        if len(matches) != 1:
            raise RuntimeError(
                "binary object does not contain exactly one requested file"
            )
        file = matches[0]
        return BinaryObjectContents(
            ref=ref, file=file, data=self._store._read_file(ref, file)
        )

    def read_file_into(
        self,
        manifest_uri: str,
        *,
        expected_format: str,
        expected_metadata: dict[str, str],
        relative_path: str,
        target: memoryview,
    ) -> BinaryObjectFile:
        with self._lock:
            if self._closed:
                raise RuntimeError("binary object receiver is closed")
        ref = self._resolve(manifest_uri, expected_format, expected_metadata)
        return self._store._read_file_into(ref, relative_path, target)

    def release(self, materialization: BinaryObjectMaterialization) -> None:
        path = self._managed_path(materialization.path)
        with self._lock:
            if path not in self._active:
                raise RuntimeError("binary object materialization is not active")
            self._active.remove(path)
        try:
            shutil.rmtree(path)
        except BaseException:
            with self._lock:
                self._active.add(path)
            raise

    def delete(self, manifest_uri: str) -> None:
        ref = self._store.resolve(manifest_uri, missing_ok=True)
        if ref is not None:
            self._store.delete(ref)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            active = tuple(self._active)
            self._active.clear()
        failures: list[BaseException] = []
        for path in active:
            try:
                shutil.rmtree(path)
            except BaseException as error:
                failures.append(error)
        self._store.close()
        if failures:
            raise BaseExceptionGroup("binary object receiver cleanup failed", failures)

    def _managed_path(self, value: str) -> Path:
        path = Path(value).resolve()
        if path.parent != self.receive_root or not path.name.startswith(".receive-"):
            raise RuntimeError("binary object materialization leaves its receive root")
        return path

    def _resolve(
        self,
        manifest_uri: str,
        expected_format: str,
        expected_metadata: dict[str, str],
    ) -> BinaryObjectRef:
        ref = self._store.resolve(manifest_uri)
        assert ref is not None
        if ref.format != expected_format:
            raise RuntimeError("binary object format does not match its consumer")
        if any(
            ref.metadata.get(key) != value for key, value in expected_metadata.items()
        ):
            raise RuntimeError("binary object metadata does not match its consumer")
        return ref


def binary_object_manifest_uri(target: BinaryObjectTarget) -> str:
    prefix = "/".join(
        value.strip("/")
        for value in (target.store.prefix, target.object_id)
        if value.strip("/")
    )
    return f"s3://{target.store.bucket}/{prefix}/_COMMITTED.json"


def vllm_lora_object_target(
    store: S3ObjectStoreConfig,
    *,
    run_id: str,
    training_session_id: str,
    generation_id: str,
    policy_step: int,
) -> BinaryObjectTarget:
    object_id = hashlib.sha256(f"{run_id}\0{generation_id}".encode()).hexdigest()
    return BinaryObjectTarget(
        store=store,
        object_id=object_id,
        format=VLLM_LORA_OBJECT_FORMAT,
        metadata={
            "run_id": run_id,
            "training_session_id": training_session_id,
            "generation_id": generation_id,
            "policy_step": str(policy_step),
        },
    )


def moe_route_object_target(
    store: S3ObjectStoreConfig,
    *,
    tenant_id: str,
    run_id: str,
    object_id: str,
) -> BinaryObjectTarget:
    return BinaryObjectTarget(
        store=store,
        object_id=object_id,
        format=MOE_ROUTE_OBJECT_FORMAT,
        metadata={"tenant_id": tenant_id, "run_id": run_id},
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required object-store credential {name} is missing")
    return value


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise ValueError("object file path must be safe and relative")
    return str(path)


def _multipart_part_bytes(total: int, *, max_part_bytes: int, concurrency: int) -> int:
    return max(
        _S3_MIN_MULTIPART_PART_BYTES,
        min(max_part_bytes, (total + concurrency - 1) // concurrency),
        (total + _S3_MAX_MULTIPART_PARTS - 1) // _S3_MAX_MULTIPART_PARTS,
    )


def _multipart_ranges(total: int, size: int) -> Iterable[tuple[int, int]]:
    for offset in range(0, total, size):
        yield offset, min(size, total - offset)


class _ChunkRangeReader(RawIOBase):
    """Seekable stream over immutable tensor-backed buffers without coalescing."""

    def __init__(
        self,
        chunks: tuple[memoryview, ...],
        *,
        offset: int = 0,
        byte_count: int | None = None,
    ) -> None:
        self._chunks = tuple(chunk.cast("B") for chunk in chunks if chunk.nbytes)
        self._offsets: list[int] = [0]
        for chunk in self._chunks:
            self._offsets.append(self._offsets[-1] + chunk.nbytes)
        total = self._offsets[-1]
        byte_count = total - offset if byte_count is None else byte_count
        if offset < 0 or byte_count < 0 or offset + byte_count > total:
            raise ValueError("chunk reader range leaves its source buffers")
        self._start = offset
        self._length = byte_count
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self._position + offset
        elif whence == 2:
            position = self._length + offset
        else:
            raise ValueError(f"unsupported seek mode {whence}")
        if position < 0 or position > self._length:
            raise ValueError("chunk reader seek leaves its range")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        remaining = self._length - self._position
        size = remaining if size is None or size < 0 else min(size, remaining)
        if size == 0:
            return b""
        absolute = self._start + self._position
        pending = size
        output: list[memoryview] = []
        first = max(0, bisect_right(self._offsets, absolute) - 1)
        for index in range(first, len(self._chunks)):
            chunk = self._chunks[index]
            start, end = self._offsets[index], self._offsets[index + 1]
            if absolute >= end:
                continue
            local = max(absolute - start, 0)
            count = min(end - start - local, pending)
            output.append(chunk[local : local + count])
            absolute += count
            pending -= count
            if pending == 0:
                break
        if pending:
            raise RuntimeError("chunk reader did not cover its declared range")
        self._position += size
        return b"".join(output)


class _MemoryviewWriter(RawIOBase):
    """Seekable S3 download target over a caller-owned writable buffer."""

    def __init__(self, target: memoryview, *, expected_bytes: int) -> None:
        self._target = target.cast("B")
        if self._target.readonly or self._target.nbytes != expected_bytes:
            raise ValueError("download target must be writable and exactly sized")
        self._position = 0
        self._intervals: list[tuple[int, int]] = []
        self._lock = Lock()

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        with self._lock:
            return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        with self._lock:
            if whence == 0:
                position = offset
            elif whence == 1:
                position = self._position + offset
            elif whence == 2:
                position = self._target.nbytes + offset
            else:
                raise ValueError(f"unsupported seek mode {whence}")
            if position < 0 or position > self._target.nbytes:
                raise ValueError("download target seek leaves its buffer")
            self._position = position
            return position

    def write(self, value: Any) -> int:
        source = memoryview(value).cast("B")
        with self._lock:
            end = self._position + source.nbytes
            if end > self._target.nbytes:
                raise RuntimeError("binary object download exceeded its target")
            self._target[self._position : end] = source
            self._intervals.append((self._position, end))
            self._position = end
        return source.nbytes

    def verify_complete(self) -> None:
        cursor = 0
        for start, end in sorted(self._intervals):
            if start > cursor:
                raise RuntimeError("binary object download left an unwritten range")
            cursor = max(cursor, end)
        if cursor != self._target.nbytes:
            raise RuntimeError("binary object download was incomplete")


def _complete_part(
    part_number: int, future: Future[dict[str, object]]
) -> dict[str, object]:
    return {"ETag": future.result()["ETag"], "PartNumber": part_number}


def _manifest_bytes(ref: BinaryObjectRef) -> bytes:
    return json.dumps(
        ref.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
    ).encode()


def _parse_s3_uri(value: str) -> tuple[str, str]:
    from urllib.parse import urlsplit

    parsed = urlsplit(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 object URI: {value!r}")
    return parsed.netloc, parsed.path.strip("/")
