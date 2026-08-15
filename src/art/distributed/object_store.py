from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from threading import Lock
from typing import Iterable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

VLLM_LORA_OBJECT_FORMAT = "art_vllm_lora_v1"
MOE_ROUTE_OBJECT_FORMAT = "art_moe_route_bundle_v1"


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
    multipart_chunk_bytes: int = Field(default=64 << 20, ge=5 << 20)
    multipart_concurrency: int = Field(default=8, ge=1)

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
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

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
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
                max_pool_connections=config.multipart_concurrency + 2,
                s3={"addressing_style": config.addressing_style},
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
        try:
            for relative_path, chunks in sorted(files.items()):
                relative_path = _safe_relative_path(relative_path)
                byte_count, digest = self._upload(f"{prefix}/{relative_path}", chunks)
                published.append(
                    BinaryObjectFile(
                        relative_path=relative_path,
                        byte_count=byte_count,
                        sha256=digest,
                    )
                )
            tree_sha256 = _tree_sha256(published)
            result = BinaryObjectRef(
                object_id=target.object_id,
                format=target.format,
                manifest_uri=self._manifest_uri(target),
                byte_count=sum(file.byte_count for file in published),
                tree_sha256=tree_sha256,
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
        except BaseException:
            self._delete_prefix(prefix, target.store.bucket)
            raise

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
                if (
                    target.stat().st_size != file.byte_count
                    or _file_sha256(target) != file.sha256
                ):
                    raise RuntimeError(
                        f"binary object file changed: {file.relative_path}"
                    )
        except BaseException:
            import shutil

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
        if (
            len(data) != file.byte_count
            or hashlib.sha256(data).hexdigest() != file.sha256
        ):
            raise RuntimeError(f"binary object file changed: {file.relative_path}")
        return data

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

    def _upload(self, key: str, chunks: tuple[memoryview, ...]) -> tuple[int, str]:
        total = sum(chunk.nbytes for chunk in chunks)
        if total < 1:
            raise ValueError("binary object files must not be empty")
        digest = hashlib.sha256()
        if total <= self.config.multipart_chunk_bytes:
            body = b"".join(chunk.tobytes() for chunk in chunks)
            digest.update(body)
            self._client.put_object(Bucket=self.config.bucket, Key=key, Body=body)
            return total, digest.hexdigest()
        upload = self._client.create_multipart_upload(
            Bucket=self.config.bucket, Key=key
        )
        upload_id = upload["UploadId"]
        futures: list[tuple[int, Future[dict[str, object]]]] = []
        completed: list[dict[str, object]] = []
        try:
            for part_number, body in enumerate(
                _multipart_chunks(chunks, self.config.multipart_chunk_bytes), 1
            ):
                digest.update(body)
                futures.append(
                    (
                        part_number,
                        self._parts.submit(
                            self._client.upload_part,
                            Bucket=self.config.bucket,
                            Key=key,
                            UploadId=upload_id,
                            PartNumber=part_number,
                            Body=body,
                        ),
                    )
                )
                if len(futures) >= self.config.multipart_concurrency:
                    completed.append(_complete_part(futures.pop(0)))
            completed.extend(_complete_part(value) for value in futures)
            self._client.complete_multipart_upload(
                Bucket=self.config.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": completed},
            )
        except BaseException:
            self._client.abort_multipart_upload(
                Bucket=self.config.bucket, Key=key, UploadId=upload_id
            )
            raise
        return total, digest.hexdigest()

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
    """Materialize verified objects into bounded, explicitly released paths."""

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


def _multipart_chunks(chunks: tuple[memoryview, ...], size: int) -> Iterable[bytes]:
    buffer = bytearray()
    for chunk in chunks:
        view = chunk.cast("B")
        offset = 0
        while offset < view.nbytes:
            count = min(size - len(buffer), view.nbytes - offset)
            buffer.extend(view[offset : offset + count])
            offset += count
            if len(buffer) == size:
                yield bytes(buffer)
                buffer.clear()
    if buffer:
        yield bytes(buffer)


def _complete_part(value: tuple[int, Future[dict[str, object]]]) -> dict[str, object]:
    part_number, future = value
    return {"ETag": future.result()["ETag"], "PartNumber": part_number}


def _tree_sha256(files: list[BinaryObjectFile]) -> str:
    return hashlib.sha256(
        json.dumps(
            [file.model_dump(mode="json") for file in files],
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()
