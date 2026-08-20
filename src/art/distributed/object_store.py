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
from typing import Annotated, Any, Iterable, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

VLLM_LORA_OBJECT_FORMAT = "art_vllm_lora_v1"
MOE_ROUTE_OBJECT_FORMAT = "art_moe_route_bundle_v2"
_S3_MIN_MULTIPART_PART_BYTES = 5 << 20
_S3_MAX_MULTIPART_PART_BYTES = 5 << 30
_S3_MAX_MULTIPART_PARTS = 10_000


class _ObjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class S3ObjectStoreNamespace(_ObjectModel):
    endpoint_url: str
    bucket: str
    prefix: str


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

    @property
    def namespace(self) -> S3ObjectStoreNamespace:
        return S3ObjectStoreNamespace(
            endpoint_url=self.endpoint_url,
            bucket=self.bucket,
            prefix=self.prefix,
        )

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
    transport: Literal["s3"] = "s3"
    store: S3ObjectStoreConfig
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class BinaryObjectFile(_ObjectModel):
    relative_path: str
    byte_count: int = Field(ge=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

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

    @model_validator(mode="after")
    def _content_addressed_files(self) -> "BinaryObjectRef":
        if any(file.sha256 is None for file in self.files):
            raise ValueError("content-addressed object files require SHA-256")
        return self


class OrderedBinaryObjectTarget(_ObjectModel):
    transport: Literal["ordered_s3_shards"] = "ordered_s3_shards"
    store: S3ObjectStoreConfig
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    shard_bytes: int = Field(default=64 << 20, ge=1, le=5 << 30)
    max_concurrent_shards: int = Field(default=16, ge=1, le=64)
    max_shards: int = Field(default=1024, ge=1, le=10_000)

    @model_validator(mode="after")
    def _bounded_by_store(self) -> "OrderedBinaryObjectTarget":
        if self.max_concurrent_shards > self.store.multipart_concurrency:
            raise ValueError("ordered shard concurrency exceeds the object-store pool")
        return self


BinaryObjectPublicationTarget = Annotated[
    BinaryObjectTarget | OrderedBinaryObjectTarget,
    Field(discriminator="transport"),
]


class OrderedBinaryObjectShard(_ObjectModel):
    index: int = Field(ge=0)
    relative_path: str
    file_offset: int = Field(ge=0)
    byte_count: int = Field(ge=1)

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class OrderedBinaryObjectPlan(_ObjectModel):
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: str = Field(min_length=1)
    files: tuple[BinaryObjectFile, ...] = Field(min_length=1)
    shards: tuple[OrderedBinaryObjectShard, ...] = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _complete_ordered_layout(self) -> "OrderedBinaryObjectPlan":
        if tuple(shard.index for shard in self.shards) != tuple(
            range(len(self.shards))
        ):
            raise ValueError("ordered object shards must have contiguous indices")
        files = {file.relative_path: file for file in self.files}
        if len(files) != len(self.files):
            raise ValueError("ordered object files must have unique paths")
        cursors = dict.fromkeys(files, 0)
        for shard in self.shards:
            if shard.relative_path not in cursors:
                raise ValueError("ordered object shard identifies an unknown file")
            if shard.file_offset != cursors[shard.relative_path]:
                raise ValueError("ordered object shards leave a file gap")
            cursors[shard.relative_path] += shard.byte_count
        if any(cursors[path] != file.byte_count for path, file in files.items()):
            raise ValueError("ordered object shards do not cover a complete file")
        return self


class OrderedBinaryObjectRef(_ObjectModel):
    transport: Literal["ordered_s3_shards"] = "ordered_s3_shards"
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: str = Field(min_length=1)
    plan_uri: str = Field(pattern=r"^s3://")
    manifest_uri: str = Field(pattern=r"^s3://")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=1)
    files: tuple[BinaryObjectFile, ...] = Field(min_length=1)
    shard_count: int = Field(ge=1, le=10_000)
    metadata: dict[str, str] = Field(default_factory=dict)


BinaryObjectPublicationRef = Annotated[
    BinaryObjectRef | OrderedBinaryObjectRef,
    Field(discriminator="transport"),
]


class StoredOrderedBinaryObjectShard(_ObjectModel):
    index: int = Field(ge=0)
    byte_count: int = Field(ge=1)
    etag: str = Field(min_length=1)


class OrderedBinaryObjectCommit(_ObjectModel):
    transport: Literal["ordered_s3_shards"] = "ordered_s3_shards"
    ref: OrderedBinaryObjectRef
    shards: tuple[StoredOrderedBinaryObjectShard, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _complete_commit(self) -> "OrderedBinaryObjectCommit":
        if tuple(shard.index for shard in self.shards) != tuple(
            range(self.ref.shard_count)
        ):
            raise ValueError("ordered object commit does not cover every shard")
        if sum(shard.byte_count for shard in self.shards) != self.ref.byte_count:
            raise ValueError("ordered object commit has the wrong byte count")
        return self


BinaryObjectPublicationCommit = Annotated[
    BinaryObjectRef | OrderedBinaryObjectCommit,
    Field(discriminator="transport"),
]
_BINARY_OBJECT_COMMIT_ADAPTER = TypeAdapter(BinaryObjectPublicationCommit)


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
        *,
        file_sha256: dict[str, str],
    ) -> BinaryObjectRef:
        self._require_target(target)
        if set(files) != set(file_sha256):
            raise ValueError("binary object digests must cover every file exactly")
        planned = tuple(
            BinaryObjectFile(
                relative_path=_safe_relative_path(relative_path),
                byte_count=sum(chunk.nbytes for chunk in chunks),
                sha256=file_sha256[relative_path],
            )
            for relative_path, chunks in sorted(files.items())
        )
        existing = self.resolve(self._manifest_uri(target), missing_ok=True)
        if existing is not None:
            if (
                not isinstance(existing, BinaryObjectRef)
                or existing.object_id != target.object_id
                or existing.format != target.format
                or existing.metadata != target.metadata
                or existing.files != planned
            ):
                raise RuntimeError("committed object identity differs from its target")
            return existing
        prefix = self._prefix(target)
        # Content-addressed payload keys cannot overwrite another writer's bytes.
        # Failed writes remain invisible until the conditional manifest commit and
        # are reclaimed later by the lifecycle owner after writer fencing.
        for file in planned:
            written = self._upload(
                _binary_object_file_key(prefix, file), files[file.relative_path]
            )
            if written != file.byte_count:
                raise RuntimeError("binary object upload changed its planned size")
        result = BinaryObjectRef(
            object_id=target.object_id,
            format=target.format,
            manifest_uri=self._manifest_uri(target),
            byte_count=sum(file.byte_count for file in planned),
            files=planned,
            metadata=target.metadata,
        )
        body = _manifest_bytes(result)
        try:
            self._client.put_object(
                Bucket=target.store.bucket,
                Key=f"{prefix}/_COMMITTED.json",
                Body=body,
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except Exception as error:
            if not _is_precondition_failed(error):
                raise
            committed = self.resolve(self._manifest_uri(target))
            if committed is None:
                raise RuntimeError(
                    "conditional object commit lost its winner"
                ) from error
            if not isinstance(committed, BinaryObjectRef) or committed != result:
                raise RuntimeError(
                    "another publisher committed different object content"
                ) from error
            return committed
        return result

    def publish_ordered(
        self,
        target: OrderedBinaryObjectTarget,
        files: dict[str, tuple[memoryview, ...]],
        *,
        file_sha256: dict[str, str] | None = None,
    ) -> OrderedBinaryObjectRef:
        """Publish caller-owned buffers as directly consumable ordered shards."""
        self._require_target(target)
        plan = _ordered_binary_object_plan(target, files, file_sha256)
        plan_body = _model_bytes(plan)
        ref = _ordered_binary_object_ref(target, plan, plan_body=plan_body)
        existing = self.resolve_ordered(ref.manifest_uri, missing_ok=True)
        if existing is not None:
            if existing != ref:
                raise RuntimeError("committed ordered object differs from its target")
            return existing
        prefix = self._prefix(target)
        self._put_immutable(f"{prefix}/_PLAN.json", plan_body)
        stored = self._upload_ordered_shards(
            target, plan, files, plan_sha256=ref.plan_sha256
        )
        commit = OrderedBinaryObjectCommit(ref=ref, shards=stored)
        body = _model_bytes(commit)
        self._put_immutable(f"{prefix}/_COMMITTED.json", body)
        return ref

    def resolve_ordered(
        self, manifest_uri: str, *, missing_ok: bool = False
    ) -> OrderedBinaryObjectRef | None:
        ref = self.resolve(manifest_uri, missing_ok=missing_ok)
        if ref is None:
            return None
        if not isinstance(ref, OrderedBinaryObjectRef):
            raise RuntimeError("binary object commit is not ordered")
        return ref

    def resolve(
        self, manifest_uri: str, *, missing_ok: bool = False
    ) -> BinaryObjectPublicationRef | None:
        commit = self._resolve_commit(manifest_uri, missing_ok=missing_ok)
        if commit is None:
            return None
        return commit.ref if isinstance(commit, OrderedBinaryObjectCommit) else commit

    def _resolve_commit(
        self, manifest_uri: str, *, missing_ok: bool = False
    ) -> BinaryObjectPublicationCommit | None:
        from botocore.exceptions import ClientError

        bucket, key = self._require_manifest_uri(manifest_uri)
        try:
            body = self._read(key, bucket=bucket)
        except ClientError as error:
            if missing_ok and _is_missing_object(error):
                return None
            raise
        commit = _BINARY_OBJECT_COMMIT_ADAPTER.validate_json(body)
        ref = commit.ref if isinstance(commit, OrderedBinaryObjectCommit) else commit
        if (
            ref.manifest_uri != manifest_uri
            or ref.object_id != PurePosixPath(key).parent.name
        ):
            raise RuntimeError("binary object commit identifies another manifest")
        return commit

    def verify(
        self,
        ref: BinaryObjectPublicationRef,
        *,
        target: BinaryObjectPublicationTarget | None = None,
        expected_byte_count: int | None = None,
    ) -> None:
        """Stream-verify an object when manifest-only recovery is insufficient."""
        if isinstance(ref, OrderedBinaryObjectRef):
            if not isinstance(target, OrderedBinaryObjectTarget):
                raise ValueError(
                    "ordered object verification requires its exact target"
                )
            if expected_byte_count is None or expected_byte_count < 1:
                raise ValueError(
                    "ordered object verification requires its exact byte count"
                )
            self._require_target(target)
            self._verify_ordered(ref, target, expected_byte_count)
            return
        if target is not None and not isinstance(target, BinaryObjectTarget):
            raise ValueError("binary object verification target changed transport")
        if expected_byte_count is not None and ref.byte_count != expected_byte_count:
            raise RuntimeError("binary object aggregate size changed")
        if self.resolve(ref.manifest_uri) != ref:
            raise RuntimeError("binary object manifest changed before verification")
        bucket, manifest_key = _parse_s3_uri(ref.manifest_uri)
        prefix = str(PurePosixPath(manifest_key).parent)
        for file in ref.files:
            try:
                response = self._client.get_object(
                    Bucket=bucket,
                    Key=_binary_object_file_key(prefix, file),
                )
            except Exception as error:
                if _is_missing_object(error):
                    raise RuntimeError(
                        f"binary object file is missing: {file.relative_path}"
                    ) from error
                raise
            body = response["Body"]
            digest, byte_count = hashlib.sha256(), 0
            try:
                while chunk := body.read(8 << 20):
                    digest.update(chunk)
                    byte_count += len(chunk)
            finally:
                body.close()
            if byte_count != file.byte_count or digest.hexdigest() != file.sha256:
                raise RuntimeError(
                    f"binary object file identity changed: {file.relative_path}"
                )

    def _verify_ordered(
        self,
        ref: OrderedBinaryObjectRef,
        target: OrderedBinaryObjectTarget,
        expected_byte_count: int,
    ) -> None:
        commit = self._resolve_commit(ref.manifest_uri)
        if not isinstance(commit, OrderedBinaryObjectCommit) or commit.ref != ref:
            raise RuntimeError("ordered object commit changed before verification")
        bucket, plan_key = self._ordered_plan_location(ref)
        try:
            plan_body = self._read(plan_key, bucket=bucket)
        except Exception as error:
            if _is_missing_object(error):
                raise RuntimeError("ordered object plan is missing") from error
            raise
        if hashlib.sha256(plan_body).hexdigest() != ref.plan_sha256:
            raise RuntimeError("ordered object plan changed identity")
        plan = OrderedBinaryObjectPlan.model_validate_json(plan_body)
        _require_ordered_plan_target(
            target, plan, expected_byte_count=expected_byte_count
        )
        if _ordered_binary_object_ref(target, plan, plan_body=plan_body) != ref:
            raise RuntimeError("ordered object plan differs from its exact target")
        source_files = {file.relative_path: file for file in plan.files}
        prefix = str(PurePosixPath(plan_key).parent)
        for shard, stored in zip(plan.shards, commit.shards, strict=True):
            if stored.byte_count != shard.byte_count:
                raise RuntimeError("ordered object commit changed a shard size")
            key = _ordered_binary_object_shard_key(prefix, shard.index)
            try:
                response = self._client.head_object(Bucket=bucket, Key=key)
            except Exception as error:
                if _is_missing_object(error):
                    raise RuntimeError(
                        f"ordered object shard is missing: {shard.index}"
                    ) from error
                raise
            file = source_files[shard.relative_path]
            expected_metadata = _ordered_binary_object_shard_metadata(
                ref.plan_sha256, file.sha256, shard
            )
            if (
                response.get("ContentLength") != shard.byte_count
                or response.get("Metadata") != expected_metadata
                or response.get("ETag") != stored.etag
            ):
                raise RuntimeError(
                    f"ordered object shard changed identity: {shard.index}"
                )

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
                    _binary_object_file_key(prefix, file),
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
            _binary_object_file_key(prefix, file),
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
            _binary_object_file_key(prefix, file),
            writer,
            Config=self._transfer,
        )
        writer.verify_complete()
        return file

    def delete(self, ref: BinaryObjectPublicationRef) -> None:
        if self.resolve(ref.manifest_uri) != ref:
            raise RuntimeError("binary object commit changed before deletion")
        bucket, manifest_key = self._require_manifest_uri(ref.manifest_uri)
        prefix = str(PurePosixPath(manifest_key).parent)
        if not isinstance(ref, OrderedBinaryObjectRef):
            self._delete_prefix(prefix, bucket)
            return
        self._ordered_plan_location(ref)
        self._delete_keys(
            bucket,
            tuple(
                _ordered_binary_object_shard_key(prefix, index)
                for index in range(ref.shard_count)
            ),
        )
        self._delete_keys(bucket, (f"{prefix}/_PLAN.json",))
        self._delete_keys(bucket, (manifest_key,))
        if self.resolve(ref.manifest_uri, missing_ok=True) is not None:
            raise RuntimeError("binary object commit survived deletion")

    def discard_uncommitted(
        self,
        target: BinaryObjectPublicationTarget,
        files: tuple[BinaryObjectFile, ...],
    ) -> None:
        """Delete only exact fenced payload keys when no commit manifest exists."""
        self._require_target(target)
        if self.resolve(self._manifest_uri(target), missing_ok=True) is not None:
            raise RuntimeError("cannot discard a committed binary object")
        prefix = self._prefix(target)
        if isinstance(target, OrderedBinaryObjectTarget):
            plan = _ordered_binary_object_layout(target, files)
            plan_key = f"{prefix}/_PLAN.json"
            try:
                existing_plan = self._read(plan_key)
            except Exception as error:
                if not _is_missing_object(error):
                    raise
            else:
                if existing_plan != _model_bytes(plan):
                    raise RuntimeError(
                        "uncommitted ordered object plan differs from its target"
                    )
            self._delete_keys(
                target.store.bucket,
                tuple(
                    _ordered_binary_object_shard_key(prefix, shard.index)
                    for shard in plan.shards
                ),
            )
            self._delete_keys(target.store.bucket, (plan_key,))
        else:
            keys = tuple(_binary_object_file_key(prefix, file) for file in files)
            if len(keys) != len(set(keys)):
                raise ValueError(
                    "binary object cleanup contains duplicate payload keys"
                )
            for key in keys:
                self._abort_uploads(key)
            self._delete_keys(target.store.bucket, keys)
        if self.resolve(self._manifest_uri(target), missing_ok=True) is not None:
            raise RuntimeError("binary object committed during fenced cleanup")

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

    def _upload_ordered_shards(
        self,
        target: OrderedBinaryObjectTarget,
        plan: OrderedBinaryObjectPlan,
        files: dict[str, tuple[memoryview, ...]],
        *,
        plan_sha256: str,
    ) -> tuple[StoredOrderedBinaryObjectShard, ...]:
        pending: dict[Future[StoredOrderedBinaryObjectShard], int] = {}
        completed: list[StoredOrderedBinaryObjectShard] = []
        shards = iter(plan.shards)
        source_sha256 = {file.relative_path: file.sha256 for file in plan.files}

        def submit() -> bool:
            try:
                shard = next(shards)
            except StopIteration:
                return False
            future = self._parts.submit(
                self._upload_ordered_shard,
                self._prefix(target),
                shard,
                files[shard.relative_path],
                plan_sha256,
                source_sha256[shard.relative_path],
            )
            pending[future] = shard.index
            return True

        for _ in range(min(target.max_concurrent_shards, len(plan.shards))):
            submit()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    value = future.result()
                    if value.index != index:
                        raise RuntimeError("ordered shard result changed its index")
                    completed.append(value)
                    submit()
        except BaseException:
            futures = tuple(pending)
            for future in futures:
                future.cancel()
            wait(futures)
            for future in futures:
                with suppress(BaseException):
                    future.result()
            raise
        return tuple(sorted(completed, key=lambda shard: shard.index))

    def _upload_ordered_shard(
        self,
        prefix: str,
        shard: OrderedBinaryObjectShard,
        chunks: tuple[memoryview, ...],
        plan_sha256: str,
        source_sha256: str | None,
    ) -> StoredOrderedBinaryObjectShard:
        key = _ordered_binary_object_shard_key(prefix, shard.index)
        metadata = _ordered_binary_object_shard_metadata(
            plan_sha256, source_sha256, shard
        )
        try:
            response = self._client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=_ChunkRangeReader(
                    chunks,
                    offset=shard.file_offset,
                    byte_count=shard.byte_count,
                ),
                ContentLength=shard.byte_count,
                Metadata=metadata,
                IfNoneMatch="*",
            )
            etag = response.get("ETag")
        except Exception as error:
            if not _is_precondition_failed(error):
                raise
            head = self._client.head_object(Bucket=self.config.bucket, Key=key)
            if (
                head.get("ContentLength") != shard.byte_count
                or head.get("Metadata") != metadata
            ):
                raise RuntimeError(
                    "committed ordered shard changed identity"
                ) from error
            etag = head.get("ETag")
        if not isinstance(etag, str) or not etag:
            raise RuntimeError("ordered shard upload returned no ETag")
        return StoredOrderedBinaryObjectShard(
            index=shard.index,
            byte_count=shard.byte_count,
            etag=etag,
        )

    def _put_immutable(self, key: str, body: bytes) -> None:
        try:
            self._client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except Exception as error:
            if not _is_precondition_failed(error):
                raise
            if self._read(key) != body:
                raise RuntimeError("immutable object metadata changed") from error

    def _abort_uploads(self, key: str) -> None:
        key_marker = None
        upload_marker = None
        while True:
            markers = (
                {}
                if key_marker is None
                else {
                    "KeyMarker": key_marker,
                    "UploadIdMarker": upload_marker,
                }
            )
            values = self._client.list_multipart_uploads(
                Bucket=self.config.bucket,
                Prefix=key,
                **markers,
            )
            for upload in values.get("Uploads", ()):
                if upload.get("Key") == key:
                    self._client.abort_multipart_upload(
                        Bucket=self.config.bucket,
                        Key=key,
                        UploadId=upload["UploadId"],
                    )
            if not values.get("IsTruncated"):
                return
            key_marker = values.get("NextKeyMarker")
            upload_marker = values.get("NextUploadIdMarker")
            if not key_marker or not upload_marker:
                raise RuntimeError("multipart cleanup pagination lost its cursor")

    def _read(self, key: str, *, bucket: str | None = None) -> bytes:
        response = self._client.get_object(Bucket=bucket or self.config.bucket, Key=key)
        return response["Body"].read()

    def _delete_keys(self, bucket: str, keys: tuple[str, ...]) -> None:
        for offset in range(0, len(keys), 1000):
            result = self._client.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [{"Key": key} for key in keys[offset : offset + 1000]],
                    "Quiet": True,
                },
            )
            if errors := result.get("Errors"):
                raise RuntimeError(f"binary object cleanup failed: {errors}")

    def _delete_prefix(self, prefix: str, bucket: str) -> None:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
            objects = [{"Key": value["Key"]} for value in page.get("Contents", ())]
            if objects:
                self._client.delete_objects(Bucket=bucket, Delete={"Objects": objects})

    def _require_target(
        self, target: BinaryObjectTarget | OrderedBinaryObjectTarget
    ) -> None:
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

    def _ordered_plan_location(self, ref: OrderedBinaryObjectRef) -> tuple[str, str]:
        bucket, manifest_key = self._require_manifest_uri(ref.manifest_uri)
        plan_bucket, plan_key = _parse_s3_uri(ref.plan_uri)
        if (plan_bucket, plan_key) != (
            bucket,
            f"{PurePosixPath(manifest_key).parent}/_PLAN.json",
        ):
            raise RuntimeError("ordered object plan leaves its manifest prefix")
        return plan_bucket, plan_key

    def _prefix(self, target: BinaryObjectTarget | OrderedBinaryObjectTarget) -> str:
        return "/".join(
            value.strip("/")
            for value in (target.store.prefix, target.object_id)
            if value.strip("/")
        )

    def _manifest_uri(
        self, target: BinaryObjectTarget | OrderedBinaryObjectTarget
    ) -> str:
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

    def read_single_file_into(
        self,
        manifest_uri: str,
        *,
        expected_format: str,
        expected_metadata: dict[str, str],
        expected_object_id: str,
        relative_path: str,
        target: memoryview,
    ) -> BinaryObjectFile:
        """Download an exact one-file object into caller-owned host memory."""
        with self._lock:
            if self._closed:
                raise RuntimeError("binary object receiver is closed")
        ref = self._resolve(manifest_uri, expected_format, expected_metadata)
        if (
            ref.object_id != expected_object_id
            or ref.byte_count != target.nbytes
            or len(ref.files) != 1
        ):
            raise RuntimeError("binary object changed its single-file identity")
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
        if not isinstance(ref, BinaryObjectRef):
            raise RuntimeError("ordered binary objects require the ordered receiver")
        if ref.format != expected_format:
            raise RuntimeError("binary object format does not match its consumer")
        if any(
            ref.metadata.get(key) != value for key, value in expected_metadata.items()
        ):
            raise RuntimeError("binary object metadata does not match its consumer")
        return ref


def binary_object_manifest_uri(
    target: BinaryObjectTarget | OrderedBinaryObjectTarget,
) -> str:
    prefix = "/".join(
        value.strip("/")
        for value in (target.store.prefix, target.object_id)
        if value.strip("/")
    )
    return f"s3://{target.store.bucket}/{prefix}/_COMMITTED.json"


def ordered_binary_object_plan_uri(target: OrderedBinaryObjectTarget) -> str:
    prefix = "/".join(
        value.strip("/")
        for value in (target.store.prefix, target.object_id)
        if value.strip("/")
    )
    return f"s3://{target.store.bucket}/{prefix}/_PLAN.json"


def ordered_binary_object_shard_uri(
    target: OrderedBinaryObjectTarget, index: int
) -> str:
    if index < 0 or index >= target.max_shards:
        raise ValueError("ordered shard index leaves its target bound")
    prefix = "/".join(
        value.strip("/")
        for value in (target.store.prefix, target.object_id)
        if value.strip("/")
    )
    return (
        f"s3://{target.store.bucket}/{_ordered_binary_object_shard_key(prefix, index)}"
    )


def ordered_binary_object_ref(
    target: OrderedBinaryObjectTarget,
    files: tuple[BinaryObjectFile, ...],
) -> OrderedBinaryObjectRef:
    """Derive the only ordered commit accepted for an exact prepared target."""
    return _ordered_binary_object_ref(
        target, _ordered_binary_object_layout(target, files)
    )


def vllm_lora_ordered_target(
    store: S3ObjectStoreConfig,
    *,
    run_id: str,
    training_session_id: str,
    generation_id: str,
    policy_step: int,
) -> OrderedBinaryObjectTarget:
    object_id = hashlib.sha256(f"{run_id}\0{generation_id}".encode()).hexdigest()
    return OrderedBinaryObjectTarget(
        store=store,
        object_id=object_id,
        format=VLLM_LORA_OBJECT_FORMAT,
        shard_bytes=store.multipart_chunk_bytes,
        max_concurrent_shards=min(store.multipart_concurrency, 64),
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


def _ordered_binary_object_plan(
    target: OrderedBinaryObjectTarget,
    files: dict[str, tuple[memoryview, ...]],
    file_sha256: dict[str, str] | None,
) -> OrderedBinaryObjectPlan:
    if file_sha256 is not None and set(files) != set(file_sha256):
        raise ValueError("ordered object digests must cover every file exactly")
    planned_files = tuple(
        BinaryObjectFile(
            relative_path=_safe_relative_path(relative_path),
            byte_count=sum(chunk.nbytes for chunk in chunks),
            sha256=None if file_sha256 is None else file_sha256[relative_path],
        )
        for relative_path, chunks in sorted(files.items())
    )
    return _ordered_binary_object_layout(target, planned_files)


def _ordered_binary_object_layout(
    target: OrderedBinaryObjectTarget,
    files: tuple[BinaryObjectFile, ...],
) -> OrderedBinaryObjectPlan:
    planned_files = tuple(sorted(files, key=lambda file: file.relative_path))
    shards: list[OrderedBinaryObjectShard] = []
    for file in planned_files:
        for offset, byte_count in _multipart_ranges(
            file.byte_count, target.shard_bytes
        ):
            shards.append(
                OrderedBinaryObjectShard(
                    index=len(shards),
                    relative_path=file.relative_path,
                    file_offset=offset,
                    byte_count=byte_count,
                )
            )
    if len(shards) > target.max_shards:
        raise RuntimeError(
            f"ordered object requires {len(shards)} shards; limit is {target.max_shards}"
        )
    plan = OrderedBinaryObjectPlan(
        object_id=target.object_id,
        format=target.format,
        files=planned_files,
        shards=tuple(shards),
        metadata=target.metadata,
    )
    _require_ordered_plan_target(
        target,
        plan,
        expected_byte_count=sum(file.byte_count for file in planned_files),
    )
    return plan


def _require_ordered_plan_target(
    target: OrderedBinaryObjectTarget,
    plan: OrderedBinaryObjectPlan,
    *,
    expected_byte_count: int,
) -> None:
    if (plan.object_id, plan.format, plan.metadata) != (
        target.object_id,
        target.format,
        target.metadata,
    ):
        raise RuntimeError("ordered object plan differs from its prepared target")
    if len(plan.shards) > target.max_shards:
        raise RuntimeError("ordered object plan exceeds its prepared shard bound")
    if any(shard.byte_count > target.shard_bytes for shard in plan.shards):
        raise RuntimeError("ordered object shard exceeds its prepared size bound")
    byte_count = sum(file.byte_count for file in plan.files)
    if byte_count != expected_byte_count:
        raise RuntimeError("ordered object aggregate size changed")
    if byte_count > target.shard_bytes * target.max_shards:
        raise RuntimeError("ordered object exceeds its prepared aggregate bound")


def _ordered_binary_object_ref(
    target: OrderedBinaryObjectTarget,
    plan: OrderedBinaryObjectPlan,
    *,
    plan_body: bytes | None = None,
) -> OrderedBinaryObjectRef:
    plan_body = _model_bytes(plan) if plan_body is None else plan_body
    return OrderedBinaryObjectRef(
        object_id=target.object_id,
        format=target.format,
        plan_uri=ordered_binary_object_plan_uri(target),
        manifest_uri=binary_object_manifest_uri(target),
        plan_sha256=hashlib.sha256(plan_body).hexdigest(),
        byte_count=sum(file.byte_count for file in plan.files),
        files=plan.files,
        shard_count=len(plan.shards),
        metadata=target.metadata,
    )


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


def _binary_object_file_key(prefix: str, file: BinaryObjectFile) -> str:
    if file.sha256 is None:
        raise ValueError("content-addressed object file has no SHA-256")
    return f"{prefix}/files/{file.sha256}"


def _ordered_binary_object_shard_key(prefix: str, index: int) -> str:
    return f"{prefix}/shards/{index:08d}"


def _ordered_binary_object_shard_metadata(
    plan_sha256: str,
    source_sha256: str | None,
    shard: OrderedBinaryObjectShard,
) -> dict[str, str]:
    metadata = {
        "plan-sha256": plan_sha256,
        "file-offset": str(shard.file_offset),
        "byte-count": str(shard.byte_count),
    }
    if source_sha256 is not None:
        metadata["source-sha256"] = source_sha256
    return metadata


def _is_precondition_failed(error: Exception) -> bool:
    from botocore.exceptions import ClientError

    return isinstance(error, ClientError) and error.response.get("Error", {}).get(
        "Code"
    ) in {"PreconditionFailed", "412"}


def _is_missing_object(error: Exception) -> bool:
    from botocore.exceptions import ClientError

    return isinstance(error, ClientError) and error.response.get("Error", {}).get(
        "Code"
    ) in {"404", "NoSuchKey", "NotFound"}


def _manifest_bytes(ref: BinaryObjectRef) -> bytes:
    return _model_bytes(ref)


def _model_bytes(value: BaseModel) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
    ).encode()


def _parse_s3_uri(value: str) -> tuple[str, str]:
    from urllib.parse import urlsplit

    parsed = urlsplit(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid S3 object URI: {value!r}")
    return parsed.netloc, parsed.path.strip("/")
