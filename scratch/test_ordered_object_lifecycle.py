from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from threading import Lock
from typing import cast

from botocore.exceptions import ClientError
import pytest

from art.distributed.object_store import (
    OrderedBinaryObjectTarget,
    S3BinaryObjectStore,
    S3ObjectStoreConfig,
    vllm_lora_ordered_target,
)


class _S3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.etags: dict[str, str] = {}
        self.puts: list[str] = []
        self.gets: list[str] = []
        self.heads: list[str] = []
        self.lock = Lock()

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body,
        ContentType: str | None = None,
        ContentLength: int | None = None,
        IfNoneMatch: str | None = None,
        Metadata: dict[str, str] | None = None,
    ) -> dict[str, str]:
        del Bucket, ContentType
        payload = Body if isinstance(Body, bytes) else Body.read()
        assert ContentLength is None or ContentLength == len(payload)
        with self.lock:
            if IfNoneMatch == "*" and Key in self.objects:
                raise ClientError(
                    {"Error": {"Code": "PreconditionFailed"}}, "PutObject"
                )
            etag = f'"{hashlib.sha256(payload).hexdigest()[:16]}"'
            self.objects[Key] = payload
            self.metadata[Key] = Metadata or {}
            self.etags[Key] = etag
            self.puts.append(Key)
        return {"ETag": etag}

    def get_object(self, *, Bucket: str, Key: str):
        del Bucket
        with self.lock:
            self.gets.append(Key)
            try:
                payload = self.objects[Key]
            except KeyError as error:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey"}}, "GetObject"
                ) from error
            return {
                "Body": BytesIO(payload),
                "ContentLength": len(payload),
                "Metadata": self.metadata[Key],
                "ETag": self.etags[Key],
            }

    def head_object(self, *, Bucket: str, Key: str):
        del Bucket
        with self.lock:
            self.heads.append(Key)
            try:
                payload = self.objects[Key]
            except KeyError as error:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey"}}, "HeadObject"
                ) from error
            return {
                "ContentLength": len(payload),
                "Metadata": self.metadata[Key],
                "ETag": self.etags[Key],
            }

    def delete_objects(self, *, Bucket: str, Delete: dict[str, object]):
        del Bucket
        for value in cast(list[dict[str, str]], Delete["Objects"]):
            key = value["Key"]
            self.objects.pop(key, None)
            self.metadata.pop(key, None)
            self.etags.pop(key, None)
        return {}

    def close(self) -> None:
        pass


def _target() -> OrderedBinaryObjectTarget:
    config = S3ObjectStoreConfig(
        endpoint_url="https://objects.invalid",
        region="test",
        bucket="bucket",
        prefix="training",
        multipart_concurrency=2,
    )
    target = vllm_lora_ordered_target(
        config,
        run_id="run",
        training_session_id="session",
        generation_id="generation",
        policy_step=7,
    )
    return OrderedBinaryObjectTarget.model_validate(
        {**target.model_dump(), "shard_bytes": 4, "max_shards": 8}
    )


def _store(target: OrderedBinaryObjectTarget, client: _S3) -> S3BinaryObjectStore:
    store = S3BinaryObjectStore.__new__(S3BinaryObjectStore)
    store.config = target.store
    store._client = client
    store._parts = ThreadPoolExecutor(max_workers=2)
    store._lock = Lock()
    store._closed = False
    return store


def _source() -> dict[str, tuple[memoryview, ...]]:
    return {
        "adapter_config.json": (memoryview(b"cfg"),),
        "adapter_model.safetensors": (
            memoryview(b"abcd"),
            memoryview(b"efghij"),
        ),
    }


def test_ordered_commit_resolves_verifies_and_deletes_by_discriminator() -> None:
    target, client = _target(), _S3()
    store = _store(target, client)
    files = _source()
    ref = store.publish_ordered(target, files)
    prefix = f"{target.store.prefix}/{target.object_id}"

    assert client.gets == [f"{prefix}/_COMMITTED.json"]
    assert json.loads(client.objects[f"{prefix}/_COMMITTED.json"])["transport"] == (
        "ordered_s3_shards"
    )
    assert client.puts[0] == f"{prefix}/_PLAN.json"
    assert client.puts[-1] == f"{prefix}/_COMMITTED.json"
    assert store.resolve(ref.manifest_uri) == ref
    assert all(file.sha256 is None for file in ref.files)
    assert all(
        "source-sha256" not in client.metadata[key]
        for key in client.metadata
        if "/shards/" in key
    )
    prior_gets = len(client.gets)
    store.verify(ref, target=target, expected_byte_count=ref.byte_count)
    verification_gets = client.gets[prior_gets:]
    assert verification_gets == [
        f"{prefix}/_COMMITTED.json",
        f"{prefix}/_PLAN.json",
    ]
    assert client.heads == [
        f"{prefix}/shards/{index:08d}" for index in range(ref.shard_count)
    ]

    with pytest.raises(RuntimeError, match="aggregate size"):
        store.verify(ref, target=target, expected_byte_count=ref.byte_count - 1)
    smaller_shards = OrderedBinaryObjectTarget.model_validate(
        {**target.model_dump(), "shard_bytes": 3}
    )
    with pytest.raises(RuntimeError, match="shard exceeds"):
        store.verify(
            ref,
            target=smaller_shards,
            expected_byte_count=ref.byte_count,
        )

    store.delete(ref)
    assert not any(key.startswith(f"{prefix}/") for key in client.objects)
    assert store.resolve(ref.manifest_uri, missing_ok=True) is None
    store.close()


def test_ordered_fenced_cleanup_reclaims_partial_plan_and_shards() -> None:
    target, client = _target(), _S3()
    store = _store(target, client)
    files = _source()
    ref = store.publish_ordered(target, files)
    prefix = f"{target.store.prefix}/{target.object_id}"

    with pytest.raises(RuntimeError, match="committed"):
        store.discard_uncommitted(target, ref.files)
    client.objects.pop(f"{prefix}/_COMMITTED.json")
    client.metadata.pop(f"{prefix}/_COMMITTED.json")
    client.etags.pop(f"{prefix}/_COMMITTED.json")
    client.objects.pop(f"{prefix}/shards/{ref.shard_count - 1:08d}")
    client.objects["unrelated/object"] = b"keep"

    store.discard_uncommitted(target, ref.files)

    assert not any(key.startswith(f"{prefix}/") for key in client.objects)
    assert client.objects["unrelated/object"] == b"keep"
    store.close()
