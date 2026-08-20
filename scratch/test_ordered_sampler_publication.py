from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
from threading import Barrier, Lock

from botocore.exceptions import ClientError

from art.distributed.object_store import (
    OrderedBinaryObjectPlan,
    OrderedBinaryObjectTarget,
    S3BinaryObjectStore,
    S3ObjectStoreConfig,
    vllm_lora_ordered_target,
)


class _S3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.events: list[str] = []
        self.lock = Lock()
        self.first_uploads = Barrier(2)
        self.shard_starts = 0
        self.active = 0
        self.max_active = 0

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
        if IfNoneMatch == "*" and Key in self.objects:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed"}}, "PutObject"
            )
        shard = "/shards/" in Key
        if shard:
            with self.lock:
                self.shard_starts += 1
                wait_for_peer = self.shard_starts <= 2
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.events.append(f"start:{Key}")
            if wait_for_peer:
                self.first_uploads.wait(timeout=2)
        payload = Body if isinstance(Body, bytes) else Body.read()
        assert ContentLength is None or ContentLength == len(payload)
        with self.lock:
            self.objects[Key] = payload
            self.metadata[Key] = Metadata or {}
            self.events.append(f"finish:{Key}")
            if shard:
                self.active -= 1
        return {"ETag": f'"{len(payload):x}"'}

    def head_object(self, *, Bucket: str, Key: str):
        del Bucket
        return {
            "ContentLength": len(self.objects[Key]),
            "Metadata": self.metadata[Key],
            "ETag": f'"{len(self.objects[Key]):x}"',
        }

    def get_object(self, *, Bucket: str, Key: str):
        del Bucket
        try:
            payload = self.objects[Key]
        except KeyError as error:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}}, "GetObject"
            ) from error
        return {"Body": BytesIO(payload)}

    def close(self) -> None:
        pass


def _store(config: S3ObjectStoreConfig, client: _S3) -> S3BinaryObjectStore:
    store = S3BinaryObjectStore.__new__(S3BinaryObjectStore)
    store.config = config
    store._client = client
    store._parts = ThreadPoolExecutor(max_workers=2)
    store._lock = Lock()
    store._closed = False
    return store


def test_sampler_publishes_direct_bounded_independently_visible_shards() -> None:
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
    target = OrderedBinaryObjectTarget.model_validate(
        {**target.model_dump(), "shard_bytes": 4}
    )
    client = _S3()
    store = _store(config, client)
    source = {
        "adapter_config.json": (memoryview(b"cfg"),),
        "adapter_model.safetensors": (
            memoryview(b"abcd"),
            memoryview(b"efghij"),
        ),
    }
    ref = store.publish_ordered(
        target,
        source,
        file_sha256={
            name: hashlib.sha256(b"".join(chunks)).hexdigest()
            for name, chunks in source.items()
        },
    )

    prefix = f"{config.prefix}/{target.object_id}"
    plan = OrderedBinaryObjectPlan.model_validate_json(
        client.objects[f"{prefix}/_PLAN.json"]
    )
    reconstructed = {name: bytearray() for name in source}
    for shard in plan.shards:
        reconstructed[shard.relative_path].extend(
            client.objects[f"{prefix}/shards/{shard.index:08d}"]
        )
    assert {
        name: bytes(value) for name, value in reconstructed.items()
    } == {name: b"".join(chunks) for name, chunks in source.items()}
    assert client.max_active == target.max_concurrent_shards
    assert client.events[0] == f"finish:{prefix}/_PLAN.json"
    assert client.events[-1] == f"finish:{prefix}/_COMMITTED.json"
    assert not any("/files/" in key for key in client.objects)
    assert store.resolve_ordered(ref.manifest_uri) == ref
    store.close()
