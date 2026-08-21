from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
from threading import Lock
import time
from typing import cast

from botocore.exceptions import ClientError
import pytest

from art.distributed.object_store import (
    VLLM_LORA_OBJECT_FORMAT,
    BinaryObjectFile,
    OrderedBinaryObjectAborted,
    OrderedBinaryObjectPlan,
    OrderedBinaryObjectShard,
    OrderedBinaryObjectTarget,
    S3BinaryObjectReceiver,
    S3BinaryObjectStore,
    S3ObjectStoreConfig,
    ordered_binary_object_plan,
    vllm_lora_ordered_target,
)
from art.megatron.optimizer_state import (
    CheckpointFile,
    acknowledge_materialized_adapter,
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


class _CommitChangingS3(_S3):
    def __init__(self) -> None:
        super().__init__()
        self.changed = False

    def get_object(self, *, Bucket: str, Key: str):
        response = super().get_object(Bucket=Bucket, Key=Key)
        if "/shards/" in Key:
            with self.lock:
                if not self.changed:
                    commit_key = next(
                        key for key in self.objects if key.endswith("/_COMMITTED.json")
                    )
                    commit = json.loads(self.objects[commit_key])
                    commit["shards"][0]["etag"] = '"superseded"'
                    self.objects[commit_key] = json.dumps(
                        commit, separators=(",", ":"), sort_keys=True
                    ).encode()
                    self.changed = True
        return response


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
        generation_id="step-00000007-0123456789abcdef0123456789abcdef",
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


def _receiver(
    target: OrderedBinaryObjectTarget,
    store: S3BinaryObjectStore,
    receive_root: Path,
) -> S3BinaryObjectReceiver:
    receive_root.mkdir()
    receiver = S3BinaryObjectReceiver.__new__(S3BinaryObjectReceiver)
    receiver.config = target.store
    receiver.receive_root = receive_root
    receiver._store = store
    receiver._active = set()
    receiver._lock = Lock()
    receiver._closed = False
    return receiver


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

    assert client.gets == []
    assert json.loads(client.objects[f"{prefix}/_COMMITTED.json"])["transport"] == (
        "ordered_s3_shards"
    )
    assert client.puts[0] == f"{prefix}/_PLAN.json"
    assert client.puts[-1] == f"{prefix}/_COMMITTED.json"
    assert store.resolve(ref.manifest_uri) == ref
    assert client.gets == [f"{prefix}/_COMMITTED.json"]
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


def test_distributed_ordered_upload_uses_rank_local_payload_offsets() -> None:
    target, client = _target(), _S3()
    store = _store(target, client)
    plan = OrderedBinaryObjectPlan(
        object_id=target.object_id,
        format=target.format,
        files=(
            BinaryObjectFile(relative_path="adapter_config.json", byte_count=3),
            BinaryObjectFile(relative_path="adapter_model.safetensors", byte_count=10),
        ),
        shards=(
            OrderedBinaryObjectShard(
                index=0,
                relative_path="adapter_config.json",
                file_offset=0,
                byte_count=3,
            ),
            OrderedBinaryObjectShard(
                index=1,
                relative_path="adapter_model.safetensors",
                file_offset=0,
                byte_count=4,
            ),
            OrderedBinaryObjectShard(
                index=2,
                relative_path="adapter_model.safetensors",
                file_offset=4,
                byte_count=4,
            ),
            OrderedBinaryObjectShard(
                index=3,
                relative_path="adapter_model.safetensors",
                file_offset=8,
                byte_count=2,
            ),
        ),
        metadata=target.metadata,
    )

    expected = store.publish_ordered_plan(target, plan)
    first = store.upload_ordered_shards(
        target,
        plan,
        {0: (memoryview(b"cfg"),), 2: (memoryview(b"efgh"),)},
    )
    second = store.upload_ordered_shards(
        target,
        plan,
        {1: (memoryview(b"abcd"),), 3: (memoryview(b"ij"),)},
    )
    assert store.commit_ordered(target, plan, (*first, *second)) == expected
    prefix = f"{target.store.prefix}/{target.object_id}/shards"
    assert [client.objects[f"{prefix}/{index:08d}"] for index in range(4)] == [
        b"cfg",
        b"abcd",
        b"efgh",
        b"ij",
    ]
    store.close()


def test_ordered_retry_reuses_the_exact_immutable_object() -> None:
    target, client = _target(), _S3()
    store = _store(target, client)
    files = _source()

    expected = store.publish_ordered(target, files)
    actual = store.publish_ordered(target, files)

    assert actual == expected
    assert client.gets == [
        f"{target.store.prefix}/{target.object_id}/_PLAN.json",
        f"{target.store.prefix}/{target.object_id}/_COMMITTED.json",
    ]
    store.close()


def test_pending_ordered_receive_overlaps_upload_and_resumes_shards(
    tmp_path: Path,
) -> None:
    target, client = _target(), _S3()
    publisher, receiver = _store(target, client), _store(target, client)
    files = _source()
    plan = ordered_binary_object_plan(
        target,
        tuple(
            BinaryObjectFile(
                relative_path=path, byte_count=sum(x.nbytes for x in chunks)
            )
            for path, chunks in sorted(files.items())
        ),
    )
    ref = publisher.publish_ordered_plan(target, plan)
    payload = {
        path: b"".join(bytes(chunk) for chunk in chunks)
        for path, chunks in files.items()
    }
    shards = {
        shard.index: (
            memoryview(payload[shard.relative_path])[
                shard.file_offset : shard.file_offset + shard.byte_count
            ],
        )
        for shard in plan.shards
    }
    destination = tmp_path / "receive"
    with ThreadPoolExecutor(max_workers=1) as tasks:
        pending = tasks.submit(
            receiver.materialize_ordered_pending,
            ref,
            destination,
            timeout_s=2.0,
            poll_interval_s=0.001,
            max_poll_interval_s=0.005,
        )
        deadline = time.monotonic() + 1
        shard_prefix = f"{target.store.prefix}/{target.object_id}/shards/"
        while not any(key.startswith(shard_prefix) for key in client.gets):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        first = publisher.upload_ordered_shards(target, plan, {0: shards[0]})
        rest = publisher.upload_ordered_shards(
            target, plan, {index: shards[index] for index in range(1, len(shards))}
        )
        assert not pending.done()
        publisher.commit_ordered(target, plan, (*first, *rest))
        assert pending.result(timeout=1) == destination
    assert (destination / "adapter_config.json").read_bytes() == b"cfg"
    assert (destination / "adapter_model.safetensors").read_bytes() == b"abcdefghij"
    assert not (destination / ".ordered-shards").exists()
    publisher.close()
    receiver.close()


def test_pending_ordered_receive_reuses_fsynced_shards(tmp_path: Path) -> None:
    target, client = _target(), _S3()
    publisher, receiver = _store(target, client), _store(target, client)
    files = _source()
    plan = ordered_binary_object_plan(
        target,
        tuple(
            BinaryObjectFile(
                relative_path=path, byte_count=sum(x.nbytes for x in chunks)
            )
            for path, chunks in sorted(files.items())
        ),
    )
    ref = publisher.publish_ordered_plan(target, plan)
    payload = {
        path: b"".join(bytes(chunk) for chunk in chunks)
        for path, chunks in files.items()
    }
    shards = {
        shard.index: (
            memoryview(payload[shard.relative_path])[
                shard.file_offset : shard.file_offset + shard.byte_count
            ],
        )
        for shard in plan.shards
    }
    first = publisher.upload_ordered_shards(target, plan, {0: shards[0]})
    destination = tmp_path / "receive"
    with pytest.raises(TimeoutError):
        receiver.materialize_ordered_pending(
            ref,
            destination,
            timeout_s=0.03,
            poll_interval_s=0.001,
            max_poll_interval_s=0.005,
        )
    assert (destination / ".ordered-shards/00000000.json").is_file()
    rest = publisher.upload_ordered_shards(
        target, plan, {index: shards[index] for index in range(1, len(shards))}
    )
    publisher.commit_ordered(target, plan, (*first, *rest))
    client.gets.clear()
    receiver.materialize_ordered_pending(
        ref,
        destination,
        timeout_s=1,
        poll_interval_s=0.001,
        max_poll_interval_s=0.005,
    )
    first_key = f"{target.store.prefix}/{target.object_id}/shards/00000000"
    assert first_key not in client.gets
    assert (destination / "adapter_model.safetensors").read_bytes() == b"abcdefghij"
    publisher.close()
    receiver.close()


def test_pending_receive_does_not_block_ready_later_shards(tmp_path: Path) -> None:
    target, client = _target(), _S3()
    publisher, receiver = _store(target, client), _store(target, client)
    files = _source()
    plan = ordered_binary_object_plan(
        target,
        tuple(
            BinaryObjectFile(
                relative_path=path, byte_count=sum(x.nbytes for x in chunks)
            )
            for path, chunks in sorted(files.items())
        ),
    )
    ref = publisher.publish_ordered_plan(target, plan)
    payload = {
        path: b"".join(bytes(chunk) for chunk in chunks)
        for path, chunks in files.items()
    }
    shards = {
        shard.index: (
            memoryview(payload[shard.relative_path])[
                shard.file_offset : shard.file_offset + shard.byte_count
            ],
        )
        for shard in plan.shards
    }
    destination = tmp_path / "out-of-order"

    with ThreadPoolExecutor(max_workers=1) as tasks:
        pending = tasks.submit(
            receiver.materialize_ordered_pending,
            ref,
            destination,
            timeout_s=2,
            poll_interval_s=0.001,
            max_poll_interval_s=0.005,
        )
        later = publisher.upload_ordered_shards(
            target, plan, {2: shards[2], 3: shards[3]}
        )
        receipts = destination / ".ordered-shards"
        deadline = time.monotonic() + 1
        while not all((receipts / f"{index:08d}.json").is_file() for index in (2, 3)):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert not any((receipts / f"{index:08d}.json").exists() for index in (0, 1))
        earlier = publisher.upload_ordered_shards(
            target, plan, {0: shards[0], 1: shards[1]}
        )
        publisher.commit_ordered(target, plan, (*earlier, *later))
        assert pending.result(timeout=1) == destination

    publisher.close()
    receiver.close()


def test_ordered_abort_interrupts_streaming_receive_and_reclaims_shards(
    tmp_path: Path,
) -> None:
    target, client = _target(), _S3()
    publisher, receiver = _store(target, client), _store(target, client)
    files = _source()
    plan = ordered_binary_object_plan(
        target,
        tuple(
            BinaryObjectFile(
                relative_path=path, byte_count=sum(x.nbytes for x in chunks)
            )
            for path, chunks in sorted(files.items())
        ),
    )
    ref = publisher.publish_ordered_plan(target, plan)
    destination = tmp_path / "aborted"

    with ThreadPoolExecutor(max_workers=1) as tasks:
        pending = tasks.submit(
            receiver.materialize_ordered_pending,
            ref,
            destination,
            timeout_s=30,
            poll_interval_s=0.001,
            max_poll_interval_s=0.005,
        )
        shard_prefix = f"{target.store.prefix}/{target.object_id}/shards/"
        deadline = time.monotonic() + 1
        while not any(key.startswith(shard_prefix) for key in client.gets):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        started = time.monotonic()
        assert publisher.abort_ordered(ref)
        with pytest.raises(OrderedBinaryObjectAborted):
            pending.result(timeout=1)
        assert time.monotonic() - started < 0.25

    prefix = f"{target.store.prefix}/{target.object_id}"
    assert json.loads(client.objects[f"{prefix}/_COMMITTED.json"])[
        "disposition"
    ] == "aborted"
    assert not any("/shards/" in key for key in client.objects)
    publisher.close()
    receiver.close()


def test_ordered_commit_and_abort_share_one_atomic_terminal() -> None:
    target, client = _target(), _S3()
    store = _store(target, client)
    ref = store.publish_ordered(target, _source())

    assert not store.abort_ordered(ref)
    assert store.resolve(ref.manifest_uri) == ref
    assert json.loads(
        client.objects[f"{target.store.prefix}/{target.object_id}/_COMMITTED.json"]
    )["disposition"] == "committed"
    assert any("/shards/" in key for key in client.objects)
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


def test_ordered_receiver_materializes_later_adapter_only_restore(
    tmp_path: Path,
) -> None:
    target, client = _target(), _S3()
    store = _store(target, client)
    source = _source()
    ref = store.publish_ordered(target, source)
    receive_root = (tmp_path / "receives").resolve()
    receiver = _receiver(target, store, receive_root)

    materialization = receiver.materialize(
        ref.manifest_uri,
        expected_format=VLLM_LORA_OBJECT_FORMAT,
        expected_metadata=target.metadata,
    )

    path = Path(materialization.path)
    assert materialization.ref == ref
    assert (path / "adapter_config.json").read_bytes() == b"cfg"
    assert (path / "adapter_model.safetensors").read_bytes() == b"abcdefghij"
    assert all(file.sha256 is None for file in materialization.ref.files)
    prefix = f"{target.store.prefix}/{target.object_id}"
    assert {
        f"{prefix}/shards/{index:08d}" for index in range(ref.shard_count)
    }.issubset(client.gets)
    files = {file.relative_path: file for file in materialization.ref.files}
    adapter = acknowledge_materialized_adapter(
        path,
        step=7,
        training_session_id=target.metadata["training_session_id"],
        generation_id=target.metadata["generation_id"],
        files=tuple(
            CheckpointFile(
                name=name,
                size_bytes=files[name].byte_count,
                sha256=files[name].sha256,
            )
            for name in ("adapter_config.json", "adapter_model.safetensors")
        ),
    )
    assert all(file.sha256 is None for file in adapter.files)

    receiver.release(materialization)
    assert not path.exists()
    receiver.close()


def test_ordered_receiver_discards_materialization_if_commit_changes(
    tmp_path: Path,
) -> None:
    target, client = _target(), _CommitChangingS3()
    store = _store(target, client)
    ref = store.publish_ordered(target, _source())
    receive_root = (tmp_path / "receives").resolve()
    receiver = _receiver(target, store, receive_root)

    with pytest.raises(RuntimeError, match="commit changed during materialization"):
        receiver.materialize(
            ref.manifest_uri,
            expected_format=VLLM_LORA_OBJECT_FORMAT,
            expected_metadata=target.metadata,
        )

    assert not tuple(receive_root.iterdir())
    receiver.close()
