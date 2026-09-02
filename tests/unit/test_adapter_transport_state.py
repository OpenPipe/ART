import asyncio
import base64
import hashlib
import json
from threading import Lock
from types import SimpleNamespace

import httpx
import pytest

from art.distributed import adapter_transport
from art.distributed.adapter_transport import (
    AdapterReceiveResult,
    AdapterSnapshotReceiver,
    AdapterSnapshotSender,
    AdapterTransferTarget,
    ExternalAdapterCommit,
    ExternalAdapterObjectSource,
    ExternalAdapterShard,
    ExternalAdapterShardedSource,
    NixlAdapterSender,
)
from art.distributed.vllm_replica import ReplicaManager


def test_adapter_transport_state_is_bounded_and_survives_close(
    tmp_path, monkeypatch
) -> None:
    receiver = AdapterSnapshotReceiver("inference-0", str(tmp_path), pool_capacity=2)
    assert receiver.state().model_dump() == {
        "schema_version": 1,
        "host_id": "inference-0",
        "pool_capacity": 2,
        "registered_slots": 0,
        "registered_capacity_bytes": 0,
        "active_registered_slots": 0,
        "pending_nixl_receives": 0,
        "pending_local_receives": 0,
        "pending_object_receives": 0,
        "materialized_generations": 0,
        "pending_notifications": 0,
        "closed": False,
    }
    receiver.close()
    assert receiver.state().closed is True

    sender = AdapterSnapshotSender()
    monkeypatch.setattr(sender, "_send_local", lambda *_args, **_kwargs: None)
    target = AdapterTransferTarget(
        transport="local",
        host_id="inference-0",
        generation_id="generation-1",
        path="/dev/shm/generation-1",
        remote_agent="/tmp/adapter.sock",
        remote_metadata_b64="-",
        remote_address=0,
        remote_device_id=0,
        slot_id=0,
        capacity_bytes=1,
        prepare_s=0,
        pool_wait_s=0,
        registration_s=0,
    )
    sender.send(
        SimpleNamespace(adapter_config={}),
        (target,),
        prepared_tensors=SimpleNamespace(),
    )
    assert sender.state().model_dump() == {
        "schema_version": 1,
        "transport": "local",
        "active_transfers": 0,
        "completed_transfers": 1,
        "registered_buffers": 0,
        "registered_capacity_bytes": 0,
        "remote_agents": 0,
        "poisoned": False,
        "unreleased_handles": 0,
        "closed": False,
    }
    sender.close()
    assert sender.state().closed is True


def _external_object_source(payload: bytes) -> ExternalAdapterObjectSource:
    config = json.dumps(
        {
            "art_lora_format": "vllm",
            "r": 1,
            "target_modules": ["q_proj"],
        },
        separators=(",", ":"),
    )
    return ExternalAdapterObjectSource(
        generation_id="generation-object",
        source_identity="sha256:" + hashlib.sha256(payload).hexdigest(),
        object_url="https://objects.example/adapter_model.safetensors?signature=secret",
        object_bytes=len(payload),
        object_sha256=hashlib.sha256(payload).hexdigest(),
        adapter_config_json=config,
        adapter_config_sha256=hashlib.sha256(config.encode()).hexdigest(),
        lora_rank=1,
        target_modules=("q_proj",),
    )


def _safetensors_payload() -> bytes:
    header = json.dumps(
        {
            "base_model.model.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    return len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00"


def test_completed_external_object_materializes_into_existing_receiver(
    tmp_path, monkeypatch
) -> None:
    payload = _safetensors_payload()
    source = _external_object_source(payload)
    real_client = httpx.Client

    def client(**kwargs):
        return real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=payload, request=request)
            ),
            **kwargs,
        )

    monkeypatch.setattr(adapter_transport.httpx, "Client", client)
    receiver = AdapterSnapshotReceiver("inference-0", str(tmp_path), pool_capacity=2)

    result = receiver.materialize_object(source)

    assert result.source_identity == source.source_identity
    assert result.tensor_bytes == len(payload)
    assert (
        tmp_path
        / "adapter_transfers"
        / source.generation_id
        / "adapter_model.safetensors"
    ).read_bytes() == payload
    assert receiver.state().materialized_generations == 1
    receiver.release(source.generation_id)
    assert receiver.state().materialized_generations == 0


def test_completed_external_object_retries_transient_get(tmp_path, monkeypatch) -> None:
    payload = _safetensors_payload()
    source = _external_object_source(payload)
    real_client = httpx.Client
    attempts = 0

    def client(**kwargs):
        def response(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                503 if attempts == 1 else 200,
                content=b"" if attempts == 1 else payload,
                request=request,
            )

        return real_client(transport=httpx.MockTransport(response), **kwargs)

    monkeypatch.setattr(adapter_transport.httpx, "Client", client)
    receiver = AdapterSnapshotReceiver("inference-0", str(tmp_path), pool_capacity=2)

    result = receiver.materialize_object(source, timeout_s=1)

    assert result.tensor_bytes == len(payload)
    assert attempts == 2


def test_external_object_digest_failure_leaves_no_materialized_state(
    tmp_path, monkeypatch
) -> None:
    payload = _safetensors_payload()
    source = _external_object_source(payload).model_copy(
        update={"object_sha256": "0" * 64}
    )
    real_client = httpx.Client

    def client(**kwargs):
        return real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=payload, request=request)
            ),
            **kwargs,
        )

    monkeypatch.setattr(adapter_transport.httpx, "Client", client)
    receiver = AdapterSnapshotReceiver("inference-0", str(tmp_path), pool_capacity=2)

    with pytest.raises(RuntimeError, match="digest changed"):
        receiver.materialize_object(source)

    assert receiver.state().pending_object_receives == 0
    assert receiver.state().materialized_generations == 0
    assert tuple((tmp_path / "adapter_transfers").iterdir()) == ()


def test_committed_external_shards_reconstruct_standard_adapter(
    tmp_path, monkeypatch
) -> None:
    model = _safetensors_payload()
    config = json.dumps(
        {
            "art_lora_format": "vllm",
            "r": 1,
            "target_modules": ["q_proj"],
        },
        separators=(",", ":"),
    ).encode()
    payloads = (config, model[:17], model[17:])
    paths = ("adapter_config.json",) + ("adapter_model.safetensors",) * 2
    offsets = (0, 0, 17)
    source = ExternalAdapterShardedSource(
        generation_id="generation-sharded",
        source_identity="manifest:" + hashlib.sha256(b"commit").hexdigest(),
        model_bytes=len(model),
        config_bytes=len(config),
        shards=tuple(
            ExternalAdapterShard(
                index=index,
                relative_path=paths[index],
                file_offset=offsets[index],
                object_url=f"https://objects.example/shard-{index}?signature=secret",
                object_bytes=len(payload),
                object_sha256=hashlib.sha256(payload).hexdigest(),
            )
            for index, payload in enumerate(payloads)
        ),
    )
    real_client = httpx.Client

    def client(**kwargs):
        def response(request):
            index = int(request.url.path.rsplit("-", 1)[1])
            return httpx.Response(200, content=payloads[index], request=request)

        return real_client(transport=httpx.MockTransport(response), **kwargs)

    monkeypatch.setattr(adapter_transport.httpx, "Client", client)
    receiver = AdapterSnapshotReceiver("inference-0", str(tmp_path), pool_capacity=2)

    result = receiver.materialize_object(source)
    root = tmp_path / "adapter_transfers" / source.generation_id

    assert result.source_identity == source.source_identity
    assert result.tensor_bytes == len(model)
    assert result.config_bytes == len(config)
    assert (root / "adapter_config.json").read_bytes() == config
    assert (root / "adapter_model.safetensors").read_bytes() == model
    assert receiver.materialize_object(source) == result
    with pytest.raises(RuntimeError, match="identity changed"):
        receiver.materialize_object(
            source.model_copy(update={"source_identity": "manifest:" + "f" * 64})
        )


def test_streaming_shards_overlap_upload_and_require_final_commit(
    tmp_path, monkeypatch
) -> None:
    model = _safetensors_payload()
    config = json.dumps(
        {"art_lora_format": "vllm", "r": 1, "target_modules": ["q_proj"]},
        separators=(",", ":"),
    ).encode()
    payloads = (config, model[:17], model[17:])
    paths = ("adapter_config.json",) + ("adapter_model.safetensors",) * 2
    offsets = (0, 0, 17)
    fields = {
        "generation_id": "generation-streaming",
        "source_identity": "stream:" + hashlib.sha256(model).hexdigest(),
        "model_bytes": len(model),
        "config_bytes": len(config),
        "shards": tuple(
            ExternalAdapterShard(
                index=index,
                relative_path=paths[index],
                file_offset=offsets[index],
                object_url=f"https://objects.example/shard-{index}?signature=secret",
                object_bytes=len(payload),
                object_sha256=hashlib.sha256(payload).hexdigest(),
            )
            for index, payload in enumerate(payloads)
        ),
        "max_parallel_downloads": 4,
    }
    draft = ExternalAdapterShardedSource(**fields)
    plan_sha256 = hashlib.sha256(
        adapter_transport._external_shard_plan(draft)
    ).hexdigest()
    commit_payload = json.dumps(
        {
            "format": "art_external_adapter_commit_v1",
            "generation_id": fields["generation_id"],
            "plan_sha256": plan_sha256,
            "source_identity": fields["source_identity"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    source = ExternalAdapterShardedSource(
        **fields,
        plan_sha256=plan_sha256,
        commit=ExternalAdapterCommit(
            object_url="https://objects.example/commit?signature=secret",
            object_bytes=len(commit_payload),
            object_sha256=hashlib.sha256(commit_payload).hexdigest(),
        ),
    )
    real_client = httpx.Client
    lock = Lock()
    shard_attempts = 0
    commit_attempts_before_last_shard = 0

    def client(**kwargs):
        def response(request):
            nonlocal shard_attempts, commit_attempts_before_last_shard
            path = request.url.path
            with lock:
                if path.endswith("commit"):
                    if shard_attempts < 2:
                        commit_attempts_before_last_shard += 1
                        return httpx.Response(404, request=request)
                    return httpx.Response(200, content=commit_payload, request=request)
                index = int(path.rsplit("-", 1)[1])
                if index == 2:
                    shard_attempts += 1
                    if shard_attempts < 2:
                        return httpx.Response(404, request=request)
                return httpx.Response(200, content=payloads[index], request=request)

        return real_client(transport=httpx.MockTransport(response), **kwargs)

    monkeypatch.setattr(adapter_transport.httpx, "Client", client)
    receiver = AdapterSnapshotReceiver("inference-0", str(tmp_path), pool_capacity=2)

    result = receiver.materialize_object(source, timeout_s=2)

    root = tmp_path / "adapter_transfers" / source.generation_id
    assert result.tensor_bytes == len(model)
    assert (root / "adapter_model.safetensors").read_bytes() == model
    assert shard_attempts == 2
    assert commit_attempts_before_last_shard >= 1


def test_external_materialization_failure_settles_and_releases_every_host() -> None:
    config = json.dumps(
        {"art_lora_format": "vllm", "r": 1, "target_modules": ["q_proj"]},
        separators=(",", ":"),
    ).encode()
    source = ExternalAdapterObjectSource(
        generation_id="generation-failed-gang",
        source_identity="sha256:" + "a" * 64,
        object_url="https://objects.example/adapter?signature=secret",
        object_bytes=4096,
        object_sha256="b" * 64,
        adapter_config_json=config.decode(),
        adapter_config_sha256=hashlib.sha256(config).hexdigest(),
        lora_rank=1,
        target_modules=("q_proj",),
    )

    class Launcher:
        def __init__(self, host_id: str, *, fail_once: bool) -> None:
            self.host_id = host_id
            self.fail_once = fail_once
            self.attempts = 0
            self.completed = False
            self.retained = False
            self.released: list[str] = []

        async def materialize_adapter_object(self, value, timeout_s):
            self.attempts += 1
            self.completed = False
            try:
                await asyncio.sleep(0 if self.fail_once else 0.01)
                assert value is source
                if self.retained:
                    raise RuntimeError("retry conflicted with retained receive")
                if self.fail_once and self.attempts == 1:
                    raise RuntimeError("host materialization failed")
                self.retained = True
                return AdapterReceiveResult(
                    host_id=self.host_id,
                    generation_id=source.generation_id,
                    path=f"/adapter/{source.generation_id}",
                    tensor_bytes=source.object_bytes,
                    config_bytes=len(config),
                    materialization_s=0.1,
                    used_bytes=source.object_bytes + len(config),
                    capacity_bytes=source.object_bytes + len(config),
                    source_identity=source.source_identity,
                )
            finally:
                self.completed = True

        async def release_adapter_receive(self, generation_id):
            assert self.completed
            self.retained = False
            self.released.append(generation_id)

    async def exercise() -> tuple[Launcher, Launcher, tuple[AdapterReceiveResult, ...]]:
        first = Launcher("inference-0", fail_once=False)
        second = Launcher("inference-1", fail_once=True)
        manager = object.__new__(ReplicaManager)
        manager._host_launchers = (first, second)
        manager._rpc_timeout_s = 1.0
        with pytest.raises(BaseExceptionGroup, match="materialization failed"):
            await manager.materialize_external_adapter(source, timeout_s=1.0)
        retried = await manager.materialize_external_adapter(source, timeout_s=1.0)
        return first, second, retried

    first, second, retried = asyncio.run(exercise())
    assert first.released == [source.generation_id]
    assert second.released == [source.generation_id]
    assert tuple(item.host_id for item in retried) == ("inference-0", "inference-1")


class _Handle:
    def __init__(self, *, release_error: bool = False) -> None:
        self.release_error = release_error

    def release(self) -> None:
        if self.release_error:
            raise RuntimeError("transfer remains active")


class _Agent:
    name = "sender"

    def __init__(self, *, state: str, release_error: bool = False) -> None:
        self.state = state
        self.release_error = release_error
        self.removed: list[str] = []

    def get_xfer_descs(self, value, **_kwargs):
        return value

    def add_remote_agent(self, metadata: bytes) -> str:
        return metadata.decode()

    def remove_remote_agent(self, remote_agent: str) -> None:
        self.removed.append(remote_agent)

    def deregister_memory(self, _registration, **_kwargs) -> None:
        return None

    def initialize_xfer(self, *_args, **_kwargs):
        return _Handle(release_error=self.release_error)

    def transfer(self, _handle) -> str:
        return self.state

    def check_xfer_state(self, _handle) -> str:
        return self.state


def _nixl_target(*, epoch: str, timeout_s: float = 1) -> AdapterTransferTarget:
    return AdapterTransferTarget(
        host_id="inference-0",
        generation_id="generation-1",
        path="/adapter/generation-1",
        remote_agent=epoch,
        remote_metadata_b64=base64.b64encode(epoch.encode()).decode(),
        remote_address=1,
        remote_device_id=0,
        slot_id=0,
        capacity_bytes=1,
        prepare_s=0,
        pool_wait_s=0,
        registration_s=0,
        transfer_timeout_s=timeout_s,
    )


def _sender(agent: _Agent, monkeypatch) -> NixlAdapterSender:
    sender = NixlAdapterSender()
    sender._agent = agent
    sender._block = SimpleNamespace(
        numel=lambda: 1,
        narrow=lambda *_args: object(),
    )
    sender._registration = object()
    monkeypatch.setattr(sender, "_ensure_capacity", lambda _used_bytes: 0)
    monkeypatch.setattr(adapter_transport, "_copy_payload", lambda *_args: None)
    return sender


def test_nixl_sender_invalidates_restarted_receiver_metadata(monkeypatch) -> None:
    agent = _Agent(state="DONE")
    sender = _sender(agent, monkeypatch)
    payload = SimpleNamespace(nbytes=1)

    sender.send(payload, {}, (_nixl_target(epoch="receiver-1"),))
    sender.send(payload, {}, (_nixl_target(epoch="receiver-2"),))

    assert agent.removed == ["receiver-1"]
    assert sender.state().remote_agents == 1
    assert sender.state().completed_transfers == 2


def test_nixl_sender_never_reuses_buffer_after_failed_cancel(monkeypatch) -> None:
    sender = _sender(_Agent(state="PROC", release_error=True), monkeypatch)
    target = _nixl_target(epoch="receiver-1", timeout_s=0.000001)

    with pytest.raises(TimeoutError) as caught:
        sender.send(SimpleNamespace(nbytes=1), {}, (target,))

    assert any("sender is poisoned" in note for note in caught.value.__notes__)
    assert sender.state().poisoned is True
    assert sender.state().unreleased_handles == 1
    with pytest.raises(RuntimeError, match="requires a runtime restart"):
        sender.send(SimpleNamespace(nbytes=1), {}, (target,))

    sender._unreleased_handles[0].release_error = False
    sender.close()
    assert sender.state().closed is True
    assert sender.state().unreleased_handles == 0
