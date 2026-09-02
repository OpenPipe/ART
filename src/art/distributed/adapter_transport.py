from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
from threading import Condition, Lock
import time
from typing import Any, Callable, Literal, TypeAlias
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

from art.utils.safetensors import PreparedSafetensors, save_prepared_safetensors


class _TransportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AdapterTransferTarget(_TransportRecord):
    transport: Literal["local", "nixl"] = "nixl"
    host_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    remote_agent: str = Field(min_length=1)
    remote_metadata_b64: str = Field(min_length=1)
    remote_address: int = Field(ge=0)
    remote_device_id: int = Field(ge=0)
    slot_id: int = Field(ge=0)
    capacity_bytes: int = Field(gt=0)
    prepare_s: float = Field(ge=0)
    pool_wait_s: float = Field(ge=0)
    registration_s: float = Field(ge=0)
    transfer_timeout_s: float = Field(default=300.0, gt=0)


class NixlMemorySource(_TransportRecord):
    """One bounded CPU buffer registered for a remote NIXL read."""

    agent: str = Field(min_length=1, max_length=255)
    metadata_b64: str = Field(min_length=1, max_length=1 << 20)
    address: int = Field(gt=0)
    byte_count: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_metadata(self) -> "NixlMemorySource":
        try:
            base64.b64decode(self.metadata_b64, validate=True)
        except ValueError as error:
            raise ValueError("NIXL source metadata is not valid base64") from error
        return self


class AdapterReceiveResult(_TransportRecord):
    host_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    tensor_bytes: int = Field(gt=0)
    config_bytes: int = Field(gt=0)
    materialization_s: float = Field(ge=0)
    slot_id: int = Field(default=0, ge=0)
    used_bytes: int = Field(default=0, ge=0)
    capacity_bytes: int = Field(default=0, ge=0)
    prepare_s: float = Field(default=0, ge=0)
    pool_wait_s: float = Field(default=0, ge=0)
    registration_s: float = Field(default=0, ge=0)
    sender_staging_s: float = Field(default=0, ge=0)
    sender_registration_s: float = Field(default=0, ge=0)
    source_identity: str | None = Field(default=None, min_length=1, max_length=512)


class ExternalAdapterObjectSource(_TransportRecord):
    """One completed immutable LoRA object plus bounded PEFT metadata."""

    generation_id: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    source_identity: str = Field(min_length=1, max_length=512)
    object_url: str = Field(min_length=1, max_length=8192, repr=False)
    object_bytes: int = Field(gt=8, le=8 << 30)
    object_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_config_json: str = Field(min_length=2, max_length=64 << 10)
    adapter_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lora_rank: int = Field(gt=0, le=4096)
    target_modules: tuple[str, ...] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def _validate_source(self) -> "ExternalAdapterObjectSource":
        _validate_exact_object_url(self.object_url)
        config_bytes = self.adapter_config_json.encode()
        if hashlib.sha256(config_bytes).hexdigest() != self.adapter_config_sha256:
            raise ValueError("external adapter config digest changed")
        config = _adapter_config(config_bytes)
        if config["r"] != self.lora_rank:
            raise ValueError("external adapter rank changed")
        if tuple(config["target_modules"]) != self.target_modules:
            raise ValueError("external adapter target modules changed")
        if len(set(self.target_modules)) != len(self.target_modules) or any(
            not module or len(module) > 512 for module in self.target_modules
        ):
            raise ValueError("external adapter target modules are invalid")
        return self


class ExternalAdapterShard(_TransportRecord):
    index: int = Field(ge=0)
    relative_path: Literal["adapter_config.json", "adapter_model.safetensors"]
    file_offset: int = Field(ge=0)
    object_url: str = Field(min_length=1, max_length=8192, repr=False)
    object_bytes: int = Field(gt=0, le=5 << 30)
    object_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_url(self) -> "ExternalAdapterShard":
        _validate_exact_object_url(self.object_url)
        return self


class ExternalAdapterCommit(_TransportRecord):
    """Final small object proving a streaming shard plan is complete."""

    object_url: str = Field(min_length=1, max_length=8192, repr=False)
    object_bytes: int = Field(gt=0, le=1 << 20)
    object_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_url(self) -> "ExternalAdapterCommit":
        _validate_exact_object_url(self.object_url)
        return self


class ExternalAdapterShardedSource(_TransportRecord):
    """Committed immutable shards covering one standard PEFT adapter."""

    generation_id: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    source_identity: str = Field(min_length=1, max_length=512)
    model_bytes: int = Field(gt=8, le=8 << 30)
    config_bytes: int = Field(gt=1, le=64 << 10)
    shards: tuple[ExternalAdapterShard, ...] = Field(min_length=2, max_length=1024)
    max_parallel_downloads: int = Field(default=16, ge=1, le=32)
    plan_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    commit: ExternalAdapterCommit | None = None

    @model_validator(mode="after")
    def _validate_coverage(self) -> "ExternalAdapterShardedSource":
        if tuple(shard.index for shard in self.shards) != tuple(
            range(len(self.shards))
        ):
            raise ValueError("external adapter shard indexes must be contiguous")
        if len({shard.object_url for shard in self.shards}) != len(self.shards):
            raise ValueError("external adapter shard URLs must be unique")
        expected = {
            "adapter_config.json": self.config_bytes,
            "adapter_model.safetensors": self.model_bytes,
        }
        for relative_path, size_bytes in expected.items():
            cursor = 0
            for shard in (
                item for item in self.shards if item.relative_path == relative_path
            ):
                if shard.file_offset != cursor:
                    raise ValueError("external adapter shards leave a file gap")
                cursor += shard.object_bytes
            if cursor != size_bytes:
                raise ValueError("external adapter shards do not cover their file")
        streaming = self.commit is not None
        if streaming != (self.plan_sha256 is not None):
            raise ValueError("streaming external adapter requires plan and commit")
        if streaming:
            assert self.plan_sha256 is not None and self.commit is not None
            plan_sha256 = hashlib.sha256(_external_shard_plan(self)).hexdigest()
            if self.plan_sha256 != plan_sha256:
                raise ValueError("external adapter streaming plan changed")
            commit = _external_shard_commit(self)
            if (
                self.commit.object_bytes != len(commit)
                or self.commit.object_sha256 != hashlib.sha256(commit).hexdigest()
            ):
                raise ValueError("external adapter commit identity changed")
        return self


ExternalAdapterSource: TypeAlias = (
    ExternalAdapterObjectSource | ExternalAdapterShardedSource
)


def _validate_exact_object_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("external adapter object requires an exact HTTPS URL")


def _external_shard_plan(source: ExternalAdapterShardedSource) -> bytes:
    payload = {
        "format": "art_external_adapter_plan_v1",
        "generation_id": source.generation_id,
        "source_identity": source.source_identity,
        "model_bytes": source.model_bytes,
        "config_bytes": source.config_bytes,
        "shards": [
            {
                "index": shard.index,
                "relative_path": shard.relative_path,
                "file_offset": shard.file_offset,
                "object_bytes": shard.object_bytes,
                "object_sha256": shard.object_sha256,
            }
            for shard in source.shards
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _external_shard_commit(source: ExternalAdapterShardedSource) -> bytes:
    assert source.plan_sha256 is not None
    return json.dumps(
        {
            "format": "art_external_adapter_commit_v1",
            "generation_id": source.generation_id,
            "plan_sha256": source.plan_sha256,
            "source_identity": source.source_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


class AdapterReceivePoolState(_TransportRecord):
    """Bounded physical receiver state after one publication boundary."""

    schema_version: Literal[1] = 1
    host_id: str = Field(min_length=1)
    pool_capacity: int = Field(ge=1)
    registered_slots: int = Field(ge=0)
    registered_capacity_bytes: int = Field(ge=0)
    active_registered_slots: int = Field(ge=0)
    pending_nixl_receives: int = Field(ge=0)
    pending_local_receives: int = Field(ge=0)
    pending_object_receives: int = Field(default=0, ge=0)
    materialized_generations: int = Field(ge=0)
    pending_notifications: int = Field(ge=0)
    closed: bool

    @model_validator(mode="after")
    def _validate_bounds(self) -> "AdapterReceivePoolState":
        if (
            self.registered_slots > self.pool_capacity
            or self.active_registered_slots > self.registered_slots
            or self.pending_nixl_receives > self.pool_capacity
            or self.pending_local_receives > self.pool_capacity
            or self.pending_object_receives > self.pool_capacity
        ):
            raise ValueError("adapter receiver state exceeds its configured pool")
        return self


class AdapterSenderState(_TransportRecord):
    """Bounded sender registrations and completed physical transfer lanes."""

    schema_version: Literal[1] = 1
    transport: Literal["idle", "local", "nixl", "mixed"]
    active_transfers: int = Field(ge=0)
    completed_transfers: int = Field(ge=0)
    registered_buffers: int = Field(ge=0, le=1)
    registered_capacity_bytes: int = Field(ge=0)
    remote_agents: int = Field(ge=0)
    poisoned: bool = False
    unreleased_handles: int = Field(default=0, ge=0, le=1)
    closed: bool


class AdapterTransferNotification(_TransportRecord):
    generation_id: str = Field(min_length=1)
    used_bytes: int = Field(gt=0)
    adapter_config: dict[str, Any]
    sender_staging_s: float = Field(ge=0)
    sender_registration_s: float = Field(ge=0)


class _PendingReceive:
    def __init__(
        self,
        *,
        target: AdapterTransferTarget,
        slot: "_RegisteredSlot",
    ) -> None:
        self.target = target
        self.slot = slot


class _PendingLocalReceive:
    def __init__(
        self,
        *,
        target: AdapterTransferTarget,
        listener: socket.socket,
    ) -> None:
        self.target = target
        self.listener = listener


class _RegisteredSlot:
    def __init__(
        self,
        slot_id: int,
        block: torch.Tensor,
        registration: Any,
    ) -> None:
        self.slot_id = slot_id
        self.block = block
        self.registration = registration
        self.generation_id: str | None = None


def _load_nixl() -> tuple[Any, Any, Any]:
    from .nixl_runtime import configure_nixl_environment

    configure_nixl_environment()
    for name in ("nixl_cu13", "nixl_cu12", "nixl"):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        return (
            module.nixl_agent,
            module.nixl_agent_config,
            module.nixl_thread_sync_t,
        )
    raise RuntimeError(
        "NIXL Python bindings are unavailable; install ART with the megatron "
        "or megatron-cu130 extra"
    )


def _new_agent(name: str) -> Any:
    agent_type, config_type, sync_type = _load_nixl()
    return agent_type(
        name,
        config_type(
            enable_prog_thread=True,
            enable_listen_thread=False,
            backends=["UCX"],
            sync_mode=sync_type.NIXL_THREAD_SYNC_STRICT,
        ),
    )


def _run_nixl_transfer(
    agent: Any, handle: Any, *, timeout_s: float, description: str
) -> None:
    state = agent.transfer(handle)
    deadline = time.monotonic() + timeout_s
    while state == "PROC":
        if time.monotonic() >= deadline:
            raise TimeoutError(f"NIXL {description} timed out")
        time.sleep(0.001)
        state = agent.check_xfer_state(handle)
    if state != "DONE":
        raise RuntimeError(f"NIXL {description} failed")


def nixl_read_bytes(
    source: NixlMemorySource, *, transfer_id: str, timeout_s: float
) -> bytearray:
    """Read one registered CPU buffer with the standard NIXL lifecycle."""

    payload = bytearray(source.byte_count)
    block = torch.frombuffer(payload, dtype=torch.uint8)
    agent = _new_agent(
        "art-nixl-reader-"
        + hashlib.sha256(f"{os.getpid()}:{transfer_id}".encode()).hexdigest()[:24]
    )
    registration = agent.register_memory((block,), backends=["UCX"])
    remote_agent = None
    handle = None
    try:
        remote_agent = agent.add_remote_agent(
            base64.b64decode(source.metadata_b64, validate=True)
        )
        if isinstance(remote_agent, bytes):
            remote_agent = remote_agent.decode()
        if remote_agent != source.agent:
            raise RuntimeError("NIXL source returned the wrong agent identity")
        handle = agent.initialize_xfer(
            "READ",
            agent.get_xfer_descs((block,)),
            agent.get_xfer_descs(
                [(source.address, source.byte_count, 0)], mem_type="DRAM"
            ),
            remote_agent,
            backends=["UCX"],
        )
        _run_nixl_transfer(agent, handle, timeout_s=timeout_s, description=transfer_id)
        return payload
    finally:
        if handle is not None:
            handle.release()
        if remote_agent is not None:
            agent.remove_remote_agent(remote_agent)
        agent.deregister_memory(registration, backends=["UCX"])


def _adapter_template_bytes(path: str) -> int:
    root = Path(path)
    model_path = root / "adapter_model.safetensors"
    model_bytes = model_path.stat().st_size
    if model_bytes <= 8:
        raise RuntimeError(f"Adapter template is empty: {path}")
    with (root / "adapter_config.json").open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise RuntimeError(f"Adapter config must be an object: {path}")
    if config.get("art_lora_format") != "vllm":
        raise RuntimeError(f"Adapter template is not in vLLM format: {path}")
    return model_bytes


def _copy_payload(payload: PreparedSafetensors, block: torch.Tensor) -> None:
    offset = 0
    for chunk in payload.chunks:
        block.narrow(0, offset, chunk.numel()).copy_(chunk)
        offset += chunk.numel()
    if offset != payload.nbytes:
        raise RuntimeError("Adapter payload copy was incomplete")


def _adapter_config(payload: bytes) -> dict[str, Any]:
    try:
        config = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("external adapter config is not valid JSON") from error
    if not isinstance(config, dict):
        raise ValueError("external adapter config must be an object")
    rank = config.get("r")
    targets = config.get("target_modules")
    if type(rank) is not int or not 0 < rank <= 4096:
        raise ValueError("external adapter config has an invalid rank")
    if (
        not isinstance(targets, list)
        or not targets
        or len(targets) > 4096
        or any(not isinstance(item, str) for item in targets)
    ):
        raise ValueError("external adapter config has invalid target modules")
    if (
        config.get("art_lora_format") != "vllm"
        and str(config.get("peft_type", "")).upper() != "LORA"
    ):
        raise ValueError("external adapter config is not a vLLM-compatible LoRA")
    return config


_SAFETENSORS_FLOAT_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def _validate_safetensors_file(path: Path, expected_bytes: int) -> None:
    with path.open("rb") as source:
        raw_header_bytes = source.read(8)
        if len(raw_header_bytes) != 8:
            raise RuntimeError("external adapter safetensors header is incomplete")
        header_bytes = int.from_bytes(raw_header_bytes, "little")
        if not 2 <= header_bytes <= min(64 << 20, expected_bytes - 8):
            raise RuntimeError("external adapter safetensors header is unbounded")
        try:
            header = json.loads(source.read(header_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "external adapter safetensors header is invalid"
            ) from error
    if not isinstance(header, dict) or len(header) > 65_537:
        raise RuntimeError("external adapter safetensors tensor table is invalid")
    data_bytes = expected_bytes - 8 - header_bytes
    extents: list[tuple[int, int]] = []
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise RuntimeError("external adapter safetensors metadata is invalid")
            continue
        if not isinstance(name, str) or not name or len(name) > 1024:
            raise RuntimeError("external adapter safetensors tensor name is invalid")
        if not isinstance(tensor, dict):
            raise RuntimeError("external adapter safetensors tensor entry is invalid")
        dtype_bytes = _SAFETENSORS_FLOAT_BYTES.get(tensor.get("dtype"))
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if (
            dtype_bytes is None
            or not isinstance(shape, list)
            or not 1 <= len(shape) <= 8
            or any(type(value) is not int or value <= 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int or value < 0 for value in offsets)
        ):
            raise RuntimeError("external adapter safetensors tensor layout is invalid")
        start, end = offsets
        elements = 1
        for value in shape:
            elements *= value
        if end <= start or end - start != elements * dtype_bytes or end > data_bytes:
            raise RuntimeError("external adapter safetensors tensor extent is invalid")
        extents.append((start, end))
    if not extents:
        raise RuntimeError("external adapter safetensors contains no tensors")
    cursor = 0
    for start, end in sorted(extents):
        if start != cursor:
            raise RuntimeError("external adapter safetensors extents are not complete")
        cursor = end
    if cursor != data_bytes:
        raise RuntimeError("external adapter safetensors payload is incomplete")


def _download_exact_object(
    source: ExternalAdapterObjectSource,
    path: Path,
    *,
    timeout_s: float,
) -> None:
    with path.open("xb") as output:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(output.fileno(), 0, source.object_bytes)

        def reset() -> None:
            output.seek(0)

        received = _stream_exact_object(
            source.object_url,
            size_bytes=source.object_bytes,
            sha256=source.object_sha256,
            timeout_s=timeout_s,
            write=output.write,
            reset=reset,
        )
        output.truncate(received)
        output.flush()
        os.fsync(output.fileno())


def _stream_exact_object(
    url: str,
    *,
    size_bytes: int,
    sha256: str,
    timeout_s: float,
    write: Any,
    reset: Callable[[], None] | None = None,
    wait_until_available: bool = False,
) -> int:
    deadline = time.monotonic() + timeout_s
    attempts = 0
    while True:
        attempts += 1
        if reset is not None:
            reset()
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError("external adapter object download timed out")
        try:
            return _stream_exact_object_once(
                url,
                size_bytes=size_bytes,
                sha256=sha256,
                timeout_s=remaining_s,
                write=write,
                wait_until_available=wait_until_available,
            )
        except _RetryableObjectError as error:
            if not wait_until_available and attempts >= 4:
                raise RuntimeError(
                    "external adapter object retries exhausted"
                ) from error
            delay_s = min(
                0.1 * (2 ** min(attempts - 1, 3)), deadline - time.monotonic()
            )
            if delay_s <= 0:
                raise TimeoutError(
                    "external adapter object download timed out"
                ) from error
            time.sleep(delay_s)


class _RetryableObjectError(RuntimeError):
    pass


def _stream_exact_object_once(
    url: str,
    *,
    size_bytes: int,
    sha256: str,
    timeout_s: float,
    write: Any,
    wait_until_available: bool,
) -> int:
    started = time.monotonic()
    digest = hashlib.sha256()
    received = 0
    try:
        with httpx.Client(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_s, connect=min(30.0, timeout_s)),
        ) as client:
            with client.stream(
                "GET", url, headers={"Accept-Encoding": "identity"}
            ) as response:
                if response.status_code != 200:
                    message = (
                        f"external adapter object returned HTTP {response.status_code}"
                    )
                    if response.status_code in {408, 425, 429, 500, 502, 503, 504} or (
                        wait_until_available and response.status_code == 404
                    ):
                        raise _RetryableObjectError(message)
                    raise RuntimeError(message)
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise RuntimeError("external adapter object used content encoding")
                length = response.headers.get("content-length")
                if length is not None:
                    try:
                        declared = int(length)
                    except ValueError as error:
                        raise RuntimeError(
                            "external adapter object has invalid content length"
                        ) from error
                    if declared != size_bytes:
                        raise RuntimeError("external adapter object size changed")
                for chunk in response.iter_bytes(1 << 20):
                    if time.monotonic() - started > timeout_s:
                        raise _RetryableObjectError(
                            "external adapter object download timed out"
                        )
                    received += len(chunk)
                    if received > size_bytes:
                        raise RuntimeError(
                            "external adapter object exceeded its byte bound"
                        )
                    write(chunk)
                    digest.update(chunk)
    except httpx.TransportError as error:
        raise _RetryableObjectError(
            "external adapter object transport failed"
        ) from error
    if received != size_bytes:
        raise _RetryableObjectError("external adapter object was incomplete")
    if digest.hexdigest() != sha256:
        raise RuntimeError("external adapter object digest changed")
    return received


def _pwrite_all(fd: int, value: bytes, offset: int) -> int:
    written = 0
    while written < len(value):
        count = os.pwrite(fd, value[written:], offset + written)
        if count <= 0:
            raise RuntimeError("external adapter shard write made no progress")
        written += count
    return written


def _materialize_external_shards(
    source: ExternalAdapterShardedSource,
    root: Path,
    *,
    timeout_s: float,
) -> None:
    files = {
        "adapter_config.json": source.config_bytes,
        "adapter_model.safetensors": source.model_bytes,
    }
    descriptors: dict[str, int] = {}
    try:
        for relative_path, size_bytes in files.items():
            fd = os.open(
                root / relative_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
            )
            descriptors[relative_path] = fd
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(fd, 0, size_bytes)
            else:
                os.ftruncate(fd, size_bytes)

        def receive(shard: ExternalAdapterShard) -> None:
            cursor = shard.file_offset

            def reset() -> None:
                nonlocal cursor
                cursor = shard.file_offset

            def write(chunk: bytes) -> int:
                nonlocal cursor
                count = _pwrite_all(descriptors[shard.relative_path], chunk, cursor)
                cursor += count
                return count

            _stream_exact_object(
                shard.object_url,
                size_bytes=shard.object_bytes,
                sha256=shard.object_sha256,
                timeout_s=timeout_s,
                write=write,
                reset=reset,
                wait_until_available=source.commit is not None,
            )

        with ThreadPoolExecutor(
            max_workers=min(
                source.max_parallel_downloads,
                len(source.shards) + (source.commit is not None),
            )
        ) as pool:
            futures = [pool.submit(receive, shard) for shard in source.shards]
            if source.commit is not None:
                commit_payload = bytearray()
                commit = source.commit
                futures.append(
                    pool.submit(
                        _stream_exact_object,
                        commit.object_url,
                        size_bytes=commit.object_bytes,
                        sha256=commit.object_sha256,
                        timeout_s=timeout_s,
                        write=commit_payload.extend,
                        reset=commit_payload.clear,
                        wait_until_available=True,
                    )
                )
            for future in futures:
                future.result()
            if source.commit is not None and bytes(commit_payload) != (
                _external_shard_commit(source)
            ):
                raise RuntimeError("external adapter final commit changed")
        for fd in descriptors.values():
            os.fsync(fd)
    finally:
        for fd in descriptors.values():
            os.close(fd)


class AdapterSnapshotReceiver:
    """Owns receive buffers for immutable LoRA generations."""

    def __init__(
        self,
        host_id: str,
        output_root: str,
        *,
        pool_capacity: int = 2,
        local_transfer_root: str | None = None,
    ) -> None:
        if pool_capacity < 1:
            raise ValueError("adapter receive pool capacity must be positive")
        self.host_id = host_id
        self.output_root = Path(output_root) / "adapter_transfers"
        self.local_transfer_root = (
            None
            if local_transfer_root is None
            else Path(local_transfer_root) / "adapter_transfers"
        )
        self.pool_capacity = pool_capacity
        self._agent: Any | None = None
        self._pending: dict[str, _PendingReceive] = {}
        self._local_pending: dict[str, _PendingLocalReceive] = {}
        self._slots: list[_RegisteredSlot] = []
        self._condition = Condition()
        self._notifications: dict[str, AdapterTransferNotification] = {}
        self._materialized: set[str] = set()
        self._materialized_objects: dict[str, AdapterReceiveResult] = {}
        self._object_pending: set[str] = set()
        self._agent_lock = Lock()
        self._closed = False

    def materialize_object(
        self,
        source: ExternalAdapterSource,
        timeout_s: float = 300.0,
    ) -> AdapterReceiveResult:
        """Download and validate committed object bytes into immutable staging."""

        started = time.monotonic()
        with self._condition:
            if self._closed:
                raise RuntimeError("adapter receive pool is closed")
            existing = self._materialized_objects.get(source.generation_id)
            if existing is not None:
                if isinstance(source, ExternalAdapterObjectSource):
                    tensor_bytes = source.object_bytes
                    config_bytes = len(source.adapter_config_json.encode())
                else:
                    tensor_bytes = source.model_bytes
                    config_bytes = source.config_bytes
                if (
                    existing.source_identity != source.source_identity
                    or existing.tensor_bytes != tensor_bytes
                    or existing.config_bytes != config_bytes
                ):
                    raise RuntimeError("materialized external adapter identity changed")
                return existing
            deadline = started + timeout_s
            while len(self._object_pending) >= self.pool_capacity:
                if self._closed:
                    raise RuntimeError("adapter receive pool is closed")
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError("external adapter receive pool remained full")
                self._condition.wait(remaining_s)
            if self._closed:
                raise RuntimeError("adapter receive pool is closed")
            if (
                source.generation_id in self._pending
                or source.generation_id in self._local_pending
                or source.generation_id in self._object_pending
                or source.generation_id in self._materialized
            ):
                raise RuntimeError(
                    f"Adapter receive already exists: {source.generation_id}"
                )
            self._object_pending.add(source.generation_id)

        path = self.output_root / source.generation_id
        temporary = self.output_root / f".{source.generation_id}.{os.getpid()}.tmp"
        try:
            if path.exists() or temporary.exists():
                raise RuntimeError("external adapter staging path already exists")
            temporary.mkdir(parents=True)
            model_path = temporary / "adapter_model.safetensors"
            config_path = temporary / "adapter_config.json"
            if isinstance(source, ExternalAdapterObjectSource):
                _download_exact_object(source, model_path, timeout_s=timeout_s)
                config_path.write_text(source.adapter_config_json, encoding="utf-8")
                tensor_bytes = source.object_bytes
                config_bytes = len(source.adapter_config_json.encode())
            else:
                _materialize_external_shards(source, temporary, timeout_s=timeout_s)
                tensor_bytes = source.model_bytes
                config_bytes = source.config_bytes
                _adapter_config(config_path.read_bytes())
            _validate_safetensors_file(model_path, tensor_bytes)
            with self._condition:
                if self._closed:
                    raise RuntimeError("adapter receive pool closed during download")
            os.rename(temporary, path)
            result = AdapterReceiveResult(
                host_id=self.host_id,
                generation_id=source.generation_id,
                path=str(path),
                tensor_bytes=tensor_bytes,
                config_bytes=config_bytes,
                materialization_s=time.monotonic() - started,
                used_bytes=tensor_bytes + config_bytes,
                capacity_bytes=tensor_bytes + config_bytes,
                source_identity=source.source_identity,
            )
            with self._condition:
                self._materialized.add(source.generation_id)
                self._materialized_objects[source.generation_id] = result
            return result
        except BaseException:
            if temporary.exists():
                from shutil import rmtree

                rmtree(temporary)
            raise
        finally:
            with self._condition:
                self._object_pending.discard(source.generation_id)
                self._condition.notify_all()

    def prepare(
        self,
        generation_id: str,
        template_path: str,
        timeout_s: float = 300.0,
        transport: Literal["local", "nixl"] = "nixl",
    ) -> AdapterTransferTarget:
        if transport == "local":
            return self._prepare_local(generation_id, template_path, timeout_s)
        prepare_started = time.monotonic()
        required_bytes = _adapter_template_bytes(template_path)
        wait_started = time.monotonic()
        with self._condition:
            if self._closed:
                raise RuntimeError("adapter receive pool is closed")
            if generation_id in self._pending:
                raise RuntimeError(f"Adapter receive already exists: {generation_id}")
            slot, registration_s = self._acquire_slot(
                required_bytes, deadline=wait_started + timeout_s
            )
            slot.generation_id = generation_id
            pool_wait_s = time.monotonic() - wait_started - registration_s
        try:
            with self._agent_lock:
                agent = self._require_agent()
                remote_agent = agent.name
                metadata = base64.b64encode(agent.get_agent_metadata()).decode()
            path = str((self.output_root / generation_id).absolute())
            target = AdapterTransferTarget(
                host_id=self.host_id,
                generation_id=generation_id,
                path=path,
                remote_agent=remote_agent,
                remote_metadata_b64=metadata,
                remote_address=slot.block.data_ptr(),
                remote_device_id=0,
                slot_id=slot.slot_id,
                capacity_bytes=slot.block.numel(),
                prepare_s=time.monotonic() - prepare_started,
                pool_wait_s=max(0.0, pool_wait_s),
                registration_s=registration_s,
                transfer_timeout_s=timeout_s,
            )
        except BaseException:
            self._release_slot(slot, generation_id)
            raise
        self._pending[generation_id] = _PendingReceive(
            target=target,
            slot=slot,
        )
        return target

    def _prepare_local(
        self,
        generation_id: str,
        template_path: str,
        timeout_s: float,
    ) -> AdapterTransferTarget:
        prepare_started = time.monotonic()
        required_bytes = _adapter_template_bytes(template_path)
        wait_started = time.monotonic()
        with self._condition:
            while len(self._local_pending) >= self.pool_capacity:
                remaining_s = wait_started + timeout_s - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError("local adapter receive pool remained full")
                self._condition.wait(remaining_s)
            if self._closed:
                raise RuntimeError("adapter receive pool is closed")
            if generation_id in self._local_pending or generation_id in self._pending:
                raise RuntimeError(f"Adapter receive already exists: {generation_id}")
            if self.local_transfer_root is None:
                raise RuntimeError("local adapter transfer root is not configured")
            self.local_transfer_root.mkdir(parents=True, exist_ok=True)
            socket_path = (
                "/tmp/art-lora-"
                + hashlib.sha256(
                    f"{self.host_id}:{generation_id}:{os.getpid()}".encode()
                ).hexdigest()[:24]
                + ".sock"
            )
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(socket_path)
                listener.listen(1)
                listener.setblocking(False)
                target = AdapterTransferTarget(
                    transport="local",
                    host_id=self.host_id,
                    generation_id=generation_id,
                    path=str(
                        (
                            self.local_transfer_root / self.host_id / generation_id
                        ).absolute()
                    ),
                    remote_agent=socket_path,
                    remote_metadata_b64="-",
                    remote_address=0,
                    remote_device_id=0,
                    slot_id=0,
                    capacity_bytes=required_bytes,
                    prepare_s=time.monotonic() - prepare_started,
                    pool_wait_s=time.monotonic() - wait_started,
                    registration_s=0.0,
                    transfer_timeout_s=timeout_s,
                )
            except BaseException:
                listener.close()
                Path(socket_path).unlink(missing_ok=True)
                raise
            self._local_pending[generation_id] = _PendingLocalReceive(
                target=target,
                listener=listener,
            )
        return target

    def poll(self, generation_id: str) -> AdapterReceiveResult | None:
        if generation_id in self._local_pending:
            return self._poll_local(generation_id)
        pending = self._pending.get(generation_id)
        if pending is None:
            raise RuntimeError(f"Unknown adapter receive: {generation_id}")
        notification = self._take_notification(generation_id)
        if notification is None:
            return None
        if notification.used_bytes > pending.slot.block.numel():
            self._finish(generation_id)
            raise RuntimeError("Adapter payload exceeds its prepared receive capacity")
        started = time.monotonic()
        path = Path(pending.target.path)
        if path.exists():
            self._finish(generation_id)
            raise RuntimeError(f"Adapter transfer path already exists: {path}")
        try:
            path.mkdir(parents=True)
            save_prepared_safetensors(
                PreparedSafetensors(
                    (pending.slot.block.narrow(0, 0, notification.used_bytes),)
                ),
                path / "adapter_model.safetensors",
            )
            with (path / "adapter_config.json").open("w", encoding="utf-8") as output:
                json.dump(notification.adapter_config, output, indent=2, sort_keys=True)
                output.write("\n")
            materialization_s = time.monotonic() - started
            model_bytes = (path / "adapter_model.safetensors").stat().st_size
            config_bytes = (path / "adapter_config.json").stat().st_size
        except BaseException:
            if path.exists():
                from shutil import rmtree

                rmtree(path)
            raise
        finally:
            self._finish(generation_id)
        self._materialized.add(generation_id)
        return AdapterReceiveResult(
            host_id=self.host_id,
            generation_id=generation_id,
            path=str(path),
            tensor_bytes=model_bytes,
            config_bytes=config_bytes,
            materialization_s=materialization_s,
            slot_id=pending.target.slot_id,
            used_bytes=notification.used_bytes,
            capacity_bytes=pending.target.capacity_bytes,
            prepare_s=pending.target.prepare_s,
            pool_wait_s=pending.target.pool_wait_s,
            registration_s=pending.target.registration_s,
            sender_staging_s=notification.sender_staging_s,
            sender_registration_s=notification.sender_registration_s,
        )

    def _poll_local(self, generation_id: str) -> AdapterReceiveResult | None:
        pending = self._local_pending[generation_id]
        try:
            connection, _ = pending.listener.accept()
        except BlockingIOError:
            return None
        try:
            connection.settimeout(60.0)
            payload = bytearray()
            while chunk := connection.recv(64 * 1024):
                payload.extend(chunk)
            notification = AdapterTransferNotification.model_validate_json(payload)
            if notification.generation_id != generation_id:
                raise RuntimeError("local adapter notification has wrong generation")
            path = Path(pending.target.path)
            model_path = path / "adapter_model.safetensors"
            config_path = path / "adapter_config.json"
            if not model_path.is_file() or not config_path.is_file():
                raise RuntimeError("local adapter transfer is incomplete")
            self._materialized.add(generation_id)
            return AdapterReceiveResult(
                host_id=self.host_id,
                generation_id=generation_id,
                path=str(path),
                tensor_bytes=model_path.stat().st_size,
                config_bytes=config_path.stat().st_size,
                materialization_s=notification.sender_staging_s,
                slot_id=pending.target.slot_id,
                used_bytes=notification.used_bytes,
                capacity_bytes=pending.target.capacity_bytes,
                prepare_s=pending.target.prepare_s,
                pool_wait_s=pending.target.pool_wait_s,
                registration_s=0.0,
                sender_staging_s=notification.sender_staging_s,
                sender_registration_s=0.0,
            )
        finally:
            connection.close()
            self._finish_local(generation_id)

    def release(self, generation_id: str) -> None:
        from shutil import rmtree

        if generation_id in self._pending:
            self._finish(generation_id)
        if generation_id in self._local_pending:
            self._finish_local(generation_id)
        with self._agent_lock:
            self._notifications.pop(generation_id, None)
        with self._condition:
            self._materialized.discard(generation_id)
            self._materialized_objects.pop(generation_id, None)
        roots = [self.output_root]
        if self.local_transfer_root is not None:
            roots.append(self.local_transfer_root / self.host_id)
        for root in roots:
            path = root / generation_id
            if path.exists():
                rmtree(path)

    def state(self) -> AdapterReceivePoolState:
        with self._condition:
            slots = tuple(self._slots)
            pending_nixl = len(self._pending)
            pending_local = len(self._local_pending)
            pending_object = len(self._object_pending)
            materialized = len(self._materialized)
            closed = self._closed
        with self._agent_lock:
            notifications = len(self._notifications)
        return AdapterReceivePoolState(
            host_id=self.host_id,
            pool_capacity=self.pool_capacity,
            registered_slots=len(slots),
            registered_capacity_bytes=sum(slot.block.numel() for slot in slots),
            active_registered_slots=sum(
                slot.generation_id is not None for slot in slots
            ),
            pending_nixl_receives=pending_nixl,
            pending_local_receives=pending_local,
            pending_object_receives=pending_object,
            materialized_generations=materialized,
            pending_notifications=notifications,
            closed=closed,
        )

    def _finish_local(self, generation_id: str) -> None:
        pending = self._local_pending.pop(generation_id)
        pending.listener.close()
        Path(pending.target.remote_agent).unlink(missing_ok=True)
        with self._condition:
            self._condition.notify()

    def _require_agent(self) -> Any:
        if self._agent is None:
            self._agent = _new_agent(f"art-lora-receiver-{self.host_id}-{os.getpid()}")
        return self._agent

    def _take_notification(
        self, generation_id: str
    ) -> AdapterTransferNotification | None:
        with self._agent_lock:
            for messages in self._require_agent().get_new_notifs().values():
                for message in messages:
                    notification = AdapterTransferNotification.model_validate_json(
                        message
                    )
                    self._notifications[notification.generation_id] = notification
            return self._notifications.pop(generation_id, None)

    def _finish(self, generation_id: str) -> None:
        pending = self._pending.pop(generation_id)
        self._release_slot(pending.slot, generation_id)

    def _release_slot(self, slot: _RegisteredSlot, generation_id: str) -> None:
        with self._condition:
            if slot.generation_id != generation_id:
                raise RuntimeError("adapter receive slot ownership changed")
            slot.generation_id = None
            self._condition.notify()

    def _acquire_slot(
        self, used_bytes: int, *, deadline: float
    ) -> tuple[_RegisteredSlot, float]:
        while True:
            free = [slot for slot in self._slots if slot.generation_id is None]
            fitting = [slot for slot in free if slot.block.numel() >= used_bytes]
            if fitting:
                return min(fitting, key=lambda slot: slot.block.numel()), 0.0
            if free or len(self._slots) < self.pool_capacity:
                previous = min(free, key=lambda slot: slot.block.numel(), default=None)
                slot_id = len(self._slots) if previous is None else previous.slot_id
                capacity = used_bytes
                started = time.monotonic()
                block = torch.empty(capacity, dtype=torch.uint8)
                with self._agent_lock:
                    agent = self._require_agent()
                    registration = agent.register_memory((block,), backends=["UCX"])
                    if previous is not None:
                        agent.deregister_memory(previous.registration, backends=["UCX"])
                if previous is None:
                    slot = _RegisteredSlot(slot_id, block, registration)
                    self._slots.append(slot)
                else:
                    previous.block = block
                    previous.registration = registration
                    slot = previous
                return slot, time.monotonic() - started
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError("adapter receive pool remained full")
            self._condition.wait(remaining_s)
            if self._closed:
                raise RuntimeError("adapter receive pool closed while waiting")

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._object_pending:
                self._condition.wait()
        for generation_id in (
            *self._pending,
            *self._local_pending,
            *self._materialized,
        ):
            self.release(generation_id)
        if self._agent is not None:
            with self._agent_lock:
                for slot in self._slots:
                    self._agent.deregister_memory(slot.registration, backends=["UCX"])
        self._slots.clear()


class NixlAdapterSender:
    """Transfers one immutable CPU snapshot to one or more prepared hosts."""

    def __init__(self) -> None:
        self._agent: Any | None = None
        self._block: torch.Tensor | None = None
        self._registration: Any | None = None
        self._payload: bytearray | None = None
        self._remote_agents: dict[str, tuple[str, str]] = {}
        self._unreleased_handles: list[Any] = []
        self._active_transfers = 0
        self._completed_transfers = 0
        self._poisoned = False
        self._closed = False

    def send(
        self,
        payload: PreparedSafetensors,
        adapter_config: dict[str, Any],
        targets: tuple[AdapterTransferTarget, ...],
    ) -> None:
        if self._closed:
            raise RuntimeError("NIXL adapter sender is closed")
        if self._poisoned:
            raise RuntimeError("NIXL adapter sender requires a runtime restart")
        if not targets:
            return
        first = targets[0]
        if any(target.generation_id != first.generation_id for target in targets[1:]):
            raise RuntimeError("Adapter transfer targets disagree")
        used_bytes = payload.nbytes
        if any(used_bytes > target.capacity_bytes for target in targets):
            raise RuntimeError("Adapter payload exceeds prepared receive capacity")
        agent = self._require_agent()
        sender_registration_s = self._ensure_capacity(used_bytes)
        assert self._block is not None
        staging_started = time.monotonic()
        _copy_payload(payload, self._block)
        notification = (
            AdapterTransferNotification(
                generation_id=first.generation_id,
                used_bytes=used_bytes,
                adapter_config=adapter_config,
                sender_staging_s=time.monotonic() - staging_started,
                sender_registration_s=sender_registration_s,
            )
            .model_dump_json()
            .encode()
        )
        for target in targets:
            local_descriptors = agent.get_xfer_descs(
                (self._block.narrow(0, 0, used_bytes),)
            )
            remote_agent = self._remote_agent_for_target(agent, target)
            self._active_transfers += 1
            handle = None
            try:
                handle = agent.initialize_xfer(
                    "WRITE",
                    local_descriptors,
                    agent.get_xfer_descs(
                        [
                            (
                                target.remote_address,
                                used_bytes,
                                target.remote_device_id,
                            )
                        ],
                        mem_type="DRAM",
                    ),
                    remote_agent,
                    notification,
                    backends=["UCX"],
                )
                _run_nixl_transfer(
                    agent,
                    handle,
                    timeout_s=target.transfer_timeout_s,
                    description=(
                        f"adapter transfer for {target.host_id}: {target.generation_id}"
                    ),
                )
                self._completed_transfers += 1
            except BaseException as error:
                if handle is not None:
                    try:
                        handle.release()
                    except BaseException as release_error:
                        self._poisoned = True
                        self._unreleased_handles.append(handle)
                        error.add_note(
                            "NIXL transfer cancellation failed; sender is poisoned: "
                            f"{type(release_error).__name__}: {release_error}"
                        )
                raise
            else:
                if handle is not None:
                    try:
                        handle.release()
                    except BaseException:
                        self._poisoned = True
                        self._unreleased_handles.append(handle)
                        raise
            finally:
                self._active_transfers -= 1

    def _ensure_capacity(self, used_bytes: int) -> float:
        if self._block is not None and self._block.numel() >= used_bytes:
            return 0.0
        capacity = max(
            used_bytes,
            2 * (0 if self._block is None else self._block.numel()),
        )
        block = torch.empty(capacity, dtype=torch.uint8)
        agent = self._require_agent()
        started = time.monotonic()
        registration = agent.register_memory((block,), backends=["UCX"])
        if self._registration is not None:
            agent.deregister_memory(self._registration, backends=["UCX"])
        self._block = block
        self._registration = registration
        return time.monotonic() - started

    def register_bytes(self, payload: bytearray) -> NixlMemorySource:
        """Register one immutable payload until this sender is closed."""

        if self._closed or self._block is not None or not payload:
            raise RuntimeError("NIXL byte source is not available")
        self._payload = payload
        self._block = torch.frombuffer(payload, dtype=torch.uint8)
        agent = self._require_agent()
        self._registration = agent.register_memory((self._block,), backends=["UCX"])
        return NixlMemorySource(
            agent=agent.name,
            metadata_b64=base64.b64encode(agent.get_agent_metadata()).decode(),
            address=self._block.data_ptr(),
            byte_count=len(payload),
        )

    def close(self) -> None:
        failures: list[BaseException] = []
        retained_handles = []
        for handle in self._unreleased_handles:
            try:
                handle.release()
            except BaseException as error:
                failures.append(error)
                retained_handles.append(handle)
        self._unreleased_handles = retained_handles
        if failures:
            raise RuntimeError(
                "NIXL adapter sender could not release an active transfer"
            ) from failures[0]
        if self._agent is not None:
            for _, remote_agent in self._remote_agents.values():
                self._agent.remove_remote_agent(remote_agent)
        self._remote_agents.clear()
        if self._agent is not None and self._registration is not None:
            self._agent.deregister_memory(self._registration, backends=["UCX"])
        self._block = None
        self._payload = None
        self._registration = None
        self._closed = True

    def state(self) -> AdapterSenderState:
        return AdapterSenderState(
            transport="nixl",
            active_transfers=self._active_transfers,
            completed_transfers=self._completed_transfers,
            registered_buffers=int(self._registration is not None),
            registered_capacity_bytes=(
                0 if self._block is None else self._block.numel()
            ),
            remote_agents=len(self._remote_agents),
            poisoned=self._poisoned,
            unreleased_handles=len(self._unreleased_handles),
            closed=self._closed,
        )

    def _remote_agent_for_target(
        self, agent: Any, target: AdapterTransferTarget
    ) -> str:
        cached = self._remote_agents.get(target.host_id)
        if cached is not None and cached[0] != target.remote_metadata_b64:
            try:
                agent.remove_remote_agent(cached[1])
            except BaseException:
                self._poisoned = True
                raise
            self._remote_agents.pop(target.host_id)
            cached = None
        if cached is None:
            remote_agent = agent.add_remote_agent(
                base64.b64decode(target.remote_metadata_b64)
            )
            if isinstance(remote_agent, bytes):
                remote_agent = remote_agent.decode()
            if remote_agent != target.remote_agent:
                try:
                    agent.remove_remote_agent(remote_agent)
                except BaseException:
                    self._poisoned = True
                    raise
                raise RuntimeError("NIXL target returned the wrong agent identity")
            self._remote_agents[target.host_id] = (
                target.remote_metadata_b64,
                remote_agent,
            )
            return remote_agent
        if cached[1] != target.remote_agent:
            raise RuntimeError("NIXL target returned the wrong agent identity")
        return cached[1]

    def _require_agent(self) -> Any:
        if self._agent is None:
            self._agent = _new_agent(f"art-nixl-sender-{os.getpid()}-{id(self):x}")
        return self._agent


class AdapterSnapshotSender:
    """Dispatches immutable snapshots over the transport selected by each target."""

    def __init__(self) -> None:
        self._nixl: NixlAdapterSender | None = None
        self._local_completed_transfers = 0
        self._closed = False

    def send(
        self,
        snapshot: Any,
        targets: tuple[AdapterTransferTarget, ...],
        *,
        prepared_tensors: PreparedSafetensors,
    ) -> None:
        transports = {target.transport for target in targets}
        if not targets:
            return
        if len(transports) != 1:
            raise RuntimeError("adapter transfer targets mix transports")
        if transports == {"nixl"}:
            if self._nixl is None:
                self._nixl = NixlAdapterSender()
            self._nixl.send(
                prepared_tensors,
                {**snapshot.adapter_config, "art_lora_format": "vllm"},
                targets,
            )
            return
        self._send_local(snapshot, targets, prepared_tensors=prepared_tensors)
        self._local_completed_transfers += len(targets)

    def state(self) -> AdapterSenderState:
        if self._nixl is not None:
            nixl = self._nixl.state()
            return nixl.model_copy(
                update={
                    "transport": (
                        "mixed" if self._local_completed_transfers else "nixl"
                    ),
                    "completed_transfers": (
                        nixl.completed_transfers + self._local_completed_transfers
                    ),
                    "closed": self._closed,
                }
            )
        return AdapterSenderState(
            transport="local" if self._local_completed_transfers else "idle",
            active_transfers=0,
            completed_transfers=self._local_completed_transfers,
            registered_buffers=0,
            registered_capacity_bytes=0,
            remote_agents=0,
            poisoned=False,
            unreleased_handles=0,
            closed=self._closed,
        )

    @staticmethod
    def _send_local(
        snapshot: Any,
        targets: tuple[AdapterTransferTarget, ...],
        *,
        prepared_tensors: PreparedSafetensors,
    ) -> None:
        from art.megatron.weights.lora_publish import save_vllm_lora_snapshot

        first = targets[0]
        snapshot_config = {**snapshot.adapter_config, "art_lora_format": "vllm"}
        if any(target.generation_id != first.generation_id for target in targets):
            raise RuntimeError("local adapter transfer target changed")
        for target in targets:
            started = time.monotonic()
            save_vllm_lora_snapshot(
                snapshot,
                target.path,
                prepared_tensors=prepared_tensors,
            )
            notification = AdapterTransferNotification(
                generation_id=target.generation_id,
                used_bytes=prepared_tensors.nbytes,
                adapter_config=snapshot_config,
                sender_staging_s=time.monotonic() - started,
                sender_registration_s=0.0,
            ).model_dump_json()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(60.0)
                client.connect(target.remote_agent)
                client.sendall(notification.encode())

    def close(self) -> None:
        if self._nixl is not None:
            self._nixl.close()
            self._nixl = None
        self._closed = True
