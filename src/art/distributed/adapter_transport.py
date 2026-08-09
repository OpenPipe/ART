from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
from threading import Condition, Lock
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.utils.safetensors import save_safetensors


class _TransportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AdapterTensorSpec(_TransportRecord):
    name: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)


class AdapterTransferTarget(_TransportRecord):
    transport: Literal["local", "nixl"] = "nixl"
    host_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    remote_agent: str = Field(min_length=1)
    remote_metadata_b64: str = Field(min_length=1)
    remote_descriptors_b64: str = Field(min_length=1)
    tensors: tuple[AdapterTensorSpec, ...]
    adapter_config: dict[str, Any]
    slot_id: int = Field(ge=0)
    used_bytes: int = Field(gt=0)
    capacity_bytes: int = Field(gt=0)
    prepare_s: float = Field(ge=0)
    pool_wait_s: float = Field(ge=0)
    registration_s: float = Field(ge=0)


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


class AdapterTransferNotification(_TransportRecord):
    generation_id: str = Field(min_length=1)
    sender_staging_s: float = Field(ge=0)
    sender_registration_s: float = Field(ge=0)


class _PendingReceive:
    def __init__(
        self,
        *,
        target: AdapterTransferTarget,
        slot: "_RegisteredSlot",
        tensors: dict[str, torch.Tensor],
    ) -> None:
        self.target = target
        self.slot = slot
        self.tensors = tensors


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


_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "C64": torch.complex64,
}
_TRAINING_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}


def _load_nixl() -> tuple[Any, Any, Any]:
    root = Path("/usr/local/art-multinode/nixl/python")
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.update(
        NIXL_PLUGIN_DIR="/usr/local/art-multinode/nixl-ucx/lib/plugins",
        UCX_MODULE_DIR="/usr/local/art-multinode/ucx/lib/ucx",
        UCX_NET_DEVICES="all",
        UCX_TLS="rc,rc_gda,cuda_copy",
        UCX_IB_GDA_RETAIN_INACTIVE_CTX="yes",
    )
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
        "NIXL Python bindings are unavailable; run scripts/setup_multinode.sh"
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


def _read_adapter_template(
    path: str, tensor_dtype: str
) -> tuple[tuple[AdapterTensorSpec, ...], dict[str, Any]]:
    root = Path(path)
    with (root / "adapter_model.safetensors").open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise RuntimeError(f"Invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 64 * 1024 * 1024:
            raise RuntimeError(f"Unreasonable safetensors header in {path}")
        header = json.loads(source.read(header_length))
    try:
        dtype = _TRAINING_DTYPES[tensor_dtype]
    except KeyError:
        raise RuntimeError(
            f"Unsupported adapter transport dtype: {tensor_dtype}"
        ) from None
    specs = tuple(
        AdapterTensorSpec(
            name=name,
            shape=tuple(int(dim) for dim in value["shape"]),
            dtype=dtype,
        )
        for name, value in sorted(header.items())
        if name != "__metadata__"
    )
    if not specs:
        raise RuntimeError(f"Adapter template has no tensors: {path}")
    with (root / "adapter_config.json").open("r", encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise RuntimeError(f"Adapter config must be an object: {path}")
    if config.get("art_lora_format") != "vllm":
        raise RuntimeError(f"Adapter template is not in vLLM format: {path}")
    return specs, config


def _allocate_blocks(
    specs: tuple[AdapterTensorSpec, ...],
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    grouped: dict[str, list[AdapterTensorSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.dtype, []).append(spec)
    blocks: list[torch.Tensor] = []
    tensors: dict[str, torch.Tensor] = {}
    for dtype_name, group in sorted(grouped.items()):
        dtype = _SAFETENSORS_DTYPES.get(dtype_name)
        if dtype is None:
            raise RuntimeError(f"Unsupported adapter transport dtype: {dtype_name}")
        counts = [spec_numel(spec) for spec in group]
        block = torch.empty(sum(counts), dtype=dtype)
        blocks.append(block)
        offset = 0
        for spec, count in zip(group, counts, strict=True):
            tensors[spec.name] = block.narrow(0, offset, count).view(spec.shape)
            offset += count
    return tuple(blocks), tensors


def _tensor_views(
    block: torch.Tensor, specs: tuple[AdapterTensorSpec, ...]
) -> tuple[dict[str, torch.Tensor], int]:
    offset = 0
    tensors = {}
    for spec in specs:
        dtype = _SAFETENSORS_DTYPES.get(spec.dtype)
        if dtype is None:
            raise RuntimeError(f"Unsupported adapter transport dtype: {spec.dtype}")
        item_size = torch.empty((), dtype=dtype).element_size()
        offset = (offset + item_size - 1) // item_size * item_size
        count = spec_numel(spec)
        byte_count = count * item_size
        tensors[spec.name] = (
            block.narrow(0, offset, byte_count).view(dtype).view(spec.shape)
        )
        offset += byte_count
    return tensors, offset


def _spec_bytes(specs: tuple[AdapterTensorSpec, ...]) -> int:
    offset = 0
    for spec in specs:
        dtype = _SAFETENSORS_DTYPES.get(spec.dtype)
        if dtype is None:
            raise RuntimeError(f"Unsupported adapter transport dtype: {spec.dtype}")
        item_size = torch.empty((), dtype=dtype).element_size()
        offset = (offset + item_size - 1) // item_size * item_size
        offset += spec_numel(spec) * item_size
    return offset


def spec_numel(spec: AdapterTensorSpec) -> int:
    count = 1
    for dim in spec.shape:
        count *= dim
    return count


def _copy_snapshot(
    tensors: dict[str, torch.Tensor],
    specs: tuple[AdapterTensorSpec, ...],
    block: torch.Tensor,
) -> int:
    expected = {spec.name for spec in specs}
    if set(tensors) != expected:
        raise RuntimeError("LoRA snapshot and transport target tensor names differ")
    targets, used_bytes = _tensor_views(block, specs)
    for spec in specs:
        tensor = tensors[spec.name]
        expected_dtype = _SAFETENSORS_DTYPES.get(spec.dtype)
        if tuple(tensor.shape) != spec.shape or tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"LoRA snapshot tensor changed shape or dtype: {spec.name}"
            )
        targets[spec.name].copy_(tensor)
    return used_bytes


class AdapterSnapshotReceiver:
    """Owns receive buffers for immutable LoRA generations."""

    def __init__(
        self, host_id: str, output_root: str, *, pool_capacity: int = 2
    ) -> None:
        if pool_capacity < 1:
            raise ValueError("adapter receive pool capacity must be positive")
        self.host_id = host_id
        self.output_root = Path(output_root) / "adapter_transfers"
        self.pool_capacity = pool_capacity
        self._agent: Any | None = None
        self._pending: dict[str, _PendingReceive] = {}
        self._local_pending: dict[str, _PendingLocalReceive] = {}
        self._slots: list[_RegisteredSlot] = []
        self._templates: dict[
            tuple[str, str], tuple[tuple[AdapterTensorSpec, ...], dict[str, Any]]
        ] = {}
        self._condition = Condition()
        self._notifications: dict[str, AdapterTransferNotification] = {}
        self._materialized: set[str] = set()
        self._agent_lock = Lock()
        self._closed = False

    def prepare(
        self,
        generation_id: str,
        template_path: str,
        tensor_dtype: str,
        timeout_s: float = 300.0,
        transport: Literal["local", "nixl"] = "nixl",
    ) -> AdapterTransferTarget:
        if transport == "local":
            return self._prepare_local(
                generation_id, template_path, tensor_dtype, timeout_s
            )
        prepare_started = time.monotonic()
        template_key = (str(Path(template_path).absolute()), tensor_dtype)
        specs, config = self._templates.get(template_key, ((), {}))
        if not specs:
            specs, config = _read_adapter_template(template_path, tensor_dtype)
            self._templates[template_key] = (specs, config)
        used_bytes = _spec_bytes(specs)
        wait_started = time.monotonic()
        with self._condition:
            if self._closed:
                raise RuntimeError("adapter receive pool is closed")
            if generation_id in self._pending:
                raise RuntimeError(f"Adapter receive already exists: {generation_id}")
            slot, registration_s = self._acquire_slot(
                used_bytes, deadline=wait_started + timeout_s
            )
            slot.generation_id = generation_id
            pool_wait_s = time.monotonic() - wait_started - registration_s
        try:
            used_block = slot.block.narrow(0, 0, used_bytes)
            tensors, actual_used_bytes = _tensor_views(slot.block, specs)
            if actual_used_bytes != used_bytes:
                raise RuntimeError("adapter byte layout changed during preparation")
            with self._agent_lock:
                agent = self._require_agent()
                descriptors = agent.get_xfer_descs((used_block,))
                remote_agent = agent.name
                metadata = base64.b64encode(agent.get_agent_metadata()).decode()
                serialized = base64.b64encode(
                    agent.get_serialized_descs(descriptors)
                ).decode()
            path = str((self.output_root / generation_id).absolute())
            target = AdapterTransferTarget(
                host_id=self.host_id,
                generation_id=generation_id,
                path=path,
                remote_agent=remote_agent,
                remote_metadata_b64=metadata,
                remote_descriptors_b64=serialized,
                tensors=specs,
                adapter_config=config,
                slot_id=slot.slot_id,
                used_bytes=used_bytes,
                capacity_bytes=slot.block.numel(),
                prepare_s=time.monotonic() - prepare_started,
                pool_wait_s=max(0.0, pool_wait_s),
                registration_s=registration_s,
            )
        except BaseException:
            self._release_slot(slot, generation_id)
            raise
        self._pending[generation_id] = _PendingReceive(
            target=target,
            slot=slot,
            tensors=tensors,
        )
        return target

    def _prepare_local(
        self,
        generation_id: str,
        template_path: str,
        tensor_dtype: str,
        timeout_s: float,
    ) -> AdapterTransferTarget:
        prepare_started = time.monotonic()
        template_key = (str(Path(template_path).absolute()), tensor_dtype)
        specs, config = self._templates.get(template_key, ((), {}))
        if not specs:
            specs, config = _read_adapter_template(template_path, tensor_dtype)
            self._templates[template_key] = (specs, config)
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
                used_bytes = _spec_bytes(specs)
                local_root = Path(
                    os.environ.get(
                        "ART_LOCAL_ADAPTER_TRANSFER_ROOT",
                        "/dev/shm/art_adapter_transfers",
                    )
                )
                target = AdapterTransferTarget(
                    transport="local",
                    host_id=self.host_id,
                    generation_id=generation_id,
                    path=str((local_root / self.host_id / generation_id).absolute()),
                    remote_agent=socket_path,
                    remote_metadata_b64="-",
                    remote_descriptors_b64="-",
                    tensors=specs,
                    adapter_config=config,
                    slot_id=0,
                    used_bytes=used_bytes,
                    capacity_bytes=used_bytes,
                    prepare_s=time.monotonic() - prepare_started,
                    pool_wait_s=time.monotonic() - wait_started,
                    registration_s=0.0,
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
        started = time.monotonic()
        path = Path(pending.target.path)
        if path.exists():
            self._finish(generation_id)
            raise RuntimeError(f"Adapter transfer path already exists: {path}")
        try:
            path.mkdir(parents=True)
            save_safetensors(pending.tensors, path / "adapter_model.safetensors")
            with (path / "adapter_config.json").open("w", encoding="utf-8") as output:
                json.dump(
                    pending.target.adapter_config, output, indent=2, sort_keys=True
                )
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
            used_bytes=pending.target.used_bytes,
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
                used_bytes=pending.target.used_bytes,
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
        self._materialized.discard(generation_id)
        for root in (
            self.output_root,
            Path(
                os.environ.get(
                    "ART_LOCAL_ADAPTER_TRANSFER_ROOT",
                    "/dev/shm/art_adapter_transfers",
                )
            )
            / self.host_id,
        ):
            path = root / generation_id
            if path.exists():
                rmtree(path)

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
                capacity = max(
                    used_bytes,
                    2 * (0 if previous is None else previous.block.numel()),
                )
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
        self._remote_agents: dict[tuple[str, str], str] = {}

    def send(self, snapshot: Any, targets: tuple[AdapterTransferTarget, ...]) -> None:
        if not targets:
            return
        first = targets[0]
        if any(
            target.generation_id != first.generation_id
            or target.tensors != first.tensors
            or target.adapter_config != first.adapter_config
            for target in targets[1:]
        ):
            raise RuntimeError("Adapter transfer targets disagree")
        snapshot_config = {**snapshot.adapter_config, "art_lora_format": "vllm"}
        if snapshot_config != first.adapter_config:
            raise RuntimeError("LoRA snapshot adapter config changed during training")
        used_bytes = first.used_bytes
        agent = self._require_agent()
        sender_registration_s = self._ensure_capacity(used_bytes)
        assert self._block is not None
        staging_started = time.monotonic()
        if _copy_snapshot(snapshot.tensors, first.tensors, self._block) != used_bytes:
            raise RuntimeError(
                "LoRA snapshot byte layout differs from transport target"
            )
        notification = (
            AdapterTransferNotification(
                generation_id=first.generation_id,
                sender_staging_s=time.monotonic() - staging_started,
                sender_registration_s=sender_registration_s,
            )
            .model_dump_json()
            .encode()
        )
        local_descriptors = agent.get_xfer_descs(
            (self._block.narrow(0, 0, used_bytes),)
        )
        for target in targets:
            key = (target.host_id, target.remote_metadata_b64)
            remote_agent = self._remote_agents.get(key)
            if remote_agent is None:
                remote_agent = agent.add_remote_agent(
                    base64.b64decode(target.remote_metadata_b64)
                )
                if isinstance(remote_agent, bytes):
                    remote_agent = remote_agent.decode()
                self._remote_agents[key] = remote_agent
            if remote_agent != target.remote_agent:
                raise RuntimeError("NIXL target returned the wrong agent identity")
            handle = agent.initialize_xfer(
                "WRITE",
                local_descriptors,
                agent.deserialize_descs(
                    base64.b64decode(target.remote_descriptors_b64)
                ),
                remote_agent,
                notification,
                backends=["UCX"],
            )
            try:
                state = agent.transfer(handle)
                while state == "PROC":
                    time.sleep(0.001)
                    state = agent.check_xfer_state(handle)
                if state != "DONE":
                    raise RuntimeError(
                        f"NIXL adapter transfer failed for {target.host_id}"
                    )
            finally:
                handle.release()

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

    def close(self) -> None:
        if self._agent is not None:
            for remote_agent in self._remote_agents.values():
                self._agent.remove_remote_agent(remote_agent)
        self._remote_agents.clear()
        if self._agent is not None and self._registration is not None:
            self._agent.deregister_memory(self._registration, backends=["UCX"])
        self._block = None
        self._registration = None

    def _require_agent(self) -> Any:
        if self._agent is None:
            self._agent = _new_agent(f"art-lora-sender-{os.getpid()}")
        return self._agent


class AdapterSnapshotSender:
    """Dispatches immutable snapshots over the transport selected by each target."""

    def __init__(self) -> None:
        self._nixl: NixlAdapterSender | None = None

    def send(self, snapshot: Any, targets: tuple[AdapterTransferTarget, ...]) -> None:
        transports = {target.transport for target in targets}
        if not targets:
            return
        if len(transports) != 1:
            raise RuntimeError("adapter transfer targets mix transports")
        if transports == {"nixl"}:
            if self._nixl is None:
                self._nixl = NixlAdapterSender()
            self._nixl.send(snapshot, targets)
            return
        self._send_local(snapshot, targets)

    @staticmethod
    def _send_local(snapshot: Any, targets: tuple[AdapterTransferTarget, ...]) -> None:
        from art.megatron.weights.lora_publish import save_vllm_lora_snapshot

        first = targets[0]
        snapshot_config = {**snapshot.adapter_config, "art_lora_format": "vllm"}
        if any(
            target.generation_id != first.generation_id
            or target.tensors != first.tensors
            or target.adapter_config != snapshot_config
            for target in targets
        ):
            raise RuntimeError("local adapter transfer target changed")
        expected = {spec.name for spec in first.tensors}
        if set(snapshot.tensors) != expected:
            raise RuntimeError("LoRA snapshot tensor names changed")
        for spec in first.tensors:
            tensor = snapshot.tensors[spec.name]
            expected_dtype = _SAFETENSORS_DTYPES.get(spec.dtype)
            if tuple(tensor.shape) != spec.shape or tensor.dtype != expected_dtype:
                raise RuntimeError(f"LoRA snapshot shape or dtype changed: {spec.name}")
        for target in targets:
            started = time.monotonic()
            save_vllm_lora_snapshot(snapshot, target.path)
            notification = AdapterTransferNotification(
                generation_id=target.generation_id,
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
