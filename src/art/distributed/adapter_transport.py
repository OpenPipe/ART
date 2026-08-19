from __future__ import annotations

import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
from threading import Condition, Lock
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
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
        self._condition = Condition()
        self._notifications: dict[str, AdapterTransferNotification] = {}
        self._materialized: set[str] = set()
        self._agent_lock = Lock()
        self._closed = False

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
                    remote_address=0,
                    remote_device_id=0,
                    slot_id=0,
                    capacity_bytes=required_bytes,
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

    def send(
        self,
        payload: PreparedSafetensors,
        adapter_config: dict[str, Any],
        targets: tuple[AdapterTransferTarget, ...],
    ) -> None:
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
