from __future__ import annotations

import base64
import importlib
import json
import os
from pathlib import Path
import struct
import sys
from threading import Lock
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
import torch


class _TransportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AdapterTensorSpec(_TransportRecord):
    name: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)


class AdapterTransferTarget(_TransportRecord):
    transport: Literal["nixl"] = "nixl"
    host_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    remote_agent: str = Field(min_length=1)
    remote_metadata_b64: str = Field(min_length=1)
    remote_descriptors_b64: str = Field(min_length=1)
    tensors: tuple[AdapterTensorSpec, ...]
    adapter_config: dict[str, Any]


class AdapterReceiveResult(_TransportRecord):
    host_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    tensor_bytes: int = Field(gt=0)
    config_bytes: int = Field(gt=0)
    materialization_s: float = Field(ge=0)


class _PendingReceive:
    def __init__(
        self,
        *,
        target: AdapterTransferTarget,
        blocks: tuple[torch.Tensor, ...],
        tensors: dict[str, torch.Tensor],
        registration: Any,
    ) -> None:
        self.target = target
        self.blocks = blocks
        self.tensors = tensors
        self.registration = registration


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


def spec_numel(spec: AdapterTensorSpec) -> int:
    count = 1
    for dim in spec.shape:
        count *= dim
    return count


def _snapshot_blocks(
    tensors: dict[str, torch.Tensor], specs: tuple[AdapterTensorSpec, ...]
) -> tuple[torch.Tensor, ...]:
    expected = {spec.name for spec in specs}
    if set(tensors) != expected:
        raise RuntimeError("LoRA snapshot and transport target tensor names differ")
    grouped: dict[str, list[tuple[AdapterTensorSpec, torch.Tensor]]] = {}
    for spec in specs:
        tensor = tensors[spec.name]
        expected_dtype = _SAFETENSORS_DTYPES.get(spec.dtype)
        if tuple(tensor.shape) != spec.shape or tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"LoRA snapshot tensor changed shape or dtype: {spec.name}"
            )
        grouped.setdefault(spec.dtype, []).append((spec, tensor))

    blocks = []
    for _dtype, group in sorted(grouped.items()):
        first = group[0][1]
        storage = first.untyped_storage()
        count = sum(tensor.numel() for _spec, tensor in group)
        expected_address = first.data_ptr()
        for _spec, tensor in group:
            if tensor.untyped_storage().data_ptr() != storage.data_ptr():
                raise RuntimeError("LoRA snapshot dtype group does not share storage")
            if tensor.data_ptr() != expected_address:
                raise RuntimeError("LoRA snapshot dtype group is not contiguous")
            expected_address += tensor.numel() * tensor.element_size()
        block = first.new_empty(0).set_(storage, first.storage_offset(), (count,), (1,))
        blocks.append(block)
    return tuple(blocks)


class NixlAdapterReceiver:
    """Owns host memory registered for immutable LoRA generations."""

    def __init__(self, host_id: str, output_root: str) -> None:
        self.host_id = host_id
        self.output_root = Path(output_root) / "adapter_transfers"
        self._agent: Any | None = None
        self._pending: dict[str, _PendingReceive] = {}
        self._notifications: set[str] = set()
        self._notification_lock = Lock()

    def prepare(
        self, generation_id: str, template_path: str, tensor_dtype: str
    ) -> AdapterTransferTarget:
        if generation_id in self._pending:
            raise RuntimeError(f"Adapter receive already exists: {generation_id}")
        specs, config = _read_adapter_template(template_path, tensor_dtype)
        blocks, tensors = _allocate_blocks(specs)
        agent = self._require_agent()
        registration = agent.register_memory(blocks, backends=["UCX"])
        descriptors = agent.get_xfer_descs(blocks)
        path = str((self.output_root / generation_id).absolute())
        target = AdapterTransferTarget(
            host_id=self.host_id,
            generation_id=generation_id,
            path=path,
            remote_agent=agent.name,
            remote_metadata_b64=base64.b64encode(agent.get_agent_metadata()).decode(),
            remote_descriptors_b64=base64.b64encode(
                agent.get_serialized_descs(descriptors)
            ).decode(),
            tensors=specs,
            adapter_config=config,
        )
        self._pending[generation_id] = _PendingReceive(
            target=target,
            blocks=blocks,
            tensors=tensors,
            registration=registration,
        )
        return target

    def wait(self, generation_id: str, timeout_s: float) -> AdapterReceiveResult:
        pending = self._pending.get(generation_id)
        if pending is None:
            raise RuntimeError(f"Unknown adapter receive: {generation_id}")
        deadline = time.monotonic() + timeout_s
        while not self._take_notification(generation_id):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Adapter transfer timed out: {generation_id}")
            time.sleep(0.001)
        from art.megatron.weights.lora_publish import (
            LoraSnapshot,
            save_vllm_lora_snapshot,
        )

        started = time.monotonic()
        path = Path(pending.target.path)
        if path.exists():
            self._finish(generation_id)
            raise RuntimeError(f"Adapter transfer path already exists: {path}")
        try:
            save_vllm_lora_snapshot(
                LoraSnapshot(
                    tensors=pending.tensors,
                    adapter_config=pending.target.adapter_config,
                ),
                str(path),
            )
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
        return AdapterReceiveResult(
            host_id=self.host_id,
            generation_id=generation_id,
            path=str(path),
            tensor_bytes=model_bytes,
            config_bytes=config_bytes,
            materialization_s=materialization_s,
        )

    def release(self, generation_id: str) -> None:
        from shutil import rmtree

        if generation_id in self._pending:
            self._finish(generation_id)
        with self._notification_lock:
            self._notifications.discard(generation_id)
        path = self.output_root / generation_id
        if path.exists():
            rmtree(path)

    def _require_agent(self) -> Any:
        if self._agent is None:
            self._agent = _new_agent(f"art-lora-receiver-{self.host_id}-{os.getpid()}")
        return self._agent

    def _take_notification(self, generation_id: str) -> bool:
        with self._notification_lock:
            for messages in self._require_agent().get_new_notifs().values():
                self._notifications.update(message.decode() for message in messages)
            if generation_id not in self._notifications:
                return False
            self._notifications.remove(generation_id)
            return True

    def _finish(self, generation_id: str) -> None:
        pending = self._pending.pop(generation_id)
        self._require_agent().deregister_memory(pending.registration, backends=["UCX"])


class NixlAdapterSender:
    """Transfers one immutable CPU snapshot to one or more prepared hosts."""

    def __init__(self) -> None:
        self._agent: Any | None = None

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
        blocks = _snapshot_blocks(snapshot.tensors, first.tensors)
        agent = self._require_agent()
        registration = agent.register_memory(blocks, backends=["UCX"])
        try:
            local_descriptors = agent.get_xfer_descs(blocks)
            for target in targets:
                remote_agent = agent.add_remote_agent(
                    base64.b64decode(target.remote_metadata_b64)
                )
                if isinstance(remote_agent, bytes):
                    remote_agent = remote_agent.decode()
                if remote_agent != target.remote_agent:
                    raise RuntimeError("NIXL target returned the wrong agent identity")
                handle = agent.initialize_xfer(
                    "WRITE",
                    local_descriptors,
                    agent.deserialize_descs(
                        base64.b64decode(target.remote_descriptors_b64)
                    ),
                    remote_agent,
                    target.generation_id.encode(),
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
                    agent.remove_remote_agent(remote_agent)
        finally:
            agent.deregister_memory(registration, backends=["UCX"])

    def _require_agent(self) -> Any:
        if self._agent is None:
            self._agent = _new_agent(f"art-lora-sender-{os.getpid()}")
        return self._agent
