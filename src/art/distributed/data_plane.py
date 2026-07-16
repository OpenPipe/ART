from __future__ import annotations

import asyncio
from collections.abc import Mapping
import gc
import math
from multiprocessing import resource_tracker, shared_memory
import os
import secrets
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

PACKED_BATCH_FORMAT = "art_packed_rl_v1"
_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TensorSpec(_Contract):
    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    shape: tuple[int, ...]
    offset: int = Field(ge=0)
    byte_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_storage(self) -> "TensorSpec":
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be non-negative")
        item_size = _DTYPE_BYTES.get(self.dtype)
        if item_size is None:
            raise ValueError(f"unsupported packed tensor dtype {self.dtype!r}")
        if _numel(self.shape) * item_size != self.byte_count:
            raise ValueError("tensor byte_count does not match dtype and shape")
        return self


class MoeRoutingReplaySpec(_Contract):
    num_layers: int = Field(ge=1)
    topk: int = Field(ge=1)
    num_experts: int = Field(ge=1)
    packed_tokens: int = Field(ge=0)


class PrefixTreePackingStatsSpec(_Contract):
    logical_tokens: int = Field(ge=0)
    physical_tokens: int = Field(ge=0)


class PackedBatchRef(_Contract):
    batch_id: str = Field(min_length=1)
    owner_actor_id: str = Field(min_length=1)
    lease_id: str = Field(min_length=1)
    format: str = PACKED_BATCH_FORMAT
    shared_memory_name: str = Field(min_length=1)
    owner_process_id: int = Field(ge=1)
    tensors: tuple[TensorSpec, ...]
    num_sequences: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    byte_count: int = Field(ge=0)
    storage_byte_count: int = Field(ge=1)
    pixel_values_present: tuple[bool, ...]
    image_grid_thw_present: tuple[bool, ...]
    moe_routing_replay: MoeRoutingReplaySpec | None = None
    prefix_tree_packing_stats: PrefixTreePackingStatsSpec | None = None
    group_ids: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()
    min_source_version: int = Field(default=0, ge=0)
    max_source_version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_manifest(self) -> "PackedBatchRef":
        if self.format != PACKED_BATCH_FORMAT:
            raise ValueError(f"unsupported packed-batch format {self.format!r}")
        names = [tensor.name for tensor in self.tensors]
        if len(set(names)) != len(names):
            raise ValueError("tensor manifest names must be unique")
        if sum(tensor.byte_count for tensor in self.tensors) != self.byte_count:
            raise ValueError("packed-batch byte_count does not match tensor manifest")
        core_dtypes = {
            "tokens": "int64",
            "group_ids": "int64",
            "parent_ids": "int64",
            "input_pos": "int64",
            "assistant_mask": "bool",
            "logprobs": "float32",
            "advantages": "float32",
            "weights": "float32",
        }
        specs = {tensor.name: tensor for tensor in self.tensors}
        if not core_dtypes.keys() <= specs.keys():
            raise ValueError("packed-batch tensor manifest is missing core tensors")
        core_shape = (self.num_sequences, self.sequence_length)
        if any(
            specs[name].dtype != dtype or specs[name].shape != core_shape
            for name, dtype in core_dtypes.items()
        ):
            raise ValueError("core packed tensor dtype or shape is invalid")
        if (
            len(self.pixel_values_present) != self.num_sequences
            or len(self.image_grid_thw_present) != self.num_sequences
        ):
            raise ValueError("multimodal presence manifests must match num_sequences")
        expected_optional = {
            f"pixel_values/{index}"
            for index, present in enumerate(self.pixel_values_present)
            if present
        } | {
            f"image_grid_thw/{index}"
            for index, present in enumerate(self.image_grid_thw_present)
            if present
        }
        if any(
            specs[name].dtype
            != ("float32" if name.startswith("pixel_values/") else "int64")
            for name in expected_optional
        ):
            raise ValueError("multimodal packed tensor dtype is invalid")
        if "original_logprobs" in specs:
            expected_optional.add("original_logprobs")
            if (
                specs["original_logprobs"].dtype != "float32"
                or specs["original_logprobs"].shape != core_shape
            ):
                raise ValueError("original_logprobs dtype or shape is invalid")
        replay_names = {
            "moe_routing_replay/expert_indices",
            "moe_routing_replay/token_mask",
        }
        if self.moe_routing_replay is not None:
            if not replay_names <= specs.keys():
                raise ValueError("MoE routing replay manifest is incomplete")
            expected_optional |= replay_names
            replay = self.moe_routing_replay
            if (
                specs["moe_routing_replay/expert_indices"].dtype != "int32"
                or specs["moe_routing_replay/expert_indices"].shape
                != core_shape + (replay.num_layers, replay.topk)
                or specs["moe_routing_replay/token_mask"].dtype != "bool"
                or specs["moe_routing_replay/token_mask"].shape != core_shape
            ):
                raise ValueError("MoE routing replay tensor dtype or shape is invalid")
        if set(specs) != set(core_dtypes) | expected_optional:
            raise ValueError("packed-batch tensor manifest has unexpected tensors")
        previous_end = 0
        for tensor in self.tensors:
            if tensor.offset < previous_end:
                raise ValueError("packed-batch tensor storage must not overlap")
            if tensor.offset + tensor.byte_count > self.storage_byte_count:
                raise ValueError(
                    f"tensor {tensor.name!r} exceeds shared-memory storage"
                )
            previous_end = tensor.offset + tensor.byte_count
        if self.max_source_version < self.min_source_version:
            raise ValueError("max_source_version must be >= min_source_version")
        return self


class PackedBatchLeaseSet(_Contract):
    """One logical batch and its host-local physical leases."""

    ref: PackedBatchRef
    host_refs: dict[str, PackedBatchRef]

    @model_validator(mode="after")
    def _validate_hosts(self) -> "PackedBatchLeaseSet":
        if not self.host_refs:
            raise ValueError("packed batch requires at least one host lease")
        logical = _logical_ref(self.ref)
        if any(_logical_ref(ref) != logical for ref in self.host_refs.values()):
            raise ValueError("host leases must describe the same logical packed batch")
        return self


class BatchReservation(_Contract):
    reservation_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    storage_byte_count: int = Field(ge=1)


class DataPlaneStats(_Contract):
    capacity_bytes: int
    used_bytes: int
    reserved_bytes: int
    peak_bytes: int
    created_bytes: int
    copied_bytes: int
    transmitted_bytes: int
    copy_count: int
    batches: int
    leases: int


class PackedBatchCapacityError(RuntimeError):
    pass


class PackedBatchLeaseError(RuntimeError):
    pass


class _Entry:
    def __init__(self, shm: shared_memory.SharedMemory, ref: PackedBatchRef) -> None:
        self.shm = shm
        self.ref = ref


class SharedMemoryPackedBatchStore:
    """Own immutable current-format batches in bounded POSIX shared memory."""

    def __init__(self, *, owner_actor_id: str, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be > 0")
        self.owner_actor_id = owner_actor_id
        self.capacity_bytes = capacity_bytes
        self._entries: dict[str, _Entry] = {}
        self._reservations: dict[str, BatchReservation] = {}
        self._reclaimed: set[str] = set()
        self._used_bytes = 0
        self._reserved_bytes = 0
        self._peak_bytes = 0
        self._created_bytes = 0
        self._copied_bytes = 0
        self._transmitted_bytes = 0
        self._copy_count = 0

    def create(
        self,
        tensors: Any,
        *,
        batch_id: str,
        group_ids: tuple[str, ...] = (),
        record_ids: tuple[str, ...] = (),
        min_source_version: int = 0,
        max_source_version: int = 0,
    ) -> PackedBatchRef:
        flat, metadata = _flatten_packed_tensors(tensors)
        manifest, storage_bytes = _layout(flat)
        if batch_id in self._reclaimed:
            raise PackedBatchLeaseError(f"packed batch {batch_id!r} was reclaimed")
        if batch_id in self._entries or any(
            reservation.batch_id == batch_id
            for reservation in self._reservations.values()
        ):
            raise ValueError(f"packed batch {batch_id!r} already exists")
        self._require_capacity(storage_bytes)
        lease_id = secrets.token_hex(16)
        shm = shared_memory.SharedMemory(create=True, size=storage_bytes)
        try:
            for spec, (_, tensor) in zip(manifest, flat, strict=True):
                destination = _tensor_from_buffer(_shm_buffer(shm), spec)
                destination.copy_(tensor)
            ref = PackedBatchRef(
                batch_id=batch_id,
                owner_actor_id=self.owner_actor_id,
                lease_id=lease_id,
                shared_memory_name=shm.name,
                owner_process_id=os.getpid(),
                tensors=manifest,
                num_sequences=metadata["num_sequences"],
                sequence_length=metadata["sequence_length"],
                byte_count=sum(spec.byte_count for spec in manifest),
                storage_byte_count=storage_bytes,
                pixel_values_present=metadata["pixel_values_present"],
                image_grid_thw_present=metadata["image_grid_thw_present"],
                moe_routing_replay=metadata["moe_routing_replay"],
                prefix_tree_packing_stats=metadata["prefix_tree_packing_stats"],
                group_ids=group_ids,
                record_ids=record_ids,
                min_source_version=min_source_version,
                max_source_version=max_source_version,
            )
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        self._entries[batch_id] = _Entry(shm, ref)
        self._used_bytes += storage_bytes
        self._peak_bytes = max(
            self._peak_bytes, self._used_bytes + self._reserved_bytes
        )
        self._created_bytes += storage_bytes
        self._copied_bytes += ref.byte_count
        self._copy_count += len(manifest)
        return ref

    def reserve(self, source: PackedBatchRef) -> BatchReservation:
        if source.batch_id in self._reclaimed:
            raise PackedBatchLeaseError(
                f"packed batch {source.batch_id!r} was reclaimed"
            )
        if source.batch_id in self._entries or any(
            reservation.batch_id == source.batch_id
            for reservation in self._reservations.values()
        ):
            raise ValueError(f"packed batch {source.batch_id!r} already exists")
        self._require_capacity(source.storage_byte_count)
        reservation = BatchReservation(
            reservation_id=secrets.token_hex(16),
            batch_id=source.batch_id,
            storage_byte_count=source.storage_byte_count,
        )
        self._reservations[reservation.reservation_id] = reservation
        self._reserved_bytes += reservation.storage_byte_count
        self._peak_bytes = max(
            self._peak_bytes, self._used_bytes + self._reserved_bytes
        )
        return reservation

    async def commit_rdma(
        self,
        reservation_id: str,
        source: PackedBatchRef,
        rdma_buffer: Any,
        *,
        timeout_s: float,
    ) -> PackedBatchRef:
        reservation = self._reservations.get(reservation_id)
        if reservation is None or reservation.batch_id != source.batch_id:
            raise PackedBatchLeaseError(
                "unknown or mismatched packed-batch reservation"
            )
        shm = shared_memory.SharedMemory(
            create=True, size=reservation.storage_byte_count
        )
        destination = None
        try:
            import torch

            destination = torch.frombuffer(
                _shm_buffer(shm),
                dtype=torch.uint8,
                count=reservation.storage_byte_count,
            )
            await rdma_buffer.read_into(destination, timeout=math.ceil(timeout_s))
            del destination
            destination = None
            return self._finish_commit(reservation, source, shm)
        except BaseException:
            if destination is not None:
                del destination
            gc.collect()
            shm.close()
            shm.unlink()
            raise

    def abort(self, reservation_id: str) -> None:
        reservation = self._reservations.pop(reservation_id, None)
        if reservation is not None:
            self._reserved_bytes -= reservation.storage_byte_count

    def drop(self, ref: PackedBatchRef) -> None:
        """Idempotently reclaim one host-owned packed batch."""

        entry = self._entries.get(ref.batch_id)
        if entry is None:
            return
        if entry.ref.lease_id != ref.lease_id:
            raise PackedBatchLeaseError("packed-batch reference has a stale lease")
        self.reclaim(ref.batch_id)

    def reclaim(self, batch_id: str, *, fence: bool = True) -> bool:
        """Release committed or in-flight storage and optionally fence late writes."""

        if fence:
            self._reclaimed.add(batch_id)
        found = False
        for reservation_id, reservation in tuple(self._reservations.items()):
            if reservation.batch_id == batch_id:
                self.abort(reservation_id)
                found = True
        entry = self._entries.pop(batch_id, None)
        if entry is not None:
            self._used_bytes -= entry.ref.storage_byte_count
            entry.shm.close()
            entry.shm.unlink()
            found = True
        return found

    def map(self, ref: PackedBatchRef) -> "MappedPackedBatch":
        entry = self._entries.get(ref.batch_id)
        if entry is None or entry.ref.lease_id != ref.lease_id:
            raise PackedBatchLeaseError("packed-batch reference has no active lease")
        return MappedPackedBatch.open(ref)

    def note_transmitted(self, byte_count: int) -> None:
        self._transmitted_bytes += byte_count

    def close(self) -> None:
        batch_ids = set(self._entries)
        batch_ids.update(
            reservation.batch_id for reservation in self._reservations.values()
        )
        for batch_id in batch_ids:
            self.reclaim(batch_id, fence=True)

    def stats(self) -> DataPlaneStats:
        return DataPlaneStats(
            capacity_bytes=self.capacity_bytes,
            used_bytes=self._used_bytes,
            reserved_bytes=self._reserved_bytes,
            peak_bytes=self._peak_bytes,
            created_bytes=self._created_bytes,
            copied_bytes=self._copied_bytes,
            transmitted_bytes=self._transmitted_bytes,
            copy_count=self._copy_count,
            batches=len(self._entries),
            leases=len(self._entries),
        )

    def _require_capacity(self, byte_count: int) -> None:
        if byte_count > self.capacity_bytes:
            raise PackedBatchCapacityError(
                f"packed batch requires {byte_count} bytes, capacity is "
                f"{self.capacity_bytes}"
            )
        available = self.capacity_bytes - self._used_bytes - self._reserved_bytes
        if byte_count > available:
            raise PackedBatchCapacityError(
                f"packed batch requires {byte_count} bytes, only {available} available"
            )

    def _finish_commit(
        self,
        reservation: BatchReservation,
        source: PackedBatchRef,
        shm: shared_memory.SharedMemory,
    ) -> PackedBatchRef:
        if self._reservations.get(reservation.reservation_id) != reservation:
            raise PackedBatchLeaseError(
                f"packed batch {source.batch_id!r} was reclaimed during transfer"
            )
        lease_id = secrets.token_hex(16)
        ref = source.model_copy(
            update={
                "owner_actor_id": self.owner_actor_id,
                "lease_id": lease_id,
                "shared_memory_name": shm.name,
                "owner_process_id": os.getpid(),
            }
        )
        self._reservations.pop(reservation.reservation_id)
        self._reserved_bytes -= reservation.storage_byte_count
        self._entries[ref.batch_id] = _Entry(shm, ref)
        self._used_bytes += ref.storage_byte_count
        self._created_bytes += ref.storage_byte_count
        self._copied_bytes += ref.storage_byte_count
        self._copy_count += 1
        return ref


class MappedPackedBatch(BaseModel):
    """Zero-copy consumer view; callers must not mutate its immutable tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ref: PackedBatchRef
    tensors: Any
    _shm: Any = None
    _closed: bool = False

    @classmethod
    def open(cls, ref: PackedBatchRef) -> "MappedPackedBatch":
        shm = shared_memory.SharedMemory(name=ref.shared_memory_name)
        if ref.owner_process_id != os.getpid():
            # Python 3.12 has no public `track=False`. The segment belongs to the
            # host inbox, so an unrelated consumer's tracker must not unlink it.
            resource_tracker.unregister(cast(Any, shm)._name, "shared_memory")
        try:
            flat = {
                spec.name: _tensor_from_buffer(_shm_buffer(shm), spec)
                for spec in ref.tensors
            }
            tensors = _unflatten_packed_tensors(flat, ref)
        except BaseException:
            shm.close()
            raise
        mapped = cls(ref=ref, tensors=tensors)
        mapped._shm = shm
        return mapped

    def close(self) -> None:
        if not self._closed:
            self.tensors = None
            gc.collect()
            self._shm.close()
            self._closed = True

    def __enter__(self) -> "MappedPackedBatch":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PackedBatchInbox:
    def __init__(self, *, host_id: str, capacity_bytes: int) -> None:
        self.host_id = host_id
        self.store = SharedMemoryPackedBatchStore(
            owner_actor_id=f"packed_batch_inbox:{host_id}",
            capacity_bytes=capacity_bytes,
        )

    async def receive_rdma(
        self, ref: PackedBatchRef, rdma_buffer: Any, *, timeout_s: float
    ) -> PackedBatchRef:
        reservation = self.store.reserve(ref)
        try:
            return await self.store.commit_rdma(
                reservation.reservation_id,
                ref,
                rdma_buffer,
                timeout_s=timeout_s,
            )
        except BaseException:
            self.store.abort(reservation.reservation_id)
            raise

    async def drop(self, ref: PackedBatchRef) -> None:
        self.store.drop(ref)

    async def reclaim(self, batch_id: str, *, fence: bool = True) -> bool:
        return self.store.reclaim(batch_id, fence=fence)


class RDMABatchSourceEndpoint(Protocol):
    async def publish(self, ref: PackedBatchRef) -> Any: ...

    async def drop(self, batch_id: str) -> None: ...

    async def note_transmitted(self, byte_count: int) -> None: ...


class RDMABatchInboxEndpoint(Protocol):
    async def receive_rdma(
        self, ref: PackedBatchRef, rdma_buffer: Any, *, timeout_s: float
    ) -> PackedBatchRef: ...

    async def drop(self, ref: PackedBatchRef) -> None: ...


async def fanout_rdma_packed_batch(
    *,
    ref: PackedBatchRef,
    source_endpoint: RDMABatchSourceEndpoint,
    inboxes: Mapping[str, RDMABatchInboxEndpoint],
    timeout_s: float,
) -> dict[str, PackedBatchRef]:
    """Publish once, read once per host, and always drop the actor-owned handle."""

    handle = await source_endpoint.publish(ref)
    try:
        tasks = {
            host_id: asyncio.create_task(
                inbox.receive_rdma(ref, handle, timeout_s=timeout_s)
            )
            for host_id, inbox in inboxes.items()
        }
        try:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        except BaseException:
            for task in tasks.values():
                task.cancel()
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            await _release_transferred(
                inboxes,
                [
                    result if isinstance(result, BaseException) else (host_id, result)
                    for host_id, result in zip(tasks, results, strict=True)
                ],
            )
            raise
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            await _release_transferred(
                inboxes,
                [
                    result if isinstance(result, BaseException) else (host_id, result)
                    for host_id, result in zip(tasks, results, strict=True)
                ],
            )
            raise failures[0]
        await source_endpoint.note_transmitted(len(inboxes) * ref.storage_byte_count)
        return dict(zip(tasks, cast(list[PackedBatchRef], results), strict=True))
    finally:
        await source_endpoint.drop(ref.batch_id)


async def _release_transferred(
    inboxes: Mapping[str, RDMABatchInboxEndpoint],
    results: list[tuple[str, PackedBatchRef] | BaseException],
) -> None:
    for result in results:
        if not isinstance(result, BaseException):
            host_id, destination_ref = result
            await inboxes[host_id].drop(destination_ref)


def _flatten_packed_tensors(
    tensors: Any,
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    import torch

    required = (
        "tokens",
        "group_ids",
        "parent_ids",
        "input_pos",
        "assistant_mask",
        "logprobs",
        "advantages",
        "weights",
    )
    flat: list[tuple[str, Any]] = []
    for name in required:
        tensor = tensors[name]
        _validate_tensor(name, tensor, torch)
        flat.append((name, tensor))
    shape = tuple(tensors["tokens"].shape)
    if len(shape) != 2 or any(tuple(tensors[name].shape) != shape for name in required):
        raise ValueError(
            "core packed tensors must share [num_sequences, sequence_length]"
        )
    for list_name in ("pixel_values", "image_grid_thw"):
        for index, tensor in enumerate(tensors[list_name]):
            if tensor is not None:
                _validate_tensor(f"{list_name}/{index}", tensor, torch)
                flat.append((f"{list_name}/{index}", tensor))
    original = tensors.get("original_logprobs")
    if original is not None:
        _validate_tensor("original_logprobs", original, torch)
        if tuple(original.shape) != shape:
            raise ValueError("original_logprobs must match the core packed shape")
        flat.append(("original_logprobs", original))
    replay = tensors.get("moe_routing_replay")
    replay_spec = None
    if replay is not None:
        for name in ("expert_indices", "token_mask"):
            tensor = getattr(replay, name)
            _validate_tensor(f"moe_routing_replay/{name}", tensor, torch)
            flat.append((f"moe_routing_replay/{name}", tensor))
        replay_spec = MoeRoutingReplaySpec(
            num_layers=replay.num_layers,
            topk=replay.topk,
            num_experts=replay.num_experts,
            packed_tokens=replay.pack_stats.packed_tokens,
        )
    return flat, {
        "num_sequences": shape[0],
        "sequence_length": shape[1],
        "pixel_values_present": tuple(x is not None for x in tensors["pixel_values"]),
        "image_grid_thw_present": tuple(
            x is not None for x in tensors["image_grid_thw"]
        ),
        "moe_routing_replay": replay_spec,
        "prefix_tree_packing_stats": tensors.get("prefix_tree_packing_stats"),
    }


def _validate_tensor(name: str, tensor: Any, torch: Any) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CPU tensor")


def _layout(flat: list[tuple[str, Any]]) -> tuple[tuple[TensorSpec, ...], int]:
    offset = 0
    specs = []
    for name, tensor in flat:
        element_size = tensor.element_size()
        offset = (offset + element_size - 1) // element_size * element_size
        byte_count = tensor.numel() * element_size
        specs.append(
            TensorSpec(
                name=name,
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=tuple(tensor.shape),
                offset=offset,
                byte_count=byte_count,
            )
        )
        offset += byte_count
    return tuple(specs), max(offset, 1)


def _tensor_from_buffer(buffer: memoryview, spec: TensorSpec) -> Any:
    import torch

    dtype = getattr(torch, spec.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {spec.dtype!r}")
    return torch.frombuffer(
        buffer, dtype=dtype, count=_numel(spec.shape), offset=spec.offset
    ).reshape(spec.shape)


def _shm_buffer(shm: shared_memory.SharedMemory) -> memoryview:
    buffer = shm.buf
    if buffer is None:
        raise RuntimeError("shared-memory buffer is closed")
    return buffer


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError("tensor dimensions must be non-negative")
        result *= dimension
    return result


def _logical_ref(ref: PackedBatchRef) -> dict[str, Any]:
    return ref.model_dump(
        exclude={
            "owner_actor_id",
            "lease_id",
            "shared_memory_name",
            "owner_process_id",
        }
    )


def _unflatten_packed_tensors(flat: dict[str, Any], ref: PackedBatchRef) -> Any:
    from art.preprocessing.moe_routing import (
        MoeRoutingPackStats,
        PackedMoeRoutingReplay,
    )

    tensors: dict[str, Any] = {
        name: flat[name]
        for name in (
            "tokens",
            "group_ids",
            "parent_ids",
            "input_pos",
            "assistant_mask",
            "logprobs",
            "advantages",
            "weights",
        )
    }
    for name, present in (
        ("pixel_values", ref.pixel_values_present),
        ("image_grid_thw", ref.image_grid_thw_present),
    ):
        tensors[name] = [
            flat[f"{name}/{index}"] if value else None
            for index, value in enumerate(present)
        ]
    replay = ref.moe_routing_replay
    tensors["moe_routing_replay"] = (
        PackedMoeRoutingReplay(
            expert_indices=flat["moe_routing_replay/expert_indices"],
            token_mask=flat["moe_routing_replay/token_mask"],
            num_layers=replay.num_layers,
            topk=replay.topk,
            num_experts=replay.num_experts,
            pack_stats=MoeRoutingPackStats(packed_tokens=replay.packed_tokens),
        )
        if replay is not None
        else None
    )
    if "original_logprobs" in flat:
        tensors["original_logprobs"] = flat["original_logprobs"]
    if ref.prefix_tree_packing_stats is not None:
        tensors["prefix_tree_packing_stats"] = (
            ref.prefix_tree_packing_stats.model_dump()
        )
    return tensors
