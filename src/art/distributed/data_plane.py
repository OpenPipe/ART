from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from multiprocessing import resource_tracker, shared_memory
import os
import secrets
import socket
from threading import Thread
import time
from typing import Any, Coroutine, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

PACKED_BATCH_FORMAT = "art_packed_rl_v2"
_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "uint16": 2,
    "int8": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}
_STREAM_CHUNK_BYTES = 4 << 20
T = TypeVar("T")


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
    num_experts: int = Field(ge=1, le=65_536)
    packed_tokens: int = Field(ge=0)


class PrefixTreePackingStatsSpec(_Contract):
    logical_tokens: int = Field(ge=0)
    physical_tokens: int = Field(ge=0)
    policy_token_counts: dict[int, int] | None

    @model_validator(mode="after")
    def _validate_policy_token_counts(self) -> "PrefixTreePackingStatsSpec":
        counts = self.policy_token_counts
        if counts is not None and any(
            version < 0 or count <= 0 for version, count in counts.items()
        ):
            raise ValueError(
                "policy_token_counts require non-negative versions and positive counts"
            )
        return self


class TokenizedOutputMapSpec(_Contract):
    packed_positions: tuple[tuple[int, ...], ...] = Field(min_length=1)
    candidate_counts: tuple[int, ...] = Field(min_length=1)


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
    training_kind: Literal["rl", "tokenized"] = "rl"
    tokenized_output_map: TokenizedOutputMapSpec | None = None
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
        tokenized_names = {
            "target_tokens",
            "loss_weights",
            "behavior_logprobs",
            "token_advantages",
        }
        if self.training_kind == "tokenized":
            if self.tokenized_output_map is None or not tokenized_names <= specs.keys():
                raise ValueError("tokenized packed-batch manifest is incomplete")
            target_shape = specs["target_tokens"].shape
            if (
                len(target_shape) != 3
                or target_shape[:2] != core_shape
                or target_shape[2] < 1
                or specs["target_tokens"].dtype != "int64"
                or any(
                    specs[name].dtype != "float32" or specs[name].shape != target_shape
                    for name in tokenized_names - {"target_tokens"}
                )
            ):
                raise ValueError("tokenized packed tensor dtype or shape is invalid")
            expected_optional |= tokenized_names
            output_map = self.tokenized_output_map
            if (
                len(output_map.packed_positions) != len(output_map.candidate_counts)
                or any(
                    count < 1 or count > target_shape[2]
                    for count in output_map.candidate_counts
                )
                or any(
                    position < 0
                    or position >= self.num_sequences * self.sequence_length
                    for positions in output_map.packed_positions
                    for position in positions
                )
            ):
                raise ValueError("tokenized output map is invalid")
        elif self.tokenized_output_map is not None:
            raise ValueError("RL packed batches cannot carry a tokenized output map")
        replay_names = {"moe_routing_replay/expert_indices"}
        if self.moe_routing_replay is not None:
            if not replay_names <= specs.keys():
                raise ValueError("MoE routing replay manifest is incomplete")
            expected_optional |= replay_names
            replay = self.moe_routing_replay
            replay_dtype = "uint8" if replay.num_experts <= 256 else "uint16"
            if specs[
                "moe_routing_replay/expert_indices"
            ].dtype != replay_dtype or specs[
                "moe_routing_replay/expert_indices"
            ].shape != (replay.num_layers, *core_shape, replay.topk):
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


class PackedBatchTransfer(_Contract):
    batch_id: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    token: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=1)


class ByteStreamTransfer(_Contract):
    stream_id: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    token: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=1)


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
                training_kind=metadata["training_kind"],
                tokenized_output_map=metadata["tokenized_output_map"],
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

    async def commit_stream(
        self,
        reservation_id: str,
        source: PackedBatchRef,
        transfer: PackedBatchTransfer,
        *,
        timeout_s: float,
    ) -> PackedBatchRef:
        reservation = self._reservations.get(reservation_id)
        if reservation is None or reservation.batch_id != source.batch_id:
            raise PackedBatchLeaseError(
                "unknown or mismatched packed-batch reservation"
            )
        if (
            transfer.batch_id != source.batch_id
            or transfer.byte_count != reservation.storage_byte_count
        ):
            raise PackedBatchLeaseError(
                "packed-batch transfer does not match reservation"
            )
        shm = shared_memory.SharedMemory(
            create=True, size=reservation.storage_byte_count
        )
        try:
            from art.utils.lifecycle import complete_to_thread

            _, cancelled = await complete_to_thread(
                lambda: _receive_stream(transfer, shm, timeout_s)
            )
            if cancelled is not None:
                raise cancelled
            return self._finish_commit(reservation, source, shm)
        except BaseException:
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
            self._shm.close()
            self._closed = True

    def __enter__(self) -> "MappedPackedBatch":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ByteStreamServerLoop:
    """A process-local I/O loop that cannot be blocked by rollout code."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run, name="art-byte-stream", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def submit(self, coroutine: Coroutine[Any, Any, T]) -> T:
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        )

    async def close(self) -> None:
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            await asyncio.to_thread(self._thread.join)
            self._loop.close()


class _AuthenticatedStreamPublisher:
    def __init__(
        self, advertise_host: str, server_loop: ByteStreamServerLoop | None = None
    ) -> None:
        self.advertise_host = advertise_host
        self._server_loop = server_loop
        self._token = secrets.token_bytes(32)
        self._server: Any = None
        self._handlers: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if self._server_loop is not None:
            return await self._server_loop.submit(self._start())
        await self._start()

    async def _start(self) -> None:
        family = socket.getaddrinfo(self.advertise_host, 0, type=socket.SOCK_STREAM)[0][
            0
        ]
        bind_host = "::" if family == socket.AF_INET6 else "0.0.0.0"
        self._server = await asyncio.start_server(
            self._handle, bind_host, 0, family=family
        )

    def _port(self) -> int:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("byte-stream publisher is not listening")
        return int(self._server.sockets[0].getsockname()[1])

    async def close(self) -> None:
        if self._server_loop is not None:
            return await self._server_loop.submit(self._close())
        await self._close()

    async def _close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        for task in self._handlers:
            task.cancel()
        await asyncio.gather(*self._handlers, return_exceptions=True)
        self._handlers.clear()
        self._server = None

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = cast(asyncio.Task[None], asyncio.current_task())
        self._handlers.add(task)
        sent = False
        try:
            token = await reader.readexactly(len(self._token))
            if not secrets.compare_digest(token, self._token):
                return
            await self._write(writer)
            sent = True
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            try:
                writer.close()
                if task.cancelling():
                    writer.transport.abort()
                else:
                    try:
                        await writer.wait_closed()
                    except asyncio.CancelledError:
                        writer.transport.abort()
                        raise
                    except Exception:
                        pass
            finally:
                self._handlers.discard(task)
                if sent and not task.cancelling():
                    self._sent()

    async def _write(self, writer: asyncio.StreamWriter) -> None:
        raise NotImplementedError

    def _sent(self) -> None:
        pass


class ByteStreamPublisher(_AuthenticatedStreamPublisher):
    """Authenticated one-shot transport for immutable byte chunks."""

    def __init__(
        self,
        stream_id: str,
        advertise_host: str,
        chunks: tuple[bytes | memoryview, ...],
        on_sent: Callable[[], None] | None,
        server_loop: ByteStreamServerLoop | None,
    ) -> None:
        super().__init__(advertise_host, server_loop)
        self.stream_id = stream_id
        self.chunks = chunks
        self.on_sent = on_sent
        self.byte_count = sum(map(len, chunks))
        if not stream_id or self.byte_count < 1:
            raise ValueError("byte stream ID and payload must be non-empty")

    @classmethod
    async def create(
        cls,
        stream_id: str,
        chunks: tuple[bytes | memoryview, ...],
        *,
        advertise_host: str,
        on_sent: Callable[[], None] | None = None,
        server_loop: ByteStreamServerLoop | None = None,
    ) -> "ByteStreamPublisher":
        publisher = cls(stream_id, advertise_host, chunks, on_sent, server_loop)
        await publisher.start()
        return publisher

    @property
    def transfer(self) -> ByteStreamTransfer:
        return ByteStreamTransfer(
            stream_id=self.stream_id,
            host=self.advertise_host,
            port=self._port(),
            token=self._token.hex(),
            byte_count=self.byte_count,
        )

    async def _write(self, writer: asyncio.StreamWriter) -> None:
        for chunk in self.chunks:
            await _write_stream_chunk(writer, chunk)

    def _sent(self) -> None:
        if self.on_sent is not None:
            self.on_sent()


class PackedBatchPublisher(_AuthenticatedStreamPublisher):
    """Batch-scoped authenticated stream over the cluster's routable TCP fabric."""

    def __init__(
        self,
        ref: PackedBatchRef,
        advertise_host: str,
        shm: shared_memory.SharedMemory,
    ) -> None:
        super().__init__(advertise_host)
        self.ref = ref
        self.shm = shm

    @classmethod
    async def create(
        cls, ref: PackedBatchRef, *, advertise_host: str
    ) -> "PackedBatchPublisher":
        shm = shared_memory.SharedMemory(name=ref.shared_memory_name)
        if ref.owner_process_id != os.getpid():
            resource_tracker.unregister(cast(Any, shm)._name, "shared_memory")
        publisher = cls(ref, advertise_host, shm)
        try:
            await publisher.start()
            return publisher
        except BaseException:
            shm.close()
            raise

    @property
    def transfer(self) -> PackedBatchTransfer:
        return PackedBatchTransfer(
            batch_id=self.ref.batch_id,
            host=self.advertise_host,
            port=self._port(),
            token=self._token.hex(),
            byte_count=self.ref.storage_byte_count,
        )

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            self.shm.close()

    async def _write(self, writer: asyncio.StreamWriter) -> None:
        source = _shm_buffer(self.shm)[: self.ref.storage_byte_count]
        try:
            await _write_stream_chunk(writer, source)
        finally:
            source.release()


class PackedBatchInbox:
    def __init__(self, *, host_id: str, capacity_bytes: int) -> None:
        self.host_id = host_id
        self.store = SharedMemoryPackedBatchStore(
            owner_actor_id=f"packed_batch_inbox:{host_id}",
            capacity_bytes=capacity_bytes,
        )

    async def receive(
        self, ref: PackedBatchRef, transfer: PackedBatchTransfer, *, timeout_s: float
    ) -> PackedBatchRef:
        reservation = self.store.reserve(ref)
        try:
            return await self.store.commit_stream(
                reservation.reservation_id,
                ref,
                transfer,
                timeout_s=timeout_s,
            )
        except BaseException:
            self.store.abort(reservation.reservation_id)
            raise

    async def drop(self, ref: PackedBatchRef) -> None:
        self.store.drop(ref)

    async def reclaim(self, batch_id: str, *, fence: bool = True) -> bool:
        return self.store.reclaim(batch_id, fence=fence)


class PackedBatchSourceEndpoint(Protocol):
    async def publish(self, ref: PackedBatchRef) -> PackedBatchTransfer: ...

    async def drop(self, batch_id: str) -> None: ...

    async def note_transmitted(self, byte_count: int) -> None: ...


class PackedBatchInboxEndpoint(Protocol):
    async def receive(
        self, ref: PackedBatchRef, transfer: PackedBatchTransfer, *, timeout_s: float
    ) -> PackedBatchRef: ...

    async def drop(self, ref: PackedBatchRef) -> None: ...


async def fanout_packed_batch(
    *,
    ref: PackedBatchRef,
    source_endpoint: PackedBatchSourceEndpoint,
    inboxes: Mapping[str, PackedBatchInboxEndpoint],
    timeout_s: float,
) -> dict[str, PackedBatchRef]:
    """Publish once, stream once per host, and always drop the source listener."""

    transfer = await source_endpoint.publish(ref)
    try:
        tasks = {
            host_id: asyncio.create_task(
                inbox.receive(ref, transfer, timeout_s=timeout_s)
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
    inboxes: Mapping[str, PackedBatchInboxEndpoint],
    results: list[tuple[str, PackedBatchRef] | BaseException],
) -> None:
    for result in results:
        if not isinstance(result, BaseException):
            host_id, destination_ref = result
            await inboxes[host_id].drop(destination_ref)


async def receive_byte_stream(
    transfer: ByteStreamTransfer, *, timeout_s: float
) -> bytearray:
    from art.utils.lifecycle import complete_to_thread

    payload = bytearray(transfer.byte_count)
    destination = memoryview(payload)
    try:
        _, cancelled = await complete_to_thread(
            lambda: _receive_into_stream(transfer, destination, timeout_s)
        )
        if cancelled is not None:
            raise cancelled
        return payload
    finally:
        destination.release()


def _receive_stream(
    transfer: PackedBatchTransfer,
    shm: shared_memory.SharedMemory,
    timeout_s: float,
) -> None:
    destination = _shm_buffer(shm)[: transfer.byte_count]
    try:
        _receive_into_stream(transfer, destination, timeout_s)
    finally:
        destination.release()


def _receive_into_stream(
    transfer: PackedBatchTransfer | ByteStreamTransfer,
    destination: memoryview,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    with socket.create_connection(
        (transfer.host, transfer.port), timeout=max(0.001, timeout_s)
    ) as connection:
        connection.sendall(bytes.fromhex(transfer.token))
        offset = 0
        while offset < len(destination):
            connection.settimeout(max(0.001, deadline - time.monotonic()))
            received = connection.recv_into(destination[offset:])
            if not received:
                raise ConnectionError(
                    f"byte stream ended after {offset} of {len(destination)} bytes"
                )
            offset += received


async def _write_stream_chunk(
    writer: asyncio.StreamWriter, source: bytes | memoryview
) -> None:
    view = memoryview(source).cast("B")
    try:
        for offset in range(0, len(view), _STREAM_CHUNK_BYTES):
            writer.write(view[offset : offset + _STREAM_CHUNK_BYTES])
            await writer.drain()
    finally:
        view.release()


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
    tokenized_names = (
        "target_tokens",
        "loss_weights",
        "behavior_logprobs",
        "token_advantages",
    )
    tokenized = [name for name in tokenized_names if name in tensors]
    if tokenized and len(tokenized) != len(tokenized_names):
        raise ValueError("tokenized packed tensors are incomplete")
    for name in tokenized:
        tensor = tensors[name]
        _validate_tensor(name, tensor, torch)
        if tensor.ndim != 3 or tuple(tensor.shape[:2]) != shape:
            raise ValueError("tokenized packed tensors must have shape [B, S, K]")
        flat.append((name, tensor))
    replay = tensors.get("moe_routing_replay")
    replay_spec = None
    if replay is not None:
        tensor = replay.expert_indices
        _validate_tensor("moe_routing_replay/expert_indices", tensor, torch)
        flat.append(("moe_routing_replay/expert_indices", tensor))
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
        "training_kind": "tokenized" if tokenized else "rl",
        "tokenized_output_map": (
            None
            if (output_map := tensors.get("tokenized_output_map")) is None
            else output_map.model_dump()
        ),
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
            num_experts=replay.num_experts,
            pack_stats=MoeRoutingPackStats(packed_tokens=replay.packed_tokens),
        )
        if replay is not None
        else None
    )
    if "original_logprobs" in flat:
        tensors["original_logprobs"] = flat["original_logprobs"]
    for name in (
        "target_tokens",
        "loss_weights",
        "behavior_logprobs",
        "token_advantages",
    ):
        if name in flat:
            tensors[name] = flat[name]
    if ref.prefix_tree_packing_stats is not None:
        tensors["prefix_tree_packing_stats"] = (
            ref.prefix_tree_packing_stats.model_dump()
        )
    return tensors
