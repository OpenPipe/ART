from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
import struct
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
import zmq
import zmq.asyncio
from zmq.utils.monitor import recv_monitor_message

from .specs import (
    VLLM_KV_EVENT_BUFFER_STEPS,
    VLLM_KV_EVENT_HWM,
    VLLM_KV_EVENT_SCHEMA_VERSION,
)
from .vllm_router import KvCacheEvent, canonical_block_hash

_LOGGER = logging.getLogger(__name__)
_MAX_PAYLOAD_BYTES = 64 << 20
_MAX_MSGPACK_DEPTH = 32
_MAX_MSGPACK_ITEMS = 1_000_000
_REPLAY_TIMEOUT_S = 5.0
_RECONNECT_DELAY_S = 0.25
_END_SEQUENCE = (-1).to_bytes(8, "big", signed=True)


class KvEventSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    replica_id: str = Field(min_length=1)
    generation: int = Field(ge=0)
    publisher_rank: int = Field(ge=0)
    endpoint: str = Field(min_length=1)
    replay_endpoint: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    version: str = VLLM_KV_EVENT_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_endpoints(self) -> "KvEventSource":
        if not self.endpoint.startswith("tcp://"):
            raise ValueError("KV event endpoint must use tcp://")
        if not self.replay_endpoint.startswith("tcp://"):
            raise ValueError("KV replay endpoint must use tcp://")
        return self


class _MsgpackDecoder:
    def __init__(self, payload: bytes) -> None:
        if len(payload) > _MAX_PAYLOAD_BYTES:
            raise ValueError("MsgPack payload exceeds the configured limit")
        self._payload = memoryview(payload)
        self._position = 0
        self._items = 0

    def decode(self) -> Any:
        value = self._read(0)
        if self._position != len(self._payload):
            raise ValueError("MsgPack payload has trailing bytes")
        return value

    def _take(self, size: int) -> memoryview:
        end = self._position + size
        if size < 0 or end > len(self._payload):
            raise ValueError("truncated MsgPack payload")
        value = self._payload[self._position : end]
        self._position = end
        return value

    def _uint(self, size: int) -> int:
        return int.from_bytes(self._take(size), "big")

    def _collection(self, size: int, depth: int, *, mapping: bool) -> Any:
        if size > _MAX_MSGPACK_ITEMS:
            raise ValueError("MsgPack collection exceeds the configured limit")
        if mapping:
            result = {}
            for _ in range(size):
                key = self._read(depth)
                try:
                    result[key] = self._read(depth)
                except TypeError as error:
                    raise ValueError("MsgPack map key is not hashable") from error
            return result
        return [self._read(depth) for _ in range(size)]

    def _read(self, depth: int) -> Any:
        if depth > _MAX_MSGPACK_DEPTH:
            raise ValueError("MsgPack nesting exceeds the configured limit")
        self._items += 1
        if self._items > _MAX_MSGPACK_ITEMS:
            raise ValueError("MsgPack item count exceeds the configured limit")
        marker = self._uint(1)
        if marker <= 0x7F:
            return marker
        if marker >= 0xE0:
            return marker - 256
        if 0x80 <= marker <= 0x8F:
            return self._collection(marker & 0x0F, depth + 1, mapping=True)
        if 0x90 <= marker <= 0x9F:
            return self._collection(marker & 0x0F, depth + 1, mapping=False)
        if 0xA0 <= marker <= 0xBF:
            return bytes(self._take(marker & 0x1F)).decode("utf-8")
        if marker == 0xC0:
            return None
        if marker in (0xC2, 0xC3):
            return marker == 0xC3
        if marker in (0xC4, 0xC5, 0xC6):
            return bytes(self._take(self._uint(1 << (marker - 0xC4))))
        if marker in (0xCA, 0xCB):
            size = 4 if marker == 0xCA else 8
            return struct.unpack(">f" if size == 4 else ">d", self._take(size))[0]
        if 0xCC <= marker <= 0xCF:
            return self._uint(1 << (marker - 0xCC))
        if 0xD0 <= marker <= 0xD3:
            size = 1 << (marker - 0xD0)
            return int.from_bytes(self._take(size), "big", signed=True)
        if marker in (0xD9, 0xDA, 0xDB):
            size = self._uint(1 << (marker - 0xD9))
            return bytes(self._take(size)).decode("utf-8")
        if marker in (0xDC, 0xDD):
            return self._collection(
                self._uint(2 if marker == 0xDC else 4),
                depth + 1,
                mapping=False,
            )
        if marker in (0xDE, 0xDF):
            return self._collection(
                self._uint(2 if marker == 0xDE else 4),
                depth + 1,
                mapping=True,
            )
        raise ValueError(f"unsupported MsgPack marker 0x{marker:02x}")


def _array(value: Any, name: str, minimum: int, maximum: int) -> list[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} must be an array of length {minimum}..{maximum}")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _hashes(value: Any) -> tuple[str, ...]:
    values = _array(value, "block_hashes", 0, _MAX_MSGPACK_ITEMS)
    return tuple(canonical_block_hash(item) for item in values)


def decode_vllm_kv_event_batch(
    payload: bytes, source: KvEventSource, sequence: int
) -> tuple[KvCacheEvent, ...]:
    batch = _array(_MsgpackDecoder(payload).decode(), "event batch", 3, 3)
    if isinstance(batch[0], bool) or not isinstance(batch[0], (int, float)):
        raise ValueError("event timestamp must be numeric")
    events = _array(batch[1], "events", 0, _MAX_MSGPACK_ITEMS)
    if _integer(batch[2], "data_parallel_rank") != source.publisher_rank:
        raise ValueError("event data_parallel_rank does not match its publisher")
    normalized: list[KvCacheEvent] = []
    for raw in events:
        event = _array(raw, "KV event", 1, 12)
        tag = event[0]
        if tag == "BlockStored":
            event = _array(event, tag, 8, 12)
            hashes = _hashes(event[1])
            if event[2] is not None:
                canonical_block_hash(event[2])
            token_ids = _array(event[3], "token_ids", 0, _MAX_MSGPACK_ITEMS)
            for token_id in token_ids:
                _integer(token_id, "token_id")
            block_size = _integer(event[4], "block_size", minimum=1)
            if event[5] is not None:
                _integer(event[5], "lora_id")
            if event[6] is not None and not isinstance(event[6], str):
                raise ValueError("event medium must be a string or null")
            if event[7] is not None and not isinstance(event[7], str):
                raise ValueError("event lora_name must be a string or null")
            group_idx = (
                _integer(event[9], "group_idx")
                if len(event) > 9 and event[9] is not None
                else 0
            )
            normalized.append(
                KvCacheEvent(
                    replica_id=source.replica_id,
                    generation=source.generation,
                    publisher_rank=source.publisher_rank,
                    sequence=sequence,
                    version=source.version,
                    block_size=block_size,
                    group_idx=group_idx,
                    operation="store" if event[6] == "GPU" else "noop",
                    block_hashes=hashes if event[6] == "GPU" else (),
                )
            )
        elif tag == "BlockRemoved":
            event = _array(event, tag, 3, 4)
            hashes = _hashes(event[1])
            if event[2] is not None and not isinstance(event[2], str):
                raise ValueError("event medium must be a string or null")
            group_idx = (
                _integer(event[3], "group_idx")
                if len(event) > 3 and event[3] is not None
                else None
            )
            normalized.append(
                KvCacheEvent(
                    replica_id=source.replica_id,
                    generation=source.generation,
                    publisher_rank=source.publisher_rank,
                    sequence=sequence,
                    version=source.version,
                    group_idx=group_idx,
                    operation="remove" if event[2] == "GPU" else "noop",
                    block_hashes=hashes if event[2] == "GPU" else (),
                )
            )
        elif tag == "AllBlocksCleared":
            _array(event, tag, 1, 1)
            normalized.append(
                KvCacheEvent(
                    replica_id=source.replica_id,
                    generation=source.generation,
                    publisher_rank=source.publisher_rank,
                    sequence=sequence,
                    version=source.version,
                    group_idx=None,
                    operation="reset",
                )
            )
        else:
            raise ValueError(f"unknown vLLM KV event tag {tag!r}")
    if normalized:
        return tuple(normalized)
    return (
        KvCacheEvent(
            replica_id=source.replica_id,
            generation=source.generation,
            publisher_rank=source.publisher_rank,
            sequence=sequence,
            version=source.version,
            group_idx=None,
            operation="noop",
        ),
    )


class VllmKvEventSubscriber:
    """One bounded vLLM 0.23 event subscriber and replay client."""

    def __init__(
        self,
        source: KvEventSource,
        apply_batch: Callable[[tuple[KvCacheEvent, ...]], object],
        invalidate: Callable[[str], object],
    ) -> None:
        self.source = source
        self._apply_batch = apply_batch
        self._invalidate_callback = invalidate
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self._next_sequence: int | None = None

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("KV event subscriber is already running")
        self._closing = False
        self._task = asyncio.create_task(
            self._run(),
            name=f"vllm-kv-{self.source.replica_id}-{self.source.publisher_rank}",
        )

    async def close(self) -> None:
        self._closing = True
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _invalidate(self) -> None:
        self._invalidate_callback(self.source.replica_id)

    async def _run(self) -> None:
        reconnect = False
        while not self._closing:
            if reconnect:
                self._invalidate()
                self._next_sequence = None
            try:
                await self._connection()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception(
                    "vLLM KV event subscriber failed for %s rank %d",
                    self.source.replica_id,
                    self.source.publisher_rank,
                )
                self._invalidate()
                self._next_sequence = None
                reconnect = True
                await asyncio.sleep(_RECONNECT_DELAY_S)

    async def _connection(self) -> None:
        context = zmq.asyncio.Context.instance()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.LINGER, 0)
        subscriber.setsockopt(zmq.RCVHWM, VLLM_KV_EVENT_HWM)
        subscriber.setsockopt(zmq.MAXMSGSIZE, _MAX_PAYLOAD_BYTES)
        subscriber.setsockopt(zmq.SUBSCRIBE, self.source.topic.encode())
        monitor = subscriber.get_monitor_socket(
            events=zmq.EVENT_CONNECTED
            | zmq.EVENT_DISCONNECTED
            | zmq.EVENT_CONNECT_RETRIED
            | zmq.EVENT_CLOSED
        )
        subscriber.connect(self.source.endpoint)
        poller = zmq.asyncio.Poller()
        poller.register(subscriber, zmq.POLLIN)
        poller.register(monitor, zmq.POLLIN)
        try:
            await self._replay(
                0 if self._next_sequence is None else self._next_sequence
            )
            while True:
                ready = dict(await poller.poll(1000))
                if monitor in ready:
                    message = await recv_monitor_message(monitor)
                    if int(message["event"]) in {
                        zmq.EVENT_DISCONNECTED,
                        zmq.EVENT_CONNECT_RETRIED,
                        zmq.EVENT_CLOSED,
                    }:
                        raise ConnectionError("vLLM KV publisher disconnected")
                if subscriber in ready:
                    frames = await subscriber.recv_multipart()
                    if len(frames) != 3 or frames[0] != self.source.topic.encode():
                        raise ValueError("invalid vLLM KV event multipart frame")
                    sequence = self._sequence(frames[1])
                    if (
                        self._next_sequence is not None
                        and sequence > self._next_sequence
                    ):
                        missing = self._next_sequence
                        self._invalidate()
                        await self._replay(missing)
                    self._consume(sequence, frames[2])
        finally:
            subscriber.disable_monitor()
            monitor.close(linger=0)
            subscriber.close(linger=0)

    async def _replay(self, start_sequence: int) -> None:
        context = zmq.asyncio.Context.instance()
        replay = context.socket(zmq.DEALER)
        replay.setsockopt(zmq.LINGER, 0)
        replay.setsockopt(zmq.SNDHWM, 1)
        replay.setsockopt(zmq.RCVHWM, VLLM_KV_EVENT_BUFFER_STEPS + 1)
        replay.setsockopt(zmq.MAXMSGSIZE, _MAX_PAYLOAD_BYTES)
        replay.connect(self.source.replay_endpoint)
        try:
            await replay.send_multipart((b"", start_sequence.to_bytes(8, "big")))
            count = 0
            while True:
                frames = await asyncio.wait_for(
                    replay.recv_multipart(), timeout=_REPLAY_TIMEOUT_S
                )
                if len(frames) != 3 or frames[0] != b"" or len(frames[1]) != 8:
                    raise ValueError("invalid vLLM KV replay frame")
                if frames[1] == _END_SEQUENCE:
                    if frames[2]:
                        raise ValueError("invalid vLLM KV replay terminator")
                    return
                count += 1
                if count > VLLM_KV_EVENT_BUFFER_STEPS:
                    raise ValueError("vLLM KV replay exceeded its configured bound")
                self._consume(self._sequence(frames[1]), frames[2])
        finally:
            replay.close(linger=0)

    @staticmethod
    def _sequence(frame: bytes) -> int:
        if len(frame) != 8:
            raise ValueError("vLLM KV sequence must be eight bytes")
        return int.from_bytes(frame, "big")

    def _consume(self, sequence: int, payload: bytes) -> None:
        expected = self._next_sequence
        if expected is not None and sequence < expected:
            return
        if expected is not None and sequence > expected:
            self._invalidate()
        try:
            batch = decode_vllm_kv_event_batch(payload, self.source, sequence)
        except Exception:
            self._invalidate()
            self._next_sequence = sequence + 1
            _LOGGER.exception(
                "invalid vLLM KV event payload for %s rank %d sequence %d",
                self.source.replica_id,
                self.source.publisher_rank,
                sequence,
            )
            return
        self._apply_batch(batch)
        self._next_sequence = sequence + 1
