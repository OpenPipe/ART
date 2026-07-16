import asyncio
import struct
from typing import Any

import pytest
import zmq
import zmq.asyncio

from art.distributed.specs import EndpointSpec
from art.distributed.vllm_gateway import VllmGateway
from art.distributed.vllm_kv_events import KvEventSource
from art.distributed.vllm_router import (
    ReplicaTelemetry,
    RoutableReplica,
    RoutingInput,
    RoutingTable,
    VllmPrefixHashConfig,
    canonical_block_hash,
    vllm_request_block_hashes,
)


def _pack(value: Any) -> bytes:
    if value is None:
        return b"\xc0"
    if isinstance(value, float):
        return b"\xcb" + struct.pack(">d", value)
    if isinstance(value, int):
        if 0 <= value < 128:
            return bytes((value,))
        return b"\xcf" + value.to_bytes(8, "big")
    if isinstance(value, bytes):
        return b"\xc6" + len(value).to_bytes(4, "big") + value
    if isinstance(value, str):
        encoded = value.encode()
        return b"\xdb" + len(encoded).to_bytes(4, "big") + encoded
    if isinstance(value, list):
        return b"\xdd" + len(value).to_bytes(4, "big") + b"".join(map(_pack, value))
    raise TypeError(type(value))


def _payload(block_hash: bytes, rank: int = 0) -> bytes:
    return _pack(
        [
            1.5,
            [
                [
                    "BlockStored",
                    [block_hash],
                    None,
                    [1, 2],
                    2,
                    None,
                    "GPU",
                    "model@7",
                    [None],
                    0,
                    "full_attention",
                    None,
                ]
            ],
            rank,
        ]
    )


class FakePublisher:
    def __init__(self, topic: bytes) -> None:
        self.topic = topic
        self.context = zmq.asyncio.Context.instance()
        self.pub = self.context.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.bind("tcp://127.0.0.1:*")
        self.endpoint = self.pub.getsockopt_string(zmq.LAST_ENDPOINT)
        self.replay = self.context.socket(zmq.ROUTER)
        self.replay.setsockopt(zmq.LINGER, 0)
        self.replay.bind("tcp://127.0.0.1:*")
        self.replay_endpoint = self.replay.getsockopt_string(zmq.LAST_ENDPOINT)
        self.buffer: list[tuple[int, bytes]] = []
        self.task = asyncio.create_task(self._serve_replay())

    async def _serve_replay(self) -> None:
        while True:
            identity, empty, start = await self.replay.recv_multipart()
            assert empty == b""
            sequence = int.from_bytes(start, "big")
            for item_sequence, payload in self.buffer:
                if item_sequence >= sequence:
                    await self.replay.send_multipart(
                        [identity, b"", item_sequence.to_bytes(8, "big"), payload]
                    )
            await self.replay.send_multipart(
                [identity, b"", (-1).to_bytes(8, "big", signed=True), b""]
            )

    async def publish(self, sequence: int, payload: bytes) -> None:
        await self.pub.send_multipart(
            [self.topic, sequence.to_bytes(8, "big"), payload]
        )

    async def close(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        self.replay.close(linger=0)
        self.pub.close(linger=0)


async def _wait_for(predicate: Any) -> None:
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not reached")


def test_policy_cache_salt_marker_is_replaced_exactly() -> None:
    config = VllmPrefixHashConfig(
        block_size=2,
        lora_name="model:active",
        policy_cache_key="model:active:7",
    )
    tokens = (1, 2)
    assert vllm_request_block_hashes(
        tokens,
        config,
        cache_salt="tenant|art_policy_cache_salt=model:active:6",
    ) == vllm_request_block_hashes(tokens, config, cache_salt="tenant")
    assert vllm_request_block_hashes(
        tokens, config, cache_salt="art_policy_cache_salt=model:active:6"
    ) == vllm_request_block_hashes(tokens, config)


@pytest.mark.asyncio
async def test_fake_vllm_publisher_replay_gap_decode_and_close() -> None:
    config = VllmPrefixHashConfig(block_size=2, lora_name="model@7")
    request_hash = vllm_request_block_hashes((1, 2), config)[0]
    fake = FakePublisher(b"art.test")
    fake.buffer = [(0, _payload(request_hash))]
    now = asyncio.get_running_loop().time()

    def replica(replica_id: str, port: int, load: int) -> RoutableReplica:
        return RoutableReplica(
            replica_id=replica_id,
            endpoint=EndpointSpec(host="127.0.0.1", port=port),
            phase="ready",
            generation=3,
            generation_digest=f"digest-{replica_id}",
            committed_version="7",
            policy_digest="policy-7",
            update_identity="update-7",
            telemetry=ReplicaTelemetry(observed_at=now, in_flight=load, capacity=8),
        )

    table = RoutingTable(
        policy_generation=0,
        policy_version="7",
        policy_digest="policy-7",
        update_identity="update-7",
        replicas=(replica("r0", 1, 2), replica("r1", 2, 0)),
        prefix_hash=config,
    )
    source = KvEventSource(
        replica_id="r0",
        generation=3,
        publisher_rank=0,
        endpoint=fake.endpoint,
        replay_endpoint=fake.replay_endpoint,
        topic="art.test",
    )
    gateway = VllmGateway(table, kv_event_sources=(source,))
    subscriber = next(iter(gateway._kv_subscribers.values()))
    await gateway.start()
    try:
        await _wait_for(lambda: ("r0", 0) in gateway.router._kv)
        reservation = await gateway.router.acquire(
            RoutingInput(
                policy_version="7",
                policy_digest="policy-7",
                prompt_token_ids=(1, 2),
            ),
            timeout_s=1,
        )
        assert reservation.replica.replica_id == "r0"
        await reservation.release()

        replacement = b"replacement"
        fake.buffer = [(2, _payload(replacement))]
        await fake.publish(2, _payload(replacement))
        await _wait_for(
            lambda: (
                canonical_block_hash(replacement)
                in next(iter(gateway.router._kv[("r0", 0)].blocks.values()))
            )
        )
        assert canonical_block_hash(request_hash) not in next(
            iter(gateway.router._kv[("r0", 0)].blocks.values())
        )

        await fake.publish(3, b"\xc1")
        await _wait_for(lambda: ("r0", 0) not in gateway.router._kv)
    finally:
        await gateway.close()
        await fake.close()
    assert subscriber._task is None
