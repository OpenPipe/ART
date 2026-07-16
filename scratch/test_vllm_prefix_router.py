import asyncio

from aiohttp import ClientSession, web

from art.distributed.specs import EndpointSpec
from art.distributed.vllm_gateway import VllmGateway
from art.distributed.vllm_router import (
    KvCacheEvent,
    PrefixBlockHashes,
    ReplicaRouter,
    ReplicaTelemetry,
    RoutableReplica,
    RoutingInput,
    RoutingTable,
    canonical_block_hash,
)


def _replica(
    replica_id: str,
    port: int,
    *,
    load: int = 0,
    publishers: int = 1,
) -> RoutableReplica:
    return RoutableReplica(
        replica_id=replica_id,
        endpoint=EndpointSpec(host="127.0.0.1", port=port),
        phase="ready",
        generation=3,
        generation_digest=f"generation-{replica_id}",
        committed_version="7",
        policy_digest="digest-7",
        update_identity="update-7",
        telemetry=ReplicaTelemetry(
            observed_at=asyncio.get_running_loop().time(),
            in_flight=load,
            capacity=8,
        ),
        kv_event_publishers=publishers,
    )


def _table(*replicas: RoutableReplica) -> RoutingTable:
    return RoutingTable(
        policy_generation=7,
        policy_version="7",
        policy_digest="digest-7",
        update_identity="update-7",
        replicas=replicas,
    )


def _event(
    replica_id: str,
    sequence: int,
    operation: str,
    hashes: tuple[str, ...] = (),
    *,
    publisher_rank: int = 0,
    block_size: int | None = None,
) -> KvCacheEvent:
    return KvCacheEvent(
        replica_id=replica_id,
        generation=3,
        publisher_rank=publisher_rank,
        sequence=sequence,
        version="cache-7",
        block_size=block_size,
        operation=operation,
        block_hashes=hashes,
    )


def _request(*hashes: str) -> RoutingInput:
    return RoutingInput(
        policy_version="7",
        policy_digest="digest-7",
        prefix=PrefixBlockHashes(version="cache-7", block_size=16, hashes=hashes),
    )


async def test_vllm_sequence_applies_the_whole_batch_once() -> None:
    assert canonical_block_hash(15) == "i:f"
    assert canonical_block_hash(b"\x00\xff") == "b:00ff"
    router = ReplicaRouter(
        _table(_replica("r0", 1), _replica("r1", 2, load=1)), random_seed=0
    )
    assert router.apply_kv_events(
        (
            _event("r0", 0, "store", ("a", "b"), block_size=16),
            _event("r0", 0, "store", ("wide",), block_size=32),
        )
    )
    assert router.apply_kv_event(_event("r1", 0, "store", ("a",), block_size=16))

    reservation = await router.acquire(_request("a", "b"), timeout_s=1)
    assert reservation.replica.replica_id == "r0"
    await reservation.release()

    assert not router.apply_kv_event(_event("r0", 2, "store", ("z",), block_size=16))
    reservation = await router.acquire(_request("a", "b"), timeout_s=1)
    assert reservation.replica.replica_id == "r1"
    await reservation.release()

    assert router.apply_kv_event(_event("r0", 1, "store", ("a", "b"), block_size=16))
    reservation = await router.acquire(_request("a", "b"), timeout_s=1)
    assert reservation.replica.replica_id == "r1"
    await reservation.release()


async def test_affinity_requires_every_internal_dp_publisher() -> None:
    router = ReplicaRouter(
        _table(
            _replica("r0", 1, publishers=2),
            _replica("r1", 2, load=1),
        ),
        random_seed=0,
    )
    router.apply_kv_event(_event("r0", 0, "store", ("a", "b"), block_size=16))
    router.apply_kv_event(
        _event(
            "r0",
            0,
            "store",
            ("a",),
            publisher_rank=1,
            block_size=16,
        )
    )
    router.apply_kv_event(_event("r1", 0, "store", ("a", "b"), block_size=16))

    reservation = await router.acquire(_request("a", "b"), timeout_s=1)
    assert reservation.replica.replica_id == "r1"
    await reservation.release()


async def test_update_identity_invalidates_affinity() -> None:
    router = ReplicaRouter(
        _table(_replica("r0", 1, load=2), _replica("r1", 2, load=1)),
        random_seed=0,
    )
    router.apply_kv_event(_event("r0", 0, "store", ("a", "b"), block_size=16))
    router.apply_kv_event(_event("r1", 0, "store", ("a",), block_size=16))
    reservation = await router.acquire(_request("a", "b"), timeout_s=1)
    assert reservation.replica.replica_id == "r0"
    await reservation.release()

    candidate = router.table.model_copy(
        update={
            "policy_generation": 8,
            "update_identity": "update-8",
            "replicas": tuple(
                replica.model_copy(update={"update_identity": "update-8"})
                for replica in router.table.replicas
            ),
        }
    )
    await router.commit(router.prepare(candidate))
    reservation = await router.acquire(_request("a", "b"), timeout_s=1)
    assert reservation.replica.replica_id == "r1"
    await reservation.release()


async def _upstream(
    *, running: int = 0, waiting: int = 0
) -> tuple[web.AppRunner, int, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    async def completion(request: web.Request) -> web.Response:
        calls.append(await request.json())
        return web.json_response({"ok": True})

    async def metrics(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "metrics": {
                    "num_requests_running": running,
                    "num_requests_waiting": waiting,
                    "max_num_seqs": 2 if running or waiting else 8,
                }
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_get("/art/metrics", metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port, calls


async def test_gateway_strips_typed_routing_hints() -> None:
    upstream, upstream_port, calls = await _upstream()
    gateway = VllmGateway(_table(_replica("r0", upstream_port)))
    gateway_port = await gateway.start("127.0.0.1")
    try:
        gateway.apply_kv_events((_event("r0", 0, "store", ("a", "b"), block_size=16),))
        async with ClientSession() as client:
            response = await client.post(
                f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                json={
                    "model": "model@7",
                    "messages": [],
                    "art_routing": {
                        "stable_key": "scenario-1",
                        "prefix": {
                            "version": "cache-7",
                            "block_size": 16,
                            "hashes": ["a", "b"],
                        },
                    },
                },
            )
            assert response.status == 200
            assert "art_routing" not in calls[0]

            response = await client.post(
                f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                json={
                    "model": "model@7",
                    "messages": [],
                    "art_routing": {"unknown": True},
                },
            )
            assert response.status == 400
            assert len(calls) == 1
    finally:
        await gateway.close()
        await upstream.cleanup()


async def test_gateway_waiting_requests_consume_capacity() -> None:
    upstream, upstream_port, calls = await _upstream(running=1, waiting=1)
    gateway = VllmGateway(_table(_replica("r0", upstream_port)), route_timeout_s=0.05)
    gateway_port = await gateway.start("127.0.0.1")
    try:
        for _ in range(100):
            if gateway.router.table.replicas[0].telemetry.in_flight == 2:
                break
            await asyncio.sleep(0.01)
        async with ClientSession() as client:
            response = await client.post(
                f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                json={"model": "model@7", "messages": []},
            )
            assert response.status == 504
            assert not calls
    finally:
        await gateway.close()
        await upstream.cleanup()
