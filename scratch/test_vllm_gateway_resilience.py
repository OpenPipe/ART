import asyncio
import socket
from typing import Any, Literal

from aiohttp import ClientError, ClientSession, web
import pytest

from art.distributed.specs import EndpointSpec
from art.distributed.vllm_gateway import VllmGateway, _TimedBytesPayload
from art.distributed.vllm_router import (
    ReplicaTelemetry,
    RoutableReplica,
    RoutingTable,
)


def _replica(
    replica_id: str,
    port: int,
    *,
    phase: Literal["ready", "quarantined"] = "ready",
    age_s: float = 0.0,
) -> RoutableReplica:
    return RoutableReplica(
        replica_id=replica_id,
        endpoint=EndpointSpec(host="127.0.0.1", port=port),
        phase=phase,
        generation=0,
        generation_digest=f"generation-{replica_id}",
        committed_version="0",
        policy_digest="digest-0",
        update_identity="update-0",
        telemetry=ReplicaTelemetry(
            observed_at=asyncio.get_running_loop().time() - age_s,
            in_flight=0,
            capacity=8,
        ),
        quarantine_reason="test quarantine" if phase == "quarantined" else None,
    )


def _table(*replicas: RoutableReplica) -> RoutingTable:
    return RoutingTable(
        policy_generation=0,
        policy_version="0",
        policy_digest="digest-0",
        update_identity="update-0",
        replicas=replicas,
    )


async def _serve(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server: Any = site._server
    assert server is not None
    port = server.sockets[0].getsockname()[1]
    return runner, int(port)


def _unused_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _payload() -> dict[str, Any]:
    return {"model": "model@0", "messages": []}


async def test_proxy_returns_502_for_upstream_connection_failure() -> None:
    gateway = VllmGateway(
        _table(_replica("failed", _unused_port())),
        upstream_connect_timeout_s=0.05,
    )
    port = await gateway.start()
    try:
        async with ClientSession() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions", json=_payload()
            )
            assert response.status == 502
            assert await response.text() == "upstream request failed"
    finally:
        await gateway.close()


async def test_proxy_returns_504_when_upstream_first_byte_stalls() -> None:
    never = asyncio.Event()

    async def completion(request: web.Request) -> web.StreamResponse:
        await request.read()
        response = web.StreamResponse()
        await response.prepare(request)
        await never.wait()
        return response

    async def metrics(_request: web.Request) -> web.Response:
        return web.json_response(
            {"metrics": {"num_requests_running": 0, "max_num_seqs": 8}}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_get("/art/metrics", metrics)
    upstream, upstream_port = await _serve(app)
    gateway = VllmGateway(
        _table(_replica("slow", upstream_port)), upstream_read_timeout_s=0.02
    )
    port = await gateway.start()
    try:
        async with ClientSession() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions", json=_payload()
            )
            assert response.status == 504
            assert await response.text() == "upstream request timed out"
    finally:
        await gateway.close()
        never.set()
        await upstream.cleanup()


async def test_proxy_returns_504_when_upstream_pool_is_exhausted() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def completion(request: web.Request) -> web.Response:
        await request.read()
        entered.set()
        await release.wait()
        return web.Response(text="ok")

    async def metrics(_request: web.Request) -> web.Response:
        return web.json_response(
            {"metrics": {"num_requests_running": 0, "max_num_seqs": 8}}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_get("/art/metrics", metrics)
    upstream, upstream_port = await _serve(app)
    gateway = VllmGateway(
        _table(_replica("busy", upstream_port)),
        upstream_pool_size=1,
        upstream_pool_timeout_s=0.02,
    )
    port = await gateway.start()
    first: asyncio.Task[Any] | None = None
    try:
        async with ClientSession() as client:
            first = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=_payload(),
                )
            )
            await asyncio.wait_for(entered.wait(), 1.0)
            response = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions", json=_payload()
            )
            assert response.status == 504
            release.set()
            assert (await first).status == 200
    finally:
        release.set()
        if first is not None:
            await asyncio.gather(first, return_exceptions=True)
        await gateway.close()
        await upstream.cleanup()


async def test_request_body_write_has_a_deadline() -> None:
    class SlowWriter:
        async def write(self, _body: bytes) -> None:
            await asyncio.sleep(1.0)

    body = _TimedBytesPayload(b"request", timeout_s=0.01)
    with pytest.raises(TimeoutError):
        await body.write_with_length(SlowWriter(), None)


async def test_health_and_metrics_only_probe_surviving_replicas() -> None:
    health_calls = 0
    metrics_calls = 0

    async def health(_request: web.Request) -> web.Response:
        nonlocal health_calls
        health_calls += 1
        return web.Response()

    async def metrics(_request: web.Request) -> web.Response:
        nonlocal metrics_calls
        metrics_calls += 1
        return web.json_response(
            {
                "metrics": {
                    "num_requests_running": 3,
                    "max_num_seqs": 7,
                    "kv_cache_usage_perc": 0.25,
                }
            }
        )

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/art/metrics", metrics)
    upstream, upstream_port = await _serve(app)
    dead_port = _unused_port()
    gateway = VllmGateway(
        _table(
            _replica("survivor", upstream_port),
            _replica("quarantined", dead_port, phase="quarantined"),
            _replica("stale", dead_port, age_s=10.0),
        )
    )
    port = await gateway.start()
    try:
        async with ClientSession() as client:
            health_response = await client.get(f"http://127.0.0.1:{port}/health")
            assert health_response.status == 200
            metrics_response = await client.get(f"http://127.0.0.1:{port}/art/metrics")
            assert metrics_response.status == 200
            assert (await metrics_response.json())["metrics"] == {
                "num_requests_running": 3.0,
                "max_num_seqs": 7.0,
                "kv_cache_usage_perc": 0.25,
            }
            assert health_calls == 1
            assert metrics_calls >= 1
            await gateway.router.quarantine(("survivor",), "test quarantine")
            assert (await client.get(f"http://127.0.0.1:{port}/health")).status == 503
            assert (
                await client.get(f"http://127.0.0.1:{port}/art/metrics")
            ).status == 503
    finally:
        await gateway.close()
        await upstream.cleanup()


async def test_close_drains_request_before_closing_upstream_client() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def completion(request: web.Request) -> web.Response:
        await request.read()
        entered.set()
        await release.wait()
        return web.Response(text="drained")

    async def metrics(_request: web.Request) -> web.Response:
        return web.json_response(
            {"metrics": {"num_requests_running": 0, "max_num_seqs": 8}}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_get("/art/metrics", metrics)
    upstream, upstream_port = await _serve(app)
    gateway = VllmGateway(
        _table(_replica("draining", upstream_port)), shutdown_timeout_s=0.5
    )
    port = await gateway.start()
    upstream_client = gateway._session
    assert upstream_client is not None
    request: asyncio.Task[Any] | None = None
    close: asyncio.Task[None] | None = None
    try:
        async with ClientSession() as client:
            request = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=_payload(),
                )
            )
            await asyncio.wait_for(entered.wait(), 1.0)
            close = asyncio.create_task(gateway.close())
            await asyncio.sleep(0.02)
            assert not close.done()
            assert not upstream_client.closed
            release.set()
            response = await request
            assert await response.text() == "drained"
            await close
            assert upstream_client.closed
    finally:
        release.set()
        if request is not None:
            await asyncio.gather(request, return_exceptions=True)
        if close is not None:
            await asyncio.gather(close, return_exceptions=True)
        await gateway.close()
        await upstream.cleanup()


async def test_close_bounds_drain_before_closing_upstream_client() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def completion(request: web.Request) -> web.Response:
        await request.read()
        entered.set()
        await release.wait()
        return web.Response(text="late")

    async def metrics(_request: web.Request) -> web.Response:
        return web.json_response(
            {"metrics": {"num_requests_running": 0, "max_num_seqs": 8}}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_get("/art/metrics", metrics)
    upstream, upstream_port = await _serve(app)
    gateway = VllmGateway(
        _table(_replica("bounded", upstream_port)), shutdown_timeout_s=0.02
    )
    port = await gateway.start()
    upstream_client = gateway._session
    assert upstream_client is not None
    request: asyncio.Task[Any] | None = None
    try:
        async with ClientSession() as client:
            request = asyncio.create_task(
                client.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json=_payload(),
                )
            )
            await asyncio.wait_for(entered.wait(), 1.0)
            await asyncio.wait_for(gateway.close(), 1.0)
            assert upstream_client.closed
    finally:
        release.set()
        if request is not None:
            try:
                await request
            except ClientError:
                pass
        await gateway.close()
        await upstream.cleanup()
