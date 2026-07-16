import asyncio

from aiohttp import ClientSession, web
import pytest

from art.distributed.specs import EndpointSpec
from art.distributed.vllm_gateway import VllmGateway
from art.distributed.vllm_replica import ReplicaUpdateReport
from art.distributed.vllm_router import (
    ReplicaTelemetry,
    RoutableReplica,
    RoutingTable,
)


async def _upstream(name: str) -> tuple[web.AppRunner, int, list[str]]:
    calls = []

    async def completion(request: web.Request) -> web.StreamResponse:
        calls.append((await request.json())["model"])
        response = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await response.prepare(request)
        await response.write(f"data: {name}\n\n".encode())
        await response.write_eof()
        return response

    async def metrics(_request: web.Request) -> web.Response:
        return web.json_response(
            {"metrics": {"num_requests_running": 0, "max_num_seqs": 2}}
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


def _table(ports: tuple[int, int], generation: int, version: str) -> RoutingTable:
    identity = f"update-{version}"
    replicas = tuple(
        RoutableReplica(
            replica_id=f"r{index}",
            endpoint=EndpointSpec(host="127.0.0.1", port=port),
            phase="ready",
            generation=0,
            generation_digest=f"generation-{index}",
            committed_version=version,
            policy_digest=f"digest-{version}",
            update_identity=identity,
            telemetry=ReplicaTelemetry(
                observed_at=asyncio.get_running_loop().time(),
                in_flight=0,
                capacity=2,
            ),
        )
        for index, port in enumerate(ports)
    )
    return RoutingTable(
        policy_generation=generation,
        policy_version=version,
        policy_digest=f"digest-{version}",
        update_identity=identity,
        replicas=replicas,
    )


async def test_gateway_streams_and_commits_atomically() -> None:
    first, first_port, first_calls = await _upstream("first")
    second, second_port, second_calls = await _upstream("second")
    gateway = VllmGateway(_table((first_port, second_port), 0, "0"))
    gateway_port = await gateway.start("127.0.0.1")
    try:
        async with ClientSession() as client:
            for index in range(16):
                response = await client.post(
                    f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                    json={
                        "model": "model@0",
                        "messages": [{"role": "user", "content": str(index)}],
                        "stream": True,
                    },
                )
                assert response.status == 200
                assert (await response.text()).startswith("data: ")
            assert first_calls and second_calls
            metrics = await client.get(f"http://127.0.0.1:{gateway_port}/art/metrics")
            assert (await metrics.json())["metrics"]["max_num_seqs"] == 4

            await gateway.add_policy(_table((first_port, second_port), 0, "0"))
            table = _table((first_port, second_port), 1, "1")
            reports = tuple(
                ReplicaUpdateReport(
                    replica_id=replica.replica_id,
                    generation=replica.generation,
                    generation_digest=replica.generation_digest,
                    policy_version="1",
                    policy_digest="digest-1",
                    update_identity="update-1",
                )
                for replica in table.replicas
            )
            await gateway.commit(table, reports)
            old = await client.post(
                f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                json={"model": "model@0", "messages": []},
            )
            assert old.status == 200
            await gateway.remove_policy("0")
            old = await client.post(
                f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                json={"model": "model@0", "messages": []},
            )
            assert old.status == 503
            new = await client.post(
                f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                json={"model": "model@1", "messages": []},
            )
            assert new.status == 200
    finally:
        await gateway.close()
        await first.cleanup()
        await second.cleanup()


async def test_gateway_rejects_public_unauthenticated_bind() -> None:
    gateway = VllmGateway(_table((1, 2), 0, "0"))
    with pytest.raises(ValueError, match="requires inbound authentication"):
        await gateway.start("0.0.0.0")
    with pytest.raises(ValueError, match="has no credentials"):
        VllmGateway(
            _table((1, 2), 0, "0"),
            upstream_headers={"Authorization": "Bearer "},
        )


async def test_gateway_only_proxies_authenticated_generation() -> None:
    calls: list[tuple[str, str]] = []

    async def completion(request: web.Request) -> web.Response:
        calls.append((request.path, request.headers["Authorization"]))
        return web.json_response({"ok": True})

    async def admin(request: web.Request) -> web.Response:
        calls.append((request.path, request.headers["Authorization"]))
        return web.json_response({"loaded": True})

    async def metrics(_request: web.Request) -> web.Response:
        return web.json_response(
            {"metrics": {"num_requests_running": 0, "max_num_seqs": 2}}
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    app.router.add_post("/v1/load_lora_adapter", admin)
    app.router.add_get("/art/metrics", metrics)
    upstream = web.AppRunner(app)
    await upstream.setup()
    site = web.TCPSite(upstream, "127.0.0.1", 0)
    await site.start()
    upstream_port = site._server.sockets[0].getsockname()[1]
    gateway = VllmGateway(
        _table((upstream_port, upstream_port), 0, "0"),
        upstream_headers={"Authorization": "Bearer upstream-admin"},
        inbound_api_key="gateway-client",
    )
    gateway_port = await gateway.start("0.0.0.0")
    url = f"http://127.0.0.1:{gateway_port}"
    try:
        async with ClientSession() as client:
            payload = {"model": "model@0", "messages": []}
            assert (
                await client.post(f"{url}/v1/chat/completions", json=payload)
            ).status == 401
            headers = {"Authorization": "Bearer gateway-client"}
            assert (
                await client.post(
                    f"{url}/v1/load_lora_adapter", json={}, headers=headers
                )
            ).status == 404
            assert (
                await client.get(f"{url}/v1/chat/completions", headers=headers)
            ).status == 405
            assert (
                await client.post(
                    f"{url}/v1/chat/completions", json=payload, headers=headers
                )
            ).status == 200
        assert calls == [("/v1/chat/completions", "Bearer upstream-admin")]
    finally:
        await gateway.close()
        await upstream.cleanup()
