from __future__ import annotations

import asyncio
from collections.abc import Iterable
from hmac import compare_digest
from ipaddress import ip_address
import json
import socket
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web
from pydantic import BaseModel, ConfigDict, Field

from .vllm_replica import ReplicaUpdateReport
from .vllm_router import (
    KvCacheEvent,
    PrefixBlockHashes,
    ReplicaRouter,
    ReplicaTelemetry,
    RoutingDeadlineExceededError,
    RoutingInput,
    RoutingQueueFullError,
    RoutingTable,
    RoutingUnavailableError,
)

_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_REQUEST_HEADER_DENYLIST = _HOP_HEADERS | {"authorization"}


class _RoutingHints(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_key: str | None = Field(default=None, min_length=1)
    prefix: PrefixBlockHashes | None = None


_ROUTED_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/art/v1/chat/completions",
}


class VllmGateway:
    """Streaming reverse proxy over one atomically committed replica table."""

    def __init__(
        self,
        table: RoutingTable,
        *,
        upstream_headers: dict[str, str] | None = None,
        inbound_api_key: str | None = None,
        max_queued: int = 128,
        route_timeout_s: float = 1200.0,
    ) -> None:
        if inbound_api_key is not None and not inbound_api_key.strip():
            raise ValueError("inbound_api_key cannot be empty")
        self.router = ReplicaRouter(table, max_queued=max_queued)
        self._policies = {table.policy_version: self.router}
        self._pinned_policies: set[str] = set()
        self._max_queued = max_queued
        self._upstream_headers = dict(upstream_headers or {})
        self._required_authorization = (
            f"Bearer {inbound_api_key}"
            if inbound_api_key is not None
            else next(
                (
                    value
                    for key, value in self._upstream_headers.items()
                    if key.lower() == "authorization"
                ),
                None,
            )
        )
        self._route_timeout_s = route_timeout_s
        self._runner: web.AppRunner | None = None
        self._socket: socket.socket | None = None
        self._session: ClientSession | None = None
        self._telemetry_task: asyncio.Task[None] | None = None

    async def start(self, bind_host: str = "127.0.0.1", port: int = 0) -> int:
        if self._runner is not None:
            raise RuntimeError("vLLM gateway is already running")
        if not _is_loopback(bind_host) and self._required_authorization is None:
            raise ValueError(
                "a non-loopback vLLM gateway requires inbound authentication"
            )
        app = web.Application(
            client_max_size=64 << 20, middlewares=(self._authenticate,)
        )
        for path in _ROUTED_PATHS:
            app.router.add_post(path, self._handle)
        app.router.add_get("/art/metrics", self._handle, allow_head=False)
        app.router.add_get("/health", self._handle, allow_head=False)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, port))
        sock.listen(2048)
        sock.setblocking(False)
        self._socket = sock
        await web.SockSite(self._runner, sock).start()
        self._session = ClientSession(
            timeout=ClientTimeout(total=None, connect=5.0),
            connector=TCPConnector(limit=0, limit_per_host=0, ttl_dns_cache=300),
            auto_decompress=False,
        )
        self._telemetry_task = asyncio.create_task(self._refresh_telemetry())
        return int(sock.getsockname()[1])

    @web.middleware
    async def _authenticate(
        self, request: web.Request, handler: Any
    ) -> web.StreamResponse:
        expected = self._required_authorization
        if expected is not None and not compare_digest(
            request.headers.get("Authorization", ""), expected
        ):
            raise web.HTTPUnauthorized(headers={"WWW-Authenticate": "Bearer"})
        return await handler(request)

    async def commit(
        self,
        table: RoutingTable,
        reports: tuple[ReplicaUpdateReport, ...],
    ) -> None:
        previous = self.router.table.policy_version
        prepared = self.router.prepare(table)
        await self.router.verify(prepared, reports)
        await self.router.commit(prepared)
        self._policies[table.policy_version] = self.router
        if previous not in self._pinned_policies:
            self._policies.pop(previous, None)

    def add_policy(self, table: RoutingTable) -> None:
        version = table.policy_version
        self._policies[version] = ReplicaRouter(table, max_queued=self._max_queued)
        self._pinned_policies.add(version)

    def remove_policy(self, version: str) -> None:
        self._pinned_policies.discard(version)
        if version != self.router.table.policy_version:
            self._policies.pop(version, None)

    def apply_kv_events(self, events: Iterable[KvCacheEvent]) -> bool:
        """Apply one normalized vLLM publisher batch to every policy view."""
        batch = tuple(events)
        applied = False
        for router in set(self._policies.values()):
            applied = router.apply_kv_events(batch) or applied
        return applied

    def invalidate_kv(self, replica_id: str) -> None:
        for router in set(self._policies.values()):
            router.invalidate_kv(replica_id)

    async def pause(self, reason: str) -> None:
        await self.router.quarantine(
            (replica.replica_id for replica in self.router.table.replicas), reason
        )

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if self._session is None:
            raise web.HTTPServiceUnavailable(text="gateway is not ready")
        path = request.path
        if path == "/art/metrics":
            return web.json_response({"metrics": await self._aggregate_metrics()})
        if path == "/health":
            if any(replica.phase != "ready" for replica in self.router.table.replicas):
                raise web.HTTPServiceUnavailable(text="a replica is not routable")
            await self._require_all_healthy()
            return web.Response()
        body = await request.read()
        try:
            payload = json.loads(body)
            version = _policy_version(payload)
            router = self._policies.get(version)
            if router is None:
                raise RoutingUnavailableError(
                    f"policy version {version!r} is not loaded"
                )
            routing = _routing_input(payload, router.table)
            if "art_routing" in payload:
                payload.pop("art_routing")
                body = json.dumps(payload, separators=(",", ":")).encode()
            reservation = await router.acquire(routing, timeout_s=self._route_timeout_s)
            async with reservation:
                return await self._proxy(
                    request, body, reservation.replica.endpoint.url
                )
        except RoutingQueueFullError as error:
            raise web.HTTPTooManyRequests(
                text=str(error), headers={"Retry-After": "1"}
            ) from error
        except RoutingDeadlineExceededError as error:
            raise web.HTTPGatewayTimeout(text=str(error)) from error
        except RoutingUnavailableError as error:
            raise web.HTTPServiceUnavailable(text=str(error)) from error
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text=f"invalid routed request: {error}") from error

    async def _proxy(
        self, request: web.Request, body: bytes, endpoint: str
    ) -> web.StreamResponse:
        assert self._session is not None
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _REQUEST_HEADER_DENYLIST
        }
        headers.update(self._upstream_headers)
        url = f"{endpoint.rstrip('/')}{request.path}"
        async with self._session.request(
            request.method,
            url,
            data=body if body else None,
            headers=headers,
            allow_redirects=False,
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers={
                    key: value
                    for key, value in upstream.headers.items()
                    if key.lower() not in _HOP_HEADERS
                },
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response

    async def _aggregate_metrics(self) -> dict[str, float]:
        snapshots = await asyncio.gather(
            *(
                self._metrics(replica.endpoint.url)
                for replica in self.router.table.replicas
            )
        )
        names = set().union(*(snapshot.keys() for snapshot in snapshots))
        return {
            name: (
                sum(snapshot.get(name, 0.0) for snapshot in snapshots) / len(snapshots)
                if name == "kv_cache_usage_perc"
                else max(snapshot.get(name, 0.0) for snapshot in snapshots)
                if name == "max_model_len"
                else sum(snapshot.get(name, 0.0) for snapshot in snapshots)
            )
            for name in names
        }

    async def _metrics(self, endpoint: str) -> dict[str, float]:
        assert self._session is not None
        async with self._session.get(
            f"{endpoint}/art/metrics", headers=self._upstream_headers
        ) as response:
            response.raise_for_status()
            values = (await response.json())["metrics"]
        return {key: float(value) for key, value in values.items()}

    async def _require_all_healthy(self) -> None:
        session = self._session
        assert session is not None

        async def health(endpoint: str) -> None:
            async with session.get(
                f"{endpoint}/health", headers=self._upstream_headers
            ) as response:
                response.raise_for_status()

        try:
            await asyncio.gather(
                *(
                    health(replica.endpoint.url)
                    for replica in self.router.table.replicas
                )
            )
        except ClientError as error:
            raise web.HTTPServiceUnavailable(text=str(error)) from error

    async def _refresh_telemetry(self) -> None:
        assert self._session is not None
        while True:
            observed_at = asyncio.get_running_loop().time()
            table = self.router.table

            async def refresh(replica: Any) -> None:
                try:
                    values = await asyncio.wait_for(
                        self._metrics(replica.endpoint.url), timeout=5.0
                    )
                    telemetry = ReplicaTelemetry(
                        observed_at=observed_at,
                        in_flight=int(values["num_requests_running"])
                        + int(values.get("num_requests_waiting", 0)),
                        capacity=int(values.get("max_num_seqs") or 256),
                    )
                    await asyncio.gather(
                        *(
                            router.update_telemetry(
                                replica.replica_id, replica.generation, telemetry
                            )
                            for router in set(self._policies.values())
                        )
                    )
                except (ClientError, KeyError, TypeError, ValueError, OSError):
                    pass

            await asyncio.gather(*(refresh(replica) for replica in table.replicas))
            await asyncio.sleep(1.0)

    async def close(self) -> None:
        task, self._telemetry_task = self._telemetry_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None


def _routing_input(payload: dict[str, Any], table: RoutingTable) -> RoutingInput:
    policy_version = _policy_version(payload)
    hints = _RoutingHints.model_validate(payload.get("art_routing", {}))
    return RoutingInput(
        policy_version=policy_version,
        policy_digest=table.policy_digest,
        stable_key=hints.stable_key,
        prefix=hints.prefix,
    )


def _policy_version(payload: dict[str, Any]) -> str:
    model = str(payload["model"])
    return model.rsplit("@", 1)[-1] if "@" in model else model


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False
