from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from hmac import compare_digest
from ipaddress import ip_address
import json
import socket
from typing import Any

from aiohttp import (
    ClientError,
    ClientResponse,
    ClientSession,
    ClientTimeout,
    TCPConnector,
    payload,
    web,
)
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .vllm_kv_events import KvEventSource, VllmKvEventSubscriber
from .vllm_replica import ReplicaUpdateReport
from .vllm_router import (
    KvCacheEvent,
    PrefixBlockHashes,
    ReplicaRouter,
    ReplicaTelemetry,
    RoutableReplica,
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
_TELEMETRY_MAX_AGE_S = 5.0


class _TimedBytesPayload(payload.BytesPayload):
    def __init__(self, value: bytes, timeout_s: float) -> None:
        super().__init__(value)
        self._timeout_s = timeout_s

    async def write(self, writer: Any) -> None:
        async with asyncio.timeout(self._timeout_s):
            await super().write(writer)

    async def write_with_length(self, writer: Any, content_length: int | None) -> None:
        async with asyncio.timeout(self._timeout_s):
            await super().write_with_length(writer, content_length)


class _RoutingHints(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_key: str | None = Field(default=None, min_length=1)
    prefix: PrefixBlockHashes | None = None
    prompt_token_ids: tuple[StrictInt, ...] | None = Field(
        default=None, max_length=1_000_000
    )


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
        upstream_connect_timeout_s: float = 5.0,
        upstream_read_timeout_s: float = 1200.0,
        upstream_write_timeout_s: float = 30.0,
        upstream_pool_timeout_s: float = 5.0,
        upstream_pool_size: int = 1024,
        shutdown_timeout_s: float = 30.0,
        kv_event_sources: Iterable[KvEventSource] = (),
    ) -> None:
        if inbound_api_key is not None and not inbound_api_key.strip():
            raise ValueError("inbound_api_key cannot be empty")
        self.router = ReplicaRouter(
            table,
            telemetry_max_age_s=_TELEMETRY_MAX_AGE_S,
            max_queued=max_queued,
        )
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
        if self._required_authorization is not None and not _has_credentials(
            self._required_authorization
        ):
            raise ValueError("configured Authorization header has no credentials")
        if any(
            timeout <= 0
            for timeout in (
                upstream_connect_timeout_s,
                upstream_read_timeout_s,
                upstream_write_timeout_s,
                upstream_pool_timeout_s,
                shutdown_timeout_s,
            )
        ):
            raise ValueError("gateway timeouts must be positive")
        if upstream_pool_size <= 0:
            raise ValueError("upstream_pool_size must be positive")
        self._route_timeout_s = route_timeout_s
        self._upstream_timeout = ClientTimeout(
            total=None,
            connect=upstream_connect_timeout_s,
            sock_connect=upstream_connect_timeout_s,
            sock_read=upstream_read_timeout_s,
        )
        self._upstream_write_timeout_s = upstream_write_timeout_s
        self._upstream_pool_timeout_s = upstream_pool_timeout_s
        self._upstream_pool_size = upstream_pool_size
        self._upstream_slots = asyncio.BoundedSemaphore(upstream_pool_size)
        self._shutdown_timeout_s = shutdown_timeout_s
        self._kv_subscribers = tuple(
            VllmKvEventSubscriber(source, self.apply_kv_events, self.invalidate_kv)
            for source in kv_event_sources
        )
        self._runner: web.AppRunner | None = None
        self._site: web.SockSite | None = None
        self._socket: socket.socket | None = None
        self._session: ClientSession | None = None
        self._telemetry_task: asyncio.Task[None] | None = None
        self._admitting = False
        self._close_lock = asyncio.Lock()

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
        self._runner = web.AppRunner(
            app, access_log=None, shutdown_timeout=self._shutdown_timeout_s
        )
        await self._runner.setup()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, port))
        sock.listen(2048)
        sock.setblocking(False)
        self._socket = sock
        self._site = web.SockSite(self._runner, sock)
        await self._site.start()
        self._session = ClientSession(
            timeout=self._upstream_timeout,
            connector=TCPConnector(limit=self._upstream_pool_size, ttl_dns_cache=300),
            auto_decompress=False,
        )
        self._telemetry_task = asyncio.create_task(self._refresh_telemetry())
        try:
            for subscriber in self._kv_subscribers:
                subscriber.start()
        except BaseException:
            await self.close()
            raise
        self._admitting = True
        return int(sock.getsockname()[1])

    @web.middleware
    async def _authenticate(
        self, request: web.Request, handler: Any
    ) -> web.StreamResponse:
        expected = self._required_authorization
        if expected is not None and not compare_digest(
            request.headers.get("Authorization", "").encode(), expected.encode()
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
        router = ReplicaRouter(
            table,
            telemetry_max_age_s=_TELEMETRY_MAX_AGE_S,
            max_queued=self._max_queued,
        )
        router.inherit_kv(self.router)
        self._policies[version] = router
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
        if not self._admitting or self._session is None:
            raise web.HTTPServiceUnavailable(text="gateway is not accepting requests")
        path = request.path
        if path == "/art/metrics":
            return web.json_response({"metrics": await self._aggregate_metrics()})
        if path == "/health":
            await self._require_survivors_healthy()
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
            routing = _routing_input(payload, router.table, path)
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
        response: web.StreamResponse | None = None
        try:
            async with self._upstream_request(
                request.method,
                url,
                data=(
                    _TimedBytesPayload(body, self._upstream_write_timeout_s)
                    if body
                    else None
                ),
                headers=headers,
                allow_redirects=False,
            ) as upstream:
                chunks = upstream.content.iter_any()
                first = await anext(chunks, None)
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
                if first is not None:
                    await response.write(first)
                async for chunk in chunks:
                    await response.write(chunk)
                await response.write_eof()
                return response
        except TimeoutError as error:
            if response is not None and response.prepared:
                raise
            raise web.HTTPGatewayTimeout(text="upstream request timed out") from error
        except (ClientError, OSError) as error:
            if response is not None and response.prepared:
                raise
            raise web.HTTPBadGateway(text="upstream request failed") from error

    @asynccontextmanager
    async def _upstream_request(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ClientResponse]:
        async with asyncio.timeout(self._upstream_pool_timeout_s):
            await self._upstream_slots.acquire()
        try:
            assert self._session is not None
            async with self._session.request(*args, **kwargs) as response:
                yield response
        finally:
            self._upstream_slots.release()

    def _surviving_replicas(self) -> tuple[RoutableReplica, ...]:
        table = self.router.table
        now = asyncio.get_running_loop().time()
        return tuple(
            replica
            for replica in table.replicas
            if replica.phase == "ready"
            and replica.committed_version == table.policy_version
            and replica.policy_digest == table.policy_digest
            and now - replica.telemetry.observed_at <= _TELEMETRY_MAX_AGE_S
        )

    async def _aggregate_metrics(self) -> dict[str, float]:
        replicas = self._surviving_replicas()
        if not replicas:
            raise web.HTTPServiceUnavailable(text="no routable replica survives")
        try:
            snapshots = await asyncio.gather(
                *(self._metrics(replica.endpoint.url) for replica in replicas)
            )
        except (
            ClientError,
            KeyError,
            OSError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as error:
            raise web.HTTPServiceUnavailable(
                text="surviving replica metrics failed"
            ) from error
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
        async with self._upstream_request(
            "GET", f"{endpoint}/art/metrics", headers=self._upstream_headers
        ) as response:
            response.raise_for_status()
            values = (await response.json())["metrics"]
        return {key: float(value) for key, value in values.items()}

    async def _require_survivors_healthy(self) -> None:
        async def health(endpoint: str) -> None:
            async with self._upstream_request(
                "GET", f"{endpoint}/health", headers=self._upstream_headers
            ) as response:
                response.raise_for_status()

        replicas = self._surviving_replicas()
        if not replicas:
            raise web.HTTPServiceUnavailable(text="no routable replica survives")
        try:
            await asyncio.gather(
                *(health(replica.endpoint.url) for replica in replicas)
            )
        except (ClientError, OSError, TimeoutError) as error:
            raise web.HTTPServiceUnavailable(
                text="surviving replica is unhealthy"
            ) from error

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

    async def _stop_admission(self) -> None:
        self._admitting = False
        site, self._site = self._site, None
        if site is not None:
            await site.stop()

    async def close(self) -> None:
        async with self._close_lock:
            await self._stop_admission()
            task, self._telemetry_task = self._telemetry_task, None
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            runner = self._runner
            try:
                if runner is not None:
                    await runner.cleanup()
            finally:
                self._runner = None
                try:
                    await asyncio.gather(
                        *(subscriber.close() for subscriber in self._kv_subscribers)
                    )
                finally:
                    if self._session is not None:
                        await self._session.close()
                        self._session = None
                    if self._socket is not None:
                        self._socket.close()
                        self._socket = None


def _routing_input(
    payload: dict[str, Any], table: RoutingTable, path: str
) -> RoutingInput:
    policy_version = _policy_version(payload)
    hints = _RoutingHints.model_validate(payload.get("art_routing", {}))
    prompt_token_ids = hints.prompt_token_ids
    if prompt_token_ids is None and hints.prefix is None and path == "/v1/completions":
        prompt = payload.get("prompt")
        if isinstance(prompt, list) and all(
            isinstance(token_id, int) and not isinstance(token_id, bool)
            for token_id in prompt
        ):
            prompt_token_ids = tuple(prompt)
    if prompt_token_ids is not None and _has_multimodal_input(payload):
        raise ValueError(
            "prompt token routing cannot represent multimodal cache hash keys"
        )
    cache_salt = payload.get("cache_salt")
    if cache_salt is not None and not isinstance(cache_salt, str):
        raise ValueError("cache_salt must be a string")
    return RoutingInput(
        policy_version=policy_version,
        policy_digest=table.policy_digest,
        stable_key=hints.stable_key,
        prefix=hints.prefix,
        prompt_token_ids=prompt_token_ids,
        cache_salt=cache_salt,
    )


def _has_multimodal_input(payload: dict[str, Any]) -> bool:
    if payload.get("prompt_embeds") is not None:
        return True
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(
            not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}
            for part in content
        ):
            return True
    return False


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


def _has_credentials(authorization: str) -> bool:
    scheme, separator, credentials = authorization.partition(" ")
    return bool(scheme and separator and credentials.strip())
