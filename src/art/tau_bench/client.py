from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import math
import os
from typing import Any, AsyncGenerator, Literal, cast
import uuid

import httpx
from pydantic import BaseModel

JsonObject = dict[str, Any]
TRANSIENT_STATUS_CODES = {429, 502, 503, 504}
DEFAULT_STATUS_RETRIES = 12
DEFAULT_RETRY_BASE_DELAY = 0.5
DEFAULT_RETRY_MAX_DELAY = 5.0
DEFAULT_REQUEST_ATTEMPT_TIMEOUT = 300.0
DEFAULT_CLEANUP_TIMEOUT = 30.0
DEFAULT_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, pool=30.0, write=30.0, read=300.0)
_HTTP_POOL_SHARDS = 64
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_PRE_SEND_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)

logger = logging.getLogger(__name__)


def _default_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=100_000,
        max_keepalive_connections=100_000,
        keepalive_expiry=60.0,
    )


def _shard_limit(total: int | None, shards: int, index: int) -> int | None:
    if total is None:
        return None
    quotient, remainder = divmod(total, shards)
    return quotient + (index < remainder)


class _CapacityReleaseStream(httpx.AsyncByteStream):
    def __init__(
        self, stream: httpx.AsyncByteStream, capacity: asyncio.Semaphore
    ) -> None:
        self._stream = stream
        self._capacity = capacity
        self._closed = False

    async def __aiter__(self) -> AsyncGenerator[bytes, None]:
        try:
            async for chunk in self._stream:
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._stream.aclose()
        finally:
            self._capacity.release()


class _ShardedAsyncHTTPTransport(httpx.AsyncBaseTransport):
    """Round-robin requests across smaller HTTP connection pools."""

    def __init__(
        self,
        *,
        limits: httpx.Limits,
        retries: int,
        shards: int = _HTTP_POOL_SHARDS,
    ) -> None:
        max_connections = limits.max_connections
        shard_count = min(shards, max_connections or shards)
        if shard_count < 1:
            raise ValueError("HTTP transport shards must be positive")
        self._capacity = (
            asyncio.Semaphore(max_connections) if max_connections is not None else None
        )
        self.transports = tuple(
            httpx.AsyncHTTPTransport(
                retries=retries,
                limits=httpx.Limits(
                    # Aggregate active capacity is enforced above the shards. Giving
                    # every shard the aggregate ceiling prevents uneven request
                    # durations from stranding capacity in another shard.
                    max_connections=max_connections,
                    max_keepalive_connections=_shard_limit(
                        limits.max_keepalive_connections, shard_count, index
                    ),
                    keepalive_expiry=limits.keepalive_expiry,
                ),
            )
            for index in range(shard_count)
        )
        self._next = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        capacity = self._capacity
        if capacity is not None:
            pool_timeout = request.extensions.get("timeout", {}).get("pool")
            try:
                async with asyncio.timeout(pool_timeout):
                    await capacity.acquire()
            except TimeoutError:
                raise httpx.PoolTimeout(
                    "Timed out waiting for an available connection slot",
                    request=request,
                ) from None
        transport = self.transports[self._next]
        self._next = (self._next + 1) % len(self.transports)
        try:
            response = await transport.handle_async_request(request)
        except BaseException:
            if capacity is not None:
                capacity.release()
            raise
        if capacity is not None:
            response.stream = _CapacityReleaseStream(
                cast(httpx.AsyncByteStream, response.stream), capacity
            )
        return response

    async def aclose(self) -> None:
        await asyncio.gather(*(transport.aclose() for transport in self.transports))


def _sharded_transport(
    *, limits: httpx.Limits, retries: int
) -> _ShardedAsyncHTTPTransport:
    return _ShardedAsyncHTTPTransport(limits=limits, retries=retries)


def _normalize_timeout(timeout: float | httpx.Timeout | None) -> httpx.Timeout | None:
    if isinstance(timeout, int | float):
        return httpx.Timeout(timeout, connect=min(float(timeout), 30.0))
    return timeout


def _default_status_retries() -> int:
    return max(
        0, int(os.getenv("TAU_BENCH_HTTP_STATUS_RETRIES", DEFAULT_STATUS_RETRIES))
    )


def _default_retry_base_delay() -> float:
    return max(
        0.0,
        float(os.getenv("TAU_BENCH_HTTP_RETRY_BASE_DELAY", DEFAULT_RETRY_BASE_DELAY)),
    )


def _default_retry_max_delay() -> float:
    return max(
        0.0,
        float(os.getenv("TAU_BENCH_HTTP_RETRY_MAX_DELAY", DEFAULT_RETRY_MAX_DELAY)),
    )


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail: Any = response.text
        try:
            parsed = response.json()
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = parsed["detail"]
        except ValueError:
            pass
        raise httpx.HTTPStatusError(
            f"{exc} Response detail: {detail}",
            request=exc.request,
            response=exc.response,
        ) from exc


class Task(BaseModel):
    id: str
    description: JsonObject | str | None = None
    user_scenario: JsonObject | str | None = None
    ticket: str | None = None
    initial_state: JsonObject | None = None
    evaluation_criteria: JsonObject | None = None
    issues: list[JsonObject | str] | None = None
    required_documents: list[str] | None = None
    user_tools: list[str] | None = None


class Scenario(BaseModel):
    domain: str
    task: Task


class ScenarioListResponse(BaseModel):
    scenarios: list[Scenario]


class EnvironmentResponse(BaseModel):
    id: str
    observation: str
    info: dict[str, Any]


class StepEnvironmentResponse(EnvironmentResponse):
    reward: float
    terminated: bool
    truncated: bool


class DeleteEnvironmentResponse(BaseModel):
    id: str
    deleted: bool


class TauBenchClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = DEFAULT_HTTP_TIMEOUT,
        limits: httpx.Limits | None = None,
        http_client: httpx.AsyncClient | None = None,
        request_attempt_timeout: float | None = DEFAULT_REQUEST_ATTEMPT_TIMEOUT,
        status_retries: int | None = None,
        retry_base_delay: float | None = None,
        retry_max_delay: float | None = None,
    ) -> None:
        self.api_key = (
            api_key if api_key is not None else os.getenv("TAU_BENCH_API_KEY")
        )
        self.status_retries = (
            status_retries if status_retries is not None else _default_status_retries()
        )
        self.retry_base_delay = (
            retry_base_delay
            if retry_base_delay is not None
            else _default_retry_base_delay()
        )
        self.retry_max_delay = (
            retry_max_delay
            if retry_max_delay is not None
            else _default_retry_max_delay()
        )
        self._owns_client = http_client is None
        if request_attempt_timeout is not None and (
            not math.isfinite(request_attempt_timeout) or request_attempt_timeout <= 0
        ):
            raise ValueError("request_attempt_timeout must be finite and positive")
        self._request_attempt_timeout = (
            request_attempt_timeout if self._owns_client else None
        )
        self._client = http_client or httpx.AsyncClient(
            base_url=(
                base_url or os.getenv("TAU_BENCH_BASE_URL") or "http://localhost:8000"
            ),
            timeout=_normalize_timeout(timeout),
            transport=_sharded_transport(limits=limits or _default_limits(), retries=2),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "TauBenchClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def get_scenarios(
        self,
        *,
        domain: str | None = None,
        split: str | None = None,
    ) -> list[Scenario]:
        response = await self._request(
            "GET",
            "/scenarios",
            params={
                key: value
                for key, value in {"domain": domain, "split": split}.items()
                if value is not None
            },
            headers=self._auth_headers(),
        )
        _raise_for_status(response)
        return ScenarioListResponse.model_validate(response.json()).scenarios

    async def create_environment(
        self,
        *,
        domain: str,
        task_id: str,
        user_llm: str | None = None,
        user_llm_args: dict[str, Any] | None = None,
        retrieval_config: str | None = None,
        retrieval_config_kwargs: dict[str, Any] | None = None,
        idle_timeout_seconds: float | None = None,
    ) -> EnvironmentResponse:
        response = await self._request(
            "POST",
            "/environments",
            json={
                key: value
                for key, value in {
                    "domain": domain,
                    "task_id": task_id,
                    "user_llm": user_llm,
                    "user_llm_args": user_llm_args,
                    "retrieval_config": retrieval_config,
                    "retrieval_config_kwargs": retrieval_config_kwargs,
                    "idle_timeout_seconds": idle_timeout_seconds,
                }.items()
                if value is not None
            },
            headers=self._auth_headers(),
        )
        _raise_for_status(response)
        return EnvironmentResponse.model_validate(response.json())

    async def step_environment(
        self,
        env_id: str,
        action: str,
    ) -> StepEnvironmentResponse:
        response = await self._request(
            "POST",
            f"/environments/{env_id}/step",
            json={"action": action},
            headers=self._auth_headers(),
        )
        _raise_for_status(response)
        return StepEnvironmentResponse.model_validate(response.json())

    async def delete_environment(self, env_id: str) -> DeleteEnvironmentResponse:
        response = await self._request(
            "DELETE",
            f"/environments/{env_id}",
            headers=self._auth_headers(),
        )
        _raise_for_status(response)
        return DeleteEnvironmentResponse.model_validate(response.json())

    @asynccontextmanager
    async def environment(
        self,
        *,
        domain: str,
        task_id: str,
        user_llm: str | None = None,
        user_llm_args: dict[str, Any] | None = None,
        retrieval_config: str | None = None,
        retrieval_config_kwargs: dict[str, Any] | None = None,
        idle_timeout_seconds: float | None = None,
    ) -> AsyncGenerator[EnvironmentResponse, None]:
        env = await self.create_environment(
            domain=domain,
            task_id=task_id,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
            retrieval_config=retrieval_config,
            retrieval_config_kwargs=retrieval_config_kwargs,
            idle_timeout_seconds=idle_timeout_seconds,
        )
        try:
            yield env
        except BaseException as error:
            await self._delete_after_error(env.id, error)
            raise
        else:
            await self.delete_environment(env.id)

    async def _delete_after_error(
        self, env_id: str, primary_error: BaseException
    ) -> None:
        cleanup = asyncio.create_task(self.delete_environment(env_id))
        try:
            async with asyncio.timeout(DEFAULT_CLEANUP_TIMEOUT):
                await asyncio.shield(cleanup)
        except BaseException as cleanup_error:
            if not cleanup.done():
                cleanup.cancel()
                cleanup.add_done_callback(_discard_task_result)
            primary_error.add_note(
                f"Failed to delete tau-bench environment {env_id}: {cleanup_error!r}"
            )
            logger.warning(
                "Failed to delete tau-bench environment %s while handling another error",
                env_id,
                exc_info=True,
            )

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key is None:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        method = method.upper()
        headers = dict(kwargs.pop("headers", {}))
        headers.setdefault("X-Request-ID", str(uuid.uuid4()))
        attempts = self.status_retries + 1
        last_transport_error: httpx.TransportError | None = None
        for attempt in range(attempts):
            try:
                try:
                    async with asyncio.timeout(self._request_attempt_timeout):
                        response = await self._client.request(
                            method,
                            url,
                            headers=headers,
                            **kwargs,
                        )
                except TimeoutError as exc:
                    raise httpx.TimeoutException(
                        f"tau-bench {method} {url} exceeded the "
                        f"{self._request_attempt_timeout:g}s attempt deadline"
                    ) from exc
            except httpx.TransportError as exc:
                last_transport_error = exc
                if attempt == attempts - 1 or not _transport_retry_is_safe(
                    method,
                    exc,
                    trusted_transport=self._owns_client,
                ):
                    raise
            else:
                if (
                    response.status_code not in TRANSIENT_STATUS_CODES
                    or method not in _SAFE_METHODS
                    or attempt == attempts - 1
                ):
                    return response
                await response.aclose()
            await asyncio.sleep(
                min(self.retry_base_delay * (2**attempt), self.retry_max_delay)
            )
        assert last_transport_error is not None
        raise last_transport_error


def _transport_retry_is_safe(
    method: str,
    error: httpx.TransportError,
    *,
    trusted_transport: bool,
) -> bool:
    return method in _SAFE_METHODS or (
        trusted_transport and isinstance(error, _PRE_SEND_TRANSPORT_ERRORS)
    )


def _discard_task_result(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:
        pass


default_client: TauBenchClient | None = None


def _get_default_client(client: TauBenchClient | None = None) -> TauBenchClient:
    if client is not None:
        return client
    global default_client
    if default_client is None:
        default_client = TauBenchClient()
    return default_client


async def get_scenarios(
    *,
    domain: (
        Literal[
            "banking_knowledge",
            "retail",
            "airline",
            "telecom",
            "telecom-workflow",
            "mock",
        ]
        | str
        | None
    ) = None,
    split: (
        Literal[
            "base",
            "train",
            "test",
            "small",
            "full",
        ]
        | str
        | None
    ) = None,
    client: TauBenchClient | None = None,
) -> list[Scenario]:
    return await _get_default_client(client).get_scenarios(domain=domain, split=split)
