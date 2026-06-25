from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from typing import Any, AsyncGenerator, Literal
import uuid

import httpx
from pydantic import BaseModel

JsonObject = dict[str, Any]
TRANSIENT_STATUS_CODES = {429, 502, 503, 504}
DEFAULT_STATUS_RETRIES = 12
DEFAULT_RETRY_BASE_DELAY = 0.5
DEFAULT_RETRY_MAX_DELAY = 5.0


def _default_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=512,
        max_keepalive_connections=512,
        keepalive_expiry=60.0,
    )


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
        timeout: float | httpx.Timeout | None = 300.0,
        limits: httpx.Limits | None = None,
        http_client: httpx.AsyncClient | None = None,
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
        self._client = http_client or httpx.AsyncClient(
            base_url=(
                base_url or os.getenv("TAU_BENCH_BASE_URL") or "http://localhost:8000"
            ),
            timeout=_normalize_timeout(timeout),
            transport=httpx.AsyncHTTPTransport(
                limits=limits or _default_limits(),
                retries=2,
            ),
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
        finally:
            await self.delete_environment(env.id)

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key is None:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers.setdefault("X-Request-ID", str(uuid.uuid4()))
        attempts = self.status_retries + 1
        last_transport_error: httpx.TransportError | None = None
        for attempt in range(attempts):
            try:
                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs,
                )
            except httpx.TransportError as exc:
                last_transport_error = exc
                if attempt == attempts - 1:
                    raise
            else:
                if (
                    response.status_code not in TRANSIENT_STATUS_CODES
                    or attempt == attempts - 1
                ):
                    return response
                await response.aclose()
            await asyncio.sleep(
                min(self.retry_base_delay * (2**attempt), self.retry_max_delay)
            )
        assert last_transport_error is not None
        raise last_transport_error


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
