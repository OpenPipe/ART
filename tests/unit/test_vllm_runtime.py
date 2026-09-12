from __future__ import annotations

import httpx
import pytest

from art.vllm_runtime import ManagedVllmRuntime


@pytest.mark.asyncio
async def test_post_with_retry_recovers_from_connect_errors() -> None:
    attempts = 0
    process_checks = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("runtime restarting", request=request)
        return httpx.Response(200, json={"loaded": True})

    def check_process() -> None:
        nonlocal process_checks
        process_checks += 1

    runtime = ManagedVllmRuntime()
    runtime.port = 8000
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await runtime.post_with_retry(
            "/v1/load_lora_adapter",
            json={"lora_name": "model@1"},
            timeout=1.0,
            max_attempts=3,
            retry_base_delay=0,
            before_attempt=check_process,
            http_client=client,
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert attempts == 3
    assert process_checks == 3


@pytest.mark.asyncio
async def test_post_with_retry_stops_when_runtime_process_fails() -> None:
    attempts = 0
    process_checks = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("runtime restarting", request=request)

    def check_process() -> None:
        nonlocal process_checks
        process_checks += 1
        if process_checks == 2:
            raise RuntimeError("vLLM runtime exited with code 9")

    runtime = ManagedVllmRuntime()
    runtime.port = 8000
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError, match="vLLM runtime exited with code 9"):
            await runtime.post_with_retry(
                "/v1/load_lora_adapter",
                timeout=1.0,
                retry_base_delay=0,
                before_attempt=check_process,
                http_client=client,
            )
    finally:
        await client.aclose()

    assert attempts == 1
    assert process_checks == 2


@pytest.mark.asyncio
async def test_post_with_retry_recovers_from_transient_status() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="runtime unavailable")
        return httpx.Response(200, json={"loaded": True})

    runtime = ManagedVllmRuntime()
    runtime.port = 8000
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await runtime.post_with_retry(
            "/v1/load_lora_adapter",
            timeout=1.0,
            max_attempts=2,
            retry_base_delay=0,
            http_client=client,
        )
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert attempts == 2


@pytest.mark.asyncio
async def test_post_with_retry_does_not_retry_client_errors() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="adapter already loaded")

    runtime = ManagedVllmRuntime()
    runtime.port = 8000
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await runtime.post_with_retry(
                "/v1/load_lora_adapter",
                timeout=1.0,
                retry_base_delay=0,
                http_client=client,
            )
    finally:
        await client.aclose()

    assert attempts == 1


@pytest.mark.asyncio
async def test_post_with_retry_reports_runtime_diagnostics() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("runtime unavailable", request=request)

    runtime = ManagedVllmRuntime()
    runtime.port = 8123
    runtime.log_path = "/tmp/art/vllm-runtime.log"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError) as exc_info:
            await runtime.post_with_retry(
                "/v1/load_lora_adapter",
                timeout=1.0,
                max_attempts=2,
                retry_base_delay=0,
                operation="LoRA adapter reload",
                failure_context="step=7; checkpoint=/tmp/checkpoints/0007",
                http_client=client,
            )
    finally:
        await client.aclose()

    message = str(exc_info.value)
    assert "LoRA adapter reload failed after 2 attempts" in message
    assert "url=http://127.0.0.1:8123/v1/load_lora_adapter" in message
    assert "process=not managed by this process" in message
    assert "log=/tmp/art/vllm-runtime.log" in message
    assert "last_error=runtime unavailable" in message
    assert "step=7; checkpoint=/tmp/checkpoints/0007" in message
