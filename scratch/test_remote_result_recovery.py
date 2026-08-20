import asyncio
from datetime import datetime, timezone

import httpx
from pydantic import ValidationError
import pytest

from art.serverless.client import (
    RemoteTrainingError,
    RemoteTrainingHttpError,
    RemoteTrainingOperation,
    RemoteTrainingServiceClient,
    _RunEventObserver,
)
from art.serverless.contracts import OperationResultRef, RunEvent
from art.serverless.data_plane import encode_operation_result
from art.training.contracts import OperationRef, OperationResult

_OPERATION_ID = "operation"
_NOW = datetime.now(timezone.utc)


def _service(
    handler, *, max_retries: int = 0
) -> tuple[RemoteTrainingServiceClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(
        base_url="http://test/v1/", transport=httpx.MockTransport(handler)
    )
    return (
        RemoteTrainingServiceClient(
            api_key="test",
            base_url="http://test/v1",
            control_http_client=http,
            transfer_http_client=http,
            max_retries=max_retries,
        ),
        http,
    )


def _operation(
    service: RemoteTrainingServiceClient, ref: OperationResultRef
) -> RemoteTrainingOperation[OperationResult]:
    terminal = asyncio.get_running_loop().create_future()
    terminal.set_result(
        RunEvent(
            cursor=1,
            run_id="run",
            operation_id=_OPERATION_ID,
            event="operation_succeeded",
            payload=ref.model_dump(mode="json"),
            created_at=_NOW,
        )
    )
    return RemoteTrainingOperation(
        OperationRef(
            run_id="run",
            operation_id=_OPERATION_ID,
            sequence_id=0,
            learner_parent_version=0,
            kind="forward",
        ),
        service,
        terminal,
        OperationResult,
    )


@pytest.mark.asyncio
async def test_event_polling_recovers_past_request_retry_budget_at_same_cursor():
    requests: list[int] = []
    remaining_failures = 4

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal remaining_failures
        after = int(request.url.params["after"])
        requests.append(after)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "events": [
                        {
                            "cursor": 5,
                            "run_id": "run",
                            "operation_id": None,
                            "event": "run_opened",
                            "payload": {},
                            "created_at": _NOW.isoformat(),
                        }
                    ],
                    "next_cursor": 5,
                },
            )
        if remaining_failures:
            remaining_failures -= 1
            if remaining_failures % 2:
                return httpx.Response(503, json={"detail": "temporarily unavailable"})
            raise httpx.ReadError("temporary disconnect", request=request)
        return httpx.Response(
            200,
            json={
                "events": [
                    {
                        "cursor": 6,
                        "run_id": "run",
                        "operation_id": _OPERATION_ID,
                        "event": "operation_succeeded",
                        "payload": {},
                        "created_at": _NOW.isoformat(),
                    }
                ],
                "next_cursor": 6,
            },
        )

    service, http = _service(handle, max_retries=1)
    observer = _RunEventObserver(service, "run", poll_interval_s=0.001)
    future = observer.reserve(_OPERATION_ID)
    observer.claim(_OPERATION_ID, future)
    try:
        event = await asyncio.wait_for(future, 1.0)
        assert event.operation_id == _OPERATION_ID
        assert requests == [0, 5, 5, 5, 5, 5]
    finally:
        await observer.close(timeout_s=0.1)
        await service.close()
        await http.aclose()


@pytest.mark.asyncio
async def test_result_restarts_after_mid_body_disconnect_and_verifies_exact_bytes():
    result = OperationResult(operation_id=_OPERATION_ID)
    ref, payload = encode_operation_result(result)
    paths: list[str] = []

    class DisconnectedBody(httpx.AsyncByteStream):
        def __init__(self, request: httpx.Request) -> None:
            self.request = request

        async def __aiter__(self):
            yield b"stale"
            raise httpx.ReadError("mid-body disconnect", request=self.request)

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            paths.append(request.url.path)
        if len(paths) == 1:
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(payload))},
                stream=DisconnectedBody(request),
            )
        return httpx.Response(200, content=payload)

    service, http = _service(handle)
    try:
        assert await asyncio.wait_for(_operation(service, ref).result(), 1.0) == result
        assert paths == [
            "/v1/training/runs/run/operations/operation/result",
            "/v1/training/runs/run/operations/operation/result",
        ]
    finally:
        await service.close()
        await http.aclose()


@pytest.mark.asyncio
async def test_result_rejects_changed_content_identity():
    ref, payload = encode_operation_result(OperationResult(operation_id=_OPERATION_ID))
    changed = payload[:-1] + bytes([payload[-1] ^ 1])

    async def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=changed)

    service, http = _service(handle)
    try:
        with pytest.raises(ValueError, match="hash differs"):
            await _operation(service, ref).result()
    finally:
        await service.close()
        await http.aclose()


@pytest.mark.asyncio
async def test_result_rejects_changed_size_without_retrying():
    ref, payload = encode_operation_result(OperationResult(operation_id=_OPERATION_ID))
    calls = 0

    async def handle(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"Content-Length": str(len(payload) + 1)},
            content=payload,
        )

    service, http = _service(handle)
    try:
        with pytest.raises(RemoteTrainingError, match="Content-Length changed"):
            await _operation(service, ref).result()
        assert calls == 1
    finally:
        await service.close()
        await http.aclose()


@pytest.mark.asyncio
async def test_result_rejects_changed_operation_identity():
    ref, payload = encode_operation_result(OperationResult(operation_id="different"))

    async def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    service, http = _service(handle)
    try:
        with pytest.raises(RemoteTrainingError, match="result identity changed"):
            await _operation(service, ref).result()
    finally:
        await service.close()
        await http.aclose()


@pytest.mark.asyncio
async def test_result_recovery_remains_cancellable():
    ref, payload = encode_operation_result(OperationResult(operation_id=_OPERATION_ID))
    retry_started = asyncio.Event()
    calls = 0

    class Body(httpx.AsyncByteStream):
        def __init__(self, request: httpx.Request) -> None:
            self.request = request

        async def __aiter__(self):
            if calls == 1:
                yield payload[:1]
                raise httpx.ReadError("mid-body disconnect", request=self.request)
            retry_started.set()
            await asyncio.Event().wait()
            yield payload

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"Content-Length": str(len(payload))},
            stream=Body(request),
        )

    service, http = _service(handle)
    task = asyncio.create_task(service.get_operation_result("run", _OPERATION_ID, ref))
    try:
        await asyncio.wait_for(retry_started.wait(), 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.1)
    finally:
        await service.close()
        await http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        ("auth", RemoteTrainingHttpError),
        ("protocol", httpx.LocalProtocolError),
        ("validation", ValidationError),
    ],
)
async def test_event_polling_does_not_retry_terminal_failures(failure, error_type):
    calls = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if failure == "auth":
            return httpx.Response(401, json={"detail": "denied"})
        if failure == "protocol":
            raise httpx.LocalProtocolError("invalid protocol", request=request)
        return httpx.Response(200, json={"events": [], "next_cursor": -1})

    service, http = _service(handle, max_retries=5)
    observer = _RunEventObserver(service, "run", poll_interval_s=0.001)
    future = observer.reserve(_OPERATION_ID)
    try:
        with pytest.raises(error_type):
            await asyncio.wait_for(future, 0.1)
        assert calls == 1
    finally:
        await observer.close(timeout_s=0.1)
        await service.close()
        await http.aclose()
