import asyncio
from datetime import datetime, timedelta, timezone

from art_vllm_runtime.route_uploads import (
    PresignedPutUploader,
    RouteUploadConflict,
    RouteUploadError,
    RouteUploadForbidden,
    RouteUploadManager,
    S3PutGrant,
)
import httpx
from pydantic import SecretStr, ValidationError
import pytest


def _grant(*, max_bytes: int = 32, reference: str = "route-1") -> S3PutGrant:
    return S3PutGrant(
        url=SecretStr("https://objects.test.example/bucket/exact-object?signature=x"),
        required_headers={"x-amz-meta-kind": SecretStr("routes")},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        max_bytes=max_bytes,
        client_reference=reference,
    )


def _uploader(handler) -> PresignedPutUploader:
    return PresignedPutUploader(
        allowed_host_suffixes=("test.example",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_grant_rejects_non_https_and_non_object_destinations() -> None:
    for url in (
        "http://objects.test.example/bucket/key",
        "https://objects.test.example/",
    ):
        with pytest.raises(ValidationError):
            S3PutGrant(
                url=SecretStr(url),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
                max_bytes=1,
                client_reference="x",
            )

    for name, value in (("bad header", "ok"), ("x-test", "bad\r\nvalue")):
        with pytest.raises(ValidationError):
            S3PutGrant(
                url=SecretStr("https://objects.test.example/bucket/key"),
                required_headers={name: SecretStr(value)},
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
                max_bytes=1,
                client_reference="x",
            )


@pytest.mark.asyncio
async def test_upload_streams_exact_chunks_and_reports_owner_scoped_completion() -> (
    None
):
    observed = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed["method"] = request.method
        observed["url"] = str(request.url)
        observed["headers"] = dict(request.headers)
        observed["body"] = await request.aread()
        return httpx.Response(200)

    manager = RouteUploadManager(
        _uploader(handler), max_pending_bytes=64, max_object_bytes=64
    )
    lease = await manager.reserve(
        owner_id="tenant-a", request_id="request-a", grant=_grant()
    )
    future = await lease.publish((b"abc", memoryview(b"def")))
    assert future.state == "pending"
    ready = await manager.wait(
        owner_id="tenant-a", operation_id=future.operation_id, timeout_s=1
    )
    assert ready.state == "ready"
    assert ready.actual_bytes == 6
    assert observed["method"] == "PUT"
    assert observed["body"] == b"abcdef"
    assert observed["headers"]["content-length"] == "6"
    with pytest.raises(RouteUploadForbidden):
        await manager.get(owner_id="tenant-b", operation_id=future.operation_id)
    await manager.close()


@pytest.mark.asyncio
async def test_pending_byte_reservation_backpressures_before_generation() -> None:
    release = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200)

    manager = RouteUploadManager(
        _uploader(handler),
        max_pending_uploads=2,
        max_pending_bytes=32,
        max_object_bytes=32,
    )
    first = await manager.reserve(
        owner_id="tenant", request_id="first", grant=_grant(max_bytes=32)
    )
    await first.publish((b"a",))
    second_task = asyncio.create_task(
        manager.reserve(
            owner_id="tenant", request_id="second", grant=_grant(max_bytes=32)
        )
    )
    await asyncio.sleep(0)
    assert not second_task.done()
    release.set()
    first_ready = await manager.wait(
        owner_id="tenant", operation_id=first.future.operation_id, timeout_s=1
    )
    assert first_ready.state == "ready"
    second = await asyncio.wait_for(second_task, timeout=1)
    await second.fail("generation failed")
    await manager.close()


@pytest.mark.asyncio
async def test_oversize_redirect_and_idempotency_conflict_fail_closed() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(307, headers={"location": "https://evil.example/steal"})

    manager = RouteUploadManager(
        _uploader(handler), max_pending_bytes=64, max_object_bytes=64
    )
    lease = await manager.reserve(
        owner_id="tenant", request_id="oversize", grant=_grant(max_bytes=4)
    )
    with pytest.raises(RouteUploadError, match="byte bound"):
        await lease.publish((b"12345",))
    assert calls == 0

    redirect_grant = _grant(max_bytes=4)
    redirect = await manager.reserve(
        owner_id="tenant", request_id="redirect", grant=redirect_grant
    )
    future = await redirect.publish((b"1234",))
    failed = await manager.wait(
        owner_id="tenant", operation_id=future.operation_id, timeout_s=1
    )
    assert failed.state == "failed"
    assert "redirect" in (failed.error or "")
    assert calls == 1

    with pytest.raises(RouteUploadConflict, match="already admitted"):
        await manager.reserve(
            owner_id="tenant", request_id="redirect", grant=redirect_grant
        )
    with pytest.raises(RouteUploadConflict, match="different grant"):
        await manager.reserve(
            owner_id="tenant",
            request_id="redirect",
            grant=_grant(max_bytes=5, reference="changed"),
        )
    await manager.close()


@pytest.mark.asyncio
async def test_expired_and_non_allowlisted_grants_fail_without_network() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    uploader = _uploader(handler)
    expired = _grant(max_bytes=4).model_copy(
        update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)}
    )
    with pytest.raises(RouteUploadError, match="expired"):
        await uploader.put(expired, (b"1234",), actual_bytes=4)
    foreign = _grant(max_bytes=4).model_copy(
        update={"url": SecretStr("https://attacker.invalid/bucket/key")}
    )
    with pytest.raises(RouteUploadError, match="allowlisted"):
        await uploader.put(foreign, (b"1234",), actual_bytes=4)
    assert calls == 0
    await uploader.close()


@pytest.mark.asyncio
async def test_lease_context_and_shutdown_settle_unpublished_reservations() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unpublished reservations must not use the network")

    manager = RouteUploadManager(
        _uploader(handler), max_pending_bytes=64, max_object_bytes=64
    )
    async with await manager.reserve(
        owner_id="tenant", request_id="context", grant=_grant()
    ) as lease:
        operation_id = lease.future.operation_id
    failed = await manager.get(owner_id="tenant", operation_id=operation_id)
    assert failed.state == "failed"

    abandoned = await manager.reserve(
        owner_id="tenant", request_id="shutdown", grant=_grant()
    )
    await manager.close()
    assert abandoned.future.state == "failed"
    assert "closed" in (abandoned.future.error or "")
