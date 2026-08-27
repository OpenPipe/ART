import asyncio
from datetime import datetime, timedelta, timezone
import gc
import weakref

from art_vllm_runtime.route_uploads import (
    PresignedPutUploader,
    RouteUploadBusy,
    RouteUploadConflict,
    RouteUploadError,
    RouteUploadForbidden,
    RouteUploadManager,
    RouteUploadNotFound,
    RouteUploadReplay,
    RouteUploadReplayResponse,
    S3PutGrant,
)
import httpx
import numpy as np
from pydantic import SecretStr, ValidationError
import pytest


def _grant(
    *,
    max_bytes: int = 32,
    reference: str = "route-1",
    required_headers: dict[str, SecretStr] | None = None,
) -> S3PutGrant:
    return S3PutGrant(
        url=SecretStr("https://objects.test.example/bucket/exact-object?signature=x"),
        required_headers=required_headers
        or {"content-type": SecretStr("application/vnd.art.routed-experts-v2")},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        max_bytes=max_bytes,
        client_reference=reference,
    )


def _uploader(handler) -> PresignedPutUploader:
    return PresignedPutUploader(
        allowed_host_suffixes=("test.example",),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def _reserve(
    manager: RouteUploadManager,
    *,
    owner_id: str,
    request_id: str,
    grant: S3PutGrant,
    request_fingerprint: str = "request-fingerprint",
):
    return await manager.reserve(
        owner_id=owner_id,
        request_id=request_id,
        request_fingerprint=request_fingerprint,
        grant=grant,
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


def test_grant_accepts_only_payload_signature_headers() -> None:
    grant = S3PutGrant(
        url=SecretStr("https://objects.test.example/bucket/key?signature=x"),
        required_headers={
            "content-type": SecretStr("application/octet-stream"),
            "x-amz-content-sha256": SecretStr("UNSIGNED-PAYLOAD"),
            "x-amz-checksum-crc32c": SecretStr("AAAAAA=="),
        },
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        max_bytes=1,
        client_reference="x",
    )

    assert set(grant.required_headers) == {
        "content-type",
        "x-amz-content-sha256",
        "x-amz-checksum-crc32c",
    }

    for name, value in (
        ("bad header", "ok"),
        ("x-test", "bad\r\nvalue"),
        ("authorization", "secret"),
        ("host", "objects.test.example"),
        ("connection", "keep-alive"),
        ("proxy-authorization", "secret"),
        ("transfer-encoding", "chunked"),
        ("x-amz-meta-kind", "routes"),
        ("x-amz-security-token", "secret"),
    ):
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
        observed["content_lengths"] = request.headers.get_list("content-length")
        observed["body"] = await request.aread()
        return httpx.Response(200)

    manager = RouteUploadManager(
        _uploader(handler), max_pending_bytes=64, max_object_bytes=64
    )
    grant = _grant(
        required_headers={
            "Content-Length": SecretStr("6"),
            "Content-Type": SecretStr("application/vnd.art.routed-experts-v2"),
        }
    )
    lease = await _reserve(
        manager, owner_id="tenant-a", request_id="request-a", grant=grant
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
    assert observed["content_lengths"] == ["6"]
    with pytest.raises(RouteUploadForbidden):
        await manager.get(owner_id="tenant-b", operation_id=future.operation_id)
    await manager.close()


@pytest.mark.asyncio
async def test_completed_upload_releases_chunk_views_and_byte_reservation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert await request.aread() == b"routes"
        return httpx.Response(200)

    manager = RouteUploadManager(
        _uploader(handler), max_pending_bytes=64, max_object_bytes=64
    )
    lease = await _reserve(
        manager, owner_id="tenant", request_id="release", grant=_grant()
    )
    array = np.frombuffer(bytearray(b"routes"), dtype=np.uint8)
    array_ref = weakref.ref(array)
    view = memoryview(array)
    future = await lease.publish((view,))
    del view, array
    assert (
        await manager.wait(
            owner_id="tenant", operation_id=future.operation_id, timeout_s=1
        )
    ).state == "ready"
    gc.collect()

    assert manager._pending_bytes == 0
    assert manager._records[future.operation_id].task is None
    assert array_ref() is None
    await manager.close()


@pytest.mark.asyncio
async def test_pending_byte_reservation_rejects_before_generation() -> None:
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
    first = await _reserve(
        manager, owner_id="tenant", request_id="first", grant=_grant(max_bytes=32)
    )
    await first.publish((b"a",))
    with pytest.raises(RouteUploadBusy, match="capacity is exhausted"):
        await asyncio.wait_for(
            _reserve(
                manager,
                owner_id="tenant",
                request_id="second",
                grant=_grant(max_bytes=32),
            ),
            timeout=0.05,
        )
    assert manager._waiters == 0
    release.set()
    first_ready = await manager.wait(
        owner_id="tenant", operation_id=first.future.operation_id, timeout_s=1
    )
    assert first_ready.state == "ready"
    assert manager._records[first.future.operation_id].task is None
    second = await _reserve(
        manager, owner_id="tenant", request_id="second", grant=_grant(max_bytes=32)
    )
    await second.fail("generation failed")
    await manager.close()


@pytest.mark.asyncio
async def test_idempotent_replay_does_not_need_new_status_capacity() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unpublished reservations must not use the network")

    manager = RouteUploadManager(
        _uploader(handler),
        max_pending_uploads=1,
        max_pending_bytes=32,
        max_object_bytes=32,
        max_status_records=1,
    )
    grant = _grant(max_bytes=32)
    first = await _reserve(manager, owner_id="tenant", request_id="first", grant=grant)
    replay = await _reserve(manager, owner_id="tenant", request_id="first", grant=grant)
    assert isinstance(replay, RouteUploadReplay)
    assert replay.future.operation_id == first.future.operation_id
    replay_waiter = asyncio.create_task(replay.response())
    await asyncio.sleep(0)
    expected_response = RouteUploadReplayResponse(
        status_code=200,
        headers=(("content-type", "application/json"),),
        body=b'{"result":"ok"}',
    )
    await first.remember_response(expected_response)
    assert await replay_waiter == expected_response
    with pytest.raises(RouteUploadBusy, match="status capacity"):
        await _reserve(
            manager, owner_id="tenant", request_id="second", grant=_grant(max_bytes=1)
        )
    await first.fail("done")
    await manager.close()


@pytest.mark.asyncio
async def test_idempotency_key_is_bound_to_inference_request() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unpublished reservations must not use the network")

    manager = RouteUploadManager(
        _uploader(handler), max_pending_bytes=64, max_object_bytes=64
    )
    grant = _grant()
    lease = await _reserve(
        manager,
        owner_id="tenant",
        request_id="same",
        request_fingerprint="prompt-a",
        grant=grant,
    )
    with pytest.raises(RouteUploadConflict, match="different inference"):
        await _reserve(
            manager,
            owner_id="tenant",
            request_id="same",
            request_fingerprint="prompt-b",
            grant=grant,
        )
    await lease.fail("done")
    await manager.close()


@pytest.mark.asyncio
async def test_oversize_response_can_replay_reserved_compact_rejection() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("rejected reservations must not use the network")

    manager = RouteUploadManager(
        _uploader(handler),
        max_pending_bytes=64,
        max_object_bytes=64,
        max_replay_response_bytes=4 << 10,
    )
    grant = _grant()
    lease = await _reserve(manager, owner_id="tenant", request_id="large", grant=grant)
    replay = await _reserve(manager, owner_id="tenant", request_id="large", grant=grant)
    assert isinstance(replay, RouteUploadReplay)
    with pytest.raises(RouteUploadError, match="replay byte limit"):
        await lease.remember_response(
            RouteUploadReplayResponse(status_code=200, headers=(), body=b"x" * 5000)
        )
    rejection = RouteUploadReplayResponse(
        status_code=413,
        headers=(("content-type", "application/json"),),
        body=b'{"error":"response too large"}',
    )
    await lease.remember_response(rejection)
    await lease.fail("response too large")

    assert await replay.response() == rejection
    await manager.close()


@pytest.mark.asyncio
async def test_status_metadata_bytes_are_bounded_and_released() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unpublished reservations must not use the network")

    grant = _grant()
    metadata_bytes = RouteUploadManager._record_metadata_bytes(
        owner_id="tenant",
        request_id="first",
        request_fingerprint="request-fingerprint",
        grant=grant,
    )
    manager = RouteUploadManager(
        _uploader(handler),
        max_pending_bytes=64,
        max_object_bytes=64,
        max_status_metadata_bytes=metadata_bytes,
    )
    first = await _reserve(manager, owner_id="tenant", request_id="first", grant=grant)
    with pytest.raises(RouteUploadBusy, match="status capacity"):
        await _reserve(
            manager,
            owner_id="tenant",
            request_id="second",
            grant=_grant(reference="second"),
        )
    await first.fail("done")
    second = await _reserve(
        manager,
        owner_id="tenant",
        request_id="second",
        grant=_grant(reference="second"),
    )
    assert manager._status_metadata_bytes <= metadata_bytes
    await second.fail("done")
    await manager.close()


@pytest.mark.asyncio
async def test_waiter_admission_is_bounded() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unpublished reservations must not use the network")

    manager = RouteUploadManager(
        _uploader(handler),
        max_pending_bytes=64,
        max_object_bytes=64,
        max_waiters=1,
        max_waiters_per_operation=1,
    )
    lease = await _reserve(
        manager, owner_id="tenant", request_id="wait", grant=_grant()
    )
    waiter = asyncio.create_task(
        manager.wait(
            owner_id="tenant", operation_id=lease.future.operation_id, timeout_s=1
        )
    )
    await asyncio.sleep(0)
    with pytest.raises(RouteUploadBusy, match="waiter capacity"):
        await manager.wait(
            owner_id="tenant", operation_id=lease.future.operation_id, timeout_s=1
        )
    await lease.fail("done")
    assert (await waiter).state == "failed"
    assert manager._waiters == 0
    await manager.close()


@pytest.mark.asyncio
async def test_transient_upload_failure_retries_without_leaking_signed_url() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200)

    manager = RouteUploadManager(
        _uploader(handler),
        max_pending_bytes=64,
        max_object_bytes=64,
        retry_base_delay_s=0,
    )
    lease = await _reserve(
        manager, owner_id="tenant", request_id="retry", grant=_grant()
    )
    future = await lease.publish((b"routes",))
    ready = await manager.wait(
        owner_id="tenant", operation_id=future.operation_id, timeout_s=1
    )
    assert ready.state == "ready"
    assert calls == 2
    await manager.close()


@pytest.mark.asyncio
async def test_unexpected_transport_error_redacts_signed_url() -> None:
    secret = "https://objects.test.example/bucket/exact-object?signature=secret"

    async def handler(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError(secret)

    manager = RouteUploadManager(
        _uploader(handler), max_pending_bytes=64, max_object_bytes=64
    )
    lease = await _reserve(
        manager, owner_id="tenant", request_id="redact", grant=_grant()
    )
    future = await lease.publish((b"routes",))
    failed = await manager.wait(
        owner_id="tenant", operation_id=future.operation_id, timeout_s=1
    )
    assert failed.state == "failed"
    assert secret not in (failed.error or "")
    assert failed.error == "signed route upload failed: RuntimeError"
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
    lease = await _reserve(
        manager, owner_id="tenant", request_id="oversize", grant=_grant(max_bytes=4)
    )
    with pytest.raises(RouteUploadError, match="byte bound"):
        await lease.publish((b"12345",))
    assert calls == 0

    redirect_grant = _grant(max_bytes=4)
    redirect = await _reserve(
        manager, owner_id="tenant", request_id="redirect", grant=redirect_grant
    )
    future = await redirect.publish((b"1234",))
    failed = await manager.wait(
        owner_id="tenant", operation_id=future.operation_id, timeout_s=1
    )
    assert failed.state == "failed"
    assert "redirect" in (failed.error or "")
    assert calls == 1

    replay = await _reserve(
        manager, owner_id="tenant", request_id="redirect", grant=redirect_grant
    )
    assert isinstance(replay, RouteUploadReplay)
    assert replay.future.operation_id == future.operation_id
    assert replay.future.state == "failed"
    with pytest.raises(RouteUploadConflict, match="different grant"):
        await _reserve(
            manager,
            owner_id="tenant",
            request_id="redirect",
            grant=_grant(max_bytes=5, reference="changed"),
        )
    await manager.close()


@pytest.mark.asyncio
async def test_terminal_status_ttl_releases_idempotency_identity() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("failed reservations must not use the network")

    manager = RouteUploadManager(
        _uploader(handler),
        max_pending_bytes=64,
        max_object_bytes=64,
        terminal_ttl_s=0.001,
    )
    first = await _reserve(
        manager, owner_id="tenant", request_id="same", grant=_grant()
    )
    assert not isinstance(first, RouteUploadReplay)
    operation_id = first.future.operation_id
    await first.fail("expected")
    await asyncio.sleep(0.01)
    with pytest.raises(RouteUploadNotFound):
        await manager.get(owner_id="tenant", operation_id=operation_id)
    replacement = await _reserve(
        manager, owner_id="tenant", request_id="same", grant=_grant()
    )
    assert not isinstance(replacement, RouteUploadReplay)
    assert replacement.future.operation_id != operation_id
    await replacement.fail("done")
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

    manager = RouteUploadManager(uploader, max_pending_bytes=64, max_object_bytes=64)
    with pytest.raises(RouteUploadError, match="expired"):
        await _reserve(manager, owner_id="tenant", request_id="expired", grant=expired)
    with pytest.raises(RouteUploadError, match="allowlisted"):
        await _reserve(manager, owner_id="tenant", request_id="foreign", grant=foreign)
    assert manager._records == {}
    assert calls == 0
    await manager.close()


@pytest.mark.asyncio
async def test_lease_context_and_shutdown_settle_unpublished_reservations() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unpublished reservations must not use the network")

    manager = RouteUploadManager(
        _uploader(handler), max_pending_bytes=64, max_object_bytes=64
    )
    async with await _reserve(
        manager, owner_id="tenant", request_id="context", grant=_grant()
    ) as lease:
        operation_id = lease.future.operation_id
    failed = await manager.get(owner_id="tenant", operation_id=operation_id)
    assert failed.state == "failed"

    abandoned = await _reserve(
        manager, owner_id="tenant", request_id="shutdown", grant=_grant()
    )
    await manager.close()
    assert abandoned.future.state == "failed"
    assert "closed" in (abandoned.future.error or "")
