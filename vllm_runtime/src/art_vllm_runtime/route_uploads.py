"""Bounded delivery of request-scoped binary routes to signed object URLs.

This module is intentionally independent of vLLM internals.  A deployment can
reserve upload capacity before admitting generation, then hand the captured
route chunks to the lease after the model response is complete.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import time
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

RouteUploadState = Literal["pending", "ready", "failed"]
_HTTP_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_REPLAY_RESPONSE_RESERVE_BYTES = 2 << 10
_SIGNED_PUT_HEADERS = frozenset(
    {
        "content-length",
        "content-type",
        "x-amz-content-sha256",
    }
)
_SIGNED_PUT_CHECKSUM_HEADERS = frozenset(
    {
        "x-amz-checksum-crc32",
        "x-amz-checksum-crc32c",
        "x-amz-checksum-crc64nvme",
        "x-amz-checksum-sha1",
        "x-amz-checksum-sha256",
    }
)


def _is_allowed_signed_header(name: str) -> bool:
    return name in _SIGNED_PUT_HEADERS or name in _SIGNED_PUT_CHECKSUM_HEADERS


class S3PutGrant(BaseModel):
    """One bounded write to one exact caller-owned S3-compatible object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: SecretStr
    required_headers: dict[str, SecretStr] = Field(default_factory=dict, max_length=64)
    expires_at: datetime
    max_bytes: int = Field(ge=1)
    client_reference: str = Field(
        min_length=1,
        max_length=512,
        description=(
            "Owner-scoped request idempotency key; identical retries replay the "
            "original inference response and upload operation."
        ),
    )

    @model_validator(mode="after")
    def _validate_destination(self) -> "S3PutGrant":
        value = self.url.get_secret_value()
        if not value or len(value) > 16_384:
            raise ValueError("route upload URL length is outside configured bounds")
        parsed = urlsplit(value)
        if parsed.scheme.casefold() != "https":
            raise ValueError("route upload destination must use HTTPS")
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("route upload destination has invalid authority")
        if parsed.fragment:
            raise ValueError("route upload destination must not include a fragment")
        if not parsed.path or parsed.path == "/":
            raise ValueError("route upload destination must name one exact object")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ValueError("route upload expiry must be timezone-aware")
        normalized_headers = [name.strip().casefold() for name in self.required_headers]
        if any(name != name.strip() for name in self.required_headers):
            raise ValueError("route upload header name must not contain whitespace")
        if any(
            _HTTP_HEADER_NAME.fullmatch(name) is None for name in normalized_headers
        ):
            raise ValueError("route upload header name is invalid")
        if len(normalized_headers) != len(set(normalized_headers)):
            raise ValueError("route upload headers must be case-insensitively unique")
        if any(len(name) > 128 for name in normalized_headers):
            raise ValueError("route upload header name is too long")
        if any(not _is_allowed_signed_header(name) for name in normalized_headers):
            raise ValueError("route upload header is not permitted for signed S3 PUT")
        if any(
            len(value.get_secret_value()) > 8_192
            for value in self.required_headers.values()
        ):
            raise ValueError("route upload header value is too long")
        if any(
            "\r" in value.get_secret_value() or "\n" in value.get_secret_value()
            for value in self.required_headers.values()
        ):
            raise ValueError("route upload header value contains a line break")
        return self


class RouteUploadFuture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1, max_length=128)
    client_reference: str = Field(min_length=1, max_length=512)
    state: RouteUploadState
    actual_bytes: int | None = Field(default=None, ge=0)
    error: str | None = Field(default=None, max_length=1024)


class RouteUploadError(RuntimeError):
    pass


class RouteUploadNotFound(RouteUploadError):
    pass


class RouteUploadForbidden(RouteUploadError):
    pass


class RouteUploadConflict(RouteUploadError):
    pass


class RouteUploadBusy(RouteUploadError):
    pass


class RouteUploadTransientError(RouteUploadError):
    pass


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes | memoryview, ...]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            # httpcore accepts the buffer protocol.  Keeping memoryviews intact
            # avoids copying large NumPy-backed route arrays before the socket.
            yield chunk  # ty: ignore[invalid-yield]


class PresignedPutUploader:
    """Streams one payload to an allowlisted signed URL without redirects."""

    def __init__(
        self,
        *,
        allowed_host_suffixes: tuple[str, ...],
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not allowed_host_suffixes:
            raise ValueError("route upload requires an explicit destination allowlist")
        if timeout_s <= 0:
            raise ValueError("route upload timeout must be positive")
        self._allowed_host_suffixes = tuple(
            self._normalize_suffix(value) for value in allowed_host_suffixes
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_s),
        )

    async def put(
        self,
        grant: S3PutGrant,
        chunks: tuple[bytes | memoryview, ...],
        *,
        actual_bytes: int,
    ) -> None:
        self.validate_admission(grant)
        if actual_bytes <= 0 or actual_bytes > grant.max_bytes:
            raise RouteUploadError("route payload exceeds its signed byte bound")
        headers = {
            name.casefold(): value.get_secret_value()
            for name, value in grant.required_headers.items()
        }
        declared_length = next(
            (
                value
                for name, value in headers.items()
                if name.casefold() == "content-length"
            ),
            None,
        )
        if declared_length is not None and declared_length != str(actual_bytes):
            raise RouteUploadError("signed Content-Length does not match route bytes")
        headers.setdefault("content-length", str(actual_bytes))
        request = self._client.build_request(
            "PUT",
            grant.url.get_secret_value(),
            headers=headers,
            content=_ChunkStream(chunks),
        )
        try:
            async with self._client.stream(
                request.method,
                request.url,
                headers=request.headers,
                content=request.stream,
                follow_redirects=False,
            ) as response:
                if 200 <= response.status_code < 300:
                    return
                if 300 <= response.status_code < 400:
                    raise RouteUploadError("signed route upload refused a redirect")
                if response.status_code in {408, 429} or response.status_code >= 500:
                    raise RouteUploadTransientError(
                        "signed route upload reached a transient object-store failure"
                    )
                raise RouteUploadError(
                    f"signed route upload failed with HTTP {response.status_code}"
                )
        except httpx.TransportError as exc:
            raise RouteUploadTransientError(
                "signed route upload transport failed"
            ) from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def validate_admission(self, grant: S3PutGrant) -> None:
        """Reject a doomed or foreign destination before GPU generation starts."""

        if datetime.now(timezone.utc) >= grant.expires_at.astimezone(timezone.utc):
            raise RouteUploadError("route upload grant expired")
        host = urlsplit(grant.url.get_secret_value()).hostname
        assert host is not None
        normalized = host.rstrip(".").casefold()
        if not any(
            normalized == suffix or normalized.endswith(f".{suffix}")
            for suffix in self._allowed_host_suffixes
        ):
            raise RouteUploadError("route upload destination is not allowlisted")

    @staticmethod
    def _normalize_suffix(value: str) -> str:
        normalized = value.strip().lstrip(".").rstrip(".").casefold()
        if not normalized or "/" in normalized or ":" in normalized:
            raise ValueError("route upload host suffix is invalid")
        return normalized


@dataclass
class _UploadRecord:
    owner_id: str
    request_id: str
    request_fingerprint: str
    grant: S3PutGrant
    future: RouteUploadFuture
    event: asyncio.Event
    response_event: asyncio.Event
    reserved_bytes: int
    metadata_bytes: int
    created_at: float
    replay_response: RouteUploadReplayResponse | None = None
    replay_error: str | None = None
    terminal_at: float | None = None
    task: asyncio.Task[None] | None = None
    waiters: int = 0


@dataclass(frozen=True)
class RouteUploadReplayResponse:
    """Bounded HTTP result retained for exact request-id replay."""

    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes

    @property
    def metadata_bytes(self) -> int:
        return len(self.body) + sum(
            len(name.encode()) + len(value.encode()) for name, value in self.headers
        )


class RouteUploadReplay:
    """A prior admission for the same owner, idempotency key, and request."""

    def __init__(self, manager: "RouteUploadManager", record: _UploadRecord) -> None:
        self._manager = manager
        self._record = record

    @property
    def future(self) -> RouteUploadFuture:
        return self._record.future

    async def response(self) -> RouteUploadReplayResponse:
        return await self._manager._wait_replay_response(self._record)


class RouteUploadLease:
    """Capacity reservation held across one admitted generation request."""

    def __init__(self, manager: "RouteUploadManager", record: _UploadRecord) -> None:
        self._manager = manager
        self._record = record
        self._settled = False

    @property
    def future(self) -> RouteUploadFuture:
        return self._record.future

    async def publish(self, chunks: Iterable[bytes | memoryview]) -> RouteUploadFuture:
        if self._settled:
            raise RouteUploadConflict("route upload lease is already settled")
        self._settled = True
        values = tuple(chunks)
        actual_bytes = sum(map(len, values))
        await self._manager._publish(self._record, values, actual_bytes=actual_bytes)
        return self._record.future

    async def remember_response(
        self, response: RouteUploadReplayResponse
    ) -> RouteUploadReplayResponse:
        await self._manager._remember_response(self._record, response)
        return response

    async def fail(self, error: str) -> RouteUploadFuture:
        if self._settled:
            raise RouteUploadConflict("route upload lease is already settled")
        self._settled = True
        await self._manager._finish(self._record, state="failed", error=error[:1024])
        return self._record.future

    async def __aenter__(self) -> "RouteUploadLease":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if not self._settled:
            await self.fail("generation ended without publishing requested routes")
        if self._record.replay_response is None:
            await self._manager._abandon_response(
                self._record, "original generation produced no replayable response"
            )


class RouteUploadManager:
    """Owner-scoped upload state with bounded pending bytes and long-poll events."""

    def __init__(
        self,
        uploader: PresignedPutUploader,
        *,
        max_pending_uploads: int = 64,
        max_pending_bytes: int = 16 << 30,
        max_object_bytes: int = 4 << 30,
        max_status_records: int = 4_096,
        max_status_metadata_bytes: int = 64 << 20,
        max_replay_response_bytes: int = 8 << 20,
        max_waiters: int = 4_096,
        max_waiters_per_operation: int = 64,
        terminal_ttl_s: float = 300.0,
        shutdown_timeout_s: float = 10.0,
        max_upload_attempts: int = 3,
        retry_base_delay_s: float = 0.05,
    ) -> None:
        if (
            min(
                max_pending_uploads,
                max_pending_bytes,
                max_object_bytes,
                max_status_records,
                max_status_metadata_bytes,
                max_replay_response_bytes,
                max_waiters,
                max_waiters_per_operation,
                max_upload_attempts,
            )
            <= 0
        ):
            raise ValueError("route upload limits must be positive")
        if max_object_bytes > max_pending_bytes:
            raise ValueError("one route object cannot exceed the pending byte budget")
        if terminal_ttl_s <= 0 or shutdown_timeout_s <= 0 or retry_base_delay_s < 0:
            raise ValueError("route upload timeouts must be positive")
        self._uploader = uploader
        self._max_pending_uploads = max_pending_uploads
        self._max_pending_bytes = max_pending_bytes
        self._max_object_bytes = max_object_bytes
        self._max_status_records = max_status_records
        self._max_status_metadata_bytes = max_status_metadata_bytes
        self._max_replay_response_bytes = max_replay_response_bytes
        self._max_waiters = max_waiters
        self._max_waiters_per_operation = max_waiters_per_operation
        self._terminal_ttl_s = terminal_ttl_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._max_upload_attempts = max_upload_attempts
        self._retry_base_delay_s = retry_base_delay_s
        self._condition = asyncio.Condition()
        self._records: dict[str, _UploadRecord] = {}
        self._request_ids: dict[tuple[str, str], str] = {}
        self._pending_uploads = 0
        self._pending_bytes = 0
        self._status_metadata_bytes = 0
        self._waiters = 0
        self._closed = False

    async def reserve(
        self,
        *,
        owner_id: str,
        request_id: str,
        request_fingerprint: str,
        grant: S3PutGrant,
    ) -> RouteUploadLease | RouteUploadReplay:
        """Replay exact request identities and reject new work when capacity is full."""

        if not owner_id or not request_id or not request_fingerprint:
            raise ValueError(
                "route upload owner, request id, and fingerprint must be non-empty"
            )
        if (
            len(owner_id) > 256
            or len(request_id) > 512
            or len(request_fingerprint) > 128
        ):
            raise ValueError("route upload admission identity is too long")
        self._uploader.validate_admission(grant)
        if grant.max_bytes > self._max_object_bytes:
            raise RouteUploadError("route upload grant exceeds the object byte limit")
        metadata_bytes = self._record_metadata_bytes(
            owner_id=owner_id,
            request_id=request_id,
            request_fingerprint=request_fingerprint,
            grant=grant,
        )
        if metadata_bytes > self._max_status_metadata_bytes:
            raise RouteUploadError(
                "route upload metadata exceeds the status byte limit"
            )
        async with self._condition:
            self._purge_expired_locked()
            key = (owner_id, request_id)
            prior_id = self._request_ids.get(key)
            if prior_id is not None:
                prior = self._records[prior_id]
                if prior.grant != grant:
                    raise RouteUploadConflict(
                        "route upload request id was reused with a different grant"
                    )
                if prior.request_fingerprint != request_fingerprint:
                    raise RouteUploadConflict(
                        "route upload request id was reused for different inference"
                    )
                return RouteUploadReplay(self, prior)
            self._make_status_room_locked(
                additional_records=1,
                additional_metadata_bytes=metadata_bytes,
            )
            self._require_open()
            # Admission never queues behind uploads. The caller can retry with the
            # same request identity, which replays the original operation.
            if not self._has_capacity(grant.max_bytes):
                raise RouteUploadBusy("route upload capacity is exhausted")
            operation_id = uuid4().hex
            record = _UploadRecord(
                owner_id=owner_id,
                request_id=request_id,
                request_fingerprint=request_fingerprint,
                grant=grant,
                future=RouteUploadFuture(
                    operation_id=operation_id,
                    client_reference=grant.client_reference,
                    state="pending",
                ),
                event=asyncio.Event(),
                response_event=asyncio.Event(),
                reserved_bytes=grant.max_bytes,
                metadata_bytes=metadata_bytes,
                created_at=time.monotonic(),
            )
            self._records[operation_id] = record
            self._request_ids[key] = operation_id
            self._pending_uploads += 1
            self._pending_bytes += grant.max_bytes
            self._status_metadata_bytes += metadata_bytes
            return RouteUploadLease(self, record)

    async def get(self, *, owner_id: str, operation_id: str) -> RouteUploadFuture:
        async with self._condition:
            self._purge_expired_locked()
            return self._owned_record(owner_id, operation_id).future

    async def wait(
        self,
        *,
        owner_id: str,
        operation_id: str,
        timeout_s: float,
    ) -> RouteUploadFuture:
        if not 0 <= timeout_s <= 30.0:
            raise ValueError("route upload wait timeout must be in [0, 30] seconds")
        async with self._condition:
            self._purge_expired_locked()
            record = self._owned_record(owner_id, operation_id)
            event = record.event
            if record.future.state != "pending" or timeout_s == 0:
                return record.future
            if (
                self._waiters >= self._max_waiters
                or record.waiters >= self._max_waiters_per_operation
            ):
                raise RouteUploadBusy("route upload waiter capacity is exhausted")
            self._waiters += 1
            record.waiters += 1
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        except TimeoutError:
            pass
        finally:
            async with self._condition:
                self._waiters -= 1
                record.waiters -= 1
        return await self.get(owner_id=owner_id, operation_id=operation_id)

    async def close(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            for record in self._records.values():
                if record.future.state == "pending" and record.task is None:
                    record.future = RouteUploadFuture(
                        operation_id=record.future.operation_id,
                        client_reference=record.future.client_reference,
                        state="failed",
                        error="route upload manager closed before publication",
                    )
                    record.terminal_at = time.monotonic()
                    record.event.set()
                    self._pending_uploads -= 1
                    self._pending_bytes -= record.reserved_bytes
                if record.replay_response is None:
                    self._abandon_response_locked(
                        record, "route upload manager closed before response replay"
                    )
            tasks = tuple(
                record.task
                for record in self._records.values()
                if record.task is not None and not record.task.done()
            )
            self._condition.notify_all()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self._shutdown_timeout_s)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if not task.cancelled():
                    task.exception()
        await self._uploader.close()

    async def _wait_replay_response(
        self, record: _UploadRecord
    ) -> RouteUploadReplayResponse:
        async with self._condition:
            current = self._records.get(record.future.operation_id)
            if current is not record:
                raise RouteUploadNotFound("route upload operation was not found")
            if record.replay_response is not None:
                return record.replay_response
            if record.replay_error is not None:
                raise RouteUploadError(record.replay_error)
            if (
                self._waiters >= self._max_waiters
                or record.waiters >= self._max_waiters_per_operation
            ):
                raise RouteUploadBusy("route upload waiter capacity is exhausted")
            self._waiters += 1
            record.waiters += 1
        try:
            await record.response_event.wait()
        finally:
            async with self._condition:
                self._waiters -= 1
                record.waiters -= 1
        async with self._condition:
            if record.replay_response is not None:
                return record.replay_response
            raise RouteUploadError(
                record.replay_error or "original response is not replayable"
            )

    async def _remember_response(
        self,
        record: _UploadRecord,
        response: RouteUploadReplayResponse,
    ) -> None:
        response_bytes = response.metadata_bytes
        if response_bytes > self._max_replay_response_bytes:
            raise RouteUploadError("inference response exceeds replay byte limit")
        additional_bytes = max(0, response_bytes - _REPLAY_RESPONSE_RESERVE_BYTES)
        async with self._condition:
            current = self._records.get(record.future.operation_id)
            if current is not record:
                raise RouteUploadNotFound("route upload operation was not found")
            if record.replay_response is not None:
                if record.replay_response != response:
                    raise RouteUploadConflict(
                        "route upload replay response was already recorded"
                    )
                return
            if record.replay_error is not None:
                raise RouteUploadConflict(
                    "route upload replay response was already abandoned"
                )
            self._make_status_room_locked(
                additional_metadata_bytes=additional_bytes,
                exclude_operation_id=record.future.operation_id,
            )
            record.replay_response = response
            record.metadata_bytes += additional_bytes
            self._status_metadata_bytes += additional_bytes
            record.response_event.set()
            self._condition.notify_all()

    async def _abandon_response(self, record: _UploadRecord, error: str) -> None:
        async with self._condition:
            if self._records.get(record.future.operation_id) is record:
                self._abandon_response_locked(record, error)

    def _abandon_response_locked(self, record: _UploadRecord, error: str) -> None:
        if record.replay_response is None and record.replay_error is None:
            record.replay_error = error[:1024]
            record.response_event.set()
            self._condition.notify_all()

    async def _publish(
        self,
        record: _UploadRecord,
        chunks: tuple[bytes | memoryview, ...],
        *,
        actual_bytes: int,
    ) -> None:
        if actual_bytes <= 0 or actual_bytes > record.grant.max_bytes:
            await self._finish(
                record,
                state="failed",
                error="route payload exceeds its signed byte bound",
            )
            raise RouteUploadError("route payload exceeds its signed byte bound")

        async def upload() -> None:
            try:
                for attempt in range(self._max_upload_attempts):
                    try:
                        await self._uploader.put(
                            record.grant, chunks, actual_bytes=actual_bytes
                        )
                        break
                    except RouteUploadTransientError:
                        if attempt + 1 >= self._max_upload_attempts:
                            raise
                        delay = self._retry_base_delay_s * (2**attempt)
                        remaining = (
                            record.grant.expires_at.astimezone(timezone.utc)
                            - datetime.now(timezone.utc)
                        ).total_seconds()
                        if remaining <= delay:
                            raise RouteUploadError(
                                "route upload grant expires before retry"
                            )
                        await asyncio.sleep(delay)
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError):
                    message = "route upload cancelled"
                elif isinstance(exc, RouteUploadError):
                    message = str(exc)[:1024]
                else:
                    # Transport exceptions can contain the signed URL. Status
                    # records expose only a bounded class, never bearer grants.
                    message = f"signed route upload failed: {type(exc).__name__}"
                await self._finish(record, state="failed", error=message)
                if isinstance(exc, asyncio.CancelledError):
                    raise
            else:
                await self._finish(
                    record,
                    state="ready",
                    actual_bytes=actual_bytes,
                )

        async with self._condition:
            if record.task is not None or record.future.state != "pending":
                raise RouteUploadConflict("route upload was already started")
            record.task = asyncio.create_task(
                upload(), name=f"route-upload-{record.future.operation_id}"
            )

    async def _finish(
        self,
        record: _UploadRecord,
        *,
        state: Literal["ready", "failed"],
        actual_bytes: int | None = None,
        error: str | None = None,
    ) -> None:
        async with self._condition:
            if record.future.state != "pending":
                return
            record.future = RouteUploadFuture(
                operation_id=record.future.operation_id,
                client_reference=record.future.client_reference,
                state=state,
                actual_bytes=actual_bytes,
                error=(error or None),
            )
            record.terminal_at = time.monotonic()
            # A completed task must not retain its stream closure or chunk views
            # for the duration of the status TTL.
            record.task = None
            self._pending_uploads -= 1
            self._pending_bytes -= record.reserved_bytes
            record.event.set()
            self._make_status_room_locked()
            self._condition.notify_all()

    def _owned_record(self, owner_id: str, operation_id: str) -> _UploadRecord:
        record = self._records.get(operation_id)
        if record is None:
            raise RouteUploadNotFound("route upload operation was not found")
        if record.owner_id != owner_id:
            raise RouteUploadForbidden(
                "route upload operation belongs to another owner"
            )
        return record

    def _has_capacity(self, reserved_bytes: int) -> bool:
        return (
            self._pending_uploads < self._max_pending_uploads
            and self._pending_bytes + reserved_bytes <= self._max_pending_bytes
        )

    def _purge_expired_locked(self) -> None:
        cutoff = time.monotonic() - self._terminal_ttl_s
        expired = [
            operation_id
            for operation_id, record in self._records.items()
            if (
                record.terminal_at is not None
                and record.terminal_at <= cutoff
                and record.waiters == 0
            )
        ]
        for operation_id in expired:
            self._evict_record_locked(operation_id)

    def _make_status_room_locked(
        self,
        *,
        additional_records: int = 0,
        additional_metadata_bytes: int = 0,
        exclude_operation_id: str | None = None,
    ) -> None:
        terminal = iter(
            sorted(
                (
                    (record.terminal_at, operation_id)
                    for operation_id, record in self._records.items()
                    if (
                        operation_id != exclude_operation_id
                        and record.terminal_at is not None
                        and record.waiters == 0
                    )
                ),
                key=lambda value: value[0],
            )
        )
        while (
            len(self._records) + additional_records > self._max_status_records
            or self._status_metadata_bytes + additional_metadata_bytes
            > self._max_status_metadata_bytes
        ):
            try:
                _terminal_at, operation_id = next(terminal)
            except StopIteration as exc:
                raise RouteUploadBusy(
                    "route upload status capacity is exhausted"
                ) from exc
            self._evict_record_locked(operation_id)

    def _evict_record_locked(self, operation_id: str) -> None:
        record = self._records.pop(operation_id)
        self._request_ids.pop((record.owner_id, record.request_id), None)
        self._status_metadata_bytes -= record.metadata_bytes

    @staticmethod
    def _grant_metadata_bytes(grant: S3PutGrant) -> int:
        return (
            len(grant.url.get_secret_value().encode())
            + len(grant.client_reference.encode())
            + sum(
                len(name.encode()) + len(value.get_secret_value().encode())
                for name, value in grant.required_headers.items()
            )
        )

    @classmethod
    def _record_metadata_bytes(
        cls,
        *,
        owner_id: str,
        request_id: str,
        request_fingerprint: str,
        grant: S3PutGrant,
    ) -> int:
        # Include bounded identity/status fields, not only caller-controlled URL
        # data, so the configured status-memory ceiling is an actual ceiling.
        return (
            cls._grant_metadata_bytes(grant)
            + len(owner_id.encode())
            + len(request_id.encode())
            + len(request_fingerprint.encode())
            + 128
            + _REPLAY_RESPONSE_RESERVE_BYTES
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("route upload manager is closed")
