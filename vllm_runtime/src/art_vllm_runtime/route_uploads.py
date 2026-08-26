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


class S3PutGrant(BaseModel):
    """One bounded write to one exact caller-owned S3-compatible object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: SecretStr
    required_headers: dict[str, SecretStr] = Field(default_factory=dict, max_length=64)
    expires_at: datetime
    max_bytes: int = Field(ge=1)
    client_reference: str = Field(min_length=1, max_length=512)

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
        if any(
            _HTTP_HEADER_NAME.fullmatch(name) is None for name in normalized_headers
        ):
            raise ValueError("route upload header name is invalid")
        if len(normalized_headers) != len(set(normalized_headers)):
            raise ValueError("route upload headers must be case-insensitively unique")
        if any(len(name) > 128 for name in normalized_headers):
            raise ValueError("route upload header name is too long")
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


class _ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes | memoryview, ...]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            # httpcore accepts the buffer protocol.  Keeping memoryviews intact
            # avoids copying large NumPy-backed route arrays before the socket.
            yield chunk  # type: ignore[misc]


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
        self._validate_grant(grant, actual_bytes=actual_bytes)
        headers = {
            name: value.get_secret_value()
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
        headers.setdefault("Content-Length", str(actual_bytes))
        request = self._client.build_request(
            "PUT",
            grant.url.get_secret_value(),
            headers=headers,
            content=_ChunkStream(chunks),
        )
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
            raise RouteUploadError(
                f"signed route upload failed with HTTP {response.status_code}"
            )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _validate_grant(self, grant: S3PutGrant, *, actual_bytes: int) -> None:
        if actual_bytes <= 0 or actual_bytes > grant.max_bytes:
            raise RouteUploadError("route payload exceeds its signed byte bound")
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
    grant: S3PutGrant
    future: RouteUploadFuture
    event: asyncio.Event
    reserved_bytes: int
    created_at: float
    terminal_at: float | None = None
    task: asyncio.Task[None] | None = None


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


class RouteUploadManager:
    """Owner-scoped upload state with bounded pending bytes and long-poll events."""

    def __init__(
        self,
        uploader: PresignedPutUploader,
        *,
        max_pending_uploads: int = 64,
        max_pending_bytes: int = 16 << 30,
        max_object_bytes: int = 4 << 30,
        max_status_records: int = 65_536,
        terminal_ttl_s: float = 300.0,
        shutdown_timeout_s: float = 10.0,
    ) -> None:
        if (
            min(
                max_pending_uploads,
                max_pending_bytes,
                max_object_bytes,
                max_status_records,
            )
            <= 0
        ):
            raise ValueError("route upload limits must be positive")
        if max_object_bytes > max_pending_bytes:
            raise ValueError("one route object cannot exceed the pending byte budget")
        if terminal_ttl_s <= 0 or shutdown_timeout_s <= 0:
            raise ValueError("route upload timeouts must be positive")
        self._uploader = uploader
        self._max_pending_uploads = max_pending_uploads
        self._max_pending_bytes = max_pending_bytes
        self._max_object_bytes = max_object_bytes
        self._max_status_records = max_status_records
        self._terminal_ttl_s = terminal_ttl_s
        self._shutdown_timeout_s = shutdown_timeout_s
        self._condition = asyncio.Condition()
        self._records: dict[str, _UploadRecord] = {}
        self._request_ids: dict[tuple[str, str], str] = {}
        self._pending_uploads = 0
        self._pending_bytes = 0
        self._closed = False

    async def reserve(
        self,
        *,
        owner_id: str,
        request_id: str,
        grant: S3PutGrant,
    ) -> RouteUploadLease:
        if not owner_id or not request_id:
            raise ValueError("route upload owner and request ids must be non-empty")
        if len(owner_id) > 256 or len(request_id) > 256:
            raise ValueError("route upload owner or request id is too long")
        if grant.max_bytes > self._max_object_bytes:
            raise RouteUploadError("route upload grant exceeds the object byte limit")
        async with self._condition:
            self._purge_expired_locked()
            self._make_status_room_locked()
            key = (owner_id, request_id)
            prior_id = self._request_ids.get(key)
            if prior_id is not None:
                prior = self._records[prior_id]
                if prior.grant != grant:
                    raise RouteUploadConflict(
                        "route upload request id was reused with a different grant"
                    )
                raise RouteUploadConflict("route upload request id is already admitted")
            while not self._has_capacity(grant.max_bytes):
                self._require_open()
                remaining = (
                    grant.expires_at.astimezone(timezone.utc)
                    - datetime.now(timezone.utc)
                ).total_seconds()
                if remaining <= 0:
                    raise RouteUploadError(
                        "route upload grant expired during admission"
                    )
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError as exc:
                    raise RouteUploadError(
                        "route upload grant expired during admission"
                    ) from exc
                self._purge_expired_locked()
            self._require_open()
            operation_id = uuid4().hex
            record = _UploadRecord(
                owner_id=owner_id,
                request_id=request_id,
                grant=grant,
                future=RouteUploadFuture(
                    operation_id=operation_id,
                    client_reference=grant.client_reference,
                    state="pending",
                ),
                event=asyncio.Event(),
                reserved_bytes=grant.max_bytes,
                created_at=time.monotonic(),
            )
            self._records[operation_id] = record
            self._request_ids[key] = operation_id
            self._pending_uploads += 1
            self._pending_bytes += grant.max_bytes
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
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
        except TimeoutError:
            pass
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
                await self._uploader.put(
                    record.grant, chunks, actual_bytes=actual_bytes
                )
            except BaseException as exc:
                if isinstance(exc, asyncio.CancelledError):
                    message = "route upload cancelled"
                else:
                    message = f"{type(exc).__name__}: {exc}"[:1024]
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
            if record.terminal_at is not None and record.terminal_at <= cutoff
        ]
        for operation_id in expired:
            record = self._records.pop(operation_id)
            self._request_ids.pop((record.owner_id, record.request_id), None)

    def _make_status_room_locked(self) -> None:
        overflow = len(self._records) - self._max_status_records + 1
        if overflow <= 0:
            return
        terminal = sorted(
            (
                (record.terminal_at, operation_id)
                for operation_id, record in self._records.items()
                if record.terminal_at is not None
            ),
            key=lambda value: value[0],
        )
        if len(terminal) < overflow:
            raise RouteUploadError("route upload status capacity is exhausted")
        for _terminal_at, operation_id in terminal[:overflow]:
            record = self._records.pop(operation_id)
            self._request_ids.pop((record.owner_id, record.request_id), None)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("route upload manager is closed")
