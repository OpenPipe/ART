from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast, overload

import aiohttp
from yarl import URL

from .core import CaptureState, begin, reset


class _CapturedStream:
    def __init__(self, stream: aiohttp.StreamReader, state: CaptureState) -> None:
        self._stream = stream
        self._state = state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    @overload
    def _record(self, value: bytes) -> bytes: ...

    @overload
    def _record(self, value: tuple[bytes, bool]) -> tuple[bytes, bool]: ...

    def _record(self, value: bytes | tuple[bytes, bool]) -> bytes | tuple[bytes, bool]:
        chunk = value[0] if isinstance(value, tuple) else value
        if isinstance(chunk, bytes):
            self._state.add(chunk)
        if self._stream.at_eof():
            self._state.finish()
        return value

    async def read(self, n: int = -1) -> bytes:
        return self._record(await self._stream.read(n))

    async def readany(self) -> bytes:
        return self._record(await self._stream.readany())

    async def readline(self) -> bytes:
        return self._record(await self._stream.readline())

    async def readexactly(self, n: int) -> bytes:
        return self._record(await self._stream.readexactly(n))

    async def readuntil(self, separator: bytes = b"\n") -> bytes:
        return self._record(await self._stream.readuntil(separator))

    def read_nowait(self, n: int = -1) -> bytes:
        return self._record(self._stream.read_nowait(n))

    async def readchunk(self) -> tuple[bytes, bool]:
        return self._record(await self._stream.readchunk())

    async def _iterate(self, iterator: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        completed = False
        try:
            async for chunk in iterator:
                yield self._record(chunk)
            completed = True
        finally:
            if completed or self._state.request.get("stream") is True:
                self._state.finish()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate(self._stream.__aiter__())

    def iter_any(self) -> AsyncIterator[bytes]:
        return self._iterate(self._stream.iter_any())

    def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
        return self._iterate(self._stream.iter_chunked(size))

    async def _iterate_chunks(
        self, iterator: AsyncIterator[tuple[bytes, bool]]
    ) -> AsyncIterator[tuple[bytes, bool]]:
        completed = False
        try:
            async for chunk in iterator:
                yield self._record(chunk)
            completed = True
        finally:
            if completed or self._state.request.get("stream") is True:
                self._state.finish()

    def iter_chunks(self) -> AsyncIterator[tuple[bytes, bool]]:
        return self._iterate_chunks(self._stream.iter_chunks())


def install() -> None:
    if getattr(aiohttp.ClientSession._request, "_art_capture", False):
        return
    original = aiohttp.ClientSession._request

    async def request(
        self: aiohttp.ClientSession,
        method: str,
        str_or_url: str | URL,
        # This private aiohttp surface is version-dependent; preserve its options.
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        body: object = kwargs.get("json")
        if body is None:
            body = kwargs.get("data")
        state, token = begin(method, str(str_or_url), body)
        try:
            response = await original(self, method, str_or_url, **kwargs)
        finally:
            reset(token)
        if state is not None:
            state.status_code = response.status
            # The proxy preserves StreamReader's runtime surface while intercepting
            # reads; aiohttp exposes no protocol type for response.content.
            response.content = cast(
                aiohttp.StreamReader, _CapturedStream(response.content, state)
            )
        return response

    setattr(request, "_art_capture", True)
    setattr(aiohttp.ClientSession, "_request", request)
