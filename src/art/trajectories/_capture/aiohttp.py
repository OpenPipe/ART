from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from .core import CaptureState, begin, reset


class _CapturedStream:
    def __init__(self, stream: Any, state: CaptureState) -> None:
        self._stream = stream
        self._state = state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def _record(self, value: Any) -> Any:
        chunk = value[0] if isinstance(value, tuple) else value
        if isinstance(chunk, bytes):
            self._state.add(chunk)
        if self._stream.at_eof():
            self._state.finish()
        return value

    async def read(self, *args: Any, **kwargs: Any) -> bytes:
        return self._record(await self._stream.read(*args, **kwargs))

    async def readany(self) -> bytes:
        return self._record(await self._stream.readany())

    async def readline(self) -> bytes:
        return self._record(await self._stream.readline())

    async def readchunk(self) -> tuple[bytes, bool]:
        return self._record(await self._stream.readchunk())

    async def _iterate(self, iterator: Any) -> AsyncIterator[bytes]:
        try:
            async for chunk in iterator:
                yield self._record(chunk)
        finally:
            self._state.finish()

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate(self._stream.__aiter__())

    def iter_any(self) -> AsyncIterator[bytes]:
        return self._iterate(self._stream.iter_any())

    def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
        return self._iterate(self._stream.iter_chunked(size))


def install() -> None:
    if getattr(aiohttp.ClientSession._request, "_art_capture", False):
        return
    original = aiohttp.ClientSession._request

    async def request(
        self: aiohttp.ClientSession,
        method: str,
        str_or_url: Any,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        body: Any = kwargs.get("json")
        if body is None:
            body = kwargs.get("data")
        state, token = begin(method, str(str_or_url), body)
        try:
            response = await original(self, method, str_or_url, **kwargs)
        finally:
            reset(token)
        if state is not None:
            state.status_code = response.status
            response.content = _CapturedStream(response.content, state)  # type: ignore[assignment]
        return response

    request._art_capture = True  # type: ignore[attr-defined]
    aiohttp.ClientSession._request = request  # type: ignore[method-assign]
