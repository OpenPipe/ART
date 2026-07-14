from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from .core import CaptureState, begin, reset

_STATE = "_art_trajectory_capture"


def install() -> None:
    if getattr(httpx.Client.send, "_art_capture", False):
        return
    original_send = httpx.Client.send
    original_async_send = httpx.AsyncClient.send
    original_iter = httpx.Response.iter_bytes
    original_aiter = httpx.Response.aiter_bytes
    original_close = httpx.Response.close
    original_aclose = httpx.Response.aclose

    def send(
        self: httpx.Client, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        try:
            body = request.content
        except httpx.RequestNotRead:
            body = None
        state, token = begin(request.method, str(request.url), body)
        try:
            response = original_send(self, request, **kwargs)
        finally:
            reset(token)
        if state is not None:
            state.status_code = response.status_code
            setattr(response, _STATE, state)
            if not kwargs.get("stream", False):
                state.add(response.content)
                state.finish()
        return response

    async def async_send(
        self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        try:
            body = request.content
        except httpx.RequestNotRead:
            body = None
        state, token = begin(request.method, str(request.url), body)
        try:
            response = await original_async_send(self, request, **kwargs)
        finally:
            reset(token)
        if state is not None:
            state.status_code = response.status_code
            setattr(response, _STATE, state)
            if not kwargs.get("stream", False):
                state.add(response.content)
                state.finish()
        return response

    def iter_bytes(
        self: httpx.Response, chunk_size: int | None = None
    ) -> Iterator[bytes]:
        state: CaptureState | None = getattr(self, _STATE, None)
        try:
            for chunk in original_iter(self, chunk_size):
                if state is not None:
                    state.add(chunk)
                yield chunk
        finally:
            if state is not None:
                state.finish()

    async def aiter_bytes(
        self: httpx.Response, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        state: CaptureState | None = getattr(self, _STATE, None)
        try:
            async for chunk in original_aiter(self, chunk_size):
                if state is not None:
                    state.add(chunk)
                yield chunk
        finally:
            if state is not None:
                state.finish()

    def close(self: httpx.Response) -> None:
        original_close(self)
        if state := getattr(self, _STATE, None):
            state.finish()

    async def aclose(self: httpx.Response) -> None:
        await original_aclose(self)
        if state := getattr(self, _STATE, None):
            state.finish()

    send._art_capture = True  # type: ignore[attr-defined]
    async_send._art_capture = True  # type: ignore[attr-defined]
    httpx.Client.send = send  # type: ignore[method-assign]
    httpx.AsyncClient.send = async_send  # type: ignore[method-assign]
    httpx.Response.iter_bytes = iter_bytes  # type: ignore[method-assign]
    httpx.Response.aiter_bytes = aiter_bytes  # type: ignore[method-assign]
    httpx.Response.close = close  # type: ignore[method-assign]
    httpx.Response.aclose = aclose  # type: ignore[method-assign]
