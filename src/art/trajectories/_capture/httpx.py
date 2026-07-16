from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
from httpx._client import UseClientDefault
from httpx._types import AuthTypes
from typing_extensions import TypedDict, Unpack

from .core import CaptureState, begin, reset

_STATE = "_art_trajectory_capture"


class _SendOptions(TypedDict, total=False):
    stream: bool
    auth: AuthTypes | UseClientDefault | None
    follow_redirects: bool | UseClientDefault


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
        self: httpx.Client,
        request: httpx.Request,
        **kwargs: Unpack[_SendOptions],
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
        self: httpx.AsyncClient,
        request: httpx.Request,
        **kwargs: Unpack[_SendOptions],
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
        completed = False
        try:
            for chunk in original_iter(self, chunk_size):
                if state is not None:
                    state.add(chunk)
                yield chunk
            completed = True
        finally:
            if state is not None and (completed or state.request.get("stream") is True):
                state.finish()

    async def aiter_bytes(
        self: httpx.Response, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        state: CaptureState | None = getattr(self, _STATE, None)
        completed = False
        try:
            async for chunk in original_aiter(self, chunk_size):
                if state is not None:
                    state.add(chunk)
                yield chunk
            completed = True
        finally:
            if state is not None and (completed or state.request.get("stream") is True):
                state.finish()

    def close(self: httpx.Response) -> None:
        original_close(self)
        state: CaptureState | None = getattr(self, _STATE, None)
        if state is not None and state.request.get("stream") is True:
            state.finish()

    async def aclose(self: httpx.Response) -> None:
        await original_aclose(self)
        state: CaptureState | None = getattr(self, _STATE, None)
        if state is not None and state.request.get("stream") is True:
            state.finish()

    setattr(send, "_art_capture", True)
    setattr(async_send, "_art_capture", True)
    setattr(httpx.Client, "send", send)
    setattr(httpx.AsyncClient, "send", async_send)
    setattr(httpx.Response, "iter_bytes", iter_bytes)
    setattr(httpx.Response, "aiter_bytes", aiter_bytes)
    setattr(httpx.Response, "close", close)
    setattr(httpx.Response, "aclose", aclose)
