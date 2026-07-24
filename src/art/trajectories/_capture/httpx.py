from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
from httpx._client import UseClientDefault
from httpx._decoders import ContentDecoder
from httpx._types import AuthTypes
from typing_extensions import TypedDict, Unpack

from .core import CaptureState, begin, reset

_STATE = "_art_trajectory_capture"
_RAW_ACTIVE = "_art_trajectory_capture_raw_active"


class _SendOptions(TypedDict, total=False):
    stream: bool
    auth: AuthTypes | UseClientDefault | None
    follow_redirects: bool | UseClientDefault


def _capture_decoder(response: httpx.Response) -> ContentDecoder:
    shadow = httpx.Response(
        response.status_code,
        headers=response.headers,
        stream=httpx.ByteStream(b""),
    )
    return shadow._get_content_decoder()


def install() -> None:
    if getattr(httpx.Client.send, "_art_capture", False):
        return
    original_send = httpx.Client.send
    original_async_send = httpx.AsyncClient.send
    original_iter = httpx.Response.iter_raw
    original_aiter = httpx.Response.aiter_raw
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

    def iter_raw(
        self: httpx.Response, chunk_size: int | None = None
    ) -> Iterator[bytes]:
        state: CaptureState | None = getattr(self, _STATE, None)
        if state is None:
            yield from original_iter(self, chunk_size)
            return
        try:
            decoder = _capture_decoder(self)
        except Exception:
            state.discard()
            yield from original_iter(self, chunk_size)
            return
        completed = False
        usable = True
        setattr(self, _RAW_ACTIVE, True)
        try:
            for chunk in original_iter(self, chunk_size):
                if usable:
                    try:
                        state.add(decoder.decode(chunk))
                    except Exception:
                        state.discard()
                        usable = False
                yield chunk
            completed = True
        finally:
            if usable and (completed or state.request.get("stream") is True):
                try:
                    state.add(decoder.flush())
                except Exception:
                    state.discard()
                    usable = False
            setattr(self, _RAW_ACTIVE, False)
            if usable and (completed or state.request.get("stream") is True):
                state.finish()

    async def aiter_raw(
        self: httpx.Response, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        state: CaptureState | None = getattr(self, _STATE, None)
        if state is None:
            async for chunk in original_aiter(self, chunk_size):
                yield chunk
            return
        try:
            decoder = _capture_decoder(self)
        except Exception:
            state.discard()
            async for chunk in original_aiter(self, chunk_size):
                yield chunk
            return
        completed = False
        usable = True
        setattr(self, _RAW_ACTIVE, True)
        try:
            async for chunk in original_aiter(self, chunk_size):
                if usable:
                    try:
                        state.add(decoder.decode(chunk))
                    except Exception:
                        state.discard()
                        usable = False
                yield chunk
            completed = True
        finally:
            if usable and (completed or state.request.get("stream") is True):
                try:
                    state.add(decoder.flush())
                except Exception:
                    state.discard()
                    usable = False
            setattr(self, _RAW_ACTIVE, False)
            if usable and (completed or state.request.get("stream") is True):
                state.finish()

    def close(self: httpx.Response) -> None:
        original_close(self)
        state: CaptureState | None = getattr(self, _STATE, None)
        if (
            state is not None
            and not getattr(self, _RAW_ACTIVE, False)
            and state.request.get("stream") is True
        ):
            state.finish()

    async def aclose(self: httpx.Response) -> None:
        await original_aclose(self)
        state: CaptureState | None = getattr(self, _STATE, None)
        if (
            state is not None
            and not getattr(self, _RAW_ACTIVE, False)
            and state.request.get("stream") is True
        ):
            state.finish()

    setattr(send, "_art_capture", True)
    setattr(async_send, "_art_capture", True)
    setattr(httpx.Client, "send", send)
    setattr(httpx.AsyncClient, "send", async_send)
    setattr(httpx.Response, "iter_raw", iter_raw)
    setattr(httpx.Response, "aiter_raw", aiter_raw)
    setattr(httpx.Response, "close", close)
    setattr(httpx.Response, "aclose", aclose)
