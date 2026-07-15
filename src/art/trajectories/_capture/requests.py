from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import requests

from .core import CaptureState, begin, reset

_STATE = "_art_trajectory_capture"


def install() -> None:
    if getattr(requests.Session.send, "_art_capture", False):
        return
    original_send = requests.Session.send
    original_iter = requests.Response.iter_content

    def send(
        self: requests.Session, request: requests.PreparedRequest, **kwargs: Any
    ) -> requests.Response:
        state, token = begin(request.method or "GET", request.url or "", request.body)
        try:
            response = original_send(self, request, **kwargs)
        finally:
            reset(token)
        if state is not None:
            state.status_code = response.status_code
            setattr(response, _STATE, state)
            if not kwargs.get("stream", self.stream):
                state.add(response.content)
                state.finish()
        return response

    def iter_content(
        self: requests.Response,
        chunk_size: int | None = 1,
        decode_unicode: bool = False,
    ) -> Iterator[str | bytes]:
        state: CaptureState | None = getattr(self, _STATE, None)
        completed = False
        try:
            for chunk in original_iter(
                self, chunk_size=chunk_size, decode_unicode=decode_unicode
            ):
                if state is not None:
                    if isinstance(chunk, str):
                        chunk = chunk.encode(self.encoding or "utf-8")
                    if isinstance(chunk, bytes):
                        state.add(chunk)
                yield chunk
            completed = True
        finally:
            if state is not None and (completed or state.request.get("stream") is True):
                state.finish()

    setattr(send, "_art_capture", True)
    setattr(requests.Session, "send", send)
    setattr(requests.Response, "iter_content", iter_content)
