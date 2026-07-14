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
            if not kwargs.get("stream", False):
                state.add(response.content)
                state.finish()
        return response

    def iter_content(
        self: requests.Response, *args: Any, **kwargs: Any
    ) -> Iterator[Any]:
        state: CaptureState | None = getattr(self, _STATE, None)
        try:
            for chunk in original_iter(self, *args, **kwargs):
                if state is not None:
                    if isinstance(chunk, str):
                        chunk = chunk.encode(self.encoding or "utf-8")
                    if isinstance(chunk, bytes):
                        state.add(chunk)
                yield chunk
        finally:
            if state is not None:
                state.finish()

    send._art_capture = True  # type: ignore[attr-defined]
    requests.Session.send = send  # type: ignore[method-assign]
    requests.Response.iter_content = iter_content  # type: ignore[method-assign]
