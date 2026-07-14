from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from typing import Any

from .._protocols import build_exchange, endpoint_for_url
from .._scope import get_current_trajectory

logger = logging.getLogger(__name__)
_adapter_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "art_capture_adapter_active", default=False
)


@dataclass
class CaptureState:
    trajectory: Any
    endpoint: str
    request: dict[str, Any]
    start_time: datetime = field(default_factory=datetime.now)
    status_code: int | None = None
    body: bytearray = field(default_factory=bytearray)
    captured: bool = False

    def add(self, chunk: bytes) -> None:
        if not self.captured:
            self.body.extend(chunk)

    def finish(self) -> None:
        if self.captured:
            return
        self.captured = True
        if self.status_code is None or not 200 <= self.status_code < 300:
            return
        try:
            name, exchange = build_exchange(
                self.endpoint,
                self.request,
                bytes(self.body),
                start_time=self.start_time,
                end_time=datetime.now(),
            )
        except Exception as exc:
            logger.debug("Ignoring incomplete trajectory exchange: %s", exc)
            return
        getattr(self.trajectory.exchanges, name).append(exchange)


def _json_body(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.encode()
    if not isinstance(value, bytes):
        return None
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def begin(
    method: str,
    url: str,
    body: Any,
) -> tuple[CaptureState | None, contextvars.Token[bool] | None]:
    trajectory = get_current_trajectory(required=False)
    endpoint = endpoint_for_url(url)
    request = _json_body(body)
    if (
        trajectory is None
        or method.upper() != "POST"
        or endpoint is None
        or request is None
        or _adapter_active.get()
    ):
        return None, None
    return (
        CaptureState(trajectory=trajectory, endpoint=endpoint, request=request),
        _adapter_active.set(True),
    )


def reset(token: contextvars.Token[bool] | None) -> None:
    if token is not None:
        _adapter_active.reset(token)
