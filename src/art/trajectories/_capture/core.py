from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from typing import Any, assert_never

from .. import (
    ChatCompletionsExchange,
    CompletionsExchange,
    MessagesExchange,
    ResponsesExchange,
    Trajectory,
)
from .._protocols import Endpoint, Exchange, build_exchange, endpoint_for_url
from .._scope import get_current_trajectory

logger = logging.getLogger(__name__)
_adapter_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "art_capture_adapter_active", default=False
)


@dataclass
class CaptureState:
    trajectory: Trajectory
    endpoint: Endpoint
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
            _, exchange = build_exchange(
                self.endpoint,
                self.request,
                bytes(self.body),
                start_time=self.start_time,
                end_time=datetime.now(),
            )
        except Exception as exc:
            logger.debug("Ignoring incomplete trajectory exchange: %s", exc)
            return
        _append_exchange(self.trajectory, exchange)


def _append_exchange(trajectory: Trajectory, exchange: Exchange) -> None:
    if isinstance(exchange, ChatCompletionsExchange):
        trajectory.exchanges.chat_completions.append(exchange)
    elif isinstance(exchange, CompletionsExchange):
        trajectory.exchanges.completions.append(exchange)
    elif isinstance(exchange, ResponsesExchange):
        trajectory.exchanges.responses.append(exchange)
    elif isinstance(exchange, MessagesExchange):
        trajectory.exchanges.messages.append(exchange)
    else:
        assert_never(exchange)


def _json_body(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return None
        return {key: item for key, item in value.items() if isinstance(key, str)}
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
    body: object,
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
