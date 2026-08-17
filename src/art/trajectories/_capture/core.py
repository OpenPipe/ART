from __future__ import annotations

import contextvars
from copy import deepcopy
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
_SSE_DELIMITERS = (b"\r\n\r\n", b"\n\n", b"\r\r")


def _terminal_sse_event(endpoint: Endpoint, block: bytes) -> bool:
    try:
        lines = block.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    event_name: str | None = None
    data_lines: list[str] = []
    for line in lines:
        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    data = "\n".join(data_lines)
    if endpoint in {"chat_completions", "completions"}:
        return data == "[DONE]"
    if endpoint == "responses" and event_name == "response.completed":
        return True
    if endpoint == "messages" and event_name == "message_stop":
        return True
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    event_type = payload.get("type")
    return (endpoint == "responses" and event_type == "response.completed") or (
        endpoint == "messages" and event_type == "message_stop"
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
    _event_start: int = field(default=0, init=False, repr=False)
    _scan_start: int = field(default=0, init=False, repr=False)

    def add(self, chunk: bytes) -> None:
        if not self.captured:
            self.body.extend(chunk)
            if self.request.get("stream") is True and self._reached_terminal_event():
                self.finish()

    def _reached_terminal_event(self) -> bool:
        while True:
            boundaries = [
                (index, len(delimiter))
                for delimiter in _SSE_DELIMITERS
                if (index := self.body.find(delimiter, self._scan_start)) >= 0
            ]
            if not boundaries:
                self._scan_start = max(self._event_start, len(self.body) - 3)
                return False
            index, delimiter_length = min(boundaries)
            block = bytes(self.body[self._event_start : index])
            self._event_start = index + delimiter_length
            self._scan_start = self._event_start
            if _terminal_sse_event(self.endpoint, block):
                return True

    def discard(self) -> None:
        self.body.clear()
        self.captured = True

    def finish(self) -> None:
        if self.captured:
            return
        self.captured = True
        if self.status_code is None or not 200 <= self.status_code < 300:
            return
        try:
            exchange = build_exchange(
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
    if (
        trajectory.messages_and_choices
        or trajectory.tools is not None
        or trajectory.additional_histories
    ):
        logger.debug("Ignoring exchange captured into a legacy trajectory")
        return
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
        try:
            return deepcopy(
                {key: item for key, item in value.items() if isinstance(key, str)}
            )
        except Exception:
            return None
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
        CaptureState(
            trajectory=trajectory,
            endpoint=endpoint,
            request=request,
        ),
        _adapter_active.set(True),
    )


def reset(token: contextvars.Token[bool] | None) -> None:
    if token is not None:
        _adapter_active.reset(token)
