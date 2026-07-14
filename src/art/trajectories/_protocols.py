from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
from typing import Any, Literal
from urllib.parse import urlsplit

from anthropic._types import NOT_GIVEN
from anthropic.lib.streaming._messages import accumulate_event
from anthropic.types import Message, ParsedMessage, RawMessageStreamEvent
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.responses import Response
from pydantic import TypeAdapter

from ..openai import init_chat_completion, update_chat_completion
from . import (
    ChatCompletionsExchange,
    ChatCompletionsRequest,
    CompletionsExchange,
    CompletionsRequest,
    MessagesExchange,
    MessagesRequest,
    ResponsesExchange,
    ResponsesRequest,
)

Endpoint = Literal["chat_completions", "completions", "responses", "messages"]
Exchange = (
    ChatCompletionsExchange | CompletionsExchange | ResponsesExchange | MessagesExchange
)
ResponseModel = ChatCompletion | Completion | Response | Message
SSEPayload = dict[str, Any] | Literal["[DONE]"]
_ENDPOINTS = {
    "/v1/chat/completions": "chat_completions",
    "/v1/completions": "completions",
    "/v1/responses": "responses",
    "/v1/messages": "messages",
}


def endpoint_for_url(url: str) -> Endpoint | None:
    path = urlsplit(url).path.rstrip("/")
    return next(
        (value for suffix, value in _ENDPOINTS.items() if path.endswith(suffix)), None
    )


def _sse_events(body: bytes) -> list[tuple[str | None, SSEPayload]]:
    text = body.decode("utf-8").replace("\r\n", "\n")
    events: list[tuple[str | None, SSEPayload]] = []
    for block in text.split("\n\n"):
        event_name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines)
        if raw == "[DONE]":
            events.append((event_name, "[DONE]"))
        else:
            value = json.loads(raw)
            if isinstance(value, dict):
                events.append((event_name, value))
    return events


def _chat_response(body: bytes, *, stream: bool) -> ChatCompletion:
    if not stream:
        return ChatCompletion.model_validate_json(body)
    response: ChatCompletion | None = None
    done = False
    for _, payload in _sse_events(body):
        if payload == "[DONE]":
            done = True
            continue
        chunk = ChatCompletionChunk.model_validate(payload)
        if response is None:
            response = init_chat_completion(chunk)
        update_chat_completion(response, chunk)
    if response is None or not done:
        raise ValueError("Incomplete Chat Completions stream")
    return response


def _completion_response(body: bytes, *, stream: bool) -> Completion:
    if not stream:
        return Completion.model_validate_json(body)
    chunks: list[dict[str, Any]] = []
    done = False
    for _, payload in _sse_events(body):
        if payload == "[DONE]":
            done = True
        elif isinstance(payload, dict):
            chunks.append(payload)
    if not chunks or not done:
        raise ValueError("Incomplete Completions stream")
    data = dict(chunks[0])
    choices: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        if isinstance(chunk.get("usage"), dict):
            data["usage"] = chunk["usage"]
        for raw in chunk.get("choices") or []:
            if not isinstance(raw, dict):
                continue
            index = raw.get("index")
            if not isinstance(index, int):
                continue
            current = choices.setdefault(
                index,
                {"index": index, "text": "", "finish_reason": "stop"},
            )
            current["text"] += raw.get("text") or ""
            if raw.get("finish_reason") is not None:
                current["finish_reason"] = raw["finish_reason"]
            for key in ("token_ids", "tokens", "token_logprobs", "text_offset"):
                values = raw.get(key)
                if isinstance(values, list):
                    current.setdefault(key, []).extend(values)
            logprobs = raw.get("logprobs")
            if isinstance(logprobs, dict):
                target = current.setdefault("logprobs", {})
                for key, values in logprobs.items():
                    if isinstance(values, list):
                        target.setdefault(key, []).extend(values)
                    elif values is not None:
                        target[key] = values
            for key, value in raw.items():
                if key not in {
                    "finish_reason",
                    "index",
                    "logprobs",
                    "text",
                    "text_offset",
                    "token_ids",
                    "token_logprobs",
                    "tokens",
                }:
                    current[key] = value
    data["object"] = "text_completion"
    data["choices"] = [choices[index] for index in sorted(choices)]
    return Completion.model_validate(data)


def _responses_response(body: bytes, *, stream: bool) -> Response:
    if not stream:
        return Response.model_validate_json(body)
    completed: dict[str, Any] | None = None
    for event_name, payload in _sse_events(body):
        if isinstance(payload, dict) and (
            event_name == "response.completed"
            or payload.get("type") == "response.completed"
        ):
            value = payload.get("response")
            if isinstance(value, dict):
                completed = value
    if completed is None:
        raise ValueError("Incomplete Responses stream")
    return Response.model_validate(completed)


def _messages_response(body: bytes, *, stream: bool) -> Message:
    if not stream:
        return Message.model_validate_json(body)
    adapter = TypeAdapter(RawMessageStreamEvent)
    snapshot: ParsedMessage[object] | None = None
    complete = False
    token_ids: list[int] = []
    logprobs: list[float] = []
    for event_name, payload in _sse_events(body):
        if not isinstance(payload, dict):
            continue
        if event_name and "type" not in payload:
            payload = {**payload, "type": event_name}
        event = adapter.validate_python(payload)
        snapshot = accumulate_event(
            event=event,
            current_snapshot=snapshot,
            output_format=NOT_GIVEN,
        )
        if event.type == "message_delta":
            event_token_ids = payload.get("token_ids")
            event_logprobs = payload.get("logprobs")
            if isinstance(event_token_ids, list) and all(
                isinstance(value, int) for value in event_token_ids
            ):
                token_ids = event_token_ids
            if isinstance(event_logprobs, list) and all(
                isinstance(value, (int, float)) for value in event_logprobs
            ):
                logprobs = [float(value) for value in event_logprobs]
        complete = complete or event.type == "message_stop"
    if snapshot is None or not complete:
        raise ValueError("Incomplete Messages stream")
    data = snapshot.model_dump(mode="python")
    if token_ids:
        data["token_ids"] = token_ids
    if logprobs:
        data["logprobs"] = logprobs
    return Message.model_validate(data)


def _model(request: Mapping[str, object], response: ResponseModel) -> str | None:
    requested = request.get("model")
    if isinstance(requested, str):
        return requested
    return response.model if isinstance(response.model, str) else None


def build_exchange(
    endpoint: Endpoint,
    request: dict[str, Any],
    body: bytes,
    *,
    start_time: datetime,
    end_time: datetime,
) -> tuple[Endpoint, Exchange]:
    stream = request.get("stream") is True
    if endpoint == "chat_completions":
        response = _chat_response(body, stream=stream)
        return endpoint, ChatCompletionsExchange(
            request=ChatCompletionsRequest(request),
            response=response,
            model=_model(request, response),
            start_time=start_time,
            end_time=end_time,
        )
    if endpoint == "completions":
        response = _completion_response(body, stream=stream)
        return endpoint, CompletionsExchange(
            request=CompletionsRequest(request),
            response=response,
            model=_model(request, response),
            start_time=start_time,
            end_time=end_time,
        )
    if endpoint == "responses":
        response = _responses_response(body, stream=stream)
        return endpoint, ResponsesExchange(
            request=ResponsesRequest(request),
            response=response,
            model=_model(request, response),
            start_time=start_time,
            end_time=end_time,
        )
    if endpoint == "messages":
        response = _messages_response(body, stream=stream)
        return endpoint, MessagesExchange(
            request=MessagesRequest(request),
            response=response,
            model=_model(request, response),
            start_time=start_time,
            end_time=end_time,
        )
    raise ValueError(f"Unsupported trajectory endpoint: {endpoint}")
