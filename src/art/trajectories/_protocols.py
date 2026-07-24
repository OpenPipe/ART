from __future__ import annotations

from datetime import datetime
import json
import math
from typing import Any, Literal
from urllib.parse import urlsplit

from anthropic._types import NOT_GIVEN
from anthropic.lib.streaming._messages import accumulate_event
from anthropic.types import Message, ParsedMessage, RawMessageStreamEvent
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.responses import Response
from pydantic import TypeAdapter, ValidationError

from ..openai import init_chat_completion, update_chat_completion
from ..preprocessing.moe_routing import attach_moe_routing_metadata_to_choice
from ..vllm_route_transport import (
    decode_routed_experts_response,
    is_routed_experts_response,
)
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
SSEPayload = dict[str, Any] | Literal["[DONE]"]
_ENDPOINTS: dict[str, Endpoint] = {
    "/chat/completions": "chat_completions",
    "/completions": "completions",
    "/responses": "responses",
    "/messages": "messages",
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
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                if raw.strip().lower() in {"ping", "keepalive"}:
                    continue
                raise
            if isinstance(value, dict):
                events.append((event_name, value))
    return events


def _chat_response(body: bytes, *, stream: bool) -> ChatCompletion:
    if not stream:
        if is_routed_experts_response(body):
            response, routes = decode_routed_experts_response(body)
            payload = response.model_dump(mode="python")
            for position, choice in enumerate(response.choices):
                attach_moe_routing_metadata_to_choice(
                    choice=choice,
                    response_payload=payload,
                    choice_index=position,
                    routed_experts=routes.get(int(choice.index)),
                )
            return response
        return ChatCompletion.model_validate_json(body)
    response: ChatCompletion | None = None
    choices: dict[int, ChatCompletion] = {}
    done = False
    for _, payload in _sse_events(body):
        if payload == "[DONE]":
            done = True
            continue
        try:
            chunk = ChatCompletionChunk.model_validate(payload)
        except ValidationError:
            if (
                set(payload)
                <= {
                    "choices",
                    "created",
                    "id",
                    "model",
                    "object",
                    "prompt_filter_results",
                    "system_fingerprint",
                }
                and payload.get("object") == ""
                and payload.get("id") == ""
                and payload.get("model") == ""
                and payload.get("choices") == []
            ):
                continue
            raise
        if response is None:
            response = init_chat_completion(chunk.model_copy(update={"choices": []}))
        update_chat_completion(response, chunk.model_copy(update={"choices": []}))
        for choice in chunk.choices:
            choice_chunk = chunk.model_copy(update={"choices": [choice]})
            choice_response = choices.get(choice.index)
            if choice_response is None:
                choice_response = init_chat_completion(choice_chunk)
                choices[choice.index] = choice_response
            update_chat_completion(choice_response, choice_chunk)
    if response is None or not done:
        raise ValueError("Incomplete Chat Completions stream")
    response.choices = [choices[index].choices[0] for index in sorted(choices)]
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
    prompt_token_ids: list[int] = []
    block_token_ids: dict[int, list[int]] = {}
    block_logprobs: dict[int, list[float]] = {}

    def parsed_token_ids(value: object, field: str) -> list[int] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(f"{field} must contain integer token IDs")
        result: list[int] = []
        for item in value:
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise ValueError(f"{field} must contain integer token IDs")
            result.append(item)
        return result

    def token_logprobs(value: object, field: str) -> list[float] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(f"{field} must contain numeric logprobs")
        result: list[float] = []
        for item in value:
            if (
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(item)
            ):
                raise ValueError(f"{field} must contain numeric logprobs")
            result.append(float(item))
        return result

    for event_name, payload in _sse_events(body):
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type") or event_name
        if event_name == "ping" or event_type == "ping":
            continue
        if event_name == "error" or event_type == "error":
            raise ValueError("Anthropic Messages stream returned an error event")
        if event_name and "type" not in payload:
            payload = {**payload, "type": event_name}
        event = adapter.validate_python(payload)
        snapshot = accumulate_event(
            event=event,
            current_snapshot=snapshot,
            output_format=NOT_GIVEN,
        )
        if event.type == "message_start":
            message = payload.get("message")
            values = (
                message.get("prompt_token_ids") if isinstance(message, dict) else None
            )
            if (
                values := parsed_token_ids(values, "Messages prompt_token_ids")
            ) is not None:
                prompt_token_ids = values
        elif event.type == "message_delta":
            if (
                values := parsed_token_ids(
                    payload.get("prompt_token_ids"), "Messages prompt_token_ids"
                )
            ) is not None:
                prompt_token_ids = values
            if (
                event_token_ids := parsed_token_ids(
                    payload.get("token_ids"), "Messages token_ids"
                )
            ) is not None:
                token_ids = event_token_ids
            if (
                event_logprobs := token_logprobs(
                    payload.get("logprobs"), "Messages logprobs"
                )
            ) is not None:
                logprobs = event_logprobs
        elif event.type in {"content_block_start", "content_block_delta"}:
            index = payload.get("index")
            event_token_ids = parsed_token_ids(
                payload.get("token_ids"), "Messages content token_ids"
            )
            event_logprobs = token_logprobs(
                payload.get("logprobs"), "Messages content logprobs"
            )
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and event_token_ids
            ):
                block_token_ids.setdefault(index, []).extend(event_token_ids)
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and event_logprobs
            ):
                block_logprobs.setdefault(index, []).extend(event_logprobs)
        complete = complete or event.type == "message_stop"
    if snapshot is None or not complete:
        raise ValueError("Incomplete Messages stream")
    data = snapshot.model_dump(mode="python")
    content = data.get("content")
    if isinstance(content, list):
        rebuilt_content: list[object] = []
        for index, raw_block in enumerate(content):
            if not isinstance(raw_block, dict):
                rebuilt_content.append(raw_block)
                continue
            block: dict[str, Any] = {
                str(key): value for key, value in raw_block.items()
            }
            if values := block_token_ids.get(index):
                block["token_ids"] = values
            if values := block_logprobs.get(index):
                block["logprobs"] = values
            rebuilt_content.append(block)
        data["content"] = rebuilt_content
    if token_ids:
        data["token_ids"] = token_ids
    if logprobs:
        data["logprobs"] = logprobs
    if prompt_token_ids:
        data["prompt_token_ids"] = prompt_token_ids
    return Message.model_validate(data)


def build_exchange(
    endpoint: Endpoint,
    request: dict[str, Any],
    body: bytes,
    *,
    start_time: datetime,
    end_time: datetime,
) -> Exchange:
    stream = request.get("stream") is True
    if endpoint == "chat_completions":
        response = _chat_response(body, stream=stream)
        return ChatCompletionsExchange(
            request=ChatCompletionsRequest(**request),
            response=response,
            start_time=start_time,
            end_time=end_time,
        )
    if endpoint == "completions":
        response = _completion_response(body, stream=stream)
        return CompletionsExchange(
            request=CompletionsRequest(**request),
            response=response,
            start_time=start_time,
            end_time=end_time,
        )
    if endpoint == "responses":
        response = _responses_response(body, stream=stream)
        return ResponsesExchange(
            request=ResponsesRequest(**request),
            response=response,
            start_time=start_time,
            end_time=end_time,
        )
    if endpoint == "messages":
        response = _messages_response(body, stream=stream)
        return MessagesExchange(
            request=MessagesRequest(**request),
            response=response,
            start_time=start_time,
            end_time=end_time,
        )
    raise ValueError(f"Unsupported trajectory endpoint: {endpoint}")
