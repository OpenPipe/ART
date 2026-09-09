from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import cast

import httpx
from openai import AsyncOpenAI, AsyncStream, Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
import pytest

from art.openai import (
    consume_chat_completion_stream,
    consume_sync_chat_completion_stream,
)
from art.trajectories._protocols import _chat_response


class _Stream:
    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def close(self) -> None:
        self.closed = True


class _SyncStream:
    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = chunks
        self.closed = False

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


class _HangingStream(_Stream):
    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        super().__init__(chunks)
        self.waiting = asyncio.Event()

    async def __anext__(self) -> ChatCompletionChunk:
        try:
            return next(self._chunks)
        except StopIteration:
            self.waiting.set()
            await asyncio.Future()
            raise AssertionError("unreachable")


def _chunk(**values: object) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "completion",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test-model",
            "choices": [],
            **values,
        }
    )


def _chunks() -> list[ChatCompletionChunk]:
    first_span = {
        "start_token": 0,
        "end_token": 1,
        "policy_version": 1,
        "lora_slot": "slot",
        "update_seq": 2,
    }
    second_span = {**first_span, "start_token": 1, "end_token": 2}
    return [
        ChatCompletionChunk.model_construct(
            id="", object="", created=0, model="", choices=[]
        ),
        _chunk(
            service_tier="default",
            system_fingerprint="fingerprint",
            choices=[
                {
                    "index": 1,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "think-b ",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-",
                                "type": "function",
                                "function": {
                                    "name": "look",
                                    "arguments": '{"x":',
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                    "token_ids": [21],
                    "policy_token_spans": [first_span],
                    "logprobs": {
                        "content": [
                            {
                                "token": "token_id:21",
                                "logprob": -0.21,
                                "bytes": [98],
                                "top_logprobs": [],
                            }
                        ]
                    },
                }
            ],
        ),
        _chunk(
            choices=[
                {
                    "index": 1,
                    "delta": {
                        "reasoning_content": "more",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "b",
                                "function": {"name": "up", "arguments": "1}"},
                            }
                        ],
                    },
                    "finish_reason": None,
                    "token_ids": [22],
                    "policy_token_spans": [second_span],
                    "logprobs": {
                        "content": [
                            {
                                "token": "token_id:22",
                                "logprob": -0.22,
                                "bytes": [99],
                                "top_logprobs": [],
                            }
                        ]
                    },
                },
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning": "think-a",
                        "content": "answer-a",
                    },
                    "finish_reason": None,
                    "token_ids": [11],
                    "policy_token_spans": [first_span],
                    "logprobs": {
                        "content": [
                            {
                                "token": "token_id:11",
                                "logprob": -0.11,
                                "bytes": [97],
                                "top_logprobs": [],
                            }
                        ]
                    },
                },
            ]
        ),
        _chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {"content": "!"},
                    "finish_reason": "stop",
                    "token_ids": [12],
                    "policy_token_spans": [second_span],
                    "logprobs": {
                        "content": [
                            {
                                "token": "token_id:12",
                                "logprob": -0.12,
                                "bytes": [33],
                                "top_logprobs": [],
                            }
                        ]
                    },
                },
                {
                    "index": 1,
                    "delta": {},
                    "finish_reason": "tool_calls",
                },
            ]
        ),
        _chunk(
            prompt_token_ids=[1, 2],
            usage={
                "prompt_tokens": 2,
                "completion_tokens": 4,
                "total_tokens": 6,
            },
        ),
    ]


def _sse(chunks: list[ChatCompletionChunk]) -> bytes:
    frames = [b": caladan-progress\n\n"]
    for chunk in chunks:
        if chunk.object == "":
            frames.append(
                b'data: {"id":"","object":"","created":0,"model":"","choices":[]}\n\n'
            )
            continue
        frames.extend(
            (
                b"data: " + chunk.model_dump_json(exclude_none=True).encode() + b"\n\n",
                b": caladan-progress\n\n",
            )
        )
    frames.append(b"data: [DONE]\n\n")
    return b"".join(frames)


def _consume(chunks: list[ChatCompletionChunk], *, require_usage: bool = False):
    return asyncio.run(
        consume_chat_completion_stream(
            cast(AsyncStream[ChatCompletionChunk], _Stream(chunks)),
            require_usage=require_usage,
        )
    )


def test_stream_consumer_matches_auto_capture_reconstruction() -> None:
    chunks = _chunks()

    consumed = _consume(chunks)
    captured = _chat_response(_sse(chunks), stream=True)

    assert consumed.model_dump(mode="json") == captured.model_dump(mode="json")
    assert [choice.index for choice in consumed.choices] == [0, 1]
    first, second = consumed.choices
    assert first.message.content == "answer-a!"
    assert first.message.model_extra == {"reasoning": "think-a"}
    assert second.message.model_extra == {"reasoning_content": "think-b more"}
    assert second.message.tool_calls is not None
    tool_call = second.message.tool_calls[0]
    assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall)
    assert tool_call.function.arguments == '{"x":1}'
    assert first.model_extra == {
        "prompt_token_ids": [1, 2],
        "token_ids": [11, 12],
        "policy_token_spans": [
            {
                "start_token": 0,
                "end_token": 1,
                "policy_version": 1,
                "lora_slot": "slot",
                "update_seq": 2,
            },
            {
                "start_token": 1,
                "end_token": 2,
                "policy_version": 1,
                "lora_slot": "slot",
                "update_seq": 2,
            },
        ],
    }
    assert second.model_extra is not None
    assert second.model_extra["token_ids"] == [21, 22]
    assert first.logprobs is not None
    assert second.logprobs is not None
    assert len(first.logprobs.content or []) == 2
    assert len(second.logprobs.content or []) == 2
    assert consumed.usage is not None
    assert consumed.usage.total_tokens == 6
    assert consumed.service_tier == "default"
    assert consumed.system_fingerprint == "fingerprint"


def test_sync_stream_consumer_matches_async_consumer() -> None:
    chunks = _chunks()

    asynchronous = _consume(chunks)
    synchronous = consume_sync_chat_completion_stream(
        cast(Stream[ChatCompletionChunk], _SyncStream(chunks))
    )

    assert synchronous.model_dump(mode="json") == asynchronous.model_dump(mode="json")


@pytest.mark.asyncio
async def test_openai_sdk_ignores_sse_comments_and_empty_prologue() -> None:
    chunks = _chunks()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(chunks),
            headers={"Content-Type": "text/event-stream"},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="test",
        base_url="http://model.test/v1",
        http_client=http_client,
    )
    try:
        stream = await client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
        )
        consumed = await consume_chat_completion_stream(stream)
    finally:
        await client.close()
        await http_client.aclose()

    captured = _chat_response(_sse(chunks), stream=True)
    assert consumed.model_dump(mode="json") == captured.model_dump(mode="json")


def test_stream_consumer_preserves_explicit_empty_text_fields() -> None:
    chunks = [
        _chunk(
            choices=[
                {
                    "index": 0,
                    "delta": {"content": "", "refusal": ""},
                    "finish_reason": "stop",
                }
            ]
        )
    ]

    consumed = _consume(chunks)

    assert consumed.choices[0].message.content == ""
    assert consumed.choices[0].message.refusal == ""


def test_stream_consumer_rejects_premature_choice_termination() -> None:
    chunks = _chunks()[:2]

    with pytest.raises(ValueError, match="ended before choices"):
        _consume(chunks)


def test_stream_consumer_can_require_usage_trailer() -> None:
    chunks = _chunks()[:-1]

    assert _consume(chunks).usage is None
    with pytest.raises(ValueError, match="usage trailer"):
        _consume(chunks, require_usage=True)


def test_auto_capture_rejects_done_with_nonterminal_choice() -> None:
    with pytest.raises(ValueError, match="were not terminal"):
        _chat_response(_sse(_chunks()[:2]), stream=True)


def _negative_tool_call_chunk() -> ChatCompletionChunk:
    return _chunk(
        choices=[
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": -1,
                            "id": "call",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ]
    )


@pytest.mark.asyncio
async def test_stream_consumer_closes_on_reconstruction_error() -> None:
    stream = _Stream([_negative_tool_call_chunk()])

    with pytest.raises(ValueError, match="must be non-negative"):
        await consume_chat_completion_stream(
            cast(AsyncStream[ChatCompletionChunk], stream)
        )

    assert stream.closed


def test_sync_stream_consumer_closes_on_reconstruction_error() -> None:
    stream = _SyncStream([_negative_tool_call_chunk()])

    with pytest.raises(ValueError, match="must be non-negative"):
        consume_sync_chat_completion_stream(cast(Stream[ChatCompletionChunk], stream))

    assert stream.closed


@pytest.mark.asyncio
async def test_stream_consumer_closes_on_cancellation() -> None:
    stream = _HangingStream(
        [_chunk(choices=[{"index": 0, "delta": {"content": "partial"}}])]
    )
    task = asyncio.create_task(
        consume_chat_completion_stream(cast(AsyncStream[ChatCompletionChunk], stream))
    )
    await stream.waiting.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed
