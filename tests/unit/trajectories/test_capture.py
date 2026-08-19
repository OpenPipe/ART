from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Coroutine,
    Generator,
    Iterable,
)
import copy
from datetime import datetime, timedelta
import gzip
import json
from typing import Any, cast
from unittest.mock import Mock
import zlib

import aiohttp
from aiohttp import web
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
import httpx
from openai import AsyncOpenAI, OpenAI
import pytest
import pytest_asyncio
import requests

import art
from art.gather import GatherContext, record_metrics
import art.trajectories as tr
from art.trajectories import (
    ChatCompletionsExchange,
    CompletionsExchange,
    MessagesExchange,
    ResponsesExchange,
    _compat,
)
from art.trajectories._capture.core import _append_exchange, begin, reset
from art.trajectories._protocols import Endpoint, build_exchange, endpoint_for_url


def test_root_trajectory_exports_are_minimal() -> None:
    expected = {
        "Trajectory",
        "TrajectoryGroup",
        "trajectory",
        "trajectory_group",
        "current_trajectory",
        "no_capture",
    }

    assert set(art.__all__) & set(art.trajectories.__all__) == expected
    assert all(hasattr(art, name) for name in expected)
    assert all(
        not hasattr(art, name) for name in set(art.trajectories.__all__) - expected
    )


CHAT: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "test/model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hello"},
            "logprobs": {
                "content": [
                    {
                        "token": "token_id:2",
                        "logprob": -0.2,
                        "bytes": [104],
                        "top_logprobs": [],
                    }
                ]
            },
            "token_ids": [2],
            "prompt_token_ids": [1],
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
}

COMPLETION: dict[str, Any] = {
    "id": "cmpl-1",
    "object": "text_completion",
    "created": 1,
    "model": "test/model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "text": "hello",
            "token_ids": [2],
            "prompt_token_ids": [1],
            "logprobs": {
                "tokens": ["token_id:2"],
                "token_logprobs": [-0.2],
                "top_logprobs": [{}],
                "text_offset": [0],
            },
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
}

RESPONSE: dict[str, Any] = {
    "id": "resp_1",
    "created_at": 1.0,
    "model": "test/model",
    "object": "response",
    "output": [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "hello",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 1,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 5,
    },
    "token_generations": [
        {
            "prompt_token_ids": [1],
            "output_tokens": [{"token_id": 2, "logprob": -0.2, "text": "hello"}],
            "output_indices": [0],
        }
    ],
}

MESSAGE: dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "test/model",
    "content": [{"type": "text", "text": "hello", "citations": None}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "prompt_token_ids": [1],
    "token_ids": [2],
    "logprobs": [-0.2],
}


class _SyncChunks(httpx.SyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __iter__(self) -> Generator[bytes, None, None]:
        for index in range(0, len(self.body), 3):
            yield self.body[index : index + 3]


class _AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index in range(0, len(self.body), 3):
            yield self.body[index : index + 3]


def _encoded(body: bytes, encoding: str) -> bytes:
    if encoding == "gzip":
        return gzip.compress(body)
    if encoding == "deflate":
        return zlib.compress(body)
    brotli = pytest.importorskip("brotli")
    return cast(bytes, brotli.compress(body))


@pytest_asyncio.fixture
async def endpoint_server(unused_tcp_port: int) -> AsyncIterator[str]:
    async def handler(request: web.Request) -> web.StreamResponse:
        request_body = await request.json()
        if request_body.get("fail"):
            return web.json_response({"error": "failed"}, status=400)
        if request_body.get("incomplete"):
            return web.Response(
                body=_sse([(None, {"type": "incomplete"})]),
                content_type="text/event-stream",
            )
        if request_body.get("early_close"):
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "application/json"}
            )
            await response.prepare(request)
            await response.write(json.dumps(CHAT).encode())
            await asyncio.sleep(0.05)
            try:
                await response.write(b" ")
            except ConnectionResetError:
                pass
            return response
        bodies = {
            "/v1/chat/completions": CHAT,
            "/v1/completions": COMPLETION,
            "/v1/responses": RESPONSE,
            "/v1/messages": MESSAGE,
        }
        return web.json_response(bodies[request.path])

    app = web.Application()
    app.router.add_post("/v1/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    yield f"http://127.0.0.1:{unused_tcp_port}/v1"
    await runner.cleanup()


async def test_contexts_are_nested_and_task_local() -> None:
    assert art.current_trajectory() is None
    with art.Trajectory() as outer:
        assert art.current_trajectory(require=True) is outer
        with art.Trajectory() as inner:
            assert art.current_trajectory() is inner
        assert art.current_trajectory() is outer

        async def child() -> art.Trajectory:
            with art.Trajectory() as item:
                await asyncio.sleep(0)
                assert art.current_trajectory() is item
            return item

        first, second = await asyncio.gather(child(), child())
        assert first is not second
    assert art.current_trajectory() is None
    with pytest.raises(RuntimeError, match="No trajectory"):
        art.current_trajectory(require=True)


def test_no_capture_hides_enclosing_trajectory_but_allows_nested_capture() -> None:
    def capture_exchange() -> None:
        state, token = begin(
            "POST",
            "https://example.test/v1/chat/completions",
            {"model": "test/model", "messages": []},
        )
        reset(token)
        if state is not None:
            state.status_code = 200
            state.add(json.dumps(CHAT).encode())
            state.finish()

    with art.Trajectory() as outer:
        capture_exchange()
        with art.no_capture():
            assert art.current_trajectory() is None
            with pytest.raises(RuntimeError, match="No trajectory"):
                art.current_trajectory(require=True)
            capture_exchange()
            with art.Trajectory() as inner:
                capture_exchange()
                with art.no_capture():
                    assert art.current_trajectory() is None
                    capture_exchange()
                    with art.Trajectory() as nested:
                        capture_exchange()
                    assert art.current_trajectory() is None
                assert art.current_trajectory() is inner
                capture_exchange()
            assert art.current_trajectory() is None
        assert art.current_trajectory() is outer
        capture_exchange()

        with pytest.raises(ValueError, match="restore"):
            with art.no_capture():
                raise ValueError("restore")
        assert art.current_trajectory() is outer

    assert len(outer.exchanges.chat_completions) == 2
    assert len(inner.exchanges.chat_completions) == 2
    assert len(nested.exchanges.chat_completions) == 1


def test_no_capture_uses_the_scope_active_when_a_request_begins() -> None:
    with art.Trajectory() as trajectory:
        state, token = begin(
            "POST",
            "https://example.test/v1/chat/completions",
            {"model": "test/model", "messages": []},
        )
        reset(token)
        assert state is not None

        with art.no_capture():
            state.status_code = 200
            state.add(json.dumps(CHAT).encode())
            state.finish()
            ignored, ignored_token = begin(
                "POST",
                "https://example.test/v1/chat/completions",
                {"model": "test/model", "messages": []},
            )
            reset(ignored_token)
            assert ignored is None

    assert len(trajectory.exchanges.chat_completions) == 1


async def test_no_capture_context_is_copied_when_tasks_are_created() -> None:
    async def current_after_scheduling() -> art.Trajectory | None:
        await asyncio.sleep(0)
        return art.current_trajectory()

    with art.Trajectory() as outer:
        inherited = asyncio.create_task(current_after_scheduling())
        with art.no_capture():
            detached = asyncio.create_task(current_after_scheduling())
            assert await current_after_scheduling() is None
        assert await inherited is outer
        assert await detached is None


async def test_async_helpers_and_group_aggregation_are_isolated() -> None:
    def capture_exchange() -> None:
        state, token = begin(
            "POST",
            "https://example.test/v1/chat/completions",
            {"model": "test/model", "messages": []},
        )
        reset(token)
        if state is not None:
            state.status_code = 200
            state.add(json.dumps(CHAT).encode())
            state.finish()

    async def rollout() -> None:
        await asyncio.sleep(0)
        capture_exchange()

    captured = await art.trajectory(rollout())
    assert isinstance(captured, art.Trajectory)
    assert len(captured.exchanges.chat_completions) == 1
    task = asyncio.create_task(rollout())
    with pytest.raises(TypeError, match="raw coroutine"):
        # Passing a Task is deliberately a static type error and a runtime error.
        await art.trajectory(task)  # ty: ignore[invalid-argument-type]
    await task

    async def unscoped() -> art.Trajectory:
        assert art.current_trajectory() is None
        capture_exchange()
        return art.Trajectory()

    async def failed() -> art.Trajectory:
        raise ValueError("boom")

    def generated() -> Generator[Coroutine[Any, Any, art.Trajectory], None, None]:
        capture_exchange()
        yield art.trajectory(rollout())

    with art.Trajectory() as outer:
        successful = art.trajectory(rollout())
        result = await art.trajectory_group(
            [successful, unscoped(), failed()],
            return_exceptions=True,
        )
        generated_result = await art.trajectory_group(generated())
        with pytest.raises(ValueError, match="boom"):
            await art.trajectory_group([failed()])
        assert art.current_trajectory() is outer

    assert len(result.trajectories) == 2
    assert len(result.trajectories[0].exchanges.chat_completions) == 1
    assert not result.trajectories[1].exchanges
    assert result.exceptions[0].message == "boom"
    assert len(generated_result.trajectories[0].exchanges.chat_completions) == 1
    assert not outer.exchanges

    calls = 0

    async def once() -> art.Trajectory:
        nonlocal calls
        calls += 1
        return art.Trajectory(reward=1)

    shared = once()
    resolved = art.Trajectory(reward=2)
    mixed = await art.trajectory_group(
        [resolved, shared, shared, ValueError("recorded")],
        exceptions=[RuntimeError("provided")],
        metadata={"source": "test"},
        metrics={"score": 3},
        logs=["log"],
    )
    assert calls == 1
    assert len(mixed.trajectories) == 3
    assert mixed.trajectories[0] is resolved
    assert mixed.trajectories[1] is mixed.trajectories[2]
    assert [error.message for error in mixed.exceptions] == [
        "recorded",
        "provided",
    ]
    assert mixed.metadata == {"source": "test"}
    assert mixed.metrics == {"score": 3}
    assert mixed.logs == ["log"]

    sibling_started = asyncio.Event()
    sibling_stopped = asyncio.Event()

    async def sibling() -> art.Trajectory:
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_stopped.set()
        raise AssertionError("unreachable")

    async def cancelled() -> art.Trajectory:
        await sibling_started.wait()
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await art.trajectory_group([sibling(), cancelled()])
    assert sibling_stopped.is_set()

    sibling_started.clear()
    sibling_stopped.clear()
    scheduled = sibling()

    def failing_iterable() -> Iterable[Coroutine[Any, Any, art.Trajectory]]:
        yield scheduled
        raise ValueError("iteration failed")

    with pytest.raises(ValueError, match="iteration failed"):
        await art.trajectory_group(failing_iterable())
    assert scheduled.cr_frame is None

    task = asyncio.create_task(once())
    scheduled = await art.trajectory_group([task, task])
    assert len(scheduled.trajectories) == 2
    assert scheduled.trajectories[0] is scheduled.trajectories[1]
    assert calls == 2

    future = asyncio.get_running_loop().create_future()
    future.set_result(resolved)
    from_future = await art.trajectory_group([future])
    assert from_future.trajectories == [resolved]


def test_sync_group_generator_initializes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initializer = Mock(wraps=_compat.init_trajectory_group)
    monkeypatch.setattr(_compat, "init_trajectory_group", initializer)

    trajectory = art.Trajectory(reward=1)
    group = art.TrajectoryGroup(item for item in [trajectory])

    assert group.trajectories == [trajectory]
    assert initializer.call_count == 1


def test_group_pydantic_round_trip_restores_trajectories_and_exceptions() -> None:
    group = art.TrajectoryGroup(
        [art.Trajectory(reward=1)],
        exceptions=[ValueError("boom")],
        metadata={"source": "test"},
    )
    payload = group.model_dump()

    restored = art.TrajectoryGroup.model_validate_json(group.model_dump_json())

    assert isinstance(restored, art.TrajectoryGroup)
    assert restored.model_dump() == payload
    assert art.TrajectoryGroup.model_validate(payload).model_dump() == payload
    assert art.TrajectoryGroup(**payload).model_dump() == payload


def test_group_copy_preserves_subclass() -> None:
    class Group(art.TrajectoryGroup):
        pass

    group = Group([art.Trajectory(reward=1)])

    assert type(copy.copy(group)) is Group
    assert type(copy.deepcopy(group)) is Group


async def test_httpx_requests_and_aiohttp_capture_once(endpoint_server: str) -> None:
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}

    def requests_stream() -> None:
        with requests.post(
            f"{endpoint_server}/chat/completions",
            json=body,
            stream=True,
            timeout=5,
        ) as response:
            list(response.iter_content(chunk_size=5, decode_unicode=True))

    with art.Trajectory() as trajectory:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{endpoint_server}/chat/completions", json=body
            )
            response.raise_for_status()

        await asyncio.to_thread(requests_stream)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint_server}/chat/completions", json=body
            ) as response:
                await response.json()

    assert len(trajectory.exchanges.chat_completions) == 3
    assert all(
        exchange.response.choices[0].message.content == "hello"
        for exchange in trajectory.exchanges.chat_completions
    )


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br"])
@pytest.mark.parametrize("mode", ["raw", "bytes", "lines"])
def test_httpx_sync_stream_consumption_captures_decoded_body_once(
    encoding: str, mode: str
) -> None:
    body = _streaming_chat_body()
    compressed = _encoded(body, encoding)

    def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": encoding},
            stream=_SyncChunks(compressed),
        )

    with art.Trajectory() as trajectory:
        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            with client.stream(
                "POST",
                "https://example.test/v1/chat/completions",
                json={"model": "test/model", "messages": [], "stream": True},
            ) as result:
                if mode == "raw":
                    assert b"".join(result.iter_raw()) == compressed
                elif mode == "bytes":
                    assert b"".join(result.iter_bytes()) == body
                else:
                    list(result.iter_lines())

    assert len(trajectory.exchanges.chat_completions) == 1


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br"])
@pytest.mark.parametrize("mode", ["raw", "bytes", "lines"])
async def test_httpx_async_stream_consumption_captures_decoded_body_once(
    encoding: str, mode: str
) -> None:
    body = _streaming_chat_body()
    compressed = _encoded(body, encoding)

    async def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": encoding},
            stream=_AsyncChunks(compressed),
        )

    with art.Trajectory() as trajectory:
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            async with client.stream(
                "POST",
                "https://example.test/v1/chat/completions",
                json={"model": "test/model", "messages": [], "stream": True},
            ) as result:
                if mode == "raw":
                    assert (
                        b"".join([chunk async for chunk in result.aiter_raw()])
                        == compressed
                    )
                elif mode == "bytes":
                    assert (
                        b"".join([chunk async for chunk in result.aiter_bytes()])
                        == body
                    )
                else:
                    _ = [line async for line in result.aiter_lines()]

    assert len(trajectory.exchanges.chat_completions) == 1


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br"])
async def test_httpx_terminal_event_captures_before_response_close(
    encoding: str,
) -> None:
    body = _streaming_chat_body()
    compressed = _encoded(body, encoding)

    async def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-encoding": encoding,
                "content-type": "text/event-stream",
            },
            stream=_AsyncChunks(compressed),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(response))
    request = client.build_request(
        "POST",
        "https://example.test/v1/chat/completions",
        json={"model": "test/model", "messages": [], "stream": True},
    )
    with art.Trajectory() as trajectory:
        result = await client.send(request, stream=True)
        iterator = cast(AsyncGenerator[bytes, None], result.aiter_bytes())
        received = bytearray()
        async for chunk in iterator:
            received.extend(chunk)
            if b"data: [DONE]\n\n" in received:
                break
        assert len(trajectory.exchanges.chat_completions) == 1

    assert not result.is_closed
    await iterator.aclose()
    await result.aclose()
    await client.aclose()


def test_httpx_raw_capture_failure_does_not_change_user_stream() -> None:
    malformed = b"not a gzip stream"

    def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_SyncChunks(malformed),
        )

    with art.Trajectory() as trajectory:
        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            with client.stream(
                "POST",
                "https://example.test/v1/chat/completions",
                json={"model": "test/model", "messages": [], "stream": True},
            ) as result:
                assert b"".join(result.iter_raw()) == malformed

    assert not trajectory.exchanges


async def test_httpx_async_raw_capture_failure_does_not_change_user_stream() -> None:
    malformed = b"not a gzip stream"

    async def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_AsyncChunks(malformed),
        )

    with art.Trajectory() as trajectory:
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            async with client.stream(
                "POST",
                "https://example.test/v1/chat/completions",
                json={"model": "test/model", "messages": [], "stream": True},
            ) as result:
                assert (
                    b"".join([chunk async for chunk in result.aiter_raw()]) == malformed
                )

    assert not trajectory.exchanges


def _streaming_chat_body() -> bytes:
    return _sse(
        [
            (
                None,
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test/model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            ),
            (None, "[DONE]"),
        ]
    )


def test_httpx_abandoned_compressed_raw_stream_is_excluded() -> None:
    compressed = gzip.compress(_streaming_chat_body())

    def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_SyncChunks(compressed),
        )

    with art.Trajectory() as trajectory:
        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            with client.stream(
                "POST",
                "https://example.test/v1/chat/completions",
                json={"model": "test/model", "messages": [], "stream": True},
            ) as result:
                iterator = cast(Generator[bytes, None, None], result.iter_raw())
                next(iterator)
                iterator.close()

    assert not trajectory.exchanges


async def test_httpx_abandoned_compressed_async_raw_stream_is_excluded() -> None:
    compressed = gzip.compress(_streaming_chat_body())

    async def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_AsyncChunks(compressed),
        )

    with art.Trajectory() as trajectory:
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            async with client.stream(
                "POST",
                "https://example.test/v1/chat/completions",
                json={"model": "test/model", "messages": [], "stream": True},
            ) as result:
                iterator = cast(AsyncGenerator[bytes, None], result.aiter_raw())
                await anext(iterator)
                await iterator.aclose()

    assert not trajectory.exchanges


async def test_aiohttp_capture_covers_stream_reader_consumption_methods(
    endpoint_server: str,
) -> None:
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}

    with art.Trajectory() as trajectory:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint_server}/chat/completions", json=body
            ) as response:
                await response.content.readexactly(1)
                await response.content.read()
            async with session.post(
                f"{endpoint_server}/chat/completions", json=body
            ) as response:
                await response.content.readuntil(b"}")
                await response.content.read()
            async with session.post(
                f"{endpoint_server}/chat/completions", json=body
            ) as response:
                async for _chunk, _end_of_http_chunk in response.content.iter_chunks():
                    pass

    assert len(trajectory.exchanges.chat_completions) == 3


async def test_requests_session_stream_default_is_preserved(
    endpoint_server: str,
) -> None:
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}

    def consume(trajectory: art.Trajectory) -> None:
        session = requests.Session()
        session.stream = True
        response = session.post(
            f"{endpoint_server}/chat/completions", json=body, timeout=5
        )
        assert not trajectory.exchanges
        list(response.iter_content())

    with art.Trajectory() as trajectory:
        await asyncio.to_thread(consume, trajectory)

    assert len(trajectory.exchanges.chat_completions) == 1


async def test_requests_decode_unicode_preserves_string_chunks(
    endpoint_server: str,
) -> None:
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}

    def consume() -> list[str | bytes]:
        with requests.post(
            f"{endpoint_server}/chat/completions",
            json=body,
            stream=True,
            timeout=5,
        ) as response:
            return list(response.iter_content(chunk_size=5, decode_unicode=True))

    with art.Trajectory() as trajectory:
        chunks = await asyncio.to_thread(consume)

    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert len(trajectory.exchanges.chat_completions) == 1


async def test_native_openai_and_anthropic_sdks(endpoint_server: str) -> None:
    openai = AsyncOpenAI(base_url=endpoint_server, api_key="test")
    anthropic = AsyncAnthropic(
        base_url=endpoint_server.removesuffix("/v1"), api_key="test"
    )
    with art.Trajectory() as trajectory:
        completion = await openai.completions.create(model="test/model", prompt="hi")
        response = await openai.responses.create(model="test/model", input="hi")
        message = await anthropic.messages.create(
            model="test/model",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
    await openai.close()
    await anthropic.close()

    assert completion.choices[0].text == "hello"
    assert response.output_text == "hello"
    assert message.content[0].type == "text"
    assert message.content[0].text == "hello"
    assert len(trajectory.exchanges.completions) == 1
    assert len(trajectory.exchanges.responses) == 1
    assert len(trajectory.exchanges.messages) == 1


async def test_native_openai_chat_stream_captures_at_done_event() -> None:
    async def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_AsyncChunks(_streaming_chat_body()),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(response))
    client = AsyncOpenAI(
        base_url="https://example.test/v1",
        api_key="test",
        http_client=http_client,
    )
    with art.Trajectory() as trajectory:
        stream = await client.chat.completions.create(
            model="test/model",
            messages=[],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]
        assert len(trajectory.exchanges.chat_completions) == 1

    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "hello"
    await stream.close()
    assert len(trajectory.exchanges.chat_completions) == 1
    await client.close()


def test_native_openai_preloaded_chat_stream_captures_once() -> None:
    def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_streaming_chat_body(),
        )

    http_client = httpx.Client(transport=httpx.MockTransport(response))
    client = OpenAI(
        base_url="https://example.test/v1",
        api_key="test",
        http_client=http_client,
    )
    with art.Trajectory() as trajectory:
        stream = client.chat.completions.create(
            model="test/model",
            messages=[],
            stream=True,
        )
        assert len(trajectory.exchanges.chat_completions) == 1
        chunks = list(stream)
        assert len(trajectory.exchanges.chat_completions) == 1

    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "hello"
    stream.close()
    assert len(trajectory.exchanges.chat_completions) == 1
    client.close()


async def test_native_async_openai_preloaded_chat_stream_captures_once() -> None:
    async def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_streaming_chat_body(),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(response))
    client = AsyncOpenAI(
        base_url="https://example.test/v1",
        api_key="test",
        http_client=http_client,
    )
    with art.Trajectory() as trajectory:
        stream = await client.chat.completions.create(
            model="test/model",
            messages=[],
            stream=True,
        )
        assert len(trajectory.exchanges.chat_completions) == 1
        chunks = [chunk async for chunk in stream]
        assert len(trajectory.exchanges.chat_completions) == 1

    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content == "hello"
    await stream.close()
    assert len(trajectory.exchanges.chat_completions) == 1
    await client.close()


def test_preloaded_malformed_stream_is_excluded() -> None:
    def response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse([(None, "corrupt"), (None, "[DONE]")]),
        )

    with art.Trajectory() as trajectory:
        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            with client.stream(
                "POST",
                "https://example.test/v1/chat/completions",
                json={"model": "test/model", "messages": [], "stream": True},
            ) as result:
                assert list(result.iter_bytes())

    assert not trajectory.exchanges


def test_stream_terminal_without_sse_boundary_is_excluded() -> None:
    body = _streaming_chat_body()[:-1]
    with art.Trajectory() as trajectory:
        state, token = begin(
            "POST",
            "https://example.test/v1/chat/completions",
            {"model": "test/model", "messages": [], "stream": True},
        )
        reset(token)
        assert state is not None
        state.status_code = 200
        state.add(body)
        assert not state.captured
        state.finish()

    assert not trajectory.exchanges


async def test_failed_and_incomplete_calls_are_excluded(endpoint_server: str) -> None:
    async with httpx.AsyncClient() as client:
        with art.Trajectory() as trajectory:
            await client.post(
                f"{endpoint_server}/chat/completions",
                json={"model": "test/model", "messages": [], "fail": True},
            )
            await client.post(
                f"{endpoint_server}/chat/completions",
                json={
                    "model": "test/model",
                    "messages": [],
                    "stream": True,
                    "incomplete": True,
                },
            )
    assert not trajectory.exchanges


async def test_abandoned_transport_streams_are_excluded(endpoint_server: str) -> None:
    body = {
        "model": "test/model",
        "messages": [],
        "early_close": True,
    }

    async def abandon_httpx() -> None:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", f"{endpoint_server}/chat/completions", json=body
            ) as response:
                iterator = cast(AsyncGenerator[bytes, None], response.aiter_bytes())
                await anext(iterator)
                await iterator.aclose()

    async def abandon_aiohttp() -> None:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint_server}/chat/completions", json=body
            ) as response:
                iterator = cast(
                    AsyncGenerator[bytes, None], response.content.iter_any()
                )
                await anext(iterator)
                await iterator.aclose()

    def abandon_requests() -> None:
        with requests.post(
            f"{endpoint_server}/chat/completions",
            json=body,
            stream=True,
            timeout=5,
        ) as response:
            iterator = cast(
                Generator[bytes, None, None],
                response.iter_content(chunk_size=None),
            )
            next(iterator)
            iterator.close()

    with art.Trajectory() as trajectory:
        await abandon_httpx()
        await abandon_aiohttp()
        await asyncio.to_thread(abandon_requests)

    assert not trajectory.exchanges


def test_capture_snapshots_nested_request_values() -> None:
    request = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "before"}],
    }
    with art.Trajectory() as trajectory:
        state, token = begin(
            "POST", "https://example.test/v1/chat/completions", request
        )
        reset(token)
        assert state is not None
        request["messages"][0]["content"] = "after"
        state.status_code = 200
        state.add(json.dumps(CHAT).encode())
        state.finish()

    assert trajectory.exchanges.chat_completions[0].request["messages"] == [
        {"role": "user", "content": "before"}
    ]


def test_all_protocols_reconstruct_typed_responses() -> None:
    now = datetime.now()
    values: list[tuple[Endpoint, dict[str, Any], dict[str, Any]]] = [
        ("chat_completions", {"model": "request-model", "messages": []}, CHAT),
        ("completions", {"model": "request-model", "prompt": "hi"}, COMPLETION),
        ("responses", {"input": "hi"}, RESPONSE),
        ("messages", {"model": "request-model", "messages": []}, MESSAGE),
    ]
    for endpoint, request, response in values:
        exchange = build_exchange(
            endpoint,
            request,
            json.dumps(response).encode(),
            start_time=now,
            end_time=now + timedelta(seconds=1),
        )
        assert exchange.end_time > exchange.start_time
        expected = request.get("model", "test/model")
        assert exchange.model == expected
        dumped = exchange.model_dump(mode="json")
        assert dumped["request"] == request
        assert dumped["model"] == expected

        exchange.request["model"] = "updated/model"
        assert exchange.model == "updated/model"
        assert exchange.model_dump(mode="json")["model"] == "updated/model"


def test_gather_metrics_sum_legacy_and_exchange_completion_tokens() -> None:
    now = datetime.now()
    chat = build_exchange(
        "chat_completions",
        {"model": "test/model", "messages": []},
        json.dumps(CHAT).encode(),
        start_time=now,
        end_time=now,
    )
    completion = build_exchange(
        "completions",
        {"model": "test/model", "prompt": "hi"},
        json.dumps(COMPLETION).encode(),
        start_time=now,
        end_time=now,
    )
    response = build_exchange(
        "responses",
        {"model": "test/model", "input": "hi"},
        json.dumps(RESPONSE).encode(),
        start_time=now,
        end_time=now,
    )
    message = build_exchange(
        "messages",
        {"model": "test/model", "messages": []},
        json.dumps(MESSAGE).encode(),
        start_time=now,
        end_time=now,
    )
    assert isinstance(chat, ChatCompletionsExchange)
    assert isinstance(completion, CompletionsExchange)
    assert isinstance(response, ResponsesExchange)
    assert isinstance(message, MessagesExchange)

    trajectory = art.Trajectory(
        exchanges=tr.TrajectoryExchanges(
            chat_completions=[chat],
            completions=[completion],
            responses=[response],
            messages=[message],
        )
    )
    record_metrics(GatherContext(), trajectory)
    assert trajectory.metrics["completion_tokens"] == 10

    first = chat.response.choices[0]
    second = first.model_copy(deep=True)
    assert second.logprobs is not None and second.logprobs.content is not None
    second.logprobs.content *= 2
    legacy = art.Trajectory(messages_and_choices=[first, second])
    record_metrics(GatherContext(), legacy)
    assert "completion_tokens" not in legacy.metrics


def test_gather_metrics_omit_partial_exchange_usage() -> None:
    now = datetime.now()
    chat = build_exchange(
        "chat_completions",
        {"model": "test/model", "messages": []},
        json.dumps(CHAT).encode(),
        start_time=now,
        end_time=now,
    )
    message = build_exchange(
        "messages",
        {"model": "test/model", "messages": []},
        json.dumps(MESSAGE).encode(),
        start_time=now,
        end_time=now,
    )
    assert isinstance(chat, ChatCompletionsExchange)
    assert isinstance(message, MessagesExchange)
    chat.response.usage = None
    trajectory = art.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[chat], messages=[message])
    )

    record_metrics(GatherContext(), trajectory)

    assert "completion_tokens" not in trajectory.metrics


@pytest.mark.parametrize(
    ("url", "endpoint"),
    [
        (
            "https://azure.test/openai/deployments/model/chat/completions?api-version=1",
            "chat_completions",
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "chat_completions",
        ),
        ("https://gateway.test/completions", "completions"),
        ("https://gateway.test/responses", "responses"),
        ("https://gateway.test/messages", "messages"),
        ("https://gateway.test/not-messages", None),
    ],
)
def test_endpoint_detection_accepts_compatible_gateway_paths(
    url: str, endpoint: Endpoint | None
) -> None:
    assert endpoint_for_url(url) == endpoint


def _sse(events: list[tuple[str | None, dict[str, Any] | str]]) -> bytes:
    return "".join(
        f"{f'event: {name}\n' if name else ''}data: "
        f"{value if isinstance(value, str) else json.dumps(value)}\n\n"
        for name, value in events
    ).encode()


def test_all_streaming_protocols_reconstruct_final_responses() -> None:
    now = datetime.now()
    chat_chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test/model",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
    }
    completion_chunk = {
        **COMPLETION,
        "object": "text_completion.chunk",
        "choices": [
            {
                **COMPLETION["choices"][0],
                "text": "hello",
                "finish_reason": None,
            }
        ],
    }
    response_event = {"type": "response.completed", "response": RESPONSE}
    message_events = [
        ("ping", {"type": "ping"}),
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **MESSAGE,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                    "prompt_token_ids": [1],
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "", "citations": None},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
                "token_ids": [2],
                "logprobs": [-0.2],
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
                "prompt_token_ids": [1],
                "token_ids": [2],
                "logprobs": [-0.2],
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    values: list[tuple[Endpoint, dict[str, Any], bytes]] = [
        (
            "chat_completions",
            {"model": "test/model", "messages": [], "stream": True},
            _sse([(None, chat_chunk), (None, "[DONE]")]),
        ),
        (
            "completions",
            {"model": "test/model", "prompt": "hi", "stream": True},
            _sse([(None, completion_chunk), (None, "[DONE]")]),
        ),
        (
            "responses",
            {"model": "test/model", "input": "hi", "stream": True},
            _sse([("response.completed", response_event)]),
        ),
        (
            "messages",
            {"model": "test/model", "messages": [], "stream": True},
            _sse(message_events),
        ),
    ]
    for endpoint, request, body in values:
        exchange = build_exchange(
            endpoint,
            request,
            body,
            start_time=now,
            end_time=now + timedelta(seconds=1),
        )
        assert exchange.model == "test/model"
        if isinstance(exchange, MessagesExchange):
            content = exchange.response.content[0]
            assert isinstance(content, TextBlock)
            assert content.text == "hello"
            assert content.model_extra is not None
            assert content.model_extra["token_ids"] == [2]
            assert content.model_extra["logprobs"] == [-0.2]
            assert getattr(exchange.response, "token_ids") == [2]
            assert getattr(exchange.response, "logprobs") == [-0.2]
            assert getattr(exchange.response, "prompt_token_ids") == [1]

        with art.Trajectory() as trajectory:
            state, token = begin(
                "POST",
                f"https://example.test/v1/{endpoint.replace('_', '/')}",
                request,
            )
            reset(token)
            assert state is not None
            state.status_code = 200
            for byte in body[:-1]:
                state.add(bytes([byte]))
            assert not state.captured
            state.add(body[-1:])
            assert state.captured

        assert (
            sum(
                len(exchanges)
                for exchanges in (
                    trajectory.exchanges.chat_completions,
                    trajectory.exchanges.completions,
                    trajectory.exchanges.responses,
                    trajectory.exchanges.messages,
                )
            )
            == 1
        )


def test_streaming_messages_error_event_is_rejected_and_not_captured() -> None:
    body = _sse(
        [
            (
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": "failed"},
                },
            )
        ]
    )
    request = {"model": "test/model", "messages": [], "stream": True}
    with pytest.raises(ValueError, match="returned an error event"):
        build_exchange(
            "messages",
            request,
            body,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    with art.Trajectory() as trajectory:
        state, token = begin("POST", "https://example.test/v1/messages", request)
        reset(token)
        assert state is not None
        state.status_code = 200
        state.add(body)
        state.finish()

    assert not trajectory.exchanges


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_token_ids", [True]),
        ("prompt_token_ids", [-1]),
        ("token_ids", [False]),
        ("token_ids", [-1]),
        ("logprobs", [True]),
        ("logprobs", [float("nan")]),
        ("logprobs", [float("inf")]),
    ],
)
def test_streaming_messages_reject_malformed_token_metadata(
    field: str, value: list[object]
) -> None:
    delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 1},
        "prompt_token_ids": [1],
        "token_ids": [2],
        "logprobs": [-0.2],
        field: value,
    }
    body = _sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        **MESSAGE,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": "", "citations": None},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", delta),
            ("message_stop", {"type": "message_stop"}),
        ]
    )

    with pytest.raises(ValueError, match="must contain"):
        build_exchange(
            "messages",
            {"model": "test/model", "messages": [], "stream": True},
            body,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )


def test_streaming_chat_choices_are_accumulated_by_index() -> None:
    now = datetime.now()

    def chunk(
        index: int, content: str, finish_reason: str | None = None
    ) -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test/model",
            "choices": [
                {
                    "index": index,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
        }

    exchange = build_exchange(
        "chat_completions",
        {"model": "test/model", "messages": [], "stream": True, "n": 2},
        _sse(
            [
                (None, chunk(1, "b")),
                (None, chunk(0, "a")),
                (None, chunk(1, "d", "stop")),
                (None, chunk(0, "c", "stop")),
                (None, "[DONE]"),
            ]
        ),
        start_time=now,
        end_time=now + timedelta(seconds=1),
    )

    assert isinstance(exchange, ChatCompletionsExchange)
    assert [choice.index for choice in exchange.response.choices] == [0, 1]
    assert [choice.message.content for choice in exchange.response.choices] == [
        "ac",
        "bd",
    ]


def test_streaming_chat_preserves_reasoning_fields() -> None:
    now = datetime.now()

    def chunk(finish_reason: str | None = None, **delta: str) -> dict[str, Any]:
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "test/model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", **delta},
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
        }

    exchange = build_exchange(
        "chat_completions",
        {"model": "test/model", "messages": [], "stream": True},
        _sse(
            [
                (None, chunk(reasoning="r1", reasoning_content="c1")),
                (
                    None,
                    chunk(
                        "stop",
                        reasoning="r2",
                        reasoning_content="c2",
                    ),
                ),
                (None, "[DONE]"),
            ]
        ),
        start_time=now,
        end_time=now + timedelta(seconds=1),
    )

    assert isinstance(exchange, ChatCompletionsExchange)
    message = exchange.response.choices[0].message
    assert getattr(message, "reasoning") == "r1r2"
    assert getattr(message, "reasoning_content") == "c1c2"


def test_streaming_chat_ignores_keepalives_and_azure_prologue() -> None:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test/model",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
    }
    body = _sse(
        [
            (None, "ping"),
            (None, {"id": "", "object": "", "created": 0, "model": "", "choices": []}),
            (None, chunk),
            (None, "[DONE]"),
        ]
    )

    exchange = build_exchange(
        "chat_completions",
        {"model": "test/model", "messages": [], "stream": True},
        body,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert isinstance(exchange, ChatCompletionsExchange)
    assert exchange.response.choices[0].message.content == "hello"

    with pytest.raises(json.JSONDecodeError):
        build_exchange(
            "chat_completions",
            {"model": "test/model", "messages": [], "stream": True},
            _sse([(None, "corrupt"), (None, chunk), (None, "[DONE]")]),
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    with pytest.raises(ValueError):
        build_exchange(
            "chat_completions",
            {"model": "test/model", "messages": [], "stream": True},
            _sse(
                [
                    (None, chunk),
                    (None, {"error": {"message": "failed"}}),
                    (None, "[DONE]"),
                ]
            ),
            start_time=datetime.now(),
            end_time=datetime.now(),
        )


def test_trajectory_rejects_mixed_representations() -> None:
    exchange = build_exchange(
        "chat_completions",
        {"model": "test/model", "messages": []},
        json.dumps(CHAT).encode(),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert isinstance(exchange, ChatCompletionsExchange)
    with pytest.raises(ValueError, match="both exchanges and legacy histories"):
        art.Trajectory(
            exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]),
            messages_and_choices=[{"role": "user", "content": "hi"}],
        )


def test_capture_does_not_mix_exchange_and_legacy_representations() -> None:
    exchange = build_exchange(
        "chat_completions",
        {"model": "test/model", "messages": []},
        json.dumps(CHAT).encode(),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert isinstance(exchange, ChatCompletionsExchange)
    trajectory = art.Trajectory(
        messages_and_choices=[{"role": "user", "content": "hi"}]
    )

    _append_exchange(trajectory, exchange)

    assert not trajectory.exchanges


def test_metadata_accepts_json_serializable_values() -> None:
    assert art.Trajectory().model_dump() == {}
    assert art.Trajectory().model_dump(
        mode="json", include={"reward"}, exclude_defaults=False
    ) == {"reward": 0.0}
    assert (
        art.Trajectory().model_dump_json(include={"reward"}, exclude_defaults=False)
        == '{"reward":0.0}'
    )
    trajectory = art.Trajectory(metadata={"nested": {"items": [1, "two"]}})
    assert trajectory.model_dump(mode="json")["metadata"] == {
        "nested": {"items": [1, "two"]}
    }
