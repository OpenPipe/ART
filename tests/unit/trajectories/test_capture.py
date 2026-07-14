from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json
from typing import Any

import aiohttp
from aiohttp import web
from anthropic import AsyncAnthropic
import httpx
from openai import AsyncOpenAI
import pytest
import pytest_asyncio
import requests

import art
from art.trajectories._protocols import build_exchange

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
    "raw_output_tokens": [{"token_id": 2, "logprob": -0.2}],
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
    "token_ids": [2],
    "logprobs": [-0.2],
}


@pytest_asyncio.fixture
async def endpoint_server(unused_tcp_port: int):
    async def handler(request: web.Request) -> web.Response:
        request_body = await request.json()
        if request_body.get("fail"):
            return web.json_response({"error": "failed"}, status=400)
        if request_body.get("incomplete"):
            return web.Response(
                body=_sse([(None, {"type": "incomplete"})]),
                content_type="text/event-stream",
            )
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
        assert art.current_trajectory(required=True) is outer
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
        art.current_trajectory(required=True)


async def test_group_context_and_async_helpers() -> None:
    with art.TrajectoryGroup() as group:
        with art.Trajectory() as first:
            pass
        with art.Trajectory() as second:
            pass
    assert group.trajectories == [first, second]

    async def rollout() -> None:
        await asyncio.sleep(0)

    captured = await art.trajectory(rollout())
    assert isinstance(captured, art.Trajectory)
    task = asyncio.create_task(rollout())
    with pytest.raises(TypeError, match="raw coroutine"):
        await art.trajectory(task)  # type: ignore[arg-type]
    await task

    async def failed() -> art.Trajectory:
        raise ValueError("boom")

    successful = art.trajectory(rollout())
    result = await art.trajectory_group([successful, failed()], return_exceptions=True)
    assert len(result.trajectories) == 1
    assert result.exceptions[0].message == "boom"


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
    assert message.content[0].text == "hello"
    assert len(trajectory.exchanges.completions) == 1
    assert len(trajectory.exchanges.responses) == 1
    assert len(trajectory.exchanges.messages) == 1


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


def test_all_protocols_reconstruct_typed_responses() -> None:
    now = datetime.now()
    values = [
        ("chat_completions", {"model": "request-model", "messages": []}, CHAT),
        ("completions", {"model": "request-model", "prompt": "hi"}, COMPLETION),
        ("responses", {"input": "hi"}, RESPONSE),
        ("messages", {"model": "request-model", "messages": []}, MESSAGE),
    ]
    for endpoint, request, response in values:
        name, exchange = build_exchange(
            endpoint,
            request,
            json.dumps(response).encode(),
            start_time=now,
            end_time=now + timedelta(seconds=1),
        )
        assert name == endpoint
        assert exchange.end_time > exchange.start_time
        expected = request.get("model", "test/model")
        assert exchange.model == expected
        assert exchange.model_dump(mode="json")["request"] == request


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
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
                "token_ids": [2],
                "logprobs": [-0.2],
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    values = [
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
        _, exchange = build_exchange(
            endpoint,
            request,
            body,
            start_time=now,
            end_time=now + timedelta(seconds=1),
        )
        assert exchange.model == "test/model"
        if endpoint == "messages":
            assert exchange.response.content[0].text == "hello"


def test_trajectory_rejects_mixed_representations() -> None:
    _, exchange = build_exchange(
        "chat_completions",
        {"model": "test/model", "messages": []},
        json.dumps(CHAT).encode(),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    with pytest.raises(ValueError, match="both exchanges and legacy histories"):
        art.Trajectory(
            exchanges=art.TrajectoryExchanges(chat_completions=[exchange]),
            messages_and_choices=[{"role": "user", "content": "hi"}],
        )


def test_metadata_accepts_json_serializable_values() -> None:
    assert art.Trajectory().model_dump() == {}
    trajectory = art.Trajectory(metadata={"nested": {"items": [1, "two"]}})
    assert trajectory.model_dump(mode="json")["metadata"] == {
        "nested": {"items": [1, "two"]}
    }
