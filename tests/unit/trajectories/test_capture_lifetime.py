"""Terminal capture must not pin a trajectory through a retained HTTP response."""

import asyncio
import json
import weakref

import httpx
from openai import AsyncOpenAI
import pytest

import art
from art.trajectories._capture import core

CHAT = {
    "id": "lifetime-fixture",
    "object": "chat.completion",
    "created": 1,
    "model": "policy",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
            "token_ids": [7],
            "prompt_token_ids": [2, 3],
            "logprobs": {
                "content": [
                    {
                        "token": "ok",
                        "logprob": -0.2,
                        "bytes": [111, 107],
                        "top_logprobs": [],
                    }
                ]
            },
        }
    ],
}


def test_terminal_state_releases_only_its_owners():
    trajectory = art.Trajectory()
    request = {"model": "policy", "messages": [{"role": "user", "content": "fixture"}]}
    state = core.CaptureState(trajectory, "chat_completions", request, status_code=200)
    state.add(json.dumps(CHAT).encode())
    ref = weakref.ref(trajectory)
    state.finish()
    (exchange,) = trajectory.exchanges.chat_completions
    assert exchange.request == request
    choice = exchange.response.choices[0]
    assert choice.logprobs is not None and choice.logprobs.content is not None
    assert choice.logprobs.content[0].logprob == -0.2
    assert choice.model_extra is not None and choice.model_extra["token_ids"] == [7]
    # Rebinding the private owner must not clear an aliased original request.
    assert request["messages"] == [{"role": "user", "content": "fixture"}]
    state.add(b"ignored after terminal capture")
    state.finish()
    assert len(trajectory.exchanges.chat_completions) == 1
    assert state.captured and not state.body and not state.request
    del trajectory
    assert ref() is None
    assert exchange.request == request


@pytest.mark.parametrize("terminal", ["discard", "status", "malformed", "append_error"])
def test_failed_capture_releases_state_and_preserves_error(terminal, monkeypatch):
    trajectory = art.Trajectory()
    state = core.CaptureState(
        trajectory,
        "chat_completions",
        {"model": "policy", "messages": []},
        status_code=200,
    )
    state.add(b"invalid" if terminal == "malformed" else json.dumps(CHAT).encode())
    ref = weakref.ref(trajectory)
    del trajectory
    assert ref() is not None  # Live capture still owns its destination.
    if terminal == "append_error":
        sentinel = KeyboardInterrupt("append sentinel")

        def fail(*args):
            raise sentinel

        monkeypatch.setattr(core, "_append_exchange", fail)
        with pytest.raises(KeyboardInterrupt) as caught:
            state.finish()
        assert caught.value is sentinel
        # An exception's traceback may legitimately own its frame and exchange.
        sentinel.__traceback__ = None
        del caught
    elif terminal == "discard":
        state.discard()
    else:
        if terminal == "status":
            state.status_code = 500
        state.finish()
    assert state.captured and state.trajectory is None
    assert not state.body and not state.request and ref() is None
    state.finish()


async def test_openai_httpx_response_cycle_does_not_retain_completed_trajectory():
    responses = []

    async def handler(request):
        response = httpx.Response(200, json=CHAT)
        responses.append(response)
        return response

    async with AsyncOpenAI(
        api_key="test",
        base_url="https://offline.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    ) as client:
        with art.Trajectory() as trajectory:
            completion = await client.chat.completions.create(
                model="policy", messages=[]
            )
        assert completion.choices[0].message.content == "ok"
        assert len(trajectory.exchanges.chat_completions) == 1
        (response,) = responses
        assert response.stream._response is response
        ref = weakref.ref(trajectory)
        del trajectory
        # Keep the actual HTTPX cycle strongly alive: no GC manipulation needed.
        assert ref() is None
        assert response.json() == CHAT


@pytest.mark.parametrize("failure", [None, "cancel", "read_error"])
async def test_stream_terminal_cleanup_preserves_delivery_and_failures(failure):
    sentinel = (
        asyncio.CancelledError("cancel sentinel")
        if failure == "cancel"
        else httpx.ReadError("read sentinel")
    )
    chunk = {
        "id": "stream-fixture",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "policy",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }
    first = b"data: " + json.dumps(chunk).encode() + b"\n\n"
    terminal = b"data: [DONE]\n\n"

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield first
            if failure:
                raise sentinel
            yield terminal

    async def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Stream()
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with art.Trajectory() as trajectory:
            async with client.stream(
                "POST",
                "https://offline.invalid/v1/chat/completions",
                json={"model": "policy", "messages": [], "stream": True},
            ) as response:
                ref = weakref.ref(trajectory)
                received = []
                try:
                    async for value in response.aiter_bytes():
                        received.append(value)
                        if value == first:
                            assert ref() is not None
                except BaseException as error:
                    assert failure and error is sentinel
                else:
                    assert failure is None
            assert b"".join(received) == first + (b"" if failure else terminal)
        assert len(trajectory.exchanges.chat_completions) == (not failure)
        del trajectory
        # Failure tracebacks are preserved, so clear only the test-owned sentinel.
        sentinel.__traceback__ = None
        assert getattr(response, "_art_trajectory_capture").trajectory is None
        assert ref() is None
