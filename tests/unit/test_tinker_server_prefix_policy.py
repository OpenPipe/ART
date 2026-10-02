"""Request policy reaches Tinker's worker and its actual prefix cache."""

import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest

from art.tinker import server as module
from art.token_prefix import prefix_edits
from art.utils.append_only import chat_prefix_scope


@pytest.fixture
def request_case(monkeypatch):
    server = module.OpenAICompatibleTinkerServer()
    sampling = SimpleNamespace(sample_async=AsyncMock(return_value=object()))
    closed = []

    @asynccontextmanager
    async def sampling_client():
        try:
            yield sampling
        finally:
            closed.append(True)

    model = SimpleNamespace(base_model="fixture-model", sampling_client=sampling_client)
    tenant = SimpleNamespace(get_samplable_model=AsyncMock(return_value=model))
    server._tenants["fixture"] = cast(Any, tenant)
    worker = SimpleNamespace(
        prompt_tokens=AsyncMock(return_value=[1, 500, 9]),
        chat_completion_and_prefixes=AsyncMock(
            return_value=(
                module.ChatCompletion(
                    id="fixture",
                    model="fixture-model",
                    created=0,
                    object="chat.completion",
                    choices=[],
                ),
                [([2, 500], [2, 101, 102], prefix_edits([2, 500], [2, 101, 102]))],
            )
        ),
    )
    server._workers = [cast(Any, worker)]
    body = {"model": "fixture-model", "messages": [{"role": "user", "content": "x"}]}

    class InProcessServer:
        def __init__(self, config):
            self.app = config.app

        async def serve(self, sockets):
            assert sockets == []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://fixture"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json=body,
                    headers={"Authorization": "Bearer fixture"},
                )
                assert response.status_code == 200, response.text
            assert client.is_closed

    monkeypatch.setattr(module, "_UvicornServer", InProcessServer)

    def request():
        asyncio.run(server._run("unused", 0, []))
        assert closed == [True]
        sampling.sample_async.assert_awaited_once()
        worker.chat_completion_and_prefixes.assert_awaited_once()

    return SimpleNamespace(
        request=request,
        body=body,
        server=server,
        worker=worker,
        sampling=sampling,
        raw_scope=json.dumps([id(tenant), model.base_model, None], sort_keys=True),
    )


@pytest.mark.parametrize(
    "parallel", [None, True, False], ids=["default", "parallel", "serial"]
)
def test_request_forwards_parallel_tool_policy(request_case, parallel):
    request_case.body["tools"] = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    if parallel is not None:
        request_case.body["parallel_tool_calls"] = parallel
    request_case.request()
    arguments = request_case.worker.chat_completion_and_prefixes.await_args.kwargs
    assert arguments.get("parallel_tool_calls") is parallel


@pytest.mark.parametrize("corrected_entry", [False, True])
def test_request_reuses_and_publishes_only_corrected_scope(
    request_case, corrected_entry
):
    cache = request_case.server._prefix_cache
    raw = request_case.raw_scope
    corrected = chat_prefix_scope(raw)
    cache.insert(raw, [1, 500], [1, 666, 777], "content")
    if corrected_entry:
        cache.insert(corrected, [1, 500], [1, 101, 102], "content")
    request_case.request()
    prompt = request_case.sampling.sample_async.await_args.kwargs["prompt"]
    assert list(prompt.chunks[0].tokens) == (
        [1, 101, 102, 9] if corrected_entry else [1, 500, 9]
    )
    published = cache.lookup(corrected, [2, 500], None)
    assert published is not None and list(published.raw_prefix) == [2, 101, 102]
    assert cache.lookup(raw, [2, 500], None) is None
