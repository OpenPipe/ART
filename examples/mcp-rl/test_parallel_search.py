"""Transport selection, cleanup and tool dispatch for the opt-in HTTP rollout."""

import asyncio
from contextlib import asynccontextmanager
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp import StdioServerParameters
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from openai.types.chat import ChatCompletion
import pytest

sys.path.insert(0, str(Path(__file__).parent))
transport = importlib.import_module("mcp_rl.transport")
rollout_module = importlib.import_module("mcp_rl.rollout")


@pytest.mark.asyncio
async def test_http_selection_and_cancellation_cleanup(monkeypatch):
    events = []

    @asynccontextmanager
    async def http(url, **kwargs):
        events.append((url, kwargs))
        try:
            yield "read", "write", lambda: None
        finally:
            events.append("closed")

    monkeypatch.setattr(transport, "streamablehttp_client", http)
    with pytest.raises(asyncio.CancelledError):
        async with transport.connect_mcp_transport(
            transport.parallel_search_params
        ) as streams:
            assert streams == ("read", "write")
            raise asyncio.CancelledError()
    assert events == [
        (
            "https://search.parallel.ai/mcp",
            {
                "headers": {
                    "User-Agent": "ART-MCP-RL/0.1 (https://github.com/OpenPipe/ART)"
                },
                "timeout": 30,
                "sse_read_timeout": 300,
            },
        ),
        "closed",
    ]


@pytest.mark.asyncio
async def test_stdio_parameters_preserved(monkeypatch):
    params = StdioServerParameters(
        command="python", args=["server.py"], env={"KEY": "value"}
    )
    seen = []

    @asynccontextmanager
    async def stdio(received):
        seen.append(received)
        yield "read", "write"
        seen.append("closed")

    monkeypatch.setattr(transport, "stdio_client", stdio)
    async with transport.connect_mcp_transport(params) as streams:
        assert streams == ("read", "write")
    assert seen == [params, "closed"]
    assert seen[0] is params


@pytest.mark.asyncio
@pytest.mark.parametrize("max_chars, expected_error", [(20000, True), (40000, False)])
async def test_rollout_discovers_and_dispatches_http_tools(
    monkeypatch, max_chars, expected_error
):
    session = SimpleNamespace(
        initialize=AsyncMock(),
        list_tools=AsyncMock(
            return_value=ListToolsResult(
                tools=[
                    Tool(
                        name="web_search",
                        description="Search",
                        inputSchema={"type": "object"},
                    )
                ]
            )
        ),
        call_tool=AsyncMock(
            return_value=CallToolResult(
                content=[
                    TextContent(
                        type="text", text="ART trains agents with GRPO" + "x" * 20000
                    )
                ]
            )
        ),
    )
    closed = []

    @asynccontextmanager
    async def streams(params):
        assert params is transport.parallel_search_params
        try:
            yield "read", "write"
        finally:
            closed.append("transport")

    @asynccontextmanager
    async def client_session(read, write):
        assert (read, write) == ("read", "write")
        try:
            yield session
        finally:
            closed.append("session")

    def response(name, arguments):
        return ChatCompletion.model_validate(
            {
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": name,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )

    create = AsyncMock(
        side_effect=[
            response("web_search", '{"objective":"ART"}'),
            response("complete_task", '{"summary":"Found GRPO"}'),
        ]
    )
    monkeypatch.setattr(rollout_module, "connect_mcp_transport", streams)
    monkeypatch.setattr(rollout_module, "ClientSession", client_session)
    monkeypatch.setattr(
        rollout_module,
        "AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    monkeypatch.setattr(
        rollout_module, "check_successful", AsyncMock(return_value=True)
    )
    model = SimpleNamespace(
        inference_api_key="fixture",
        inference_base_url="http://fixture",
        get_inference_name=lambda: "fixture",
    )
    scenario = rollout_module.McpScenario(
        "Find ART", transport.parallel_search_params, max_tool_result_chars=max_chars
    )
    trajectory = await rollout_module.rollout(model, scenario)
    session.initialize.assert_awaited_once()
    session.call_tool.assert_awaited_once_with("web_search", {"objective": "ART"})
    assert trajectory.tools[0]["function"]["name"] == "web_search"
    message = next(m for m in trajectory.messages() if m.get("role") == "tool")
    assert message["content"].startswith("Error:") is expected_error
    if not expected_error:
        assert "GRPO" in message["content"]
    assert trajectory.metrics["success"] is True
    assert closed == ["session", "transport"]


@pytest.mark.asyncio
async def test_connection_failure_returns_error_trajectory(monkeypatch):
    @asynccontextmanager
    async def failed_connection(params):
        raise RuntimeError("Connection unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr(rollout_module, "connect_mcp_transport", failed_connection)
    scenario = rollout_module.McpScenario("Find ART", transport.parallel_search_params)
    trajectory = await rollout_module.rollout(SimpleNamespace(), scenario)
    assert trajectory.metrics["num_turns"] == 0
    assert trajectory.metrics["success"] is False
    assert any("Connection unavailable" in log for log in trajectory.logs)
