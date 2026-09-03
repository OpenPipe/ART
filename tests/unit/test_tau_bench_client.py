from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import importlib
import json
from types import SimpleNamespace
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import pytest

import art
import art.tau_bench.client as client_module
from art.tau_bench.client import (
    DeleteEnvironmentResponse,
    EnvironmentResponse,
    Scenario,
    StepEnvironmentResponse,
    Task,
    TauBenchClient,
)
import art.trajectories as tr


def test_client_reuses_connections_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    transport_kwargs: list[dict[str, Any]] = []

    class FakeTransport:
        def __init__(self, **kwargs: Any) -> None:
            transport_kwargs.append(kwargs)

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(client_module.httpx, "AsyncHTTPTransport", FakeTransport)
    TauBenchClient(base_url="http://tau.test", api_key="secret")

    assert len(transport_kwargs) == 64
    limits = [kwargs["limits"] for kwargs in transport_kwargs]
    assert all(isinstance(limit, httpx.Limits) for limit in limits)
    assert {limit.max_connections for limit in limits} == {100_000}
    assert sum(limit.max_keepalive_connections or 0 for limit in limits) == 100_000
    assert {kwargs["retries"] for kwargs in transport_kwargs} == {2}
    assert seen["timeout"] == httpx.Timeout(
        connect=10.0,
        pool=30.0,
        write=30.0,
        read=300.0,
    )


@pytest.mark.asyncio
async def test_sharded_transport_routes_round_robin_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transports: list[Any] = []

    class FakeTransport:
        def __init__(self, **kwargs: Any) -> None:
            self.index = len(transports)
            self.closed = False
            transports.append(self)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, request=request, extensions={"shard": self.index}
            )

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(client_module.httpx, "AsyncHTTPTransport", FakeTransport)
    transport = client_module._ShardedAsyncHTTPTransport(
        limits=httpx.Limits(
            max_connections=100_000,
            max_keepalive_connections=100_000,
        ),
        retries=2,
    )
    request = httpx.Request("GET", "http://tau.test")

    shards = [
        (await transport.handle_async_request(request)).extensions["shard"]
        for _ in range(65)
    ]
    await transport.aclose()

    assert shards == [*range(64), 0]
    assert all(item.closed for item in transports)


@pytest.mark.asyncio
async def test_sharded_transport_does_not_strand_aggregate_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTransport:
        def __init__(self, **kwargs: Any) -> None:
            self.capacity = asyncio.Semaphore(kwargs["limits"].max_connections)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await self.capacity.acquire()

            async def release() -> None:
                self.capacity.release()

            return httpx.Response(
                200,
                request=request,
                stream=ClosingStream(release),
            )

        async def aclose(self) -> None:
            pass

    class ClosingStream(httpx.AsyncByteStream):
        def __init__(self, close: Any) -> None:
            self.close = close

        async def __aiter__(self) -> AsyncGenerator[bytes, None]:
            if False:
                yield b""

        async def aclose(self) -> None:
            await self.close()

    monkeypatch.setattr(client_module.httpx, "AsyncHTTPTransport", FakeTransport)
    transport = client_module._ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=2, max_keepalive_connections=2),
        retries=0,
        shards=2,
    )
    request = httpx.Request(
        "GET",
        "http://tau.test",
        extensions={"timeout": {"pool": 0.1}},
    )

    slow = await transport.handle_async_request(request)
    completed = await transport.handle_async_request(request)
    with pytest.raises(httpx.PoolTimeout):
        await transport.handle_async_request(request)
    await completed.aclose()
    replacement = await transport.handle_async_request(request)

    await replacement.aclose()
    await slow.aclose()
    await transport.aclose()


@pytest.mark.asyncio
async def test_client_sends_auth_and_parses_scenarios() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["query"] = str(request.url.query, "utf-8")
        return httpx.Response(
            200,
            json={
                "scenarios": [
                    {"domain": "banking_knowledge", "task": {"id": "task_001"}}
                ]
            },
        )

    http_client = httpx.AsyncClient(
        base_url="http://tau.test",
        transport=httpx.MockTransport(handler),
    )
    client = TauBenchClient(api_key="secret", http_client=http_client)
    scenarios = await client.get_scenarios(domain="banking_knowledge", split="base")
    await client.close()
    await http_client.aclose()

    assert seen["authorization"] == "Bearer secret"
    assert seen["query"] == "domain=banking_knowledge&split=base"
    assert scenarios[0].task.id == "task_001"


@pytest.mark.asyncio
async def test_client_retries_transient_status_with_same_request_id() -> None:
    attempts: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.headers.get("x-request-id"))
        if len(attempts) < 3:
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(
            200,
            json={"scenarios": [{"domain": "telecom", "task": {"id": "task_001"}}]},
        )

    http_client = httpx.AsyncClient(
        base_url="http://tau.test",
        transport=httpx.MockTransport(handler),
    )
    client = TauBenchClient(
        api_key="secret",
        http_client=http_client,
        status_retries=3,
        retry_base_delay=0,
    )
    scenarios = await client.get_scenarios(domain="telecom")
    await client.close()
    await http_client.aclose()

    assert scenarios[0].task.id == "task_001"
    assert len(attempts) == 3
    assert attempts[0] is not None
    assert len(set(attempts)) == 1


@pytest.mark.asyncio
async def test_client_retries_transport_errors() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("temporary connect failure", request=request)
        return httpx.Response(
            200,
            json={"scenarios": [{"domain": "telecom", "task": {"id": "task_001"}}]},
        )

    http_client = httpx.AsyncClient(
        base_url="http://tau.test",
        transport=httpx.MockTransport(handler),
    )
    client = TauBenchClient(
        api_key="secret",
        http_client=http_client,
        status_retries=3,
        retry_base_delay=0,
    )
    scenarios = await client.get_scenarios(domain="telecom")
    await client.close()
    await http_client.aclose()

    assert scenarios[0].task.id == "task_001"
    assert attempts == 3


@pytest.mark.asyncio
async def test_owned_client_retries_pre_send_error_for_stateful_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def request(*args: Any, **kwargs: Any) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary connect failure")
        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://tau.test/environments"),
            json={"id": "env-1", "observation": "user: hello", "info": {}},
        )

    client = TauBenchClient(
        base_url="http://tau.test",
        api_key="secret",
        status_retries=3,
        retry_base_delay=0,
    )
    monkeypatch.setattr(client._client, "request", request)

    environment = await client.create_environment(domain="telecom", task_id="task_001")
    await client.close()

    assert environment.id == "env-1"
    assert attempts == 2


@pytest.mark.asyncio
async def test_caller_owned_client_does_not_retry_stateful_connect_error() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("custom transport failure", request=request)

    http_client = httpx.AsyncClient(
        base_url="http://tau.test",
        transport=httpx.MockTransport(handler),
    )
    client = TauBenchClient(
        api_key="secret",
        http_client=http_client,
        status_retries=3,
        retry_base_delay=0,
    )

    with pytest.raises(httpx.ConnectError):
        await client.create_environment(domain="telecom", task_id="task_001")
    await http_client.aclose()

    assert attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadError("response connection failed"),
        httpx.ReadTimeout("response timed out"),
        httpx.WriteError("request write failed"),
        httpx.RemoteProtocolError("response was incomplete"),
    ],
)
async def test_client_does_not_retry_ambiguous_stateful_transport_error(
    failure: httpx.TransportError,
) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        failure.request = request
        raise failure

    http_client = httpx.AsyncClient(
        base_url="http://tau.test",
        transport=httpx.MockTransport(handler),
    )
    client = TauBenchClient(
        api_key="secret",
        http_client=http_client,
        status_retries=3,
        retry_base_delay=0,
    )

    with pytest.raises(type(failure)):
        await client.create_environment(domain="telecom", task_id="task_001")
    await http_client.aclose()

    assert attempts == 1


@pytest.mark.asyncio
async def test_client_does_not_retry_transient_status_for_stateful_request() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="Unavailable")

    http_client = httpx.AsyncClient(
        base_url="http://tau.test",
        transport=httpx.MockTransport(handler),
    )
    client = TauBenchClient(
        api_key="secret",
        http_client=http_client,
        status_retries=3,
        retry_base_delay=0,
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.create_environment(domain="telecom", task_id="task_001")
    await http_client.aclose()

    assert attempts == 1


@pytest.mark.asyncio
async def test_owned_client_has_outer_request_attempt_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TauBenchClient(
        base_url="http://tau.test",
        api_key="secret",
        request_attempt_timeout=0.01,
        status_retries=0,
    )

    async def request(*args: Any, **kwargs: Any) -> httpx.Response:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(client._client, "request", request)
    with pytest.raises(httpx.TimeoutException, match="attempt deadline"):
        await client.get_scenarios()
    await client.close()


@pytest.mark.asyncio
async def test_caller_owned_client_is_not_wrapped_in_attempt_deadline() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.02)
        return httpx.Response(200, json={"scenarios": []})

    http_client = httpx.AsyncClient(
        base_url="http://tau.test",
        transport=httpx.MockTransport(handler),
    )
    client = TauBenchClient(
        api_key="secret",
        http_client=http_client,
        request_attempt_timeout=0.001,
    )

    assert await client.get_scenarios() == []
    await http_client.aclose()


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_client_rejects_invalid_request_attempt_timeout(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        TauBenchClient(request_attempt_timeout=value)


@pytest.mark.asyncio
async def test_client_sends_create_environment_idle_timeout() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"id": "env-1", "observation": "user: hello", "info": {}},
        )

    http_client = httpx.AsyncClient(
        base_url="http://tau.test",
        transport=httpx.MockTransport(handler),
    )
    client = TauBenchClient(api_key="secret", http_client=http_client)
    await client.create_environment(
        domain="telecom",
        task_id="task_001",
        idle_timeout_seconds=120,
    )
    await client.close()
    await http_client.aclose()

    assert seen["json"] == {
        "domain": "telecom",
        "task_id": "task_001",
        "idle_timeout_seconds": 120,
    }


@pytest.mark.asyncio
async def test_module_default_client_can_be_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tau_bench = importlib.import_module("art.tau_bench")
    client_module = importlib.import_module("art.tau_bench.client")

    class FakeClient(TauBenchClient):
        def __init__(self) -> None:
            pass

        async def get_scenarios(
            self,
            *,
            domain: str | None = None,
            split: str | None = None,
        ) -> list[Scenario]:
            return [Scenario(domain=domain or "", task=Task(id="task_001"))]

    original = client_module.default_client
    monkeypatch.setattr(client_module, "default_client", FakeClient())
    try:
        assert await tau_bench.get_scenarios(domain="telecom") == [
            Scenario(domain="telecom", task=Task(id="task_001"))
        ]
    finally:
        monkeypatch.setattr(client_module, "default_client", original)


class FakeTauBenchClient(TauBenchClient):
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def create_environment(
        self,
        *,
        domain: str,
        task_id: str,
        user_llm: str | None = None,
        user_llm_args: dict[str, Any] | None = None,
        retrieval_config: str | None = None,
        retrieval_config_kwargs: dict[str, Any] | None = None,
        idle_timeout_seconds: float | None = None,
    ) -> EnvironmentResponse:
        self.create_kwargs = {
            "domain": domain,
            "task_id": task_id,
            "user_llm": user_llm,
            "user_llm_args": user_llm_args,
            "retrieval_config": retrieval_config,
            "retrieval_config_kwargs": retrieval_config_kwargs,
            "idle_timeout_seconds": idle_timeout_seconds,
        }
        return EnvironmentResponse(
            id="env-1",
            observation="user: hello",
            info={"policy": "policy", "tools": []},
        )

    async def step_environment(
        self, env_id: str, action: str
    ) -> StepEnvironmentResponse:
        return StepEnvironmentResponse(
            id=env_id,
            observation=f"user: saw {action}",
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"user_message_cost": 0.25},
        )

    async def delete_environment(self, env_id: str) -> DeleteEnvironmentResponse:
        self.deleted.append(env_id)
        return DeleteEnvironmentResponse(id=env_id, deleted=True)


class FailingDeleteTauBenchClient(FakeTauBenchClient):
    async def delete_environment(self, env_id: str) -> DeleteEnvironmentResponse:
        raise RuntimeError(f"could not delete {env_id}")


class HangingDeleteTauBenchClient(FakeTauBenchClient):
    async def delete_environment(self, env_id: str) -> DeleteEnvironmentResponse:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class CancellationResistantDeleteTauBenchClient(FakeTauBenchClient):
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def delete_environment(self, env_id: str) -> DeleteEnvironmentResponse:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
        return DeleteEnvironmentResponse(id=env_id, deleted=True)


@pytest.mark.asyncio
async def test_environment_preserves_primary_error_when_cleanup_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FailingDeleteTauBenchClient()

    with pytest.raises(ValueError, match="rollout failed") as exc_info:
        async with client.environment(domain="telecom", task_id="task_001"):
            raise ValueError("rollout failed")

    assert exc_info.value.__notes__ == [
        "Failed to delete tau-bench environment env-1: RuntimeError('could not delete env-1')"
    ]
    assert "Failed to delete tau-bench environment env-1" in caplog.text


@pytest.mark.asyncio
async def test_environment_preserves_cancellation_when_cleanup_fails() -> None:
    client = FailingDeleteTauBenchClient()

    with pytest.raises(asyncio.CancelledError):
        async with client.environment(domain="telecom", task_id="task_001"):
            raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_environment_bounds_cleanup_while_preserving_primary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "DEFAULT_CLEANUP_TIMEOUT", 0.01)
    client = HangingDeleteTauBenchClient()

    with pytest.raises(ValueError, match="rollout failed"):
        async with client.environment(domain="telecom", task_id="task_001"):
            raise ValueError("rollout failed")


@pytest.mark.asyncio
async def test_environment_does_not_await_cancellation_resistant_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "DEFAULT_CLEANUP_TIMEOUT", 0.01)
    client = CancellationResistantDeleteTauBenchClient()

    with pytest.raises(ValueError, match="rollout failed"):
        async with client.environment(domain="telecom", task_id="task_001"):
            raise ValueError("rollout failed")

    await asyncio.wait_for(client.cancelled.wait(), timeout=0.1)
    client.release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_environment_propagates_cleanup_error_after_success() -> None:
    client = FailingDeleteTauBenchClient()

    with pytest.raises(RuntimeError, match="could not delete env-1"):
        async with client.environment(domain="telecom", task_id="task_001"):
            pass


class FakeCompletions:
    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return ChatCompletion.model_validate(
            {
                "id": "chat-1",
                "object": "chat.completion",
                "created": 0,
                "model": "default",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )


class FakeAsyncOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=FakeCompletions())


@pytest.mark.asyncio
async def test_rollout_supports_string_model_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout_module = importlib.import_module("art.tau_bench.rollout")
    rollout_module.openai_clients.clear()
    monkeypatch.setattr(rollout_module, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        rollout_module,
        "DefaultAsyncHttpxClient",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        rollout_module.httpx,
        "AsyncHTTPTransport",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    client = FakeTauBenchClient()
    scenario = Scenario(domain="banking_knowledge", task=Task(id="task_001"))

    trajectory = await rollout_module.rollout(
        scenario,
        "http://model.test/v1",
        "model-key",
        "default",
        client=client,
        base_model="Qwen/Qwen3.6-35B-A3B",
        max_turns=1,
    )

    assert trajectory.reward == 1.0
    assert trajectory.metrics["cost/user"] == 0.25
    assert trajectory.metrics["latency/environment_startup"] >= 0
    assert trajectory.metrics["latency/policy"] >= 0
    assert trajectory.metrics["latency/policy_max"] >= 0
    assert trajectory.metrics["latency/environment"] >= 0
    assert trajectory.metrics["latency/active"] >= 0
    assert trajectory.metrics["tokens/prompt"] == 10
    assert trajectory.metrics["tokens/completion"] == 5
    assert client.deleted == ["env-1"]
    assert client.create_kwargs["user_llm"] == "gpt-4.1-2025-04-14"
    assert client.create_kwargs["idle_timeout_seconds"] == 30 * 60
    policy_client: Any = rollout_module.openai_clients[
        ("http://model.test/v1", "model-key")
    ]
    assert policy_client.chat.completions.kwargs["stream"] is False
    assert (
        policy_client.chat.completions.kwargs["max_completion_tokens"]
        == rollout_module.DEFAULT_MAX_COMPLETION_TOKENS
    )
    assert policy_client.kwargs["max_retries"] == 1
    http_client = policy_client.kwargs["http_client"]
    assert http_client.timeout == httpx.Timeout(
        connect=10,
        read=10 * 60,
        write=30,
        pool=30,
    )
    transports = http_client.transport.transports
    assert len(transports) == 64
    assert {transport.retries for transport in transports} == {2}
    assert {transport.limits.max_connections for transport in transports} == {100_000}
    assert (
        sum(transport.limits.max_keepalive_connections for transport in transports)
        == 100_000
    )


@pytest.mark.asyncio
async def test_rollout_supports_art_model_like_args() -> None:
    rollout_module = importlib.import_module("art.tau_bench.rollout")
    model = art.Model(
        name="registered-model",
        project="test",
        inference_api_key="test-key",
        inference_base_url="http://model.test/v1",
    )
    object.__setattr__(model, "_openai_client", FakeAsyncOpenAI())
    client = FakeTauBenchClient()
    scenario = Scenario(domain="banking_knowledge", task=Task(id="task_001"))

    trajectory = await rollout_module.rollout(
        scenario,
        model,
        client=client,
        max_turns=1,
    )

    assert trajectory.metadata["scenario_id"] == "task_001"
    assert trajectory.metrics["num_turns"] == 1
    assert client.create_kwargs["idle_timeout_seconds"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["max_tokens", "max_completion_tokens"])
async def test_rollout_preserves_explicit_completion_limit(
    argument: str,
) -> None:
    rollout_module = importlib.import_module("art.tau_bench.rollout")
    model = art.Model(
        name="registered-model",
        project="test",
        inference_api_key="test-key",
        inference_base_url="http://model.test/v1",
    )
    client = FakeTauBenchClient()
    completion_client = FakeAsyncOpenAI()
    object.__setattr__(model, "_openai_client", completion_client)

    await rollout_module.rollout(
        Scenario(domain="banking_knowledge", task=Task(id="task_001")),
        model,
        client=client,
        max_turns=1,
        chat_completion_kwargs={argument: 123},
    )

    assert completion_client.chat.completions.kwargs[argument] == 123
    other = {"max_tokens", "max_completion_tokens"} - {argument}
    assert other.isdisjoint(completion_client.chat.completions.kwargs)


@pytest.mark.asyncio
async def test_rollout_preserves_server_lease_for_explicit_policy_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout_module = importlib.import_module("art.tau_bench.rollout")
    rollout_module.openai_clients.clear()
    monkeypatch.setattr(rollout_module, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(
        rollout_module,
        "DefaultAsyncHttpxClient",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        rollout_module.httpx,
        "AsyncHTTPTransport",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    client = FakeTauBenchClient()

    await rollout_module.rollout(
        Scenario(domain="banking_knowledge", task=Task(id="task_001")),
        "http://model.test/v1",
        "model-key",
        "default",
        client=client,
        max_turns=1,
        chat_completion_kwargs={"timeout": None},
    )

    assert client.create_kwargs["idle_timeout_seconds"] is None


@pytest.mark.asyncio
async def test_rollout_captures_two_turn_tool_exchange_with_exact_tokens() -> None:
    rollout_module = importlib.import_module("art.tau_bench.rollout")
    rollout_module.openai_clients.clear()
    request_bodies: list[dict[str, Any]] = []

    class ToolTauBenchClient(FakeTauBenchClient):
        async def step_environment(
            self, env_id: str, action: str
        ) -> StepEnvironmentResponse:
            if action == "lookup(key='x')":
                return StepEnvironmentResponse(
                    id=env_id,
                    observation="tool: result",
                    reward=0.25,
                    terminated=False,
                    truncated=False,
                    info={},
                )
            assert action == "hello"
            return StepEnvironmentResponse(
                id=env_id,
                observation="user: done",
                reward=0.75,
                terminated=True,
                truncated=False,
                info={"user_message_cost": 0.25},
            )

    async def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        if len(request_bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chat-tool",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "default",
                    "prompt_token_ids": [10, 11],
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"key":"x"}',
                                        },
                                    }
                                ],
                            },
                            "token_ids": [12],
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "token_id:12",
                                        "logprob": -0.25,
                                        "bytes": None,
                                        "top_logprobs": [],
                                    }
                                ]
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chat-exact",
                "object": "chat.completion",
                "created": 0,
                "model": "default",
                "prompt_token_ids": [10, 11, 12, 13],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello"},
                        "token_ids": [14],
                        "logprobs": {
                            "content": [
                                {
                                    "token": "token_id:14",
                                    "logprob": -0.5,
                                    "bytes": [104, 101, 108, 108, 111],
                                    "top_logprobs": [],
                                }
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    openai_client = AsyncOpenAI(
        api_key="model-key",
        base_url="http://model.test/v1",
        http_client=http_client,
    )
    rollout_module.openai_clients[("http://model.test/v1", "model-key")] = openai_client
    try:
        trajectory = await rollout_module.rollout(
            Scenario(domain="banking_knowledge", task=Task(id="task_001")),
            "http://model.test/v1",
            "model-key",
            "default",
            client=ToolTauBenchClient(),
            max_turns=2,
        )
    finally:
        await openai_client.close()
        await http_client.aclose()
        rollout_module.openai_clients.clear()

    assert request_bodies[0]["messages"] == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello"},
    ]
    assert request_bodies[1]["messages"][2]["tool_calls"][0]["id"] == "call-1"
    assert request_bodies[1]["messages"][3] == {
        "role": "tool",
        "content": "result",
        "tool_call_id": "call-1",
    }
    assert len(trajectory.exchanges.chat_completions) == 2
    assert not trajectory.messages_and_choices
    assert trajectory.tools is None
    restored = art.Trajectory.model_validate_json(trajectory.model_dump_json())
    tokenized = restored.tokenize()
    assert tokenized.tokens == [10, 11, 12, 13, 14]
    assert tokenized.logprobs[2] == -0.25
    assert tokenized.logprobs[3] != tokenized.logprobs[3]
    assert tokenized.logprobs[4] == -0.5
    assert tokenized.flags == [
        tr.TokenFlag.EXACT,
        tr.TokenFlag.EXACT,
        tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.ASSISTANT
        | tr.TokenFlag.OUTPUT,
        tr.TokenFlag.EXACT,
        tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.ASSISTANT
        | tr.TokenFlag.OUTPUT,
    ]


class FakeBadRequestError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class MaxTokensCompletions:
    async def create(self, **kwargs: Any) -> Any:
        raise FakeBadRequestError("max_tokens is too large for this model")


class MaxTokensAsyncOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.chat = SimpleNamespace(completions=MaxTokensCompletions())


class CountingTauBenchClient(FakeTauBenchClient):
    def __init__(self) -> None:
        super().__init__()
        self.steps = 0

    async def step_environment(
        self, env_id: str, action: str
    ) -> StepEnvironmentResponse:
        self.steps += 1
        return StepEnvironmentResponse(
            id=env_id,
            observation=f"user: saw {action}",
            reward=1.0,
            terminated=False,
            truncated=False,
            info={},
        )


@pytest.mark.asyncio
async def test_rollout_stops_on_max_tokens_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout_module = importlib.import_module("art.tau_bench.rollout")
    rollout_module.openai_clients.clear()
    monkeypatch.setattr(rollout_module, "AsyncOpenAI", MaxTokensAsyncOpenAI)
    monkeypatch.setattr(rollout_module, "BadRequestError", FakeBadRequestError)
    client = CountingTauBenchClient()
    scenario = Scenario(domain="banking_knowledge", task=Task(id="task_001"))

    trajectory = await rollout_module.rollout(
        scenario,
        "http://model.test/v1",
        "model-key",
        "default",
        client=client,
        max_turns=10,
    )

    assert trajectory.metrics["num_turns"] == 0
    assert client.steps == 0
    assert client.deleted == ["env-1"]


class NearContextLimitCompletions:
    async def create(self, **kwargs: Any) -> Any:
        return ChatCompletion.model_validate(
            {
                "id": "chat-near-limit",
                "object": "chat.completion",
                "created": 0,
                "model": "default",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 32_000,
                    "completion_tokens": 700,
                    "total_tokens": 32_700,
                },
            }
        )


class NearContextLimitAsyncOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.chat = SimpleNamespace(completions=NearContextLimitCompletions())


@pytest.mark.asyncio
async def test_rollout_stops_before_next_turn_exceeds_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout_module = importlib.import_module("art.tau_bench.rollout")
    rollout_module.openai_clients.clear()
    monkeypatch.setattr(rollout_module, "AsyncOpenAI", NearContextLimitAsyncOpenAI)
    client = CountingTauBenchClient()
    scenario = Scenario(domain="banking_knowledge", task=Task(id="task_001"))

    trajectory = await rollout_module.rollout(
        scenario,
        "http://model.test/v1",
        "model-key",
        "default",
        client=client,
        max_turns=10,
    )

    assert trajectory.metrics["num_turns"] == 1
    assert client.steps == 1
    assert client.deleted == ["env-1"]
