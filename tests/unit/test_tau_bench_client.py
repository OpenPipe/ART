from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import Any

import httpx
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


def test_client_reuses_connections_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class FakeTransport:
        def __init__(self, **kwargs: Any) -> None:
            seen["transport_kwargs"] = kwargs

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(client_module.httpx, "AsyncHTTPTransport", FakeTransport)
    TauBenchClient(base_url="http://tau.test", api_key="secret")

    limits = seen["transport_kwargs"]["limits"]
    assert isinstance(limits, httpx.Limits)
    assert limits.max_connections == 512
    assert limits.max_keepalive_connections == 512
    assert seen["transport_kwargs"]["retries"] == 2
    assert isinstance(seen["timeout"], httpx.Timeout)


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


class FakeCompletions:
    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        choice = SimpleNamespace(
            message=SimpleNamespace(content="hello", tool_calls=None)
        )
        return SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
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
    assert client.deleted == ["env-1"]
    assert client.create_kwargs["user_llm"] == "gpt-4.1-2025-04-14"


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
