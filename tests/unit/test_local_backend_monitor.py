import asyncio
from pathlib import Path

import pytest

from art import TrainableModel
from art.local import LocalBackend


class _FakeResponse:
    def __init__(self, body: str, status: int = 200) -> None:
        self._body = body
        self.status = status

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def text(self) -> str:
        return self._body


class _FakeSession:
    def __init__(self, urls: list[str]) -> None:
        self._urls = urls

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url: str, timeout) -> _FakeResponse:
        del timeout
        self._urls.append(url)
        if url.endswith("/metrics"):
            return _FakeResponse(
                "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
            )
        if url.endswith("/health"):
            return _FakeResponse("ok")
        raise AssertionError(f"Unexpected URL: {url}")


@pytest.mark.asyncio
async def test_monitor_openai_server_uses_health_probe_when_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    model = TrainableModel(
        name="qwen35-monitor",
        project="unit-tests",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        base_path=str(tmp_path),
    )

    class _FakeService:
        async def vllm_engine_is_sleeping(self) -> bool:
            return False

    backend._services[model.name] = _FakeService()  # type: ignore[index]
    requested_urls: list[str] = []
    sleep_calls = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr("art.local.backend.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "art.local.backend.aiohttp.ClientSession",
        lambda: _FakeSession(requested_urls),
    )

    with pytest.raises(asyncio.CancelledError):
        await backend._monitor_openai_server(
            model,
            "http://127.0.0.1:1234/v1",
            "default",
        )

    assert requested_urls == [
        "http://127.0.0.1:1234/metrics",
        "http://127.0.0.1:1234/health",
    ]
