from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from art.local import backend as backend_module
from art.local.backend import LocalBackend


def _runtime_metrics_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    loggers = ModuleType("vllm.v1.metrics.loggers")
    setattr(loggers, "StatLoggerBase", object)
    for name in ("vllm", "vllm.v1", "vllm.v1.metrics"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, loggers.__name__, loggers)
    path = Path(__file__).parents[1] / "vllm_runtime/src/art_vllm_runtime/metrics.py"
    spec = importlib.util.spec_from_file_location("test_art_vllm_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_world_size_includes_data_parallel_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _runtime_metrics_module(monkeypatch)._ArtRuntimeMetricsState()
    state.configure(
        SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=8,
                max_num_batched_tokens=1024,
                max_num_scheduled_tokens=1024,
            ),
            model_config=SimpleNamespace(max_model_len=4096),
            parallel_config=SimpleNamespace(world_size=8, world_size_across_dp=16),
        ),
        engine_idx=0,
    )

    assert state.snapshot()["metrics"]["world_size"] == 16.0


def _metrics(
    *, prompt: float, generation: float, queries: float, hits: float
) -> dict[str, float]:
    return {
        "prompt_tokens_total": prompt,
        "generation_tokens_total": generation,
        "prefix_cache_queries_total": queries,
        "prefix_cache_hits_total": hits,
        "num_preempted_reqs_total": 1.0,
        "num_requests_running": 1.0,
        "num_requests_waiting": 2.0,
        "num_requests_waiting_capacity": 1.0,
        "kv_cache_usage_perc": 0.25,
        "max_num_seqs": 8.0,
        "max_num_batched_tokens": 1024.0,
        "max_num_scheduled_tokens": 1024.0,
        "max_model_len": 8192.0,
        "world_size": 16.0,
    }


@pytest.mark.asyncio
async def test_backend_reads_leader_metrics_and_fences_counter_generations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payloads = iter(
        [
            {
                "process_uuid": "leader-a",
                "generation": 0,
                "metrics": _metrics(prompt=100, generation=50, queries=20, hits=10),
            },
            {
                "process_uuid": "leader-a",
                "generation": 0,
                "metrics": _metrics(prompt=200, generation=100, queries=40, hits=20),
            },
            {
                "process_uuid": "leader-b",
                "generation": 1,
                "metrics": _metrics(
                    prompt=10_000, generation=5_000, queries=2_000, hits=1_000
                ),
            },
            {
                "process_uuid": "leader-b",
                "generation": 1,
                "metrics": _metrics(
                    prompt=10_100, generation=5_050, queries=2_020, hits=1_010
                ),
            },
        ]
    )
    requests: list[tuple[str, dict[str, str] | None]] = []

    class Response:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str] | None) -> Response:
            requests.append((url, headers))
            return Response(next(payloads))

    times = iter((0.0, 10.0, 20.0, 30.0))
    monkeypatch.setattr(backend_module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(
        backend_module, "time", SimpleNamespace(monotonic=lambda: next(times))
    )
    backend = LocalBackend(path=str(tmp_path))
    model: Any = SimpleNamespace(
        name="test-model",
        inference_base_url="http://leader.test/v1",
        inference_api_key="secret",
        _serving_capabilities=SimpleNamespace(require=lambda *_args, **_kwargs: None),
    )

    first = await backend.collect_train_step_vllm_metrics(model)
    second = await backend.collect_train_step_vllm_metrics(model)
    restarted = await backend.collect_train_step_vllm_metrics(model)
    recovered = await backend.collect_train_step_vllm_metrics(model)

    assert "vllm/prompt_tok_per_s" not in first
    assert second["vllm/prompt_tok_per_s"] == 10.0
    assert second["vllm/completion_tok_per_s"] == 5.0
    assert "vllm/prompt_tok_per_s" not in restarted
    assert restarted["vllm/prefix_cache_hit_rate"] == 0.5
    assert recovered["vllm/prompt_tok_per_s"] == 10.0
    assert recovered["vllm/completion_tok_per_s"] == 5.0
    assert recovered["vllm/world_size"] == 16.0
    assert set(backend._vllm_metric_snapshots) == {("test-model", "leader-b", 1)}
    assert (
        requests
        == [
            (
                "http://leader.test/art/metrics",
                {"Authorization": "Bearer secret"},
            )
        ]
        * 4
    )
