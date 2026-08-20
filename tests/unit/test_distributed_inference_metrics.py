from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import cast

import httpx
import pytest

from art.local import backend as backend_module
from art.local.backend import LocalBackend
from art.model import Model
from art.serving_capabilities import ART_SERVING_PROTOCOL_VERSION, ServingCapabilities


def _runtime_metrics_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    loggers = ModuleType("vllm.v1.metrics.loggers")
    setattr(loggers, "StatLoggerBase", object)
    fast_metrics = ModuleType("art_vllm_runtime.fast_metrics")
    setattr(fast_metrics, "FastMetricsSharedWriter", object)
    for name in ("vllm", "vllm.v1", "vllm.v1.metrics"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, loggers.__name__, loggers)
    monkeypatch.setitem(sys.modules, fast_metrics.__name__, fast_metrics)
    path = Path(__file__).parents[2] / "vllm_runtime/src/art_vllm_runtime/metrics.py"
    spec = importlib.util.spec_from_file_location("test_art_vllm_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(*, prompt: float, generation: float) -> dict[str, float]:
    return {
        "prompt_tokens_total": prompt,
        "generation_tokens_total": generation,
        "prefix_cache_queries_total": prompt / 5,
        "prefix_cache_hits_total": prompt / 10,
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


def _snapshot(
    process_uuid: str,
    generation: int,
    record_count: int,
    *,
    prompt: float,
    completion: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "art_vllm_runtime",
        "last_update_unix_s": float(record_count),
        "record_count": record_count,
        "engine_count": 1,
        "process_uuid": process_uuid,
        "generation": generation,
        "metrics": _metrics(prompt=prompt, generation=completion),
    }


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


@pytest.mark.asyncio
async def test_metrics_endpoint_and_counter_generation_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payloads = iter(
        (
            _snapshot("leader-a", 0, 1, prompt=100, completion=50),
            _snapshot("leader-a", 0, 2, prompt=200, completion=100),
            _snapshot("leader-b", 1, 1, prompt=10_000, completion=5_000),
            _snapshot("leader-b", 1, 2, prompt=10_100, completion=5_050),
        )
    )
    requests: list[tuple[str, dict[str, str] | None]] = []

    class Client:
        async def get(
            self, url: str, *, headers: dict[str, str] | None
        ) -> httpx.Response:
            requests.append((url, headers))
            return httpx.Response(
                200,
                json=next(payloads),
                request=httpx.Request("GET", url, headers=headers),
            )

    times = iter((0.0, 10.0, 20.0, 30.0))
    monkeypatch.setattr(
        backend_module, "time", SimpleNamespace(monotonic=lambda: next(times))
    )
    backend = LocalBackend(path=str(tmp_path))
    backend._vllm_metrics_client = cast(httpx.AsyncClient, Client())
    model = Model(
        name="test-model",
        project="test",
        inference_base_url="http://leader.test/v1",
        inference_api_key="secret",
    )
    object.__setattr__(
        model,
        "_serving_capabilities",
        ServingCapabilities(
            runtime="art_vllm",
            protocol_version=ART_SERVING_PROTOCOL_VERSION,
            fast_metrics={"url": "http://leader.test/art/metrics"},
        ),
    )

    first = await backend.collect_train_step_vllm_metrics(model)
    second = await backend.collect_train_step_vllm_metrics(model)
    restarted = await backend.collect_train_step_vllm_metrics(model)
    recovered = await backend.collect_train_step_vllm_metrics(model)

    assert "vllm/prompt_tok_per_s" not in first
    assert (second["vllm/prompt_tok_per_s"], second["vllm/completion_tok_per_s"]) == (
        10.0,
        5.0,
    )
    assert "vllm/prompt_tok_per_s" not in restarted
    assert restarted["vllm/prefix_cache_hit_rate"] == 0.5
    assert (
        recovered["vllm/prompt_tok_per_s"],
        recovered["vllm/completion_tok_per_s"],
        recovered["vllm/world_size"],
    ) == (10.0, 5.0, 16.0)
    assert set(backend._vllm_metric_snapshots) == {
        ("test", "test-model", "leader-b", 1)
    }
    assert (
        requests
        == [("http://leader.test/art/metrics", {"Authorization": "Bearer secret"})] * 4
    )
