from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

from aiohttp import web
import pytest

from art.distributed.specs import EndpointSpec
from art.distributed.vllm_gateway import VllmGateway, _aggregate_metric_snapshots
from art.distributed.vllm_router import (
    ReplicaTelemetry,
    RoutableReplica,
    RoutingTable,
)
from art.errors import ArtVllmMetricsTimeoutError
from art.local import backend as backend_module
from art.local.backend import LocalBackend
from art.pipeline_tuner import PipelineAutotuneConfig
from art.pipeline_tuner import attachment as attachment_module
from art.pipeline_tuner.attachment import PipelineAutotunerAttachment


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


def test_runtime_world_size_includes_data_parallel_replicas(
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


def _replica(replica_id: str, port: int, generation: int) -> RoutableReplica:
    return RoutableReplica(
        replica_id=replica_id,
        endpoint=EndpointSpec(host="127.0.0.1", port=port),
        phase="ready",
        generation=generation,
        generation_digest=f"digest-{replica_id}-{generation}",
        committed_version="0",
        policy_digest="policy-digest",
        update_identity="update-0",
        telemetry=ReplicaTelemetry(
            observed_at=asyncio.get_running_loop().time(),
            in_flight=0,
            capacity=8,
        ),
    )


def _table(*replicas: RoutableReplica) -> RoutingTable:
    return RoutingTable(
        policy_generation=0,
        policy_version="0",
        policy_digest="policy-digest",
        update_identity="update-0",
        replicas=replicas,
    )


def _metrics(
    *,
    prompt: float,
    generation: float,
    queries: float,
    hits: float,
    running: float = 1.0,
    kv_pressure: float = 0.25,
    world_size: float = 8.0,
    max_model_len: float = 8192.0,
) -> dict[str, float]:
    return {
        "prompt_tokens_total": prompt,
        "generation_tokens_total": generation,
        "prefix_cache_queries_total": queries,
        "prefix_cache_hits_total": hits,
        "external_prefix_cache_queries_total": queries / 10,
        "external_prefix_cache_hits_total": hits / 10,
        "num_preempted_reqs_total": 1.0,
        "num_requests_running": running,
        "num_requests_waiting": 2.0,
        "num_requests_waiting_capacity": 1.0,
        "num_requests_waiting_deferred": 1.0,
        "kv_cache_usage_perc": kv_pressure,
        "max_num_seqs": 8.0,
        "max_num_batched_tokens": 1024.0,
        "max_num_scheduled_tokens": 1024.0,
        "max_model_len": max_model_len,
        "world_size": world_size,
        "prefix_cache_hit_rate": hits / queries if queries else 0.0,
        "external_prefix_cache_hit_rate": hits / queries if queries else 0.0,
    }


@pytest.mark.asyncio
async def test_gateway_returns_semantic_partial_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replicas = (
        _replica("r0", 10000, 3),
        _replica("r1", 10001, 4),
        _replica("r2", 10002, 5),
    )
    gateway = VllmGateway(_table(*replicas))
    responses: dict[str, dict[str, float] | BaseException] = {
        replicas[0].endpoint.url: _metrics(
            prompt=100, generation=50, queries=100, hits=50, kv_pressure=0.4
        ),
        replicas[1].endpoint.url: _metrics(
            prompt=10,
            generation=5,
            queries=10,
            hits=1,
            running=3,
            kv_pressure=0.8,
            max_model_len=4096,
        ),
        replicas[2].endpoint.url: OSError("replica unavailable"),
    }

    async def scrape(endpoint: str) -> dict[str, float]:
        result = responses[endpoint]
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(gateway, "_metrics", scrape)
    result = await gateway._aggregate_metrics()

    assert result.incomplete is True
    assert [(item.replica_id, item.generation) for item in result.replicas] == [
        ("r0", 3),
        ("r1", 4),
    ]
    assert result.metrics["num_requests_running"] == 4.0
    assert result.metrics["max_num_seqs"] == 16.0
    assert result.metrics["world_size"] == 16.0
    assert result.metrics["max_model_len"] == 8192.0
    assert result.metrics["kv_cache_usage_perc"] == 0.8
    assert result.metrics["prefix_cache_hit_rate"] == pytest.approx(51 / 110)


@pytest.mark.asyncio
async def test_gateway_returns_503_only_when_every_scrape_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replicas = (_replica("r0", 10000, 0), _replica("r1", 10001, 0))
    gateway = VllmGateway(_table(*replicas))
    calls: list[str] = []

    async def scrape(endpoint: str) -> dict[str, float]:
        calls.append(endpoint)
        raise OSError("replica unavailable")

    monkeypatch.setattr(gateway, "_metrics", scrape)
    with pytest.raises(web.HTTPServiceUnavailable) as error:
        await gateway._aggregate_metrics()

    assert error.value.status == 503
    assert set(calls) == {replica.endpoint.url for replica in replicas}


def _gateway_payload(
    replicas: list[tuple[str, int, dict[str, float]]], *, incomplete: bool = False
) -> dict[str, Any]:
    return {
        "metrics": _aggregate_metric_snapshots(metrics for _, _, metrics in replicas),
        "replicas": [
            {"replica_id": replica_id, "generation": generation, "metrics": metrics}
            for replica_id, generation, metrics in replicas
        ],
        "incomplete": incomplete,
    }


@pytest.mark.asyncio
async def test_backend_deltas_are_keyed_by_replica_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payloads = iter(
        [
            _gateway_payload(
                [
                    ("r0", 0, _metrics(prompt=100, generation=50, queries=20, hits=10)),
                    ("r1", 0, _metrics(prompt=100, generation=50, queries=20, hits=10)),
                ]
            ),
            _gateway_payload(
                [
                    (
                        "r0",
                        0,
                        _metrics(prompt=200, generation=100, queries=40, hits=20),
                    ),
                    (
                        "r1",
                        0,
                        _metrics(prompt=200, generation=100, queries=40, hits=20),
                    ),
                ]
            ),
            _gateway_payload(
                [
                    (
                        "r0",
                        0,
                        _metrics(prompt=300, generation=150, queries=60, hits=30),
                    ),
                    (
                        "r1",
                        1,
                        _metrics(
                            prompt=10_000, generation=5_000, queries=2_000, hits=1_000
                        ),
                    ),
                ]
            ),
            _gateway_payload(
                [("r0", 0, _metrics(prompt=400, generation=200, queries=80, hits=40))],
                incomplete=True,
            ),
        ]
    )

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

        async def get(self, *_args: Any, **_kwargs: Any) -> Response:
            return Response(next(payloads))

    times = iter((0.0, 10.0, 20.0, 30.0))
    monkeypatch.setattr(backend_module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(
        backend_module, "time", SimpleNamespace(monotonic=lambda: next(times))
    )
    backend = LocalBackend(path=str(tmp_path))
    model: Any = SimpleNamespace(
        name="test-model",
        inference_base_url="http://metrics.test/v1",
        inference_api_key=None,
        _serving_capabilities=SimpleNamespace(require=lambda *_args, **_kwargs: None),
    )

    first = await backend.collect_train_step_vllm_metrics(model)
    second = await backend.collect_train_step_vllm_metrics(model)
    restarted = await backend.collect_train_step_vllm_metrics(model)
    incomplete = await backend.collect_train_step_vllm_metrics(model)

    assert "vllm/prompt_tok_per_s" not in first
    assert second["vllm/prompt_tok_per_s"] == 20.0
    assert second["vllm/completion_tok_per_s"] == 10.0
    assert restarted["vllm/prompt_tok_per_s"] == 10.0
    assert restarted["vllm/completion_tok_per_s"] == 5.0
    assert restarted["vllm/prefix_cache_hit_rate"] == 0.5
    assert incomplete["vllm/prompt_tok_per_s"] == 10.0
    assert incomplete["vllm/metrics_incomplete"] == 1.0
    assert set(backend._vllm_metric_snapshots) == {("test-model", "r0", 0)}


@pytest.mark.asyncio
async def test_online_tuner_records_incomplete_scrape_as_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = {
        "vllm/num_requests_running": 1.0,
        "vllm/num_requests_waiting": 0.0,
        "vllm/num_requests_waiting_capacity": 0.0,
        "vllm/kv_cache_usage_perc": 0.5,
        "vllm/num_preemptions_total": 0.0,
        "vllm/metrics_incomplete": 1.0,
    }

    class Backend:
        async def collect_train_step_vllm_metrics(
            self, _model: object
        ) -> dict[str, float]:
            return required

    class State:
        checks = 0

        @property
        def done(self) -> bool:
            self.checks += 1
            return self.checks > 1

    trainer = SimpleNamespace(
        backend=Backend(), model=object(), state=State(), request_stop=Mock()
    )
    attachment = PipelineAutotunerAttachment(PipelineAutotuneConfig())
    attachment.trainer = trainer

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(attachment_module.asyncio, "sleep", no_sleep)
    with pytest.raises(ArtVllmMetricsTimeoutError):
        await attachment._collect_required_serving_metrics()
    await attachment._sample_serving_metrics()

    assert len(attachment._poll_health) == 1
    assert attachment._poll_health[0].timed_out is True
    assert attachment._sampler_error is None
    trainer.request_stop.assert_not_called()
