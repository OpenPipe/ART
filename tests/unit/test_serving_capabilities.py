from copy import deepcopy
from typing import cast

import httpx
from pydantic import ValidationError
import pytest

from art.local.backend import LocalBackend
from art.model import Model
from art.serving_capabilities import (
    ART_SERVING_PROTOCOL_VERSION,
    FastMetricsSnapshot,
    ServingCapabilities,
)


def _snapshot_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": "art_vllm_runtime",
        "last_update_unix_s": 10.0,
        "record_count": 7,
        "engine_count": 1,
        "metrics": {
            "prompt_tokens_total": 100.0,
            "generation_tokens_total": 50.0,
            "prefix_cache_queries_total": 20.0,
            "prefix_cache_hits_total": 15.0,
            "num_preempted_reqs_total": 2.0,
            "num_requests_running": 3.0,
            "num_requests_waiting": 4.0,
            "num_requests_waiting_capacity": 2.0,
            "kv_cache_usage_perc": 0.5,
        },
        "process_uuid": "runtime-process",
        "generation": 3,
    }


def test_serving_capabilities_validate_isolated_metrics_endpoint() -> None:
    capabilities = ServingCapabilities(
        runtime="art_vllm",
        protocol_version=ART_SERVING_PROTOCOL_VERSION,
        fast_metrics={"url": "http://10.20.30.40:43123/art/metrics"},
    )
    assert capabilities.model_dump(mode="json")["fast_metrics"] == {
        "url": "http://10.20.30.40:43123/art/metrics"
    }

    for invalid in (
        {"protocol_version": ART_SERVING_PROTOCOL_VERSION - 1},
        {"fast_metrics": True},
        {"fast_metrics": {"url": "http://0.0.0.0:43123/art/metrics"}},
        {"fast_metrics": {"url": "/art/metrics"}},
    ):
        values = {
            "runtime": "art_vllm",
            "protocol_version": ART_SERVING_PROTOCOL_VERSION,
            **invalid,
        }
        with pytest.raises(ValidationError):
            ServingCapabilities.model_validate(values)


@pytest.mark.parametrize("invalid", [True, "1", [1.0], {"value": 1.0}, float("inf")])
def test_fast_metrics_snapshot_requires_finite_numeric_scalars(invalid: object) -> None:
    payload = deepcopy(_snapshot_payload())
    metrics = cast(dict[str, object], payload["metrics"])
    metrics["prompt_tokens_total"] = invalid
    with pytest.raises(ValidationError):
        FastMetricsSnapshot.model_validate(payload)


async def test_local_backend_collects_only_from_advertised_metrics_endpoint(
    tmp_path,
) -> None:
    metrics_url = "http://metrics.internal:43123/art/metrics"
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_snapshot_payload())

    model = Model(
        name="metrics-test",
        project="tests",
        inference_api_key="secret",
        inference_base_url="http://main-api.invalid/v1",
    )
    object.__setattr__(
        model,
        "_serving_capabilities",
        ServingCapabilities(
            runtime="art_vllm",
            protocol_version=ART_SERVING_PROTOCOL_VERSION,
            fast_metrics={"url": metrics_url},
        ),
    )
    backend = LocalBackend(path=str(tmp_path))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        metrics = await backend._collect_train_step_vllm_metrics(
            model, client=client, snapshots={}
        )

    assert [str(request.url) for request in requests] == [metrics_url]
    assert requests[0].headers["Authorization"] == "Bearer secret"
    assert metrics["vllm/num_requests_running"] == 3.0
    assert metrics["vllm/num_requests_waiting_capacity"] == 2.0
    assert metrics["vllm/prefix_cache_hit_rate"] == 0.75
