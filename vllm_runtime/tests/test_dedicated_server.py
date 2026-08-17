import asyncio
from http.client import HTTPConnection
import json
import os
from types import SimpleNamespace

from art_vllm_runtime import dedicated_server
from art_vllm_runtime.fast_metrics import FAST_METRIC_NAMES, FastMetricsSidecar
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.datastructures import URL

_PAYLOAD: dict[str, object] = {
    "schema_version": 1,
    "source": "art_vllm_runtime",
    "last_update_unix_s": 1.0,
    "record_count": 1,
    "engine_count": 1,
    "metrics": {
        **dict.fromkeys(FAST_METRIC_NAMES, 0.0),
        "num_requests_running": 2.0,
        "prompt_tokens_total": 3.0,
    },
    "process_uuid": "runtime-process",
    "generation": 4,
}


def _get(
    connection: HTTPConnection, *, token: str | None = None
) -> tuple[int, int, dict[str, object]]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    connection.request("GET", "/art/metrics", headers=headers)
    response = connection.getresponse()
    return response.status, response.version, json.loads(response.read())


def _start_sidecar(*, tokens: list[str], port: int = 0) -> FastMetricsSidecar:
    sidecar = FastMetricsSidecar.start(
        "127.0.0.1",
        tokens,
        process_uuid="runtime-process",
        generation=4,
        port=port,
    )
    sidecar.writer.publish(
        last_update_unix_s=1.0,
        record_count=1,
        engine_count=1,
        metrics=_PAYLOAD["metrics"],  # type: ignore[arg-type]
    )
    return sidecar


def test_fast_metrics_listener_auth_keepalive_and_scalar_payload() -> None:
    sidecar = _start_sidecar(tokens=["first", "second"])
    assert sidecar.process.pid != os.getpid()
    connection = HTTPConnection("127.0.0.1", sidecar.port, timeout=1.0)
    try:
        assert _get(connection)[0] == 401
        reused_socket = connection.sock
        status, version, payload = _get(connection, token="second")
        assert (status, version) == (200, 11)
        assert connection.sock is reused_socket
        assert _get(connection, token="second")[0] == 200
        assert connection.sock is reused_socket
        assert payload == _PAYLOAD
        metrics = payload["metrics"]
        assert isinstance(metrics, dict)
        assert all(type(value) in {int, float} for value in metrics.values())
    finally:
        connection.close()
        sidecar.close()
    assert sidecar.process.poll() == 0


def test_fast_metrics_listener_reads_updated_shared_snapshot() -> None:
    sidecar = _start_sidecar(tokens=[])
    connection = HTTPConnection("127.0.0.1", sidecar.port, timeout=1.0)
    try:
        metrics = dict(_PAYLOAD["metrics"])  # type: ignore[arg-type]
        metrics["num_requests_running"] = 7.0
        sidecar.writer.publish(
            last_update_unix_s=2.0,
            record_count=2,
            engine_count=1,
            metrics=metrics,
        )
        _, _, payload = _get(connection)
        assert payload["record_count"] == 2
        assert payload["last_update_unix_s"] == 2.0
        assert payload["metrics"]["num_requests_running"] == 7.0  # type: ignore[index]
    finally:
        connection.close()
        sidecar.close()


def test_fast_metrics_listener_reports_unpublished_snapshot() -> None:
    sidecar = FastMetricsSidecar.start(
        "127.0.0.1", [], process_uuid="runtime-process", generation=4
    )
    connection = HTTPConnection("127.0.0.1", sidecar.port, timeout=1.0)
    try:
        status, _, payload = _get(connection)
        assert status == 503
        assert payload == {"error": "Metrics unavailable"}
    finally:
        connection.close()
        sidecar.close()


def test_fast_metrics_listener_stops_and_restarts_on_same_port() -> None:
    sidecar = _start_sidecar(tokens=[])
    port = sidecar.port
    sidecar.close()
    assert sidecar.process.poll() == 0

    restarted = _start_sidecar(tokens=[], port=port)
    connection = HTTPConnection("127.0.0.1", port, timeout=1.0)
    try:
        assert _get(connection)[0] == 200
    finally:
        connection.close()
        restarted.close()
    assert restarted.process.poll() == 0


def test_fast_metrics_listener_accepts_parent_shutdown_sigterm() -> None:
    sidecar = _start_sidecar(tokens=[])
    sidecar.process.terminate()
    sidecar.close()
    assert sidecar.process.poll() < 0


def test_fast_metrics_url_uses_controller_routable_host(monkeypatch) -> None:
    monkeypatch.setattr(dedicated_server, "_fast_metrics_port", 43123)
    monkeypatch.setitem(dedicated_server._runtime_state, "nnodes", 2)
    request = SimpleNamespace(url=URL("https://10.20.30.40:8000/art/capabilities"))
    assert (
        dedicated_server._fast_metrics_url(request)
        == "http://10.20.30.40:43123/art/metrics"
    )

    for host in ("0.0.0.0", "127.0.0.1", "[::]"):
        request = SimpleNamespace(url=URL(f"http://{host}:8000/art/capabilities"))
        with pytest.raises(RuntimeError, match="unroutable host"):
            dedicated_server._fast_metrics_url(request)


def test_runtime_sleep_route_returns_engine_validation_error(monkeypatch) -> None:
    from vllm.entrypoints.openai import api_server

    monkeypatch.setattr(api_server, "build_app", lambda *args, **kwargs: FastAPI())
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    dedicated_server._patch_art_runtime_routes()
    app = api_server.build_app()

    class Engine:
        async def sleep(self, *, level: int, mode: str) -> None:
            raise ValueError(f"invalid {level=} {mode=}")

    app.state.engine_client = Engine()
    response = TestClient(app).post("/sleep?level=1&mode=wait")
    assert response.status_code == 400
    assert response.json() == {"error": "invalid level=1 mode='wait'"}


@pytest.mark.asyncio
async def test_lora_update_request_timing_captures_body_boundary() -> None:
    observed: dict[str, float] = {}

    async def app(scope, receive, _send) -> None:
        await receive()
        observed.update(scope["state"])

    middleware = dedicated_server._ArtRequestTimingMiddleware(app)

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"{}", "more_body": False}

    await middleware(
        {"type": "http", "method": "POST", "path": "/art/in_flight_lora_update"},
        receive,
        lambda _message: None,
    )

    assert (
        observed[dedicated_server._BODY_RECEIVED_AT]
        >= observed[dedicated_server._ASGI_STARTED_AT]
    )


def test_launch_policy_version_resolves_loaded_slot_alias(monkeypatch) -> None:
    monkeypatch.setitem(dedicated_server._runtime_state, "loaded_adapter", "policy@5")
    monkeypatch.setitem(dedicated_server._runtime_state, "policy_version", 5)

    assert (
        dedicated_server._launch_policy_version_for_slot(
            lora_slot="policy", public_model_name="policy@6"
        )
        == 5
    )


@pytest.mark.asyncio
async def test_lora_mutations_are_serialized_across_slots() -> None:
    models = SimpleNamespace()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    events: list[str] = []

    async def mutate(slot: str) -> None:
        async with dedicated_server._lora_mutation_lock(models, slot):
            events.append(f"{slot}:enter")
            if slot == "first":
                first_entered.set()
                await release_first.wait()
            events.append(f"{slot}:exit")

    first = asyncio.create_task(mutate("first"))
    await first_entered.wait()
    second = asyncio.create_task(mutate("second"))
    await asyncio.sleep(0)
    assert events == ["first:enter"]

    release_first.set()
    await asyncio.gather(first, second)
    assert events == ["first:enter", "first:exit", "second:enter", "second:exit"]


def test_kv_preflight_runs_before_pipeline_route_mutation(monkeypatch) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enable_return_routed_experts=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            distributed_executor_backend="mp",
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        use_v2_model_runner=False,
        kv_transfer_config=SimpleNamespace(is_kv_transfer_instance=True),
    )

    class EngineArgs:
        def create_engine_config(self):
            return config

    monkeypatch.setattr(
        dedicated_server,
        "_register_model_route_layout",
        lambda _model_config: None,
    )
    dedicated_server._patch_engine_config(EngineArgs, pipeline_route_capture=True)

    with pytest.raises(ValueError, match="KV connectors"):
        EngineArgs().create_engine_config()

    assert config.model_config.enable_return_routed_experts is False
