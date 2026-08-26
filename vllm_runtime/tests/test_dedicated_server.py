import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
import json
import os
from types import SimpleNamespace

from art_vllm_runtime import dedicated_server
from art_vllm_runtime.fast_metrics import FAST_METRIC_NAMES, FastMetricsSidecar
from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
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


def _fake_vllm_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def original_chat_completion():
        raise AssertionError("ART must replace vLLM's original route")

    @app.post("/v1/completions")
    async def original_completion():
        raise AssertionError("ART must replace vLLM's original route")

    return app


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

    monkeypatch.setattr(
        api_server, "build_app", lambda *args, **kwargs: _fake_vllm_app()
    )
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


@pytest.mark.parametrize("prompt_length", [128, 4096])
def test_completion_schema_preserves_exact_prompt_token_ids(prompt_length: int) -> None:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.renderers.inputs.preprocess import parse_model_prompt

    prompt_ids = [index % 257 for index in range(prompt_length)]
    request = CompletionRequest(
        model="model",
        prompt=prompt_ids,
        add_special_tokens=False,
        return_token_ids=True,
    )

    parsed = parse_model_prompt(
        SimpleNamespace(is_encoder_decoder=False), request.prompt
    )

    assert parsed["prompt_token_ids"] == prompt_ids


def test_binary_completion_route_uses_exact_token_prompt(monkeypatch) -> None:
    from art_vllm_runtime import binary_routes
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.completion import api_router

    from art.vllm_route_transport import (
        decode_routed_experts_completion_response_stream,
    )

    prompt_ids = list(range(128))
    observed: dict[str, object] = {}

    async def create_completion(request, _raw_request):
        observed["request"] = request
        return dedicated_server.JSONResponse(
            content={
                "id": "cmpl-route-test",
                "object": "text_completion",
                "created": 0,
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": "length",
                        "prompt_token_ids": prompt_ids,
                        "token_ids": [7],
                    }
                ],
                "usage": {
                    "prompt_tokens": 128,
                    "completion_tokens": 1,
                    "total_tokens": 129,
                },
            }
        )

    @contextmanager
    def capture_routed_experts():
        routes = binary_routes._CapturedRoutes(num_experts=8, padding_layers=())
        routes[0] = np.zeros((128, 1, 1), dtype=np.uint8)
        yield routes

    monkeypatch.setattr(
        api_server, "build_app", lambda *args, **kwargs: _fake_vllm_app()
    )
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    monkeypatch.setattr(api_router, "create_completion", create_completion)
    monkeypatch.setattr(binary_routes, "capture_routed_experts", capture_routed_experts)
    dedicated_server._patch_art_runtime_routes()
    app = api_server.build_app()
    response = TestClient(app).post(
        "/art/v1/completions",
        json={
            "model": "model",
            "prompt": prompt_ids,
            "max_tokens": 1,
            "stream": False,
            "add_special_tokens": False,
            "return_token_ids": True,
        },
    )
    assert response.status_code == 200

    async def chunks():
        yield response.content

    decoded, routes = asyncio.run(
        decode_routed_experts_completion_response_stream(chunks())
    )
    request = observed["request"]
    assert request.prompt == prompt_ids
    assert decoded.choices[0].prompt_token_ids == prompt_ids
    assert routes[0].shape == (128, 1, 1)


def test_public_completion_without_route_grant_preserves_native_response(
    monkeypatch,
) -> None:
    from art_vllm_runtime import binary_routes
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.completion import api_router

    async def create_completion(request, _raw_request):
        assert getattr(request, "route_upload", None) is None
        return dedicated_server.JSONResponse(
            content={
                "id": "cmpl-public-test",
                "object": "text_completion",
                "created": 0,
                "model": request.model,
                "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    @contextmanager
    def forbidden_capture():
        raise AssertionError("ordinary public inference must not capture routes")
        yield

    monkeypatch.setattr(
        api_server, "build_app", lambda *args, **kwargs: _fake_vllm_app()
    )
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    monkeypatch.setattr(api_router, "create_completion", create_completion)
    monkeypatch.setattr(binary_routes, "capture_routed_experts", forbidden_capture)
    dedicated_server._patch_art_runtime_routes()
    app = api_server.build_app()

    response = TestClient(app).post(
        "/v1/completions",
        json={"model": "model", "prompt": [1], "max_tokens": 1},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "ok"


def test_public_completion_uploads_routes_and_exposes_event_completion(
    monkeypatch,
) -> None:
    from art_vllm_runtime import binary_routes, patches, route_uploads
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.completion import api_router

    patches.subclass_chat_completion_request()
    uploaded: dict[str, object] = {}

    class Uploader:
        def __init__(self, *, allowed_host_suffixes):
            assert allowed_host_suffixes == ("test.example",)

        async def put(self, grant, chunks, *, actual_bytes):
            uploaded["client_reference"] = grant.client_reference
            uploaded["body"] = b"".join(chunks)
            uploaded["actual_bytes"] = actual_bytes

        async def close(self):
            uploaded["closed"] = True

    async def create_completion(request, _raw_request):
        assert request.route_upload.client_reference == "routes-1"
        return dedicated_server.JSONResponse(
            content={
                "id": "cmpl-public-upload",
                "object": "text_completion",
                "created": 0,
                "model": request.model,
                "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )

    @contextmanager
    def capture_routed_experts():
        routes = binary_routes._CapturedRoutes(num_experts=8, padding_layers=())
        routes[0] = np.zeros((2, 1, 1), dtype=np.uint8)
        yield routes

    monkeypatch.setenv(dedicated_server._ROUTE_UPLOAD_ALLOWED_HOSTS_ENV, "test.example")
    monkeypatch.setitem(dedicated_server._runtime_state, "binary_routed_experts", True)
    monkeypatch.setattr(
        api_server, "build_app", lambda *args, **kwargs: _fake_vllm_app()
    )
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    monkeypatch.setattr(api_router, "create_completion", create_completion)
    monkeypatch.setattr(binary_routes, "capture_routed_experts", capture_routed_experts)
    monkeypatch.setattr(route_uploads, "PresignedPutUploader", Uploader)
    dedicated_server._patch_art_runtime_routes()
    app = api_server.build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "model",
                "prompt": [1],
                "max_tokens": 1,
                "route_upload": {
                    "url": "https://objects.test.example/bucket/routes-1",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(minutes=1)
                    ).isoformat(),
                    "max_bytes": 1024,
                    "client_reference": "routes-1",
                },
            },
        )
        assert response.status_code == 200
        operation = response.json()["route_upload"]
        status = client.get(
            f"/v1/route_uploads/{operation['operation_id']}/wait?timeout=1"
        )
        assert status.json()["state"] == "ready"

    assert uploaded["client_reference"] == "routes-1"
    assert uploaded["actual_bytes"] == len(uploaded["body"])
    assert uploaded["closed"] is True


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


def test_lora_only_runtime_advertises_update_capabilities(monkeypatch) -> None:
    class RuntimeConfigured(Exception):
        pass

    def stop_after_configuration() -> None:
        raise RuntimeConfigured

    monkeypatch.setattr(
        dedicated_server,
        "_configure_index_shared_pp",
        lambda _model, _engine_args: None,
    )
    monkeypatch.setattr(
        dedicated_server,
        "apply_vllm_runtime_patches",
        stop_after_configuration,
    )

    with pytest.raises(RuntimeConfigured):
        dedicated_server.main(
            [
                "--model=model",
                "--port=8000",
                "--cuda-visible-devices=0",
                "--served-model-name=policy",
            ]
        )

    assert dedicated_server._runtime_state["in_flight_lora_updates"] is True
    assert dedicated_server._runtime_state["policy_token_spans"] is True


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
