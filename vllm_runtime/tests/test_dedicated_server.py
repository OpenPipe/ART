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


def _fake_vllm_app(*, authenticated_principal: bool = True) -> FastAPI:
    app = FastAPI()

    if authenticated_principal:

        @app.middleware("http")
        async def inject_authenticated_principal(request, call_next):
            request.scope[dedicated_server._AUTHENTICATED_PRINCIPAL_SCOPE_KEY] = (
                request.headers.get("x-test-authenticated-principal", "test-tenant")
            )
            return await call_next(request)

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


def test_capabilities_expose_versioned_model_backend_contract(monkeypatch) -> None:
    from vllm.entrypoints.openai import api_server

    model_backend = {
        "schema_version": 1,
        "base_model": "Qwen/Qwen3.5-35B-A3B",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "backend": "vllm",
        "backend_version": "0.25.1",
        "validation_status": "validated",
        "lora_implementation": "native",
        "exact_token_ids": True,
        "exact_token_logprobs": True,
        "prompt_policy_spans": True,
        "decode_policy_spans": True,
        "in_flight_lora_updates": True,
        "active_request_kv_continuation": True,
        "new_request_policy_cache_isolation": True,
        "binary_route_capture": True,
        "route_capture_dcp": 1,
        "route_capture_pcp": 1,
    }
    monkeypatch.setitem(dedicated_server._runtime_state, "model_backend", model_backend)
    monkeypatch.setattr(
        dedicated_server,
        "_fast_metrics_url",
        lambda _request: "http://runtime.test/art/metrics",
    )
    monkeypatch.setattr(
        api_server, "build_app", lambda *args, **kwargs: _fake_vllm_app()
    )
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    dedicated_server._patch_art_runtime_routes()

    response = TestClient(api_server.build_app()).get("/art/capabilities")

    assert response.status_code == 200
    assert response.json()["protocol_version"] == 8
    assert response.json()["model_backend"] == model_backend


@pytest.mark.asyncio
async def test_static_token_auth_does_not_invent_route_owner(monkeypatch) -> None:
    observed: dict[str, object] = {}

    async def app(scope, _receive, _send) -> None:
        observed.update(scope)

    monkeypatch.setattr(dedicated_server, "_auth_tokens", ["secret"])
    middleware = dedicated_server._ArtAuthenticationMiddleware(app)
    await middleware(
        {
            "type": "http",
            "method": "GET",
            "path": "/art/state",
            "headers": [(b"authorization", b"Bearer secret")],
        },
        lambda: None,
        lambda _message: None,
    )

    assert observed["path"] == "/art/state"
    assert dedicated_server._AUTHENTICATED_PRINCIPAL_SCOPE_KEY not in observed


@pytest.mark.asyncio
async def test_explicit_local_principal_populates_scope(monkeypatch) -> None:
    observed: dict[str, object] = {}

    async def app(scope, _receive, _send) -> None:
        observed.update(scope)

    monkeypatch.setattr(dedicated_server, "_auth_tokens", [])
    middleware = dedicated_server._ArtLocalRoutePrincipalMiddleware(
        app,
        principal="local-development",
    )
    await middleware(
        {"type": "http", "method": "GET", "path": "/v1/route_uploads/id"},
        lambda: None,
        lambda _message: None,
    )

    assert (
        observed[dedicated_server._AUTHENTICATED_PRINCIPAL_SCOPE_KEY]
        == "local-development"
    )


def test_local_principal_rejects_authenticated_server(monkeypatch) -> None:
    monkeypatch.setattr(dedicated_server, "_auth_tokens", ["secret"])
    monkeypatch.setenv(
        dedicated_server._ROUTE_UPLOAD_LOCAL_PRINCIPAL_ENV,
        "local-development",
    )

    with pytest.raises(ValueError, match="requires unauthenticated local mode"):
        dedicated_server._local_route_upload_principal()


def test_route_owner_is_stable_across_bearer_token_rotation() -> None:
    first = SimpleNamespace(
        scope={dedicated_server._AUTHENTICATED_PRINCIPAL_SCOPE_KEY: "tenant-7"},
        headers={"authorization": "Bearer old-token"},
    )
    rotated = SimpleNamespace(
        scope={dedicated_server._AUTHENTICATED_PRINCIPAL_SCOPE_KEY: "tenant-7"},
        headers={"authorization": "Bearer rotated-token"},
    )

    assert dedicated_server._route_upload_owner(first) == "tenant-7"
    assert dedicated_server._route_upload_owner(rotated) == "tenant-7"


@pytest.mark.asyncio
async def test_art_authentication_explicitly_guards_route_status_paths(
    monkeypatch,
) -> None:
    observed: list[str] = []
    sent: list[dict[str, object]] = []

    async def app(scope, _receive, _send) -> None:
        observed.append(scope["path"])

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    monkeypatch.setattr(dedicated_server, "_auth_tokens", ["secret"])
    middleware = dedicated_server._ArtAuthenticationMiddleware(app)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/route_uploads/operation",
        "headers": [],
    }
    await middleware(scope, receive, send)

    assert observed == []
    assert sent[0]["status"] == 401

    scope["headers"] = [(b"authorization", b"Bearer secret")]
    await middleware(scope, receive, send)
    assert observed == ["/v1/route_uploads/operation"]


@pytest.mark.asyncio
async def test_auth_middleware_passes_lifespan_scope(monkeypatch) -> None:
    observed: list[str] = []

    async def app(scope, _receive, _send) -> None:
        observed.append(scope["type"])

    monkeypatch.setattr(dedicated_server, "_auth_tokens", ["secret"])
    middleware = dedicated_server._ArtAuthenticationMiddleware(app)
    await middleware({"type": "lifespan"}, lambda: None, lambda _message: None)

    assert observed == ["lifespan"]


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


def test_public_chat_without_training_fields_preserves_native_response(
    monkeypatch,
) -> None:
    from art_vllm_runtime import binary_routes
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.chat_completion import api_router

    async def create_chat_completion(request, _raw_request):
        assert getattr(request, "route_upload", None) is None
        assert getattr(request, "return_policy_spans", False) is False
        return dedicated_server.JSONResponse(
            content={
                "id": "chatcmpl-public-test",
                "object": "chat.completion",
                "created": 0,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
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
    monkeypatch.setattr(api_router, "create_chat_completion", create_chat_completion)
    monkeypatch.setattr(binary_routes, "capture_routed_experts", forbidden_capture)
    dedicated_server._patch_art_runtime_routes()
    app = api_server.build_app()

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def test_public_completion_uploads_routes_and_exposes_event_completion(
    monkeypatch,
) -> None:
    from art_vllm_runtime import binary_routes, patches, route_uploads
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.completion import api_router

    patches.subclass_chat_completion_request()
    uploaded: dict[str, object] = {}
    generated = 0
    uploads = 0

    class Uploader:
        def __init__(self, *, allowed_host_suffixes):
            assert allowed_host_suffixes == ("test.example",)

        def validate_admission(self, _grant):
            return None

        async def put(self, grant, chunks, *, actual_bytes):
            nonlocal uploads
            uploads += 1
            uploaded["client_reference"] = grant.client_reference
            uploaded["body"] = b"".join(chunks)
            uploaded["actual_bytes"] = actual_bytes

        async def close(self):
            uploaded["closed"] = True

    async def create_completion(request, _raw_request):
        nonlocal generated
        generated += 1
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
    monkeypatch.setenv(dedicated_server._ROUTE_UPLOAD_TRUSTED_PRINCIPAL_ENV, "1")
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
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "model",
                "prompt": [1],
                "max_tokens": 1,
                "route_upload": {
                    "url": "https://objects.test.example/bucket/routes-1",
                    "expires_at": expires_at,
                    "max_bytes": 1024,
                    "client_reference": "routes-1",
                },
            },
        )
        assert response.status_code == 200
        operation = response.json()["route_upload"]
        replay = client.post(
            "/v1/completions",
            json={
                "model": "model",
                "prompt": [1],
                "max_tokens": 1,
                "route_upload": {
                    "url": "https://objects.test.example/bucket/routes-1",
                    "expires_at": expires_at,
                    "max_bytes": 1024,
                    "client_reference": "routes-1",
                },
            },
        )
        assert replay.status_code == 200
        replayed = replay.json()
        assert replayed["route_upload"]["operation_id"] == operation["operation_id"]
        assert replayed["route_upload"]["client_reference"] == "routes-1"
        assert replayed["choices"] == response.json()["choices"]
        conflict = client.post(
            "/v1/completions",
            json={
                "model": "model",
                "prompt": [2],
                "max_tokens": 1,
                "route_upload": {
                    "url": "https://objects.test.example/bucket/routes-1",
                    "expires_at": expires_at,
                    "max_bytes": 1024,
                    "client_reference": "routes-1",
                },
            },
        )
        assert conflict.status_code == 400
        assert "different inference" in conflict.json()["error"]
        status = client.get(
            f"/v1/route_uploads/{operation['operation_id']}/wait?timeout=1"
        )
        assert status.json()["state"] == "ready"
        foreign_status = client.get(
            f"/v1/route_uploads/{operation['operation_id']}",
            headers={"x-test-authenticated-principal": "other-tenant"},
        )
        assert foreign_status.status_code == 403
        assert "another owner" in foreign_status.json()["error"]

    assert uploaded["client_reference"] == "routes-1"
    assert uploaded["actual_bytes"] == len(uploaded["body"])
    assert uploaded["closed"] is True
    assert generated == 1
    assert uploads == 1


def test_public_chat_route_upload_retry_reuses_generation_and_upload(
    monkeypatch,
) -> None:
    from art_vllm_runtime import binary_routes, patches, route_uploads
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.chat_completion import api_router

    patches.subclass_chat_completion_request()
    generated = 0
    uploads = 0

    class Uploader:
        def __init__(self, *, allowed_host_suffixes):
            assert allowed_host_suffixes == ("test.example",)

        def validate_admission(self, _grant):
            return None

        async def put(self, _grant, chunks, *, actual_bytes):
            nonlocal uploads
            uploads += 1
            assert actual_bytes == sum(map(len, chunks))

        async def close(self):
            pass

    async def create_chat_completion(request, _raw_request):
        nonlocal generated
        generated += 1
        assert request.route_upload.client_reference == "chat-routes-1"
        return dedicated_server.JSONResponse(
            content={
                "id": "chatcmpl-public-upload",
                "object": "chat.completion",
                "created": 0,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
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
    monkeypatch.setenv(dedicated_server._ROUTE_UPLOAD_TRUSTED_PRINCIPAL_ENV, "1")
    monkeypatch.setitem(dedicated_server._runtime_state, "binary_routed_experts", True)
    monkeypatch.setattr(
        api_server, "build_app", lambda *args, **kwargs: _fake_vllm_app()
    )
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    monkeypatch.setattr(api_router, "create_chat_completion", create_chat_completion)
    monkeypatch.setattr(binary_routes, "capture_routed_experts", capture_routed_experts)
    monkeypatch.setattr(route_uploads, "PresignedPutUploader", Uploader)
    dedicated_server._patch_art_runtime_routes()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1,
        "route_upload": {
            "url": "https://objects.test.example/bucket/chat-routes-1",
            "expires_at": expires_at,
            "max_bytes": 1024,
            "client_reference": "chat-routes-1",
        },
    }

    with TestClient(api_server.build_app()) as client:
        response = client.post("/v1/chat/completions", json=request)
        replay = client.post("/v1/chat/completions", json=request)
        operation = response.json()["route_upload"]
        ready = client.get(
            f"/v1/route_uploads/{operation['operation_id']}/wait?timeout=1"
        )

    assert response.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == response.json()
    assert ready.json()["state"] == "ready"
    assert generated == 1
    assert uploads == 1


def test_route_upload_requires_trusted_authenticated_principal(monkeypatch) -> None:
    from art_vllm_runtime import patches, route_uploads
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.completion import api_router

    patches.subclass_chat_completion_request()

    class Uploader:
        def __init__(self, *, allowed_host_suffixes):
            pass

        def validate_admission(self, _grant):
            return None

        async def close(self):
            pass

    async def forbidden_completion(_request, _raw_request):
        raise AssertionError(
            "unauthenticated route capture must fail before generation"
        )

    monkeypatch.setenv(dedicated_server._ROUTE_UPLOAD_ALLOWED_HOSTS_ENV, "test.example")
    monkeypatch.setenv(dedicated_server._ROUTE_UPLOAD_TRUSTED_PRINCIPAL_ENV, "1")
    monkeypatch.setitem(dedicated_server._runtime_state, "binary_routed_experts", True)
    monkeypatch.setattr(
        api_server,
        "build_app",
        lambda *args, **kwargs: _fake_vllm_app(authenticated_principal=False),
    )
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    monkeypatch.setattr(api_router, "create_completion", forbidden_completion)
    monkeypatch.setattr(route_uploads, "PresignedPutUploader", Uploader)
    dedicated_server._patch_art_runtime_routes()

    with TestClient(api_server.build_app()) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "model",
                "prompt": [1],
                "max_tokens": 1,
                "route_upload": {
                    "url": "https://objects.test.example/bucket/routes",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(minutes=1)
                    ).isoformat(),
                    "max_bytes": 1024,
                    "client_reference": "routes",
                },
            },
        )

    assert response.status_code == 403
    assert "trusted authenticated principal" in response.json()["error"]


def test_route_upload_capability_requires_configured_principal_provider(
    monkeypatch,
) -> None:
    from vllm.entrypoints.openai import api_server

    monkeypatch.setenv(dedicated_server._ROUTE_UPLOAD_ALLOWED_HOSTS_ENV, "test.example")
    monkeypatch.delenv(
        dedicated_server._ROUTE_UPLOAD_TRUSTED_PRINCIPAL_ENV, raising=False
    )
    monkeypatch.setitem(dedicated_server._runtime_state, "binary_routed_experts", True)
    monkeypatch.setattr(
        dedicated_server,
        "_fast_metrics_url",
        lambda _request: "http://runtime.test/art/metrics",
    )
    monkeypatch.setattr(
        api_server, "build_app", lambda *args, **kwargs: _fake_vllm_app()
    )
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    dedicated_server._patch_art_runtime_routes()

    with TestClient(api_server.build_app()) as client:
        capabilities = client.get("/art/capabilities")

    assert capabilities.status_code == 200
    assert capabilities.json()["presigned_route_uploads"] is False


def test_explicit_local_principal_enables_owner_scoped_route_status(
    monkeypatch,
) -> None:
    from art_vllm_runtime import route_uploads
    from vllm.entrypoints.openai import api_server

    class Uploader:
        def __init__(self, *, allowed_host_suffixes):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(dedicated_server, "_auth_tokens", [])
    monkeypatch.setenv(dedicated_server._ROUTE_UPLOAD_ALLOWED_HOSTS_ENV, "test.example")
    monkeypatch.setenv(
        dedicated_server._ROUTE_UPLOAD_LOCAL_PRINCIPAL_ENV,
        "local-development",
    )
    monkeypatch.delenv(
        dedicated_server._ROUTE_UPLOAD_TRUSTED_PRINCIPAL_ENV,
        raising=False,
    )
    monkeypatch.setitem(dedicated_server._runtime_state, "binary_routed_experts", True)
    monkeypatch.setattr(
        dedicated_server,
        "_fast_metrics_url",
        lambda _request: "http://runtime.test/art/metrics",
    )
    monkeypatch.setattr(
        api_server,
        "build_app",
        lambda *args, **kwargs: _fake_vllm_app(authenticated_principal=False),
    )
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    monkeypatch.setattr(route_uploads, "PresignedPutUploader", Uploader)
    dedicated_server._patch_art_runtime_routes()

    with TestClient(api_server.build_app()) as client:
        capabilities = client.get("/art/capabilities")
        missing = client.get("/v1/route_uploads/missing")

    assert capabilities.status_code == 200
    assert capabilities.json()["presigned_route_uploads"] is True
    assert missing.status_code == 404
    assert "not found" in missing.json()["error"]


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


def test_lora_update_identity_separates_policy_generation_from_source() -> None:
    first = dedicated_server._InFlightLoraUpdateRequest(
        model_name="run:active",
        lora_slot="run:active",
        lora_path="/holder-a/materialized",
        operation_id="operation-1",
        adapter_source="caios://bucket/immutable-version-1",
        generation_id="policy-generation-1",
        expected_generation_id=None,
        policy_version=1,
    )
    prior_response = {"status": "updated", "generation_id": first.generation_id}
    applied = dedicated_server._AppliedLoraUpdate(
        identity=dedicated_server._lora_update_identity(first),
        response=prior_response,
    )

    # A holder-local materialization path is not transport or policy identity.
    replay = first.model_copy(update={"lora_path": "/holder-b/materialized"})
    assert (
        dedicated_server._admit_lora_update(
            replay,
            applied,
            launch_policy_version=None,
            launch_generation_id=None,
        )
        == prior_response
    )

    changed_source = replay.model_copy(
        update={"adapter_source": "caios://bucket/immutable-version-2"}
    )
    with pytest.raises(ValueError, match="operation identity was reused"):
        dedicated_server._admit_lora_update(
            changed_source,
            applied,
            launch_policy_version=None,
            launch_generation_id=None,
        )

    next_update = first.model_copy(
        update={
            "operation_id": "operation-2",
            "adapter_source": "caios://bucket/immutable-version-2",
            "generation_id": "policy-generation-2",
            "expected_generation_id": "policy-generation-1",
            "policy_version": 2,
        }
    )
    assert (
        dedicated_server._admit_lora_update(
            next_update,
            applied,
            launch_policy_version=None,
            launch_generation_id=None,
        )
        is None
    )
    with pytest.raises(ValueError, match="expected generation"):
        dedicated_server._admit_lora_update(
            next_update.model_copy(update={"expected_generation_id": "stale"}),
            applied,
            launch_policy_version=None,
            launch_generation_id=None,
        )


def test_lora_update_checks_launch_generation_lineage() -> None:
    request = dedicated_server._InFlightLoraUpdateRequest(
        model_name="run:active",
        lora_path="/holder/materialized",
        operation_id="operation-2",
        adapter_source="caios://bucket/immutable-version-2",
        generation_id="policy-generation-2",
        expected_generation_id="policy-generation-1",
        policy_version=2,
    )
    assert (
        dedicated_server._admit_lora_update(
            request,
            None,
            launch_policy_version=1,
            launch_generation_id="policy-generation-1",
        )
        is None
    )
    with pytest.raises(ValueError, match="launch state"):
        dedicated_server._admit_lora_update(
            request.model_copy(update={"expected_generation_id": "wrong"}),
            None,
            launch_policy_version=1,
            launch_generation_id="policy-generation-1",
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
