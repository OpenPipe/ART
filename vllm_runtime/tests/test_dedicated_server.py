from http.client import HTTPConnection
import json
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from art_vllm_runtime import dedicated_server
from art_vllm_runtime.fast_metrics import FAST_METRIC_NAMES, FastMetricsSidecar
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.datastructures import URL

from art.serving_capabilities import PairedInferenceEndpoint, ServingProfile

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
    assert os.getsid(sidecar.process.pid) == sidecar.process.pid
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


def test_lora_update_accepts_only_a_bounded_staged_descriptor() -> None:
    request = dedicated_server._InFlightLoraUpdateRequest.model_validate(
        {
            "operation_id": "operation-1",
            "model_name": "policy:active",
            "generation_id": "generation-2",
            "expected_generation_id": "generation-1",
            "policy_version": 2,
            "lora_slot": "policy:active",
            "source": {
                "path": "/adapter/generation-2",
                "source_identity": "sha256:" + "a" * 64,
                "layout": "peft_safetensors_v1",
                "model_bytes": 4096,
                "config_bytes": 512,
            },
        }
    )

    assert request.source.path == "/adapter/generation-2"
    assert request.source.total_bytes == 4608
    with pytest.raises(ValueError, match="runtime byte bound"):
        dedicated_server.StagedLoraDescriptor.model_validate(
            {**request.source.model_dump(), "model_bytes": 9 << 30}
        )


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


@pytest.mark.asyncio
async def test_serving_profile_reports_resolved_runtime_geometry(monkeypatch) -> None:
    identity = {
        "base_model": "test/model",
        "model_identifier": "test/model",
        "model_revision": "default",
        "model_support_key": "test-support",
        "handler_name": "test-handler",
        "lora_rank": 32,
        "lora_alpha": 32.0,
        "lora_target_modules": ["q_proj", "v_proj"],
        "trainer_dtype": "bfloat16",
        "route_replay": True,
        "paired_transfer": {
            "lora_backend": "nixl",
            "trainer_endpoints": [
                {"host_id": "trainer", "domain": "trainer", "root": "/dev/shm"}
            ],
            "inference_endpoints": [
                {
                    "host_id": "inference",
                    "domain": "inference",
                    "root": "/dev/shm",
                }
            ],
            "lora_source_host_id": "trainer",
            "route_source": {
                "host_id": "trainer",
                "domain": "trainer",
                "root": "/dev/shm",
            },
            "route_delivery": "local",
        },
        "lora_transport": "nixl",
        "retained_route_transport": "holder_local",
        "retained_route_delivery": "local",
        "retained_route_max_bytes": 4096,
        "retained_route_max_bundles": 4,
    }
    monkeypatch.setitem(
        dedicated_server._runtime_state, "serving_profile_identity", identity
    )
    monkeypatch.setitem(dedicated_server._runtime_state, "route_capture", True)
    monkeypatch.setitem(dedicated_server._runtime_state, "runtime_model", "test/model")
    monkeypatch.setitem(dedicated_server._runtime_state, "runtime_revision", None)
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(
                return_value={
                    "kv_block_size": 16,
                    "kv_block_bytes_per_rank": 65_536,
                    "kv_capacity_blocks_per_rank": 4096,
                    "kv_capacity_bytes_per_rank": 268_435_456,
                }
            )
        ),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                model="/offline/cache/models--test--model/snapshots/revision",
                revision=None,
                hf_config=SimpleNamespace(
                    to_dict=lambda: {
                        "architectures": ["TestModel"],
                        "num_hidden_layers": 24,
                    }
                ),
                hf_text_config=SimpleNamespace(num_hidden_layers=24),
                tokenizer="test/model",
                tokenizer_revision=None,
                dtype="bfloat16",
                quantization="fp8",
                max_model_len=16_384,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=8,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                prefill_context_parallel_size=1,
                enable_expert_parallel=True,
            ),
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=32_768,
                max_num_seqs=64,
                max_num_partial_prefills=2,
            ),
            cache_config=SimpleNamespace(
                cache_dtype="fp8",
                block_size=16,
                enable_prefix_caching=True,
                prefix_caching_hash_algo="sha256_cbor",
            ),
            lora_config=SimpleNamespace(
                max_loras=2,
                max_lora_rank=32,
                lora_dtype="bfloat16",
            ),
            speculative_config=SimpleNamespace(method="mtp"),
        ),
    )

    profile = ServingProfile.model_validate(
        await dedicated_server._serving_profile(engine)
    )

    assert profile.identity.model_identifier == "test/model"
    assert profile.architecture.loaded_layer_count == 24
    assert profile.architecture.handler_name == "test-handler"
    assert profile.architecture.expert_parallel_size == 8
    assert profile.runtime_model == "test/model"
    assert profile.tensor_parallel_size == 8
    assert profile.quantization == "fp8"
    assert profile.max_lora_rank == profile.identity.lora_rank == 32
    assert profile.multi_token_prediction
    assert profile.kv_block_bytes_per_rank == 65_536
    assert profile.kv_capacity_bytes_per_rank == 268_435_456
    assert profile.route_capture_format == "art_inference_route_bundle_v1"
    endpoint = PairedInferenceEndpoint(
        url="http://127.0.0.1:8000/art/internal/v1/chat/completions",
        target_id="a" * 64,
        runtime_generation=1,
        runtime_source_id="source",
        runtime_source_epoch=1,
        authorization_token="secret" * 6,
        profile=profile,
    )
    headers = endpoint.request_headers(
        request_identity="b" * 64,
        cache_identity="c" * 64,
        tenant_id="tenant",
        run_id="run",
        service_tier="standard",
        route_capture_max_bytes=2048,
    )
    assert headers["x-art-route-capture"] == "retained"
    assert headers["x-art-route-max-bytes"] == "2048"


def test_private_dispatch_uses_distinct_auth_and_fences_runtime_target(
    monkeypatch,
) -> None:
    from starlette.requests import Request

    monkeypatch.setattr(dedicated_server, "_auth_tokens", ["public-token"])
    monkeypatch.setattr(dedicated_server, "_private_dispatch_token", "p" * 32)
    monkeypatch.setattr(dedicated_server, "_runtime_target_id", "a" * 64)
    monkeypatch.setitem(dedicated_server._runtime_state, "route_capture", True)
    monkeypatch.setitem(
        dedicated_server._runtime_state,
        "serving_profile_identity",
        {
            "retained_route_transport": "holder_local",
            "retained_route_max_bytes": 4096,
            "retained_route_max_bundles": 4,
        },
    )

    app = FastAPI()
    app.get("/v1/models")(lambda: {"ok": True})
    app.get("/art/state")(lambda: {"ok": True})
    app.post(dedicated_server._PRIVATE_DISPATCH_PATH)(lambda: {"ok": True})
    middleware = dedicated_server._ArtAuthenticationMiddleware(app)

    async def authenticated_app(scope, receive, send):
        await middleware(scope, receive, send)

    client = TestClient(authenticated_app)
    assert client.get("/v1/models").status_code == 401
    assert (
        client.get(
            "/v1/models", headers={"Authorization": "Bearer public-token"}
        ).status_code
        == 200
    )
    assert client.get("/art/state").status_code == 401
    assert (
        client.get(
            "/art/state", headers={"Authorization": "Bearer public-token"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            dedicated_server._PRIVATE_DISPATCH_PATH,
            headers={"Authorization": "Bearer public-token"},
        ).status_code
        == 401
    )
    assert (
        client.post(
            dedicated_server._PRIVATE_DISPATCH_PATH,
            headers={"Authorization": "Bearer " + "p" * 32},
        ).status_code
        == 200
    )

    scope = {
        "type": "http",
        "method": "POST",
        "path": dedicated_server._PRIVATE_DISPATCH_PATH,
        "headers": [
            (b"authorization", b"Bearer " + b"p" * 32),
            (b"x-art-runtime-target", b"a" * 64),
            (b"x-art-request-identity", b"b" * 64),
            (b"x-art-cache-identity", b"c" * 64),
            (b"x-art-tenant-id", b"tenant"),
            (b"x-art-run-id", b"run"),
            (b"x-art-service-tier", b"standard"),
            (b"x-art-route-capture", b"retained"),
            (b"x-art-route-max-bytes", b"2048"),
        ],
    }
    request = Request(scope)

    assert dedicated_server._verify_bearer(request.headers, "p" * 32)
    assert dedicated_server._private_dispatch_context(request) == (
        "b" * 64,
        "c" * 64,
        "tenant",
        "run",
        "standard",
    )
    assert dedicated_server._private_route_capture_max_bytes(request) == 2048
    scope["headers"][1] = (b"x-art-runtime-target", b"d" * 64)
    stale = dedicated_server._private_dispatch_context(Request(scope))
    assert stale.status_code == 409
    assert b'"execution":"not_started"' in stale.body
    stale_receipt = dedicated_server._private_runtime_target_error(
        Request(scope), execution="unknown"
    )
    assert stale_receipt is not None
    assert b'"execution":"unknown"' in stale_receipt.body


@pytest.mark.asyncio
async def test_private_execution_receipts_are_bounded_and_fail_closed() -> None:
    receipts = dedicated_server._PrivateExecutionReceipts(capacity=2)

    assert await receipts.claim("a" * 64, "payload-a") is None
    duplicate = await receipts.claim("a" * 64, "payload-a")
    assert duplicate is not None
    assert duplicate.execution == "started"
    assert duplicate.fingerprint == "payload-a"

    await receipts.settle("a" * 64, "payload-a", "completed")
    assert await receipts.claim("b" * 64, "payload-b") is None
    assert await receipts.claim("c" * 64, "payload-c") is None
    assert await receipts.get("a" * 64) is None
    assert (await receipts.get("b" * 64)).execution == "started"  # type: ignore[union-attr]

    with pytest.raises(RuntimeError, match="capacity exhausted"):
        await receipts.claim("d" * 64, "payload-d")


@pytest.mark.asyncio
async def test_completed_lora_update_replays_before_generation_validation(
    monkeypatch,
) -> None:
    receipts = dedicated_server._CompletedLoraUpdates(capacity=2)
    monkeypatch.setattr(dedicated_server, "_completed_lora_updates", receipts)
    body = dedicated_server._InFlightLoraUpdateRequest(
        operation_id="operation-1",
        model_name="run:active",
        lora_slot="run:active",
        source={
            "path": "/adapter/generation-1",
            "source_identity": "manifest:" + "a" * 64,
            "layout": "peft_safetensors_v1",
            "model_bytes": 4096,
            "config_bytes": 128,
        },
        generation_id="generation-1",
        expected_generation_id="generation-1",
        policy_version=1,
    )
    fingerprint = dedicated_server._lora_update_fingerprint(body)
    result = {
        "status": "updated",
        "generation_id": body.generation_id,
        "update_seq": 1,
    }
    await receipts.settle(body.operation_id, fingerprint, result)

    replay = await dedicated_server._replay_lora_update(body, fingerprint)
    assert replay is not None
    assert replay.status_code == 200
    assert json.loads(replay.body) == result

    changed = body.model_copy(update={"policy_version": 2})
    conflict = await dedicated_server._replay_lora_update(
        changed, dedicated_server._lora_update_fingerprint(changed)
    )
    assert conflict is not None
    assert conflict.status_code == 409

    await receipts.settle("operation-2", "fingerprint-2", {"status": "updated"})
    await receipts.settle("operation-3", "fingerprint-3", {"status": "updated"})
    assert await receipts.replay(body.operation_id, fingerprint) is None


def test_lora_staging_precedes_target_admission_gate(monkeypatch) -> None:
    from art_vllm_runtime import policy_spans
    from vllm.entrypoints.openai import api_server

    events: list[str] = []

    class Gate:
        entered = False

        async def __aenter__(self):
            self.entered = True
            events.append("gate_enter")

        async def __aexit__(self, *_args):
            events.append("gate_exit")
            self.entered = False

    gate = Gate()
    current = policy_spans.PolicyLoRARequest(
        lora_name="run:active",
        lora_int_id=7,
        lora_path="/adapter/generation-1",
        load_inplace=False,
        generation_id="generation-1",
        policy_version=1,
        update_seq=1,
    )

    class Models:
        lora_requests = {"run:active": current}
        lora_resolver_lock = {"run:active": gate}

        async def _check_load_lora_adapter_request(self, _request):
            assert gate.entered
            return None

        def is_base_model(self, _model_name):
            return False

    class EngineCore:
        async def call_utility_async(self, method, *_args):
            if method == "art_prepare_staged_lora_policy":
                assert not gate.entered
                events.append("prepare")
                return {"workers": 1, "ready": True}
            if method == "art_commit_staged_lora_policy_update":
                assert gate.entered
                events.append("commit")
                return {"workers": 1, "cache_transition": {}}
            raise AssertionError(method)

    class Coordinator:
        async def preflight_publication(self, *_args, **_kwargs):
            assert gate.entered

        async def begin_publication(self, *_args, **_kwargs):
            assert gate.entered
            return 2

        async def commit_update(self, *_args, **_kwargs):
            assert gate.entered
            events.append("settle")

    monkeypatch.setattr(api_server, "build_app", lambda *args, **kwargs: FastAPI())
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    monkeypatch.setattr(
        policy_spans, "lora_update_coordinator", lambda *_args: Coordinator()
    )
    monkeypatch.setattr(
        policy_spans, "register_lora_alias", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        policy_spans, "publish_lora_slot_policy", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        dedicated_server,
        "_completed_lora_updates",
        dedicated_server._CompletedLoraUpdates(capacity=2),
    )
    dedicated_server._patch_art_runtime_routes()
    app = api_server.build_app()
    app.state.openai_serving_models = Models()
    app.state.engine_client = SimpleNamespace(engine_core=EngineCore())

    response = TestClient(app).post(
        "/art/in_flight_lora_update",
        json={
            "operation_id": "operation-2",
            "model_name": "run:active",
            "lora_slot": "run:active",
            "generation_id": "generation-2",
            "expected_generation_id": "generation-1",
            "policy_version": 2,
            "source": {
                "path": "/adapter/generation-2",
                "source_identity": "manifest:" + "a" * 64,
                "layout": "peft_safetensors_v1",
                "model_bytes": 4096,
                "config_bytes": 128,
            },
        },
    )

    assert response.status_code == 200
    assert events == ["prepare", "gate_enter", "commit", "settle", "gate_exit"]


@pytest.mark.asyncio
async def test_private_route_responses_reserve_replay_ack_and_discard_exact_bytes(
    monkeypatch,
) -> None:
    responses = dedicated_server._PrivateRouteResponses()
    identity = "a" * 64
    object_ref = {"store": "holder_local", "locator": "/route"}
    released: list[dict[str, object]] = []
    await responses.reserve(
        identity,
        "payload-a",
        8,
        capacity_bytes=8,
        capacity_bundles=1,
    )
    with pytest.raises(RuntimeError, match="capacity is exhausted"):
        await responses.reserve(
            "b" * 64,
            "payload-b",
            1,
            capacity_bytes=8,
            capacity_bundles=1,
        )
    await responses.complete(
        identity,
        "payload-a",
        b"response",
        retained_bytes=5,
        object_ref=object_ref,
    )
    replay = await responses.replay(identity, "payload-a")
    assert replay is not None and replay.body == b"response"
    with pytest.raises(RuntimeError, match="identity was reused"):
        await responses.replay(identity, "different")
    assert await responses.state() == {
        "active_route_reservations": 0,
        "retained_route_responses": 1,
        "reserved_route_bytes": 8,
        "retained_route_bytes": 5,
    }
    monkeypatch.setattr(dedicated_server, "_local_route_store", None)
    with pytest.raises(RuntimeError, match="local route store is unavailable"):
        await responses.acknowledge(identity)
    monkeypatch.setattr(
        dedicated_server,
        "_local_route_store",
        SimpleNamespace(discard=released.append),
    )
    assert await responses.acknowledge(identity) == 5
    assert released == [object_ref]
    assert await responses.acknowledge(identity) is None
    await responses.reserve(
        "b" * 64,
        "payload",
        8,
        capacity_bytes=8,
        capacity_bundles=1,
    )
    await responses.complete(
        "b" * 64,
        "payload",
        b"response",
        retained_bytes=5,
        object_ref=object_ref,
    )
    assert await responses.discard("b" * 64) == 5
    assert released == [object_ref, object_ref]
    assert await responses.discard("b" * 64) is None
    assert await responses.state() == {
        "active_route_reservations": 0,
        "retained_route_responses": 0,
        "reserved_route_bytes": 0,
        "retained_route_bytes": 0,
    }


@pytest.mark.asyncio
async def test_route_capture_failure_is_ambiguous_and_releases_bytes(
    monkeypatch,
) -> None:
    responses = dedicated_server._PrivateRouteResponses()
    receipts = dedicated_server._PrivateExecutionReceipts(capacity=1)
    identity = "a" * 64
    fingerprint = "payload"
    monkeypatch.setattr(dedicated_server, "_private_route_responses", responses)
    monkeypatch.setattr(dedicated_server, "_private_execution_receipts", receipts)
    await responses.reserve(
        identity, fingerprint, 8, capacity_bytes=8, capacity_bundles=1
    )
    assert await receipts.claim(identity, fingerprint) is None

    response = await dedicated_server._private_route_capture_failed(
        identity, fingerprint, "missing routes"
    )

    assert response.status_code == 500
    assert json.loads(response.body) == {
        "error": "missing routes",
        "type": "route_capture_incomplete",
        "execution": "ambiguous",
    }
    receipt = await receipts.get(identity)
    assert receipt is not None
    assert receipt.execution == "ambiguous"
    assert await responses.state() == {
        "active_route_reservations": 0,
        "retained_route_responses": 0,
        "reserved_route_bytes": 0,
        "retained_route_bytes": 0,
    }


def test_private_request_fingerprint_uses_authenticated_request_identity() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8,
    }
    requests = [ChatCompletionRequest.model_validate(payload) for _ in range(2)]
    assert requests[0].request_id != requests[1].request_id
    for request in requests:
        request.request_id = "a" * 64
        request.cache_salt = "art-private-cache-v1:" + "b" * 64
    assert dedicated_server._private_request_fingerprint(
        requests[0]
    ) == dedicated_server._private_request_fingerprint(requests[1])
    assert dedicated_server._private_request_fingerprint(
        requests[0], route_capture_max_bytes=4096
    ) != dedicated_server._private_request_fingerprint(requests[0])


@pytest.mark.asyncio
async def test_private_stream_receipt_is_completed_only_at_terminal_chunk(
    monkeypatch,
) -> None:
    receipts = dedicated_server._PrivateExecutionReceipts(capacity=2)
    monkeypatch.setattr(dedicated_server, "_private_execution_receipts", receipts)
    identity = "a" * 64
    fingerprint = "payload"
    assert await receipts.claim(identity, fingerprint) is None

    async def chunks():
        yield b"first"
        yield b"second"

    observed = []
    async for chunk in dedicated_server._track_private_stream(
        identity, fingerprint, chunks()
    ):
        observed.append(chunk)
        assert (await receipts.get(identity)).execution == "started"  # type: ignore[union-attr]

    assert observed == [b"first", b"second"]
    assert (await receipts.get(identity)).execution == "completed"  # type: ignore[union-attr]


def test_runtime_sleep_route_returns_engine_validation_error(monkeypatch) -> None:
    from vllm.entrypoints.openai import api_server

    monkeypatch.setattr(api_server, "build_app", lambda *args, **kwargs: FastAPI())
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    dedicated_server._patch_art_runtime_routes()
    app = api_server.build_app()
    assert {
        "/art/internal/v1/chat/completions",
        "/art/internal/v1/completions",
        "/art/internal/v1/requests/{request_identity}/routes:discard",
    } <= {route.path for route in app.routes}

    class Engine:
        async def sleep(self, *, level: int, mode: str) -> None:
            raise ValueError(f"invalid {level=} {mode=}")

    app.state.engine_client = Engine()
    response = TestClient(app).post("/sleep?level=1&mode=wait")
    assert response.status_code == 400
    assert response.json() == {"error": "invalid level=1 mode='wait'"}


def test_private_completion_preserves_exact_token_request_and_logprobs(
    monkeypatch,
) -> None:
    from art_vllm_runtime.runtime_usage import RuntimeUsageJournal
    from fastapi.responses import JSONResponse
    from vllm.entrypoints.openai import api_server
    from vllm.entrypoints.openai.completion import api_router as completion_router

    async def exact_completion(request, _raw_request):
        assert request.prompt == [11, 12]
        assert request.max_tokens == 2
        assert request.logprobs == 2
        assert request.return_token_ids is True
        return JSONResponse(
            {
                "id": "cmpl-exact",
                "object": "text_completion",
                "created": 0,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "finish_reason": "length",
                        "prompt_token_ids": [11, 12],
                        "token_ids": [13, 14],
                        "logprobs": {"token_logprobs": [-0.1, -0.2]},
                    }
                ],
            }
        )

    monkeypatch.setattr(api_server, "build_app", lambda *args, **kwargs: FastAPI())
    monkeypatch.setattr(api_server, "_art_runtime_routes_patched", False, raising=False)
    monkeypatch.setattr(completion_router, "create_completion", exact_completion)
    monkeypatch.setattr(dedicated_server, "_runtime_target_id", "a" * 64)
    receipts = dedicated_server._PrivateExecutionReceipts(capacity=2)
    monkeypatch.setattr(dedicated_server, "_private_execution_receipts", receipts)
    journal = RuntimeUsageJournal("runtime-source", 1, capacity=2)
    monkeypatch.setattr(dedicated_server, "runtime_usage_journal", lambda: journal)
    dedicated_server._patch_art_runtime_routes()
    app = api_server.build_app()

    client = TestClient(app)
    response = client.post(
        "/art/internal/v1/completions",
        headers={
            "x-art-runtime-target": "a" * 64,
            "x-art-request-identity": "b" * 64,
            "x-art-cache-identity": "c" * 64,
            "x-art-tenant-id": "tenant-1",
            "x-art-run-id": "run-1",
            "x-art-service-tier": "standard",
        },
        json={
            "model": "runtime-model",
            "prompt": [11, 12],
            "max_tokens": 2,
            "logprobs": 2,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "add_special_tokens": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0] == {
        "index": 0,
        "text": "",
        "finish_reason": "length",
        "prompt_token_ids": [11, 12],
        "token_ids": [13, 14],
        "logprobs": {"token_logprobs": [-0.1, -0.2]},
    }
    receipt = client.get(
        f"/art/internal/v1/requests/{'b' * 64}",
        headers={"x-art-runtime-target": "a" * 64},
    )
    assert receipt.status_code == 200
    assert receipt.json()["execution"] == "completed"
    client.close()
