import asyncio
from http.client import HTTPConnection
import json
import os
from types import SimpleNamespace

from art_vllm_runtime import dedicated_server
from art_vllm_runtime.fast_metrics import FAST_METRIC_NAMES, FastMetricsSidecar
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
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
async def test_completed_lora_update_replays_without_another_mutation(
    monkeypatch,
) -> None:
    receipts = dedicated_server._CompletedLoraUpdates(capacity=2)
    monkeypatch.setattr(dedicated_server, "_completed_lora_updates", receipts)
    body = dedicated_server._InFlightLoraUpdateRequest(
        operation_id="operation-1",
        model_name="run:active",
        lora_slot="run:active",
        lora_path="/adapter/generation-2",
        generation_id="generation-2",
        expected_generation_id="generation-1",
        policy_version=2,
    )
    fingerprint = dedicated_server._lora_update_fingerprint(body)
    result = {
        "status": "updated",
        "generation_id": body.generation_id,
        "update_seq": 1,
    }
    owner, _ = await receipts.reserve(body.operation_id, fingerprint)
    assert owner
    await receipts.settle(body.operation_id, fingerprint, result)

    owns_replay, replay = await dedicated_server._reserve_lora_update(body, fingerprint)
    assert not owns_replay
    assert replay is not None
    assert replay.status_code == 200
    assert json.loads(replay.body) == result

    changed = body.model_copy(update={"policy_version": 3})
    owns_conflict, conflict = await dedicated_server._reserve_lora_update(
        changed, dedicated_server._lora_update_fingerprint(changed)
    )
    assert not owns_conflict
    assert conflict is not None
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_active_lora_operation_rejects_changed_contents_and_joins_retry() -> None:
    receipts = dedicated_server._CompletedLoraUpdates(capacity=2)
    owner, completion = await receipts.reserve("operation", "fingerprint")
    assert owner

    retry_owner, retry_completion = await receipts.reserve("operation", "fingerprint")
    assert not retry_owner
    assert retry_completion is completion
    with pytest.raises(ValueError, match="identity changed"):
        await receipts.reserve("operation", "different")

    result = {"status": "updated"}
    await receipts.settle("operation", "fingerprint", result)
    assert await retry_completion == result


def test_lora_preparation_precedes_admission_and_atomic_commit(monkeypatch) -> None:
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
            if method == "art_prepare_lora_policy":
                assert not gate.entered
                events.append("prepare")
                return {"workers": 1, "ready": True}
            if method == "art_commit_prepared_lora_policy_update":
                assert gate.entered
                events.append("commit")
                return {"workers": 1, "cache_transition": {}}
            raise AssertionError(method)

    class Coordinator:
        async def begin_update(self, *_args, **_kwargs):
            assert gate.entered
            return 2

        async def commit_update(self, *_args, **_kwargs):
            assert gate.entered
            events.append("publish")

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
            "lora_path": "/adapter/generation-2",
            "generation_id": "generation-2",
            "expected_generation_id": "generation-1",
            "policy_version": 2,
        },
    )

    assert response.status_code == 200
    assert events == ["prepare", "gate_enter", "commit", "publish", "gate_exit"]

    replay = TestClient(app).post(
        "/art/in_flight_lora_update",
        json={
            "operation_id": "operation-2",
            "model_name": "run:active",
            "lora_slot": "run:active",
            "lora_path": "/adapter/generation-2",
            "generation_id": "generation-2",
            "expected_generation_id": "generation-1",
            "policy_version": 2,
        },
    )
    assert replay.json() == response.json()
    assert events == ["prepare", "gate_enter", "commit", "publish", "gate_exit"]


@pytest.mark.asyncio
async def test_different_slots_prepare_concurrently_and_commit_serially(
    monkeypatch,
) -> None:
    from art_vllm_runtime import policy_spans
    from vllm.entrypoints.openai import api_server

    slots = ("slot-a", "slot-b")
    current = {
        slot: policy_spans.PolicyLoRARequest(
            lora_name=slot,
            lora_int_id=index,
            lora_path=f"/{slot}-1",
            generation_id=f"{slot}-1",
            policy_version=1,
            update_seq=1,
        )
        for index, slot in enumerate(slots, start=1)
    }

    class Models:
        lora_requests = dict(current)
        lora_resolver_lock = {slot: asyncio.Lock() for slot in slots}

        async def _check_load_lora_adapter_request(self, _request):
            return None

        def is_base_model(self, _model_name):
            return False

    class EngineCore:
        def __init__(self) -> None:
            self.commit_lock = asyncio.Lock()
            self.preparing = 0
            self.max_preparing = 0
            self.committing = 0
            self.max_committing = 0

        async def call_utility_async(self, method, *_args):
            if method == "art_prepare_lora_policy":
                self.preparing += 1
                self.max_preparing = max(self.max_preparing, self.preparing)
                await asyncio.sleep(0.01)
                self.preparing -= 1
                return {"workers": 1, "ready": True}
            if method == "art_commit_prepared_lora_policy_update":
                async with self.commit_lock:
                    self.committing += 1
                    self.max_committing = max(self.max_committing, self.committing)
                    await asyncio.sleep(0.01)
                    self.committing -= 1
                    return {"workers": 1, "cache_transition": {}}
            raise AssertionError(method)

    class Coordinator:
        async def begin_update(self, slot, **_kwargs):
            return 2

        async def commit_update(self, *_args, **_kwargs):
            return None

    engine_core = EngineCore()
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
        dedicated_server._CompletedLoraUpdates(capacity=4),
    )
    dedicated_server._patch_art_runtime_routes()
    app = api_server.build_app()
    app.state.openai_serving_models = Models()
    app.state.engine_client = SimpleNamespace(engine_core=engine_core)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            *(
                client.post(
                    "/art/in_flight_lora_update",
                    json={
                        "operation_id": f"operation-{slot}",
                        "model_name": slot,
                        "lora_slot": slot,
                        "lora_path": f"/{slot}-2",
                        "generation_id": f"{slot}-2",
                        "expected_generation_id": f"{slot}-1",
                        "policy_version": 2,
                    },
                )
                for slot in slots
            )
        )

    assert [response.status_code for response in responses] == [200, 200]
    assert engine_core.max_preparing == 2
    assert engine_core.max_committing == 1
    assert {
        slot: app.state.openai_serving_models.lora_requests[slot].generation_id
        for slot in slots
    } == {"slot-a": "slot-a-2", "slot-b": "slot-b-2"}
