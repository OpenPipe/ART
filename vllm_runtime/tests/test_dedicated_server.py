import asyncio
from http.client import HTTPConnection
import json
import threading
from types import SimpleNamespace

from art_vllm_runtime import dedicated_server
import pytest
from starlette.datastructures import URL

_PAYLOAD: dict[str, object] = {
    "schema_version": 1,
    "source": "art_vllm_runtime",
    "last_update_unix_s": 1.0,
    "record_count": 1,
    "engine_count": 1,
    "metrics": {"num_requests_running": 2.0, "prompt_tokens_total": 3.0},
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


def test_fast_metrics_listener_auth_keepalive_and_scalar_payload(monkeypatch) -> None:
    handler_threads: list[int] = []

    def snapshot() -> dict[str, object]:
        handler_threads.append(threading.get_ident())
        return _PAYLOAD

    monkeypatch.setattr(dedicated_server, "_art_metrics_snapshot", snapshot)
    server, thread = dedicated_server._start_fast_metrics_server(
        "127.0.0.1", ["first", "second"]
    )
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=1.0)
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
        assert len(set(handler_threads)) == 1
        assert handler_threads[0] != threading.get_ident()
    finally:
        connection.close()
        dedicated_server._stop_fast_metrics_server(server, thread)
    assert not thread.is_alive()


def test_fast_metrics_listener_responds_while_main_event_loop_is_blocked(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dedicated_server, "_art_metrics_snapshot", lambda: _PAYLOAD)
    blocked = threading.Event()
    release = threading.Event()

    async def block_main_loop() -> None:
        blocked.set()
        release.wait(5.0)

    main_loop = threading.Thread(
        target=lambda: asyncio.run(block_main_loop()), name="blocked_main_api_loop"
    )
    server, server_thread = dedicated_server._start_fast_metrics_server("127.0.0.1", [])
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=1.0)
    try:
        main_loop.start()
        assert blocked.wait(1.0)
        for _ in range(10):
            assert _get(connection)[0] == 200
        assert main_loop.is_alive()
    finally:
        release.set()
        main_loop.join(1.0)
        connection.close()
        dedicated_server._stop_fast_metrics_server(server, server_thread)
    assert not main_loop.is_alive()


def test_fast_metrics_listener_stops_and_restarts_on_same_port(monkeypatch) -> None:
    monkeypatch.setattr(dedicated_server, "_art_metrics_snapshot", lambda: _PAYLOAD)
    server, thread = dedicated_server._start_fast_metrics_server("127.0.0.1", [])
    port = server.server_port
    dedicated_server._stop_fast_metrics_server(server, thread)
    assert not thread.is_alive()

    restarted, restarted_thread = dedicated_server._start_fast_metrics_server(
        "127.0.0.1", [], port
    )
    connection = HTTPConnection("127.0.0.1", port, timeout=1.0)
    try:
        assert _get(connection)[0] == 200
    finally:
        connection.close()
        dedicated_server._stop_fast_metrics_server(restarted, restarted_thread)
    assert not restarted_thread.is_alive()


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
