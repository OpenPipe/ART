import asyncio
import errno
import signal
import socket
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest

from art.tinker import server as module
from art.tinker.server import OpenAICompatibleTinkerServer


@pytest.fixture(autouse=True)
def signal_handlers():
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in original.items():
        signal.signal(sig, handler)


@pytest.fixture
def workers(monkeypatch):
    created = []

    def create(*args, **kwargs):
        worker = Mock()
        created.append(worker)
        return worker

    monkeypatch.setattr(module, "move_to_child_process", create)
    monkeypatch.setattr(
        module.os, "sched_getaffinity", lambda _: set(range(128)), raising=False
    )
    return created


@pytest.mark.parametrize("port", [None, 0])
async def test_concurrent_servers_reserve_distinct_ports(workers, port):
    servers = [
        OpenAICompatibleTinkerServer(host="127.0.0.1", port=port) for _ in range(3)
    ]
    tasks = []
    lifespans = []
    addresses = []
    try:
        addresses = await asyncio.wait_for(
            asyncio.gather(*(server.start() for server in servers)), 10
        )
        assert len({port for _, port in addresses}) == 3
        assert all(port > 0 for _, port in addresses)
        assert [len(server._workers) for server in servers] == [8, 8, 8]
        async with httpx.AsyncClient(trust_env=False) as client:
            for host, actual_port in addresses:
                response = await client.get(f"http://{host}:{actual_port}/health")
                assert response.json() == {"status": "ok"}
        tasks = [server._task for server in servers]
        for server in servers:
            assert server._server is not None
            lifespans.append(server._server.lifespan)
    finally:
        for server in reversed(servers):
            await server.stop()
    assert all(
        task is not None and task.done() and not task.cancelled() for task in tasks
    )
    assert all(lifespan.shutdown_event.is_set() for lifespan in lifespans)
    for worker in workers:
        worker.close.assert_called_once()
    for host, actual_port in addresses:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, actual_port))
            sock.listen()
    assert all(not server._sockets and not server._workers for server in servers)


async def test_explicit_worker_override_and_restart(workers):
    server = OpenAICompatibleTinkerServer(host="127.0.0.1", num_workers=10)
    try:
        await server.start()
        assert len(server._workers) == 10
        with pytest.raises(RuntimeError, match="already started"):
            await server.start()
        await server.stop()
        await server.stop()
        await server.start()
        assert len(server._workers) == 10
    finally:
        await server.stop()
    assert len(workers) == 20
    for worker in workers:
        worker.close.assert_called_once()


async def test_ipv6_only_hostname_with_explicit_port(workers, monkeypatch):
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
            port = probe.getsockname()[1]
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")
    resolve = socket.getaddrinfo

    def resolve_test_host(host, *args, **kwargs):
        return resolve("::1" if host == "tinker.test" else host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve_test_host)
    server = OpenAICompatibleTinkerServer(host="tinker.test", port=port, num_workers=1)
    try:
        assert await server.start() == ("tinker.test", port)
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(f"http://[::1]:{port}/health")
            assert response.status_code == 200
    finally:
        await server.stop()


@pytest.mark.parametrize("occupied_second_address", [False, True])
async def test_hostname_reserves_all_addresses_on_one_port(
    workers, monkeypatch, occupied_second_address
):
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")
    resolve = socket.getaddrinfo

    def resolve_test_host(host, *args, **kwargs):
        if host == "tinker.test":
            v6 = resolve("::1", *args, **kwargs)
            return v6 + resolve("127.0.0.1", *args, **kwargs) + v6
        return resolve(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve_test_host)
    with socket.socket() as competitor:
        port = 0
        if occupied_second_address:
            competitor.bind(("127.0.0.1", 0))
            competitor.listen()
            port = competitor.getsockname()[1]
        server = OpenAICompatibleTinkerServer(
            host="tinker.test", port=port, num_workers=1
        )
        try:
            if occupied_second_address:
                with pytest.raises(OSError) as caught:
                    await server.start()
                assert caught.value.errno == errno.EADDRINUSE
                assert not workers and not server._sockets
                with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
                    probe.bind(("::1", port))
                    probe.listen()
            else:
                _, port = await server.start()
                assert len(server._sockets) == 2
                assert {sock.getsockname()[1] for sock in server._sockets} == {port}
                async with httpx.AsyncClient(trust_env=False) as client:
                    for address in ("[::1]", "127.0.0.1"):
                        response = await client.get(f"http://{address}:{port}/health")
                        assert response.status_code == 200
        finally:
            await server.stop()


async def test_cancel_serving_then_restart_same_port(workers):
    server = OpenAICompatibleTinkerServer(host="127.0.0.1", num_workers=1)
    try:
        host, port = await server.start()
        assert server._task is not None
        server._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await server._task
    finally:
        await server.stop()
    server.port = port
    try:
        assert await server.start() == (host, port)
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await asyncio.wait_for(
                client.get(f"http://{host}:{port}/health"), 5
            )
            assert response.json() == {"status": "ok"}
    finally:
        await server.stop()
    for worker in workers:
        worker.close.assert_called_once()


@pytest.mark.parametrize("action", ["cancel", "stop"])
async def test_interrupted_real_startup_closes_listener_and_lifespan(
    workers, monkeypatch, action
):
    server = OpenAICompatibleTinkerServer(host="127.0.0.1", num_workers=1)
    loop = asyncio.get_running_loop()
    create_server = loop.create_server
    entered, release = asyncio.Event(), asyncio.Event()
    listeners = []

    async def pause_after_binding(*args, **kwargs):
        listener = await create_server(*args, **kwargs)
        listeners.append(listener)
        entered.set()
        await release.wait()
        return listener

    monkeypatch.setattr(loop, "create_server", pause_after_binding)
    start = asyncio.create_task(server.start())
    stop = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert server._server is not None and server._sockets
        uv_server = server._server
        port = server._sockets[0].getsockname()[1]
        if action == "cancel":
            start.cancel()
        else:
            stop = asyncio.create_task(server.stop())
        await asyncio.sleep(0.02)
        release.set()
        with pytest.raises(
            asyncio.CancelledError if action == "cancel" else RuntimeError
        ):
            await asyncio.wait_for(start, 5)
        if stop is not None:
            await asyncio.wait_for(stop, 5)
        assert uv_server.lifespan.shutdown_event.is_set()
        assert all(not listener.is_serving() for listener in listeners)
        monkeypatch.setattr(loop, "create_server", create_server)
        server.port = port
        await server.start()
        async with httpx.AsyncClient(trust_env=False) as client:
            assert (
                await client.get(f"http://127.0.0.1:{port}/health")
            ).status_code == 200
    finally:
        release.set()
        await server.stop()
        await asyncio.gather(start, *([stop] if stop else []), return_exceptions=True)


async def test_stop_before_readiness_poll_does_not_return_address(workers, monkeypatch):
    server = OpenAICompatibleTinkerServer(host="127.0.0.1", num_workers=1)
    main_loop = module.uvicorn.Server.main_loop
    stops = []

    async def stop_at_ready(uv_server):
        stops.append(asyncio.create_task(server.stop()))
        await main_loop(uv_server)

    monkeypatch.setattr(module.uvicorn.Server, "main_loop", stop_at_ready)
    try:
        with pytest.raises(RuntimeError, match="stopped during startup"):
            await asyncio.wait_for(server.start(), 5)
    finally:
        await asyncio.gather(*stops)
        await server.stop()
    assert not server._sockets and not server._workers


@pytest.mark.parametrize(
    "error", [errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL, errno.EADDRINUSE]
)
async def test_hostname_address_fallback_excludes_collisions(
    workers, monkeypatch, error
):
    resolve, new_socket = socket.getaddrinfo, socket.socket
    first = Mock()
    first.bind.side_effect = OSError(error, "first address unavailable")

    def resolve_test_host(host, port, *args, **kwargs):
        addresses = resolve(
            "127.0.0.1" if host == "tinker.test" else host, port, *args, **kwargs
        )
        if host == "tinker.test":
            return [
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", port, 0, 0)),
                *addresses,
            ]
        return addresses

    def create_socket(family=socket.AF_INET, *args, **kwargs):
        return (
            first if family == socket.AF_INET6 else new_socket(family, *args, **kwargs)
        )

    monkeypatch.setattr(socket, "getaddrinfo", resolve_test_host)
    monkeypatch.setattr(socket, "socket", create_socket)
    server = OpenAICompatibleTinkerServer(host="tinker.test", num_workers=1)
    try:
        if error == errno.EADDRINUSE:
            with pytest.raises(OSError) as caught:
                await server.start()
            assert caught.value.errno == error
            assert not workers
        else:
            _, port = await server.start()
            async with httpx.AsyncClient(trust_env=False) as client:
                assert (
                    await client.get(f"http://127.0.0.1:{port}/health")
                ).status_code == 200
        first.close.assert_called_once()
    finally:
        await server.stop()


async def test_port_reserved_before_workers_and_explicit_port_collision(
    workers, monkeypatch
):
    first = OpenAICompatibleTinkerServer(host="127.0.0.1", num_workers=1)
    create = module.move_to_child_process

    def verify_reservation(*args, **kwargs):
        assert first._sockets
        with socket.socket() as competitor:
            competitor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError):
                competitor.bind(first._sockets[0].getsockname())
        return create(*args, **kwargs)

    monkeypatch.setattr(module, "move_to_child_process", verify_reservation)
    try:
        host, port = await first.start()
        second = OpenAICompatibleTinkerServer(host=host, port=port)
        with pytest.raises(OSError):
            await second.start()
        assert not second._sockets
        assert not second._workers
        assert len(workers) == 1
    finally:
        await first.stop()
    # Explicit fixed ports work once the previous owner releases them.
    monkeypatch.setattr(module, "move_to_child_process", create)
    try:
        assert await second.start() == (host, port)
    finally:
        await second.stop()


async def test_partial_worker_failure_releases_socket_and_workers(workers, monkeypatch):
    server = OpenAICompatibleTinkerServer(host="127.0.0.1")
    create = module.move_to_child_process
    address = None

    def fail_second(*args, **kwargs):
        nonlocal address
        assert server._sockets
        address = server._sockets[0].getsockname()
        if workers:
            raise RuntimeError("worker failed")
        return create(*args, **kwargs)

    monkeypatch.setattr(module, "move_to_child_process", fail_second)
    with pytest.raises(RuntimeError, match="worker failed"):
        await server.start()
    assert not server._sockets and not server._workers
    workers[0].close.assert_called_once()
    with socket.socket() as sock:
        assert address is not None
        sock.bind(address)


@pytest.mark.parametrize("failure", ["error", "return", "timeout", "cancel", "stop"])
async def test_failed_start_closes_owned_resources(workers, monkeypatch, failure):
    server = OpenAICompatibleTinkerServer(host="127.0.0.1", num_workers=1)
    entered = asyncio.Event()
    address = None

    async def run(host, port, sockets):
        nonlocal address
        address = sockets[0].getsockname()
        entered.set()
        if failure == "error":
            raise ValueError("serve failed")
        if failure == "return":
            return
        await asyncio.Event().wait()

    monkeypatch.setattr(server, "_run", run)
    monkeypatch.setenv("ART_SERVER_TIMEOUT", "0.05" if failure == "timeout" else "5")
    task = asyncio.create_task(server.start())
    await asyncio.wait_for(entered.wait(), 5)
    if failure == "cancel":
        task.cancel()
    elif failure == "stop":
        await server.stop()
    expected = {
        "error": ValueError,
        "return": RuntimeError,
        "timeout": TimeoutError,
        "cancel": asyncio.CancelledError,
        "stop": RuntimeError,
    }
    with pytest.raises(expected[failure]):
        await asyncio.wait_for(task, 5)
    assert server._task is None and not server._sockets and not server._workers
    workers[0].close.assert_called_once()
    with socket.socket() as sock:
        assert address is not None
        sock.bind(address)


@pytest.mark.parametrize("cpus, expected", [(128, 8), (2, 2), (0, 1)])
def test_default_workers_respect_affinity(monkeypatch, cpus, expected):
    monkeypatch.setattr(
        module.os, "sched_getaffinity", lambda _: set(range(cpus)), raising=False
    )
    assert OpenAICompatibleTinkerServer()._default_num_workers() == expected


@pytest.mark.parametrize("cpus, expected", [(128, 8), (2, 2), (None, 1)])
def test_default_workers_without_affinity(monkeypatch, cpus, expected):
    monkeypatch.delattr(module.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(module.os, "cpu_count", lambda: cpus)
    assert OpenAICompatibleTinkerServer()._default_num_workers() == expected


async def test_real_worker_process_retires_on_stop(monkeypatch):
    monkeypatch.setenv("ART_MP_ACTOR_START_TIMEOUT", "30")
    server = OpenAICompatibleTinkerServer(host="127.0.0.1", num_workers=1)
    try:
        host, port = await server.start()
        worker = cast(Any, server._workers[0])
        assert worker._process.is_alive()
        async with httpx.AsyncClient(trust_env=False) as client:
            assert (await client.get(f"http://{host}:{port}/health")).status_code == 200
    finally:
        await server.stop()
    assert not worker._process.is_alive()
    assert not worker._dispatcher.is_alive()
