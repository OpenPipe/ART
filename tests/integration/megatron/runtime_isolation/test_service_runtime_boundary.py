import asyncio
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import httpx
import pytest


def _process_is_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


def test_vllm_start_releases_the_host_service_mailbox() -> None:
    from art.distributed.monarch_actor import ArtHostService

    start = ArtHostService.__dict__["start_vllm_member"]
    assert getattr(start._method, "_monarch_concurrent_endpoint_wrapper", False)


@pytest.mark.asyncio
async def test_megatron_runtime_override_is_sent_to_host_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import art.distributed.art_runtime as art_runtime

    runtime_python = "/runtime/.venv/bin/art-megatron-python"
    monkeypatch.setenv("ART_MEGATRON_RUNTIME_PYTHON", runtime_python)
    runtime = art_runtime.ArtRuntime(
        object(),
        cast(
            Any,
            SimpleNamespace(
                cluster=SimpleNamespace(nixl_transport=None, artifact_root=None)
            ),
        ),
    )
    endpoint = object()
    runtime._host_services = {
        "host": SimpleNamespace(ensure_megatron_runtime=endpoint)
    }
    info = SimpleNamespace(
        model_dump_json=lambda: "runtime-contract",
        profile="cuda13",
        variant="hybrid_ep",
        runtime=SimpleNamespace(sha256="runtime-sha"),
    )
    calls: list[tuple[object, tuple[object, ...]]] = []

    async def call_remote(target: object, *args: object) -> object:
        calls.append((target, args))
        return info

    monkeypatch.setattr(art_runtime, "call_remote", call_remote)

    assert (
        await runtime._ensure_megatron_runtime(
            ("host",), require_hybrid_ep=True, multinode=False
        )
        is info
    )
    assert calls == [(endpoint, (True, False, runtime_python))]


@pytest.mark.asyncio
async def test_publication_wait_is_reserved_before_next_train_can_expire_it() -> None:
    from art.megatron.runtime.monarch import (
        MonarchTrainerRun,
        _PublicationState,
    )

    run = MonarchTrainerRun.__new__(MonarchTrainerRun)
    future = asyncio.get_running_loop().create_future()
    state = _PublicationState("generation-1", future)
    state.train_done = True
    run._publications = {state.generation_id: state}

    waiter = run.wait_for_publication(state.generation_id)
    assert state.active_waiters == 1
    run._expire_prior_publications()
    assert state.generation_id in run._publications

    future.set_result(())
    assert await waiter == ()
    assert state.generation_id not in run._publications


@pytest.mark.asyncio
async def test_snapshot_publication_waits_for_receiver_drain_before_reuse() -> None:
    from art.megatron.runtime.monarch import (
        MonarchTrainerRun,
        _PublicationState,
    )

    run = MonarchTrainerRun.__new__(MonarchTrainerRun)
    future = asyncio.get_running_loop().create_future()
    state = _PublicationState("generation-1", future)
    state.train_done = True
    state.drain_done = False
    run._publications = {state.generation_id: state}

    publication = asyncio.create_task(
        run._await_retired_snapshot_publication(
            run.wait_for_publication(state.generation_id), state
        )
    )
    future.set_result(())
    await asyncio.sleep(0)

    assert not publication.done()
    assert state.generation_id in run._publications

    state.drain_done = True
    run._retire_publication(state)

    assert await publication == ()
    assert state.generation_id not in run._publications


@pytest.mark.skipif(
    sys.platform != "linux", reason="requires Linux parent-death signal"
)
def test_owned_local_worker_dies_when_controller_is_sigkilled(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "worker.pid"
    program = """
import signal
import sys
from pathlib import Path

from art.distributed.monarch_bootstrap import _start_worker

worker = _start_worker("tcp://127.0.0.1:0")
Path(sys.argv[1]).write_text(str(worker.process.pid))
signal.pause()
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", program, str(pid_path)],
        cwd=Path(__file__).resolve().parents[4],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert parent.stderr is not None
    worker_pid: int | None = None
    try:
        deadline = time.monotonic() + 30
        while not pid_path.exists() and parent.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if not pid_path.exists():
            detail = parent.stderr.read() if parent.poll() is not None else "timeout"
            pytest.fail(f"controller did not start a worker: {detail}")
        worker_pid = int(pid_path.read_text())
        assert _process_is_running(worker_pid)

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=10)
        deadline = time.monotonic() + 10
        while _process_is_running(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _process_is_running(worker_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        if worker_pid is not None and _process_is_running(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process identity")
def test_local_start_reconciles_legacy_owned_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import art.distributed.monarch_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_WORKER_LOCK_ROOT", tmp_path)
    address = bootstrap._resolve_ephemeral_worker_address("tcp://127.0.0.1:0")
    bootstrap._worker_lock_path(address).touch()
    program = f"""
import os
import subprocess
import sys

worker_code = {bootstrap._LEGACY_OWNED_WORKER_CODE!r}

worker = subprocess.Popen(
    [sys.executable, "-c", worker_code, sys.argv[1]],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
print(worker.pid, flush=True)
os._exit(0)
"""
    launcher = subprocess.run(
        [sys.executable, "-c", program, address],
        cwd=Path(__file__).resolve().parents[4],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    orphan_pid = int(launcher.stdout)
    worker = None
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            identity = bootstrap._process_identity(orphan_pid)
            if identity is not None and identity[1] == 1:
                break
            time.sleep(0.05)
        else:
            pytest.fail("legacy worker was not reparented")

        worker = bootstrap._start_worker("tcp://127.0.0.1:0", startup_timeout_s=30)
        assert not _process_is_running(orphan_pid)
    finally:
        if worker is not None:
            bootstrap._stop_worker(worker)
        if _process_is_running(orphan_pid):
            os.kill(orphan_pid, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process identity")
def test_orphan_reconciliation_never_targets_unrelated_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import art.distributed.monarch_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_WORKER_LOCK_ROOT", tmp_path)
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        identity = bootstrap._process_identity(unrelated.pid)
        assert identity is not None
        address = "tcp://127.0.0.1:43219"
        metadata = bootstrap._OwnedWorkerMetadata(
            address=address,
            controller_pid=2**30,
            controller_start_time=1,
            worker_pid=unrelated.pid,
            worker_start_time=identity[0],
            python_executable=os.path.realpath(sys.executable),
            worker_code_sha256="0" * 64,
            ownership_token="0" * 32,
        )
        bootstrap._worker_lock_path(address).write_text(metadata.model_dump_json())

        bootstrap._reconcile_orphaned_workers()

        assert unrelated.poll() is None
        assert bootstrap._worker_lock_path(address).exists()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)


@pytest.mark.asyncio
async def test_trainer_run_close_retries_failed_proc_mesh_stop() -> None:
    from art.megatron.runtime.monarch import MonarchTrainerRun

    class ProcMesh:
        def __init__(self) -> None:
            self.stop_calls = 0

        async def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("injected stop failure")

    proc_mesh = ProcMesh()
    supervision = SimpleNamespace(close=Mock())
    run = MonarchTrainerRun.__new__(MonarchTrainerRun)
    run.run_spec = SimpleNamespace(shutdown_timeout_s=1.0)
    run._proc_mesh = cast(Any, proc_mesh)
    run._supervision = supervision
    run._stop_task = None
    run._close_task = None
    run._closed = False
    run._valid = False
    run._active_job_id = None
    run._active_receive = None
    run._active_collective = None

    with pytest.raises(RuntimeError, match="injected stop failure"):
        await run.close()
    assert proc_mesh.stop_calls == 1
    supervision.close.assert_not_called()

    await run.close()
    assert proc_mesh.stop_calls == 2
    supervision.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_art_runtime_stop_trainer_retains_failed_run() -> None:
    from art.distributed.art_runtime import ArtRuntime

    class Run:
        def __init__(self) -> None:
            self.close = AsyncMock(
                side_effect=[RuntimeError("injected close failure"), None]
            )

    runtime = ArtRuntime.__new__(ArtRuntime)
    run = Run()
    runtime._trainer_runs = {run}

    with pytest.raises(RuntimeError, match="injected close failure"):
        await runtime.stop_trainer(run)
    assert run in runtime._trainer_runs

    await runtime.stop_trainer(run)
    assert run not in runtime._trainer_runs
    assert run.close.await_count == 2


@pytest.mark.asyncio
async def test_distributed_service_close_retries_owned_resources(
    tmp_path: Path,
) -> None:
    from art.megatron.distributed_service import DistributedMegatronService

    class Runtime:
        def __init__(self) -> None:
            self.trainer_stops = 0
            self.model_stops = 0

        async def stop_trainer(self, _trainer: object) -> None:
            self.trainer_stops += 1
            if self.trainer_stops == 1:
                raise RuntimeError("injected trainer stop failure")

        async def stop_model_service(self, _name: str) -> None:
            self.model_stops += 1
            if self.model_stops == 1:
                raise RuntimeError("injected model stop failure")

    runtime = Runtime()
    service = DistributedMegatronService(
        model_name="model",
        base_model="base",
        config=cast(Any, {}),
        output_dir=str(tmp_path),
        runtime=cast(Any, runtime),
        enable_expert_replay=False,
    )
    trainer = object()
    service._trainer = trainer
    service._managed_service_name = "model"

    with pytest.raises(BaseExceptionGroup):
        await service.aclose()
    assert service._trainer is trainer
    assert service._managed_service_name == "model"

    await service.aclose()
    assert service._trainer is None
    assert service._managed_service_name is None
    assert (runtime.trainer_stops, runtime.model_stops) == (2, 2)


@pytest.mark.asyncio
async def test_failed_vllm_start_rollback_remains_runtime_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import art.distributed.art_runtime as art_runtime

    class Manager:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.stop_calls = 0

        async def start(self) -> None:
            raise RuntimeError("injected startup rollback failure")

        async def stop(self) -> object:
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("injected rollback retry failure")
            return object()

    monkeypatch.setattr(art_runtime, "ReplicaManager", Manager)
    runtime = art_runtime.ArtRuntime.__new__(art_runtime.ArtRuntime)
    runtime._started = True
    runtime._closed = False
    runtime._model_services = {}
    runtime._host_services = {"host": object()}
    runtime._adapter_services = {"host": object()}
    runtime._preflight_launch = AsyncMock()
    spec = SimpleNamespace(
        name="model",
        members=(SimpleNamespace(host_id="host", gpu_ids=(0,)),),
        rendezvous=SimpleNamespace(host="127.0.0.1"),
    )
    runtime.topology = SimpleNamespace(
        cluster=SimpleNamespace(startup_timeout_s=1.0, rpc_timeout_s=1.0),
        model_services=(spec,),
    )

    with pytest.raises(RuntimeError, match="injected startup rollback failure"):
        await runtime.start_model_service(cast(Any, spec), cast(Any, object()))
    assert "model" in runtime._model_services

    with pytest.raises(RuntimeError, match="injected rollback retry failure"):
        await runtime.stop_model_service("model")
    assert "model" in runtime._model_services

    await runtime.stop_model_service("model")
    assert "model" not in runtime._model_services


@pytest.mark.asyncio
async def test_vllm_host_member_close_retries_without_losing_owner(
    tmp_path: Path,
) -> None:
    from art.distributed.vllm_replica import ManagedVllmHostLauncher

    class MemberRuntime:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected member close failure")

    key = ("replica", "member", 0)
    member_runtime = MemberRuntime()
    launcher = ManagedVllmHostLauncher(str(tmp_path))
    launcher._members[key] = cast(
        Any,
        SimpleNamespace(
            runtime=member_runtime,
            supervisor=SimpleNamespace(close=Mock()),
        ),
    )

    with pytest.raises(RuntimeError, match="injected member close failure"):
        await launcher.stop_member(*key)
    assert key in launcher._members

    await launcher.stop_member(*key)
    assert key not in launcher._members
    assert member_runtime.close_calls == 2


@pytest.mark.asyncio
async def test_cancelled_megatron_close_keeps_runtimes_until_services_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.backend import MegatronBackend

    service_started = asyncio.Event()
    release_service = asyncio.Event()
    events: list[str] = []

    class Service:
        propagate_close_errors = True

        async def aclose(self) -> None:
            events.append("service_started")
            service_started.set()
            await release_service.wait()
            events.append("service_stopped")

    class Runtime:
        async def close(self) -> None:
            events.append("runtime_stopped")

    monkeypatch.setattr("art.local.backend.close_proxy", lambda _service: None)
    monkeypatch.setattr("art.local.backend.torch.cuda.is_available", lambda: False)
    backend = MegatronBackend(path=str(tmp_path))
    key = ("project", "model")
    backend._services[key] = cast(Any, Service())
    backend._owned_runtimes[key] = cast(Any, Runtime())

    close = asyncio.create_task(backend.close())
    await service_started.wait()
    close.cancel()
    await asyncio.sleep(0)
    assert events == ["service_started"]

    release_service.set()
    with pytest.raises(asyncio.CancelledError):
        await close
    assert events == ["service_started", "service_stopped", "runtime_stopped"]
    assert not backend._services
    assert not backend._owned_runtimes


@pytest.mark.asyncio
async def test_megatron_close_retries_services_before_owned_runtimes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.backend import MegatronBackend

    class Service:
        propagate_close_errors = True

        def __init__(self) -> None:
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected service close failure")

    class Runtime:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("injected runtime close failure")

    monkeypatch.setattr("art.local.backend.close_proxy", lambda _service: None)
    monkeypatch.setattr("art.local.backend.torch.cuda.is_available", lambda: False)
    backend = MegatronBackend(path=str(tmp_path))
    key = ("project", "model")
    service = Service()
    runtime = Runtime()
    backend._services[key] = cast(Any, service)
    backend._owned_runtimes[key] = cast(Any, runtime)

    with pytest.raises(BaseExceptionGroup):
        await backend.close()
    assert backend._services[key] is service
    assert backend._owned_runtimes[key] is runtime
    assert runtime.close_calls == 0

    with pytest.raises(BaseExceptionGroup):
        await backend.close()
    assert key not in backend._services
    assert backend._owned_runtimes[key] is runtime

    await backend.close()
    assert not backend._owned_runtimes
    assert (service.close_calls, runtime.close_calls) == (2, 2)


@pytest.mark.asyncio
async def test_owned_model_runtimes_reserve_disjoint_local_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from art.distributed.art_runtime import ArtRuntime
    from art.megatron.backend import MegatronBackend

    class Model:
        project = "project"
        base_model = "/tmp/base"
        _internal_config: dict[str, object] = {}

        def __init__(self, name: str) -> None:
            self.name = name

        def _storage_name(self) -> str:
            return self.name

    async def start_local(topology: object) -> object:
        return SimpleNamespace(topology=topology, close=AsyncMock())

    monkeypatch.setattr(ArtRuntime, "start_local", staticmethod(start_local))
    monkeypatch.setattr(
        "art.megatron.runtime.local.get_megatron_runtime_config",
        lambda: SimpleNamespace(
            topology={"tp": 1, "ep": 1, "etp": 1, "cp": 1, "pp": 1}
        ),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr("art.local.backend.torch.cuda.is_available", lambda: False)
    backend = MegatronBackend(path=str(tmp_path))
    first = Model("first")
    second = Model("second")

    first_runtime = await backend._ensure_runtime(
        cast(Any, first), cast(Any, {"trainer_gpu_ids": [0]})
    )
    second_runtime = await backend._ensure_runtime(
        cast(Any, second), cast(Any, {"trainer_gpu_ids": [1]})
    )
    first_service = first_runtime.topology.model_services[0]
    second_service = second_runtime.topology.model_services[0]
    first_ports = {
        first_service.leader_endpoint.port,
        first_service.rendezvous.port,
    }
    second_ports = {
        second_service.leader_endpoint.port,
        second_service.rendezvous.port,
    }

    assert len(first_ports) == len(second_ports) == 2
    assert first_ports.isdisjoint(second_ports)
    with pytest.raises(ValueError, match="already reserved"):
        await backend._configure_owned_api_port(
            cast(Any, first), second_service.leader_endpoint.port
        )

    await backend.close()
    assert not backend._owned_runtime_ports
    assert not backend._local_endpoints._owned


class _AsyncOkResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None


class _RecordingAsyncClient:
    def __init__(self, posts: list[tuple[str, object, float]]) -> None:
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(
        self,
        url: str,
        *,
        params: object = None,
        json: object = None,
        timeout: float,
    ) -> _AsyncOkResponse:
        self._posts.append((url, json if json is not None else params, timeout))
        return _AsyncOkResponse()


@pytest.mark.asyncio
async def test_unsloth_shared_start_requires_runtime_sleep_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsloth_service = pytest.importorskip("art.unsloth.service")
    service = unsloth_service.UnslothService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={"engine_args": {"enable_sleep_mode": False}},
        output_dir=str(tmp_path),
    )
    service.__dict__["_state"] = SimpleNamespace(
        trainer=SimpleNamespace(save_model=lambda path: None),
        offload_to_cpu=lambda: None,
    )
    monkeypatch.setattr(
        "art.unsloth.service.get_last_checkpoint_dir", lambda _output_dir: "/tmp/lora"
    )
    monkeypatch.setattr("art.unsloth.service.get_step_from_dir", lambda _output_dir: 0)
    monkeypatch.setattr(service, "_start_vllm_subprocess", AsyncMock())

    with pytest.raises(
        ValueError,
        match="Shared-GPU mode requires engine_args.enable_sleep_mode=True",
    ):
        await service.start_openai_server(None)


@pytest.mark.asyncio
async def test_unsloth_runtime_sleep_and_wake_use_runtime_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsloth_service = pytest.importorskip("art.unsloth.service")
    service = unsloth_service.UnslothService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={},
        output_dir=str(tmp_path),
    )
    service._vllm_port = 8123
    posts: list[tuple[str, object, float]] = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda: _RecordingAsyncClient(posts))

    await service._sleep_runtime()
    await service._wake_runtime()

    assert posts == [
        ("http://127.0.0.1:8123/sleep", {"level": 1, "mode": "wait"}, 300.0),
        ("http://127.0.0.1:8123/wake_up", None, 300.0),
    ]
    assert service._is_sleeping is False
