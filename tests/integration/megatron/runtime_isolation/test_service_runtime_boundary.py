import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest

from art.megatron.service import MegatronService
from art.types import MegatronTopologyConfig
from art.unsloth.service import UnslothService


class _AsyncOkResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None


class _RecordingAsyncClient:
    def __init__(
        self, posts: list[tuple[str, dict[str, object] | None, float]]
    ) -> None:
        self._posts = posts

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def post(
        self,
        url: str,
        *,
        params: dict[str, object] | None = None,
        timeout: float,
    ) -> _AsyncOkResponse:
        self._posts.append((url, params, timeout))
        return _AsyncOkResponse()


class _RecordingVllmClient:
    def __init__(self) -> None:
        self.gets: list[tuple[str, dict[str, str] | None, float]] = []
        self.posts: list[
            tuple[str, dict[str, object], dict[str, str] | None, float]
        ] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float,
    ) -> _AsyncOkResponse:
        self.gets.append((url, headers, timeout))
        return _AsyncOkResponse()

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str] | None = None,
        timeout: float,
    ) -> _AsyncOkResponse:
        self.posts.append((url, json, headers, timeout))
        return _AsyncOkResponse()


class _FakeAsyncioProcess:
    returncode: int | None = None

    async def wait(self) -> int:
        await asyncio.Event().wait()
        return 0


@pytest.mark.asyncio
async def test_megatron_shared_start_requires_runtime_sleep_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "rollout_weights_mode": "lora",
            "engine_args": {"enable_sleep_mode": False},
        },
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr(service, "_resolve_active_lora_path", lambda: "/tmp/lora")
    monkeypatch.setattr(service, "_start_vllm_subprocess", AsyncMock())

    with pytest.raises(
        ValueError,
        match="Shared-GPU mode requires engine_args.enable_sleep_mode=True",
    ):
        await service.start_openai_server(None)


@pytest.mark.asyncio
async def test_unsloth_shared_start_requires_runtime_sleep_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = UnslothService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "rollout_weights_mode": "lora",
            "engine_args": {"enable_sleep_mode": False},
        },
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
async def test_megatron_runtime_sleep_and_wake_use_runtime_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={"rollout_weights_mode": "lora"},
        output_dir=str(tmp_path),
    )
    service._vllm_port = 8123
    posts: list[tuple[str, dict[str, object] | None, float]] = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda: _RecordingAsyncClient(posts))

    await service._sleep_runtime()
    await service._wake_runtime()

    assert posts == [
        ("http://127.0.0.1:8123/sleep", {"level": 1, "mode": "wait"}, 300.0),
        ("http://127.0.0.1:8123/wake_up", None, 300.0),
    ]
    assert service._is_sleeping is False


@pytest.mark.asyncio
async def test_unsloth_runtime_sleep_and_wake_use_runtime_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = UnslothService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={"rollout_weights_mode": "lora"},
        output_dir=str(tmp_path),
    )
    service._vllm_port = 8123
    posts: list[tuple[str, dict[str, object] | None, float]] = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda: _RecordingAsyncClient(posts))

    await service._sleep_runtime()
    await service._wake_runtime()

    assert posts == [
        ("http://127.0.0.1:8123/sleep", {"level": 1, "mode": "wait"}, 300.0),
        ("http://127.0.0.1:8123/wake_up", None, 300.0),
    ]
    assert service._is_sleeping is False


@pytest.mark.asyncio
async def test_megatron_dedicated_merged_start_syncs_initial_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "trainer_gpu_ids": [0],
            "inference_gpu_ids": [1],
            "rollout_weights_mode": "merged",
        },
        output_dir=str(tmp_path),
    )
    start_vllm = AsyncMock(return_value=("127.0.0.1", 8000))
    sync_merged = AsyncMock()
    monkeypatch.setattr(service, "_resolve_active_lora_path", lambda: "/tmp/lora")
    monkeypatch.setattr(service, "_start_vllm_subprocess", start_vllm)
    monkeypatch.setattr(service, "_sync_dedicated_merged_weights", sync_merged)

    location = await service.start_openai_server(None)

    assert location == ("127.0.0.1", 8000)
    start_vllm.assert_awaited_once()
    sync_merged.assert_awaited_once_with(
        lora_path="/tmp/lora",
        step=0,
        megatron_topology=None,
    )


@pytest.mark.asyncio
async def test_megatron_external_vllm_start_attaches_and_loads_mapped_lora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "trainer_gpu_ids": [2, 3],
            "rollout_weights_mode": "lora",
            "vllm_runtime": {
                "mode": "external",
                "server_url": "http://inference-node:8000",
                "api_key": "secret",
                "local_checkpoint_root": "/mnt/ws_pvc/ws",
                "server_checkpoint_root": "/remote/ws",
            },
        },
        output_dir=str(tmp_path),
    )
    client = _RecordingVllmClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(
        service,
        "_resolve_active_lora_path",
        lambda: "/mnt/ws_pvc/ws/checkpoints/model/0000",
    )

    location = await service.start_openai_server(None)

    assert location == ("http://inference-node:8000", 0)
    assert client.gets == [
        (
            "http://inference-node:8000/health",
            {"Authorization": "Bearer secret"},
            5.0,
        ),
        (
            "http://inference-node:8000/v1/models",
            {"Authorization": "Bearer secret"},
            5.0,
        ),
    ]
    assert client.posts == [
        (
            "http://inference-node:8000/v1/load_lora_adapter",
            {
                "lora_name": "test-model@0",
                "lora_path": "/remote/ws/checkpoints/model/0000",
                "load_inplace": True,
            },
            {"Authorization": "Bearer secret"},
            60.0,
        )
    ]


@pytest.mark.asyncio
async def test_unsloth_external_vllm_start_attaches_and_loads_lora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = UnslothService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "rollout_weights_mode": "lora",
            "vllm_runtime": {
                "mode": "external",
                "server_url": "http://inference-node:8000/v1",
            },
        },
        output_dir=str(tmp_path),
    )
    client = _RecordingVllmClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(
        "art.unsloth.service.get_last_checkpoint_dir",
        lambda _output_dir: "/mnt/ws_pvc/ws/checkpoints/model/0000",
    )
    monkeypatch.setattr("art.unsloth.service.get_step_from_dir", lambda _output_dir: 0)

    location = await service.start_openai_server(None)

    assert location == ("http://inference-node:8000", 0)
    assert client.posts == [
        (
            "http://inference-node:8000/v1/load_lora_adapter",
            {
                "lora_name": "test-model@0",
                "lora_path": "/mnt/ws_pvc/ws/checkpoints/model/0000",
                "load_inplace": True,
            },
            None,
            60.0,
        )
    ]


@pytest.mark.asyncio
async def test_megatron_dedicated_merged_start_uses_configured_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "trainer_gpu_ids": [0],
            "inference_gpu_ids": [1],
            "rollout_weights_mode": "merged",
            "megatron_topology": {"tp": 1, "cp": 2, "ep": 2, "etp": 1},
        },
        output_dir=str(tmp_path),
    )
    start_vllm = AsyncMock(return_value=("127.0.0.1", 8000))
    sync_merged = AsyncMock()
    monkeypatch.setattr(service, "_resolve_active_lora_path", lambda: "/tmp/lora")
    monkeypatch.setattr(service, "_start_vllm_subprocess", start_vllm)
    monkeypatch.setattr(service, "_sync_dedicated_merged_weights", sync_merged)

    await service.start_openai_server(None)

    sync_merged.assert_awaited_once_with(
        lora_path="/tmp/lora",
        step=0,
        megatron_topology=MegatronTopologyConfig(tp=1, cp=2, ep=2, etp=1),
    )


@pytest.mark.asyncio
async def test_megatron_worker_uses_active_python_for_torchrun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("megatron.bridge")
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "trainer_gpu_ids": [0],
            "inference_gpu_ids": [1],
            "rollout_weights_mode": "lora",
        },
        output_dir=str(tmp_path),
    )
    recorded: dict[str, object] = {}

    async def _fake_create_subprocess_exec(
        *command: str,
        cwd: str,
        env: dict[str, str],
        stdout,
        stderr,
        start_new_session: bool,
    ) -> _FakeAsyncioProcess:
        recorded["command"] = list(command)
        recorded["cwd"] = cwd
        recorded["env"] = env
        recorded["stdout"] = stdout
        recorded["stderr"] = stderr
        recorded["start_new_session"] = start_new_session
        return _FakeAsyncioProcess()

    monkeypatch.setattr(
        "art.megatron.service.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
    )
    monkeypatch.setattr(service, "_install_parent_signal_cleanup", lambda: None)
    monkeypatch.setattr(service, "_allocate_master_port", lambda: 12345)

    await service._ensure_megatron_running()
    command = cast(list[str], recorded["command"])
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert command[1].endswith("managed_process.py")
    separator = command.index("--")
    assert command[separator + 1 : separator + 4] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
    ]
    assert "uv run" not in command
    assert recorded["cwd"] == str(Path(__file__).resolve().parents[4])
    service._child_processes.close()
    service._megatron_log_file.close()
