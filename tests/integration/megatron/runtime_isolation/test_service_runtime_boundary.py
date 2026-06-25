import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest

import art
from art.megatron.service import MegatronService


@pytest.fixture(autouse=True)
def _init_megatron_runtime_config() -> None:
    art.init_megatron_runtime_config(
        topology=art.MegatronTopologyConfig(tp=1, cp=2, ep=2, etp=1),
        packed_sequence_length=1024,
    )


class _AsyncOkResponse:
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


class _FakeAsyncioProcess:
    returncode: int | None = None

    async def wait(self) -> int:
        await asyncio.Event().wait()
        return 0


def test_megatron_default_lora_adapter_config_uses_model_lora_config(
    tmp_path: Path,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "lora_config": {
                "rank": 8,
                "target_modules": ["q_proj", "down_proj"],
            },
        },
        output_dir=str(tmp_path),
    )

    config = service._default_lora_adapter_config()

    assert config.r == 8
    assert config.target_modules == {"q_proj", "down_proj"}


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
    unsloth_service = pytest.importorskip("art.unsloth.service")
    service = unsloth_service.UnslothService(
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
    unsloth_service = pytest.importorskip("art.unsloth.service")
    service = unsloth_service.UnslothService(
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
    )


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
    )
    assert service.runtime_config.topology.cp == 2


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
            "lora_config": {
                "rank": 8,
                "target_modules": ["q_proj", "down_proj"],
            },
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
    env = cast(dict[str, str], recorded["env"])
    assert env["ART_MEGATRON_LORA_RANK"] == "8"
    assert json.loads(env["ART_MEGATRON_LORA_TARGET_MODULES"]) == [
        "q_proj",
        "down_proj",
    ]
    service._child_processes.close()
    service._megatron_log_file.close()
