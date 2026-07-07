import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from safetensors.torch import save_file
import torch

import art
from art.megatron import service as service_module
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
        json: dict[str, object] | None = None,
        timeout: float,
    ) -> _AsyncOkResponse:
        self._posts.append((url, json if json is not None else params, timeout))
        return _AsyncOkResponse()


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


def test_megatron_reused_step0_identity_adapter_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lora_path = tmp_path / "checkpoints" / "0000"
    lora_path.mkdir(parents=True)
    save_file(
        {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(1, 1)},
        lora_path / "adapter_model.safetensors",
    )
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        service_module,
        "normalize_lora_checkpoint_to_vllm",
        lambda path, *, allow_unvalidated_arch=False: calls.append(
            (str(path), allow_unvalidated_arch)
        ),
    )
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={"allow_unvalidated_arch": True},
        output_dir=str(tmp_path),
    )

    service._ensure_identity_lora(str(lora_path), normalize_existing=True)

    assert calls == [(str(lora_path), True)]


def test_megatron_in_flight_lora_uses_derived_slot_name(tmp_path: Path) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "rollout_weights_mode": "lora",
            "rollout_weight_update_mode": "in_flight_lora",
        },
        output_dir=str(tmp_path),
    )
    service._latest_step = 3

    assert service._in_flight_lora_slot == "test-model:active"
    assert service._initial_served_model_name == "test-model:active"


@pytest.mark.asyncio
async def test_megatron_in_flight_lora_update_uses_derived_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "rollout_weights_mode": "lora",
            "rollout_weight_update_mode": "in_flight_lora",
        },
        output_dir=str(tmp_path),
    )
    service._vllm_runtime.port = 8123
    posts: list[tuple[str, dict[str, object] | None, float]] = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda: _RecordingAsyncClient(posts))

    await service._update_in_flight_adapter("/tmp/art/checkpoints/4", 4)

    assert posts == [
        (
            "http://127.0.0.1:8123/art/in_flight_lora_update",
            {
                "model_name": "test-model@4",
                "lora_slot": "test-model:active",
                "lora_path": "/tmp/art/checkpoints/4",
                "policy_version": 4,
            },
            60.0,
        )
    ]
    assert service._latest_step == 4
    assert service._loaded_adapter_steps == {4}


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
    real_popen = subprocess.Popen

    def _fake_popen(command: Any, *args: Any, **kwargs: Any) -> Any:
        if not (
            isinstance(command, list)
            and len(command) > 2
            and command[1].endswith("managed_process.py")
        ):
            return real_popen(command, *args, **kwargs)
        recorded["command"] = command
        recorded["cwd"] = kwargs["cwd"]
        recorded["env"] = kwargs["env"]
        recorded["stdout"] = kwargs["stdout"]
        recorded["stderr"] = kwargs["stderr"]
        recorded["start_new_session"] = kwargs["start_new_session"]
        return SimpleNamespace(pid=12345, wait=lambda: 0)

    monkeypatch.setattr(
        "art.megatron.service.subprocess.Popen",
        _fake_popen,
    )
    monkeypatch.setattr(
        service._child_processes,
        "watch_popen",
        lambda name, process, *, log_path: recorded.update(
            {"watch_name": name, "watch_process": process, "watch_log_path": log_path}
        ),
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
    assert recorded["watch_name"] == "Megatron worker"
    service._child_processes.close()
    service._megatron_log_file.close()
