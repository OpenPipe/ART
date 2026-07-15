import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest

import art
from art.megatron.runtime.jobs import MegatronOptimizerSaveJob
from art.megatron.service import MegatronService
from art.serving_capabilities import ServingCapabilities


@pytest.fixture(autouse=True)
def _init_megatron_runtime_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from art.megatron import runtime_config

    monkeypatch.setattr(runtime_config, "_MEGATRON_RUNTIME_CONFIG", None)
    art.init_megatron_runtime_config(
        topology=art.MegatronTopologyConfig(tp=1, cp=2, ep=2, etp=1),
        packed_sequence_length=1024,
        streaming_weight_offload=True,
    )


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


@pytest.mark.asyncio
async def test_megatron_in_flight_eval_uses_immutable_adapter_slot(
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

    checkpoint_path = str(tmp_path / "checkpoints" / "4")
    assert (
        await service.acquire_exact_adapter(4, checkpoint_path) == "test-model:eval@4"
    )
    assert (
        await service.acquire_exact_adapter(4, checkpoint_path) == "test-model:eval@4"
    )

    assert posts == [
        (
            "http://127.0.0.1:8123/v1/load_lora_adapter",
            {
                "lora_name": "test-model:eval@4",
                "lora_path": checkpoint_path,
            },
            60.0,
        )
    ]
    assert service._loaded_exact_adapter_steps == {4}

    await service.release_exact_adapter(4)
    assert service._loaded_exact_adapter_steps == {4}
    await service.release_exact_adapter(4)

    assert posts[-1] == (
        "http://127.0.0.1:8123/v1/unload_lora_adapter",
        {"lora_name": "test-model:eval@4"},
        30.0,
    )
    assert service._loaded_exact_adapter_steps == set()

    service._loaded_exact_adapter_steps.add(5)
    await service.prune_loaded_adapters(retain_steps=set())

    assert posts[-1] == (
        "http://127.0.0.1:8123/v1/unload_lora_adapter",
        {"lora_name": "test-model:eval@5"},
        30.0,
    )
    assert service._loaded_exact_adapter_steps == set()


@pytest.mark.asyncio
async def test_external_in_flight_update_maps_checkpoint_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = str(tmp_path / "local")
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={
            "rollout_weights_mode": "lora",
            "rollout_weight_update_mode": "in_flight_lora",
            "vllm_runtime": {
                "mode": "external",
                "server_url": "http://inference:8000",
                "local_checkpoint_root": local_root,
                "server_checkpoint_root": "/remote",
            },
        },
        output_dir=str(tmp_path),
    )
    service._serving_capabilities = ServingCapabilities(
        runtime="art_vllm",
        protocol_version=1,
        in_flight_lora_updates=True,
        policy_token_spans=True,
    )
    posts: list[tuple[str, dict[str, object] | None, float]] = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda: _RecordingAsyncClient(posts))

    await service._update_in_flight_adapter(f"{local_root}/model/0004", 4)

    assert posts[0][1] == {
        "model_name": "test-model@4",
        "lora_slot": "test-model:active",
        "lora_path": "/remote/model/0004",
        "policy_version": 4,
    }


@pytest.mark.asyncio
async def test_clean_training_finalization_submits_latest_optimizer_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="test-model",
        base_model="Qwen/Qwen3-0.6B",
        config={},
        output_dir=str(tmp_path),
    )
    service._latest_step = 4
    service._megatron_process = cast(Any, object())
    written: list[MegatronOptimizerSaveJob] = []

    monkeypatch.setattr(
        "art.megatron.service.read_optimizer_commit", lambda _path: None
    )
    monkeypatch.setattr(
        service,
        "_create_megatron_job_paths",
        lambda: (str(tmp_path / "job.json"), str(tmp_path / "job.log")),
    )
    monkeypatch.setattr(
        "art.megatron.service.write_megatron_job",
        lambda job, **_kwargs: written.append(job),
    )

    async def completed_job(*_args: Any, **_kwargs: Any):
        if False:
            yield {}

    monkeypatch.setattr("art.megatron.service.stream_megatron_job", completed_job)

    await service.finalize_training_session()

    assert len(written) == 1
    assert written[0].step == 4
    assert written[0].training_mode == "rl"
    assert written[0].training_session_id == service._training_session_id


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
    discover_capabilities = AsyncMock()
    monkeypatch.setattr(service, "_resolve_active_lora_path", lambda: "/tmp/lora")
    monkeypatch.setattr(service, "_start_vllm_subprocess", start_vllm)
    monkeypatch.setattr(service, "_sync_dedicated_merged_weights", sync_merged)
    monkeypatch.setattr(
        service, "_discover_serving_capabilities", discover_capabilities
    )

    location = await service.start_openai_server(None)

    assert location == ("127.0.0.1", 8000)
    start_vllm.assert_awaited_once()
    discover_capabilities.assert_awaited_once_with(external=False)
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
    discover_capabilities = AsyncMock()
    monkeypatch.setattr(service, "_resolve_active_lora_path", lambda: "/tmp/lora")
    monkeypatch.setattr(service, "_start_vllm_subprocess", start_vllm)
    monkeypatch.setattr(service, "_sync_dedicated_merged_weights", sync_merged)
    monkeypatch.setattr(
        service, "_discover_serving_capabilities", discover_capabilities
    )

    await service.start_openai_server(None)

    sync_merged.assert_awaited_once_with(
        lora_path="/tmp/lora",
        step=0,
    )
    discover_capabilities.assert_awaited_once_with(external=False)
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
    assert env["ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD"] == "1"
    assert recorded["watch_name"] == "Megatron worker"
    service._child_processes.close()
    service._megatron_log_file.close()
