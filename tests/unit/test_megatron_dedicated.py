import asyncio
import os
from pathlib import Path
import shlex
import sys
import types as pytypes
from typing import Any

import pytest

pytest.importorskip("vllm")

from art import TrainableModel, types
from art.dev.model import InternalModelConfig
from art.megatron.backend import MegatronBackend
from art.megatron.service import MegatronService


@pytest.mark.asyncio
async def test_megatron_backend_dedicated_uses_trainer_gpus_without_child_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = InternalModelConfig(
        trainer_gpu_ids=[0],
        inference_gpu_ids=[1],
        rollout_weights_mode="lora",
    )
    model = TrainableModel(
        name="megatron-dedicated",
        project="unit-tests",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        base_path=str(tmp_path),
        _internal_config=config,
    )
    backend = MegatronBackend(path=str(tmp_path))
    validated: dict[str, Any] = {}

    class FakeService:
        def __init__(
            self,
            *,
            model_name: str,
            base_model: str,
            config: InternalModelConfig,
            output_dir: str,
        ) -> None:
            self.model_name = model_name
            self.base_model = base_model
            self.config = config
            self.output_dir = output_dir

    monkeypatch.setattr(
        "art.dev.get_model_config.get_model_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "art.dev.validate.validate_dedicated_config",
        lambda cfg: validated.setdefault("config", cfg),
    )
    monkeypatch.setattr(
        "art.megatron.backend.move_to_child_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "Dedicated Megatron service should not move to a child process"
            )
        ),
    )
    monkeypatch.setattr("art.megatron.service.MegatronService", FakeService)

    service = await backend._get_service(model)

    assert isinstance(service, FakeService)
    assert validated["config"] is config
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.asyncio
async def test_megatron_service_ensure_megatron_running_uses_trainer_gpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="megatron-dedicated",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config=InternalModelConfig(
            trainer_gpu_ids=[0, 1],
            inference_gpu_ids=[2],
            rollout_weights_mode="lora",
        ),
        output_dir=str(tmp_path),
    )
    megatron_module = pytypes.ModuleType("megatron")
    megatron_bridge_module = pytypes.ModuleType("megatron.bridge")
    monkeypatch.setitem(sys.modules, "megatron", megatron_module)
    monkeypatch.setitem(sys.modules, "megatron.bridge", megatron_bridge_module)

    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        "art.megatron.service.subprocess.run", lambda *args, **kwargs: None
    )

    async def fake_create_subprocess_shell(
        command: str,
        cwd: str,
        env: dict[str, str],
    ) -> Any:
        seen["command"] = command
        seen["cwd"] = cwd
        seen["env"] = env
        return pytypes.SimpleNamespace(returncode=None)

    monkeypatch.setattr(
        "art.megatron.service.asyncio.create_subprocess_shell",
        fake_create_subprocess_shell,
    )

    await service._ensure_megatron_running()

    assert shlex.quote(sys.executable) in seen["command"]
    assert "torch.distributed.run" in seen["command"]
    assert "--nproc_per_node 2" in seen["command"]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert seen["env"]["MODEL_IDENTIFIER"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"


@pytest.mark.asyncio
async def test_megatron_service_start_openai_server_dedicated_starts_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "0000"
    checkpoint_dir.mkdir(parents=True)
    service = MegatronService(
        model_name="megatron-dedicated",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            rollout_weights_mode="lora",
        ),
        output_dir=str(tmp_path),
    )
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        "art.megatron.service.get_last_checkpoint_dir",
        lambda _output_dir: str(checkpoint_dir),
    )
    monkeypatch.setattr(service, "_ensure_identity_lora", lambda _path: None)
    monkeypatch.setattr(
        service, "_ensure_lora_adapter_config", lambda _path, source_path=None: None
    )

    async def fake_start_vllm_subprocess(
        lora_path: str,
        port: int,
        config: dict[str, Any] | None,
    ) -> tuple[str, int]:
        seen["lora_path"] = lora_path
        seen["port"] = port
        seen["config"] = config
        return ("127.0.0.1", port)

    monkeypatch.setattr(service, "_start_vllm_subprocess", fake_start_vllm_subprocess)

    location = await service.start_openai_server({"server_args": {"port": 8123}})

    assert location == ("127.0.0.1", 8123)
    assert seen["lora_path"] == str(checkpoint_dir)
    assert seen["port"] == 8123


@pytest.mark.asyncio
async def test_megatron_service_register_lora_for_step_dedicated_reloads_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="megatron-dedicated",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            rollout_weights_mode="lora",
        ),
        output_dir=str(tmp_path),
    )
    seen: list[tuple[str, int]] = []

    monkeypatch.setattr(
        service,
        "_reload_adapter",
        lambda checkpoint_dir, step: (
            seen.append((checkpoint_dir, step)) or asyncio.sleep(0)
        ),
    )

    await service.register_lora_for_step(3, "/tmp/checkpoints/3")

    assert seen == [("/tmp/checkpoints/3", 3)]
