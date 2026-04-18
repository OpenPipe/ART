import asyncio
from contextlib import nullcontext
import os
from pathlib import Path
import shlex
import sys
import types as pytypes
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("vllm")

from art import TrainableModel, types
from art.dev.model import InternalModelConfig
from art.dev.validate import QWEN3_5_MOE_MODELS
from art.megatron.backend import MegatronBackend
from art.megatron.jobs import (
    MegatronMergedTrainJob,
    MergedWeightTransferInitInfo,
)
from art.megatron.service import MegatronService, create_identity_lora
from art.megatron.train import _compile_enabled, _unwrap_art_wrapper_name


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
        start_new_session: bool,
    ) -> Any:
        seen["command"] = command
        seen["cwd"] = cwd
        seen["env"] = env
        seen["start_new_session"] = start_new_session
        return pytypes.SimpleNamespace(returncode=None)

    monkeypatch.setattr(
        "art.megatron.service.asyncio.create_subprocess_shell",
        fake_create_subprocess_shell,
    )

    await service._ensure_megatron_running()

    assert "uv run --project" in seen["command"]
    assert "torchrun" in seen["command"]
    assert "--nproc_per_node 2" in seen["command"]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert seen["env"]["MODEL_IDENTIFIER"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert seen["start_new_session"] is True


def test_unwrap_art_wrapper_name_strips_compiled_wrapper_segments() -> None:
    assert (
        _unwrap_art_wrapper_name(
            "module.module.decoder.layers.0._orig_mod.self_attention.linear_proj.linear_proj.weight"
        )
        == "decoder.layers.0.self_attention.linear_proj.weight"
    )
    assert (
        _unwrap_art_wrapper_name(
            "module.module.decoder.layers.0._orig_mod.mlp.experts.linear_fc1.linear_fc1.weight7"
        )
        == "decoder.layers.0.mlp.experts.linear_fc1.weight7"
    )


def test_compile_enabled_disables_qwen35_moe_by_default() -> None:
    assert _compile_enabled("Qwen/Qwen3-30B-A3B-Instruct-2507") is True
    assert _compile_enabled("Qwen/Qwen3.5-32B-Instruct") is True
    for model_identifier in QWEN3_5_MOE_MODELS:
        assert _compile_enabled(model_identifier) is False


def test_compile_enabled_honors_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ART_DISABLE_MEGATRON_COMPILE", "0")
    assert _compile_enabled("Qwen/Qwen3.5-35B-A3B") is True
    monkeypatch.setenv("ART_DISABLE_MEGATRON_COMPILE", "1")
    assert _compile_enabled("Qwen/Qwen3-30B-A3B-Instruct-2507") is False


def test_create_identity_lora_uses_nested_text_config_when_top_level_lacks_vocab_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    top_level_config = SimpleNamespace(
        text_config=SimpleNamespace(vocab_size=128),
    )
    seen: dict[str, Any] = {}

    class FakeModel:
        name_or_path = ""

        def named_parameters(self) -> list[tuple[str, torch.Tensor]]:
            return [
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    torch.empty(1, device="meta"),
                ),
                (
                    "model.layers.0.linear_attn.in_proj_qkv.weight",
                    torch.empty(1, device="meta"),
                ),
            ]

    class FakePeftModel:
        def save_pretrained(self, lora_path: str) -> None:
            Path(lora_path).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: top_level_config,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_config",
        lambda config, **_kwargs: seen.setdefault("config", config) or FakeModel(),
    )
    monkeypatch.setattr("accelerate.init_empty_weights", nullcontext)
    monkeypatch.setattr(
        "peft.get_peft_model",
        lambda _model, lora_config, **_kwargs: (
            seen.setdefault("lora_config", lora_config) or FakePeftModel()
        ),
    )
    monkeypatch.setattr(
        "art.megatron.service.convert_checkpoint_if_needed",
        lambda _path: None,
    )

    create_identity_lora("Qwen/Qwen3.5-35B-A3B", str(tmp_path))

    assert seen["config"] is top_level_config.text_config
    assert (
        "model.layers.0.linear_attn.in_proj_qkv.weight"
        in seen["lora_config"].target_parameters
    )


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


@pytest.mark.asyncio
async def test_megatron_service_start_openai_server_merged_syncs_step_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "0000"
    checkpoint_dir.mkdir(parents=True)
    service = MegatronService(
        model_name="megatron-merged",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            rollout_weights_mode="merged",
        ),
        output_dir=str(tmp_path),
    )
    calls: list[object] = []
    ensured_identity_paths: list[str] = []

    monkeypatch.setattr(
        "art.megatron.service.get_last_checkpoint_dir",
        lambda _output_dir: str(checkpoint_dir),
    )
    monkeypatch.setattr(
        service,
        "_ensure_identity_lora",
        lambda path: ensured_identity_paths.append(path),
    )
    monkeypatch.setattr(service, "_ensure_lora_adapter_config", lambda _path: None)
    monkeypatch.setattr(
        service,
        "_start_vllm_subprocess",
        lambda lora_path, port, config: asyncio.sleep(0, result=("127.0.0.1", port)),
    )
    monkeypatch.setattr(service, "_clear_pending_jobs", lambda: calls.append("clear"))
    monkeypatch.setattr(
        service,
        "_sync_dedicated_merged_weights",
        lambda *, lora_path, step: calls.append((lora_path, step)) or asyncio.sleep(0),
    )

    location = await service.start_openai_server({"server_args": {"port": 8123}})

    assert location == ("127.0.0.1", 8123)
    assert ensured_identity_paths == [str(checkpoint_dir)]
    assert calls == ["clear", (str(checkpoint_dir), 0)]


@pytest.mark.asyncio
async def test_megatron_service_start_openai_server_shared_lora_bootstraps_step_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "0000"
    checkpoint_dir.mkdir(parents=True)
    service = MegatronService(
        model_name="megatron-shared",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config=InternalModelConfig(),
        output_dir=str(tmp_path),
    )
    ensured_identity_paths: list[str] = []

    monkeypatch.setattr(
        "art.megatron.service.get_last_checkpoint_dir",
        lambda _output_dir: str(checkpoint_dir),
    )
    monkeypatch.setattr(
        service,
        "_ensure_identity_lora",
        lambda path: ensured_identity_paths.append(path),
    )
    monkeypatch.setattr(service, "_ensure_lora_adapter_config", lambda _path: None)
    monkeypatch.setattr(
        "art.megatron.service.dev.get_openai_server_config",
        lambda **_kwargs: {"server_args": {"port": 8123}, "engine_args": {}},
    )
    monkeypatch.setattr(
        "art.megatron.service.openai_server_task",
        lambda **_kwargs: asyncio.sleep(0),
    )

    location = await service.start_openai_server({"server_args": {"port": 8123}})

    assert location == ("0.0.0.0", 8123)
    assert ensured_identity_paths == [str(checkpoint_dir)]


@pytest.mark.asyncio
async def test_megatron_service_register_lora_for_step_merged_sets_served_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MegatronService(
        model_name="megatron-merged",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            rollout_weights_mode="merged",
        ),
        output_dir=str(tmp_path),
    )
    calls: list[int] = []

    monkeypatch.setattr(
        service,
        "_set_served_model_name",
        lambda step: calls.append(step) or asyncio.sleep(0),
    )

    await service.register_lora_for_step(3, "/tmp/checkpoints/3")

    assert calls == [3]


@pytest.mark.asyncio
async def test_megatron_service_train_merged_writes_merged_job_and_does_not_reload_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "0000"
    checkpoint_dir.mkdir(parents=True)
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    adapter_path.write_bytes(b"adapter")
    service = MegatronService(
        model_name="megatron-merged",
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            rollout_weights_mode="merged",
        ),
        output_dir=str(tmp_path),
    )
    events: list[Any] = []

    monkeypatch.setattr(
        service,
        "_ensure_megatron_running",
        lambda: events.append("ensure") or asyncio.sleep(0),
    )
    monkeypatch.setattr(
        service, "_resolve_active_lora_path", lambda: str(checkpoint_dir)
    )
    monkeypatch.setattr(
        service,
        "_init_merged_weight_transfer",
        lambda: events.append("init") or asyncio.sleep(0),
    )
    monkeypatch.setattr(
        "art.megatron.service.create_megatron_job_paths",
        lambda *, jobs_dir, training_log_dir: ("/tmp/job.json", "/tmp/log.jsonl"),
    )
    monkeypatch.setattr(
        "art.megatron.service.write_megatron_job",
        lambda job, *, job_path: events.append(job),
    )

    async def fake_stream_megatron_job(job: Any, *, job_path: str):
        events.append(("stream", job_path, job.lora_path))
        yield {"loss": 1.0}

    monkeypatch.setattr(
        "art.megatron.service.stream_megatron_job",
        fake_stream_megatron_job,
    )
    monkeypatch.setattr(
        service,
        "_ensure_lora_adapter_config",
        lambda _path, source_path=None: None,
    )
    monkeypatch.setattr(
        service,
        "_reload_adapter",
        lambda checkpoint_dir, step: (_ for _ in ()).throw(
            AssertionError("merged mode should not hot-reload a LoRA adapter")
        ),
    )
    service._merged_weight_transfer_init_info = MergedWeightTransferInitInfo(
        master_address="127.0.0.1",
        master_port=1234,
        rank_offset=1,
        world_size=2,
    )

    results = []
    async for result in service.train(
        {"dir": "/tmp/tensors", "num_sequences": 1, "sequence_length": 16},
        types.TrainConfig(learning_rate=5e-5),
        {},
    ):
        results.append(result)

    assert results == [{"loss": 1.0}]
    assert events[0:2] == ["ensure", "init"]
    job = events[2]
    assert isinstance(job, MegatronMergedTrainJob)
    assert job.merged_weight_transfer.served_model_name == "megatron-merged@1"
    assert events[3] == ("stream", "/tmp/job.json", str(checkpoint_dir))
