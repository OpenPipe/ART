from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import socket
import tempfile
from typing import cast

from pydantic import BaseModel, Field
import torch

import art
from art import dev
from art.megatron.service import MegatronService
from art.utils.output_dirs import get_step_checkpoint_dir

from ..model_support.oracle_harness import (
    ORACLE_TOPOLOGY,
    OracleCaseConfig,
    ensure_case_artifacts,
)
from ..model_support.oracle_worker import provider_topology_env
from ..model_support.workflow_resources import (
    handler_workflow_resources_for_base_model,
    resolve_stage_resources_for_visible_gpus,
    validate_dedicated_test_resources,
)

_TRAINER_GPU_IDS_ENV = "ART_MODEL_SUPPORT_TRAINER_GPU_IDS"
_INFERENCE_GPU_IDS_ENV = "ART_MODEL_SUPPORT_INFERENCE_GPU_IDS"


class NativeVllmLoraServingReport(BaseModel):
    base_model: str
    output_dir: str
    host: str
    port: int
    trainer_gpu_ids: list[int]
    inference_gpu_ids: list[int]
    rollout_weights_mode: str = "lora"
    step0_name: str
    step1_name: str
    model_ids_before: list[str] = Field(default_factory=list)
    model_ids_after: list[str] = Field(default_factory=list)
    step0_served: bool
    step1_served: bool
    step0_completion_text: str = ""
    step1_completion_text: str = ""


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_gpu_id_env(name: str) -> list[int] | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _resolve_dedicated_gpu_ids() -> tuple[list[int], list[int]]:
    trainer_gpu_ids = _parse_gpu_id_env(_TRAINER_GPU_IDS_ENV)
    inference_gpu_ids = _parse_gpu_id_env(_INFERENCE_GPU_IDS_ENV)
    if trainer_gpu_ids is not None or inference_gpu_ids is not None:
        if trainer_gpu_ids is None or inference_gpu_ids is None:
            raise RuntimeError(
                f"{_TRAINER_GPU_IDS_ENV} and {_INFERENCE_GPU_IDS_ENV} must both be set"
            )
        return trainer_gpu_ids, inference_gpu_ids

    visible_gpu_count = int(torch.cuda.device_count())
    if visible_gpu_count < 2:
        raise RuntimeError(
            f"Need at least 2 visible GPUs for native LoRA serving, found {visible_gpu_count}"
        )
    return [0], [1]


async def _model_ids(client, base_url: str) -> list[str]:
    response = await client.get(f"{base_url}/v1/models", timeout=60.0)
    response.raise_for_status()
    return [
        str(model_info["id"])
        for model_info in response.json().get("data", [])
        if isinstance(model_info, dict) and "id" in model_info
    ]


async def _completion_text(client, base_url: str, model_name: str) -> str:
    response = await client.post(
        f"{base_url}/v1/completions",
        json={
            "model": model_name,
            "prompt": "Hello",
            "max_tokens": 1,
            "temperature": 0.0,
        },
        timeout=900.0,
    )
    response.raise_for_status()
    return str(response.json().get("choices", [{}])[0].get("text", ""))


def _copy_adapter_checkpoint(source_dir: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy(Path(source_dir) / filename, Path(dest_dir) / filename)


def _init_runtime_config(case_config: OracleCaseConfig) -> None:
    art.init_megatron_runtime_config(
        topology=art.MegatronTopologyConfig(
            tp=ORACLE_TOPOLOGY.tp,
            cp=ORACLE_TOPOLOGY.cp,
            ep=ORACLE_TOPOLOGY.ep,
            pp=ORACLE_TOPOLOGY.pp,
            etp=ORACLE_TOPOLOGY.etp,
        ),
        packed_sequence_length=case_config.packed_tensors.sequence_length,
    )


async def _run_native_vllm_lora(
    case_config: OracleCaseConfig,
) -> NativeVllmLoraServingReport:
    workflow_resources = handler_workflow_resources_for_base_model(
        case_config.base_model,
        allow_unvalidated_arch=case_config.allow_unvalidated_arch,
    )
    stage_resources = (
        workflow_resources.native_vllm_lora if workflow_resources is not None else None
    )
    if stage_resources is not None:
        stage_resources = resolve_stage_resources_for_visible_gpus(
            "native_vllm_lora",
            stage_resources,
            visible_gpu_count=int(torch.cuda.device_count()),
        )
        if stage_resources.vllm is None:
            raise RuntimeError("native_vllm_lora resources require vLLM")
        trainer_gpu_ids = [0]
        inference_gpu_ids = list(stage_resources.vllm.gpu_ids)
        validate_dedicated_test_resources(
            stage_name="native_vllm_lora",
            trainer_gpu_ids=trainer_gpu_ids,
            inference_gpu_ids=inference_gpu_ids,
            allow_overlap=True,
        )
        engine_args = cast(dev.EngineArgs, stage_resources.vllm.engine_args())
    else:
        trainer_gpu_ids, inference_gpu_ids = _resolve_dedicated_gpu_ids()
        engine_args = dev.EngineArgs()
    service_name = "model_support_native_lora_validation"
    case_artifacts = ensure_case_artifacts(case_config)
    output_root = Path(case_artifacts.case_dir) / "native_vllm_lora"
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = tempfile.mkdtemp(prefix="run_", dir=output_root)
    internal_config = dev.InternalModelConfig(
        trainer_gpu_ids=trainer_gpu_ids,
        inference_gpu_ids=inference_gpu_ids,
        rollout_weights_mode="lora",
        allow_unvalidated_arch=case_config.allow_unvalidated_arch,
        engine_args=engine_args,
    )
    if stage_resources is None:
        dev.validate_dedicated_config(internal_config)
    with provider_topology_env(ORACLE_TOPOLOGY):
        _init_runtime_config(case_config)
        service = MegatronService(
            model_name=service_name,
            base_model=case_config.base_model,
            config=internal_config,
            output_dir=output_dir,
        )
        port = _find_free_port()
        try:
            host, resolved_port = await service.start_openai_server(
                {"server_args": {"port": port}}
            )
            import httpx

            base_url = f"http://{host}:{resolved_port}"
            step0_name = f"{service_name}@0"
            step1_name = f"{service_name}@1"
            async with httpx.AsyncClient() as client:
                model_ids_before = await _model_ids(client, base_url)
                step0_completion_text = await _completion_text(
                    client,
                    base_url,
                    step0_name,
                )
                step0_dir = get_step_checkpoint_dir(output_dir, 0)
                step1_dir = get_step_checkpoint_dir(output_dir, 1)
                _copy_adapter_checkpoint(step0_dir, step1_dir)
                await service.register_lora_for_step(1, step1_dir)
                model_ids_after = await _model_ids(client, base_url)
                step1_completion_text = await _completion_text(
                    client,
                    base_url,
                    step1_name,
                )

            return NativeVllmLoraServingReport(
                base_model=case_config.base_model,
                output_dir=output_dir,
                host=host,
                port=resolved_port,
                trainer_gpu_ids=trainer_gpu_ids,
                inference_gpu_ids=inference_gpu_ids,
                step0_name=step0_name,
                step1_name=step1_name,
                model_ids_before=model_ids_before,
                model_ids_after=model_ids_after,
                step0_served=True,
                step1_served=True,
                step0_completion_text=step0_completion_text,
                step1_completion_text=step1_completion_text,
            )
        finally:
            service.close()


def run_native_vllm_lora(
    case_config: OracleCaseConfig,
) -> NativeVllmLoraServingReport:
    return asyncio.run(_run_native_vllm_lora(case_config))
