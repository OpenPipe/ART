from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import cast
from urllib.parse import urlparse

from pydantic import BaseModel, Field
import torch

import art
from art import dev
from art.megatron.backend import MegatronBackend
from art.megatron.distributed_service import DistributedMegatronService

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
_EXTERNAL_VLLM_URL_ENV = "ART_MODEL_SUPPORT_EXTERNAL_VLLM_URL"
_EXTERNAL_VLLM_API_KEY_ENV = "ART_MODEL_SUPPORT_EXTERNAL_VLLM_API_KEY"
_EXTERNAL_VLLM_ENGINE_ARGS_ENV = "ART_MODEL_SUPPORT_EXTERNAL_VLLM_ENGINE_ARGS"


class NativeVllmLoraServingReport(BaseModel):
    base_model: str
    output_dir: str
    host: str
    port: int
    trainer_gpu_ids: list[int]
    inference_gpu_ids: list[int]
    rollout_weights_mode: str = "lora"
    external_vllm_reused: bool = False
    vllm_engine_args: dict[str, object] = Field(default_factory=dict)
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


async def _model_ids(client) -> list[str]:
    return [model_info.id for model_info in (await client.models.list()).data]


async def _completion_text(client, model_name: str) -> str:
    response = await client.completions.create(
        model=model_name,
        prompt=[100],
        max_tokens=1,
        temperature=0.0,
        timeout=900.0,
    )
    return response.choices[0].text


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
    configured_gpu_ids = any(
        _parse_gpu_id_env(name) is not None
        for name in (_TRAINER_GPU_IDS_ENV, _INFERENCE_GPU_IDS_ENV)
    )
    if stage_resources is not None:
        stage_resources = resolve_stage_resources_for_visible_gpus(
            "native_vllm_lora",
            stage_resources,
            visible_gpu_count=int(torch.cuda.device_count()),
        )
        if stage_resources.vllm is None:
            raise RuntimeError("native_vllm_lora resources require vLLM")
        engine_args = cast(dev.EngineArgs, stage_resources.vllm.engine_args())
    else:
        engine_args = dev.EngineArgs(enforce_eager=True)
    external_url = os.environ.get(_EXTERNAL_VLLM_URL_ENV)
    external_api_key = os.environ.get(_EXTERNAL_VLLM_API_KEY_ENV)
    if (external_url is None) != (external_api_key is None):
        raise RuntimeError(
            f"{_EXTERNAL_VLLM_URL_ENV} and {_EXTERNAL_VLLM_API_KEY_ENV} must both be set"
        )
    if raw_engine_args := os.environ.get(_EXTERNAL_VLLM_ENGINE_ARGS_ENV):
        if external_url is None:
            raise RuntimeError(
                f"{_EXTERNAL_VLLM_ENGINE_ARGS_ENV} requires {_EXTERNAL_VLLM_URL_ENV}"
            )
        parsed_engine_args = json.loads(raw_engine_args)
        if not isinstance(parsed_engine_args, dict):
            raise RuntimeError(
                f"{_EXTERNAL_VLLM_ENGINE_ARGS_ENV} must encode an object"
            )
        engine_args = cast(dev.EngineArgs, parsed_engine_args)
    if configured_gpu_ids:
        trainer_gpu_ids, inference_gpu_ids = _resolve_dedicated_gpu_ids()
    elif stage_resources is not None:
        assert stage_resources.vllm is not None
        trainer_gpu_ids = [0]
        inference_gpu_ids = list(stage_resources.vllm.gpu_ids)
    else:
        trainer_gpu_ids, inference_gpu_ids = _resolve_dedicated_gpu_ids()
    validate_dedicated_test_resources(
        stage_name="native_vllm_lora",
        trainer_gpu_ids=trainer_gpu_ids,
        inference_gpu_ids=inference_gpu_ids,
        allow_overlap=True,
    )
    service_name = "model_support_native_lora_validation"
    case_artifacts = ensure_case_artifacts(case_config)
    output_root = Path(case_artifacts.case_dir) / "native_vllm_lora"
    output_root.mkdir(parents=True, exist_ok=True)
    backend_root = tempfile.mkdtemp(prefix="run_", dir=output_root)
    internal_config = dev.InternalModelConfig(
        rollout_weights_mode="lora",
        allow_unvalidated_arch=case_config.allow_unvalidated_arch,
        engine_args=engine_args,
    )
    if external_url is not None:
        internal_config["vllm_runtime"] = {
            "mode": "external",
            "server_url": external_url,
            "api_key": external_api_key,
        }
    if set(trainer_gpu_ids).isdisjoint(inference_gpu_ids):
        internal_config["trainer_gpu_ids"] = trainer_gpu_ids
        internal_config["inference_gpu_ids"] = inference_gpu_ids
    else:
        trainer_gpu_ids = list(inference_gpu_ids)
    if stage_resources is None:
        dev.validate_dedicated_config(internal_config)
    with provider_topology_env(ORACLE_TOPOLOGY):
        _init_runtime_config(case_config)
        backend = MegatronBackend(path=backend_root)
        model = art.TrainableModel(
            name=service_name,
            run_name=service_name,
            project="model-support-validation",
            base_model=case_config.base_model,
            _internal_config=internal_config,
            report_metrics=[],
        )
        port = _find_free_port() if external_url is None else None
        try:
            await model.register(
                backend,
                {"server_args": {"port": port}} if port is not None else None,
            )
            service = cast(
                DistributedMegatronService, await backend._get_service(model)
            )
            endpoint = urlparse(model.inference_base_url or "")
            host, resolved_port = (
                endpoint.hostname or "127.0.0.1",
                int(endpoint.port or (443 if endpoint.scheme == "https" else 80)),
            )
            step0_name = model.get_inference_name(step=0)
            step1_name = model.get_inference_name(step=1)
            async with model.openai_client() as client:
                model_ids_before = await _model_ids(client)
                step0_completion_text = await _completion_text(client, step0_name)
                await service.advance_without_training(
                    expected_step=0,
                    learner_version=1,
                )
                await service.wait_for_serving(1)
                model_ids_after = await _model_ids(client)
                step1_completion_text = await _completion_text(client, step1_name)

            return NativeVllmLoraServingReport(
                base_model=case_config.base_model,
                output_dir=service.output_dir,
                host=host,
                port=resolved_port,
                trainer_gpu_ids=trainer_gpu_ids,
                inference_gpu_ids=inference_gpu_ids,
                external_vllm_reused=external_url is not None,
                vllm_engine_args=dict(engine_args),
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
            await backend.close()


def run_native_vllm_lora(
    case_config: OracleCaseConfig,
) -> NativeVllmLoraServingReport:
    return asyncio.run(_run_native_vllm_lora(case_config))
