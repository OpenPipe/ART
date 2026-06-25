from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket

from pydantic import BaseModel, Field
import torch

import art
from art import dev
from art.megatron.service import MegatronService

from ..model_support.oracle_harness import (
    ORACLE_TOPOLOGY,
    OracleCaseConfig,
    ensure_case_artifacts,
)
from ..model_support.oracle_worker import provider_topology_env

_TRAINER_GPU_IDS_ENV = "ART_MODEL_SUPPORT_TRAINER_GPU_IDS"
_INFERENCE_GPU_IDS_ENV = "ART_MODEL_SUPPORT_INFERENCE_GPU_IDS"


class MergedVllmServingReport(BaseModel):
    base_model: str
    output_dir: str
    host: str
    port: int
    trainer_gpu_ids: list[int]
    inference_gpu_ids: list[int]
    served_model_name: str
    model_ids: list[str] = Field(default_factory=list)
    completion_text: str = ""


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
            f"Need at least 2 visible GPUs for merged serving, found {visible_gpu_count}"
        )
    return [0], [1]


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


async def _run_merged_vllm_serving(
    case_config: OracleCaseConfig,
) -> MergedVllmServingReport:
    trainer_gpu_ids, inference_gpu_ids = _resolve_dedicated_gpu_ids()
    service_name = "model_support_merged_validation"
    case_artifacts = ensure_case_artifacts(case_config)
    output_dir = str(Path(case_artifacts.case_dir) / "merged_vllm_serving")
    os.makedirs(output_dir, exist_ok=True)
    internal_config = dev.InternalModelConfig(
        trainer_gpu_ids=trainer_gpu_ids,
        inference_gpu_ids=inference_gpu_ids,
        rollout_weights_mode="merged",
        allow_unvalidated_arch=case_config.allow_unvalidated_arch,
    )
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

            async with httpx.AsyncClient() as client:
                models_response = await client.get(
                    f"http://{host}:{resolved_port}/v1/models",
                    timeout=60.0,
                )
                models_response.raise_for_status()
                model_ids = [
                    str(model_info["id"])
                    for model_info in models_response.json().get("data", [])
                    if isinstance(model_info, dict) and "id" in model_info
                ]

                served_model_name = f"{service_name}@{service._latest_step}"
                completion_response = await client.post(
                    f"http://{host}:{resolved_port}/v1/completions",
                    json={
                        "model": served_model_name,
                        "prompt": "Hello",
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                    timeout=900.0,
                )
                completion_response.raise_for_status()
                completion_json = completion_response.json()
                completion_text = str(
                    completion_json.get("choices", [{}])[0].get("text", "")
                )
            return MergedVllmServingReport(
                base_model=case_config.base_model,
                output_dir=output_dir,
                host=host,
                port=resolved_port,
                trainer_gpu_ids=trainer_gpu_ids,
                inference_gpu_ids=inference_gpu_ids,
                served_model_name=served_model_name,
                model_ids=model_ids,
                completion_text=completion_text,
            )
        finally:
            service.close()


def run_merged_vllm_serving(
    case_config: OracleCaseConfig,
) -> MergedVllmServingReport:
    return asyncio.run(_run_merged_vllm_serving(case_config))
