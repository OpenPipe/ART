import argparse
import asyncio
from collections.abc import Callable
import json
import os
from pathlib import Path
import time
import traceback

import httpx
from pydantic import BaseModel, ConfigDict

from art.megatron.model_support.spec import ArchitectureReport
from art.serving_capabilities import FastMetricsSnapshot
from art.utils.lifecycle import ChildProcessSupervisor
from art.utils.network import find_free_tcp_port
from art.vllm_runtime import ManagedVllmRuntime, VllmRuntimeLaunchConfig

from .workflow import (
    run_chat_template_rollout_stage,
    run_correctness_sensitivity_stage,
    run_e2e_throughput_stage,
    run_hf_parity_stage,
    run_length_trainability_stage,
    run_lora_coverage_stage,
    run_merged_vllm_serving_stage,
    run_native_vllm_lora_stage,
    run_packing_invariance_stage,
    run_train_inf_mismatch_stage,
    run_yes_no_trainability_stage,
)
from .workflow_fixtures import FIXTURE_PATH_ENV, ensure_workflow_fixture

FUNCTIONAL_LORA_VLLM_MODE = "functional_lora_vllm"
FUNCTIONAL_LORA_VLLM_STAGES = ("length_trainability", "native_vllm_lora")
BASE_MEGATRON_MODE = "base_megatron"
BASE_MEGATRON_STAGES = ("hf_parity", "packing_invariance")
EXTERNAL_VLLM_ENGINE_ARGS_ENV = "ART_MODEL_SUPPORT_EXTERNAL_VLLM_ENGINE_ARGS"
_STAGE_RUNNERS = {
    "hf_parity": run_hf_parity_stage,
    "lora_coverage": run_lora_coverage_stage,
    "train_inf_mismatch": run_train_inf_mismatch_stage,
    "merged_vllm_serving": run_merged_vllm_serving_stage,
    "correctness_sensitivity": run_correctness_sensitivity_stage,
    "chat_template_rollout": run_chat_template_rollout_stage,
    "packing_invariance": run_packing_invariance_stage,
    "length_trainability": run_length_trainability_stage,
    "e2e_throughput": run_e2e_throughput_stage,
    "yes_no_trainability": run_yes_no_trainability_stage,
    "native_vllm_lora": run_native_vllm_lora_stage,
}


class WorkflowStageWorkerItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    stage_dir: str
    output_json: str
    environment: dict[str, str]


class FunctionalVllmSessionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu_count: int
    launch: VllmRuntimeLaunchConfig
    trainer_gpu_ids: dict[str, tuple[int, ...]]


class WorkflowStageWorkerSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_model: str
    architecture_json: str
    allow_unvalidated_arch: bool = False
    functional_vllm: FunctionalVllmSessionSpec | None = None
    base_megatron: bool = False
    items: tuple[WorkflowStageWorkerItem, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-json")
    parser.add_argument("--stage")
    parser.add_argument("--base-model")
    parser.add_argument("--architecture-json")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--allow-unsupported-arch",
        dest="allow_unvalidated_arch",
        action="store_true",
    )
    args = parser.parse_args()
    if args.session_json is None and not all(
        (args.stage, args.base_model, args.architecture_json, args.output_json)
    ):
        parser.error(
            "--session-json or --stage/--base-model/--architecture-json/--output-json "
            "is required"
        )
    return args


def _runtime_json(client: httpx.Client, method: str, path: str, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def _reset_vllm(client: httpx.Client, baseline: tuple[str, ...]) -> dict[str, object]:
    def model_ids() -> tuple[str, ...]:
        models = _runtime_json(client, "GET", "/v1/models")["data"]
        return tuple(sorted(str(model["id"]) for model in models))

    def idle() -> dict[str, float]:
        for _ in range(600):
            metrics = FastMetricsSnapshot.model_validate(
                _runtime_json(client, "GET", "/art/metrics")
            ).metrics
            if not any(
                value
                for key, value in metrics.items()
                if key.startswith("num_requests_")
            ):
                return {
                    key: float(value)
                    for key, value in metrics.items()
                    if key.startswith("num_requests_")
                }
            time.sleep(0.1)
        raise TimeoutError("functional vLLM requests did not drain")

    idle_before = idle()
    before = model_ids()
    aliases = tuple(sorted(set(before) - set(baseline)))
    for alias in aliases:
        _runtime_json(
            client,
            "POST",
            "/v1/unload_lora_adapter",
            json={"lora_name": alias},
        )
    reset = _runtime_json(
        client,
        "POST",
        "/art/reset_prefix_cache",
        json={"reset_running_requests": False, "reset_connector": True},
    )
    idle_after = idle()
    if reset.get("success") is not True or (after := model_ids()) != baseline:
        raise RuntimeError(
            f"functional reset failed: baseline={baseline}, after={after}"
        )
    return {
        "baseline_model_ids": baseline,
        "model_ids_before": before,
        "unloaded_aliases": aliases,
        "prefix_cache_reset": True,
        "model_ids_after": after,
        "requests_before": idle_before,
        "requests_after": idle_after,
    }


async def _run_functional_session(request: WorkflowStageWorkerSession) -> None:
    from . import workflow

    spec = request.functional_vllm
    assert spec is not None
    launch_spec = spec.launch
    stages = tuple(item.stage for item in request.items)
    if stages != FUNCTIONAL_LORA_VLLM_STAGES:
        raise ValueError("functional vLLM stages do not match worker items")
    if tuple(spec.trainer_gpu_ids) != stages:
        raise ValueError("functional vLLM trainer placements do not match worker items")
    prepared = ensure_workflow_fixture(
        request.base_model,
        allow_unvalidated_arch=request.allow_unvalidated_arch,
        required_stages=frozenset(FUNCTIONAL_LORA_VLLM_STAGES),
    )
    host_environments = {stage: prepared.environment(stage) for stage in stages}
    host_paths = {
        environment[FIXTURE_PATH_ENV] for environment in host_environments.values()
    }
    if len(host_paths) != 1:
        raise RuntimeError(f"selected host prepared different fixtures: {host_paths}")
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible) != spec.gpu_count:
        raise RuntimeError(
            f"functional vLLM expected {spec.gpu_count} GPUs, received {len(visible)}"
        )
    inference_gpu_ids = tuple(map(int, launch_spec.visible_devices.split(",")))
    launch = launch_spec.model_copy(
        update={
            "base_model": host_paths.pop(),
            "port": find_free_tcp_port(),
            "cuda_visible_devices": ",".join(
                visible[index] for index in inference_gpu_ids
            ),
        }
    )
    runtime = ManagedVllmRuntime()
    supervisor = ChildProcessSupervisor(lambda _error: None)
    try:
        with workflow._temporary_env(**host_environments[stages[0]]):
            await runtime.start(
                launch_config=launch,
                output_dir=str(
                    Path(request.architecture_json).parent / "functional_vllm"
                ),
                child_processes=supervisor,
                install_parent_cleanup=lambda: None,
            )
        assert runtime.api_key is not None
        external = {
            "ART_MODEL_SUPPORT_EXTERNAL_VLLM_URL": runtime.base_url,
            "ART_MODEL_SUPPORT_EXTERNAL_VLLM_API_KEY": runtime.api_key,
            "ART_MODEL_SUPPORT_INFERENCE_GPU_IDS": ",".join(
                map(str, inference_gpu_ids)
            ),
            EXTERNAL_VLLM_ENGINE_ARGS_ENV: json.dumps(
                launch_spec.engine_args, sort_keys=True
            ),
        }
        for stage, trainer_gpu_ids in spec.trainer_gpu_ids.items():
            if (
                not trainer_gpu_ids
                or set(trainer_gpu_ids) & set(inference_gpu_ids)
                or any(gpu_id not in range(len(visible)) for gpu_id in trainer_gpu_ids)
            ):
                raise RuntimeError(
                    f"invalid functional GPU partition for {stage}: "
                    f"trainer={trainer_gpu_ids}, inference={inference_gpu_ids}"
                )
        request = request.model_copy(
            update={
                "functional_vllm": None,
                "items": tuple(
                    item.model_copy(
                        update={
                            "environment": item.environment
                            | host_environments[item.stage]
                            | external
                            | {
                                "ART_MODEL_SUPPORT_TRAINER_GPU_IDS": ",".join(
                                    map(str, spec.trainer_gpu_ids[item.stage])
                                )
                            }
                        }
                    )
                    for item in request.items
                ),
            }
        )
        with httpx.Client(
            base_url=runtime.base_url, **runtime.request_kwargs()
        ) as client:
            models = _runtime_json(client, "GET", "/v1/models")["data"]
            baseline = tuple(sorted(str(model["id"]) for model in models))

            def reset() -> dict[str, object]:
                supervisor.raise_if_failed()
                return _reset_vllm(client, baseline)

            await asyncio.to_thread(_run_session, request, reset)
    finally:
        supervisor.close()
        runtime.close()


def _run_session(
    request: WorkflowStageWorkerSession,
    after_stage: Callable[[], dict[str, object]] | None = None,
) -> None:
    from . import workflow

    architecture = ArchitectureReport.model_validate_json(
        Path(request.architecture_json).read_text(encoding="utf-8")
    )
    for item in request.items:
        started = time.monotonic()
        log_path = Path(item.stage_dir) / "worker.log"
        reset_error: Exception | None = None
        try:
            with workflow._temporary_env(
                **item.environment,
                **{workflow.WORKFLOW_STAGE_DIR_ENV: item.stage_dir},
            ):
                with workflow._redirect_output(log_path):
                    result = _STAGE_RUNNERS[item.stage](
                        base_model=request.base_model,
                        architecture=architecture,
                        allow_unvalidated_arch=request.allow_unvalidated_arch,
                    )
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log:
                traceback.print_exc(file=log)
            result = workflow.ValidationStageResult(
                name=item.stage,
                passed=False,
                metrics=workflow._stage_error_metrics(exc),
            )
        if after_stage is not None:
            try:
                result.metrics["functional_vllm_reset"] = after_stage()
            except Exception as exc:
                with log_path.open("a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
                result.passed = False
                result.metrics["functional_vllm_reset_error"] = (
                    workflow._stage_error_metrics(exc)
                )
                reset_error = exc
        result.metrics.update(
            {
                "workflow_stage_artifact_dir": item.stage_dir,
                "workflow_stage_duration_s": time.monotonic() - started,
            }
        )
        Path(item.output_json).write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        if reset_error is not None:
            raise RuntimeError(
                f"functional vLLM reset failed after {item.stage}"
            ) from reset_error


def _run_base_megatron_session(request: WorkflowStageWorkerSession) -> None:
    stages = tuple(item.stage for item in request.items)
    if stages != BASE_MEGATRON_STAGES:
        raise ValueError("base Megatron stages do not match worker items")
    from .base_megatron_session import base_megatron_session

    with base_megatron_session():
        _run_session(request.model_copy(update={"base_megatron": False}))


def main() -> None:
    args = _parse_args()
    if args.session_json is not None:
        request = WorkflowStageWorkerSession.model_validate_json(
            Path(args.session_json).read_text(encoding="utf-8")
        )
        if request.functional_vllm is not None:
            asyncio.run(_run_functional_session(request))
        elif request.base_megatron:
            _run_base_megatron_session(request)
        else:
            _run_session(request)
        return
    architecture = ArchitectureReport.model_validate_json(
        Path(args.architecture_json).read_text(encoding="utf-8")
    )
    result = _STAGE_RUNNERS[args.stage](
        base_model=args.base_model,
        architecture=architecture,
        allow_unvalidated_arch=args.allow_unvalidated_arch,
    )
    Path(args.output_json).write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
