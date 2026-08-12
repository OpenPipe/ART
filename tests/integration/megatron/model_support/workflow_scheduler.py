from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from threading import Lock
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from art.megatron.lora_config import (
    MEGATRON_LORA_RANK_ENV,
    default_lora_rank_for_handler,
)
from art.megatron.model_support.registry import (
    get_model_support_handler_for_spec,
    get_model_support_spec,
    model_uses_expert_parallel,
)
from art.megatron.model_support.spec import ArchitectureReport
from art.vllm_runtime import VllmRuntimeLaunchConfig

from .validation_spec import ValidationReport, ValidationStageResult
from .workflow_fixtures import (
    FIXTURE_PATH_ENV,
    WorkflowFixture,
    ensure_workflow_fixture,
)
from .workflow_resources import (
    HANDLER_WORKFLOW_RESOURCES,
    WorkflowStageResources,
    resolve_stage_resources_for_visible_gpus,
)
from .workflow_runtime import (
    WorkflowDevice,
    WorkflowOperation,
    WorkflowPlacement,
    WorkflowResourceRequest,
    WorkflowRuntimeKey,
    WorkflowSession,
    compile_workflow,
    execute_workflow,
)
from .workflow_stage_worker import (
    FUNCTIONAL_LORA_VLLM_MODE,
    FUNCTIONAL_LORA_VLLM_STAGES,
    WorkflowStageWorkerItem,
    WorkflowStageWorkerSession,
)

_REDUCED_FIXTURE_GPU_SHARE = 0.125
_STAGE_DURATION_ESTIMATES_S = {
    "hf_parity": 120.0,
    "lora_coverage": 60.0,
    "train_inf_mismatch": 360.0,
    "merged_vllm_serving": 180.0,
    "correctness_sensitivity": 600.0,
    "chat_template_rollout": 45.0,
    "packing_invariance": 90.0,
    "length_trainability": 360.0,
    "e2e_throughput": 360.0,
    "native_vllm_lora": 180.0,
    "yes_no_trainability": 360.0,
}
_LIGHTWEIGHT_GPU_STAGES = frozenset(
    {"hf_parity", "lora_coverage", "packing_invariance"}
)
_CPU_STAGES = frozenset({"chat_template_rollout"})
_DEFAULT_STAGE_GPU_COUNTS = {
    "train_inf_mismatch": 4,
    "merged_vllm_serving": 2,
    "native_vllm_lora": 2,
    "yes_no_trainability": 2,
}
_WORKFLOW_HOSTS_ENV = "ART_MODEL_SUPPORT_WORKFLOW_HOSTS"


class PreparedWorkflow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    report: ValidationReport
    architecture: ArchitectureReport
    fixture: WorkflowFixture
    run_dir: Path
    output_json: Path | None
    stages: tuple[str, ...]
    allow_unvalidated_arch: bool
    include_sensitivity: bool | None
    fixture_provisioning_s: float
    _fixture_metric_recorded: bool = PrivateAttr(default=False)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    def record(self, result: ValidationStageResult) -> None:
        with self._lock:
            stage = next(
                stage for stage in self.report.stages if stage.name == result.name
            )
            stage.passed = result.passed
            stage.skipped = result.skipped
            stage.metrics = dict(result.metrics)
            stage.artifact_dir = result.artifact_dir
            if self.output_json is not None:
                self.output_json.parent.mkdir(parents=True, exist_ok=True)
                self.output_json.write_text(
                    self.report.model_dump_json(indent=2), encoding="utf-8"
                )

    def record_fixture_metric(self, metrics: dict[str, Any]) -> None:
        with self._lock:
            if self._fixture_metric_recorded:
                return
            metrics["fixture_provisioning_s"] = self.fixture_provisioning_s
            self._fixture_metric_recorded = True


def prepare_workflow(
    *,
    base_model: str,
    include_native_vllm_lora: bool,
    include_yes_no_trainability: bool,
    include_sensitivity: bool | None,
    output_json: Path | None,
    skip_stages: set[str],
    allow_unvalidated_arch: bool,
) -> PreparedWorkflow:
    from . import workflow

    report = workflow.initialize_validation_report(
        base_model=base_model,
        include_native_vllm_lora=include_native_vllm_lora,
        include_yes_no_trainability=include_yes_no_trainability,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    run_dir = workflow._new_workflow_run_dir(
        output_json=output_json,
        model_key=report.model_key,
    )
    dependency = next(
        stage for stage in report.stages if stage.name == "dependency_resolution"
    )
    started = time.monotonic()
    dependency.passed = True
    dependency.metrics = dict(report.dependency_versions)
    workflow._record_stage_duration(dependency, started=started)

    architecture_stage = next(
        stage for stage in report.stages if stage.name == "architecture_discovery"
    )
    started = time.monotonic()
    architecture = workflow._inspect_architecture_for_workflow(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    architecture_stage.passed = not architecture.unresolved_risks
    architecture_stage.metrics = {
        "recommended_min_layers": architecture.recommended_min_layers,
        "layer_families": [
            family.model_dump() for family in architecture.layer_families
        ],
        "unresolved_risks": list(architecture.unresolved_risks),
    }
    workflow._record_stage_duration(architecture_stage, started=started)

    stages = tuple(
        stage.name
        for stage in report.stages
        if stage.name not in {"dependency_resolution", "architecture_discovery"}
        and stage.name not in skip_stages
    )
    for stage in report.stages:
        if stage.name not in skip_stages:
            continue
        stage.skipped = True
        stage.metrics = {
            "skipped": True,
            "reason": "--skip-stage",
            "workflow_stage_duration_s": 0.0,
        }
    fixture_started = time.monotonic()
    fixture = ensure_workflow_fixture(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
        required_stages=frozenset(stages),
    )
    prepared = PreparedWorkflow(
        report=report,
        architecture=architecture,
        fixture=fixture,
        run_dir=run_dir,
        output_json=output_json,
        stages=stages,
        allow_unvalidated_arch=allow_unvalidated_arch,
        include_sensitivity=include_sensitivity,
        fixture_provisioning_s=time.monotonic() - fixture_started,
    )
    workflow._write_validation_report(report, output_json)
    return prepared


def _minimum_gpu_count(
    stage_name: str,
    resources: WorkflowStageResources,
    *,
    visible_gpu_count: int,
) -> int:
    minimum = resources.required_physical_gpus or 1
    for count in range(minimum, visible_gpu_count + 1):
        try:
            resolve_stage_resources_for_visible_gpus(
                stage_name,
                resources,
                visible_gpu_count=count,
            )
        except RuntimeError:
            continue
        return count
    raise RuntimeError(
        f"{stage_name} does not fit any allocation up to {visible_gpu_count} GPUs"
    )


def _stage_gpu_count(
    prepared: PreparedWorkflow, stage_name: str, available: int
) -> int:
    if stage_name in _CPU_STAGES:
        return 0
    if stage_name in _LIGHTWEIGHT_GPU_STAGES:
        return 1
    if stage_name == "correctness_sensitivity":
        from .oracle_harness import selected_suite_topologies

        handler = get_model_support_handler_for_spec(
            get_model_support_spec(
                prepared.report.base_model,
                allow_unvalidated_arch=prepared.allow_unvalidated_arch,
            )
        )
        return max(
            topology.world_size()
            for topology in selected_suite_topologies(
                is_moe=handler.is_moe,
                cp_supported=bool(handler.cp_supported),
            )
        )
    resources = getattr(
        HANDLER_WORKFLOW_RESOURCES.get(prepared.report.model_key), stage_name, None
    )
    if resources is None:
        if stage_name == "length_trainability":
            return (
                3
                if model_uses_expert_parallel(
                    prepared.report.base_model,
                    allow_unvalidated_arch=prepared.allow_unvalidated_arch,
                )
                else 2
            )
        try:
            return _DEFAULT_STAGE_GPU_COUNTS[stage_name]
        except KeyError:
            raise RuntimeError(
                "missing workflow resources for "
                f"{prepared.report.model_key}/{stage_name}"
            ) from None
    return _minimum_gpu_count(stage_name, resources, visible_gpu_count=available)


def _runtime_key(
    prepared: PreparedWorkflow,
    stage_name: str,
    *,
    gpu_count: int,
) -> WorkflowRuntimeKey:
    environment = prepared.fixture.environment(stage_name)
    if stage_name in _CPU_STAGES:
        kind = "cpu"
        mode = stage_name
    elif stage_name in {"lora_coverage", "correctness_sensitivity"}:
        kind = "megatron"
        mode = stage_name
    elif stage_name == "e2e_throughput":
        kind = "joint"
        mode = "throughput"
    else:
        kind = "joint"
        mode = stage_name
    return WorkflowRuntimeKey(
        source_fingerprint=str(prepared.report.git["commit"]),
        handler=prepared.report.model_key,
        fixture=environment["ART_MODEL_SUPPORT_FIXTURE_PATH"],
        kind=kind,
        topology=f"gpus={gpu_count}",
        mode=mode,
        static_options=stage_name if mode == stage_name else "",
    )


def _stage_gpu_share(prepared: PreparedWorkflow, stage_name: str) -> float:
    if stage_name not in _LIGHTWEIGHT_GPU_STAGES:
        return 1.0
    fixture_path = prepared.fixture.environment(stage_name)[
        "ART_MODEL_SUPPORT_FIXTURE_PATH"
    ]
    return (
        1.0
        if fixture_path == prepared.fixture.canonical_path
        else _REDUCED_FIXTURE_GPU_SHARE
    )


def _functional_session(
    prepared: PreparedWorkflow,
    stage_gpu_counts: dict[str, int],
) -> VllmRuntimeLaunchConfig | None:
    stages = tuple(
        stage for stage in prepared.stages if stage in FUNCTIONAL_LORA_VLLM_STAGES
    )
    if stages != FUNCTIONAL_LORA_VLLM_STAGES:
        return None
    support = get_model_support_spec(
        prepared.report.base_model,
        allow_unvalidated_arch=prepared.allow_unvalidated_arch,
    )
    if (
        support.native_vllm_lora_status == "disabled"
        or support.default_rollout_weights_mode != "lora"
    ):
        return None
    handler = get_model_support_handler_for_spec(support)
    counts = {stage_gpu_counts[stage] for stage in stages}
    if len(counts) != 1:
        return None
    count = counts.pop()
    length_env = os.environ | prepared.fixture.environment(stages[0])
    capacities = {
        "max_model_len": int(
            length_env.get("ART_MODEL_SUPPORT_LENGTH_MAX_MODEL_LEN", 1024)
        ),
        "max_num_seqs": int(
            length_env.get("ART_MODEL_SUPPORT_LENGTH_MAX_NUM_SEQS", 32)
        ),
        "max_loras": 2,
        "max_lora_rank": int(
            length_env.get(
                MEGATRON_LORA_RANK_ENV, default_lora_rank_for_handler(handler)
            )
        ),
    }
    configured = HANDLER_WORKFLOW_RESOURCES.get(prepared.report.model_key)
    engine_specs = []
    for stage in stages:
        stage_resources = getattr(configured, stage, None)
        if stage_resources is None:
            gpu_ids = (count - 1,)
            engine_args: dict[str, object] = {"tensor_parallel_size": 1}
        else:
            resolved = resolve_stage_resources_for_visible_gpus(
                stage, stage_resources, visible_gpu_count=count
            )
            if resolved.vllm is None:
                return None
            gpu_ids = tuple(resolved.vllm.gpu_ids)
            if resolved.megatron is not None and set(resolved.megatron.gpu_ids) != set(
                range(count)
            ) - set(gpu_ids):
                return None
            engine_args = resolved.vllm.engine_args()
        for key in (*capacities, "max_num_batched_tokens"):
            value = engine_args.pop(key, None)
            if value is not None:
                if not isinstance(value, int):
                    return None
                capacities[key] = max(capacities.get(key, 0), value)
        engine_specs.append((gpu_ids, engine_args))
    fixtures = {
        prepared.fixture.environment(stage)[FIXTURE_PATH_ENV] for stage in stages
    }
    if engine_specs[0] != engine_specs[1] or len(fixtures) != 1:
        return None
    inference_gpu_ids, resource_args = engine_specs[0]
    return VllmRuntimeLaunchConfig(
        base_model=fixtures.pop(),
        port=0,
        cuda_visible_devices=",".join(map(str, inference_gpu_ids)),
        served_model_name="__art_functional_base__",
        rollout_weights_mode="lora",
        engine_args={
            "enforce_eager": True,
            "generation_config": "vllm",
            "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
            **handler.vllm_engine_args(rollout_weights_mode="lora"),
            **resource_args,
            **capacities,
        },
        server_args={
            "return_tokens_as_token_ids": True,
            "enable_auto_tool_choice": True,
            "tool_call_parser": "hermes",
            **handler.vllm_server_args(),
            "api_key": "art-functional-vllm",
        },
    )


def compile_prepared_workflows(
    workflows: list[PreparedWorkflow], *, visible_gpu_count: int
):
    operations = []
    for prepared in workflows:
        stage_gpu_counts = {
            stage_name: _stage_gpu_count(prepared, stage_name, visible_gpu_count)
            for stage_name in prepared.stages
        }
        functional_session = _functional_session(prepared, stage_gpu_counts)
        for stage_name in prepared.stages:
            shared = (
                functional_session
                if functional_session is not None
                and stage_name in FUNCTIONAL_LORA_VLLM_STAGES
                else None
            )
            gpu_count = stage_gpu_counts[stage_name]
            runtime = _runtime_key(prepared, stage_name, gpu_count=gpu_count)
            if shared is not None:
                runtime = runtime.model_copy(
                    update={
                        "mode": FUNCTIONAL_LORA_VLLM_MODE,
                        "static_options": shared.model_dump_json(),
                    }
                )
            operations.append(
                WorkflowOperation(
                    id=f"{prepared.report.model_key}:{stage_name}",
                    stage=stage_name,
                    runtime=runtime,
                    resources=WorkflowResourceRequest(
                        gpu_count=gpu_count,
                        gpu_share=_stage_gpu_share(prepared, stage_name),
                    ),
                    estimated_duration_s=_STAGE_DURATION_ESTIMATES_S[stage_name],
                )
            )
    return compile_workflow(operations)


def _visible_devices() -> list[WorkflowDevice]:
    import torch

    count = int(torch.cuda.device_count())
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = raw.split(",") if raw else [str(index) for index in range(count)]
    if len(gpu_ids) != count:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES exposes {len(gpu_ids)} ids but torch sees {count} GPUs"
        )
    hosts = [
        host.strip()
        for host in os.environ.get(_WORKFLOW_HOSTS_ENV, socket.gethostname()).split(",")
        if host.strip()
    ]
    if not hosts or len(set(hosts)) != len(hosts):
        raise RuntimeError(f"{_WORKFLOW_HOSTS_ENV} must contain unique host names")
    return [
        WorkflowDevice(host=host, gpu=gpu_id) for host in hosts for gpu_id in gpu_ids
    ]


def run_prepared_workflows(workflows: list[PreparedWorkflow]) -> list[ValidationReport]:
    from . import workflow

    devices = _visible_devices()
    if not devices:
        raise RuntimeError("the scheduled model-support workflow requires CUDA GPUs")
    gpu_counts = {
        host: sum(device.host == host for device in devices)
        for host in {device.host for device in devices}
    }
    if len(set(gpu_counts.values())) != 1:
        raise RuntimeError(f"workflow hosts expose different GPU counts: {gpu_counts}")
    by_model_key = {prepared.report.model_key: prepared for prepared in workflows}
    plan = compile_prepared_workflows(
        workflows, visible_gpu_count=next(iter(gpu_counts.values()))
    )

    def runner(session: WorkflowSession, placement: WorkflowPlacement) -> list[str]:
        session_dir = (
            by_model_key[session.runtime.handler].run_dir / ".sessions" / session.id
        )
        session_dir.mkdir(parents=True, exist_ok=False)
        prepared = by_model_key[session.runtime.handler]
        architecture_json = session_dir / "architecture.json"
        request_json = session_dir / "request.json"
        session_log = session_dir / "worker.log"
        architecture_json.write_text(
            prepared.architecture.model_dump_json(indent=2), encoding="utf-8"
        )
        items = []
        for operation in session.operations:
            prepared = by_model_key[operation.runtime.handler]
            stage_dir = prepared.run_dir / operation.stage
            stage_dir.mkdir(parents=True, exist_ok=False)
            environment = prepared.fixture.environment(operation.stage)
            environment[workflow.WORKFLOW_RUN_DIR_ENV] = str(prepared.run_dir)
            if prepared.include_sensitivity is not None:
                environment[workflow.SKIP_SENSITIVITY_ENV] = (
                    "0" if prepared.include_sensitivity else "1"
                )
            items.append(
                WorkflowStageWorkerItem(
                    stage=operation.stage,
                    stage_dir=str(stage_dir),
                    output_json=str(stage_dir / "stage_result.json"),
                    environment=environment,
                )
            )
        request = WorkflowStageWorkerSession(
            base_model=prepared.report.base_model,
            architecture_json=str(architecture_json),
            allow_unvalidated_arch=prepared.allow_unvalidated_arch,
            functional_vllm=(
                VllmRuntimeLaunchConfig.model_validate_json(
                    session.runtime.static_options
                )
                if session.runtime.mode == FUNCTIONAL_LORA_VLLM_MODE
                else None
            ),
            items=tuple(items),
        )
        request_json.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(
                    device.gpu for device in placement.devices
                ),
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            str(workflow.TESTS_DIR),
                            environment.get("PYTHONPATH"),
                        ),
                    )
                ),
                "WANDB_MODE": "disabled",
            }
        )
        command = [
            sys.executable,
            "-m",
            "integration.megatron.model_support.workflow_stage_worker",
            "--session-json",
            str(request_json),
        ]
        placement_hosts = {device.host for device in placement.devices}
        if len(placement_hosts) > 1:
            raise RuntimeError("one workflow session cannot span hosts")
        execution_host = next(iter(placement_hosts), socket.gethostname())
        if execution_host != socket.gethostname():
            runtime_profile = Path(sys.prefix) / "art-megatron-env.sh"
            if not runtime_profile.is_file():
                raise RuntimeError(
                    "remote workflow sessions require the Megatron runtime profile: "
                    f"{runtime_profile}"
                )
            remote_command = (
                "unset LD_LIBRARY_PATH && "
                f"source {shlex.quote(str(runtime_profile))} && "
                f"cd {shlex.quote(str(workflow.REPO_ROOT))} && exec "
                + shlex.join(
                    [
                        "env",
                        *(
                            f"{key}={environment[key]}"
                            for key in (
                                "CUDA_VISIBLE_DEVICES",
                                "PYTHONPATH",
                                "WANDB_MODE",
                            )
                        ),
                        *command,
                    ]
                )
            )
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                execution_host,
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                shlex.quote(remote_command),
            ]
        timeout_s = sum(
            workflow._WORKFLOW_STAGE_TIMEOUT_OVERRIDES_S.get(
                (operation.stage, prepared.report.base_model),
                workflow._WORKFLOW_STAGE_TIMEOUT_S,
            )
            for operation in session.operations
        )
        with session_log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command,
                cwd=workflow.REPO_ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = workflow._wait_stage_process(process, timeout_s=timeout_s)
            except subprocess.TimeoutExpired:
                returncode = None
        completed = []
        for operation, item in zip(session.operations, items, strict=True):
            output_json = Path(item.output_json)
            if output_json.exists():
                result = ValidationStageResult.model_validate_json(
                    output_json.read_text(encoding="utf-8")
                )
            else:
                detail = (
                    f"session exceeded {timeout_s:g}s"
                    if returncode is None
                    else "session worker did not write stage_result.json"
                    if returncode == 0
                    else workflow._subprocess_log_tail(session_log)
                    or f"session exited with code {returncode}"
                )
                result = ValidationStageResult(
                    name=operation.stage,
                    passed=False,
                    metrics={"error": detail},
                )
            result.metrics["workflow_session_id"] = session.id
            result.metrics["workflow_gpu_ids"] = [
                device.gpu for device in placement.devices
            ]
            prepared.record_fixture_metric(result.metrics)
            if operation.stage in workflow._RUNTIME_CLEANUP_STAGES:
                try:
                    result.metrics.update(
                        workflow._prune_runtime_artifacts(
                            prepared.run_dir / operation.stage
                        )
                    )
                except Exception as exc:
                    result.passed = False
                    result.metrics["runtime_artifact_cleanup_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            prepared.record(result)
            completed.append(operation.id)
        return completed

    execute_workflow(plan, devices=devices, runner=runner)
    for prepared in workflows:
        workflow._finalize_validation_report(prepared.report, partial=False)
        workflow._write_validation_report(prepared.report, prepared.output_json)
    return [prepared.report for prepared in workflows]


def build_scheduled_validation_reports(
    *,
    base_models: list[str],
    include_native_vllm_lora: bool = False,
    include_yes_no_trainability: bool = False,
    include_sensitivity: bool | None = None,
    output_json_by_model: dict[str, Path | None] | None = None,
    skip_stages: set[str] | None = None,
    allow_unvalidated_arch: bool = False,
) -> list[ValidationReport]:
    skip_stages = skip_stages or set()
    output_json_by_model = output_json_by_model or {}

    def prepare(base_model: str) -> PreparedWorkflow:
        return prepare_workflow(
            base_model=base_model,
            include_native_vllm_lora=include_native_vllm_lora,
            include_yes_no_trainability=include_yes_no_trainability,
            include_sensitivity=include_sensitivity,
            output_json=output_json_by_model.get(base_model),
            skip_stages=skip_stages,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )

    with ThreadPoolExecutor(max_workers=len(base_models)) as executor:
        workflows = list(executor.map(prepare, base_models))
    return run_prepared_workflows(workflows)
