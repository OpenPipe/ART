import argparse
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping
import uuid

from pydantic import BaseModel, Field

from art.megatron.model_support.registry import (
    VALIDATED_MODEL_SUPPORT_SPECS,
    get_model_support_handler_for_spec,
    get_model_support_spec,
)
from art.megatron.model_support.spec import ArchitectureReport

from ..artifacts import pinned_git_state
from .validation_spec import (
    MinimalLayerCoverageReport,
    ValidationReport,
    ValidationStageResult,
)
from .workflow_fixtures import WorkflowFixture, ensure_workflow_fixture

REPO_ROOT = Path(__file__).resolve().parents[4]
TESTS_DIR = REPO_ROOT / "tests"
ORACLE_LIVE_TRAINING_LOG_ENV = "ART_ORACLE_LIVE_TRAINING_LOG"
WORKFLOW_RUN_DIR_ENV = "ART_MODEL_SUPPORT_WORKFLOW_RUN_DIR"
WORKFLOW_STAGE_DIR_ENV = "ART_MODEL_SUPPORT_WORKFLOW_STAGE_DIR"
SKIP_SENSITIVITY_ENV = "ART_MODEL_SUPPORT_SKIP_SENSITIVITY"
INCLUDE_FLASH_SENSITIVITY_ENV = "ART_MODEL_SUPPORT_INCLUDE_FLASH_SENSITIVITY"
KEEP_TOPOLOGY_ARTIFACTS_ENV = "ART_ORACLE_KEEP_TOPOLOGY_ARTIFACTS"
CORRECTNESS_ARTIFACT_ROOT_ENV = "ART_MODEL_SUPPORT_CORRECTNESS_ARTIFACT_ROOT"
CORRECTNESS_PHASE_ENV = "ART_MODEL_SUPPORT_CORRECTNESS_PHASE"
CORRECTNESS_REFERENCE_STAGE = "correctness_reference"
WORKFLOW_ARTIFACT_SUITE_NAME = "Megatron model-support validation workflow"
FLASH_SENSITIVITY_MUTATION = "attn_skip_flash_lse_normalize"
_HANDLER_INAPPLICABLE_SENSITIVITY_MUTATIONS = {
    "glm52": frozenset(
        {"attn_skip_nested_grad_sanitize", "attn_skip_flash_lse_normalize"}
    )
}

MANDATORY_VALIDATION_STAGES = (
    "dependency_resolution",
    "architecture_discovery",
    "hf_parity",
    "lora_coverage",
    "train_inf_mismatch",
    "correctness_sensitivity",
    "chat_template_rollout",
    "packing_invariance",
    "length_trainability",
    "e2e_throughput",
)
YES_NO_TRAINABILITY_STAGE = "yes_no_trainability"
OPTIONAL_VALIDATION_STAGES = (YES_NO_TRAINABILITY_STAGE,)
ALL_VALIDATION_STAGES = (*MANDATORY_VALIDATION_STAGES, *OPTIONAL_VALIDATION_STAGES)
ARCHITECTURE_REPRESENTATIVE_MODELS = {
    "llama3_dense": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
    "qwen3_dense": "Qwen/Qwen3-32B",
    "qwen3_5_moe": "Qwen/Qwen3.5-35B-A3B",
    "qwen3_5_dense": "Qwen/Qwen3.5-27B",
    "gemma4_moe": "google/gemma-4-26B-A4B-it",
    "gemma4_dense": "google/gemma-4-31B-it",
    "dsv4": "deepseek-ai/DeepSeek-V4-Flash",
    "glm52": "zai-org/GLM-5.2",
    "gpt_oss_moe": "openai/gpt-oss-20b",
}
SUBPROCESS_VALIDATION_STAGES = frozenset(
    {
        "hf_parity",
        "lora_coverage",
        "train_inf_mismatch",
        "correctness_sensitivity",
        "chat_template_rollout",
        "packing_invariance",
        "length_trainability",
        "e2e_throughput",
        YES_NO_TRAINABILITY_STAGE,
    }
)
_RUNTIME_CLEANUP_STAGES = frozenset(
    {"length_trainability", "e2e_throughput", YES_NO_TRAINABILITY_STAGE}
)
_RUNTIME_ARTIFACT_DIR_NAMES = frozenset(
    {
        "checkpoints",
        "megatron_runtime",
        "optimizer_states",
        "trajectories",
    }
)
_WORKFLOW_STAGE_TIMEOUT_S = 30 * 60
_WORKFLOW_STAGE_TIMEOUT_OVERRIDES_S = {
    ("e2e_throughput", "deepseek-ai/DeepSeek-V4-Flash"): 40 * 60,
}


class AllArchitecturesValidationReport(BaseModel):
    passed: bool = False
    complete: bool = False
    reports: list[ValidationReport] = Field(default_factory=list)


def build_validation_stage_names(
    *,
    include_yes_no_trainability: bool = False,
) -> list[str]:
    stages = list(MANDATORY_VALIDATION_STAGES)
    if include_yes_no_trainability:
        stages.append(YES_NO_TRAINABILITY_STAGE)
    return stages


def detect_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in ("transformers", "vllm", "megatron-bridge"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def initialize_validation_report(
    *,
    base_model: str,
    include_yes_no_trainability: bool = False,
    allow_unvalidated_arch: bool = False,
) -> ValidationReport:
    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return ValidationReport(
        git=pinned_git_state(WORKFLOW_ARTIFACT_SUITE_NAME).model_dump(mode="json"),
        base_model=base_model,
        model_key=spec.key,
        dependency_versions=detect_dependency_versions(),
        stages=[
            ValidationStageResult(name=stage_name)
            for stage_name in build_validation_stage_names(
                include_yes_no_trainability=include_yes_no_trainability,
            )
        ],
    )


def _stage_error_metrics(exc: Exception) -> dict[str, Any]:
    return {"error": f"{type(exc).__name__}: {exc}"}


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _import_integration_module(module_name: str) -> Any:
    tests_dir = str(TESTS_DIR)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    return importlib.import_module(module_name)


def _subprocess_log_tail(log_path: Path, *, max_lines: int = 40) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _inspect_architecture_for_workflow(
    base_model: str,
    *,
    allow_unvalidated_arch: bool,
) -> ArchitectureReport:
    from art.megatron.model_support.discovery import inspect_architecture

    # Discovery only inspects layer families, so use a minimal topology instead
    # of inheriting visible GPU count and tripping model-specific TP limits.
    with _temporary_env(
        ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE="1",
        ART_MEGATRON_CONTEXT_PARALLEL_SIZE="1",
        ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE="1",
        ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE="1",
    ):
        return (
            inspect_architecture(base_model, allow_unvalidated_arch=True)
            if allow_unvalidated_arch
            else inspect_architecture(base_model)
        )


@contextmanager
def _redirect_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            yield


@contextmanager
def _temporary_env(**updates: str):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
                continue
            os.environ[key] = value


def _new_workflow_run_dir(*, output_json: str | Path | None, model_key: str) -> Path:
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    if output_json is None:
        root = REPO_ROOT / ".local" / "model_support_workflow_runs" / model_key
    else:
        output_path = Path(output_json).resolve()
        root = output_path.parent / f"{output_path.stem}.artifacts"
    path = root / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _workflow_stage_dir() -> Path:
    raw = os.environ.get(WORKFLOW_STAGE_DIR_ENV)
    if raw is None:
        raise RuntimeError(f"missing {WORKFLOW_STAGE_DIR_ENV}")
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stage_artifact_dir() -> Path:
    path = _workflow_stage_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_stage_workspace(path: Path) -> None:
    if os.environ.get(KEEP_TOPOLOGY_ARTIFACTS_ENV) != "1" and path.exists():
        shutil.rmtree(path)


def _oracle_case_config(
    oracle_harness: Any,
    *,
    base_model: str,
    model_support_key: str,
    is_moe: bool,
    precision: str,
    num_layers: int,
    target_modules: list[str],
    allow_unvalidated_arch: bool,
) -> Any:
    artifact_root = os.environ.get(CORRECTNESS_ARTIFACT_ROOT_ENV)
    oracle_harness.ARTIFACT_ROOT = (
        Path(artifact_root) if artifact_root is not None else _stage_artifact_dir()
    )
    num_layers = int(
        os.environ.get("ART_MODEL_SUPPORT_FUNCTIONAL_NUM_LAYERS", num_layers)
    )
    return oracle_harness.OracleCaseConfig(
        base_model=base_model,
        model_support_key=model_support_key,
        is_moe=is_moe,
        precision=precision,
        num_layers=num_layers,
        num_steps=1,
        lora={"target_modules": target_modules},
        allow_unvalidated_arch=allow_unvalidated_arch,
    )


def _write_validation_report(
    report: ValidationReport,
    output_json: str | Path | None,
) -> None:
    if output_json is None:
        return
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _record_stage_duration(stage: ValidationStageResult, *, started: float) -> None:
    stage.metrics["workflow_stage_duration_s"] = time.monotonic() - started


def _prune_runtime_artifacts(stage_dir: Path) -> dict[str, int]:
    paths = sorted(
        (
            path
            for path in stage_dir.rglob("*")
            if path.is_dir() and path.name in _RUNTIME_ARTIFACT_DIR_NAMES
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    removed_bytes = 0
    for path in paths:
        removed_bytes += sum(
            child.stat().st_size for child in path.rglob("*") if child.is_file()
        )
        shutil.rmtree(path)
    return {
        "workflow_pruned_runtime_artifact_dirs": len(paths),
        "workflow_pruned_runtime_artifact_bytes": removed_bytes,
    }


def _write_all_architectures_report(
    report: AllArchitecturesValidationReport,
    output_json: str | Path | None,
) -> None:
    if output_json is None:
        return
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _per_architecture_output_json(output_json: str | Path, model_key: str) -> Path:
    path = Path(output_json)
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}.{model_key}{suffix}")


def validated_architecture_representative_models() -> list[str]:
    missing_keys = {
        spec.key
        for spec in VALIDATED_MODEL_SUPPORT_SPECS
        if spec.key not in ARCHITECTURE_REPRESENTATIVE_MODELS
    }
    unknown_keys = set(ARCHITECTURE_REPRESENTATIVE_MODELS) - {
        spec.key for spec in VALIDATED_MODEL_SUPPORT_SPECS
    }
    if missing_keys or unknown_keys:
        raise RuntimeError(
            "Architecture representative mapping does not match validated specs: "
            f"missing={sorted(missing_keys)}, unknown={sorted(unknown_keys)}"
        )
    representatives: list[str] = []
    for spec in VALIDATED_MODEL_SUPPORT_SPECS:
        base_model = ARCHITECTURE_REPRESENTATIVE_MODELS[spec.key]
        if base_model not in spec.model_names:
            raise RuntimeError(
                f"{base_model!r} is not registered under model support spec {spec.key!r}"
            )
        representatives.append(base_model)
    return representatives


def _mark_remaining_stages_skipped(
    report: ValidationReport,
    *,
    after_stage_name: str,
    reason: str | None = None,
) -> None:
    past_failure = False
    for stage in report.stages:
        if past_failure:
            stage.passed = False
            stage.skipped = True
            stage.metrics = {
                "skipped": True,
                "reason": reason or f"stopped after {after_stage_name} failed",
                "workflow_stage_duration_s": 0.0,
            }
            continue
        past_failure = stage.name == after_stage_name


def _finalize_validation_report(
    report: ValidationReport,
    *,
    partial: bool,
) -> None:
    executed = [stage for stage in report.stages if not stage.skipped]
    report.passed = bool(executed) and all(stage.passed for stage in executed)
    report.complete = (
        not partial and len(executed) == len(report.stages) and report.passed
    )


def _only_stage_run_set(only_stage: str | None) -> set[str] | None:
    if only_stage is None:
        return None
    if only_stage not in ALL_VALIDATION_STAGES:
        raise ValueError(f"unknown workflow stage: {only_stage}")
    if only_stage == "dependency_resolution":
        return {only_stage}
    if only_stage == "architecture_discovery":
        return {"dependency_resolution", only_stage}
    return {"dependency_resolution", "architecture_discovery", only_stage}


def _run_stage_in_subprocess(
    *,
    stage_name: str,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
    run_dir: Path | None = None,
    stage_environment: Mapping[str, str] | None = None,
    visible_gpu_ids: tuple[str, ...] | None = None,
) -> ValidationStageResult:
    run_dir = run_dir or Path(os.environ[WORKFLOW_RUN_DIR_ENV])
    stage_dir = run_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=False)
    architecture_json = stage_dir / "architecture.json"
    output_json = stage_dir / "stage_result.json"
    log_path = stage_dir / "worker.log"
    architecture_json.write_text(
        architecture.model_dump_json(indent=2),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "-m",
        "integration.megatron.model_support.workflow_stage_worker",
        "--stage",
        stage_name,
        "--base-model",
        base_model,
        "--architecture-json",
        str(architecture_json),
        "--output-json",
        str(output_json),
    ]
    if allow_unvalidated_arch:
        cmd.append("--allow-unsupported-arch")
    env = os.environ.copy()
    if stage_environment is not None:
        env.update(stage_environment)
    if visible_gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_gpu_ids)
    env["WANDB_MODE"] = "disabled"
    env[WORKFLOW_STAGE_DIR_ENV] = str(stage_dir)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(TESTS_DIR)
        if not existing_pythonpath
        else f"{TESTS_DIR}{os.pathsep}{existing_pythonpath}"
    )
    started = time.monotonic()
    timeout_s = _WORKFLOW_STAGE_TIMEOUT_OVERRIDES_S.get(
        (stage_name, base_model), _WORKFLOW_STAGE_TIMEOUT_S
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = _wait_stage_process(process, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            returncode = None
    duration_s = time.monotonic() - started
    common_metrics = {
        "workflow_stage_artifact_dir": str(stage_dir),
        "workflow_stage_duration_s": duration_s,
    }
    if returncode is None:
        return ValidationStageResult(
            name=stage_name,
            passed=False,
            metrics={
                **common_metrics,
                "error": f"stage exceeded {timeout_s:g}s; log={log_path}",
            },
        )
    if returncode != 0:
        tail = _subprocess_log_tail(log_path)
        return ValidationStageResult(
            name=stage_name,
            passed=False,
            metrics={
                **common_metrics,
                "error": tail or f"subprocess exited with code {returncode}",
            },
        )
    if not output_json.exists():
        return ValidationStageResult(
            name=stage_name,
            passed=False,
            metrics={
                **common_metrics,
                "error": "stage worker did not write output_json",
            },
        )
    result = ValidationStageResult.model_validate_json(output_json.read_text())
    result.metrics.update(common_metrics)
    output_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def _raise_signal_exit(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def _wait_stage_process(process: subprocess.Popen[Any], *, timeout_s: float) -> int:
    owns_signals = threading.current_thread() is threading.main_thread()
    previous_sigterm = (
        signal.signal(signal.SIGTERM, _raise_signal_exit) if owns_signals else None
    )
    try:
        return process.wait(timeout=timeout_s)
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def run_hf_parity_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    hf_parity = _import_integration_module(
        "integration.megatron.model_support.hf_parity"
    )
    oracle_harness = _import_integration_module(
        "integration.megatron.model_support.oracle_harness"
    )
    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    handler = get_model_support_handler_for_spec(spec)
    case_config = _oracle_case_config(
        oracle_harness,
        base_model=base_model,
        model_support_key=spec.key,
        is_moe=handler.is_moe,
        precision=handler.correctness_precision(),
        num_layers=max(1, architecture.recommended_min_layers),
        target_modules=list(spec.default_target_modules),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    case_config = hf_parity.hf_parity_case_config(case_config)
    report = hf_parity.run_hf_parity(case_config=case_config, in_process=True)
    artifact_dir = str(
        Path(oracle_harness.ARTIFACT_ROOT)
        / report.case_id
        / hf_parity.HF_PARITY_OUTPUT_DIRNAME
    )
    return ValidationStageResult(
        name="hf_parity",
        passed=report.signal == "pass",
        metrics={
            "requested_num_layers": report.requested_num_layers,
            "coverage": report.coverage.model_dump(mode="json"),
            "signal": report.signal,
            "pass_count": report.pass_count,
            "fail_count": report.fail_count,
            "phases": [row.model_dump(mode="json") for row in report.metrics],
        },
        artifact_dir=artifact_dir,
    )


def run_lora_coverage_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    lora_coverage = _import_integration_module(
        "integration.megatron.model_support.lora_coverage"
    )
    oracle_harness = _import_integration_module(
        "integration.megatron.model_support.oracle_harness"
    )
    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    handler = get_model_support_handler_for_spec(spec)
    case_config = _oracle_case_config(
        oracle_harness,
        base_model=base_model,
        model_support_key=spec.key,
        is_moe=handler.is_moe,
        precision=handler.correctness_precision(),
        num_layers=max(1, architecture.recommended_min_layers),
        target_modules=list(spec.default_target_modules),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    report = lora_coverage.run_lora_coverage(case_config)
    return ValidationStageResult(
        name="lora_coverage",
        passed=not report.missing_wrapped_target_modules
        and not report.missing_exported_target_modules,
        metrics=report.model_dump(mode="json"),
    )


def run_train_inf_mismatch_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    del architecture
    train_inf_mismatch = _import_integration_module(
        "integration.megatron.train_inf_mismatch.workflow_stage"
    )
    report = train_inf_mismatch.run_train_inf_mismatch(
        base_model=base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return ValidationStageResult(
        name="train_inf_mismatch",
        passed=report.passed,
        metrics=report.model_dump(mode="json"),
        artifact_dir=report.artifact_dir,
    )


def run_correctness_sensitivity_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    stage_dir = _workflow_stage_dir()
    phase = os.environ.get(CORRECTNESS_PHASE_ENV, "all")
    if phase not in {"all", "reference", "variants"}:
        raise ValueError(f"unsupported correctness phase: {phase}")
    artifact_root = os.environ.get(CORRECTNESS_ARTIFACT_ROOT_ENV)
    correctness_log = (
        Path(artifact_root).parent / "reference.log"
        if phase == "reference" and artifact_root is not None
        else stage_dir / "correctness.log"
    )
    sensitivity_log = stage_dir / "sensitivity.log"
    live_training_log = stage_dir / "live_training.log"
    oracle_harness = _import_integration_module(
        "integration.megatron.model_support.oracle_harness"
    )
    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    handler = get_model_support_handler_for_spec(spec)
    cp_supported = bool(handler.cp_supported)
    correctness_precision = handler.correctness_precision()
    correctness_use_fp32_lora_reference = handler.correctness_use_fp32_lora_reference()
    correctness_phase_pass_fns = handler.correctness_phase_pass_fns(oracle_harness)
    suite_topologies = list(
        oracle_harness.selected_suite_topologies(
            is_moe=handler.is_moe,
            cp_supported=cp_supported,
        )
    )
    objectives = list(oracle_harness.SUPPORTED_ORACLE_OBJECTIVES)
    skip_sensitivity = _truthy_env(SKIP_SENSITIVITY_ENV)
    available_gpu_count = oracle_harness.available_gpu_count()
    max_world_size = available_gpu_count
    oracle_world_size = oracle_harness.oracle_topology(
        is_moe=handler.is_moe
    ).world_size()
    required_gpu_count = (
        oracle_world_size
        if phase == "reference"
        else max(
            oracle_world_size,
            *(topology.world_size() for topology in suite_topologies),
        )
    )
    if available_gpu_count < required_gpu_count:
        raise RuntimeError(
            "Need "
            f"{required_gpu_count} GPUs for the complete correctness topology set, "
            f"found {available_gpu_count}"
        )
    selected_suite_topologies = suite_topologies
    excluded_suite_topologies: list[Any] = []
    pipeline_layer_multiple = math.lcm(
        *(topology.pp * topology.vpp for topology in selected_suite_topologies)
    )
    minimum_layers = max(1, architecture.recommended_min_layers)
    num_layers = (
        (minimum_layers + pipeline_layer_multiple - 1) // pipeline_layer_multiple
    ) * pipeline_layer_multiple
    case_config = _oracle_case_config(
        oracle_harness,
        base_model=base_model,
        model_support_key=spec.key,
        is_moe=handler.is_moe,
        precision=correctness_precision,
        num_layers=num_layers,
        target_modules=list(spec.default_target_modules),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    case_artifacts = oracle_harness.ensure_case_artifacts(case_config)
    if phase == "reference":
        live_training_log.write_text("", encoding="utf-8")
        with _temporary_env(**{oracle_harness.ORACLE_OBJECTIVE_ENV: "all"}):
            with _temporary_env(
                **{ORACLE_LIVE_TRAINING_LOG_ENV: str(live_training_log)}
            ):
                with _redirect_output(correctness_log):
                    oracle_harness.prepare_suite_references(
                        case_config=case_config,
                        use_fp32_lora_reference=correctness_use_fp32_lora_reference,
                    )
        return ValidationStageResult(
            name=CORRECTNESS_REFERENCE_STAGE,
            passed=True,
            metrics={
                "correctness_reference_log_path": str(correctness_log),
                "live_training_log_path": str(live_training_log),
                "requested_num_layers": case_config.num_layers,
                "precision": correctness_precision,
                "use_fp32_lora_reference": correctness_use_fp32_lora_reference,
                "is_moe": handler.is_moe,
                "available_gpu_count": available_gpu_count,
                "required_gpu_count": required_gpu_count,
            },
            artifact_dir=case_artifacts.case_dir,
        )
    mutations: list[str] = []
    inapplicable_sensitivity_mutations: list[str] = []
    default_excluded_sensitivity_mutations: list[str] = []
    excluded_sensitivity_mutations: list[str] = []
    if not skip_sensitivity:
        for objective in objectives:
            for (
                mutation
            ) in oracle_harness.supported_sensitivity_mutations_for_objective(
                objective,
                is_moe=handler.is_moe,
            ):
                if mutation not in mutations:
                    mutations.append(mutation)
        inapplicable = _HANDLER_INAPPLICABLE_SENSITIVITY_MUTATIONS.get(handler.key, ())
        inapplicable_sensitivity_mutations = [
            mutation for mutation in mutations if mutation in inapplicable
        ]
        mutations = [mutation for mutation in mutations if mutation not in inapplicable]
        excluded_sensitivity_mutations = [
            mutation
            for mutation in mutations
            if oracle_harness.sensitivity_topology_for_mutation(
                mutation,
                is_moe=handler.is_moe,
            ).world_size()
            > max_world_size
            or (
                not cp_supported
                and oracle_harness.sensitivity_topology_for_mutation(
                    mutation,
                    is_moe=handler.is_moe,
                ).cp
                > 1
            )
        ]
        if FLASH_SENSITIVITY_MUTATION not in inapplicable and not _truthy_env(
            INCLUDE_FLASH_SENSITIVITY_ENV
        ):
            default_excluded_sensitivity_mutations.append(FLASH_SENSITIVITY_MUTATION)
        mutations = [
            mutation
            for mutation in mutations
            if mutation
            not in {
                *excluded_sensitivity_mutations,
                *default_excluded_sensitivity_mutations,
            }
        ]
    live_training_log.write_text("", encoding="utf-8")
    with _temporary_env(**{oracle_harness.ORACLE_OBJECTIVE_ENV: "all"}):
        with _temporary_env(**{ORACLE_LIVE_TRAINING_LOG_ENV: str(live_training_log)}):
            with _redirect_output(correctness_log):
                suite_reports = oracle_harness.run_suite(
                    case_config=case_config,
                    max_world_size=max_world_size,
                    cp_supported=cp_supported,
                    phase_pass_fns=correctness_phase_pass_fns,
                    use_fp32_lora_reference=correctness_use_fp32_lora_reference,
                    require_existing_references=phase == "variants",
                    prune_reference_artifacts=skip_sensitivity or not mutations,
                    prune_case_artifacts=skip_sensitivity or not mutations,
                )
            sensitivity_reports = []
            if skip_sensitivity:
                sensitivity_log.write_text(
                    (
                        "Sensitivity suite skipped. "
                        f"Set {SKIP_SENSITIVITY_ENV}=0 to re-enable workflow sensitivity.\n"
                    ),
                    encoding="utf-8",
                )
            elif not mutations:
                sensitivity_log.write_text(
                    (
                        "Sensitivity suite skipped. "
                        f"No sensitivity mutations fit max_world_size={max_world_size}.\n"
                    ),
                    encoding="utf-8",
                )
            else:
                with _redirect_output(sensitivity_log):
                    sensitivity_reports = oracle_harness.run_sensitivity_suite(
                        case_config=case_config,
                        mutations=mutations,
                        max_world_size=max_world_size,
                    )
    return ValidationStageResult(
        name="correctness_sensitivity",
        passed=True,
        metrics={
            "correctness_log_path": str(correctness_log),
            "correctness_reference_log_path": (
                str(Path(artifact_root).parent / "reference.log")
                if phase == "variants" and artifact_root is not None
                else None
            ),
            "sensitivity_log_path": str(sensitivity_log),
            "live_training_log_path": str(live_training_log),
            "requested_num_layers": case_config.num_layers,
            "pipeline_layer_multiple": pipeline_layer_multiple,
            "precision": correctness_precision,
            "use_fp32_lora_reference": correctness_use_fp32_lora_reference,
            "is_moe": handler.is_moe,
            "cp_supported": cp_supported,
            "allow_unvalidated_arch": allow_unvalidated_arch,
            "objectives": objectives,
            "sensitivity_mutations": mutations,
            "inapplicable_sensitivity_mutations": (inapplicable_sensitivity_mutations),
            "excluded_sensitivity_mutations": excluded_sensitivity_mutations,
            "default_excluded_sensitivity_mutations": (
                default_excluded_sensitivity_mutations
            ),
            "available_gpu_count": available_gpu_count,
            "max_world_size": max_world_size,
            "required_gpu_count": required_gpu_count,
            "topology_artifacts_retained": oracle_harness.keep_topology_artifacts(),
            "correctness_variant_count": len(suite_reports),
            "correctness_excluded_topology_count": len(excluded_suite_topologies),
            "correctness_excluded_topologies": [
                topology.slug() for topology in excluded_suite_topologies
            ],
            "correctness_selected_topologies": [
                topology.slug() for topology in selected_suite_topologies
            ],
            "correctness_variants": [
                {
                    "variant": report.variant,
                    "topology": report.topology,
                    "signal": report.signal,
                    "fail_count": report.fail_count,
                }
                for report in suite_reports
            ],
            "sensitivity_skipped": skip_sensitivity,
            "sensitivity_skip_reason": (
                f"{SKIP_SENSITIVITY_ENV}=1" if skip_sensitivity else None
            ),
            "sensitivity_variant_count": len(sensitivity_reports),
            "sensitivity_variants": [
                {
                    "variant": report.variant,
                    "topology": report.topology,
                    "signal": report.signal,
                    "expected_signal": report.expected_signal,
                    "fail_count": report.fail_count,
                }
                for report in sensitivity_reports
            ],
        },
        artifact_dir=case_artifacts.case_dir,
    )


def run_chat_template_rollout_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    del architecture
    del allow_unvalidated_arch
    chat_template_rollout = _import_integration_module(
        "integration.megatron.model_support.chat_template_rollout"
    )
    chat_template_rollout._artifact_dir = lambda _base_model: _stage_artifact_dir()
    report = chat_template_rollout.run_chat_template_rollout(base_model=base_model)
    return ValidationStageResult(
        name="chat_template_rollout",
        passed=report.passed,
        metrics=report.model_dump(mode="json"),
        artifact_dir=report.output_dir,
    )


def run_yes_no_trainability_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    del architecture
    yes_no_trainability = _import_integration_module(
        "integration.megatron.trainability.yes_no_trainability"
    )
    report = yes_no_trainability.run_yes_no_trainability(
        base_model=base_model,
        artifact_root=_stage_artifact_dir(),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    passed = yes_no_trainability.yes_no_trainability_passed(report)
    return ValidationStageResult(
        name=YES_NO_TRAINABILITY_STAGE,
        passed=passed,
        metrics=report.model_dump(mode="json"),
        artifact_dir=report.output_dir,
    )


def run_length_trainability_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    del architecture
    length_trainability = _import_integration_module(
        "integration.megatron.trainability.test_live_length_trainability"
    )
    length_trainability.LATEST_SUMMARY_LOG_PATH = (
        _workflow_stage_dir() / "length_trainability.log"
    )
    artifact_dir = _stage_artifact_dir()
    length_trainability._artifact_dir = lambda _base_model: artifact_dir
    try:
        report = length_trainability.run_length_trainability(
            base_model=base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
    finally:
        _cleanup_stage_workspace(artifact_dir / "megatron_dedicated_workspace")
    return ValidationStageResult(
        name="length_trainability",
        passed=length_trainability.length_trainability_passed(report),
        metrics=report.model_dump(mode="json"),
        artifact_dir=str(Path(report.summary_log_path).parent),
    )


def run_packing_invariance_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    packing_invariance = _import_integration_module(
        "integration.megatron.model_support.packing_invariance"
    )
    packing_invariance._artifact_dir = lambda _base_model: _stage_artifact_dir()
    report = packing_invariance.run_packing_invariance(
        base_model=base_model,
        num_layers=max(1, architecture.recommended_min_layers),
        allow_unvalidated_arch=allow_unvalidated_arch,
        in_process=True,
    )
    metrics = report.model_dump(mode="json")
    passed = bool(metrics["scenarios"]) and all(
        scenario["matched"] and scenario["checked_token_count"] > 0
        for scenario in metrics["scenarios"]
    )
    return ValidationStageResult(
        name="packing_invariance",
        passed=passed,
        metrics=metrics,
        artifact_dir=report.output_dir,
    )


def run_e2e_throughput_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    from .workflow_throughput import run_e2e_throughput

    return run_e2e_throughput(
        base_model=base_model,
        architecture=architecture,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )


def build_validation_report(
    *,
    base_model: str,
    include_yes_no_trainability: bool = False,
    include_sensitivity: bool | None = None,
    output_json: str | Path | None = None,
    skip_stages: set[str] | None = None,
    only_stage: str | None = None,
    stop_on_failure: bool = False,
    allow_unvalidated_arch: bool = False,
) -> ValidationReport:
    if only_stage is not None and skip_stages:
        raise ValueError("only_stage cannot be combined with skip_stages")
    only_stage_run_set = _only_stage_run_set(only_stage)
    report = initialize_validation_report(
        base_model=base_model,
        include_yes_no_trainability=(
            include_yes_no_trainability or only_stage == YES_NO_TRAINABILITY_STAGE
        ),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    skip_stages = skip_stages or set()
    selected_subprocess_stages = {
        stage.name
        for stage in report.stages
        if stage.name in SUBPROCESS_VALIDATION_STAGES
        and stage.name not in skip_stages
        and (only_stage_run_set is None or stage.name in only_stage_run_set)
    }
    run_dir = _new_workflow_run_dir(
        output_json=output_json,
        model_key=report.model_key,
    )
    stage_runners = {
        "hf_parity": run_hf_parity_stage,
        "lora_coverage": run_lora_coverage_stage,
        "train_inf_mismatch": run_train_inf_mismatch_stage,
        "correctness_sensitivity": run_correctness_sensitivity_stage,
        "chat_template_rollout": run_chat_template_rollout_stage,
        "packing_invariance": run_packing_invariance_stage,
        "length_trainability": run_length_trainability_stage,
        "e2e_throughput": run_e2e_throughput_stage,
        YES_NO_TRAINABILITY_STAGE: run_yes_no_trainability_stage,
    }
    env = {WORKFLOW_RUN_DIR_ENV: str(run_dir)}
    if include_sensitivity is not None:
        env[SKIP_SENSITIVITY_ENV] = "0" if include_sensitivity else "1"
    architecture: ArchitectureReport | None = None
    fixture: WorkflowFixture | None = None
    fixture_error: Exception | None = None
    fixture_attempted = False
    with _temporary_env(**env):
        for stage in report.stages:
            stage_started = time.monotonic()
            if only_stage_run_set is not None and stage.name not in only_stage_run_set:
                stage.passed = False
                stage.skipped = True
                stage.metrics = {
                    "skipped": True,
                    "reason": f"--only-stage={only_stage}",
                }
                _record_stage_duration(stage, started=stage_started)
                _write_validation_report(report, output_json)
                continue
            if stage.name in skip_stages:
                stage.passed = False
                stage.skipped = True
                stage.metrics = {"skipped": True, "reason": "--skip-stage"}
                _record_stage_duration(stage, started=stage_started)
                _write_validation_report(report, output_json)
                if stage.name == "architecture_discovery":
                    _mark_remaining_stages_skipped(
                        report,
                        after_stage_name=stage.name,
                        reason="architecture_discovery was skipped",
                    )
                    break
                continue
            if stage.name == "dependency_resolution":
                stage.passed = True
                stage.metrics = dict(report.dependency_versions)
                _record_stage_duration(stage, started=stage_started)
                _write_validation_report(report, output_json)
                continue
            if stage.name == "architecture_discovery":
                try:
                    architecture = _inspect_architecture_for_workflow(
                        base_model,
                        allow_unvalidated_arch=allow_unvalidated_arch,
                    )
                    stage.passed = not architecture.unresolved_risks
                    stage.metrics = {
                        "recommended_min_layers": architecture.recommended_min_layers,
                        "layer_families": [
                            family.model_dump()
                            for family in architecture.layer_families
                        ],
                        "unresolved_risks": list(architecture.unresolved_risks),
                    }
                except Exception as exc:
                    stage.passed = False
                    stage.metrics = _stage_error_metrics(exc)
                _record_stage_duration(stage, started=stage_started)
                _write_validation_report(report, output_json)
                if architecture is None:
                    _mark_remaining_stages_skipped(
                        report,
                        after_stage_name=stage.name,
                        reason="architecture_discovery failed",
                    )
                    break
                if stop_on_failure and not stage.passed:
                    _mark_remaining_stages_skipped(report, after_stage_name=stage.name)
                    _finalize_validation_report(
                        report,
                        partial=only_stage is not None or bool(skip_stages),
                    )
                    _write_validation_report(report, output_json)
                    break
                continue
            if architecture is None:
                raise RuntimeError(
                    "architecture_discovery must run before subprocess stages"
                )
            stage_runner = stage_runners[stage.name]
            if stage.name in SUBPROCESS_VALIDATION_STAGES:
                fixture_provisioning_s: float | None = None
                if not fixture_attempted:
                    fixture_started = time.monotonic()
                    fixture_attempted = True
                    try:
                        fixture = ensure_workflow_fixture(
                            base_model,
                            allow_unvalidated_arch=allow_unvalidated_arch,
                            required_stages=selected_subprocess_stages,
                        )
                    except Exception as exc:
                        fixture_error = exc
                    fixture_provisioning_s = time.monotonic() - fixture_started
                if fixture_error is not None:
                    stage_result = ValidationStageResult(
                        name=stage.name,
                        passed=False,
                        metrics=_stage_error_metrics(fixture_error),
                    )
                else:
                    assert fixture is not None
                    with _temporary_env(**fixture.environment(stage.name)):
                        stage_result = _run_stage_in_subprocess(
                            stage_name=stage.name,
                            base_model=base_model,
                            architecture=architecture,
                            allow_unvalidated_arch=allow_unvalidated_arch,
                        )
                if fixture_provisioning_s is not None:
                    stage_result.metrics["fixture_provisioning_s"] = (
                        fixture_provisioning_s
                    )
            else:
                try:
                    stage_result = stage_runner(
                        base_model=base_model,
                        architecture=architecture,
                        allow_unvalidated_arch=allow_unvalidated_arch,
                    )
                except Exception as exc:
                    stage_result = ValidationStageResult(
                        name=stage.name,
                        passed=False,
                        metrics=_stage_error_metrics(exc),
                    )
            stage.passed = stage_result.passed
            stage.metrics = dict(stage_result.metrics)
            stage.artifact_dir = stage_result.artifact_dir
            if stage.name in _RUNTIME_CLEANUP_STAGES:
                try:
                    stage.metrics.update(_prune_runtime_artifacts(run_dir / stage.name))
                except Exception as exc:
                    stage.passed = False
                    stage.metrics["runtime_artifact_cleanup_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            _record_stage_duration(stage, started=stage_started)
            _write_validation_report(report, output_json)
            if stop_on_failure and not stage.passed:
                _mark_remaining_stages_skipped(report, after_stage_name=stage.name)
                _finalize_validation_report(
                    report,
                    partial=only_stage is not None or bool(skip_stages),
                )
                _write_validation_report(report, output_json)
                break
    _finalize_validation_report(
        report,
        partial=only_stage is not None or bool(skip_stages),
    )
    _write_validation_report(report, output_json)
    return report


def build_all_architectures_validation_report(
    *,
    include_yes_no_trainability: bool = False,
    include_sensitivity: bool | None = None,
    output_json: str | Path | None = None,
    skip_stages: set[str] | None = None,
    only_stage: str | None = None,
    stop_on_failure: bool = False,
    allow_unvalidated_arch: bool = False,
) -> AllArchitecturesValidationReport:
    aggregate = AllArchitecturesValidationReport()
    representatives = validated_architecture_representative_models()
    _write_all_architectures_report(aggregate, output_json)
    for base_model in representatives:
        model_key = get_model_support_spec(
            base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        ).key
        report = build_validation_report(
            base_model=base_model,
            include_yes_no_trainability=include_yes_no_trainability,
            include_sensitivity=include_sensitivity,
            output_json=(
                _per_architecture_output_json(output_json, model_key)
                if output_json is not None
                else None
            ),
            skip_stages=skip_stages,
            only_stage=only_stage,
            stop_on_failure=stop_on_failure,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
        aggregate.reports.append(report)
        aggregate.passed = all(
            model_report.passed for model_report in aggregate.reports
        )
        aggregate.complete = len(aggregate.reports) == len(representatives) and all(
            model_report.complete for model_report in aggregate.reports
        )
        _write_all_architectures_report(aggregate, output_json)
        if stop_on_failure and not report.passed:
            break
    return aggregate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ART Megatron model support workflow"
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--base-model")
    model_group.add_argument("--all-architectures", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--allow-unsupported-arch", action="store_true")
    parser.add_argument("--include-sensitivity", action="store_true")
    parser.add_argument("--include-yes-no-trainability", action="store_true")
    parser.add_argument("--skip-stage", action="append", default=[])
    parser.add_argument("--only-stage", choices=ALL_VALIDATION_STAGES)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)
    if args.only_stage and args.skip_stage:
        parser.error("--only-stage cannot be combined with --skip-stage")
    return args


def _print_stage_result(stage: ValidationStageResult, *, indent: str = "") -> None:
    status = "SKIP" if stage.skipped else "PASS" if stage.passed else "FAIL"
    print(f"{indent}{stage.name}: {status}", flush=True)
    child_indent = f"{indent}  "
    if stage.artifact_dir:
        print(f"{child_indent}artifact_dir={stage.artifact_dir}", flush=True)
    summary = stage.metrics.get("readable_summary")
    if isinstance(summary, list):
        for line in summary:
            print(f"{child_indent}{line}", flush=True)
    if not stage.passed and not stage.skipped:
        print(f"{child_indent}metrics={stage.metrics}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.only_stage is None and not args.stop_on_failure:
        from .workflow_scheduler import build_scheduled_validation_reports

        base_models = (
            validated_architecture_representative_models()
            if args.all_architectures
            else [args.base_model]
        )
        output_json_by_model: dict[str, Path | None] = {
            base_model: (
                _per_architecture_output_json(
                    args.output_json,
                    get_model_support_spec(
                        base_model,
                        allow_unvalidated_arch=args.allow_unsupported_arch,
                    ).key,
                )
                if args.all_architectures
                else Path(args.output_json)
            )
            for base_model in base_models
        }
        reports = build_scheduled_validation_reports(
            base_models=base_models,
            include_sensitivity=args.include_sensitivity,
            include_yes_no_trainability=args.include_yes_no_trainability,
            output_json_by_model=output_json_by_model,
            skip_stages=set(args.skip_stage),
            allow_unvalidated_arch=args.allow_unsupported_arch,
        )
        if args.all_architectures:
            all_report = AllArchitecturesValidationReport(
                reports=reports,
                passed=all(report.passed for report in reports),
                complete=all(report.complete for report in reports),
            )
            _write_all_architectures_report(all_report, args.output_json)
            for report in reports:
                print(f"base_model={report.base_model}", flush=True)
                for stage in report.stages:
                    _print_stage_result(stage, indent="  ")
            print(f"report_json={args.output_json}", flush=True)
            return 0 if all_report.passed else 1
        report = reports[0]
        for stage in report.stages:
            _print_stage_result(stage)
        print(f"report_json={args.output_json}", flush=True)
        return 0 if report.passed else 1
    if args.all_architectures:
        all_report = build_all_architectures_validation_report(
            include_yes_no_trainability=args.include_yes_no_trainability,
            include_sensitivity=args.include_sensitivity,
            output_json=args.output_json,
            skip_stages=set(args.skip_stage),
            only_stage=args.only_stage,
            stop_on_failure=args.stop_on_failure,
            allow_unvalidated_arch=args.allow_unsupported_arch,
        )
        for report in all_report.reports:
            print(f"base_model={report.base_model}", flush=True)
            for stage in report.stages:
                _print_stage_result(stage, indent="  ")
        print(f"report_json={args.output_json}", flush=True)
        return 0 if all_report.passed else 1
    report = build_validation_report(
        base_model=args.base_model,
        include_yes_no_trainability=args.include_yes_no_trainability,
        include_sensitivity=args.include_sensitivity,
        output_json=args.output_json,
        skip_stages=set(args.skip_stage),
        only_stage=args.only_stage,
        stop_on_failure=args.stop_on_failure,
        allow_unvalidated_arch=args.allow_unsupported_arch,
    )
    for stage in report.stages:
        _print_stage_result(stage)
    print(f"report_json={args.output_json}", flush=True)
    return 0 if report.passed else 1


def assess_minimal_layer_coverage(
    *,
    base_model: str,
    num_layers: int,
    architecture: ArchitectureReport | None = None,
    allow_unvalidated_arch: bool = False,
) -> MinimalLayerCoverageReport:
    architecture_report = architecture or (
        _inspect_architecture_for_workflow(
            base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
    )
    missing_layer_families = [
        family.key
        for family in architecture_report.layer_families
        if family.layer_index is not None and family.layer_index >= num_layers
    ]
    return MinimalLayerCoverageReport(
        base_model=base_model,
        model_key=architecture_report.model_key,
        requested_num_layers=num_layers,
        recommended_min_layers=architecture_report.recommended_min_layers,
        covered=not missing_layer_families and not architecture_report.unresolved_risks,
        missing_layer_families=missing_layer_families,
        unresolved_risks=list(architecture_report.unresolved_risks),
    )


if __name__ == "__main__":
    raise SystemExit(main())
