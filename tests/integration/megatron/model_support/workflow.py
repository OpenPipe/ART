import argparse
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import importlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from pydantic import BaseModel, Field

from art.megatron.model_support.discovery import inspect_architecture
from art.megatron.model_support.registry import (
    VALIDATED_MODEL_SUPPORT_SPECS,
    get_model_support_handler_for_spec,
    get_model_support_spec,
)
from art.megatron.model_support.spec import (
    ArchitectureReport,
    NativeVllmLoraStatus,
)

from ..artifacts import pinned_git_state
from .validation_spec import (
    MinimalLayerCoverageReport,
    ValidationReport,
    ValidationStageResult,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
TESTS_DIR = REPO_ROOT / "tests"
LOCAL_LOG_DIR = REPO_ROOT / ".local"
CORRECTNESS_LOG_PATH = LOCAL_LOG_DIR / "correctness.log"
SENSITIVITY_LOG_PATH = LOCAL_LOG_DIR / "sensitivity.log"
LIVE_TRAINING_LOG_PATH = LOCAL_LOG_DIR / "live_training.log"
ORACLE_LIVE_TRAINING_LOG_ENV = "ART_ORACLE_LIVE_TRAINING_LOG"
SKIP_SENSITIVITY_ENV = "ART_MODEL_SUPPORT_SKIP_SENSITIVITY"
INCLUDE_FLASH_SENSITIVITY_ENV = "ART_MODEL_SUPPORT_INCLUDE_FLASH_SENSITIVITY"
KEEP_TOPOLOGY_ARTIFACTS_ENV = "ART_ORACLE_KEEP_TOPOLOGY_ARTIFACTS"
WORKFLOW_ARTIFACT_SUITE_NAME = "Megatron model-support validation workflow"
FLASH_SENSITIVITY_MUTATION = "attn_skip_flash_lse_normalize"

MANDATORY_VALIDATION_STAGES = (
    "dependency_resolution",
    "architecture_discovery",
    "hf_parity",
    "lora_coverage",
    "train_inf_mismatch",
    "merged_vllm_serving",
    "correctness_sensitivity",
    "chat_template_rollout",
    "packed_position_ids",
    "length_trainability",
)
NATIVE_VLLM_LORA_STAGE = "native_vllm_lora"
YES_NO_TRAINABILITY_STAGE = "yes_no_trainability"
OPTIONAL_VALIDATION_STAGES = (
    YES_NO_TRAINABILITY_STAGE,
    NATIVE_VLLM_LORA_STAGE,
)
ALL_VALIDATION_STAGES = (*MANDATORY_VALIDATION_STAGES, *OPTIONAL_VALIDATION_STAGES)
ARCHITECTURE_REPRESENTATIVE_MODELS = {
    "qwen3_moe": "Qwen/Qwen3-30B-A3B",
    "qwen3_dense": "Qwen/Qwen3-32B",
    "qwen3_5_moe": "Qwen/Qwen3.5-35B-A3B",
    "qwen3_5_dense": "Qwen/Qwen3.5-27B",
    "gemma4_moe": "google/gemma-4-26B-A4B-it",
    "gemma4_dense": "google/gemma-4-31B-it",
    "dsv4": "deepseek-ai/DeepSeek-V4-Flash",
    "gpt_oss_moe": "openai/gpt-oss-20b",
}
SUBPROCESS_VALIDATION_STAGES = frozenset(
    {
        "hf_parity",
        "lora_coverage",
        "train_inf_mismatch",
        "merged_vllm_serving",
        "correctness_sensitivity",
        "chat_template_rollout",
        "packed_position_ids",
        "length_trainability",
        YES_NO_TRAINABILITY_STAGE,
        NATIVE_VLLM_LORA_STAGE,
    }
)


class AllArchitecturesValidationReport(BaseModel):
    passed: bool = False
    reports: list[ValidationReport] = Field(default_factory=list)


def build_validation_stage_names(
    *,
    include_native_vllm_lora: bool = False,
    include_yes_no_trainability: bool = False,
    native_vllm_lora_status: NativeVllmLoraStatus | None = None,
) -> list[str]:
    stages = list(MANDATORY_VALIDATION_STAGES)
    if include_yes_no_trainability:
        stages.append(YES_NO_TRAINABILITY_STAGE)
    if include_native_vllm_lora or native_vllm_lora_status not in {None, "disabled"}:
        stages.append(NATIVE_VLLM_LORA_STAGE)
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
    include_native_vllm_lora: bool = False,
    include_yes_no_trainability: bool = False,
    allow_unvalidated_arch: bool = False,
) -> ValidationReport:
    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    handler = get_model_support_handler_for_spec(spec)
    return ValidationReport(
        git=pinned_git_state(WORKFLOW_ARTIFACT_SUITE_NAME).model_dump(mode="json"),
        base_model=base_model,
        model_key=spec.key,
        dependency_versions=detect_dependency_versions(),
        stages=[
            ValidationStageResult(name=stage_name)
            for stage_name in build_validation_stage_names(
                include_native_vllm_lora=include_native_vllm_lora,
                include_yes_no_trainability=include_yes_no_trainability,
                native_vllm_lora_status=handler.native_vllm_lora_status,
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


def _write_validation_report(
    report: ValidationReport,
    output_json: str | Path | None,
) -> None:
    if output_json is None:
        return
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


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
) -> None:
    past_failure = False
    for stage in report.stages:
        if past_failure:
            stage.metrics = {
                "skipped": True,
                "reason": f"stopped after {after_stage_name} failed",
            }
            continue
        past_failure = stage.name == after_stage_name


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
) -> ValidationStageResult:
    with tempfile.TemporaryDirectory(prefix=f"model_support_{stage_name}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        architecture_json = tmp_path / "architecture.json"
        output_json = tmp_path / "stage_result.json"
        log_path = tmp_path / "stage.log"
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
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(TESTS_DIR)
            if not existing_pythonpath
            else f"{TESTS_DIR}{os.pathsep}{existing_pythonpath}"
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            tail = _subprocess_log_tail(log_path)
            error = (
                f"subprocess exited with code {completed.returncode}"
                if not tail
                else tail
            )
            return ValidationStageResult(
                name=stage_name,
                passed=False,
                metrics={"error": error},
            )
        if not output_json.exists():
            return ValidationStageResult(
                name=stage_name,
                passed=False,
                metrics={"error": "stage worker did not write output_json"},
            )
        return ValidationStageResult.model_validate_json(
            output_json.read_text(encoding="utf-8")
        )


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
    case_config = oracle_harness.OracleCaseConfig(
        base_model=base_model,
        is_moe=handler.is_moe,
        precision="fp32",
        num_layers=max(1, architecture.recommended_min_layers),
        num_steps=1,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    case_config = hf_parity.hf_parity_case_config(case_config)
    report = hf_parity.run_hf_parity(case_config=case_config)
    case_artifacts = oracle_harness.ensure_case_artifacts(case_config)
    artifact_dir = str(
        Path(case_artifacts.case_dir) / hf_parity.HF_PARITY_OUTPUT_DIRNAME
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
    case_config = oracle_harness.OracleCaseConfig(
        base_model=base_model,
        is_moe=handler.is_moe,
        precision="fp32",
        num_layers=max(1, architecture.recommended_min_layers),
        num_steps=1,
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
    case_config = oracle_harness.OracleCaseConfig(
        base_model=base_model,
        is_moe=handler.is_moe,
        precision=correctness_precision,
        num_layers=max(1, architecture.recommended_min_layers),
        num_steps=1,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    suite_topologies = list(
        oracle_harness.selected_suite_topologies(
            is_moe=handler.is_moe,
            cp_supported=cp_supported,
        )
    )
    objectives = list(oracle_harness.selected_oracle_objectives())
    skip_sensitivity = _truthy_env(SKIP_SENSITIVITY_ENV)
    available_gpu_count = oracle_harness.available_gpu_count()
    max_world_size = available_gpu_count
    oracle_world_size = oracle_harness.oracle_topology(
        is_moe=handler.is_moe
    ).world_size()
    if available_gpu_count < oracle_world_size:
        raise RuntimeError(
            "Need "
            f"{oracle_world_size} GPUs for oracle topology, found {available_gpu_count}"
        )
    selected_suite_topologies = [
        topology
        for topology in suite_topologies
        if topology.world_size() <= max_world_size
    ]
    excluded_suite_topologies = [
        topology
        for topology in suite_topologies
        if topology.world_size() > max_world_size
    ]
    mutations: list[str] = []
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
        if not _truthy_env(INCLUDE_FLASH_SENSITIVITY_ENV):
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
    LIVE_TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_TRAINING_LOG_PATH.write_text("", encoding="utf-8")
    with _temporary_env(**{ORACLE_LIVE_TRAINING_LOG_ENV: str(LIVE_TRAINING_LOG_PATH)}):
        with _redirect_output(CORRECTNESS_LOG_PATH):
            suite_reports = oracle_harness.run_suite(
                case_config=case_config,
                max_world_size=max_world_size,
                cp_supported=cp_supported,
                phase_pass_fns=correctness_phase_pass_fns,
                use_fp32_lora_reference=correctness_use_fp32_lora_reference,
                prune_reference_artifacts=skip_sensitivity or not mutations,
                prune_case_artifacts=skip_sensitivity or not mutations,
            )
        sensitivity_reports = []
        if skip_sensitivity:
            SENSITIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            SENSITIVITY_LOG_PATH.write_text(
                (
                    "Sensitivity suite skipped. "
                    f"Set {SKIP_SENSITIVITY_ENV}=0 to re-enable workflow sensitivity.\n"
                ),
                encoding="utf-8",
            )
        elif not mutations:
            SENSITIVITY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            SENSITIVITY_LOG_PATH.write_text(
                (
                    "Sensitivity suite skipped. "
                    f"No sensitivity mutations fit max_world_size={max_world_size}.\n"
                ),
                encoding="utf-8",
            )
        else:
            with _redirect_output(SENSITIVITY_LOG_PATH):
                sensitivity_reports = oracle_harness.run_sensitivity_suite(
                    case_config=case_config,
                    mutations=mutations,
                    max_world_size=max_world_size,
                )
    case_artifacts = oracle_harness.ensure_case_artifacts(case_config)
    return ValidationStageResult(
        name="correctness_sensitivity",
        passed=True,
        metrics={
            "requested_num_layers": case_config.num_layers,
            "precision": correctness_precision,
            "use_fp32_lora_reference": correctness_use_fp32_lora_reference,
            "is_moe": handler.is_moe,
            "cp_supported": cp_supported,
            "allow_unvalidated_arch": allow_unvalidated_arch,
            "objectives": objectives,
            "sensitivity_mutations": mutations,
            "excluded_sensitivity_mutations": excluded_sensitivity_mutations,
            "default_excluded_sensitivity_mutations": (
                default_excluded_sensitivity_mutations
            ),
            "available_gpu_count": available_gpu_count,
            "max_world_size": max_world_size,
            "required_gpu_count": oracle_world_size,
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


def run_merged_vllm_serving_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    merged_vllm_serving = _import_integration_module(
        "integration.megatron.lora.merged_vllm_serving"
    )
    oracle_harness = _import_integration_module(
        "integration.megatron.model_support.oracle_harness"
    )
    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    handler = get_model_support_handler_for_spec(spec)
    case_config = oracle_harness.OracleCaseConfig(
        base_model=base_model,
        is_moe=handler.is_moe,
        precision="fp32",
        num_layers=max(1, architecture.recommended_min_layers),
        num_steps=1,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    report = merged_vllm_serving.run_merged_vllm_serving(case_config)
    metrics = report.model_dump(mode="json")
    warning_lines = _read_vllm_reload_warnings(report.output_dir)
    metrics["vllm_reload_warning_count"] = len(warning_lines)
    metrics["vllm_reload_warnings"] = warning_lines
    metrics["readable_summary"] = _merged_vllm_serving_summary(metrics)
    return ValidationStageResult(
        name="merged_vllm_serving",
        passed=bool(report.model_ids),
        metrics=metrics,
        artifact_dir=report.output_dir,
    )


def _read_vllm_reload_warnings(output_dir: str) -> list[str]:
    log_path = Path(output_dir) / "logs" / "vllm-runtime.log"
    if not log_path.exists():
        return []
    return [
        line.strip()
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "Failed to load weights" in line
    ]


def _merged_vllm_serving_summary(metrics: dict[str, Any]) -> list[str]:
    lines = [
        f"served_model_name={metrics.get('served_model_name', '')}",
        f"model_ids={metrics.get('model_ids', [])}",
        f"completion_text={metrics.get('completion_text', '')!r}",
        f"vllm_reload_warning_count={metrics.get('vllm_reload_warning_count', 0)}",
    ]
    for warning in metrics.get("vllm_reload_warnings", []):
        lines.append(f"vllm_reload_warning={warning}")
    return lines


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
    report = length_trainability.run_length_trainability(
        base_model=base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return ValidationStageResult(
        name="length_trainability",
        passed=length_trainability.length_trainability_passed(report),
        metrics=report.model_dump(mode="json"),
        artifact_dir=str(Path(report.summary_log_path).parent),
    )


def run_native_vllm_lora_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    native_vllm_lora = _import_integration_module(
        "integration.megatron.lora.native_vllm_lora"
    )
    oracle_harness = _import_integration_module(
        "integration.megatron.model_support.oracle_harness"
    )
    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    handler = get_model_support_handler_for_spec(spec)
    case_config = oracle_harness.OracleCaseConfig(
        base_model=base_model,
        is_moe=handler.is_moe,
        precision="fp32",
        num_layers=max(1, architecture.recommended_min_layers),
        num_steps=1,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    report = native_vllm_lora.run_native_vllm_lora(case_config)
    passed = (
        report.rollout_weights_mode == "lora"
        and report.step0_served
        and report.step1_served
        and report.step0_name in report.model_ids_before
        and report.step1_name not in report.model_ids_before
        and report.step0_name in report.model_ids_after
        and report.step1_name in report.model_ids_after
    )
    return ValidationStageResult(
        name=NATIVE_VLLM_LORA_STAGE,
        passed=passed,
        metrics=report.model_dump(mode="json"),
        artifact_dir=report.output_dir,
    )


def run_packed_position_ids_stage(
    *,
    base_model: str,
    architecture: ArchitectureReport,
    allow_unvalidated_arch: bool = False,
) -> ValidationStageResult:
    packed_position_ids = _import_integration_module(
        "integration.megatron.model_support.packed_position_ids"
    )
    report = packed_position_ids.run_packed_position_ids(
        base_model=base_model,
        num_layers=max(1, architecture.recommended_min_layers),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    metrics = report.model_dump(mode="json")
    passed = bool(metrics["scenarios"]) and all(
        scenario["matched"] and scenario["checked_token_count"] > 0
        for scenario in metrics["scenarios"]
    )
    return ValidationStageResult(
        name="packed_position_ids",
        passed=passed,
        metrics=metrics,
        artifact_dir=report.output_dir,
    )


def build_validation_report(
    *,
    base_model: str,
    include_native_vllm_lora: bool = False,
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
        include_native_vllm_lora=(
            include_native_vllm_lora or only_stage == NATIVE_VLLM_LORA_STAGE
        ),
        include_yes_no_trainability=(
            include_yes_no_trainability or only_stage == YES_NO_TRAINABILITY_STAGE
        ),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    stage_runners = {
        "hf_parity": run_hf_parity_stage,
        "lora_coverage": run_lora_coverage_stage,
        "train_inf_mismatch": run_train_inf_mismatch_stage,
        "merged_vllm_serving": run_merged_vllm_serving_stage,
        "correctness_sensitivity": run_correctness_sensitivity_stage,
        "chat_template_rollout": run_chat_template_rollout_stage,
        "packed_position_ids": run_packed_position_ids_stage,
        "length_trainability": run_length_trainability_stage,
        YES_NO_TRAINABILITY_STAGE: run_yes_no_trainability_stage,
        NATIVE_VLLM_LORA_STAGE: run_native_vllm_lora_stage,
    }
    env = {}
    if include_sensitivity is not None:
        env[SKIP_SENSITIVITY_ENV] = "0" if include_sensitivity else "1"
    if include_sensitivity:
        env[KEEP_TOPOLOGY_ARTIFACTS_ENV] = "1"
    skip_stages = skip_stages or set()
    architecture: ArchitectureReport | None = None
    context = _temporary_env(**env) if env else nullcontext()
    with context:
        for stage in report.stages:
            if only_stage_run_set is not None and stage.name not in only_stage_run_set:
                stage.passed = True
                stage.metrics = {
                    "skipped": True,
                    "reason": f"--only-stage={only_stage}",
                }
                _write_validation_report(report, output_json)
                continue
            if stage.name in skip_stages:
                stage.passed = True
                stage.metrics = {"skipped": True, "reason": "--skip-stage"}
                _write_validation_report(report, output_json)
                continue
            if stage.name == "dependency_resolution":
                stage.passed = True
                stage.metrics = dict(report.dependency_versions)
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
                _write_validation_report(report, output_json)
                if stop_on_failure and not stage.passed:
                    _mark_remaining_stages_skipped(report, after_stage_name=stage.name)
                    _write_validation_report(report, output_json)
                    break
                continue
            if architecture is None:
                raise RuntimeError(
                    "architecture_discovery must run before subprocess stages"
                )
            stage_runner = stage_runners[stage.name]
            if stage.name in SUBPROCESS_VALIDATION_STAGES:
                stage_result = _run_stage_in_subprocess(
                    stage_name=stage.name,
                    base_model=base_model,
                    architecture=architecture,
                    allow_unvalidated_arch=allow_unvalidated_arch,
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
            _write_validation_report(report, output_json)
            if stop_on_failure and not stage.passed:
                _mark_remaining_stages_skipped(report, after_stage_name=stage.name)
                _write_validation_report(report, output_json)
                break
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
    _write_all_architectures_report(aggregate, output_json)
    for base_model in validated_architecture_representative_models():
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
            all(stage.passed for stage in model_report.stages)
            for model_report in aggregate.reports
        )
        _write_all_architectures_report(aggregate, output_json)
        if stop_on_failure and not all(stage.passed for stage in report.stages):
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
    status = "PASS" if stage.passed else "FAIL"
    print(f"{indent}{stage.name}: {status}", flush=True)
    child_indent = f"{indent}  "
    if stage.artifact_dir:
        print(f"{child_indent}artifact_dir={stage.artifact_dir}", flush=True)
    summary = stage.metrics.get("readable_summary")
    if isinstance(summary, list):
        for line in summary:
            print(f"{child_indent}{line}", flush=True)
    if not stage.passed:
        print(f"{child_indent}metrics={stage.metrics}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
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
    return 0 if all(stage.passed for stage in report.stages) else 1


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
