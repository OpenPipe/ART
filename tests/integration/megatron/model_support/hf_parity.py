from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from pydantic import BaseModel, Field

from ..artifacts import GitRepoState, pinned_git_state
from .oracle_harness import (
    NON_FINITE_METRIC_VALUE,
    ORACLE_TOPOLOGY,
    DiffAccumulator,
    DiskPackedTensorsSpec,
    MetricThresholdRule,
    OracleCaseConfig,
    PackedTensorConfig,
    PhasePassFn,
    _prune_case_artifacts,
    _read_json,
    _write_json,
    ensure_case_artifacts,
)
from .oracle_worker import provider_topology_env
from .validation_spec import MinimalLayerCoverageReport
from .workflow import assess_minimal_layer_coverage

HF_PARITY_ENABLE_ENV = "ART_RUN_HF_PARITY"
HF_PARITY_OUTPUT_DIRNAME = "hf_parity_sft"
HF_PARITY_REPORT_FILENAME = "report.json"
HF_PARITY_PACKED_TENSORS = PackedTensorConfig(
    sequence_length=256,
    prefill_tokens=64,
    decode_tokens=64,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
HF_PARITY_ARTIFACT_SUITE_NAME = "Megatron HF parity artifacts"


class HfParityMetricRow(BaseModel):
    phase: str
    param: str
    numel: float
    mean_abs_diff: float
    relative_l2: float
    typical_abs_scale: float
    candidate_abs_scale: float
    mean_abs_pct: float
    pass_signal: bool = True
    failure_reasons: list[str] = Field(default_factory=list)


class HfParityRunRequest(BaseModel):
    git: GitRepoState
    case_id: str
    case_config: OracleCaseConfig
    packed_tensors: DiskPackedTensorsSpec
    output_dir: str
    coverage: MinimalLayerCoverageReport


class HfParityReport(BaseModel):
    git: GitRepoState
    case_id: str
    base_model: str
    model_key: str
    requested_num_layers: int
    coverage: MinimalLayerCoverageReport
    signal: str
    pass_count: int
    fail_count: int
    metrics: list[HfParityMetricRow] = Field(default_factory=list)


def _hf_parity_phase_pass_fns() -> dict[str, PhasePassFn]:
    non_zero_scales = {"typical_abs_scale": 0.0, "candidate_abs_scale": 0.0}
    fwd_out = MetricThresholdRule(
        limits={"relative_l2": 1e-2, "mean_abs_pct": 1.0},
        minimums=non_zero_scales,
    )
    loss = MetricThresholdRule(
        limits={"relative_l2": 2e-2, "mean_abs_pct": 2.0},
        minimums=non_zero_scales,
    )
    grads_deltas = MetricThresholdRule(
        limits={"mean_abs_pct": 3.0},
        minimums=non_zero_scales,
    )
    return {
        "forward": fwd_out,
        "outputs": fwd_out,
        "losses": loss,
        "grads": grads_deltas,
        "deltas": grads_deltas,
    }


def hf_parity_case_config(case_config: OracleCaseConfig) -> OracleCaseConfig:
    return case_config.model_copy(
        update={"packed_tensors": HF_PARITY_PACKED_TENSORS.model_copy(deep=True)}
    )


def hf_parity_enabled() -> bool:
    value = os.environ.get(HF_PARITY_ENABLE_ENV)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _inf_summary() -> dict[str, float]:
    return {
        "numel": 0.0,
        "mean_abs_diff": NON_FINITE_METRIC_VALUE,
        "relative_l2": NON_FINITE_METRIC_VALUE,
        "typical_abs_scale": 0.0,
        "candidate_abs_scale": 0.0,
        "mean_abs_pct": NON_FINITE_METRIC_VALUE,
    }


def _build_metric_row(
    *,
    phase: str,
    param: str,
    summary: dict[str, float],
    structural_failure: str | None = None,
    phase_pass_fns: dict[str, PhasePassFn] | None = None,
) -> HfParityMetricRow:
    row = HfParityMetricRow(
        phase=phase,
        param=param,
        numel=summary["numel"],
        mean_abs_diff=summary["mean_abs_diff"],
        relative_l2=summary["relative_l2"],
        typical_abs_scale=summary["typical_abs_scale"],
        candidate_abs_scale=summary["candidate_abs_scale"],
        mean_abs_pct=summary["mean_abs_pct"],
    )
    pass_fn = (phase_pass_fns or _hf_parity_phase_pass_fns()).get(phase)
    if pass_fn is None:
        row.pass_signal = structural_failure is None
        if structural_failure is not None:
            row.failure_reasons = [structural_failure]
        return row
    row.pass_signal = bool(pass_fn(summary))
    explain = getattr(pass_fn, "failure_reasons", None)
    if callable(explain) and not row.pass_signal:
        row.failure_reasons = list(explain(summary))
    if structural_failure is not None:
        row.pass_signal = False
        row.failure_reasons = [structural_failure, *row.failure_reasons]
    return row


def summarize_tensor_pair(reference: Any, candidate: Any) -> dict[str, float]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return _inf_summary()
    accumulator = DiffAccumulator()
    accumulator.update(reference, candidate)
    return accumulator.as_summary()


def build_tensor_map_metric_rows(
    *,
    phase: str,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    phase_pass_fns: dict[str, PhasePassFn] | None = None,
) -> list[HfParityMetricRow]:
    reference_keys = set(reference.keys())
    candidate_keys = set(candidate.keys())
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        extra = sorted(candidate_keys - reference_keys)
        return [
            _build_metric_row(
                phase=phase,
                param="__tensor_set__",
                summary=_inf_summary(),
                structural_failure=f"missing={missing[:5]} extra={extra[:5]}",
                phase_pass_fns=phase_pass_fns,
            )
        ]
    rows: list[HfParityMetricRow] = []
    for key in sorted(reference_keys):
        if tuple(reference[key].shape) != tuple(candidate[key].shape):
            rows.append(
                _build_metric_row(
                    phase=phase,
                    param=key,
                    summary=_inf_summary(),
                    structural_failure=f"shape mismatch for '{key}'",
                    phase_pass_fns=phase_pass_fns,
                )
            )
            continue
        rows.append(
            _build_metric_row(
                phase=phase,
                param=key,
                summary=summarize_tensor_pair(reference[key], candidate[key]),
                phase_pass_fns=phase_pass_fns,
            )
        )
    return rows


def build_parity_sample_indices(
    *,
    num_sequences: int,
    global_grad_accumulation_sequences: int,
) -> list[int | None]:
    return [
        index if index < num_sequences else None
        for index in range(global_grad_accumulation_sequences)
    ]


def _iter_hf_layer_config_views(config: Any) -> list[tuple[str, Any]]:
    views: list[tuple[str, Any]] = [("", config)]
    base_config_key = getattr(config, "base_config_key", None)
    candidate_names = [
        name
        for name in [
            base_config_key if isinstance(base_config_key, str) else None,
            "text_config",
            "language_config",
            "llm_config",
            "decoder_config",
        ]
        if isinstance(name, str)
    ]
    seen_ids = {id(config)}
    for name in candidate_names:
        nested = getattr(config, name, None)
        if nested is None or id(nested) in seen_ids:
            continue
        seen_ids.add(id(nested))
        views.append((f"{name}.", nested))
    return views


def set_hf_config_num_layers(config: Any, num_layers: int) -> str:
    for prefix, config_view in _iter_hf_layer_config_views(config):
        for field in ("num_hidden_layers", "num_layers", "n_layer"):
            if not hasattr(config_view, field):
                continue
            setattr(config_view, field, num_layers)
            layer_types = getattr(config_view, "layer_types", None)
            if isinstance(layer_types, (list, tuple)):
                setattr(config_view, "layer_types", list(layer_types[:num_layers]))
            mlp_only_layers = getattr(config_view, "mlp_only_layers", None)
            if isinstance(mlp_only_layers, (list, tuple)):
                setattr(
                    config_view,
                    "mlp_only_layers",
                    [layer for layer in mlp_only_layers if int(layer) < num_layers],
                )
            return f"{prefix}{field}"
    raise ValueError(
        f"Could not find a supported layer-count field on HF config type {type(config)}"
    )


def zero_hf_dropout_config(config: Any) -> None:
    for field in (
        "attention_dropout",
        "hidden_dropout",
        "dropout",
        "embd_pdrop",
        "resid_pdrop",
        "attn_pdrop",
        "classifier_dropout",
    ):
        if hasattr(config, field):
            setattr(config, field, 0.0)


def assert_hf_parity_pass(report: HfParityReport, *, report_path: Path) -> None:
    if report.signal == "pass":
        return
    first_failure = next(row for row in report.metrics if not row.pass_signal)
    raise AssertionError(
        f"HF parity failed: phase={first_failure.phase} param={first_failure.param} "
        f"reasons={'; '.join(first_failure.failure_reasons)} report={report_path}"
    )


def run_hf_parity_subprocess(request: HfParityRunRequest, output_dir: Path) -> None:
    request_path = output_dir / "run_request.json"
    _write_json(request_path, request.model_dump(mode="json"))
    worker_cwd = REPO_ROOT / "tests"
    command = [
        sys.executable,
        "-m",
        "integration.megatron.model_support.hf_parity_worker",
        "--run-request",
        str(request_path),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    run = subprocess.run(
        command,
        cwd=str(worker_cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = f"{run.stdout}\n{run.stderr}".strip()
    (output_dir / "worker.log").write_text(combined_output + "\n", encoding="utf-8")
    if run.returncode != 0:
        tail = "\n".join(combined_output.splitlines()[-80:])
        raise RuntimeError(
            f"HF parity worker failed with exit code {run.returncode}.\n{tail}"
        )


def run_hf_parity(
    *,
    case_config: OracleCaseConfig,
) -> HfParityReport:
    case_config = hf_parity_case_config(case_config)
    if case_config.precision != "fp32":
        raise ValueError("HF parity currently requires fp32 precision")
    if case_config.num_steps != 1:
        raise ValueError("HF parity currently requires num_steps=1")

    coverage = assess_minimal_layer_coverage(
        base_model=case_config.base_model,
        num_layers=case_config.num_layers,
        allow_unvalidated_arch=case_config.allow_unvalidated_arch,
    )
    if not coverage.covered:
        raise AssertionError(
            "HF parity toy model does not cover required layer families: "
            f"missing={coverage.missing_layer_families} "
            f"risks={coverage.unresolved_risks}"
        )

    git = pinned_git_state(HF_PARITY_ARTIFACT_SUITE_NAME)
    case_artifacts = ensure_case_artifacts(case_config)
    output_dir = Path(case_artifacts.case_dir) / HF_PARITY_OUTPUT_DIRNAME
    report_path = output_dir / HF_PARITY_REPORT_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        report_path.unlink()
    request = HfParityRunRequest(
        git=git,
        case_id=case_artifacts.case_id,
        case_config=case_config,
        packed_tensors=case_artifacts.packed_tensors,
        output_dir=str(output_dir),
        coverage=coverage,
    )
    with provider_topology_env(ORACLE_TOPOLOGY):
        run_hf_parity_subprocess(request, output_dir)
    report = HfParityReport.model_validate(_read_json(report_path))
    assert_hf_parity_pass(report, report_path=report_path)
    _prune_case_artifacts(Path(case_artifacts.case_dir))
    return report


def build_hf_parity_report(
    *,
    request: HfParityRunRequest,
    outputs_summary: dict[str, float],
    loss_summary: dict[str, float],
    grads_rows: list[HfParityMetricRow],
) -> HfParityReport:
    phase_pass_fns = _hf_parity_phase_pass_fns()
    rows = [
        _build_metric_row(
            phase="outputs",
            param="trainable_token_losses",
            summary=outputs_summary,
            phase_pass_fns=phase_pass_fns,
        ),
        _build_metric_row(
            phase="losses",
            param="loss",
            summary=loss_summary,
            phase_pass_fns=phase_pass_fns,
        ),
        *grads_rows,
    ]
    pass_count = sum(1 for row in rows if row.pass_signal)
    fail_count = len(rows) - pass_count
    return HfParityReport(
        git=request.git,
        case_id=request.case_id,
        base_model=request.case_config.base_model,
        model_key=request.coverage.model_key,
        requested_num_layers=request.case_config.num_layers,
        coverage=request.coverage,
        signal="pass" if fail_count == 0 else "fail",
        pass_count=pass_count,
        fail_count=fail_count,
        metrics=rows,
    )


__all__ = [
    "HF_PARITY_ENABLE_ENV",
    "HF_PARITY_OUTPUT_DIRNAME",
    "HF_PARITY_PACKED_TENSORS",
    "HF_PARITY_REPORT_FILENAME",
    "HfParityMetricRow",
    "HfParityReport",
    "HfParityRunRequest",
    "assert_hf_parity_pass",
    "build_hf_parity_report",
    "build_parity_sample_indices",
    "build_tensor_map_metric_rows",
    "hf_parity_case_config",
    "hf_parity_enabled",
    "run_hf_parity",
    "set_hf_config_num_layers",
    "summarize_tensor_pair",
    "zero_hf_dropout_config",
]
