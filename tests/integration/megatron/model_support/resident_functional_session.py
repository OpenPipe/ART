from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any, Literal

from .lora_coverage import build_lora_coverage_report
from .validation_spec import ValidationStageResult

_PARITY_LEARNING_RATE = 1e-6


async def run_resident_functional_session(
    *,
    base_model: str,
    allow_unvalidated_arch: bool,
    stage_dirs: dict[str, Path],
    serving_ready: asyncio.Future[tuple[str, ...]] | None = None,
) -> tuple[ValidationStageResult, ...]:
    from art.megatron.runtime.specs import ResidentLoraInspectionResult

    from ..train_inf_mismatch.real_path import (
        config_from_env,
        run_resident_train_inf_mismatch,
    )
    from ..train_inf_mismatch.workflow_stage import _attempt_limit
    from ..trainability import test_live_length_trainability as length_trainability

    coverage_report = None
    coverage_rank_summaries: list[dict[str, Any]] = []
    coverage_run_id: str | None = None
    mismatch_report = None
    coverage_s = 0.0
    mismatch_s = 0.0
    session_started = time.monotonic()
    length_dir = stage_dirs["length_trainability"] / "artifacts"
    length_trainability.LATEST_SUMMARY_LOG_PATH = (
        stage_dirs["length_trainability"] / "length_trainability.log"
    )

    async def hook(
        phase: Literal["registered", "first_update"],
        backend: Any,
        model: Any,
        step: int,
    ) -> None:
        nonlocal coverage_report, coverage_rank_summaries, coverage_run_id
        nonlocal coverage_s, mismatch_report, mismatch_s
        if phase == "registered":
            if step != 0:
                raise RuntimeError(
                    f"resident functional session must start at step 0, got {step}"
                )
            started = time.monotonic()
            inspection = ResidentLoraInspectionResult.model_validate(
                await backend.inspect_resident_lora(
                    model, expected_learner_version=step
                )
            )
            coverage_run_id = inspection.run_id
            coverage_rank_summaries = [
                summary.model_dump(mode="json") for summary in inspection.rank_summaries
            ]
            coverage_report = build_lora_coverage_report(
                base_model=base_model,
                target_modules=list(inspection.target_modules),
                adapter_prefixes=set(inspection.wrapped_adapter_prefixes),
                adapter_weights_by_base={
                    export.base_name: list(export.adapter_keys)
                    for export in inspection.exports
                },
                trainable_lora_parameter_names=set(
                    inspection.trainable_lora_parameter_names
                ),
                unexpected_trainable_parameter_names=set(
                    inspection.unexpected_trainable_parameter_names
                ),
            )
            coverage_s = time.monotonic() - started
            artifact_dir = stage_dirs["lora_coverage"] / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "resident_lora_coverage.json").write_text(
                coverage_report.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            return
        if phase != "first_update" or step != 1:
            raise RuntimeError(f"unexpected resident functional hook {phase}@{step}")
        started = time.monotonic()
        config = config_from_env()
        config.output_parity.base_model = base_model
        config.output_parity.allow_unvalidated_arch = allow_unvalidated_arch
        mismatch_report = await run_resident_train_inf_mismatch(
            backend=backend,
            model=model,
            policy_step=step,
            config=config,
            artifact_dir=stage_dirs["train_inf_mismatch"] / "artifacts",
            max_attempts=_attempt_limit(),
        )
        if mismatch_report.run_id != coverage_run_id:
            raise RuntimeError("resident functional phases used different trainer runs")
        mismatch_s = time.monotonic() - started

    try:
        report = await length_trainability.run_length_trainability_async(
            base_model=base_model,
            artifact_dir=length_dir,
            allow_unvalidated_arch=allow_unvalidated_arch,
            resident_hook=hook,
            registration_ready=serving_ready,
            first_update_learning_rate=_PARITY_LEARNING_RATE,
        )
    finally:
        from .workflow import _cleanup_stage_workspace

        _cleanup_stage_workspace(length_dir / "megatron_dedicated_workspace")
    if coverage_report is None or mismatch_report is None:
        raise RuntimeError("resident functional session did not execute every phase")

    first_update_s = sum(
        phase.duration_s for phase in report.phases if phase.name == "first_update"
    )
    continuation_s = sum(
        phase.duration_s for phase in report.phases if phase.name == "continuation"
    )
    session_s = time.monotonic() - session_started
    common = {"resident_functional_session_s": session_s}
    return (
        ValidationStageResult(
            name="lora_coverage",
            passed=not coverage_report.missing_wrapped_target_modules
            and not coverage_report.missing_exported_target_modules
            and coverage_report.trainable_lora_parameter_count > 0
            and not coverage_report.unexpected_trainable_parameter_names
            and all(
                summary["module_count"] > 0 and summary["trainable_parameter_count"] > 0
                for summary in coverage_rank_summaries
            ),
            metrics=coverage_report.model_dump(mode="json")
            | common
            | {
                "resident_rank_summaries": coverage_rank_summaries,
                "workflow_stage_duration_s": coverage_s,
            },
            artifact_dir=str(stage_dirs["lora_coverage"] / "artifacts"),
        ),
        ValidationStageResult(
            name="train_inf_mismatch",
            passed=mismatch_report.passed,
            metrics=mismatch_report.model_dump(mode="json")
            | common
            | {"workflow_stage_duration_s": first_update_s + mismatch_s},
            artifact_dir=mismatch_report.artifact_dir,
        ),
        ValidationStageResult(
            name="length_trainability",
            passed=length_trainability.length_trainability_passed(report),
            metrics=report.model_dump(mode="json")
            | common
            | {"workflow_stage_duration_s": continuation_s},
            artifact_dir=str(length_dir),
        ),
    )
