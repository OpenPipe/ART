import argparse
from pathlib import Path
import time
import traceback

from pydantic import BaseModel, ConfigDict

from art.megatron.model_support.spec import ArchitectureReport

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


class WorkflowStageWorkerSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_model: str
    architecture_json: str
    allow_unvalidated_arch: bool = False
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


def _run_session(request: WorkflowStageWorkerSession) -> None:
    from . import workflow

    architecture = ArchitectureReport.model_validate_json(
        Path(request.architecture_json).read_text(encoding="utf-8")
    )
    for item in request.items:
        started = time.monotonic()
        stage_dir = Path(item.stage_dir)
        log_path = stage_dir / "worker.log"
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
        result.metrics.update(
            {
                "workflow_stage_artifact_dir": item.stage_dir,
                "workflow_stage_duration_s": time.monotonic() - started,
            }
        )
        Path(item.output_json).write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )


def main() -> None:
    args = _parse_args()
    if args.session_json is not None:
        request = WorkflowStageWorkerSession.model_validate_json(
            Path(args.session_json).read_text(encoding="utf-8")
        )
        _run_session(request)
        return
    architecture = ArchitectureReport.model_validate_json(
        Path(args.architecture_json).read_text(encoding="utf-8")
    )
    stage_runner = _STAGE_RUNNERS[args.stage]
    result = stage_runner(
        base_model=args.base_model,
        architecture=architecture,
        allow_unvalidated_arch=args.allow_unvalidated_arch,
    )
    Path(args.output_json).write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
