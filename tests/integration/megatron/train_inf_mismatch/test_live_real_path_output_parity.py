from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import traceback

import pytest

from art.megatron.model_support.registry import get_model_support_spec

from .output_parity import model_support_is_moe
from .real_path import (
    RealPathConfig,
    RealPathTrainInfReport,
    config_from_env,
    run_real_path_train_inf_mismatch,
)
from .workflow_stage import (
    ATTEMPT_ASSERTION_EXIT_CODE,
    ATTEMPT_ERROR_EXIT_CODE,
    TrainInfMismatchWorkerResult,
)

_TEST_NODEID = (
    "tests/integration/megatron/train_inf_mismatch/"
    "test_live_real_path_output_parity.py::test_real_path_train_inf_mismatch_live"
)


def _require_visible_gpus(gpu_ids: list[int]) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real-path train/inf mismatch")
    visible_count = int(torch.cuda.device_count())
    required = max(gpu_ids) + 1 if gpu_ids else 0
    if visible_count < required:
        pytest.skip(
            f"Need visible CUDA device ids through {required - 1}, "
            f"but torch sees {visible_count} devices"
        )


async def _run_live_real_path_output_parity(
    artifact_dir: Path,
) -> tuple[RealPathConfig, RealPathTrainInfReport]:
    config = config_from_env()
    parity_config = config.output_parity
    _require_visible_gpus(
        parity_config.trainer_gpu_ids + parity_config.inference_gpu_ids
    )

    report = await run_real_path_train_inf_mismatch(
        config=config,
        artifact_dir=artifact_dir,
    )
    return config, report


def assert_live_real_path_output_parity(
    config: RealPathConfig,
    report: RealPathTrainInfReport,
) -> None:
    parity_config = config.output_parity
    assert report.logical_prompt_count > 0
    assert report.logical_token_count > 0
    handler_key = get_model_support_spec(
        parity_config.base_model,
        allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
    ).handler_key
    if handler_key == "dsv4":
        assert report.prompt_tree_depth == 2
        assert report.prompt_tree_branch_count == 6
    elif config.sliding_window is None:
        assert report.prompt_tree_depth > 2
        assert report.prompt_tree_branch_count >= 14
    if model_support_is_moe(
        parity_config.base_model,
        allow_unvalidated_arch=parity_config.allow_unvalidated_arch,
    ):
        assert report.moe_routing_packed_tokens > 0
    assert report.passed, report.model_dump_json(indent=2)
    assert report.lora.mean_abs_pct <= report.mean_abs_pct_limit
    assert (
        report.lora_topk.top20_intersection_kl_candidate_to_target
        <= report.top20_kl_candidate_to_target_limit
    )


@pytest.mark.asyncio
async def test_real_path_train_inf_mismatch_live(artifact_dir: Path) -> None:
    config, report = await _run_live_real_path_output_parity(artifact_dir)
    assert_live_real_path_output_parity(config, report)


def _run_workflow_attempt(result_path: Path) -> int:
    from .artifacts import create_artifact_dir, require_clean_git_state

    artifact_dir: Path | None = None
    comparison_completed = False
    exception_type: str | None = None
    exception_message: str | None = None
    try:
        require_clean_git_state()
        artifact_dir = create_artifact_dir(_TEST_NODEID)
        config, report = asyncio.run(_run_live_real_path_output_parity(artifact_dir))
        comparison_completed = True
        assert_live_real_path_output_parity(config, report)
        outcome = "passed"
        returncode = 0
    except pytest.skip.Exception as error:
        traceback.print_exc()
        outcome = "skipped"
        returncode = 0
        exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
        exception_message = str(error)
    except AssertionError as error:
        traceback.print_exc()
        outcome = "failed" if comparison_completed else "error"
        returncode = (
            ATTEMPT_ASSERTION_EXIT_CODE
            if comparison_completed
            else ATTEMPT_ERROR_EXIT_CODE
        )
        exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
        exception_message = str(error)
    except Exception as error:
        traceback.print_exc()
        outcome = "error"
        returncode = ATTEMPT_ERROR_EXIT_CODE
        exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
        exception_message = str(error)
    result = TrainInfMismatchWorkerResult(
        outcome=outcome,
        artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
        comparison_completed=comparison_completed,
        exception_type=exception_type,
        exception_message=exception_message,
    )
    result_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-attempt-result", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(_run_workflow_attempt(args.workflow_attempt_result))
