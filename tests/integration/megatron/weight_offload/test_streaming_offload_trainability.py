import json
import os
from pathlib import Path

import pytest

from ..trainability.yes_no_trainability import run_yes_no_trainability_async

torch = pytest.importorskip("torch")

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"
LIVE_ENV = "ART_RUN_LIVE_YES_NO_TRAINABILITY"


def _require_opt_in() -> None:
    if os.environ.get(LIVE_ENV) != "1":
        pytest.skip(f"set {LIVE_ENV}=1 to run live yes/no trainability validation")


def _base_model() -> str:
    return os.environ.get(
        "ART_LIVE_YES_NO_BASE_MODEL",
        os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
    )


def _assert_passed(report) -> None:
    assert report.saturated_step is not None
    assert report.saturated_step > 0
    assert report.initial_eval_reward < report.reward_threshold
    assert report.final_eval_reward is not None
    assert report.final_eval_reward >= report.reward_threshold
    assert report.final_eval_reward > report.initial_eval_reward
    assert report.latest_step > 0
    assert report.step0_name in report.model_ids_before
    assert report.latest_name in report.model_ids_after
    assert report.step0_name in report.model_ids_after
    assert report.latest_snapshot["has_logprobs"] is True


def _write_report(artifact_dir: Path, name: str, report) -> None:
    (artifact_dir / name).write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for live streaming offload trainability",
)
@pytest.mark.asyncio
async def test_megatron_dedicated_streaming_offload_yes_no_trainability_live(
    artifact_dir: Path,
) -> None:
    _require_opt_in()
    report = await run_yes_no_trainability_async(
        base_model=_base_model(),
        variant_name="megatron_dedicated",
        artifact_root=artifact_dir / "megatron_dedicated_streaming_offload_workspace",
        extra_env={
            "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD": "1",
            "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_NUM_LAYERS": "8",
            "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_RESIDENT_LAYERS": "2",
            "ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD_NUM_SLOTS": "4",
        },
    )
    _write_report(
        artifact_dir,
        "megatron_dedicated_streaming_offload_yes_no_trainability.json",
        report,
    )
    _assert_passed(report)
