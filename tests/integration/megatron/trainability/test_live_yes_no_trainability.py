import json
import os
from pathlib import Path

import pytest

from .yes_no_trainability import (
    run_yes_no_trainability_async,
    yes_no_trainability_passed,
)

torch = pytest.importorskip("torch")

DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LIVE_ENV = "ART_RUN_LIVE_YES_NO_TRAINABILITY"


def _require_opt_in() -> None:
    if os.environ.get(LIVE_ENV) != "1":
        pytest.skip(f"set {LIVE_ENV}=1 to run live yes/no trainability validation")


def _base_model() -> str:
    return os.environ.get(
        "ART_LIVE_YES_NO_BASE_MODEL",
        os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
    )


def _unsloth_base_model() -> str:
    return os.environ.get("ART_LIVE_UNSLOTH_YES_NO_BASE_MODEL", _base_model())


def _assert_passed(report) -> None:
    assert yes_no_trainability_passed(report)
    assert report.latest_step > 0
    assert report.step0_name in report.model_ids_before
    assert report.latest_name in report.model_ids_after
    if report.rollout_weights_mode == "merged":
        assert report.step0_name not in report.model_ids_after
    else:
        assert report.step0_name in report.model_ids_after
    assert report.latest_snapshot["has_logprobs"] is True


def _write_report(artifact_dir: Path, name: str, report) -> None:
    (artifact_dir / name).write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for live yes/no trainability validation",
)
@pytest.mark.asyncio
async def test_megatron_shared_yes_no_trainability_live(
    artifact_dir: Path,
) -> None:
    _require_opt_in()
    report = await run_yes_no_trainability_async(
        base_model=_base_model(),
        variant_name="megatron_shared",
        artifact_root=artifact_dir / "megatron_shared_workspace",
    )
    _write_report(artifact_dir, "megatron_shared_yes_no_trainability.json", report)
    _assert_passed(report)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for live yes/no trainability validation",
)
@pytest.mark.asyncio
async def test_megatron_dedicated_yes_no_trainability_live(
    artifact_dir: Path,
) -> None:
    _require_opt_in()
    report = await run_yes_no_trainability_async(
        base_model=_base_model(),
        variant_name="megatron_dedicated",
        artifact_root=artifact_dir / "megatron_dedicated_workspace",
    )
    _write_report(artifact_dir, "megatron_dedicated_yes_no_trainability.json", report)
    _assert_passed(report)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for live yes/no trainability validation",
)
@pytest.mark.asyncio
async def test_unsloth_dedicated_yes_no_trainability_live(
    artifact_dir: Path,
) -> None:
    _require_opt_in()
    report = await run_yes_no_trainability_async(
        base_model=_unsloth_base_model(),
        variant_name="unsloth_dedicated",
        artifact_root=artifact_dir / "unsloth_dedicated_workspace",
    )
    _write_report(artifact_dir, "unsloth_dedicated_yes_no_trainability.json", report)
    _assert_passed(report)
