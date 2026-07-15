from __future__ import annotations

from pathlib import Path

import pytest

from art.megatron.model_support.registry import get_model_support_spec

from .output_parity import model_support_is_moe
from .real_path import (
    config_from_env,
    run_real_path_train_inf_mismatch,
)

torch = pytest.importorskip("torch")


def _require_visible_gpus(gpu_ids: list[int]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real-path train/inf mismatch")
    visible_count = int(torch.cuda.device_count())
    required = max(gpu_ids) + 1 if gpu_ids else 0
    if visible_count < required:
        pytest.skip(
            f"Need visible CUDA device ids through {required - 1}, "
            f"but torch sees {visible_count} devices"
        )


@pytest.mark.asyncio
async def test_real_path_train_inf_mismatch_live(artifact_dir: Path) -> None:
    config = config_from_env()
    parity_config = config.output_parity
    _require_visible_gpus(
        parity_config.trainer_gpu_ids + parity_config.inference_gpu_ids
    )

    report = await run_real_path_train_inf_mismatch(
        config=config,
        artifact_dir=artifact_dir,
    )

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
