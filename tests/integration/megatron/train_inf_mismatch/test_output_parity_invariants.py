from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from . import workflow_stage
from .output_parity import (
    TOP20_KL_CANDIDATE_TO_TARGET_LIMIT,
    TOP_K,
    EngineSide,
    ScoreBundle,
    TokenTopK,
    TrainInfOutputParityConfig,
    WeightState,
    aggregate_mean_abs_pct,
    build_logical_token_map,
    compare_rollout,
    compare_topk,
    config_from_env,
    fwd_mean_abs_pct_limit_for_model,
    top20_kl_candidate_to_target_limit_for_model,
)
from .real_path import (
    RealPathConfig,
    _delete_adapter_safetensors_on_pass,
    _real_path_rollout_mode,
    _real_path_rollout_weights_mode,
)


def test_logical_map_flattens_shared_prefix_branches() -> None:
    packed = {
        "tokens": torch.tensor([[10, 11, 12, 13, 14, 12, 15, 16]]),
        "group_ids": torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]]),
        "parent_ids": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]]),
    }

    logical_map = build_logical_token_map(packed)

    assert [prompt.token_ids for prompt in logical_map.prompts] == [
        [10, 11, 12, 13, 14],
        [10, 11, 12, 15, 16],
    ]
    assert [prompt.packed_prompt_length for prompt in logical_map.prompts] == [2, 2]
    assert [prompt.scored_token_start_index for prompt in logical_map.prompts] == [
        3,
        3,
    ]
    assert [token.token_id for token in logical_map.tokens] == [13, 14, 15, 16]
    assert [token.art_logit_index for token in logical_map.tokens] == [2, 3, 5, 6]
    assert [token.vllm_prompt_token_index for token in logical_map.tokens] == [
        3,
        4,
        3,
        4,
    ]


def test_aggregate_mean_abs_pct_uses_vllm_merge_formula() -> None:
    summary = aggregate_mean_abs_pct(
        candidate=torch.tensor([2.0, 4.0]),
        target=torch.tensor([1.0, 3.0]),
        sequence_ids=[0, 0],
    )

    assert summary.source_numel == 2
    assert summary.trimmed_numel == 0
    assert summary.mean_abs_pct == pytest.approx((2.0 / 4.0) * 100.0)


def test_aggregate_mean_abs_pct_does_not_trim_or_average_sequence_summaries() -> None:
    target = torch.ones(80)
    candidate = target.clone()
    candidate[0] = 101.0
    candidate[1] = 51.0
    candidate[2] = 26.0
    candidate[3] = 2.0

    summary = aggregate_mean_abs_pct(
        candidate=candidate,
        target=target,
        sequence_ids=[0] * 40 + [1] * 40,
    )

    assert summary.source_numel == 80
    assert summary.sequence_count == 2
    assert summary.trimmed_numel == 0
    assert summary.mean_abs_pct == pytest.approx((176.0 / 80.0) * 100.0)


def _score(
    values: list[float],
    *,
    side: EngineSide,
    state: WeightState,
) -> ScoreBundle:
    return ScoreBundle(
        side=side,
        weight_state=state,
        target_logprobs=values,
        topk=[
            TokenTopK(
                token_ids=list(range(TOP_K)),
                logprobs=[-float(index) for index in range(TOP_K)],
            )
            for _ in values
        ],
    )


def test_compare_rollout_reports_base_lora_and_delta_separately() -> None:
    packed = {
        "tokens": torch.tensor([[10, 11, 12, 13, 14]]),
        "group_ids": torch.tensor([[0, 0, 1, 1, 1]]),
        "parent_ids": torch.tensor([[0, 0, 0, 0, 0]]),
    }
    logical_map = build_logical_token_map(packed)

    report = compare_rollout(
        rollout_mode="native_lora",
        megatron_base=_score([-1.0, -2.0], side="megatron", state="base"),
        megatron_lora=_score([-1.5, -2.5], side="megatron", state="lora"),
        vllm_base=_score([-1.1, -2.2], side="vllm", state="base"),
        vllm_lora=_score([-1.7, -2.8], side="vllm", state="lora"),
        logical_map=logical_map,
    )

    assert report.base.mean_abs_pct > 0
    assert report.lora.mean_abs_pct > 0
    assert report.delta.mean_abs_pct > 0


def test_real_path_default_generates_16_tokens_per_rollout() -> None:
    assert RealPathConfig().max_completion_tokens == 16


def test_real_path_rollout_mode_follows_config() -> None:
    native_config = TrainInfOutputParityConfig(
        base_model="Qwen/Qwen3.5-35B-A3B",
    )
    merged_config = TrainInfOutputParityConfig(
        base_model="unvalidated/native-disabled",
        allow_unvalidated_arch=True,
    )

    assert _real_path_rollout_mode(native_config) == "native_lora"
    assert _real_path_rollout_weights_mode(native_config) == "lora"
    assert _real_path_rollout_mode(merged_config) == "merged"
    assert _real_path_rollout_weights_mode(merged_config) == "merged"


def test_real_path_deletes_only_adapter_safetensors_on_pass(tmp_path) -> None:
    run_dir = tmp_path / "run"
    active_lora = run_dir / "real_path_active_lora"
    checkpoint = run_dir / "art_path" / "models" / "m" / "checkpoints" / "0000"
    active_lora.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    for directory in (active_lora, checkpoint):
        (directory / "adapter_model.safetensors").write_bytes(b"adapter")
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    score_path = run_dir / "real_path_vllm_lora_scores.json"
    score_path.write_text("{}", encoding="utf-8")

    _delete_adapter_safetensors_on_pass(run_dir, passed=False)

    assert len(list(run_dir.rglob("adapter_model.safetensors"))) == 2

    _delete_adapter_safetensors_on_pass(run_dir, passed=True)

    assert list(run_dir.rglob("adapter_model.safetensors")) == []
    assert len(list(run_dir.rglob("adapter_config.json"))) == 2
    assert score_path.exists()


def test_architecture_specific_real_path_limits() -> None:
    assert fwd_mean_abs_pct_limit_for_model("Qwen/Qwen3-30B-A3B") == 7.0
    assert fwd_mean_abs_pct_limit_for_model("Qwen/Qwen3.5-35B-A3B") == 5.0
    assert TOP20_KL_CANDIDATE_TO_TARGET_LIMIT == 0.002


def test_gemma4_real_path_limits() -> None:
    assert (
        fwd_mean_abs_pct_limit_for_model(
            "google/gemma-4-31B-it",
            allow_unvalidated_arch=True,
        )
        == 8.0
    )
    assert (
        top20_kl_candidate_to_target_limit_for_model(
            "google/gemma-4-31B-it",
            allow_unvalidated_arch=True,
        )
        == 0.003
    )
    assert (
        fwd_mean_abs_pct_limit_for_model(
            "google/gemma-4-26B-A4B-it",
            allow_unvalidated_arch=True,
        )
        == 8.0
    )
    assert (
        top20_kl_candidate_to_target_limit_for_model(
            "google/gemma-4-26B-A4B-it",
            allow_unvalidated_arch=True,
        )
        == 0.008
    )
    assert TOP20_KL_CANDIDATE_TO_TARGET_LIMIT == 0.002


def test_compare_topk_reports_restricted_intersection_kl() -> None:
    target = ScoreBundle(
        side="megatron",
        weight_state="base",
        target_logprobs=[0.0],
        topk=[
            TokenTopK(
                token_ids=[10, 11],
                logprobs=[math.log(0.75), math.log(0.25)],
            )
        ],
    )
    candidate = ScoreBundle(
        side="vllm",
        weight_state="base",
        target_logprobs=[0.0],
        topk=[
            TokenTopK(
                token_ids=[10, 11],
                logprobs=[math.log(0.5), math.log(0.5)],
            )
        ],
    )

    report = compare_topk(candidate, target)

    assert report.top20_intersection_kl_target_to_candidate == pytest.approx(
        0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    )
    assert report.top20_intersection_kl_candidate_to_target == pytest.approx(
        0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    )


def test_config_from_env_accepts_lora_target_module_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ART_TRAIN_INF_MISMATCH_LORA_TARGET_MODULES",
        "experts,in_proj_qkv,in_proj_z",
    )

    config = config_from_env()

    assert config.lora_target_modules == ["experts", "in_proj_qkv", "in_proj_z"]


def test_default_rollout_modes_follow_model_support_native_lora_status() -> None:
    assert TrainInfOutputParityConfig(
        base_model="Qwen/Qwen3.5-35B-A3B"
    ).rollout_modes == ["native_lora", "merged"]
    assert TrainInfOutputParityConfig(
        base_model="unvalidated/native-disabled",
        allow_unvalidated_arch=True,
    ).rollout_modes == ["merged"]


def test_config_from_env_rollout_modes_override_handler_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ART_TRAIN_INF_MISMATCH_BASE_MODEL",
        "unvalidated/native-disabled",
    )
    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_ALLOW_UNVALIDATED_ARCH", "1")
    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_ROLLOUT_MODES", "native_lora")

    config = config_from_env()

    assert config.rollout_modes == ["native_lora"]


def test_workflow_stage_enables_live_train_inf_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import subprocess

    captured_env = {}
    real_run = workflow_stage.subprocess.run

    def fake_run(*args, **kwargs):
        if "env" not in kwargs:
            return real_run(*args, **kwargs)
        captured_env.update(kwargs["env"])
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="1 passed\n",
            stderr="",
        )

    monkeypatch.setattr(workflow_stage, "create_artifact_dir", lambda _nodeid: tmp_path)
    monkeypatch.setattr(workflow_stage.subprocess, "run", fake_run)

    report = workflow_stage.run_train_inf_mismatch(
        base_model="Qwen/Qwen3.5-35B-A3B",
        allow_unvalidated_arch=True,
    )

    assert report.passed is True
    assert captured_env["ART_RUN_TRAIN_INF_MISMATCH_LIVE"] == "1"
    assert captured_env["ART_TRAIN_INF_MISMATCH_ALLOW_UNVALIDATED_ARCH"] == "1"
    assert captured_env["ART_REAL_PATH_MAX_COMPLETION_TOKENS"] == "16"
