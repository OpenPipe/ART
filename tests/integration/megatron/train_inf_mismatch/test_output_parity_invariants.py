from __future__ import annotations

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest

torch = pytest.importorskip("torch")

import art

from . import workflow_stage
from .output_parity import (
    TOP20_KL_CANDIDATE_TO_TARGET_LIMIT,
    TOP_K,
    EngineSide,
    LogicalPrompt,
    LogicalToken,
    LogicalTokenMap,
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
    _choice_score_index,
    _collect_real_trajectory_groups,
    _delete_adapter_safetensors_on_pass,
    _real_path_rollout_mode,
    _topk_from_chat_logprob,
    _vllm_scores_from_real_choices,
)


def test_choice_score_index_disambiguates_equal_completions_by_prompt() -> None:
    def choice(prompt_id: int) -> Choice:
        return Choice.model_validate(
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "same"},
                "prompt_token_ids": [prompt_id],
                "token_ids": [7],
            }
        )

    first, second = choice(1), choice(2)
    groups = [[SimpleNamespace(messages_and_choices=[first, second])]]

    indexed = _choice_score_index(groups, require_routing_metadata=False)

    assert indexed == {(1, 7): [first], (2, 7): [second]}


def test_equal_token_paths_match_packed_source_logprobs() -> None:
    def choice(score: float) -> Choice:
        return Choice.model_validate(
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "same"},
                "prompt_token_ids": [1],
                "token_ids": [7],
                "logprobs": {
                    "content": [
                        {
                            "token": "token_id:7",
                            "logprob": score,
                            "top_logprobs": [{"token": "token_id:7", "logprob": score}],
                        }
                    ]
                },
            }
        )

    first, second = choice(-1.0), choice(-2.0)
    groups = [[SimpleNamespace(messages_and_choices=[first, second])]]
    logical_map = LogicalTokenMap(
        prompts=[
            LogicalPrompt(
                prompt_id=0,
                sample_id=0,
                family_id=0,
                completion_id=0,
                packed_prompt_length=1,
                scored_token_start_index=1,
                token_ids=[1, 7],
            )
        ],
        tokens=[
            LogicalToken(
                token_id=7,
                sample_id=0,
                family_id=0,
                completion_id=0,
                prompt_id=0,
                art_packed_token_index=1,
                art_logit_index=0,
                vllm_prompt_token_index=1,
                source_logprob=-2.0,
            ),
            LogicalToken(
                token_id=7,
                sample_id=0,
                family_id=0,
                completion_id=1,
                prompt_id=0,
                art_packed_token_index=2,
                art_logit_index=1,
                vllm_prompt_token_index=1,
                source_logprob=-1.0,
            ),
        ],
    )

    scores = _vllm_scores_from_real_choices(
        trajectory_groups=groups,
        logical_map=logical_map,
        require_routing_metadata=False,
        weight_state="lora",
        rollout_mode="native_lora",
    )

    assert scores.target_logprobs == [-2.0, -1.0]


def _write_workflow_worker_result(
    command: list[str],
    result: workflow_stage.TrainInfMismatchWorkerResult,
) -> None:
    result_path = Path(command[command.index("--workflow-attempt-result") + 1])
    result_path.write_text(result.model_dump_json(), encoding="utf-8")


def test_logical_map_flattens_prefix_tree_branches() -> None:
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


def test_logical_map_handles_unscored_prompt_suffix_inside_leaf() -> None:
    packed = {
        "tokens": torch.tensor([[10, 11, 12, 13, 14, 15]]),
        "group_ids": torch.tensor([[0, 0, 1, 1, 1, 1]]),
        "parent_ids": torch.tensor([[0, 0, 0, 0, 0, 0]]),
        "assistant_mask": torch.tensor([[False, False, False, False, True, True]]),
    }

    logical_map = build_logical_token_map(packed)

    assert logical_map.prompts[0].token_ids == [10, 11, 12, 13, 14, 15]
    assert logical_map.prompts[0].scored_token_start_index == 4
    assert [token.vllm_prompt_token_index for token in logical_map.tokens] == [4, 5]


def test_logical_map_flattens_nested_prefix_tree_leaves() -> None:
    packed = {
        "tokens": torch.tensor(
            [[10, 11, 20, 30, 31, 32, 33, 34, 35, 40, 50, 51, 52, 60, 61, 62]]
        ),
        "group_ids": torch.tensor([[0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 6]]),
        "parent_ids": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 4, 4, 4, 0, 0, 0]]),
    }

    logical_map = build_logical_token_map(packed)

    assert [prompt.token_ids for prompt in logical_map.prompts] == [
        [10, 11, 20, 30, 31, 32],
        [10, 11, 20, 33, 34, 35],
        [10, 11, 40, 50, 51, 52],
        [10, 11, 60, 61, 62],
    ]
    assert [prompt.packed_prompt_length for prompt in logical_map.prompts] == [
        3,
        3,
        3,
        2,
    ]
    assert [token.token_id for token in logical_map.tokens] == [
        31,
        32,
        34,
        35,
        51,
        52,
        61,
        62,
    ]
    assert [token.art_logit_index for token in logical_map.tokens] == [
        3,
        4,
        6,
        7,
        10,
        11,
        13,
        14,
    ]
    assert [token.vllm_prompt_token_index for token in logical_map.tokens] == [
        4,
        5,
        4,
        5,
        4,
        5,
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


@pytest.mark.asyncio
async def test_real_path_rollouts_use_stable_unique_seeds_concurrently() -> None:
    calls = []
    active_requests = 0
    max_active_requests = 0

    async def create(**kwargs):
        nonlocal active_requests, max_active_requests
        calls.append(kwargs)
        active_requests += 1
        max_active_requests = max(max_active_requests, active_requests)
        await asyncio.sleep(0)
        active_requests -= 1
        return SimpleNamespace(
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="maybe"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    model = SimpleNamespace(
        openai_client=lambda: client,
        get_inference_name=lambda *, step=None: "fake",
    )
    config = RealPathConfig(
        output_parity=TrainInfOutputParityConfig(seed=41),
        rollouts_per_prompt=3,
    )

    groups = await _collect_real_trajectory_groups(
        model=model,
        config=config,
        prompts=["first", "second"],
        extra_body={"return_tokens_as_token_ids": True},
    )

    assert len(groups) == 2
    assert max_active_requests > 1
    assert sorted(call["seed"] for call in calls) == list(range(41, 47))
    assert all(
        call["extra_body"] == {"return_tokens_as_token_ids": True} for call in calls
    )


def test_real_path_topk_sorts_vllm_sampled_token_prefix() -> None:
    entry = SimpleNamespace(
        top_logprobs=[SimpleNamespace(token="token_id:999", logprob=-100.0)]
        + [
            SimpleNamespace(token=f"token_id:{token_id}", logprob=-float(token_id))
            for token_id in range(TOP_K)
        ]
    )

    topk = _topk_from_chat_logprob(entry)

    assert topk.token_ids == list(range(TOP_K))
    assert topk.logprobs == [-float(token_id) for token_id in range(TOP_K)]


def test_real_path_rollout_mode_follows_config() -> None:
    config = TrainInfOutputParityConfig(
        base_model="Qwen/Qwen3.5-35B-A3B",
    )

    assert _real_path_rollout_mode(config) == "native_lora"


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
    assert fwd_mean_abs_pct_limit_for_model("meta-llama/Llama-3.2-1B-Instruct") == 5.75
    assert fwd_mean_abs_pct_limit_for_model("Qwen/Qwen3-30B-A3B") == 8.0
    assert fwd_mean_abs_pct_limit_for_model("Qwen/Qwen3.5-27B") == 8.05
    assert top20_kl_candidate_to_target_limit_for_model("Qwen/Qwen3.5-27B") == 0.003
    assert fwd_mean_abs_pct_limit_for_model("Qwen/Qwen3.5-35B-A3B") == 8.0
    assert top20_kl_candidate_to_target_limit_for_model("Qwen/Qwen3.5-35B-A3B") == 0.005
    assert top20_kl_candidate_to_target_limit_for_model("openai/gpt-oss-20b") == 0.005
    assert TOP20_KL_CANDIDATE_TO_TARGET_LIMIT == 0.002


def test_gemma4_real_path_limits() -> None:
    assert (
        fwd_mean_abs_pct_limit_for_model(
            "google/gemma-4-31B-it",
            allow_unvalidated_arch=True,
        )
        == 15.0
    )
    assert (
        top20_kl_candidate_to_target_limit_for_model(
            "google/gemma-4-31B-it",
            allow_unvalidated_arch=True,
        )
        == 0.008
    )
    assert (
        fwd_mean_abs_pct_limit_for_model(
            "google/gemma-4-26B-A4B-it",
            allow_unvalidated_arch=True,
        )
        == 25.0
    )
    assert (
        top20_kl_candidate_to_target_limit_for_model(
            "google/gemma-4-26B-A4B-it",
            allow_unvalidated_arch=True,
        )
        == 0.012
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


def test_config_from_env_accepts_vllm_memory_utilization_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_VLLM_GPU_MEMORY_UTILIZATION", "0.5")

    config = config_from_env()

    assert config.engine_args["gpu_memory_utilization"] == 0.5


def test_config_from_env_accepts_gdn_prefill_backend_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_GDN_PREFILL_BACKEND", "triton")

    config = config_from_env()

    assert config.engine_args["additional_config"] == {"gdn_prefill_backend": "triton"}


def test_default_rollout_mode_is_native_lora() -> None:
    assert TrainInfOutputParityConfig(
        base_model="Qwen/Qwen3.5-35B-A3B"
    ).rollout_modes == ["native_lora"]
    assert TrainInfOutputParityConfig(
        base_model="unvalidated/native-disabled",
        allow_unvalidated_arch=True,
    ).rollout_modes == ["native_lora"]


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
    captured_command = []
    real_run = subprocess.run

    def fake_run(*args, **kwargs):
        if "env" not in kwargs:
            return real_run(*args, **kwargs)
        command = args[0]
        captured_command.extend(command)
        captured_env.update(kwargs["env"])
        _write_workflow_worker_result(
            command,
            workflow_stage.TrainInfMismatchWorkerResult(outcome="passed"),
        )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
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
    assert captured_env["ART_TRAIN_INF_MISMATCH_VLLM_GPU_MEMORY_UTILIZATION"] == "0.50"
    assert "pytest" not in captured_command


def test_workflow_stage_does_not_accept_a_skipped_live_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import subprocess

    real_run = subprocess.run
    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_ATTEMPTS", "1")
    monkeypatch.setattr(workflow_stage, "create_artifact_dir", lambda _nodeid: tmp_path)

    def fake_run(*args, **kwargs):
        if "env" not in kwargs:
            return real_run(*args, **kwargs)
        _write_workflow_worker_result(
            args[0],
            workflow_stage.TrainInfMismatchWorkerResult(outcome="skipped"),
        )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(workflow_stage.subprocess, "run", fake_run)

    report = workflow_stage.run_train_inf_mismatch(base_model="Qwen/Qwen3.5-35B-A3B")

    assert report.passed is False
    assert report.passed_count == 0
    assert report.skipped_count == 1


def test_workflow_stage_retries_numerical_mismatch_and_transient_startup_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import subprocess

    calls = 0
    real_run = subprocess.run

    def fake_run(*args, **kwargs):
        nonlocal calls
        if "env" not in kwargs:
            return real_run(*args, **kwargs)
        calls += 1
        _write_workflow_worker_result(
            args[0],
            workflow_stage.TrainInfMismatchWorkerResult(
                outcome="failed",
                comparison_completed=True,
                exception_type="builtins.AssertionError",
                exception_message="TimeoutError in completed numerical evidence",
            ),
        )
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr=""
        )

    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_ATTEMPTS", "3")
    monkeypatch.setattr(workflow_stage, "create_artifact_dir", lambda _nodeid: tmp_path)
    monkeypatch.setattr(workflow_stage.subprocess, "run", fake_run)

    report = workflow_stage.run_train_inf_mismatch(base_model="openai/gpt-oss-20b")

    assert calls == report.attempt_count == 3
    assert report.failed_count == 1
    assert all(attempt.retryable for attempt in report.attempts)
    assert workflow_stage._retryable_attempt_failure(
        returncode=2,
        result=workflow_stage.TrainInfMismatchWorkerResult(
            outcome="error",
            exception_type="builtins.TimeoutError",
        ),
        output="",
    )

    calls = 0

    def transient_then_pass(*args, **kwargs):
        nonlocal calls
        if "env" not in kwargs:
            return real_run(*args, **kwargs)
        calls += 1
        worker_result = workflow_stage.TrainInfMismatchWorkerResult(
            outcome="error" if calls == 1 else "passed",
            exception_type="builtins.TimeoutError" if calls == 1 else None,
        )
        _write_workflow_worker_result(args[0], worker_result)
        return subprocess.CompletedProcess(
            args=args,
            returncode=2 if calls == 1 else 0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(workflow_stage.subprocess, "run", transient_then_pass)
    report = workflow_stage.run_train_inf_mismatch(base_model="openai/gpt-oss-20b")

    assert report.passed is True
    assert calls == report.attempt_count == 2
    assert [attempt.retryable for attempt in report.attempts] == [True, False]
