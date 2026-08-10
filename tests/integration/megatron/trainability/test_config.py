import asyncio
from types import SimpleNamespace
from typing import cast

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest
import torch

import art

from .test_live_length_trainability import (
    MOE_DEDICATED_TRAINING_TOPOLOGY,
    LengthSampleReport,
    LengthTrainabilityReport,
    _default_learning_rate,
    _length_current_step_demand,
    _length_max_steps,
    _length_rollout_seed,
    _length_rollout_temperature,
    _length_rollouts_per_prompt,
    _length_trainability_thresholds,
    _prompt_for_index,
    _target_tokens,
    _use_default_moe_dedicated_placement,
    length_trainability_passed,
)
from .test_live_length_trainability import (
    _extra_body as _length_extra_body,
)
from .test_live_length_trainability import (
    _prompt_tree_shape as _length_prompt_tree_shape,
)
from .yes_no_trainability import (
    TrainabilityStepReport,
    YesNoTrainabilityReport,
    _build_internal_config,
    _build_variant,
    _default_variant_name,
    _engine_args_for_yes_no_trainability,
    _evaluate_groups,
    _get_env_int_list,
    _rescore_groups,
    _select_answer_target,
    _TrainabilityVariant,
    _variant_init_args,
    _variant_max_steps,
    _variant_packed_sequence_length,
    _variant_rollouts_per_prompt,
    _variant_train_kwargs,
    build_prompts,
    reward_for_answer,
    yes_no_trainability_passed,
)
from .yes_no_trainability import (
    _extra_body as _yes_no_extra_body,
)
from .yes_no_trainability import (
    _prompt_tree_shape as _yes_no_prompt_tree_shape,
)


class _ConcurrentCompletions:
    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.started = 0
        self.active = 0
        self.max_active = 0
        self.all_started = asyncio.Event()

    async def create(self, **kwargs):
        self.started += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.started == self.expected:
            self.all_started.set()
        try:
            await asyncio.wait_for(self.all_started.wait(), timeout=1.0)
            return ChatCompletion(
                id=f"completion-{self.started}",
                choices=[
                    Choice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content="maybe",
                        ),
                    )
                ],
                created=0,
                model=str(kwargs["model"]),
                object="chat.completion",
            )
        finally:
            self.active -= 1


class _FakeChat:
    def __init__(self, completions: _ConcurrentCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _ConcurrentCompletions) -> None:
        self.chat = _FakeChat(completions)


class _FakeModel:
    def __init__(self, client: _FakeClient) -> None:
        self.client = client

    def openai_client(self) -> _FakeClient:
        return self.client

    def get_inference_name(self, *, step: int | None = None) -> str:
        return f"fake@{step}"


def test_optional_sampling_controls(monkeypatch) -> None:
    monkeypatch.setenv("ART_MODEL_SUPPORT_YES_NO_ALLOWED_TOKEN_IDS", "9829,902,36569")
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_ALLOWED_TOKEN_IDS", "154820,38069")
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_MIN_TOKENS", "2")
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_FREQUENCY_PENALTY", "0.5")

    assert _yes_no_extra_body("Qwen/Qwen3.5-35B-A3B")["allowed_token_ids"] == [
        9829,
        902,
        36569,
    ]
    assert _length_extra_body({}) == {
        "allowed_token_ids": [154820, 38069],
        "min_tokens": 2,
        "frequency_penalty": 0.5,
    }
    assert _length_extra_body({}, seed=1234)["seed"] == 1234


def test_yes_no_engine_args_use_model_context_budget() -> None:
    assert (
        _engine_args_for_yes_no_trainability(
            base_model="Qwen/Qwen3.5-35B-A3B", inference_gpu_ids=[0]
        )["max_model_len"]
        == 128
    )
    assert (
        _engine_args_for_yes_no_trainability(
            base_model="openai/gpt-oss-20b", inference_gpu_ids=[0]
        )["max_model_len"]
        == 512
    )


def test_integer_list_rejects_empty_values(monkeypatch) -> None:
    monkeypatch.setenv("INVALID_INTEGER_LIST", "1,,2")
    with pytest.raises(ValueError, match="Invalid integer list"):
        _get_env_int_list("INVALID_INTEGER_LIST")


def _answer_group(answers: list[str]) -> art.TrajectoryGroup:
    return art.TrajectoryGroup(
        [
            art.Trajectory(
                messages_and_choices=[
                    Choice(
                        finish_reason="stop",
                        index=index,
                        message=ChatCompletionMessage(role="assistant", content=answer),
                    )
                ],
                reward=-1.0,
            )
            for index, answer in enumerate(answers)
        ]
    )


def test_answer_target_requires_support_and_prefers_least_common() -> None:
    assert _select_answer_target([_answer_group(["maybe", "maybe", "yes"])]) == "yes"
    assert _select_answer_target([_answer_group(["yes", "no"])]) == "yes"
    assert _select_answer_target([_answer_group(["maybe", "maybe"])]) is None


def test_reward_for_answer_preserves_default_and_supports_target() -> None:
    answers = ["yes", "No.", "MAYBE!", "invalid"]

    assert [reward_for_answer(answer) for answer in answers] == [0.5, 0.75, 1.0, 0.0]
    assert [reward_for_answer(answer, target="no") for answer in answers] == [
        0.0,
        1.0,
        0.0,
        0.0,
    ]


def test_initial_groups_can_be_rescored_for_selected_target() -> None:
    group = _answer_group(["yes", "no", "maybe", "invalid"])

    _rescore_groups([group], target="no")

    assert [trajectory.reward for trajectory in group.trajectories] == [
        0.0,
        1.0,
        0.0,
        0.0,
    ]


@pytest.mark.asyncio
async def test_eval_prompts_are_submitted_concurrently_with_target_reward() -> None:
    completions = _ConcurrentCompletions(expected=3)

    groups = await _evaluate_groups(
        cast(art.TrainableModel, _FakeModel(_FakeClient(completions))),
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        prompts=["a", "b", "c"],
        step=1,
        target="yes",
    )

    assert len(groups) == 3
    assert completions.started == 3
    assert completions.max_active == 3
    assert [group.trajectories[0].reward for group in groups] == [0.0, 0.0, 0.0]


def test_megatron_variants_keep_short_packed_sequence_default(monkeypatch) -> None:
    monkeypatch.delenv("ART_MODEL_SUPPORT_YES_NO_PACKED_SEQUENCE_LENGTH", raising=False)
    variant = _TrainabilityVariant(
        name="megatron_shared",
        backend_name="megatron",
        placement_mode="shared",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[0, 1],
    )

    assert _variant_packed_sequence_length(variant) == 1024
    assert _variant_train_kwargs(variant) == {}
    config = _build_internal_config(
        variant, base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"
    )
    assert config["init_args"]["max_seq_length"] == 1024
    assert config["rollout_weights_mode"] == "lora"
    assert (
        _default_variant_name("Qwen/Qwen3-30B-A3B-Instruct-2507") == "megatron_shared"
    )
    assert _variant_rollouts_per_prompt(variant) == 4
    assert _variant_max_steps(variant) == 4


def test_unsloth_variant_uses_chunk_aligned_training_length(monkeypatch) -> None:
    monkeypatch.delenv("ART_MODEL_SUPPORT_YES_NO_PACKED_SEQUENCE_LENGTH", raising=False)
    variant = _TrainabilityVariant(
        name="unsloth_dedicated",
        backend_name="local",
        placement_mode="dedicated",
        trainer_gpu_ids=[0],
        inference_gpu_ids=[1],
    )

    assert _variant_packed_sequence_length(variant) == 1024
    assert _variant_train_kwargs(variant) == {"packed_sequence_length": 1024}
    assert _variant_init_args(variant) == {"max_seq_length": 1024}
    assert _build_internal_config(
        variant, base_model="Qwen/Qwen3-30B-A3B-Instruct-2507"
    )["init_args"] == {"max_seq_length": 1024}
    assert _variant_rollouts_per_prompt(variant) == 8
    assert _variant_max_steps(variant) == 12


def test_qwen3_5_defaults_to_shared_lora_rollout() -> None:
    variant = _TrainabilityVariant(
        name="megatron_shared",
        backend_name="megatron",
        placement_mode="shared",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[0, 1],
    )

    config = _build_internal_config(variant, base_model="Qwen/Qwen3.5-35B-A3B")

    assert _default_variant_name("Qwen/Qwen3.5-35B-A3B") == "megatron_shared"
    assert config["rollout_weights_mode"] == "lora"
    assert "trainer_gpu_ids" not in config
    assert "inference_gpu_ids" not in config


def test_dense_yes_no_default_uses_dedicated_placement(monkeypatch) -> None:
    monkeypatch.delenv("ART_MODEL_SUPPORT_YES_NO_VARIANT", raising=False)

    assert _default_variant_name("Qwen/Qwen3-32B") == "megatron_dedicated"


def test_yes_no_default_variant_env_override(monkeypatch) -> None:
    monkeypatch.setenv("ART_MODEL_SUPPORT_YES_NO_VARIANT", "megatron_shared")

    assert _default_variant_name("Qwen/Qwen3-32B") == "megatron_shared"


def test_yes_no_trainability_rejects_initially_saturated_stable_report() -> None:
    report = YesNoTrainabilityReport(
        variant="megatron_shared",
        backend_name="megatron",
        placement_mode="shared",
        base_model="google/gemma-4-31B-it",
        target_answer="maybe",
        output_dir="/tmp/report",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[0, 1],
        rollout_weights_mode="lora",
        reward_threshold=0.9,
        max_steps=4,
        prompt_count=8,
        eval_prompt_count=8,
        rollouts_per_prompt=4,
        latest_step=1,
        initial_eval_reward=0.9375,
        final_eval_reward=0.9375,
        saturated_step=1,
        step0_name="model@0",
        latest_name="model@1",
        steps=[
            TrainabilityStepReport(
                step=1,
                eval_reward=0.9375,
                train_reward=0.875,
                train_metrics={"loss/grad_norm": 54.0},
            )
        ],
    )

    assert yes_no_trainability_passed(report) is False


def test_yes_no_trainability_requires_gradient_correlation_and_learning() -> None:
    report = YesNoTrainabilityReport(
        variant="megatron_dedicated",
        backend_name="megatron",
        placement_mode="dedicated",
        base_model="deepseek-ai/DeepSeek-V4-Flash",
        target_answer="yes",
        output_dir="/tmp/report",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[2, 3],
        rollout_weights_mode="lora",
        reward_threshold=0.9,
        max_steps=4,
        prompt_count=8,
        eval_prompt_count=8,
        rollouts_per_prompt=4,
        latest_step=1,
        initial_eval_reward=0.5,
        final_eval_reward=1.0,
        saturated_step=1,
        step0_name="model@0",
        latest_name="model@1",
        steps=[
            TrainabilityStepReport(
                step=1,
                eval_reward=1.0,
                train_reward=0.75,
                train_metrics={
                    "loss/grad_norm": 1.0,
                    "loss/probs_corr": 0.9,
                },
            )
        ],
    )

    assert yes_no_trainability_passed(report) is True
    assert (
        yes_no_trainability_passed(
            report.model_copy(
                update={
                    "steps": [
                        report.steps[0].model_copy(
                            update={"train_metrics": {"loss/probs_corr": 0.9}}
                        )
                    ]
                }
            )
        )
        is False
    )
    assert (
        yes_no_trainability_passed(
            report.model_copy(
                update={
                    "steps": [
                        report.steps[0].model_copy(
                            update={"train_metrics": {"loss/grad_norm": 1.0}}
                        )
                    ]
                }
            )
        )
        is False
    )


def test_yes_no_prompts_form_prefix_tree_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ART_MODEL_SUPPORT_YES_NO_PROMPT", raising=False)
    monkeypatch.setenv("ART_MODEL_SUPPORT_YES_NO_PROMPT_COUNT", "8")

    prompts = build_prompts()

    assert _yes_no_prompt_tree_shape(prompts) == (3, 6)


def test_qwen3_5_length_trainability_uses_stable_moe_defaults() -> None:
    assert _default_learning_rate("Qwen/Qwen3.5-35B-A3B") == 1e-4
    assert _length_rollouts_per_prompt("Qwen/Qwen3.5-35B-A3B") == 32
    assert _length_max_steps("Qwen/Qwen3.5-35B-A3B") == 30
    assert _length_max_steps("meta-llama/Llama-3.2-1B-Instruct") == 30
    assert _length_rollout_seed("Qwen/Qwen3.5-35B-A3B") == 20261833
    assert _length_rollout_temperature("Qwen/Qwen3.5-35B-A3B") == 0.8
    assert _length_current_step_demand("Qwen/Qwen3.5-35B-A3B") is True
    assert _default_learning_rate("Qwen/Qwen3-30B-A3B-Instruct-2507") == 1e-4
    assert _length_rollouts_per_prompt("Qwen/Qwen3-30B-A3B-Instruct-2507") == 4
    assert _length_max_steps("Qwen/Qwen3-30B-A3B-Instruct-2507") == 20
    assert _length_rollout_seed("Qwen/Qwen3-30B-A3B-Instruct-2507") is None
    assert _length_rollout_temperature("Qwen/Qwen3-30B-A3B-Instruct-2507") == 1.1
    assert _length_current_step_demand("Qwen/Qwen3-30B-A3B-Instruct-2507") is False
    assert _length_current_step_demand("openai/gpt-oss-20b") is True


def test_length_trainability_environment_overrides_model_defaults(monkeypatch) -> None:
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_MAX_STEPS", "9")
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_ROLLOUTS_PER_PROMPT", "6")
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_ROLLOUT_SEED", "17")
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_ROLLOUT_TEMPERATURE", "0.7")
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_CURRENT_STEP_DEMAND", "0")

    assert _length_max_steps("Qwen/Qwen3.5-35B-A3B") == 9
    assert _length_rollouts_per_prompt("Qwen/Qwen3.5-35B-A3B") == 6
    assert _length_rollout_seed("Qwen/Qwen3.5-35B-A3B") == 17
    assert _length_rollout_seed("Qwen/Qwen3-30B-A3B-Instruct-2507") == 17
    assert _length_rollout_temperature("Qwen/Qwen3.5-35B-A3B") == 0.7
    assert _length_current_step_demand("Qwen/Qwen3.5-35B-A3B") is False


def test_gpt_oss_length_target_accounts_for_harmony_tokens(monkeypatch) -> None:
    assert _target_tokens("google/gemma-4-31B-it") == 22
    assert _target_tokens("openai/gpt-oss-20b") == 20
    assert _target_tokens("Qwen/Qwen3.5-35B-A3B") == 10
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_TARGET_TOKENS", "24")
    assert _target_tokens("openai/gpt-oss-20b") == 24


def test_length_prompts_form_prefix_tree_by_default() -> None:
    prompts = [_prompt_for_index(index)[0] for index in range(4)]

    assert _length_prompt_tree_shape(prompts) == (3, 6)


def test_length_trainability_accepts_near_baseline_learning_signal() -> None:
    report = LengthTrainabilityReport(
        base_model="google/gemma-4-31B-it",
        max_steps=10,
        max_steps_off_policy=0,
        latest_step=3,
        variant_name="megatron_dedicated",
        trainer_gpu_ids=[0],
        inference_gpu_ids=[1],
        training_topology={"tp": 1, "cp": 1, "ep": 1, "etp": 1, "dp": 1, "sp": False},
        rollout_weights_mode="lora",
        rollouts_per_prompt=4,
        normalize_advantages=True,
        summary_log_path="/tmp/length_trainability.log",
        latest_summary_log_path="/tmp/latest_length_trainability.log",
        thresholds=_length_trainability_thresholds("google/gemma-4-31B-it"),
        initial_train_abs_error=5.5,
        best_train_abs_error=0.5,
        success_step=3,
        final_train_reward=-0.05,
        final_train_abs_error=0.5,
        model_ids_after=["length@0", "length@3"],
        samples=[
            LengthSampleReport(
                split="train",
                step=0,
                scenario_index=0,
                target_step=0,
                target_tokens=10,
                max_tokens=142,
                prompt_word_count=300,
                generated_tokens=16,
                abs_error=6,
                reward=-0.6,
                text="a short answer",
            ),
            LengthSampleReport(
                split="train",
                step=0,
                scenario_index=1,
                target_step=0,
                target_tokens=10,
                max_tokens=142,
                prompt_word_count=300,
                generated_tokens=5,
                abs_error=5,
                reward=-0.5,
                text="brief",
            ),
            LengthSampleReport(
                split="train",
                step=3,
                scenario_index=2,
                target_step=3,
                target_tokens=10,
                max_tokens=142,
                prompt_word_count=300,
                generated_tokens=10,
                abs_error=0,
                reward=0.0,
                text="a target length answer",
            ),
            LengthSampleReport(
                split="train",
                step=3,
                scenario_index=3,
                target_step=3,
                target_tokens=10,
                max_tokens=142,
                prompt_word_count=300,
                generated_tokens=11,
                abs_error=1,
                reward=-0.1,
                text="a slightly long answer",
            ),
        ],
    )

    assert length_trainability_passed(report) is True


def test_validated_dense_model_uses_dense_shared_topology(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ART_MODEL_SUPPORT_SHARED_GPU_IDS", "0,1")
    built_variant = _build_variant(
        "megatron_shared",
        base_model="Qwen/Qwen3.5-4B",
    )
    assert built_variant.topology is not None
    assert built_variant.topology.tp == 1
    assert built_variant.topology.cp == 2
    assert built_variant.topology.ep == 1
    assert built_variant.topology.etp == 1

    variant = _TrainabilityVariant(
        name="megatron_shared",
        backend_name="megatron",
        placement_mode="shared",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[0, 1],
    )

    config = _build_internal_config(variant, base_model="Qwen/Qwen3.5-4B")
    assert config["rollout_weights_mode"] == "lora"
    assert config["engine_args"]["enable_sleep_mode"] is True
    assert "enable_expert_parallel" not in config["engine_args"]


def test_qwen3_5_moe_shared_variant_enables_expert_parallel(monkeypatch) -> None:
    monkeypatch.setenv("ART_MODEL_SUPPORT_SHARED_GPU_IDS", "0,1")
    variant = _TrainabilityVariant(
        name="megatron_shared",
        backend_name="megatron",
        placement_mode="shared",
        trainer_gpu_ids=[0, 1],
        inference_gpu_ids=[0, 1],
    )

    config = _build_internal_config(variant, base_model="Qwen/Qwen3.5-35B-A3B")

    assert config["rollout_weights_mode"] == "lora"
    assert config["engine_args"]["enable_expert_parallel"] is True


def test_dsv4_trainability_uses_large_model_dedicated_resources(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=284 * 1024**3),
    )
    monkeypatch.setattr(
        "tests.integration.megatron.trainability.yes_no_trainability."
        "_safe_gpu_memory_utilization",
        lambda device_ids: 0.5,
    )
    monkeypatch.setenv("ART_MODEL_SUPPORT_EXTERNAL_VLLM_URL", "http://127.0.0.1:8000")
    default_variant = _default_variant_name(
        "deepseek-ai/DeepSeek-V4-Flash",
    )
    variant = _build_variant(
        default_variant,
        base_model="deepseek-ai/DeepSeek-V4-Flash",
    )
    config = _build_internal_config(
        variant,
        base_model="deepseek-ai/DeepSeek-V4-Flash",
    )

    assert default_variant == "megatron_dedicated"
    assert variant.topology is not None
    assert variant.topology.tp == 2
    assert variant.topology.ep == 2
    assert variant.topology.cp == 1
    assert variant.topology.dp == 1
    assert variant.topology.sp is True
    assert variant.trainer_gpu_ids == [0, 1]
    assert variant.inference_gpu_ids == [2, 3]
    assert config["engine_args"]["tensor_parallel_size"] == 2
    assert config["engine_args"]["enable_expert_parallel"] is True
    assert config["engine_args"]["kv_cache_dtype"] == "fp8"
    assert config["engine_args"].get("moe_backend") == "triton"
    assert "megatron_topology" not in config
    assert config["vllm_runtime"] == {
        "mode": "external",
        "server_url": "http://127.0.0.1:8000",
        "api_key": "art-external-vllm",
    }


def test_dsv4_length_trainability_keeps_handler_resources(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=140 * 1024**3),
    )

    variant = _build_variant(
        "megatron_dedicated",
        base_model="deepseek-ai/DeepSeek-V4-Flash",
        resource_stage_name="length_trainability",
    )
    _use_default_moe_dedicated_placement(
        variant,
        base_model="deepseek-ai/DeepSeek-V4-Flash",
    )

    assert variant.topology is not None
    assert variant.trainer_gpu_ids == [0, 1, 2, 3, 4, 5, 6, 7]
    assert variant.inference_gpu_ids == [4, 5, 6, 7]
    assert variant.topology.tp == 2
    assert variant.topology.ep == 8
    assert variant.topology.cp == 1


def test_explicit_length_gpu_placement_keeps_default_moe_topology(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ART_MODEL_SUPPORT_TRAINER_GPU_IDS", "3,4")
    monkeypatch.setenv("ART_MODEL_SUPPORT_INFERENCE_GPU_IDS", "6")
    variant = _build_variant(
        "megatron_dedicated",
        base_model="openai/gpt-oss-20b",
        resource_stage_name="length_trainability",
    )

    _use_default_moe_dedicated_placement(
        variant,
        base_model="openai/gpt-oss-20b",
    )

    assert variant.trainer_gpu_ids == [3, 4]
    assert variant.inference_gpu_ids == [6]
    assert variant.topology is not None
    assert variant.topology.model_dump() == MOE_DEDICATED_TRAINING_TOPOLOGY.model_dump()
