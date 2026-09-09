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
    _max_tokens,
    _render_chat_messages,
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


def test_qwen3_5_length_trainability_uses_stable_moe_defaults() -> None:
    assert _default_learning_rate("Qwen/Qwen3.5-35B-A3B") == 1e-4
    assert _length_rollouts_per_prompt("Qwen/Qwen3.5-35B-A3B") == 32
    assert _length_max_steps("Qwen/Qwen3.5-35B-A3B") == 40
    assert _length_max_steps("meta-llama/Llama-3.2-1B-Instruct") == 30
    assert _length_max_steps("openai/gpt-oss-20b") == 30
    assert _length_rollout_seed("Qwen/Qwen3.5-35B-A3B") == 20261833
    assert _length_rollout_temperature("Qwen/Qwen3.5-35B-A3B") == 0.8
    assert _length_current_step_demand("Qwen/Qwen3.5-35B-A3B") is True
    assert _default_learning_rate("Qwen/Qwen3-30B-A3B-Instruct-2507") == 1e-4
    assert _length_rollouts_per_prompt("Qwen/Qwen3-30B-A3B-Instruct-2507") == 4
    assert _length_max_steps("Qwen/Qwen3-30B-A3B-Instruct-2507") == 20
    assert _length_rollout_seed("Qwen/Qwen3-30B-A3B-Instruct-2507") is None
    assert _length_rollout_temperature("Qwen/Qwen3-30B-A3B-Instruct-2507") == 1.1
    assert _length_current_step_demand("Qwen/Qwen3-30B-A3B-Instruct-2507") is False
    assert _length_rollout_seed("openai/gpt-oss-20b") == 20261833
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
    assert _target_tokens("zai-org/GLM-5.2") == 12
    monkeypatch.setenv("ART_MODEL_SUPPORT_LENGTH_TARGET_TOKENS", "24")
    assert _target_tokens("openai/gpt-oss-20b") == 24


def test_length_trainability_default_success_threshold() -> None:
    thresholds = _length_trainability_thresholds("zai-org/GLM-5.2")

    assert thresholds.success_abs_error_max == 2


def test_length_prompts_form_prefix_tree_by_default() -> None:
    prompts = [_prompt_for_index(index)[0] for index in range(4)]

    assert _length_prompt_tree_shape(prompts) == (3, 6)


def test_glm52_length_prompt_requests_a_fuller_initial_answer() -> None:
    default_prompt = _prompt_for_index(0)[0]
    glm52_prompt = _prompt_for_index(0, base_model="zai-org/GLM-5.2")[0]

    assert "Use one sentence." in default_prompt
    assert (
        "Use two complete sentences with one concrete detail in each." in glm52_prompt
    )


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

    def unexpected_memory_probe(_device_ids) -> float:
        raise AssertionError("external vLLM must not probe resident inference memory")

    monkeypatch.setattr(
        "tests.integration.megatron.trainability.yes_no_trainability."
        "_safe_gpu_memory_utilization",
        unexpected_memory_probe,
    )
    monkeypatch.setenv("ART_MODEL_SUPPORT_EXTERNAL_VLLM_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("ART_MODEL_SUPPORT_EXTERNAL_VLLM_HEALTH_TIMEOUT", "1200")
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
    assert variant.topology.ep == 4
    assert variant.topology.cp == 1
    assert variant.topology.dp == 2
    assert variant.topology.sp is True
    assert variant.trainer_gpu_ids == [0, 1, 2, 3]
    assert variant.inference_gpu_ids == [2, 3]
    assert config["engine_args"]["tensor_parallel_size"] == 2
    assert config["engine_args"]["enable_expert_parallel"] is True
    assert config["engine_args"]["kv_cache_dtype"] == "fp8"
    assert config["engine_args"].get("moe_backend") == "auto"
    assert "megatron_topology" not in config
    assert config["vllm_runtime"] == {
        "mode": "external",
        "server_url": "http://127.0.0.1:8000",
        "api_key": "art-external-vllm",
        "health_timeout_s": 1200.0,
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
