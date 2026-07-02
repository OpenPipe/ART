import asyncio
from typing import cast

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pytest

import art

from .yes_no_trainability import (
    _build_internal_config,
    _build_variant,
    _default_variant_name,
    _evaluate_groups,
    _TrainabilityVariant,
    _variant_init_args,
    _variant_max_steps,
    _variant_packed_sequence_length,
    _variant_rollouts_per_prompt,
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


@pytest.mark.asyncio
async def test_eval_prompts_are_submitted_concurrently() -> None:
    completions = _ConcurrentCompletions(expected=3)

    groups = await _evaluate_groups(
        cast(art.TrainableModel, _FakeModel(_FakeClient(completions))),
        base_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        prompts=["a", "b", "c"],
        step=1,
    )

    assert len(groups) == 3
    assert completions.started == 3
    assert completions.max_active == 3
    assert [group.trajectories[0].reward for group in groups] == [1.0, 1.0, 1.0]


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


def test_validated_dense_model_uses_dense_shared_topology(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ART_MODEL_SUPPORT_SHARED_GPU_IDS", "0,1")
    built_variant = _build_variant(
        "megatron_shared",
        base_model="Qwen/Qwen3.5-4B",
    )
    assert built_variant.topology is not None
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
