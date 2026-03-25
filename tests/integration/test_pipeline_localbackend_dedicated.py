"""Dedicated LocalBackend smoke test for PipelineTrainer."""

import asyncio
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

import openai
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import art
from art.local import LocalBackend
from art.pipeline_trainer import PipelineTrainer

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.2
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_MAX_SEQ_LENGTH = 2048


def get_base_model() -> str:
    return os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)


def get_safe_gpu_memory_utilization() -> float:
    requested = float(
        os.environ.get(
            "ART_TEST_GPU_MEMORY_UTILIZATION",
            str(DEFAULT_GPU_MEMORY_UTILIZATION),
        )
    )
    min_free_gib = float(os.environ.get("ART_TEST_MIN_FREE_GPU_GIB", "8"))
    free_ratios: list[float] = []
    for device in (0, 1):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / (1024**3)
        if free_gib < min_free_gib:
            pytest.skip(
                "Insufficient free GPU memory for dedicated LocalBackend smoke test: "
                f"GPU {device} has {free_gib:.1f} GiB free < {min_free_gib:.1f} GiB required."
            )
        free_ratios.append(free_bytes / total_bytes)
    return max(0.02, min(requested, min(free_ratios) * 0.8))


def get_dedicated_vllm_test_config() -> art.dev.InternalModelConfig:
    return {
        "trainer_gpu_ids": [0],
        "inference_gpu_ids": [1],
        "engine_args": {
            "gpu_memory_utilization": get_safe_gpu_memory_utilization(),
            "max_model_len": int(
                os.environ.get("ART_TEST_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN))
            ),
            "max_num_seqs": 8,
            "enforce_eager": True,
        },
        "init_args": {
            "max_seq_length": int(
                os.environ.get("ART_TEST_MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH))
            ),
        },
    }


def reward_for_answer(text: str) -> float:
    content = text.lower()
    if "yes" in content:
        return 1.0
    if "no" in content:
        return 0.5
    if "maybe" in content:
        return 0.25
    return 0.0


async def assert_chat_logprobs(
    client: openai.AsyncOpenAI,
    model_name: str,
) -> None:
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello."}],
        model=model_name,
        max_tokens=8,
        timeout=60,
        logprobs=True,
        top_logprobs=0,
    )
    assert completion.choices[0].logprobs is not None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for dedicated LocalBackend PipelineTrainer test",
)
async def test_pipeline_trainer_local_backend_dedicated_smoke() -> None:
    model_name = f"test-pipeline-local-dedicated-{uuid.uuid4().hex[:8]}"
    prompts = [
        "Say yes",
        "Say no",
        "Say maybe",
        "Say hello",
        "Say yes again",
        "Say no again",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        async with LocalBackend(path=tmpdir) as backend:
            model = art.TrainableModel(
                name=model_name,
                project="integration-tests",
                base_model=get_base_model(),
                _internal_config=get_dedicated_vllm_test_config(),
            )

            async def scenario_iter():
                for prompt in prompts:
                    yield {"prompt": prompt}

            await model.register(backend)
            client = model.openai_client()
            try:

                async def rollout_fn(
                    rollout_model: art.TrainableModel,
                    scenario: dict[str, str],
                    _config: None,
                ) -> art.TrajectoryGroup:
                    await asyncio.sleep(0.2)
                    messages: art.Messages = [
                        {"role": "user", "content": scenario["prompt"]}
                    ]
                    completion = await client.chat.completions.create(
                        messages=messages,
                        model=rollout_model.get_inference_name(),
                        max_tokens=10,
                        timeout=60,
                        temperature=1,
                        n=2,
                        logprobs=True,
                        top_logprobs=0,
                    )
                    return art.TrajectoryGroup(
                        [
                            art.Trajectory(
                                messages_and_choices=[*messages, choice],
                                reward=reward_for_answer(choice.message.content or ""),
                            )
                            for choice in completion.choices
                        ]
                    )

                trainer = PipelineTrainer(
                    model=model,
                    backend=backend,
                    rollout_fn=rollout_fn,
                    scenarios=scenario_iter(),
                    config=None,
                    num_rollout_workers=2,
                    min_batch_size=1,
                    max_batch_size=1,
                    max_steps=2,
                    kl_penalty_coef=0.25,
                    kl_penalty_reference_step=0,
                    loss_fn="cispo",
                    eval_fn=None,
                )

                await trainer.train()

                latest_step = await model.get_step()
                assert latest_step >= 2

                await assert_chat_logprobs(client, model.get_inference_name(step=0))
                await assert_chat_logprobs(
                    client, model.get_inference_name(step=latest_step)
                )

                model_ids = [m.id async for m in client.models.list()]
                assert f"{model.name}@0" in model_ids
                assert f"{model.name}@{latest_step}" in model_ids

                history_path = (
                    Path(tmpdir)
                    / model.project
                    / "models"
                    / model.name
                    / "history.jsonl"
                )
                history_rows = [
                    json.loads(line) for line in history_path.read_text().splitlines()
                ]
                kl_values = [
                    row["loss/kl_policy_ref"]
                    for row in history_rows
                    if "loss/kl_policy_ref" in row
                ]
                assert kl_values
                assert all(math.isfinite(value) for value in kl_values)
            finally:
                await client.close()
