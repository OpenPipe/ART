"""Integration test for TinkerNativeBackend based on yes-no-maybe."""

import os
import tempfile
import uuid

import openai
import pytest

import art
from art.tinker_native import TinkerNativeBackend

DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def get_base_model() -> str:
    return os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)


def ensure_reward_variance(groups) -> None:
    for group in groups:
        rewards = [t.reward for t in group]
        if len(rewards) < 2:
            continue
        if len(set(rewards)) <= 1:
            group.trajectories[0].reward = 1.0
            group.trajectories[1].reward = 0.0


async def simple_rollout(
    client: openai.AsyncOpenAI, model_name: str, prompt: str
) -> art.Trajectory:
    messages: art.Messages = [{"role": "user", "content": prompt}]
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=10,
        timeout=60,
        temperature=1,
    )
    choice = chat_completion.choices[0]
    content = (choice.message.content or "").lower()
    if "yes" in content:
        reward = 1.0
    elif "no" in content:
        reward = 0.5
    elif "maybe" in content:
        reward = 0.25
    else:
        reward = 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)  # type: ignore[attr-defined]


@pytest.mark.skipif(
    "TINKER_API_KEY" not in os.environ,
    reason="TINKER_API_KEY not set - skipping TinkerNativeBackend test",
)
async def test_tinker_native_backend():
    model_name = f"test-tinker-native-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = TinkerNativeBackend(path=tmpdir)
        model = art.TrainableModel(
            name=model_name,
            project="integration-tests",
            base_model=get_base_model(),
        )
        try:
            await model.register(backend)

            openai_client = model.openai_client()
            current_step = await model.get_step()
            model_name_step = model.get_inference_name(step=current_step)
            prompts = ["Say yes", "Say no", "Say maybe"]

            async def make_group(prompt: str) -> art.TrajectoryGroup:
                import asyncio

                trajectories = await asyncio.gather(
                    *[
                        simple_rollout(openai_client, model_name_step, prompt)
                        for _ in range(2)
                    ]
                )
                return art.TrajectoryGroup(trajectories)  # type: ignore[attr-defined]

            train_groups = await art.gather_trajectory_groups(  # type: ignore[attr-defined]
                [make_group(prompt) for prompt in prompts]
            )
            ensure_reward_variance(train_groups)

            result = await backend.train(
                model,
                train_groups,
                learning_rate=1e-5,
            )
            await model.log(
                train_groups, metrics=result.metrics, step=result.step, split="train"
            )

            assert result.step > current_step

            await openai_client.chat.completions.create(
                messages=[{"role": "user", "content": "Say hello"}],
                model=model.get_inference_name(step=result.step),
                max_tokens=10,
                timeout=30,
            )
            await openai_client.chat.completions.create(
                messages=[{"role": "user", "content": "Say hello"}],
                model=model.get_inference_name(step=0),
                max_tokens=10,
                timeout=30,
            )
        finally:
            await backend.close()
