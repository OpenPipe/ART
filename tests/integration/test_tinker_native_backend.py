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


@pytest.mark.skipif(
    "TINKER_API_KEY" not in os.environ,
    reason="TINKER_API_KEY not set - skipping TinkerNativeBackend fork test",
)
async def test_tinker_native_fork_checkpoint():
    """Train model A for 1 step with save_checkpoint, fork to model B, train model B."""
    run_id = uuid.uuid4().hex[:8]
    model_a_name = f"test-fork-src-{run_id}"
    model_b_name = f"test-fork-dst-{run_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = TinkerNativeBackend(path=tmpdir)
        model_a = art.TrainableModel(
            name=model_a_name,
            project="integration-tests",
            base_model=get_base_model(),
        )
        model_b = art.TrainableModel(
            name=model_b_name,
            project="integration-tests",
            base_model=get_base_model(),
        )
        try:
            # Train model A for 1 step with save_checkpoint=True
            await model_a.register(backend)
            openai_client_a = model_a.openai_client()
            step_a = await model_a.get_step()
            model_a_inf = model_a.get_inference_name(step=step_a)
            prompts = ["Say yes", "Say no", "Say maybe"]

            async def make_group_a(prompt: str) -> art.TrajectoryGroup:
                import asyncio

                trajectories = await asyncio.gather(
                    *[
                        simple_rollout(openai_client_a, model_a_inf, prompt)
                        for _ in range(2)
                    ]
                )
                return art.TrajectoryGroup(trajectories)  # type: ignore[attr-defined]

            train_groups_a = await art.gather_trajectory_groups(  # type: ignore[attr-defined]
                [make_group_a(prompt) for prompt in prompts]
            )
            ensure_reward_variance(train_groups_a)

            result_a = await backend.train(
                model_a,
                train_groups_a,
                learning_rate=1e-5,
                save_checkpoint=True,
            )
            assert result_a.step > 0
            print(f"Model A trained to step {result_a.step}")

            # Register model B, then fork from A
            await model_b.register(backend)
            await backend._experimental_fork_checkpoint(
                model_b,
                from_model=model_a_name,
                from_project="integration-tests",
                verbose=True,
            )
            print(f"Forked checkpoint from {model_a_name} to {model_b_name}")

            # Verify model B is at the same step as model A
            step_b = await model_b.get_step()
            assert step_b == result_a.step, (
                f"Expected model B at step {result_a.step}, got {step_b}"
            )

            # Train model B for 1 more step
            openai_client_b = model_b.openai_client()
            model_b_inf = model_b.get_inference_name(step=step_b)

            async def make_group_b(prompt: str) -> art.TrajectoryGroup:
                import asyncio

                trajectories = await asyncio.gather(
                    *[
                        simple_rollout(openai_client_b, model_b_inf, prompt)
                        for _ in range(2)
                    ]
                )
                return art.TrajectoryGroup(trajectories)  # type: ignore[attr-defined]

            train_groups_b = await art.gather_trajectory_groups(  # type: ignore[attr-defined]
                [make_group_b(prompt) for prompt in prompts]
            )
            ensure_reward_variance(train_groups_b)

            result_b = await backend.train(
                model_b,
                train_groups_b,
                learning_rate=1e-5,
            )
            assert result_b.step > step_b
            print(f"Model B trained to step {result_b.step}")
        finally:
            await backend.close()
