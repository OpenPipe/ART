"""Integration tests for multi-checkpoint inference with real training loops.

These tests run actual training loops with different backends to verify that
multi-checkpoint inference works end-to-end without crashing.

Usage:
    # Run all integration tests (requires appropriate backend setup)
    uv run pytest tests/integration/test_multi_checkpoint_training.py -v -s

Environment variables:
    BASE_MODEL: The base model to use (default: Qwen/Qwen3-0.6B)
    WANDB_API_KEY: Required for ServerlessBackend test
    TINKER_API_KEY: Required for TinkerBackend test
"""

import os
import tempfile
from typing import Union
import uuid

import openai
import pytest

import art
from art.local import LocalBackend
from art.tinker import TinkerBackend
from art.types import LocalTrainResult, ServerlessTrainResult, TrainResult

# Use a small model for fast testing
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"


def get_base_model() -> str:
    """Get the base model to use for testing."""
    return os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)


async def simple_rollout(
    client: openai.AsyncOpenAI, model_name: str, prompt: str
) -> art.Trajectory:
    """A simple rollout function for testing."""
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
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


async def run_training_loop(
    model: art.TrainableModel,
    backend: art.Backend,
    num_steps: int = 1,
    rollouts_per_step: int = 4,
) -> list[TrainResult]:
    """Run a simple training loop and return the TrainResults from each train call."""
    openai_client = model.openai_client()
    prompts = ["Say yes", "Say no", "Say maybe", "Say hello"]
    results: list[TrainResult] = []

    for _ in range(num_steps):
        current_step = await model.get_step()
        # Use get_inference_name(step=current_step) to target the current checkpoint
        model_name = model.get_inference_name(step=current_step)
        train_groups = await art.gather_trajectory_groups(
            [
                art.TrajectoryGroup(
                    [
                        simple_rollout(openai_client, model_name, prompt)
                        for _ in range(rollouts_per_step)
                    ]
                )
                for prompt in prompts
            ]  # ty:ignore[invalid-argument-type]
        )
        result = await backend.train(model, train_groups, learning_rate=1e-5)
        await model.log(
            train_groups, metrics=result.metrics, step=result.step, split="train"
        )
        results.append(result)

    return results


async def _run_inference_on_step(
    model: art.TrainableModel,
    step: int,
) -> None:
    openai_client = model.openai_client()
    model_name = model.get_inference_name(step=step)
    await openai_client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello"}],
        model=model_name,
        max_tokens=10,
        timeout=30,
    )


@pytest.mark.skipif(
    "TINKER_API_KEY" not in os.environ,
    reason="TINKER_API_KEY not set - skipping TinkerBackend test",
)
async def test_tinker_backend():
    """Test multi-checkpoint inference with TinkerBackend."""
    model_name = f"test-multi-ckpt-tinker-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = TinkerBackend(path=tmpdir)
        model = art.TrainableModel(
            name=model_name,
            project="integration-tests",
            base_model=get_base_model(),
        )
        try:
            await model.register(backend)
            results = await run_training_loop(
                model, backend, num_steps=1, rollouts_per_step=2
            )
            # Verify TrainResult structure
            assert len(results) == 1
            assert isinstance(results[0], LocalTrainResult)
            assert results[0].step > 0
            await _run_inference_on_step(model, step=results[-1].step)
            await _run_inference_on_step(model, step=0)
        finally:
            await backend.close()


@pytest.mark.skipif(
    not os.path.exists("/dev/nvidia0"),
    reason="No GPU available - skipping LocalBackend test",
)
async def test_local_backend():
    """Test multi-checkpoint inference with LocalBackend (UnslothService)."""
    model_name = f"test-multi-ckpt-local-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalBackend(path=tmpdir)
        model = art.TrainableModel(
            name=model_name,
            project="integration-tests",
            base_model=get_base_model(),
        )
        try:
            await model.register(backend)
            results = await run_training_loop(
                model, backend, num_steps=1, rollouts_per_step=2
            )
            # Verify TrainResult structure
            assert len(results) == 1
            assert isinstance(results[0], LocalTrainResult)
            assert results[0].step > 0
            assert results[0].checkpoint_path is not None
            await _run_inference_on_step(model, step=results[-1].step)
            await _run_inference_on_step(model, step=0)
        finally:
            await backend.close()


@pytest.mark.skipif(
    "WANDB_API_KEY" not in os.environ,
    reason="WANDB_API_KEY not set - skipping ServerlessBackend test",
)
async def test_serverless_backend():
    """Test multi-checkpoint inference with ServerlessBackend."""
    model_name = f"test-multi-ckpt-serverless-{uuid.uuid4().hex[:8]}"
    backend = art.ServerlessBackend()
    model = art.TrainableModel(
        name=model_name,
        project="integration-tests",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
    )
    try:
        await model.register(backend)
        results = await run_training_loop(
            model, backend, num_steps=1, rollouts_per_step=2
        )
        # Verify TrainResult structure
        assert len(results) == 1
        assert isinstance(results[0], ServerlessTrainResult)
        assert results[0].step > 0
        assert results[0].artifact_name is not None
        await _run_inference_on_step(model, step=results[-1].step)
        await _run_inference_on_step(model, step=0)
    finally:
        try:
            await backend.delete(model)
        except Exception:
            pass
        await backend.close()
