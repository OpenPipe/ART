"""Integration test for _experimental_push_to_s3 and _experimental_fork_checkpoint.

Trains a model, pushes its checkpoint(s) to S3, then creates a second model
by forking from the first model's checkpoint, and continues training.

Usage:
    uv run pytest tests/integration/test_push_and_fork.py -v -s

Environment variables:
    WANDB_API_KEY: Required (used by ServerlessBackend)
    BACKUP_BUCKET: S3 bucket for checkpoint storage (or pass explicitly)
    AWS credentials: Required for S3 operations
"""

import filecmp
import os
import tempfile
import uuid

import openai
import pytest

import art
from art.serverless.backend import ServerlessBackend
from art.types import ServerlessTrainResult

BASE_MODEL = "OpenPipe/Qwen3-14B-Instruct"
ROLLOUTS_PER_GROUP = 2


async def simple_rollout(
    client: openai.AsyncOpenAI, model_name: str, prompt: str
) -> art.Trajectory:
    """A simple rollout function that rewards 'yes' responses."""
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
    reward = 1.0 if "yes" in content else 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


async def train_one_step(
    model: art.TrainableModel,
    backend: ServerlessBackend,
) -> ServerlessTrainResult:
    """Run one training step and return the result."""
    openai_client = model.openai_client()
    current_step = await model.get_step()
    model_name = model.get_inference_name(step=current_step)
    prompts = ["Say yes", "Say no", "Say maybe", "Say hello"]

    train_groups = await art.gather_trajectory_groups(
        [
            art.TrajectoryGroup(
                [
                    simple_rollout(openai_client, model_name, prompt)
                    for _ in range(ROLLOUTS_PER_GROUP)
                ]
            )
            for prompt in prompts
        ]  # ty:ignore[invalid-argument-type]
    )

    result = await backend.train(model, train_groups, learning_rate=1e-5)
    await model.log(
        train_groups, metrics=result.metrics, step=result.step, split="train"
    )
    return result


@pytest.mark.skipif(
    "WANDB_API_KEY" not in os.environ,
    reason="WANDB_API_KEY not set",
)
async def test_push_to_s3():
    """Train a model, then push its checkpoints to S3."""
    run_id = uuid.uuid4().hex[:8]
    model_name = f"test-push-s3-{run_id}"
    s3_bucket = os.environ.get("BACKUP_BUCKET")

    backend = ServerlessBackend()
    model = art.TrainableModel(
        name=model_name,
        project="integration-tests",
        base_model=BASE_MODEL,
    )
    try:
        await model.register(backend)

        # Train one step
        result = await train_one_step(model, backend)
        assert isinstance(result, ServerlessTrainResult)
        assert result.step > 0

        # Push to S3
        await backend._experimental_push_to_s3(
            model,
            s3_bucket=s3_bucket,
            verbose=True,
        )
    finally:
        try:
            await backend.delete(model)
        except Exception:
            pass
        await backend.close()


@pytest.mark.skipif(
    "WANDB_API_KEY" not in os.environ,
    reason="WANDB_API_KEY not set",
)
async def test_fork_checkpoint_from_wandb():
    """Train model A, fork its checkpoint to model B via W&B, then train model B."""
    run_id = uuid.uuid4().hex[:8]
    model_a_name = f"test-fork-src-{run_id}"
    model_b_name = f"test-fork-dst-{run_id}"

    backend = ServerlessBackend()
    model_a = art.TrainableModel(
        name=model_a_name,
        project="integration-tests",
        base_model=BASE_MODEL,
    )
    model_b = art.TrainableModel(
        name=model_b_name,
        project="integration-tests",
        base_model=BASE_MODEL,
    )
    try:
        # Train model A
        await model_a.register(backend)
        result_a = await train_one_step(model_a, backend)
        assert result_a.step > 0
        print(f"Model A trained to step {result_a.step}")

        # Register model B, then fork from A (via W&B artifacts)
        await model_b.register(backend)
        await backend._experimental_fork_checkpoint(
            model_b,
            from_model=model_a_name,
            from_project="integration-tests",
            verbose=True,
        )
        print(f"Forked checkpoint from {model_a_name} to {model_b_name}")

        # Verify the forked checkpoint matches model A's checkpoint.
        # Pull both via W&B directly (the fork uploaded the artifact
        # with a step{N} alias matching the source step).
        import wandb

        api = wandb.Api(api_key=backend._client.api_key)  # ty:ignore[possibly-missing-attribute]
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a = os.path.join(tmpdir, "a")
            dir_b = os.path.join(tmpdir, "b")
            api.artifact(
                f"{model_a.entity}/integration-tests/{model_a_name}:step{result_a.step}",
                type="lora",
            ).download(root=dir_a)
            api.artifact(
                f"{model_b.entity}/integration-tests/{model_b_name}:step{result_a.step}",
                type="lora",
            ).download(root=dir_b)
            cmp = filecmp.dircmp(dir_a, dir_b)
            assert not cmp.left_only, (
                f"Files only in model A checkpoint: {cmp.left_only}"
            )
            assert not cmp.right_only, (
                f"Files only in model B checkpoint: {cmp.right_only}"
            )
            assert not cmp.diff_files, (
                f"Files differ between checkpoints: {cmp.diff_files}"
            )
            print("Verified: forked checkpoint matches model A's checkpoint")

        # Continue training model B
        result_b = await train_one_step(model_b, backend)
        assert result_b.step > 0
        print(f"Model B trained to step {result_b.step}")
    finally:
        for m in (model_a, model_b):
            try:
                await backend.delete(m)
            except Exception:
                pass
        await backend.close()


@pytest.mark.skipif(
    "WANDB_API_KEY" not in os.environ,
    reason="WANDB_API_KEY not set",
)
async def test_push_then_fork_from_s3():
    """Train model A, push to S3, fork from S3 into model B, then train model B."""
    run_id = uuid.uuid4().hex[:8]
    model_a_name = f"test-s3fork-src-{run_id}"
    model_b_name = f"test-s3fork-dst-{run_id}"
    s3_bucket = os.environ.get("BACKUP_BUCKET")

    backend = ServerlessBackend()
    model_a = art.TrainableModel(
        name=model_a_name,
        project="integration-tests",
        base_model=BASE_MODEL,
    )
    model_b = art.TrainableModel(
        name=model_b_name,
        project="integration-tests",
        base_model=BASE_MODEL,
    )
    try:
        # Train model A
        await model_a.register(backend)
        result_a = await train_one_step(model_a, backend)
        assert result_a.step > 0
        print(f"Model A trained to step {result_a.step}")

        # Push model A to S3
        await backend._experimental_push_to_s3(
            model_a,
            s3_bucket=s3_bucket,
            verbose=True,
        )
        print(f"Pushed model A to S3")

        # Register model B, then fork from S3
        await model_b.register(backend)
        await backend._experimental_fork_checkpoint(
            model_b,
            from_model=model_a_name,
            from_project="integration-tests",
            from_s3_bucket=s3_bucket,
            verbose=True,
        )
        print(f"Forked checkpoint from S3 into {model_b_name}")

        # Continue training model B
        result_b = await train_one_step(model_b, backend)
        assert result_b.step > 0
        print(f"Model B trained to step {result_b.step}")
    finally:
        for m in (model_a, model_b):
            try:
                await backend.delete(m)
            except Exception:
                pass
        await backend.close()
