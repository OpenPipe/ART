"""
Trains a Llama 3.1 8B model (static name) for 10 yes-no-maybe steps, then forks
it into a fresh run (dynamic name) and verifies the fork starts from step 10.

Checkpoint persistence:
- Before training: pulls the step-10 checkpoint from W&B if it exists (skips retraining).
- After training: pushes the checkpoint to W&B as a LoRA artifact.

Usage:
    uv run dev/yes-no-maybe-fork.py
"""

import asyncio
import os
import uuid

from dotenv import load_dotenv
import openai

import art
from art.local import LocalBackend
from art.utils.deployment.wandb import deploy_wandb
from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_NAME = "ynm-fork-llama-31-8b-base"
PROJECT = "yes-no-maybe-fork"
TRAIN_STEPS = 10
ROLLOUTS_PER_GROUP = 8
PROMPTS = ["Say yes", "Say no", "Say maybe"]
STEP0_PASS_RATE_FILE = os.path.expanduser(
    "~/.art_data/yes-no-maybe-fork-step0-pass-rate.json"
)


async def rollout(
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
        reward = 0.5
    elif "no" in content:
        reward = 0.75
    elif "maybe" in content:
        reward = 1.0
    else:
        reward = 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


def compute_pass_rate(groups: list[art.TrajectoryGroup]) -> float:
    trajectories = [t for g in groups for t in g]
    passing = sum(1 for t in trajectories if t.reward > 0)
    return passing / len(trajectories) if trajectories else 0.0


async def evaluate(model: art.TrainableModel, step: int) -> float:
    """Run one rollout batch at the given checkpoint step and return pass rate."""
    client = model.openai_client()
    model_name = model.get_inference_name(step=step)
    groups = await art.gather_trajectory_groups(
        [
            art.TrajectoryGroup(
                [rollout(client, model_name, prompt) for _ in range(ROLLOUTS_PER_GROUP)]
            )
            for prompt in PROMPTS
        ]
    )
    return compute_pass_rate(groups)


async def run_training_steps(
    model: art.TrainableModel,
    backend: LocalBackend,
    num_steps: int,
) -> None:
    client = model.openai_client()
    for _ in range(num_steps):
        current_step = await model.get_step()
        model_name = model.get_inference_name(step=current_step)
        train_groups = await art.gather_trajectory_groups(
            [
                art.TrajectoryGroup(
                    [
                        rollout(client, model_name, prompt)
                        for _ in range(ROLLOUTS_PER_GROUP)
                    ]
                )
                for prompt in PROMPTS
            ]
        )
        result = await backend.train(model, train_groups, learning_rate=1e-4)
        print(f"  Step {result.step} done.")


def try_pull_from_wandb(
    model: art.TrainableModel, backend: LocalBackend, step: int
) -> bool:
    """Download a checkpoint from W&B to local disk. Returns True if successful."""
    if "WANDB_API_KEY" not in os.environ:
        return False
    try:
        import wandb

        api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
        if model.entity is None:
            model.entity = api.default_entity
        artifact = api.artifact(
            f"{model.entity}/{model.project}/{model.name}:step{step}", type="lora"
        )
        checkpoint_path = get_step_checkpoint_dir(
            get_model_dir(model, backend._path), step
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        artifact.download(root=checkpoint_path)
        print(f"Pulled step-{step} checkpoint from W&B to {checkpoint_path}.")
        return True
    except Exception as e:
        print(f"No W&B checkpoint found ({e}), will train from scratch.")
        return False


def push_to_wandb(model: art.TrainableModel, backend: LocalBackend, step: int) -> None:
    """Upload a local checkpoint to W&B as a LoRA artifact."""
    if "WANDB_API_KEY" not in os.environ:
        print("WANDB_API_KEY not set, skipping W&B upload.")
        return
    checkpoint_path = get_step_checkpoint_dir(get_model_dir(model, backend._path), step)
    deploy_wandb(model, checkpoint_path, step, verbose=True)


async def _wait_for_gpu_free(
    target_free_gib: float = 100.0, poll_interval: float = 3.0, timeout: float = 30.0
) -> None:
    """Poll nvidia-smi until enough GPU memory is free, force-killing stragglers if needed."""
    import subprocess
    import time

    def free_gib() -> float | None:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().split("\n")[0]) / 1024
        return None

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        gib = free_gib()
        if gib is not None:
            print(f"GPU free memory: {gib:.1f} GiB")
            if gib > target_free_gib:
                return
        await asyncio.sleep(poll_interval)

    # Natural wait timed out — force-kill remaining CUDA processes.
    print("GPU not yet free; force-killing lingering CUDA processes...")
    subprocess.run(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
        shell=True,
    )
    await asyncio.sleep(5)

    gib = free_gib()
    print(f"GPU free memory after force-kill: {gib:.1f} GiB")
    if gib is not None and gib > target_free_gib:
        return
    raise TimeoutError(f"GPU did not free {target_free_gib} GiB even after force-kill")


async def main() -> None:
    load_dotenv()

    backend = LocalBackend()

    # --- Phase 1: train the base model (static name, resume-safe) ---
    model_a = art.TrainableModel(
        run_name=BASE_MODEL_NAME,
        name=BASE_MODEL_NAME,
        project=PROJECT,
        base_model=BASE_MODEL,
    )
    await model_a.register(backend)

    start_step = await model_a.get_step()

    # Load persisted step-0 pass rate if available (survives W&B checkpoint pulls).
    import json as _json

    base_pass_rate: float | None = None
    if os.path.exists(STEP0_PASS_RATE_FILE):
        with open(STEP0_PASS_RATE_FILE) as f:
            base_pass_rate = _json.load(f)["pass_rate"]
        print(f"Loaded step-0 pass rate from cache: {base_pass_rate:.1%}")

    if start_step >= TRAIN_STEPS:
        print(f"Base model already at step {start_step}, skipping training.")
    else:
        # Only try W&B when there's no local progress (avoid pulling a stale artifact
        # over a partially-trained local checkpoint).
        if start_step == 0 and try_pull_from_wandb(model_a, backend, TRAIN_STEPS):
            start_step = await model_a.get_step()
        else:
            # Evaluate the base model before training begins. Step 0 is only
            # served by vLLM when there is no pre-existing checkpoint, so this
            # must happen before the first training step.
            print("Evaluating base model (step 0)...")
            base_pass_rate = await evaluate(model_a, step=0)
            print(f"Base model pass rate: {base_pass_rate:.1%}")
            os.makedirs(os.path.dirname(STEP0_PASS_RATE_FILE), exist_ok=True)
            with open(STEP0_PASS_RATE_FILE, "w") as f:
                _json.dump({"pass_rate": base_pass_rate}, f)
            steps_needed = TRAIN_STEPS - start_step
            print(f"Training base model from step {start_step} → {TRAIN_STEPS}...")
            await run_training_steps(model_a, backend, steps_needed)
            push_to_wandb(model_a, backend, await model_a.get_step())

    final_step_a = await model_a.get_step()
    assert final_step_a >= TRAIN_STEPS, (
        f"Expected base model at >= step {TRAIN_STEPS}, got {final_step_a}"
    )
    print(f"Base model is at step {final_step_a}.")

    # Close the backend and wait for the GPU memory to actually be released
    # before starting the forked model's service.
    await backend.close()
    await _wait_for_gpu_free()
    backend = LocalBackend()

    # --- Phase 2: fork into a fresh run (dynamic name) ---
    model_b_name = f"ynm-fork-{uuid.uuid4().hex[:8]}"
    print(f"Forking into '{model_b_name}'...")
    model_b = art.TrainableModel(
        run_name=model_b_name,
        name=model_b_name,
        project=PROJECT,
        base_model=BASE_MODEL,
    )
    # Fork first (pure file copy, no GPU needed), then register.
    await backend._experimental_fork_checkpoint(
        model_b,
        from_model=BASE_MODEL_NAME,
        from_project=PROJECT,
        verbose=True,
    )
    await model_b.register(backend)

    step_b = await model_b.get_step()
    assert step_b == final_step_a, (
        f"Forked model should start at step {final_step_a}, but is at step {step_b}"
    )
    print(f"Fork verified: model_b starts at step {step_b} ✓")

    print("Evaluating forked model...")
    forked_pass_rate = await evaluate(model_b, step=step_b)
    print(f"Forked model pass rate: {forked_pass_rate:.1%}")

    print(f"\n--- Pass rate comparison ---")
    if base_pass_rate is not None:
        print(f"  Base model   (step  0): {base_pass_rate:.1%}")
    else:
        print(f"  Base model   (step  0): N/A")
    print(f"  Forked model (step {step_b:2d}): {forked_pass_rate:.1%}")
    print(f"----------------------------\n")

    # Train the forked model to confirm it can continue from where model_a left off.
    print("Training forked model for 2 more steps...")
    await run_training_steps(model_b, backend, 2)
    final_step_b = await model_b.get_step()
    assert final_step_b == step_b + 2, (
        f"Expected forked model at step {step_b + 2}, got {final_step_b}"
    )
    print(f"Success: forked model trained from step {step_b} → {final_step_b}.")


if __name__ == "__main__":
    asyncio.run(main())
