"""
Trains a Llama 3.1 8B model using PipelineTrainer for 10 steps, then forks
it into a fresh run (dynamic name) and verifies the fork starts from step 10.

Requires 2 GPUs: GPU 0 for training (Unsloth), GPU 1 for inference (vLLM).

Checkpoint persistence:
- Before training: pulls the step-10 checkpoint from W&B if it exists.
- After training: pushes the checkpoint to W&B as a LoRA artifact.

Usage:
    uv run dev/yes-no-maybe-fork-pipeline.py
"""

import asyncio
import json
import os
import uuid

os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "bf16")

from dotenv import load_dotenv
import openai

import art
from art.local import LocalBackend
from art.pipeline_trainer import PipelineRuntimeConfig, PipelineTrainer
from art.utils.deployment.wandb import deploy_wandb
from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_NAME = "ynm-fork-pipeline-llama-31-8b-base"
PROJECT = "yes-no-maybe-fork-pipeline"
TRAIN_STEPS = 5
ROLLOUTS_PER_SCENARIO = 8
PROMPTS = ["Say yes", "Say no", "Say maybe"]
STEP0_PASS_RATE_FILE = os.path.expanduser(
    "~/.art_data/yes-no-maybe-fork-pipeline-step0-pass-rate.json"
)

DEDICATED_CONFIG: art.dev.InternalModelConfig = {
    "trainer_gpu_ids": [0],
    "inference_gpu_ids": [1],
    "init_args": {
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "load_in_16bit": True,
    },
    "engine_args": {
        "gpu_memory_utilization": 0.4,
        "max_model_len": 2048,
        "max_num_seqs": 16,
        "enforce_eager": True,
    },
}


def reward_for_answer(text: str) -> float:
    content = text.lower()
    if "maybe" in content:
        return 1.0
    if "no" in content:
        return 0.75
    if "yes" in content:
        return 0.5
    return 0.0


def compute_pass_rate(groups: list[art.TrajectoryGroup]) -> float:
    trajectories = [t for g in groups for t in g]
    passing = sum(1 for t in trajectories if t.reward > 0)
    return passing / len(trajectories) if trajectories else 0.0


async def evaluate(model: art.TrainableModel, step: int) -> float:
    client = model.openai_client()
    model_name = model.get_inference_name(step=step)
    groups = await art.gather_trajectory_groups(
        [
            art.TrajectoryGroup(
                [
                    _rollout(client, model_name, prompt)
                    for _ in range(ROLLOUTS_PER_SCENARIO)
                ]
            )
            for prompt in PROMPTS
        ]
    )
    return compute_pass_rate(groups)


async def _rollout(
    client: openai.AsyncOpenAI, model_name: str, prompt: str
) -> art.Trajectory:
    messages: art.Messages = [{"role": "user", "content": prompt}]
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=10,
        timeout=60,
        temperature=1,
        logprobs=True,
        top_logprobs=0,
    )
    choice = chat_completion.choices[0]
    return art.Trajectory(
        messages_and_choices=[*messages, choice],
        reward=reward_for_answer(choice.message.content or ""),
    )


def scenario_iter():
    """Infinite cycle over the training prompts."""
    while True:
        for prompt in PROMPTS:
            yield {"prompt": prompt}


def make_rollout_fn():
    async def rollout_fn(
        model: art.TrainableModel,
        scenario: dict[str, str],
        _config: None,
    ) -> art.TrajectoryGroup:
        messages: art.Messages = [{"role": "user", "content": scenario["prompt"]}]
        completion = await model.openai_client().chat.completions.create(
            messages=messages,
            model=model.get_inference_name(),
            max_tokens=10,
            timeout=60,
            temperature=1,
            n=ROLLOUTS_PER_SCENARIO,
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

    return rollout_fn


def make_eval_fn(on_result=None):
    """Returns an eval_fn. on_result(step, pass_rate) is called if provided."""

    async def eval_fn(
        model: art.TrainableModel,
        step: int,
        _config: None,
    ) -> list[art.TrajectoryGroup]:
        client = model.openai_client()
        model_name = model.get_inference_name(step=step)
        groups = await art.gather_trajectory_groups(
            [
                art.TrajectoryGroup(
                    [
                        _rollout(client, model_name, prompt)
                        for _ in range(ROLLOUTS_PER_SCENARIO)
                    ]
                )
                for prompt in PROMPTS
            ]
        )
        rate = compute_pass_rate(groups)
        print(f"  Eval at step {step}: {rate:.1%}")
        if on_result is not None:
            on_result(step, rate)
        return groups

    return eval_fn


def try_pull_from_wandb(
    model: art.TrainableModel, backend: LocalBackend, step: int
) -> bool:
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
    if "WANDB_API_KEY" not in os.environ:
        print("WANDB_API_KEY not set, skipping W&B upload.")
        return
    checkpoint_path = get_step_checkpoint_dir(get_model_dir(model, backend._path), step)
    deploy_wandb(model, checkpoint_path, step, verbose=True)


async def _wait_for_gpu_free(
    target_free_gib: float = 100.0, poll_interval: float = 3.0, timeout: float = 30.0
) -> None:
    import subprocess
    import time

    def free_gib() -> float | None:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            # Sum free memory across all GPUs; training needs both to be mostly free.
            values = [float(v) for v in r.stdout.strip().split("\n") if v.strip()]
            return min(values) / 1024 if values else None
        return None

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        gib = free_gib()
        if gib is not None:
            print(f"GPU min-free memory: {gib:.1f} GiB")
            if gib > target_free_gib:
                return
        await asyncio.sleep(poll_interval)

    print("GPU not yet free; force-killing lingering CUDA processes...")
    subprocess.run(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
        shell=True,
    )
    await asyncio.sleep(5)

    gib = free_gib()
    print(f"GPU min-free memory after force-kill: {gib:.1f} GiB")
    if gib is not None and gib > target_free_gib:
        return
    raise TimeoutError(f"GPU did not free {target_free_gib} GiB even after force-kill")


async def main() -> None:
    load_dotenv()

    backend = LocalBackend()

    # --- Phase 1: train the base model (static name, resume-safe) ---
    model_a = art.TrainableModel(
        name=BASE_MODEL_NAME,
        project=PROJECT,
        base_model=BASE_MODEL,
        _internal_config=DEDICATED_CONFIG,
    )
    await model_a.register(backend)

    start_step = await model_a.get_step()

    # Load cached step-0 pass rate (survives W&B checkpoint pulls).
    base_pass_rate: float | None = None
    if os.path.exists(STEP0_PASS_RATE_FILE):
        with open(STEP0_PASS_RATE_FILE) as f:
            base_pass_rate = json.load(f)["pass_rate"]
        print(f"Loaded step-0 pass rate from cache: {base_pass_rate:.1%}")

    # Capture final-step pass rate from PipelineTrainer callback (populated only when training runs).
    base_final_pass_rate: dict[str, float] = {}

    if start_step >= TRAIN_STEPS:
        print(f"Base model already at step {start_step}, skipping training.")
    else:
        if start_step == 0 and try_pull_from_wandb(model_a, backend, TRAIN_STEPS):
            start_step = await model_a.get_step()
        else:
            # Capture step-0 pass rate from PipelineTrainer callback.
            step0_captured: dict[str, float] = {}

            def on_eval(step: int, rate: float) -> None:
                if step == 0 and base_pass_rate is None:
                    step0_captured["rate"] = rate
                if step == TRAIN_STEPS:
                    base_final_pass_rate["rate"] = rate

            trainer_a = PipelineTrainer(
                model=model_a,
                backend=backend,
                rollout_fn=make_rollout_fn(),
                scenarios=scenario_iter(),
                config=None,
                pipeline=PipelineRuntimeConfig(
                    num_rollout_workers=len(PROMPTS),
                    min_batch_size=len(PROMPTS),
                    max_batch_size=len(PROMPTS),
                ),
                max_steps=TRAIN_STEPS - start_step,
                learning_rate=1e-4,
                loss_fn="cispo",
                eval_fn=make_eval_fn(on_eval),
                eval_at_start=base_pass_rate is None,
                eval_every_n_steps=TRAIN_STEPS,
                discard_queue_multiplier=10000,
            )
            print(f"Training base model from step {start_step} → {TRAIN_STEPS}...")
            await trainer_a.train()

            if "rate" in step0_captured:
                base_pass_rate = step0_captured["rate"]
                os.makedirs(os.path.dirname(STEP0_PASS_RATE_FILE), exist_ok=True)
                with open(STEP0_PASS_RATE_FILE, "w") as f:
                    json.dump({"pass_rate": base_pass_rate}, f)

            push_to_wandb(model_a, backend, await model_a.get_step())

    final_step_a = await model_a.get_step()
    assert final_step_a >= TRAIN_STEPS, (
        f"Expected base model at >= step {TRAIN_STEPS}, got {final_step_a}"
    )
    print(f"Base model is at step {final_step_a}.")

    await backend.close()
    await _wait_for_gpu_free()
    backend = LocalBackend()

    # --- Phase 2: fork into a fresh run (dynamic name) ---
    model_b_name = f"ynm-fork-pipeline-{uuid.uuid4().hex[:8]}"
    print(f"Forking into '{model_b_name}'...")
    model_b = art.TrainableModel(
        name=model_b_name,
        project=PROJECT,
        base_model=BASE_MODEL,
        _internal_config=DEDICATED_CONFIG,
    )
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

    # Train forked model for 2 more steps.
    # Use eval_at_start=True to fire an eval at the forked model's starting step
    # (step_b). discard_queue_multiplier=1 lets training exit quickly after the
    # eval completes (once 3 zero-variance groups are discarded).
    forked_eval_results: dict[int, float] = {}

    def on_forked_eval(step: int, rate: float) -> None:
        forked_eval_results[step] = rate

    trainer_b = PipelineTrainer(
        model=model_b,
        backend=backend,
        rollout_fn=make_rollout_fn(),
        scenarios=scenario_iter(),
        config=None,
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=len(PROMPTS),
            min_batch_size=len(PROMPTS),
            max_batch_size=len(PROMPTS),
        ),
        max_steps=2,
        learning_rate=1e-4,
        loss_fn="cispo",
        eval_fn=make_eval_fn(on_forked_eval),
        eval_at_start=True,
        eval_every_n_steps=2,
        discard_queue_multiplier=1,
    )
    print(f"Evaluating forked model at starting step {step_b} via pipeline...")
    await trainer_b.train()

    forked_pass_rate = forked_eval_results.get(step_b)
    assert forked_pass_rate is not None, (
        f"Expected eval at step {step_b}, got keys: {list(forked_eval_results)}"
    )
    print(f"Forked model pass rate at step {step_b}: {forked_pass_rate:.1%}")

    print(f"\n--- Pass rate comparison ---")
    if base_pass_rate is not None:
        print(f"  Base model   (step  0):  {base_pass_rate:.1%}")
    else:
        print(f"  Base model   (step  0):  N/A")
    if "rate" in base_final_pass_rate:
        print(
            f"  Base model   (step {final_step_a:2d}): {base_final_pass_rate['rate']:.1%}  ← trained performance"
        )
    else:
        print(
            f"  Base model   (step {final_step_a:2d}): see forked model below  (checkpoint reused)"
        )
    print(
        f"  Forked model (step {step_b:2d}): {forked_pass_rate:.1%}"
        f"  ← fork preserved base model step {final_step_a} weights"
    )
    print(f"----------------------------\n")
    print(f"Success: forked model starts at step {step_b} with trained performance.")


if __name__ == "__main__":
    asyncio.run(main())
