"""Minimal yes/no/maybe RL training example using PipelineTrainer."""

from __future__ import annotations

import asyncio
from datetime import datetime
from functools import partial
from itertools import cycle, permutations
import os
import re

from dotenv import load_dotenv

import art
from art.tinker_native import TinkerNativeBackend

from . import PipelineTrainer

# Training config
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # or Qwen/Qwen3-4B-Instruct-2507
MODEL_NAME = "pipeline-yes-no-maybe"
PROJECT = "yes-no-maybe-pipeline"
ROLLOUTS_PER_SCENARIO = 32
MAX_TOKENS = 5
MAX_STEPS = 20
EVAL_TRAJECTORY_COUNT = 30
EVAL_EVERY_N_STEPS = 2


def build_scenarios() -> list[dict]:
    """Generate all scenario variations."""
    scenarios: list[dict] = []
    for prefix in ["respond", "just respond"]:
        for use_quotes in [True, False]:
            for n in [3, 2]:
                for words in permutations(["yes", "no", "maybe"], n):
                    quoted = [f"'{w}'" if use_quotes else w for w in words]
                    if len(words) == 3:
                        body = ", ".join(quoted)
                    else:
                        body = " or ".join(quoted)
                    scenarios.append({"prompt": f"{prefix} with {body}"})
    return scenarios


def reward_for_answer(text: str) -> float:
    """Score: maybe=1.0, no=0.75, yes=0.5, other=0.0."""
    if not text:
        return 0.0
    first_word = re.split(r"\s+", text.strip().lower())[0].strip(".,!?:;\"'()[]{}")
    return {"maybe": 1.0, "no": 0.75, "yes": 0.5}.get(first_word, 0.0)


async def eval_fn(
    model: art.TrainableModel,
    step: int,
    _config: None,
    *,
    scenarios: list[dict],
) -> list[art.Trajectory]:
    trajectories: list[art.Trajectory] = []
    openai_client = model.openai_client()
    model_name = model.get_inference_name(step)
    for scenario in scenarios:
        messages: art.Messages = [{"role": "user", "content": scenario["prompt"]}]
        response = await openai_client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_tokens=MAX_TOKENS,
            n=1,
        )
        choice = response.choices[0]
        trajectories.append(
            art.Trajectory(
                messages_and_choices=[*messages, choice],
                reward=reward_for_answer(choice.message.content or ""),
            )
        )
    return trajectories


async def rollout_fn(model, scenario, _config) -> art.TrajectoryGroup:
    """Single inference call returns N completions for the group."""
    messages: art.Messages = [{"role": "user", "content": scenario["prompt"]}]
    response = await model.openai_client().chat.completions.create(
        messages=messages,
        model=model.get_inference_name(),
        max_tokens=MAX_TOKENS,
        n=ROLLOUTS_PER_SCENARIO,
    )
    return art.TrajectoryGroup(
        [
            art.Trajectory(
                messages_and_choices=[*messages, choice],
                reward=reward_for_answer(choice.message.content or ""),
            )
            for choice in response.choices
        ]
    )


async def main() -> None:
    load_dotenv()
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY environment variable is required")

    model_name = f"{MODEL_NAME}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print("Initializing TinkerNativeBackend")
    backend = TinkerNativeBackend()

    print(f"Initializing TrainableModel: {model_name}")
    model = art.TrainableModel(name=model_name, project=PROJECT, base_model=BASE_MODEL)

    print("Registering model with backend")
    await model.register(backend)
    print("Model registered")

    openai_client = model.openai_client()
    base_scenarios = build_scenarios()
    scenarios = cycle(base_scenarios)
    eval_scenarios = base_scenarios[:EVAL_TRAJECTORY_COUNT]

    eval_callback = partial(eval_fn, scenarios=eval_scenarios)

    trainer = PipelineTrainer(
        model=model,
        backend=backend,
        rollout_fn=rollout_fn,
        scenarios=scenarios,
        config=None,
        learning_rate=5e-5,
        loss_fn="cispo",
        eval_fn=eval_callback,
        max_steps=MAX_STEPS,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        eval_step_0=False,
        total_scenarios=None,
    )

    print(
        f"Training {model_name}: {MAX_STEPS} steps, "
        f"{len(base_scenarios)} unique scenarios (cycling)"
    )
    await trainer.train()
    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
