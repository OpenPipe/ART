"""Yes-no-maybe training with KL-penalized advantage adjustment.

Demonstrates the kl_penalty_coef feature: tokens where the policy has drifted
more from the reference model get reduced advantages, while tokens that have
drifted less get increased advantages.

Uses meta-llama/Meta-Llama-3.1-8B-Instruct as the base model (trained locally).
"""

import asyncio
from itertools import permutations
import os

from dotenv import load_dotenv
import openai

import art
from art.local import LocalBackend


async def rollout(
    client: openai.AsyncOpenAI, model: art.TrainableModel, prompt: str
) -> art.Trajectory:
    messages: art.Messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    chat_completion = await client.chat.completions.create(
        messages=messages, model=model.get_inference_name(), max_tokens=100, timeout=100
    )
    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str)
    if content == "yes":
        reward = 0.5
    elif content == "no":
        reward = 0.75
    elif content == "maybe":
        reward = 1.0
    else:
        reward = 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


def with_quotes(w: str) -> str:
    return f"'{w}'"


async def main():
    load_dotenv()

    backend = LocalBackend()
    base_model = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    kl_penalty_coef = float(os.environ.get("KL_PENALTY_COEF", "0.1"))
    model = art.TrainableModel(
        name=os.environ.get("MODEL_NAME", f"kl-{kl_penalty_coef}"),
        project="yes-no-maybe",
        base_model=base_model,
    )
    await model.register(backend)

    kl_penalty_reference_step: int | None = (
        int(os.environ["KL_REF_STEP"])
        if os.environ.get("KL_REF_STEP") is not None
        else None
    )
    kl_ref_adapter_path: str | None = os.environ.get("KL_REF_ADAPTER_PATH") or None

    prompts = [
        f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
        for prefix in ["respond", "just respond"]
        for use_quotes in [True, False]
        for words in (
            list(p) for n in [3, 2] for p in permutations(["yes", "no", "maybe"], n)
        )
    ]

    openai_client = model.openai_client()
    max_steps = int(os.environ.get("NUM_STEPS", "20"))
    start_step = await model.get_step()
    for step in range(start_step, start_step + max_steps):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(openai_client, model, prompt) for _ in range(32)
                )
                for prompt in prompts
            )
        )
        result = await backend.train(
            model,
            train_groups,
            learning_rate=1e-4,
            kl_penalty_coef=kl_penalty_coef,
            kl_penalty_reference_step=kl_penalty_reference_step,
            kl_ref_adapter_path=kl_ref_adapter_path,
        )
        await model.log(
            train_groups,
            metrics=result.metrics,
            step=result.step,
            split="train",
        )
        print(f"step {result.step}: {result.metrics}")


if __name__ == "__main__":
    asyncio.run(main())
