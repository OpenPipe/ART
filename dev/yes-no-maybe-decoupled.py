#!/usr/bin/env python3
import asyncio
import time
import art
from art.local import LocalBackend
from dotenv import load_dotenv
import openai

load_dotenv()


async def rollout(
    client: openai.AsyncOpenAI, prompt: str, model_name: str
) -> art.Trajectory:
    messages: art.Messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    chat_completion = await client.chat.completions.create(
        messages=messages, model=model_name, max_tokens=100, timeout=100
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


def with_quotes(w):
    return f"'{w}'"


async def main():
    backend = LocalBackend()
    model = art.TrainableModel(
        name="001-decoupled",
        project="yes-no-maybe",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        _internal_config={
            "use_decoupled_unsloth": True,  # This triggers the use of DecoupledUnslothService
        },
    )
    await model.register(backend)

    prompts = [
        f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
        for prefix in ["respond", "just respond"]
        for use_quotes in [True, False]
        for words in [
            ["yes", "no", "maybe"],
            ["maybe", "yes", "no"],
            ["no", "yes", "maybe"],
            ["yes", "maybe", "no"],
            ["yes", "no"],
            ["maybe", "no"],
            ["no", "maybe"],
            ["no", "yes"],
            ["yes", "no"],
        ]
    ]

    openai_client = model.openai_client()

    # Track reward improvements
    start_time = time.time()
    initial_rewards = []
    latest_rewards = []

    start_step = await model.get_step()
    print(f"Starting training from step {start_step} using DecoupledUnslothService")

    for step in range(start_step, 1_000):
        # Check if 6 minutes have elapsed
        elapsed_time = time.time() - start_time
        if elapsed_time > 360:  # 6 minutes
            print(f"\n6 minutes elapsed. Stopping training at step {step}.")
            break

        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(openai_client, prompt, model.name) for _ in range(32)
                )
                for prompt in prompts
            ),
            pbar_desc="gather",
            pbar_total_completion_tokens=False,
        )

        # Track rewards
        all_rewards = []
        for group in train_groups:
            trajectories = await group
            for trajectory in trajectories:
                all_rewards.append(trajectory.reward)

        avg_reward = sum(all_rewards) / len(all_rewards)

        # Store initial rewards from first step
        if step == start_step:
            initial_rewards = all_rewards
            initial_avg = avg_reward
            print(f"Initial average reward at step {step}: {initial_avg:.4f}")

        # Store latest rewards
        latest_rewards = all_rewards
        latest_avg = avg_reward

        print(f"Step {step}: Average reward = {avg_reward:.4f}")

        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-4),
        )

    # Calculate improvement
    if initial_rewards and latest_rewards:
        improvement = ((latest_avg - initial_avg) / initial_avg) * 100
        print(f"\nTraining completed.")
        print(f"Initial average reward: {initial_avg:.4f}")
        print(f"Final average reward: {latest_avg:.4f}")
        print(f"Reward improvement: {improvement:.2f}%")

        if improvement >= 2.0:
            print("✓ Target of 2% improvement achieved!")
        else:
            print(f"✗ Target of 2% improvement not met (got {improvement:.2f}%)")
    else:
        print("Unable to calculate reward improvement")


if __name__ == "__main__":
    asyncio.run(main())
