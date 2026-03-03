import asyncio
import json

from dotenv import load_dotenv

import art
from art.local import LocalBackend

load_dotenv()


async def rollout(model: art.TrainableModel, prompt: str) -> art.Trajectory:
    messages: art.Messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    client = model.openai_client()
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model.name,
        max_tokens=100,
        timeout=100,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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


async def main():
    with open("dev/new_models/prompts.json", "r") as f:
        prompts = json.load(f)
    print(prompts)

    backend = LocalBackend()
    model = art.TrainableModel(
        name="007",
        project="yes-no-maybe-s",
        base_model="Qwen/Qwen3-0.6B",
        # base_model="Qwen/Qwen2.5-0.5B-Instruct",
    )
    await model.register(backend)
    for _ in range(await model.get_step(), 1_000):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(rollout(model, prompt) for _ in range(32))
                for prompt in prompts
            ),
            pbar_desc="gather",
        )
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-4),
        )


if __name__ == "__main__":
    asyncio.run(main())
