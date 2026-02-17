import argparse
import asyncio
import itertools
import os
import re
from typing import Iterator, TypedDict, cast

import polars as pl

import art
from art.local import LocalBackend


class DecodedImage(TypedDict):
    bytes: bytes


class Scenario(TypedDict):
    pid: int
    question: str
    answer: str
    image: str
    decoded_image: DecodedImage


async def main(model_name: str, steps: int) -> None:
    # Load and shuffle the dataset
    df = pl.read_parquet(
        "hf://datasets/AI4Math/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
    ).sample(fraction=1.0, shuffle=True, seed=42)

    val_scenarios = cast(list[Scenario], df.head(64).to_dicts())
    train_scenarios_iter = cast(Iterator[Scenario], df.tail(-64).iter_rows(named=True))

    # Initialize trainable model and backend
    model = art.TrainableModel(
        name=model_name,
        project="math-vista",
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    async def rollout(scenario: Scenario) -> art.Trajectory:
        image_path = f"/tmp/{scenario['image']}"
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(scenario["decoded_image"]["bytes"])

        trajectory = art.Trajectory(messages_and_choices=[], reward=0.0)
        trajectory.messages_and_choices = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": scenario["question"]
                        + "\n\nNote: Provide your answer in a LaTeX box.",
                    },
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
                ],
            }
        ]

        chat_completion = await client.chat.completions.create(
            model=model.name, messages=trajectory.messages()
        )
        choice = chat_completion.choices[0]
        trajectory.messages_and_choices.append(choice)
        content = choice.message.content
        assert content is not None

        if matches := list(re.finditer(r"\\boxed\{(.*?)\}", content, re.DOTALL)):
            match = matches[-1]
            answer = match.group(1)
            if answer.lower() == scenario["answer"].lower():
                trajectory.reward = 1.0
        return trajectory

    SCENARIOS_PER_STEP = 8
    TRAJECTORY_GROUP_SIZE = 8

    with LocalBackend() as backend:
        await model.register(backend)
        client = model.openai_client()

        start = await model.get_step()
        train_scenarios_iter = itertools.cycle(train_scenarios_iter)
        for _ in range(start * SCENARIOS_PER_STEP):
            next(train_scenarios_iter)

        # Training loop
        for _ in range(start, steps):
            train_scenarios = [
                next(train_scenarios_iter) for _ in range(SCENARIOS_PER_STEP)
            ]
            val_trajectories, train_trajectory_groups = await asyncio.gather(
                art.gather_trajectories(
                    (rollout(scenario) for scenario in val_scenarios),
                    pbar_desc="gather(val)",
                    max_exceptions=32,
                ),
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(scenario) for _ in range(TRAJECTORY_GROUP_SIZE)
                        )
                        for scenario in train_scenarios
                    ),
                    pbar_desc="gather(train)",
                    max_exceptions=32,
                ),
            )
            await model.log(val_trajectories)
            await model.train(train_trajectory_groups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MathVista trainer script")
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="Run/model name to use for the TrainableModel",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=1000,
        help="Number of training steps to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.name, args.steps))
