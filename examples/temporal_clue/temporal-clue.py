import art
import asyncio
from dotenv import load_dotenv
import json
import random
import re
from typing import TypedDict
import os

from art.local import LocalBackend

load_dotenv()


class TemporalCluePuzzle(TypedDict):
    num_clues: int
    prompt: str
    solution: dict[str, str]


# download the puzzles from the github repo
puzzles_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "temporal-clue", "puzzles.json"
)

puzzles: list[TemporalCluePuzzle] = json.loads(open(puzzles_path).read())
val_puzzles = puzzles[:64]
test_puzzles = puzzles[64:128]
train_puzzles = puzzles[128:]
random.seed(42)
random.shuffle(train_puzzles)


async def rollout(model: art.Model, puzzle: TemporalCluePuzzle) -> art.Trajectory:
    messages: art.Messages = [{"role": "user", "content": puzzle["prompt"]}]
    client = model.openai_client()
    chat_completion = await client.chat.completions.create(
        messages=messages, model=model.name
    )
    choice = chat_completion.choices[0]
    content = choice.message.content
    assert isinstance(content, str)
    num_correct = 0
    for key, value in puzzle["solution"].items():
        if matches := re.findall(rf"{key}\. ([A-Za-z \.:-]+)", content):
            match = matches[-1]
            if match.strip().lower() == value.lower():
                num_correct += 1
    reward = acc = num_correct / len(puzzle["solution"])
    return art.Trajectory(
        messages_and_choices=[*messages, choice], reward=reward, metrics={"acc": acc}
    )


async def main():
    model = art.TrainableModel(
        name="001",
        project="temporal-clue",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        _internal_config={"init_args": {"gpu_memory_utilization": 0.775}},
    )
    backend = LocalBackend()
    await backend._experimental_pull_from_s3(model)
    await model.register(backend)

    stride = 4
    for i in range(await model.get_step(), 1_000):
        val_groups, train_groups = await asyncio.gather(
            art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(rollout(model, puzzle) for _ in range(2))
                    for puzzle in val_puzzles
                ),
                pbar_desc="val",
            ),
            art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(rollout(model, puzzle) for _ in range(50))
                    for puzzle in train_puzzles[i * stride : (i + 1) * stride]
                ),
                pbar_desc="train",
            ),
        )
        await model.log(val_groups)
        await model.delete_checkpoints()
        await backend._experimental_push_to_s3(model)
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=5e-5),
        )


if __name__ == "__main__":
    asyncio.run(main())
