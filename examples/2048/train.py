from dotenv import load_dotenv
import asyncio
import random

import art
from art.local import LocalBackend
from rollout import rollout
from group_judge import GroupJudge

load_dotenv()

random.seed(42)

# Declare the model
model = art.TrainableModel(
    name="tutorial-001",
    project="2048",
    base_model="Qwen/Qwen2.5-3B-Instruct",
)
model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(
        max_seq_length=8192,
    ),
)

TRAIN_STEPS = 40
SIMULTANEOUS_GAMES = 18
ENABLE_GROUP_JUDGE = True


async def train():
    # Initialize the server
    backend = LocalBackend()

    # Register the model with the local backend (sets up logging, inference, and training)
    await model.register(backend)

    await backend._experimental_pull_from_s3(
        model,
        verbose=True,
    )

    if ENABLE_GROUP_JUDGE:
        group_judge = GroupJudge(
            project=model.project,
            judge_model="openai/o4-mini",
        )

    # train for 40 steps
    for i in range(await model.get_step(), TRAIN_STEPS):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    # for each step, rollout 18 trajectories
                    rollout(model, i, is_validation=False)
                    for _ in range(SIMULTANEOUS_GAMES)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=10,
        )

        if ENABLE_GROUP_JUDGE:
            for group in train_groups:
                group.trajectories = await group_judge.judge(group.trajectories)

        # save the model to S3
        await backend._experimental_push_to_s3(
            model,
        )

        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )


if __name__ == "__main__":
    asyncio.run(train())
