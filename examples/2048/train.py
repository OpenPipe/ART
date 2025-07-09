import asyncio
import os
from dotenv import load_dotenv
import random

import art
from art.local import LocalBackend
from rollout import rollout

load_dotenv()

random.seed(42)

# Declare the model
model = art.TrainableModel(
    name="tutorial-001",
    project="2048",
    base_model="Qwen/Qwen2.5-3B-Instruct",
)


async def train():
    # Initialize the server
    backend = LocalBackend()

    print(f"Pulling from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
    await backend._experimental_pull_from_s3(
        model,
        verbose=True,
    )

    # Register the model with the local backend (sets up logging, inference, and training)
    await model.register(backend)

    # train for 40 steps
    for i in range(await model.get_step(), 40):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    # for each step, rollout 18 trajectories
                    rollout(model, i, is_validation=False)
                    for _ in range(18)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=10,
        )
        await model.delete_checkpoints()
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
