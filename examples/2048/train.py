import asyncio
import random

from dotenv import load_dotenv
from rollout import rollout

import art
from art.local import LocalBackend
from art.rewards import ruler_score_group

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
ENABLE_RULER = True


async def train():
    # Initialize the server
    backend = LocalBackend()

    # Register the model with the local backend (sets up logging, inference, and training)
    await model.register(backend)

    await backend._experimental_pull_from_s3(
        model,
        verbose=True,
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
            after_each=lambda group: (
                ruler_score_group(
                    group,
                    "openai/o4-mini",
                    debug=True,
                    swallow_exceptions=True,  # Return None on error, filtering out the group
                )
                if ENABLE_RULER
                else None
            ),
            pbar_desc="gather",
            max_exceptions=10,
        )

        # save the model to S3
        await backend._experimental_push_to_s3(
            model,
        )

        result = await backend.train(model, train_groups, learning_rate=1e-5)
        await model.log(
            train_groups, metrics=result.metrics, step=result.step, split="train"
        )


if __name__ == "__main__":
    asyncio.run(train())
