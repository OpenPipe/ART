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
    name="003",
    project="2048",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)
# To run on a T4, we need to override some config defaults.
model._internal_config = art.dev.InternalModelConfig(
    init_args=art.dev.InitArgs(
        max_seq_length=8192,
    ),
    engine_args=art.dev.EngineArgs(
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        num_scheduler_steps=1,
    ),
)


async def main():
    # Initialize the server
    backend = LocalBackend()

    print(f"Pulling from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
    await backend._experimental_pull_from_s3(
        model,
        verbose=True,
    )

    # Register the model with the local backend (sets up logging, inference, and training)
    await model.register(backend)

    for i in range(await model.get_step(), 100):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, i, is_validation=False) for _ in range(48)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=10,
        )
        await model.delete_checkpoints()
        await backend._experimental_push_to_s3(
            model,
        )
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=3e-5),
            # Lowering the logprob_calculation_chunk_size is a memory saving measure
            # to allow longer sequences (up to 4096 tokens) to be processed on a T4.
            _config={"logprob_calculation_chunk_size": 8},
        )


if __name__ == "__main__":
    asyncio.run(main())
