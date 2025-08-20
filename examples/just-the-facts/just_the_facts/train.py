import asyncio
import os

from dotenv import load_dotenv
from rollout import rollout

import art
from art.skypilot.backend import SkyPilotBackend
from art.utils import iterate_dataset
from just_the_facts.experiments import JustTheFactsConfig, models
from just_the_facts.scenarios import train_scenarios, val_scenarios

load_dotenv()


async def train(model: art.TrainableModel[JustTheFactsConfig]):
    backend = SkyPilotBackend.initialize_cluster(
        cluster_name="just-the-facts",
        gpu="H100-SXM",
        tail_logs=False,
        env_path="../../.env",
    )

    print(f"Pulling latest checkpoint from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
    await backend._experimental_pull_from_s3(
        model,
        s3_bucket=os.environ["BACKUP_BUCKET"],
        verbose=True,
        only_step="latest",  # Only pull the latest checkpoint
        exclude=["trajectories"],  # Exclude trajectories to save space/time
    )

    await model.register(backend)

    print(f"Training data size: {len(train_scenarios)}")
    print(f"Validation data size: {len(val_scenarios)}")

    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=model.config.groups_per_step,
        num_epochs=model.config.num_epochs,
        initial_step=await model.get_step(),
    )

    for batch in train_iterator:
        if batch.step % model.config.eval_steps == 0:
            print(f"\n--- Evaluating at Iteration {batch.step} ---")

            val_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        (
                            rollout(model, scenario)
                            for _ in range(model.config.trajectories_per_group)
                        )
                    )
                    for scenario in val_scenarios
                ),
            )

            await model.log(val_groups, split="val")

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(model.config.trajectories_per_group)
                    )
                )
                for scenario in batch.items
            ),
        )

        await model.train(
            groups,
            config=art.TrainConfig(learning_rate=model.config.learning_rate),
        )

    await backend._experimental_push_to_s3(model)
    print("Training finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Name of the model to train")
    args = parser.parse_args()

    model = models[args.model]

    asyncio.run(train(model))
