import asyncio
import os

from dotenv import load_dotenv
from rollout import rollout
import weave

import art
from art.utils import iterate_dataset
from just_the_facts.experiments import JustTheFactsConfig, models
from just_the_facts.scenarios import train_scenarios, val_scenarios

load_dotenv()

os.environ["WEAVE_LOG_LEVEL"] = "CRITICAL"

weave.init(project_name="just-the-facts")


async def train(model: art.TrainableModel[JustTheFactsConfig]):
    from art.local import LocalBackend

    backend = LocalBackend()

    print(f"Pulling latest checkpoint from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
    await backend._experimental_pull_from_s3(
        model,
        s3_bucket=os.environ["BACKUP_BUCKET"],
        verbose=True,
        only_step="latest",  # Only pull the latest checkpoint
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
                    art.TrajectoryGroup((rollout(model, scenario) for _ in range(2)))
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

        result = await backend.train(
            model,
            groups,
            learning_rate=model.config.learning_rate,
            scale_rewards=model.config.scale_rewards,
        )
        await model.log(groups, metrics=result.metrics, step=result.step, split="train")

        await backend._experimental_push_to_s3(model)

    print("Training finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Name of the model to train")
    args = parser.parse_args()

    model = models[args.model]

    asyncio.run(train(model=model))
