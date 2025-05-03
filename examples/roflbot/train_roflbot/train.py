import art
import asyncio
from dotenv import load_dotenv
import os
from typing import List
from rollout import rollout
from dataset import get_jokes, RedditJoke
from project_types import PolicyConfig
from art.utils import iterate_dataset
from art.local import LocalBackend
from benchmark import benchmark_model
import argparse
import json

load_dotenv()


async def run_training(model: art.TrainableModel):
    assert isinstance(model.config, PolicyConfig)
    if model.config.training_config is None:
        raise ValueError("Training config is not set for the model")

    training_config = model.config.training_config
    backend = LocalBackend()

    backup_bucket = os.environ.get("REMOTE_BUCKET")
    if backup_bucket:
        await backend._experimental_pull_from_s3(
            model,
            s3_bucket=backup_bucket,
            verbose=True,
        )
    await model.register(backend)

    print("Loading training data...")
    train_scenarios: List[RedditJoke] = get_jokes(
        split="train", limit=training_config.training_dataset_size
    )
    print("Loading validation data...")
    # Using 'test' split for validation as per typical dataset splits
    val_scenarios: List[RedditJoke] = get_jokes(
        split="test", limit=training_config.val_set_size
    )

    print(f"Training data size: {len(train_scenarios)}")
    print(f"Validation data size: {len(val_scenarios)}")

    initial_step = await model.get_step()
    print(f"Starting training from step: {initial_step}")

    train_iterator = iterate_dataset(
        train_scenarios,
        groups_per_step=training_config.groups_per_step,
        num_epochs=training_config.num_epochs,
        initial_step=initial_step,
    )

    for batch, epoch, global_step, epoch_step in train_iterator:
        # Evaluation Step
        if global_step > 0 and global_step % training_config.eval_steps == 0:
            print(
                f"--- Evaluating at Iteration {global_step} (Epoch {epoch}, Step {epoch_step}) ---"
            )
            await benchmark_model(model)
            await model.delete_checkpoints()
            if backup_bucket:
                await backend._experimental_push_to_s3(
                    model, prefix="roflbot", s3_bucket=backup_bucket
                )

        # Training Step
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(training_config.trajectories_per_group)
                    )
                )
                for scenario in batch
            )
        )

        await model.train(
            groups,
            config=art.TrainConfig(learning_rate=training_config.learning_rate),
        )

    await benchmark_model(model)
    if backup_bucket:
        await backend._experimental_push_to_s3(
            model, prefix="roflbot", s3_bucket=backup_bucket
        )
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_json", help="JSON string serialization of the Model")
    args = parser.parse_args()

    print("Model JSON: ", args.model_json)

    model_dict = json.loads(args.model_json)
    model = art.TrainableModel(**model_dict)
    model.config = PolicyConfig(**model_dict["config"])
    asyncio.run(run_training(model))
