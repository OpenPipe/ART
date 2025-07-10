import art
from art.local import LocalBackend
import asyncio
from dotenv import load_dotenv
from typing import List, cast
from rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
from art_e.data.types_enron import SyntheticQuery
from art_e.data.local_email_db import generate_database
from art.utils import iterate_dataset
from art_e.project_types import ProjectPolicyConfig
from art_e.evaluate.benchmark import benchmark_model
from art_e.rollout import ProjectTrajectory
import os
import statistics
from report_trajectory import report_trajectory
from group_judge import GroupJudge

load_dotenv()


async def train(model: art.TrainableModel[ProjectPolicyConfig]):
    generate_database()

    with LocalBackend() as backend:
        print(f"Pulling from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
        await backend._experimental_pull_from_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
            verbose=True,
        )
        await model.register(backend)

        print("Loading training data...")
        # Load the training data with deterministic shuffling if a seed is provided.
        seed = model.config.training_dataset_seed
        train_scenarios: List[SyntheticQuery] = load_synthetic_queries(
            split="train",
            limit=model.config.training_dataset_size,
            seed=seed,
        )
        print("Loading validation data...")
        val_scenarios: List[SyntheticQuery] = load_synthetic_queries(
            split="test", limit=model.config.val_set_size
        )

        print(f"Training data size: {len(train_scenarios)}")
        print(f"Validation data size: {len(val_scenarios)}")

        train_iterator = iterate_dataset(
            train_scenarios,
            groups_per_step=model.config.groups_per_step,
            num_epochs=model.config.num_epochs,
            initial_step=await model.get_step(),
        )

        if model.config.group_judge_model is not None:
            judge_model = model.config.group_judge_model

            if model.config.group_judge_model in ["self", "base_model"]:
                judge_model = model

            group_judge = GroupJudge(
                project=model.project,
                judge_model=judge_model,
                judge_model_use_base_model=judge_model == "base_model",
            )

        for batch in train_iterator:
            if batch.step % model.config.eval_steps == 0:
                print(f"\n--- Evaluating at Iteration {batch.step} ---")
                await benchmark_model(model, step=batch.step)
                await model.delete_checkpoints()
                await backend._experimental_push_to_s3(
                    model,
                    s3_bucket=os.environ["BACKUP_BUCKET"],
                )

            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        (
                            rollout(model, scenario)
                            for _ in range(model.config.trajectories_per_group)
                        )
                    )
                    for scenario in batch.items
                )
            )

            # Optionally rescore each trajectory group with the LLM-judge before training.
            if model.config.use_judge_group_variant is not None:
                judge_tasks = [
                    group_judge.judge(
                        cast(list[ProjectTrajectory], g.trajectories),
                    )
                    for g in groups
                ]

                results = await asyncio.gather(*judge_tasks, return_exceptions=True)

                # Determine which groups succeeded.
                successful_groups = []
                for grp_idx, (g, res) in enumerate(zip(groups, results)):
                    if isinstance(res, Exception):
                        print(
                            f"WARNING:JUDGE_GROUP_FAILED group={grp_idx} step={batch.step}: {res!r}",
                            flush=True,
                        )
                    else:
                        successful_groups.append(g)

                # Replace `groups` with the subset that passed judgement so
                # that training only uses trajectories with judge rewards.
                groups = successful_groups

                for g in groups:
                    for t in g.trajectories:
                        report_trajectory(model, t, batch.step)

                # If every group failed, skip this training step entirely.
                if not groups:
                    print(
                        f"WARNING:ALL_JUDGE_GROUPS_FAILED step={batch.step}; skipping training step",
                        flush=True,
                    )
                    continue  # Proceed to next batch/epoch without training.

            # Drop groups with reward standard deviation below threshold
            if (
                model.config.minimum_reward_std_dev is not None
                and model.config.minimum_reward_std_dev > 0
            ):
                filtered_groups = []
                for grp_idx, g in enumerate(groups):
                    rewards = [t.reward for t in g.trajectories]
                    if len(rewards) < 2:
                        std_dev = 0.0
                    else:
                        std_dev = statistics.pstdev(rewards)
                    if std_dev < model.config.minimum_reward_std_dev:
                        print(
                            f"WARNING:REWARD_STD_DEV_TOO_LOW group={grp_idx} step={batch.step} stddev={std_dev:.4f}; dropping group",
                            flush=True,
                        )
                        continue
                    filtered_groups.append(g)

                # Replace groups with only those meeting the std dev threshold
                groups = filtered_groups

                # If every group failed the std dev filter, skip this training step
                if not groups:
                    print(
                        f"WARNING:ALL_GROUPS_DROPPED_LOW_STD_DEV step={batch.step}; skipping training step",
                        flush=True,
                    )
                    continue  # Proceed to next batch/epoch without training.

            await model.train(
                groups,
                config=art.TrainConfig(learning_rate=model.config.learning_rate),
                _config=art.dev.TrainConfig(
                    allow_training_without_logprobs=True
                    if model.config.messages_only
                    else False
                ),
            )

        await benchmark_model(model, step=batch.step)
        await backend._experimental_push_to_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
        )
        print("Training finished.")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("model_json", help="JSON string serialization of the Model")
    args = parser.parse_args()

    print("Model JSON: ", args.model_json)

    model_dict = json.loads(args.model_json)
    model_dict["config"] = ProjectPolicyConfig(**model_dict["config"])
    model: art.TrainableModel[ProjectPolicyConfig] = art.TrainableModel(
        **model_dict,
    )
    asyncio.run(train(model))
