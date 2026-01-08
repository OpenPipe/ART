#!/usr/bin/env python3
"""
ART-style training script for SWE-bench using a simplified rollout function.
Inspired by qwen_rollout.py but following ART idioms.
"""

import argparse
import asyncio
import os

import torch
from art_style_rollout import ARTModelConfig, art_style_rollout
from dotenv import load_dotenv
from instance_filter import filter_quality_instances
from instances import Instance, as_instances_iter, get_filtered_swe_smith_instances_df
from sandbox import new_sandbox

import art


def setup_environment():
    """Set up environment variables and directories."""
    load_dotenv()

    # Create necessary directories
    os.makedirs("replays", exist_ok=True)
    os.makedirs("trajectories", exist_ok=True)

    # Print environment info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")


async def rollout_with_sandbox(
    model: art.Model[ARTModelConfig],
    instance: Instance,
    reward_power: float = 1.0,
) -> art.Trajectory:
    """Run a single rollout with sandbox management."""
    async with new_sandbox(image=instance["image_name"], provider="daytona") as sandbox:
        # Apply the patch to break tests
        await sandbox.apply_patch(instance["patch"], timeout=10)

        # Run the rollout
        trajectory = await art_style_rollout(
            model=model,
            instance=instance,
            sandbox=sandbox,
            reward_power=reward_power,
        )

    return trajectory


async def train_inference_model(
    model_name: str = "willcb/Qwen3-32B",
    api_base: str = "http://localhost:8000/v1",
    num_instances: int = 10,
    rollouts_per_instance: int = 4,
    reward_power: float = 1.0,
    use_quality_filter: bool = True,
    require_non_zero_tests: bool = True,
):
    """Train using an inference-only model (no gradient updates)."""
    # Initialize model
    model = art.Model(
        name=model_name,
        project="swebench-art-style",
        config=ARTModelConfig(),
    )

    # Set up OpenAI client if using local inference
    if api_base:
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "default")
        os.environ["OPENAI_BASE_URL"] = api_base

    # Load instances
    all_instances_df = get_filtered_swe_smith_instances_df()

    # Apply quality filter if requested
    if use_quality_filter:
        instances_df = filter_quality_instances(
            all_instances_df, require_non_zero_tests=require_non_zero_tests
        )
        print(
            f"Filtered to {len(instances_df)} quality instances from {len(all_instances_df)} total"
        )
    else:
        instances_df = all_instances_df
        print(f"Using all {len(instances_df)} instances (no quality filter)")

    instances = list(
        instances_df.sample(n=min(num_instances, len(instances_df))).pipe(
            as_instances_iter
        )
    )

    print(f"Loaded {len(instances)} instances")

    # Collect trajectories
    all_trajectories = []

    for i, instance in enumerate(instances):
        print(f"\n{'=' * 60}")
        print(f"Instance {i + 1}/{len(instances)}: {instance['instance_id']}")
        print(f"{'=' * 60}")

        # Run multiple rollouts for this instance
        trajectory_group = await art.gather_trajectories(
            [
                rollout_with_sandbox(model, instance, reward_power)
                for _ in range(rollouts_per_instance)
            ]
        )

        # Log results
        rewards = [t.reward for t in trajectory_group]
        resolved = [t.metrics.get("resolved", False) for t in trajectory_group]

        print(f"Rewards: {rewards}")
        print(f"Resolved: {sum(resolved)}/{len(resolved)}")
        print(f"Average reward: {sum(rewards) / len(rewards):.3f}")

        all_trajectories.extend(trajectory_group)

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total trajectories: {len(all_trajectories)}")
    print(
        f"Average reward: {sum(t.reward for t in all_trajectories) / len(all_trajectories):.3f}"
    )
    print(
        f"Resolved: {sum(t.metrics.get('resolved', False) for t in all_trajectories)}/{len(all_trajectories)}"
    )

    return all_trajectories


async def train_trainable_model(
    base_model: str = "willcb/Qwen3-32B",
    model_name: str = "001",
    batch_size: int = 4,
    rollouts_per_instance: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 5e-5,
    reward_power: float = 1.33,
    max_concurrent_batches: int = 2,
    use_quality_filter: bool = True,
    require_non_zero_tests: bool = True,
):
    """Train a model with gradient updates using ART."""
    from art.local import LocalBackend

    # Initialize backend
    backend = LocalBackend()

    # Initialize trainable model
    model = art.TrainableModel(
        name=model_name,
        project="swebench-art-style",
        base_model=base_model,
        config=ARTModelConfig(),
        _internal_config=art.dev.InternalModelConfig(
            engine_args=art.dev.EngineArgs(
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=0.85,
            ),
            torchtune_args=art.dev.TorchtuneArgs(
                model="qwen3_32b",
                model_type="QWEN3",
                async_weight_syncing=True,
            ),
        ),
    )

    await model.register(backend)

    # Load instances
    all_instances_df = get_filtered_swe_smith_instances_df()

    # Apply quality filter if requested
    if use_quality_filter:
        instances_df = filter_quality_instances(
            all_instances_df, require_non_zero_tests=require_non_zero_tests
        )
        print(
            f"Filtered to {len(instances_df)} quality instances from {len(all_instances_df)} total"
        )
    else:
        instances_df = all_instances_df
        print(f"Using all {len(instances_df)} instances (no quality filter)")

    instances = list(instances_df.sample(fraction=1.0, seed=42).pipe(as_instances_iter))

    print(f"Loaded {len(instances)} instances")
    print(f"Starting training for {num_epochs} epochs")

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'=' * 60}")

        # Process instances in batches
        for batch_start in range(0, len(instances), batch_size):
            batch_end = min(batch_start + batch_size, len(instances))
            batch_instances = instances[batch_start:batch_end]

            print(
                f"\nBatch {batch_start // batch_size + 1}: instances {batch_start}-{batch_end - 1}"
            )

            # Gather trajectory groups for this batch
            trajectory_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout_with_sandbox(model, instance, reward_power)
                        for _ in range(rollouts_per_instance)
                    )
                    for instance in batch_instances
                ),
                pbar_desc=f"Batch {batch_start // batch_size + 1}",
            )

            # Log batch statistics
            all_trajectories = [t for group in trajectory_groups for t in group]
            avg_reward = sum(t.reward for t in all_trajectories) / len(all_trajectories)
            resolved_count = sum(
                t.metrics.get("resolved", False) for t in all_trajectories
            )

            print(f"Batch average reward: {avg_reward:.3f}")
            print(f"Batch resolved: {resolved_count}/{len(all_trajectories)}")

            # Train on this batch
            await model.train(
                trajectory_groups,
                config=art.TrainConfig(learning_rate=learning_rate),
                _config=art.dev.TrainConfig(allow_training_without_logprobs=True),
                verbose=True,
            )

            # Log the trajectories
            await model.log(trajectory_groups)

    print("\nTraining complete!")
    return model


async def main():
    parser = argparse.ArgumentParser(
        description="Train SWE-bench models using ART-style rollouts"
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "train"],
        default="inference",
        help="Mode: inference (no gradients) or train (with gradients)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name or path",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="API base URL for inference mode",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=10,
        help="Number of instances to use (inference mode)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--rollouts-per-instance",
        type=int,
        default=4,
        help="Number of rollouts per instance",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (train mode)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--reward-power",
        type=float,
        default=1.33,
        help="Power to apply to progress metric in reward",
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Disable quality filtering and use all instances (not recommended)",
    )
    parser.add_argument(
        "--require-non-zero-tests",
        action="store_true",
        default=True,
        help="When using quality filter, require instances to have non-zero tests",
    )

    args = parser.parse_args()

    # Set up environment
    setup_environment()

    # Run appropriate mode
    if args.mode == "inference":
        await train_inference_model(
            model_name=args.model,
            api_base=args.api_base,
            num_instances=args.num_instances,
            rollouts_per_instance=args.rollouts_per_instance,
            reward_power=args.reward_power,
            use_quality_filter=not args.no_quality_filter,
            require_non_zero_tests=args.require_non_zero_tests,
        )
    else:
        await train_trainable_model(
            base_model=args.model,
            batch_size=args.batch_size,
            rollouts_per_instance=args.rollouts_per_instance,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            reward_power=args.reward_power,
            use_quality_filter=not args.no_quality_filter,
            require_non_zero_tests=args.require_non_zero_tests,
        )


if __name__ == "__main__":
    asyncio.run(main())
