#!/usr/bin/env python3
"""
Test script for ART-style rollout implementation.
Tests a single instance to verify everything is working.
"""

import asyncio
import os

from art_style_rollout import ARTModelConfig, art_style_rollout
from dotenv import load_dotenv
from instances import as_instances_iter, get_filtered_swe_smith_instances_df
from sandbox import new_sandbox

import art


async def test_single_rollout():
    """Test a single rollout with detailed logging."""
    # Load environment
    load_dotenv()

    # Set up OpenAI client for local inference
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "default")
    os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"

    # Initialize model
    model = art.Model(
        name="willcb/Qwen3-32B",  # Use the same model as qwen_rollout
        project="swebench-art-test",
        config=ARTModelConfig(),
    )

    # Get a single instance
    instance = next(
        get_filtered_swe_smith_instances_df()
        .tail(-5)  # Skip first 5 instances
        .pipe(as_instances_iter)
    )

    print(f"Testing with instance: {instance['instance_id']}")
    print(f"Repository: {instance.get('repo', 'Unknown')}")
    print(f"Problem statement preview: {instance['problem_statement'][:200]}...")
    print("\n" + "=" * 60 + "\n")

    # Run rollout with sandbox
    async with new_sandbox(image=instance["image_name"], provider="daytona") as sandbox:
        print("Sandbox created successfully")

        # Apply patch to break tests
        print("Applying patch to break tests...")
        await sandbox.apply_patch(instance["patch"], timeout=10)

        # Verify tests are broken
        print("\nVerifying tests are broken after patch...")
        failed, passed = await sandbox.run_tests(
            instance["FAIL_TO_PASS"][:1],
            timeout=60,  # Test just one for speed
        )
        print(f"FAIL_TO_PASS after patch: {failed} failed, {passed} passed")
        assert failed > 0, "Tests should be failing after patch"

        # Run the rollout
        print("\nRunning ART-style rollout...")
        trajectory = await art_style_rollout(
            model=model,
            instance=instance,
            sandbox=sandbox,
            reward_power=1.0,
        )

        # Print results
        print("\n" + "=" * 60)
        print("ROLLOUT RESULTS")
        print("=" * 60)
        print(f"Steps taken: {trajectory.metrics['steps_taken']}")
        print(f"Reward: {trajectory.reward:.3f}")
        print(f"Progress: {trajectory.metrics['progress']:.3f}")
        print(f"Maintenance: {trajectory.metrics['maintenance']:.3f}")
        print(f"Resolved: {trajectory.metrics['resolved']}")
        print(
            f"FAIL_TO_PASS: {trajectory.metrics['passed_f2p']}/{trajectory.metrics['passed_f2p'] + trajectory.metrics['failed_f2p']} passed"
        )
        print(
            f"PASS_TO_PASS: {trajectory.metrics['passed_p2p']}/{trajectory.metrics['passed_p2p'] + trajectory.metrics['failed_p2p']} passed"
        )

        # Print some logs
        print("\nTrajectory logs (last 10):")
        for log in trajectory.logs[-10:]:
            print(f"  {log}")

        return trajectory


async def test_trajectory_group():
    """Test creating a trajectory group (multiple rollouts)."""
    load_dotenv()

    # Set up OpenAI client
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "default")
    os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"

    # Initialize model
    model = art.Model(
        name="willcb/Qwen3-32B",
        project="swebench-art-test",
        config=ARTModelConfig(max_steps=5),  # Very limited for testing
    )

    # Get an instance
    instance = next(
        get_filtered_swe_smith_instances_df().tail(-10).pipe(as_instances_iter)
    )

    print(f"Testing trajectory group with instance: {instance['instance_id']}")

    # Helper function for rollout with sandbox
    async def rollout_helper():
        async with new_sandbox(
            image=instance["image_name"], provider="daytona"
        ) as sandbox:
            await sandbox.apply_patch(instance["patch"], timeout=10)
            return await art_style_rollout(model, instance, sandbox)

    # Create trajectory group
    trajectory_group = await art.gather_trajectories(
        [rollout_helper() for _ in range(2)]  # Just 2 rollouts for testing
    )

    print(f"\nCompleted {len(trajectory_group)} rollouts")
    print(f"Rewards: {[t.reward for t in trajectory_group]}")
    print(
        f"Average reward: {sum(t.reward for t in trajectory_group) / len(trajectory_group):.3f}"
    )

    return trajectory_group


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "group":
        print("Testing trajectory group...")
        asyncio.run(test_trajectory_group())
    else:
        print("Testing single rollout...")
        asyncio.run(test_single_rollout())
