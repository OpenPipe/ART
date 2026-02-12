#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "openpipe-art[backend]",
# ]
#
# [tool.uv.sources]
# openpipe-art = { path = "../../../", editable = true }
# ///

import asyncio
import os
import tempfile
import uuid

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.dev import TrainConfig as DevTrainConfig
from art.local import LocalBackend
from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir

# Define base trajectory outside of tests
BASE_TRAJECTORY = Trajectory(
    messages_and_choices=[
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi there!",
        },
    ],
    reward=1.0,
)

train_config = DevTrainConfig(allow_training_without_logprobs=True)


async def test_step_skipping():
    """Test that step counting works correctly when training is skipped."""

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up backend with custom art path
        art_path = os.path.join(tmpdir, ".art")

        with LocalBackend(path=art_path) as backend:
            # Create a test model
            model = TrainableModel(
                name=f"test-step-skip-{uuid.uuid4()}",
                project="test-project",
                base_model="Qwen/Qwen2.5-0.5B-Instruct",  # Small model for testing
            )

            # Register the model
            await model.register(
                backend,
                _openai_client_config={"engine_args": {"enforce_eager": True}},
            )

            # Check initial step is 0
            initial_step = await model.get_step()
            assert initial_step == 0, f"Initial step should be 0, got {initial_step}"

            print(f"Initial step: {initial_step}")

            # Check initial filesystem state
            model_dir = get_model_dir(model, art_path)
            checkpoints_dir = os.path.join(model_dir, "checkpoints")

            # After registration, checkpoint 0000 should exist
            checkpoint_0 = get_step_checkpoint_dir(model_dir, 0)
            assert os.path.exists(checkpoint_0), (
                f"Initial checkpoint {checkpoint_0} should exist"
            )
            print(f"✓ Found initial checkpoint: {checkpoint_0}")

            # Test 1: Train with multiple groups - some trainable, some not
            print(
                "\nTest 1: Training with multiple groups (3 trainable, 2 not trainable)..."
            )
            # Group 1: trainable (different rewards)
            group1 = TrajectoryGroup(
                [
                    BASE_TRAJECTORY,
                    BASE_TRAJECTORY.model_copy(update={"reward": 0.5}),
                ]
            )
            # Group 2: not trainable (same rewards)
            group2 = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": 2.0}),
                    BASE_TRAJECTORY.model_copy(update={"reward": 2.0}),
                ]
            )
            # Group 3: trainable (different rewards)
            group3 = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": 1.5}),
                    BASE_TRAJECTORY.model_copy(update={"reward": 3.0}),
                ]
            )
            # Group 4: not trainable (same rewards)
            group4 = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": 0.0}),
                    BASE_TRAJECTORY.model_copy(update={"reward": 0.0}),
                ]
            )
            # Group 5: trainable (different rewards)
            group5 = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": -1.0}),
                    BASE_TRAJECTORY.model_copy(update={"reward": 1.0}),
                ]
            )

            await model.train(
                [group1, group2, group3, group4, group5],
                _config=train_config,
                verbose=True,
            )

            step_after_train = await model.get_step()
            assert step_after_train == 1, (
                f"Step after training should be 1, got {step_after_train}"
            )
            print(f"Step after training: {step_after_train}")

            # Check that checkpoint 0001 was created
            checkpoint_1 = get_step_checkpoint_dir(model_dir, 1)
            assert os.path.exists(checkpoint_1), (
                f"Checkpoint {checkpoint_1} should exist after training"
            )
            assert os.path.exists(checkpoint_0), (
                f"Previous checkpoint {checkpoint_0} should still exist"
            )
            print(f"✓ Found checkpoint after training: {checkpoint_1}")

            # Test 2: Try to train with all non-trainable groups - should skip but still increment step
            print(
                "\nTest 2: Training with all non-trainable groups (0 trainable, 3 not trainable)..."
            )
            # All groups have same rewards within group
            group1_skip = TrajectoryGroup(
                [
                    BASE_TRAJECTORY,
                    BASE_TRAJECTORY.model_copy(),  # Same reward
                ]
            )
            group2_skip = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": 5.0}),
                    BASE_TRAJECTORY.model_copy(update={"reward": 5.0}),
                ]
            )
            group3_skip = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": -2.0}),
                    BASE_TRAJECTORY.model_copy(update={"reward": -2.0}),
                ]
            )

            await model.train(
                [group1_skip, group2_skip, group3_skip],
                _config=train_config,
                verbose=True,
            )

            step_after_skip = await model.get_step()
            assert step_after_skip == 2, (
                f"Step after skipping should be 2, got {step_after_skip}"
            )
            print(f"Step after skipping: {step_after_skip}")

            # Check that checkpoint 0002 exists (renamed from 0001)
            checkpoint_2 = get_step_checkpoint_dir(model_dir, 2)
            assert os.path.exists(checkpoint_2), (
                f"Checkpoint {checkpoint_2} should exist after skipping"
            )
            assert not os.path.exists(checkpoint_1), (
                f"Checkpoint {checkpoint_1} should have been renamed"
            )
            assert os.path.exists(checkpoint_0), (
                f"Checkpoint {checkpoint_0} should still exist"
            )
            print(f"✓ Found renamed checkpoint after skip: {checkpoint_2}")
            print("✓ Verified checkpoint 0001 was renamed (no longer exists)")

            # Test 3: Train again with mix of trainable and non-trainable
            print(
                "\nTest 3: Training with mixed groups (2 trainable, 2 not trainable)..."
            )
            # Mix of trainable and non-trainable groups
            group1_final = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": 2.0}),
                    BASE_TRAJECTORY.model_copy(
                        update={"reward": 1.5}
                    ),  # Different - trainable
                ]
            )
            group2_final = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": 0.5}),
                    BASE_TRAJECTORY.model_copy(
                        update={"reward": 0.5}
                    ),  # Same - not trainable
                ]
            )
            group3_final = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": 10.0}),
                    BASE_TRAJECTORY.model_copy(
                        update={"reward": -10.0}
                    ),  # Different - trainable
                ]
            )
            group4_final = TrajectoryGroup(
                [
                    BASE_TRAJECTORY.model_copy(update={"reward": 3.14}),
                    BASE_TRAJECTORY.model_copy(
                        update={"reward": 3.14}
                    ),  # Same - not trainable
                ]
            )

            await model.train(
                [group1_final, group2_final, group3_final, group4_final],
                _config=train_config,
                verbose=True,
            )

            final_step = await model.get_step()
            assert final_step == 3, f"Final step should be 3, got {final_step}"
            print(f"Final step: {final_step}")

            # Check that checkpoint 0003 was created
            checkpoint_3 = get_step_checkpoint_dir(model_dir, 3)
            assert os.path.exists(checkpoint_3), (
                f"Checkpoint {checkpoint_3} should exist after final training"
            )
            assert os.path.exists(checkpoint_2), (
                f"Previous checkpoint {checkpoint_2} should still exist"
            )
            assert os.path.exists(checkpoint_0), (
                f"Initial checkpoint {checkpoint_0} should still exist"
            )
            print(f"✓ Found checkpoint after final training: {checkpoint_3}")

            # List all checkpoints to verify
            all_checkpoints = sorted(os.listdir(checkpoints_dir))
            print(f"\n📁 All checkpoints in {checkpoints_dir}:")
            for cp in all_checkpoints:
                print(f"  - {cp}")

            # Verify we have exactly the expected checkpoints
            expected_checkpoints = ["0000", "0002", "0003"]
            assert all_checkpoints == expected_checkpoints, (
                f"Expected {expected_checkpoints}, got {all_checkpoints}"
            )

            print("\n✅ All tests passed!")
            print(
                "Successfully tested step progression: 0 -> 1 (train) -> 2 (skip) -> 3 (train)"
            )
            print("✅ Filesystem verification passed!")
            print("\n📊 Expected metrics logged:")
            print("  Step 1: num_groups_submitted=5, num_groups_trainable=3")
            print("  Step 2: num_groups_submitted=3, num_groups_trainable=0")
            print("  Step 3: num_groups_submitted=4, num_groups_trainable=2")


if __name__ == "__main__":
    asyncio.run(test_step_skipping())
