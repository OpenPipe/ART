import os
import shutil

from art.utils.get_model_step import get_step_from_dir


def delete_checkpoints(output_dir: str, excluding: list[int]) -> None:
    checkpoint_base_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_base_dir):
        return

    for dir in os.listdir(checkpoint_base_dir):
        if (
            os.path.isdir(os.path.join(checkpoint_base_dir, dir))
            and dir.isdigit()
            and int(dir) not in excluding
        ):
            checkpoint_dir = os.path.join(checkpoint_base_dir, dir)
            shutil.rmtree(checkpoint_dir)
            print(f"Deleted checkpoint {checkpoint_dir}")


def get_last_checkpoint_dir(output_dir: str) -> str | None:
    step = get_step_from_dir(output_dir)

    checkpoint_dir = os.path.join(output_dir, "checkpoints", f"{step:04d}")
    if os.path.exists(checkpoint_dir):
        return checkpoint_dir

    return None


def migrate_checkpoints_to_new_structure(output_dir: str) -> None:
    """
    Migrate existing checkpoints from the old structure to the new structure.
    Old: .art/{project}/models/{model_name}/{step}
    New: .art/{project}/models/{model_name}/checkpoints/{step}
    """
    # Create checkpoints directory if it doesn't exist
    checkpoint_base_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_base_dir, exist_ok=True)

    # Find all directories in the output_dir that are checkpoints (parseable as ints)
    migrated_count = 0
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            # This is a checkpoint directory in the old structure
            new_checkpoint_path = os.path.join(checkpoint_base_dir, item)

            # Skip if already exists in new location
            if os.path.exists(new_checkpoint_path):
                print(
                    f"Checkpoint {item} already exists in new location, skipping migration"
                )
                continue

            # Move the checkpoint to the new location
            print(f"Migrating checkpoint {item} to new structure...")
            shutil.move(item_path, new_checkpoint_path)
            migrated_count += 1

    if migrated_count > 0:
        print(f"Successfully migrated {migrated_count} checkpoint(s) to new structure")
