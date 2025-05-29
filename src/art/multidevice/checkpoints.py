import os
import shutil


def delete_checkpoints(output_dir: str, excluding: list[int]) -> None:
    for dir in os.listdir(output_dir):
        if (
            os.path.isdir(os.path.join(output_dir, dir))
            and dir.isdigit()
            and int(dir) not in excluding
        ):
            checkpoint_dir = os.path.join(output_dir, dir)
            shutil.rmtree(checkpoint_dir)
            print(f"Deleted checkpoint {checkpoint_dir}")


def get_last_checkpoint_dir(output_dir: str) -> str | None:
    last_checkpoint_dir = os.path.join(output_dir, f"{get_step(output_dir):04d}")
    return last_checkpoint_dir if os.path.exists(last_checkpoint_dir) else None


def get_step(output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    return max(
        (
            int(subdir)
            for subdir in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, subdir)) and subdir.isdigit()
        ),
        default=0,
    )
