import os
import shutil


def clear_iteration_dirs(output_dir: str, excluding: list[int]) -> None:
    for dir in os.listdir(output_dir):
        if (
            os.path.isdir(os.path.join(output_dir, dir))
            and dir.isdigit()
            and int(dir) not in excluding
        ):
            iteration_dir = os.path.join(output_dir, dir)
            shutil.rmtree(iteration_dir)
            print(f"Deleted iteration directory {iteration_dir}")


def get_iteration(output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    return max(
        (
            int(subdir)
            for subdir in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, subdir)) and subdir.isdigit()
        ),
        default=0,
    )


def get_last_iteration_dir(output_dir: str) -> str | None:
    last_iteration_dir = os.path.join(output_dir, f"{get_iteration(output_dir):04d}")
    return last_iteration_dir if os.path.exists(last_iteration_dir) else None
