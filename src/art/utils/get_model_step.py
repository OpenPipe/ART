import os
from typing import TYPE_CHECKING
from art.utils.output_dirs import get_model_dir

if TYPE_CHECKING:
    from art.model import TrainableModel


def get_step_from_dir(output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    return max(
        (
            int(subdir)
            for subdir in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, subdir)) and subdir.isdigit()
        ),
        default=0,
    )


def get_model_step(model: "TrainableModel", art_path: str) -> int:
    return get_step_from_dir(get_model_dir(model=model, art_path=art_path))
