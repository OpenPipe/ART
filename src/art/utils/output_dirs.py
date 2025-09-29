import os

from art.model import Model
from art.utils.get_repo_root_path import get_repo_root_path


def get_default_art_path() -> str:
    root_path = get_repo_root_path()
    return os.path.join(root_path, ".art")


def get_models_dir(project_name: str, art_path: str | None = None) -> str:
    if art_path is None:
        art_path = get_default_art_path()
    return f"{art_path}/{project_name}/models"


def get_model_dir(model: Model, art_path: str | None = None) -> str:
    if art_path is None:
        art_path = get_default_art_path()
    return f"{art_path}/{model.project}/models/{model.name}"


def get_output_dir_from_model_properties(
    project: str, name: str, art_path: str | None = None
) -> str:
    if art_path is None:
        art_path = get_default_art_path()
    return f"{art_path}/{project}/models/{name}"


def get_step_checkpoint_dir(model_output_dir: str, step: int) -> str:
    return f"{model_output_dir}/checkpoints/{step:04d}"


def get_trajectories_dir(model_output_dir: str) -> str:
    return f"{model_output_dir}/trajectories"


def get_trajectories_split_dir(model_output_dir: str, split: str) -> str:
    return f"{model_output_dir}/trajectories/{split}"
