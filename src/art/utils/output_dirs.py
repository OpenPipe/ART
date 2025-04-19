from art.model import Model


def get_model_dir(art_path: str, model: Model) -> str:
    return f"{art_path}/{model.project}/models/{model.name}"


def get_output_dir_from_model_properties(
    project: str, name: str, path: str = "./.art"
) -> str:
    return f"{path}/{project}/models/{name}"


def get_trajectories_split_dir(model_output_dir: str, split: str) -> str:
    return f"{model_output_dir}/trajectories/{split}"


def get_benchmarks_dir(project: str, path: str = "./.art") -> str:
    return f"{path}/{project}/benchmarks"
