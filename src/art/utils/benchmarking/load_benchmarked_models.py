import os
import copy

from art.utils.benchmarking.types import BenchmarkedModelKey, BenchmarkedModel
from art.utils.output_dirs import get_output_dir_from_model_properties, get_trajectories_split_dir
from art.utils.trajectory_logging import deserialize_trajectory_groups


def load_benchmarked_models(
    project: str,
    benchmark_keys: list[BenchmarkedModelKey],
    metrics: list[str] = ["reward"],
    api_path: str = "./.art"
) -> list[BenchmarkedModel]:

    benchmark_keys_copy = copy.deepcopy(benchmark_keys)

    benchmarked_models = []

    for benchmark_key in benchmark_keys_copy:
        benchmarked_model = BenchmarkedModel(benchmark_key)
        model_output_dir = get_output_dir_from_model_properties(project, benchmark_key.model, api_path)
        split_dir = get_trajectories_split_dir(model_output_dir, benchmark_key.split)

        if benchmark_key.iteration is None:
            # get last file in split_dir
            file_name = os.listdir(split_dir)[-1]
            benchmark_key.iteration = int(file_name.split(".")[0])

        file_path = os.path.join(split_dir, f"{benchmark_key.iteration:04d}.yaml")

        with open(file_path, "r") as f:
            trajectory_groups = deserialize_trajectory_groups(f.read())

        # add "reward" to trajectory metrics to ensure it is treated like a metric
        for trajectory_group in trajectory_groups:
            for trajectory in trajectory_group.trajectories:
                if "reward" not in trajectory.metrics:
                    trajectory.metrics["reward"] = trajectory.reward

        for metric in metrics:
            group_averages = []
            for trajectory_group in trajectory_groups:
                trajectories_with_metric = [trajectory for trajectory in trajectory_group.trajectories if metric in trajectory.metrics]
                if len(trajectories_with_metric) == 0:
                    continue
                average = sum(trajectory.metrics[metric] for trajectory in trajectories_with_metric) / len(trajectories_with_metric)
                group_averages.append(average)
            if len(group_averages) == 0:
                continue
            benchmarked_model.metrics[metric] = sum(group_averages) / len(group_averages)

        benchmarked_models.append(benchmarked_model)

    return benchmarked_models



