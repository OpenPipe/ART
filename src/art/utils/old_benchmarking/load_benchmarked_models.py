import os
import copy
import json

from art.utils.old_benchmarking.calculate_step_metrics import calculate_step_std_dev
from art.utils.old_benchmarking.types import (
    BenchmarkedModelStep,
    BenchmarkedModelKey,
    BenchmarkedModel,
)
from art.utils.output_dirs import (
    get_output_dir_from_model_properties,
    get_trajectories_split_dir,
)
from art.utils.trajectory_logging import deserialize_trajectory_groups


def load_benchmarked_models(
    project: str,
    benchmark_keys: list[BenchmarkedModelKey],
    metrics: list[str] = ["reward"],
    api_path: str = "./.art",
) -> list[BenchmarkedModel]:
    benchmark_keys_copy = copy.deepcopy(benchmark_keys)

    benchmarked_models = []

    for benchmark_key in benchmark_keys_copy:
        benchmarked_model = BenchmarkedModel(benchmark_key)
        model_output_dir = get_output_dir_from_model_properties(
            project=project, name=benchmark_key.model, art_path=api_path
        )
        split_dir = get_trajectories_split_dir(model_output_dir, benchmark_key.split)

        history_logs = []

        with open(os.path.join(model_output_dir, "history.jsonl"), "r") as f:
            for line in f:
                # only include logs that have a recorded_at value
                log = json.loads(line)
                if "recorded_at" in log:
                    history_logs.append(log)

        # get last file name in split_dir
        max_step_index = -1

        try:
            max_step_index = int(os.listdir(split_dir)[-1].split(".")[0])
        except Exception as e:
            print(f"Error getting max iteration index for {benchmark_key}")
            raise e

        if benchmark_key.step_indices is None:
            # load all iterations
            benchmark_key.step_indices = list(range(max_step_index + 1))

        # allow users to count backward from max_step_index using negative indices
        benchmark_key.step_indices = [
            index - 1 + max_step_index if index < 0 else index
            for index in benchmark_key.step_indices
        ]

        for index in benchmark_key.step_indices:
            step = BenchmarkedModelStep(index)

            # find the most recent log that has a step value equal to index
            for log in reversed(history_logs):
                if log["step"] == index:
                    step.recorded_at = log["recorded_at"]
                    break

            file_path = os.path.join(split_dir, f"{index:04d}.yaml")

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
                    trajectories_with_metric = [
                        trajectory
                        for trajectory in trajectory_group.trajectories
                        if metric in trajectory.metrics
                    ]
                    if len(trajectories_with_metric) == 0:
                        continue
                    average = sum(
                        trajectory.metrics[metric]
                        for trajectory in trajectories_with_metric
                    ) / len(trajectories_with_metric)
                    group_averages.append(average)
                if len(group_averages) == 0:
                    continue
                step.metrics[metric] = sum(group_averages) / len(group_averages)

            step.metrics["reward_std_dev"] = calculate_step_std_dev(trajectory_groups)

            benchmarked_model.steps.append(step)

        benchmarked_models.append(benchmarked_model)

    return benchmarked_models
