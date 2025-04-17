import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

from .load_benchmarked_models import load_benchmarked_models
from .types import BenchmarkedModelKey
from ..output_dirs import get_benchmarks_dir

# returns an array of paths to image files, one for each metric
def generate_line_graphs(
    project: str,
    line_graph_keys: list[BenchmarkedModelKey],
    comparison_keys: list[BenchmarkedModelKey],
    metrics: list[str] = ["reward"],
    api_path: str = "./.art"
) -> list[str]:
    benchmarks_dir = get_benchmarks_dir(project, api_path)
    os.makedirs(benchmarks_dir, exist_ok=True)

    line_graph_models = load_benchmarked_models(project, line_graph_keys, metrics, api_path)
    comparison_models = load_benchmarked_models(project, comparison_keys, metrics, api_path)
    image_paths = []

    for metric in metrics:
        plt.figure()  # Create a new figure for each metric
        for model in line_graph_models:
            steps = [step.index for step in model.steps]
            values = [step.metrics.get(metric, float('nan')) for step in model.steps]
            label = f"{model.model_key.model} {model.model_key.split}"
            plt.plot(steps, values, label=label)

            # Add a dot only at the last point
            if steps and values:
                plt.scatter(steps[-1], values[-1], s=10)

        for model in comparison_models:
            last_step = model.steps[-1]
            # draw horizontal black dashed line at the last step's value
            plt.axhline(y=last_step.metrics[metric], color='black', linestyle='--')
            plt.text(steps[-1], last_step.metrics[metric], f"{model.model_key.model} {model.model_key.split}",
                     ha='right', va='bottom', fontsize=8, color='black')

        plt.title(metric)
        plt.xlabel("Step")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(axis='y', color='lightgray', linestyle='--', linewidth=0.25)

        # 2025-04-17_22:09:57.865_reward_line_graph.png
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]
        metric_graph_path = os.path.join(benchmarks_dir, f"{current_time}_{metric}_line_graph.png")
        plt.savefig(metric_graph_path)
        plt.close()
        image_paths.append(metric_graph_path)

    return image_paths

