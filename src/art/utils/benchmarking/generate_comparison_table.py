import pandas as pd

from .load_benchmarked_models import load_benchmarked_models
from .types import BenchmarkedModelKey, BenchmarkedModel

def generate_comparison_table(
    project: str,
    benchmark_keys: list[BenchmarkedModelKey],
    metrics: list[str] = ["reward"],
    api_path: str = "./.art"
) -> pd.DataFrame:
    benchmarked_models = load_benchmarked_models(project, benchmark_keys, metrics, api_path)

    rows = []

    for benchmarked_model in benchmarked_models:
        for step in benchmarked_model.steps:
            row = {
                "Model": benchmarked_model.model_key.model,
                "Split": benchmarked_model.model_key.split,
                "Step": f"{step.index:04d}"
            }
            for metric in metrics:
                row[metric] = step.metrics.get(metric, "N/A")
            rows.append(row)

    return pd.DataFrame(rows, columns=["Model", "Split", "Step"] + metrics)
