"""Aggregation utilities for trajectory data."""

import json
from pathlib import Path
from typing import Any

import polars as pl

from art.utils.benchmarking.load_trajectories import load_trajectories
from art.utils.output_dirs import get_default_art_path, get_models_dir


async def load_aggregated_trajectories(
    project_name: str,
    models: list[str] | None = None,
    metrics: list[str] | None = None,
    art_path: str | None = None,
    include_history: bool = False,
) -> pl.DataFrame:
    """
    Load trajectories and aggregate metrics at the step level.

    This function builds on top of load_trajectories to provide step-level
    aggregation similar to load_benchmarked_models, but returns a DataFrame
    instead of custom objects.

    Parameters
    ----------
    project_name : str
        Name of the project
    models : list[str] | None
        List of model names to load. If None, loads all models.
    metrics : list[str] | None
        List of metrics to aggregate. If None, aggregates all metrics.
    art_path : str | None
        Path to ART directory. If None, uses default.
    include_history : bool
        Whether to include recorded_at timestamps from history.jsonl files.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: model, split, step, metric_*, recorded_at (optional)
        Metrics are averaged first within groups, then across groups.
    """
    # Load raw trajectory data
    df = await load_trajectories(project_name, models=models, art_path=art_path)

    if df.is_empty():
        return df

    # Get all metric columns
    metric_cols = [col for col in df.columns if col.startswith("metric_")]
    if metrics:
        # Filter to requested metrics
        requested_metric_cols = [f"metric_{m}" for m in metrics]
        metric_cols = [col for col in metric_cols if col in requested_metric_cols]

    # First aggregate within groups (average trajectories in each group)
    group_agg = df.group_by(["model", "split", "step", "group_number"]).agg(
        [pl.col(col).mean() for col in metric_cols]
    )

    # Then aggregate across groups (average of group averages)
    step_agg = group_agg.group_by(["model", "split", "step"]).agg(
        [pl.col(col).mean().alias(col) for col in metric_cols]
    )

    # Calculate reward standard deviation if reward is available
    if "metric_reward" in metric_cols:
        # Calculate std dev within groups, then average across groups
        reward_std = (
            df.group_by(["model", "split", "step", "group_number"])
            .agg(pl.col("metric_reward").std().alias("group_reward_std"))
            .group_by(["model", "split", "step"])
            .agg(pl.col("group_reward_std").mean().alias("metric_reward_std_dev"))
        )
        step_agg = step_agg.join(reward_std, on=["model", "split", "step"], how="left")

    # Add history timestamps if requested
    if include_history:
        history_data = _load_history_timestamps(
            project_name, models, art_path or get_default_art_path()
        )
        if history_data:
            history_df = pl.DataFrame(history_data)
            step_agg = step_agg.join(history_df, on=["model", "step"], how="left")

    return step_agg.sort(["model", "split", "step"])


def _load_history_timestamps(
    project_name: str, models: list[str] | None, art_path: str
) -> list[dict[str, Any]]:
    """Load recorded_at timestamps from history.jsonl files."""
    history_data = []
    root = Path(get_models_dir(project_name=project_name, art_path=art_path))

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        if models is not None and model_name not in models:
            continue

        history_path = model_dir / "history.jsonl"
        if not history_path.exists():
            continue

        # Read history file and extract step timestamps
        step_timestamps = {}
        with open(history_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "recorded_at" in entry and "step" in entry:
                        # Keep the most recent timestamp for each step
                        step = entry["step"]
                        if step not in step_timestamps:
                            step_timestamps[step] = entry["recorded_at"]
                except json.JSONDecodeError:
                    continue

        # Add to history data
        for step, timestamp in step_timestamps.items():
            history_data.append(
                {"model": model_name, "step": step, "recorded_at": timestamp}
            )

    return history_data


async def load_latest_metrics(
    project_name: str,
    models: list[str] | None = None,
    split: str = "val",
    metrics: list[str] | None = None,
    art_path: str | None = None,
) -> pl.DataFrame:
    """
    Load only the latest step metrics for each model.

    This is a convenience function for comparing final model performance.

    Parameters
    ----------
    project_name : str
        Name of the project
    models : list[str] | None
        List of model names to load. If None, loads all models.
    split : str
        Which split to load (default: "val")
    metrics : list[str] | None
        List of metrics to include. If None, includes all metrics.
    art_path : str | None
        Path to ART directory. If None, uses default.

    Returns
    -------
    pl.DataFrame
        DataFrame with one row per model containing latest metrics
    """
    df = await load_aggregated_trajectories(
        project_name, models=models, metrics=metrics, art_path=art_path
    )

    if df.is_empty():
        return df

    # Filter to requested split and get latest step for each model
    latest = (
        df.filter(pl.col("split") == split)
        .group_by("model")
        .agg(
            [
                pl.col("step").max().alias("step"),
                *[
                    pl.col(col).last()
                    for col in df.columns
                    if col.startswith("metric_")
                ],
            ]
        )
    )

    return latest.sort("model")
