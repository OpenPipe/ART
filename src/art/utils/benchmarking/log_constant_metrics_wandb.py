"""Utilities for logging constant baseline metrics to Weights & Biases."""

import art
import wandb


async def log_constant_metrics_wandb(
    model: art.Model,
    num_steps: int,
    split: str,
    metrics: dict[str, float],
) -> None:
    """
    Log constant metrics to W&B as horizontal lines across all training steps.

    Creates a W&B run and logs the same values at every step from 0 to
    `num_steps`, producing horizontal reference lines on charts. Useful for
    comparing training curves against static baselines.

    Parameters
    ----------
    model : art.Model
        The model whose `project` and `name` are used for the W&B run.
    num_steps : int
        Total training steps. Metrics are logged at steps 0 through `num_steps`.
    split : str
        Data split name (e.g., "val"). Used as prefix: "{split}/{metric_name}".
    metrics : dict[str, float]
        Metric names mapped to their constant values.
    """
    run = wandb.init(
        project=model.project,
        name=model.name,
        reinit="create_new",
    )

    # Prefix metrics with split
    prefixed_metrics = {f"{split}/{key}": value for key, value in metrics.items()}

    # Log at every step to create a horizontal line
    for step in range(num_steps + 1):
        run.log(prefixed_metrics, step=step)

    run.finish()
