"""Utilities for logging constant baseline metrics to Weights & Biases."""

import wandb

import art


async def log_constant_metrics_wandb(
    model: art.Model,
    num_steps: int,
    split_metrics: dict[str, dict[str, float]],
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
    split_metrics : dict[str, dict[str, float]]
        Nested dict mapping split names (e.g., "train", "val") to metric dicts.
        Each metric is logged as "{split}/{metric_name}".

        Example: `{"train": {"loss": 0.5}, "val": {"loss": 0.4, "accuracy": 0.8}}`
    """
    run = wandb.init(
        project=model.project,
        name=model.name,
        reinit="create_new",
    )

    # Prefix metrics with their split names
    prefixed_metrics = {
        f"{split}/{key}": value
        for split, metrics in split_metrics.items()
        for key, value in metrics.items()
    }

    # Log at every step to create horizontal lines
    for step in range(num_steps + 1):
        run.log(prefixed_metrics, step=step)

    run.finish()
