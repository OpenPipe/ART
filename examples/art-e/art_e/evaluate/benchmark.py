import art
from art_e.rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
import polars as pl


async def benchmark_model(
    model: art.Model, limit: int = 100, swallow_exceptions: bool = True
) -> pl.DataFrame:
    val_scenarios = load_synthetic_queries(split="test", limit=limit)
    val_trajectories = await art.gather_trajectories(
        (rollout(model, scenario) for scenario in val_scenarios),
        pbar_desc=f"validation {model.name}",
        max_exceptions=limit if swallow_exceptions else 0,
    )

    valid_trajectories = [t for t in val_trajectories if isinstance(t, art.Trajectory)]

    if model._backend is not None:
        await model.log(valid_trajectories)

    metrics = pl.DataFrame(
        [{**t.metrics, "reward": t.reward} for t in valid_trajectories]
    )

    avg_metrics = metrics.select(
        [pl.mean(c).alias(c) for c in metrics.columns]
    ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))

    return avg_metrics
