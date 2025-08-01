import art
from art_e.rollout import rollout
from art_e.data.query_iterators import load_synthetic_queries
import polars as pl
from art_e.report_trajectory import report_trajectory
from art_e.data.types_enron import SyntheticQuery
from tqdm.asyncio import tqdm
from art_e.project_types import ProjectPolicyConfig
import uuid
from art.langgraph import wrap_rollout


async def benchmark_model(
    model: art.Model,
    limit: int = 100,
    step: int = 0,
    report: bool = True,
) -> pl.DataFrame:
    val_scenarios = load_synthetic_queries(split="test", limit=limit)

    async def rollout_and_report(scenario: SyntheticQuery):
        traj = await wrap_rollout(model, rollout)(scenario)
        if report:
            report_trajectory(model, traj, step)
        return traj

    val_trajectories = await tqdm.gather(
        *(rollout_and_report(scenario) for scenario in val_scenarios),
        desc=f"validation {model.name}",
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


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(
        benchmark_model(
            art.Model(
                name="openai/gpt-4.1",
                project="email_agent",
                config=ProjectPolicyConfig(litellm_model_name="openai/gpt-4.1"),
            ),
            limit=2,
        )
    )
