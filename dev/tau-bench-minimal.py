import asyncio
import itertools as it
from typing import Annotated

from dotenv import load_dotenv
import more_itertools as mit
import typer

import art
from art import tau_bench
from art.tinker import TinkerBackend

app = typer.Typer()

DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_NAME = "001"


async def train(
    base_model: str,
    name: str,
):
    load_dotenv()

    steps = 100
    groups = 1
    trajectories = 16

    backend = TinkerBackend()
    model = art.TrainableModel(
        name=name,
        project="tau-bench",
        base_model=base_model,
    )
    await model.register(backend)
    scenarios = await tau_bench.get_scenarios(domain="telecom", split="test")

    async def val() -> None:
        val_trajectories = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    tau_bench.rollout(scenario, model, max_turns=30) for _ in range(2)
                )
                for scenario in scenarios
            ),
            pbar_desc="val",
        )
        await model.log(val_trajectories)

    for step in range(await model.get_step(), steps):
        if step == 0:
            await val()
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    tau_bench.rollout(scenario, model, max_turns=30)
                    for _ in range(trajectories)
                )
                for scenario in mit.take((step + 1) * groups, it.cycle(scenarios))[
                    -groups:
                ]
            ),
            pbar_desc=f"gather({step:03d})",
        )
        await model.delete_checkpoints()
        await backend.train(model, train_groups)
        if step + 1 == steps:
            await val()


@app.command()
def main(
    base_model: Annotated[
        str,
        typer.Option(
            "--base-model",
            "-bm",
            help="Base model to train.",
        ),
    ] = DEFAULT_BASE_MODEL,
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="Trainable model name.",
        ),
    ] = DEFAULT_NAME,
):
    asyncio.run(train(base_model, name))


if __name__ == "__main__":
    app()
