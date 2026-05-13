from __future__ import annotations

import asyncio
import random

from dotenv import load_dotenv
from rollout import rollout
from scenarios import SCENARIOS

import art
from art.local import LocalBackend

load_dotenv()

random.seed(42)

TRAIN_STEPS = 30
SIMULTANEOUS_ROLLOUTS = 12
VALIDATION_ROLLOUTS = 3


async def train() -> None:
    backend = LocalBackend()

    model = art.TrainableModel(
        name="art-e-email-search-001",
        project="art-e",
        base_model="Qwen/Qwen2.5-3B-Instruct",
    )

    await model.register(backend)

    scenarios = list(SCENARIOS)
    for step in range(await model.get_step(), TRAIN_STEPS):
        random.shuffle(scenarios)

        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario, step=step, is_validation=False)
                    for _ in range(SIMULTANEOUS_ROLLOUTS)
                )
                for scenario in scenarios
            ),
            pbar_desc="train",
            max_exceptions=10,
        )

        val_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario, step=step, is_validation=True)
                    for _ in range(VALIDATION_ROLLOUTS)
                )
                for scenario in SCENARIOS
            ),
            pbar_desc="val",
            max_exceptions=10,
        )

        await model.log(val_groups)
        await model.delete_checkpoints()
        result = await backend.train(model, train_groups, learning_rate=1e-5)
        await model.log(
            train_groups,
            metrics=result.metrics,
            step=result.step,
            split="train",
        )


if __name__ == "__main__":
    asyncio.run(train())
