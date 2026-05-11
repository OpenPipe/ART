import asyncio
import os

from dotenv import load_dotenv

import art
from art.langgraph import wrap_rollout
from art.local import LocalBackend
from art.utils import iterate_dataset

from .data import train_scenarios
from .rollout import EmailScenario, rollout

load_dotenv()


def build_model() -> art.TrainableModel:
    return art.TrainableModel(
        name=os.getenv("ART_E_MODEL_NAME", "art-e-email-agent-001"),
        project="art-e",
        base_model=os.getenv("ART_E_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        inference_model_name=os.getenv("ART_E_INFERENCE_MODEL"),
        inference_base_url=os.getenv("ART_E_INFERENCE_BASE_URL"),
        inference_api_key=os.getenv("ART_E_INFERENCE_API_KEY")
        or os.getenv("OPENAI_API_KEY"),
    )


async def main() -> None:
    backend = LocalBackend()
    model = build_model()
    await model.register(backend)

    for batch in iterate_dataset(
        train_scenarios,
        groups_per_step=2,
        num_epochs=3,
        initial_step=await model.get_step(),
    ):
        groups = [
            art.TrajectoryGroup(
                wrap_rollout(model, rollout)(
                    model,
                    EmailScenario(step=batch.step, scenario=scenario),
                )
                for _ in range(2)
            )
            for scenario in batch.items
        ]
        finished_groups = await art.gather_trajectory_groups(groups, pbar_desc="gather")
        await model.train(
            finished_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )

        if batch.step >= 5:
            break


if __name__ == "__main__":
    asyncio.run(main())
