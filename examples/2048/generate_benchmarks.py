import asyncio
import os
import random

from dotenv import load_dotenv
from rollout import rollout

import art
from art.local import LocalBackend

load_dotenv()

random.seed(42)

# Initialize the server
backend = LocalBackend()

# comparison models
gpt_4o_mini = art.Model(
    name="gpt-4o-mini",
    project="2048",
    inference_model_name="openai/gpt-4o-mini",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)
gpt_4o = art.Model(
    name="gpt-4o",
    project="2048",
    inference_model_name="openai/gpt-4o",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)
gpt_4_1 = art.Model(
    name="gpt-4.1",
    project="2048",
    inference_model_name="openai/gpt-4.1",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)


async def log_comparison_model(comparison_model: art.Model):
    trajectories = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(
                rollout(comparison_model, 0, is_validation=True) for _ in range(12)
            )
            for _ in range(1)
        ),
        pbar_desc=f"gather {comparison_model.name}",
        max_exceptions=1,
    )

    await comparison_model.log(
        trajectories,
        split="val",
    )
    await backend._experimental_push_to_s3(
        comparison_model,
    )


async def run_benchmarks():
    await gpt_4o_mini.register(backend)
    await gpt_4o.register(backend)
    await gpt_4_1.register(backend)

    promises = []

    for comparison_model in [gpt_4o_mini, gpt_4o, gpt_4_1]:
        promises.append(log_comparison_model(comparison_model))

    await asyncio.gather(*promises)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
