import asyncio
import os
import random

from dotenv import load_dotenv

import art
from art.skypilot import SkyPilotBackend
from just_the_facts.rollout import FactsScenario, rollout
from just_the_facts.scenarios import val_urls

load_dotenv()

random.seed(42)

# Initialize the server
backend = None


# comparison models
gpt_4o_mini = art.Model(
    name="gpt-4o-mini",
    project="just-the-facts",
    inference_model_name="openai/gpt-4o-mini",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)
gpt_4o = art.Model(
    name="gpt-4o",
    project="just-the-facts",
    inference_model_name="openai/gpt-4o",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)
gpt_4_1 = art.Model(
    name="gpt-4.1",
    project="just-the-facts",
    inference_model_name="openai/gpt-4.1",
    inference_base_url="https://openrouter.ai/api/v1",
    inference_api_key=os.getenv("OPENROUTER_API_KEY"),
)


async def log_comparison_model(comparison_model: art.Model):
    scenarios = [FactsScenario(article_url=url) for url in val_urls]

    trajectory_groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(rollout(comparison_model, scenario) for _ in range(4))
            for scenario in scenarios
        ),
        pbar_desc=f"gather {comparison_model.name}",
        max_exceptions=1,
    )

    await comparison_model.log(
        trajectory_groups,
        split="val",
    )
    await backend._experimental_push_to_s3(
        comparison_model,
    )


async def run_benchmarks():
    global backend
    backend = await SkyPilotBackend.initialize_cluster(
        cluster_name="just-the-facts", gpu="H100-SXM"
    )
    await gpt_4o_mini.register(backend)
    await gpt_4o.register(backend)
    await gpt_4_1.register(backend)

    promises = []

    for comparison_model in [
        gpt_4o_mini,
        # gpt_4o,
        # gpt_4_1,
    ]:
        promises.append(log_comparison_model(comparison_model))

    await asyncio.gather(*promises)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
