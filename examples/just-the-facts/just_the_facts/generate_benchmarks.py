import asyncio
import copy
import os
import random

from dotenv import load_dotenv

import art
from art.local import LocalBackend
from just_the_facts.rollout import rollout
from just_the_facts.scenarios import val_scenarios

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

gpt_4o = copy.deepcopy(gpt_4o_mini)
gpt_4o.name = "gpt-4o"
gpt_4o.inference_model_name = "openai/gpt-4o"

gpt_4_1 = copy.deepcopy(gpt_4o_mini)
gpt_4_1.name = "gpt-4.1"
gpt_4_1.inference_model_name = "openai/gpt-4.1"

grok_3 = copy.deepcopy(gpt_4o_mini)
grok_3.name = "grok-3"
grok_3.inference_model_name = "x-ai/grok-3"

grok_4 = copy.deepcopy(gpt_4o_mini)
grok_4.name = "grok-4"
grok_4.inference_model_name = "x-ai/grok-4"

sonnet_4 = copy.deepcopy(gpt_4o_mini)
sonnet_4.name = "sonnet-4"
sonnet_4.inference_model_name = "anthropic/claude-sonnet-4"

gemini_2_5_pro = copy.deepcopy(gpt_4o_mini)
gemini_2_5_pro.name = "gemini-2.5-pro"
gemini_2_5_pro.inference_model_name = "google/gemini-2.5-pro"

r1_0528 = copy.deepcopy(gpt_4o_mini)
r1_0528.name = "r1-0528"
r1_0528.inference_model_name = "deepseek/deepseek-r1-0528"


async def log_comparison_model(comparison_model: art.Model):
    trajectory_groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(rollout(comparison_model, scenario) for _ in range(2))
            for scenario in val_scenarios
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
    backend = LocalBackend()
    await gpt_4o_mini.register(backend)
    await gpt_4o.register(backend)
    await gpt_4_1.register(backend)
    await grok_3.register(backend)
    await grok_4.register(backend)
    await sonnet_4.register(backend)
    await gemini_2_5_pro.register(backend)
    await r1_0528.register(backend)

    promises = []

    for comparison_model in [
        gpt_4o_mini,
        gpt_4o,
        gpt_4_1,
        grok_3,
        grok_4,
        sonnet_4,
        gemini_2_5_pro,
        r1_0528,
    ]:
        promises.append(log_comparison_model(comparison_model))

    await asyncio.gather(*promises)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
