# To run:
# uv run scripts/benchmark_prompted_models.py

import art
import asyncio
import polars as pl
from art.local import LocalBackend
from dotenv import load_dotenv
from art_e.data.local_email_db import generate_database
from art_e.project_types import ProjectPolicyConfig
from art_e.evaluate.benchmark import benchmark_model
import os

load_dotenv()
generate_database()

# Turn this on to trace rollouts to weave.
# weave.init(project_name="email_agent")

MODELS_TO_BENCHMARK = [
    # ("gpt-4o", "openai/gpt-4o"),
    # ("gpt-4.1-mini", "openai/gpt-4.1-mini"),
    # ("gpt-4.1", "openai/gpt-4.1"),
    # ("o4-mini", "openai/o4-mini"),
    # ("o3", "openai/o3"),
    # ("qwen3-235b", "openrouter/qwen/qwen3-235b-a22b"),
    # ("qwen3-32b", "openrouter/qwen/qwen3-32b"),
    # ("deepseek-r1", "openrouter/deepseek-ai/DeepSeek-R1"),
    # ("gemini-2.5-flash", "gemini/gemini-2.5-flash"),
    # ("sonnet 4", "openrouter/anthropic/claude-sonnet-4"),
    # ("gemini-2.5-pro", "gemini/gemini-2.5-pro"),
    ("kimi-k2", "openrouter/moonshotai/kimi-k2"),
]

TEST_SET_ENTRIES = 100


async def main():
    backend = LocalBackend()
    models = []
    for model_name, model_id in MODELS_TO_BENCHMARK:
        model = art.Model(
            name=model_name.split("/")[-1],
            project="email_agent",
            inference_model_name=model_id,
            config=ProjectPolicyConfig(),
        )
        await model.register(backend)
        models.append(model)

    results = await asyncio.gather(
        *[benchmark_model(model, TEST_SET_ENTRIES) for model in models]
    )
    for model in models:
        print(f"Pushing {model.name} to S3")
        await backend._experimental_push_to_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
        )

    df: pl.DataFrame = pl.concat(results)
    df = df.transpose(include_header=True)

    col_names = {"column": "metric"}
    for i, model in enumerate(MODELS_TO_BENCHMARK):
        col_names[f"column_{i}"] = model[0]

    df = df.rename(col_names)
    with open("data/benchmark_prompted_models.html", "w") as f:
        f.write(df.to_pandas().to_html())

    print(df.to_pandas().to_markdown())


asyncio.run(main())
