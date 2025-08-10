# %%

import asyncio

from all_experiments import models
from art_e.data.query_iterators import load_synthetic_queries
from art_e.evaluate.charts import comparison_models_bar_chart, training_progress_chart

from art.utils.benchmarking.load_trajectories import load_trajectories
from art.utils.benchmarking.pull_model_trajectories import pull_model_trajectories

ds_size_models = [m for m in models.values() if m.name.startswith("ea-210")]

await asyncio.gather(*[pull_model_trajectories(m) for m in ds_size_models])  # type: ignore  # noqa: F704

# %%

models = [
    "o3",
    "o4-mini",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "qwen3-32b",
    *[m.name for m in ds_size_models],
]

# await load_trajectories.bust_cache()
df = await load_trajectories(  # noqa: F704
    "../../.art/email_agent",
    models=models,
)  # type: ignore

# %%

training_progress_chart(
    df,
    "val",
    "answer_correct",
    models=models,
    title="Fraction of Questions Answered Correctly",
    y_label="Val set success rate",
)

# %%
import polars as pl  # noqa: E402

labeled_models = []
for m in ds_size_models:
    model_size = m.name.split("-")[-1]
    unit = "scenario" if model_size == "1" else "scenarios"
    labeled_models.append((m.name, f"{model_size} {unit}"))

comparison_models_bar_chart(
    df.filter(pl.col("step").ne(144)),
    "val",
    "answer_correct",
    models=["gemini-2.5-flash", "o3", "gemini-2.5-pro", *labeled_models],
    title="Eval Accuracy vs Training Dataset Size",
    y_label="Val set success rate",
    figsize=(20, 5),
)

# %%


# df.filter(pl.col("model").eq("ea-210-16")).filter(pl.col("split").eq("val")).group_by("step").count()


# %%

scenarios = load_synthetic_queries(split="train", limit=10)

for i, scenario in enumerate(scenarios):
    print(scenario.inbox_address)
