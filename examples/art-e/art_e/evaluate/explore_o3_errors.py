# %%
import polars as pl

from art_e.evaluate.load_trajectories import load_trajectories
from art_e.evaluate.display_run_html import display_run_html
from art_e.data.query_iterators import load_synthetic_queries

df = await load_trajectories(".art/email_agent", models=["o3"])  # type: ignore
scenarios = load_synthetic_queries(
    split="test", limit=100, exclude_known_bad_queries=False
)

# %%


# Let's look at the ones that o3 is getting wrong
failures_o3 = (
    df.filter(pl.col("model") == "o3")
    .filter(pl.col("metric_answer_correct") == 0)
    .filter(pl.col("split") == "val")
    # .filter(pl.col("step") == 594) # Or filter by a specific step if needed
).limit(5)  # Limit to 5 failures for inspection
print(f"Found {len(failures_o3)} failures for o3")

for i, row in enumerate(failures_o3.rows(named=True)):
    print(f"Processing failure {i + 1}/{len(failures_o3)}")
    display_run_html(row, scenarios)
