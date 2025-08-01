# %%

import importlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import os

import art_e.evaluate.charts

importlib.reload(art_e.evaluate.charts)

import polars as pl  # noqa: E402
from art.utils.benchmarking.load_trajectories import load_trajectories  # noqa: E402
from art_e.evaluate.charts import comparison_models_bar_chart, training_progress_chart  # noqa: E402

# Define and create the temporary directory early
TMP_DIR = "/tmp/art-e"
os.makedirs(TMP_DIR, exist_ok=True)

# await load_trajectories.bust_cache()
df = await load_trajectories(  # noqa: F704
    ".art/email_agent",
    models=[
        "email-agent-008",
        "email-agent-014",
        "gpt-4.1",
        "gemini-2.5-pro",
        "o4-mini",
        "o3",
        "deepseek-r1",
        "gpt-4o",
        "gemini-2.0-flash",
    ],
)  # type: ignore

df = df.filter(pl.col("step").lt(510))

# Create the first chart (bar chart)
fig1 = comparison_models_bar_chart(
    df,
    split="val",
    metric_name="answer_correct",
    models=[
        ("gpt-4.1", "GPT-4.1"),
        ("gemini-2.5-pro", "Gemini\n2.5 Pro"),
        # ("gemini-2.0-flash", "Gemini\n2.0 Flash"),
        ("o4-mini", "o4-mini"),
        "o3",
        ("email-agent-008", "ART·E\n(Qwen 2.5 14B)"),
    ],
    title="Percentage of Questions Answered Correctly",
)

fig1.savefig(
    "/Users/kyle/proj/openpipe-web/public/blog-images/art-e-accuracy-comparison.svg"
)
# Save PNG copy to tmp dir
fig1.savefig(os.path.join(TMP_DIR, "art-e-accuracy-comparison.png"))

comparison_models_bar_chart(
    df,
    split="val",
    metric_name="answer_correct",
    models=[
        ("gpt-4.1", "GPT-4.1"),
        ("gemini-2.5-pro", "Gemini\n2.5 Pro"),
        # ("gemini-2.0-flash", "Gemini\n2.0 Flash"),
        ("o4-mini", "o4-mini"),
        "o3",
        ("email-agent-008", "ART·E\n(Qwen 2.5 14B)"),
    ],
    title="Percentage of Questions Answered Correctly",
    figsize=(4, 4),
).savefig("/tmp/art-e/art-e-accuracy-comparison-small.png")


# Create the second chart (line chart)
fig2 = training_progress_chart(
    df.filter(pl.col("step").ne(592)),
    "val",
    "answer_correct",
    models=[
        "o3",
        ("o4-mini", "o4-mini"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gpt-4.1", "GPT-4.1"),
        ("email-agent-008", "ART·E (Qwen 2.5 14B)"),
    ],
    title="Fraction of Questions Answered Correctly",
    y_label="Val set success rate",
)
# Save the second chart using the helper
fig2.savefig(
    "/Users/kyle/proj/openpipe-web/public/blog-images/art-e-accuracy-training-progress.svg"
)
# Save PNG copy to tmp dir
fig2.savefig(os.path.join(TMP_DIR, "art-e-accuracy-training-progress.png"))

# %%
# --- Create Latency and Cost Charts ---
plt.rcParams["figure.dpi"] = 300
sns.set_theme(style="ticks")

# models_cost_latency = ["o4-mini", "o3", "ART·E"] # Old order
models_cost_latency = ["o3", "o4-mini", "ART·E"]  # New order
# latency_values = [3.4, 5.6, 1.1] # Old order
latency_values = [5.6, 3.4, 1.1]  # New order
# cost_values = [7.88, 55.19, 0.85] # Old order
cost_values = [55.19, 7.88, 0.85]  # New order

# Define colors - using the specific hex codes from comparison_models_bar_chart
ORANGE = "#e67a30"
GREY = "#e0dcd5"
model_colors = {"o4-mini": GREY, "o3": GREY, "ART·E": ORANGE}
# colors = sns.color_palette("muted", n_colors=len(models_cost_latency)) # Old way
# model_colors = {model: color for model, color in zip(models_cost_latency, colors)} # Old way
bar_colors = [model_colors[model] for model in models_cost_latency]


def plot_latency_chart(ax, models, values, colors):
    """Plots the latency bar chart onto the given Axes."""
    x_pos = np.arange(len(models))
    bar_width = 0.6
    bars_latency = ax.bar(x_pos, values, width=bar_width, color=colors)
    ax.set_title("Full-Run Latency (Seconds)", pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=0, ha="center")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.grid(False)

    # Add latency value labels
    max_latency = max(values) if values else 0
    for bar in bars_latency:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max_latency * 0.02,  # Position label slightly above bar
            f"{yval:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    return ax


def plot_cost_chart(ax, models, values, colors):
    """Plots the cost bar chart onto the given Axes."""
    x_pos = np.arange(len(models))
    bar_width = 0.6
    bars_cost = ax.bar(x_pos, values, width=bar_width, color=colors)
    ax.set_title("Cost per 1K Runs", pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=0, ha="center")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.grid(False)

    # Add cost value labels
    max_cost = max(values) if values else 0
    for bar in bars_cost:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max_cost * 0.02,  # Position label slightly above bar
            f"${yval:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    return ax


def bold_arte_label(ax):
    """Bolds the 'ART·E' label on the x-axis."""
    labels = ax.get_xticklabels()
    for label in labels:
        if label.get_text() == "ART·E":
            label.set_fontweight("bold")
    ax.set_xticklabels(labels)


# --- Create and Save Combined Chart ---
fig_combined, axes_combined = plt.subplots(
    1, 2, figsize=(8, 3.5)
)  # Adjusted figsize slightly

ax_latency_combined = plot_latency_chart(
    axes_combined[0], models_cost_latency, latency_values, bar_colors
)
ax_cost_combined = plot_cost_chart(
    axes_combined[1], models_cost_latency, cost_values, bar_colors
)

# Bold the 'ART·E' label on both axes of the combined plot
bold_arte_label(ax_latency_combined)
bold_arte_label(ax_cost_combined)

plt.tight_layout(pad=2.0)  # Add padding between plots and title
# save_figure_for_blog(fig_combined, "art-e-cost-latency-comparison") # If using a helper

combined_save_path = (
    "/Users/kyle/proj/openpipe-web/public/blog-images/art-e-cost-latency-comparison.svg"
)
fig_combined.savefig(combined_save_path)
# Save PNG copy to tmp dir
fig_combined.savefig(os.path.join(TMP_DIR, "art-e-cost-latency-comparison.png"))
print(f"Saved combined cost/latency chart to {combined_save_path}")
# plt.close(fig_combined) # Close figure if not needed later


# --- Create and Save Separate Charts ---
# TMP_DIR = "/tmp/art-e" # Moved to top
# os.makedirs(TMP_DIR, exist_ok=True) # Moved to top

# Create and save separate latency chart
fig_latency, ax_latency_single = plt.subplots(
    figsize=(4.5, 3.8)
)  # Slightly adjusted size
plot_latency_chart(ax_latency_single, models_cost_latency, latency_values, bar_colors)
bold_arte_label(ax_latency_single)
fig_latency.tight_layout(pad=1.5)
latency_save_path = os.path.join(TMP_DIR, "art-e-latency-comparison.png")
fig_latency.savefig(latency_save_path)
print(f"Saved separate latency chart to {latency_save_path}")
plt.close(fig_latency)

# Create and save separate cost chart
fig_cost, ax_cost_single = plt.subplots(figsize=(4.5, 3.8))  # Slightly adjusted size
plot_cost_chart(ax_cost_single, models_cost_latency, cost_values, bar_colors)
bold_arte_label(ax_cost_single)
fig_cost.tight_layout(pad=1.5)
cost_save_path = os.path.join(TMP_DIR, "art-e-cost-comparison.png")
fig_cost.savefig(cost_save_path)
print(f"Saved separate cost chart to {cost_save_path}")
plt.close(fig_cost)


print("Chart generation and conversion finished.")

# %%
# --- Create Bar Chart with All Prompted Models ---

fig_all_models = comparison_models_bar_chart(
    df,
    split="val",
    metric_name="answer_correct",
    models=[
        ("deepseek-r1", "DS\nR1"),
        ("gpt-4o", "GPT-4o"),
        ("gpt-4.1", "GPT-4.1"),
        ("gemini-2.0-flash", "Gemini\n2.0 Flash"),
        ("gemini-2.5-pro", "Gemini\n2.5 Pro"),
        ("o4-mini", "o4-mini"),
        ("o3", "o3"),
    ],
    title="Percentage of Questions Answered Correctly (Prompted Models)",
)

# Save the new chart
fig_all_models.savefig(
    "/Users/kyle/proj/openpipe-web/public/blog-images/art-e-accuracy-comparison-prompted-models.svg"
)
# Save PNG copy to tmp dir
fig_all_models.savefig(
    os.path.join(TMP_DIR, "art-e-accuracy-comparison-prompted-models.png")
)

# %%

fig = training_progress_chart(
    df.filter(pl.col("step").ne(592)),
    "val",
    "num_turns",
    models=[
        ("email-agent-008", "ART·E"),
        ("gpt-4.1", "GPT-4.1"),
        "o3",
        ("o4-mini", "o4-mini"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ],
    title="Average Number of Turns to Answer Question",
    y_label="Number of turns",
    legend_loc="upper right",
)

fig.savefig(
    "/Users/kyle/proj/openpipe-web/public/blog-images/art-e-num-turns-training-progress.svg"
)
# Save PNG copy to tmp dir
fig.savefig(os.path.join(TMP_DIR, "art-e-num-turns-training-progress.png"))

# %%

df = df.with_columns(
    (
        pl.col("metric_attempted_answer").cast(bool)
        & ~pl.col("metric_answer_correct").cast(bool)
    ).alias("metric_wrong_answer")
)

fig = training_progress_chart(
    df.filter(pl.col("step").ne(592)),
    "val",
    "wrong_answer",
    models=[
        ("email-agent-008", "ART·E"),
        ("gpt-4.1", "GPT-4.1"),
        "o3",
        ("o4-mini", "o4-mini"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ],
    title="Fraction of Answers Hallucinated",
    x_label="Training Step",
    legend_loc="upper right",
)

fig.tight_layout(pad=1.0)
fig.savefig(
    "/Users/kyle/proj/openpipe-web/public/blog-images/art-e-wrong-answer-training-progress.svg",
    bbox_inches="tight",
)
# Save PNG copy to tmp dir
fig.savefig(
    os.path.join(TMP_DIR, "art-e-wrong-answer-training-progress.png"),
    bbox_inches="tight",
)

# %%

# --- Export Example Trajectories ---

# Reload data if needed, or use the existing df if it contains all necessary models/steps
# It might be safer to reload with the specific models needed if the initial load was limited
# Ensure 'email-agent-014' and 'email-agent-008' are loaded
# df = await load_trajectories(".art/email_agent", models=["o3", "o4-mini", "email-agent-008", "email-agent-014"])

target_scenario_ids = [str(id) for id in [300, 34, 128, 383]]

# Define the filtering conditions
conditions = (
    ((pl.col("model") == "email-agent-014") & (pl.col("step") == 0))
    | ((pl.col("model") == "o3") & (pl.col("step") == 0))
    | ((pl.col("model") == "o4-mini") & (pl.col("step") == 0))
    | ((pl.col("model") == "email-agent-008") & (pl.col("step") == 505))
)

# Filter the DataFrame
filtered_df = df.filter(
    pl.col("split").eq("val")
    & pl.col("metadata_scenario_id").is_in(target_scenario_ids)
    & conditions
)

# Define pretty model names mapping
pretty_model_names = {
    "o3": "o3",
    "o4-mini": "o4-mini",
    "email-agent-008": "ART·E",
    "email-agent-014": "Qwen 2.5 14B",
}

# Prepare the nested dictionary
exported_trajectories = {}

for row in filtered_df.to_dicts():
    scenario_id = row["metadata_scenario_id"]
    model_raw = row["model"]
    pretty_name = pretty_model_names[model_raw]

    if scenario_id not in exported_trajectories:
        exported_trajectories[scenario_id] = {}

    exported_trajectories[scenario_id][pretty_name] = {
        "messages": row["messages"],
        "tools": row.get("tools", None),
    }

# Export to JSON
output_path = "/Users/kyle/proj/ART/examples/art-e/exported_trajectories.json"
with open(output_path, "w") as f:
    json.dump(exported_trajectories, f, indent=2)

print(f"Exported example trajectories to {output_path}")

# %%


# df.filter(pl.col("split").eq("val")).group_by("metadata_scenario_id").first()


for row in (
    df.filter(pl.col("messages").list.len() > 1)
    .filter(pl.col("split").eq("val"))
    .group_by("metadata_scenario_id")
    .first()
    .to_dicts()
):
    print(f"{row['metadata_scenario_id']}: {row['messages'][1]['content']}")

# %%

df.filter(pl.col("split").eq("val")).group_by("metadata_scenario_id").first()
