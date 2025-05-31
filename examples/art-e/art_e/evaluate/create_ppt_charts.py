# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import asyncio

import polars as pl
from art_e.evaluate.load_trajectories import load_trajectories

plt.rcParams.update(
    {
        "figure.facecolor": "none",  # Transparent figure background
        "axes.facecolor": "none",  # Transparent axes background
        "text.color": "white",  # White text
        "axes.labelcolor": "white",  # White axis labels
        "xtick.color": "white",  # White x-axis ticks
        "ytick.color": "white",  # White y-axis ticks
        "axes.edgecolor": "white",  # White axes edges
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 600,
    }
)


def setup_dark_theme():
    """Set up seaborn theme while preserving dark background settings."""
    # Store our dark theme settings
    dark_settings = {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.edgecolor": "white",
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }

    # Apply seaborn theme first
    sns.set_theme(style="ticks")

    # Then override with our dark settings
    plt.rcParams.update(dark_settings)


def training_progress_chart(
    df: pl.DataFrame,
    split: str,
    metric_name: str,
    models: list[str | tuple[str, str]] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    perfect_score: float | None = None,
    legend_loc: str | None = "lower right",
    figsize: tuple[float, float] | None = None,
):
    """Plot and save a line chart of *metric_name* over training *step* for every model.

    Parameters
    ----------
    df : pl.DataFrame
        Table produced by :func:`load_trajectories`.
    split : str
        Which evaluation split to visualise (e.g. ``"train"`` / ``"val"``).
    models : list[str | tuple[str, str]] | None
        Optional explicit ordering / subset of model names.  Each element can
        either be a plain string (``"gpt-4o"``) or a 2‑tuple where the first
        item is the *internal* model identifier as it appears in *df* and the
        second item is the *display* name that should be used in the plot
        (e.g. ``("gemini-2.0-flash", "Gemini 2.0\nFlash")``).  When a tuple is
        supplied, the function automatically renames the corresponding rows in
        the DataFrame so that downstream processing and the final legend use
        the display name.
    title : str | None, optional
        Custom figure title.  If *None* (default) a sensible title will be
        generated automatically.
    x_label : str | None, optional
        Custom label for the *x*-axis.  If *None* (default) the label will be
        set to ``"Step"`` (the previous hard‑coded behaviour).  Pass an empty
        string (``""``) to remove the label entirely.
    y_label : str | None, optional
        Custom label for the *y*-axis.  If *None* (default) the *y*-axis will
        remain unlabeled (replicating the previous behaviour).  Pass an empty
        string (``""``) to explicitly clear the label.
    perfect_score : float | None, optional
        If provided, draws a horizontal reference line at this *y*‑value and
        annotates it with the text "Perfect score" on the left‑hand side of
        the plot.  The reference line is excluded from the legend.
    legend_loc : str | None, optional
        Location of the legend, passed directly to `matplotlib.pyplot.legend`.
        Defaults to ``"lower right"``.
    figsize : tuple[float, float] | None, optional
        Figure size in inches (width, height). If *None* (default), uses
        ``(6, 4)``.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure so that it can be displayed inline in IPython /
        Jupyter notebooks.
    """

    full_metric_col = (
        metric_name if metric_name in df.columns else f"metric_{metric_name}"
    )

    if full_metric_col not in df.columns:
        raise ValueError(
            f"Column '{full_metric_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    df = df.filter(pl.col("split") == split)

    # Determine plotting / legend order
    if models is not None:
        # Support internal→display name mapping via (internal, display) tuples
        internal_names: list[str] = []  # models to keep in the DataFrame
        display_names: list[str] = []  # names as they should appear in plots
        rename_pairs: list[tuple[str, str]] = []

        for entry in models:
            if isinstance(entry, tuple):
                internal, display = entry
            else:
                internal = display = entry  # plain string → same internal/display

            internal_names.append(internal)
            display_names.append(display)

            if internal != display:
                rename_pairs.append((internal, display))

        # Restrict to the requested models (internal identifiers)
        df = df.filter(pl.col("model").is_in(internal_names))

        # Apply renaming so that subsequent logic only sees display names
        for internal, display in rename_pairs:
            df = df.with_columns(
                pl.when(pl.col("model") == internal)
                .then(pl.lit(display))
                .otherwise(pl.col("model"))
                .alias("model")
            )

        # Preserve caller‑specified order *after* renaming
        plot_models = display_names
    else:
        # Separate trained models (multiple distinct steps) from comparison
        # models (exactly one step) and list trained models first.
        step_counts = (
            df.group_by("model")
            .agg(pl.col("step").n_unique().alias("n_steps"))
            .to_dict(as_series=False)
        )

        trained_models: list[str] = []
        comparison_models: list[str] = []
        for model_name, n_steps in zip(step_counts["model"], step_counts["n_steps"]):
            if n_steps > 1:
                trained_models.append(model_name)
            else:
                comparison_models.append(model_name)

        # Sort alphabetically within each group to keep things predictable
        plot_models = sorted(trained_models) + sorted(comparison_models)

    # Guard against empty selection
    if df.is_empty():
        raise ValueError(
            "No rows left after applying the split filter. Check your 'split' argument or the DataFrame contents."
        )

    agg_df = (
        df.group_by(["model", "step"])
        .agg(pl.col(full_metric_col).mean().alias(full_metric_col))
        .sort("step")
    )

    # Use a clean theme without background grid lines while *preserving* any
    # pre-configured matplotlib colour overrides (e.g. white text on a dark
    # background).  Passing ``plt.rcParams`` ensures that seaborn does not
    # reset our customised colours.
    setup_dark_theme()
    fig, ax = plt.subplots(figsize=figsize if figsize is not None else (6, 4))

    # Explicitly remove grid lines that might be enabled by default
    ax.grid(False)

    # Ensure tick marks and axis labels use the globally-configured colour
    tick_col = plt.rcParams.get("text.color", "white")
    ax.tick_params(colors=tick_col)
    ax.yaxis.label.set_color(tick_col)
    ax.xaxis.label.set_color(tick_col)

    # Determine global x‑range so that we can draw horizontal reference lines
    all_steps = agg_df["step"].unique().to_list()
    if not all_steps:
        raise ValueError(
            "Unable to determine the range of training steps – no data available."
        )
    min_step, max_step = min(all_steps), max(all_steps)

    # Optional: draw a reference line indicating a perfect score
    if perfect_score is not None:
        # Draw the line *before* plotting the model curves so that it sits
        # behind them; exclude it from the legend by using a special label.
        ax.plot(
            [min_step, max_step],
            [perfect_score, perfect_score],
            linestyle="-",
            linewidth=1,
            color="lightgray",
            label="_nolegend_",
        )

        # Place the annotation slightly below the line to avoid overlap.
        # Calculate a small offset based on the y-axis limits
        y_min, y_max = ax.get_ylim()
        offset = (y_max - y_min) * 0.1  # Adjust multiplier as needed
        ax.text(
            min_step,
            perfect_score - offset,  # Adjusted y-position
            "Perfect score",
            verticalalignment="top",
            horizontalalignment="left",
            # fontsize="small", # Removed to use default size
            color="lightgray",
        )

    # Draw trained models first so that prompted comparison models (usually
    # baselines) are rendered *on top* and therefore remain visible even if
    # lines overlap.

    # Identify models with a single evaluation step (= prompted / comparison)
    step_counts = (
        df.group_by("model")
        .agg(pl.col("step").n_unique().alias("n_steps"))
        .to_dict(as_series=False)
    )
    step_count_map = dict(zip(step_counts["model"], step_counts["n_steps"]))

    trained_first = [m for m in plot_models if step_count_map.get(m, 2) > 1]
    comparison_last = [m for m in plot_models if step_count_map.get(m, 1) == 1]

    # Colour assignment
    # We want the most distinctive colours to belong to the trained models even
    # though they are drawn *after* the comparison models.  Therefore we
    # construct a palette and assign colours explicitly: first to the trained
    # models, then to the comparison / baseline models.

    ordered_for_palette = trained_first + comparison_last
    palette = sns.color_palette(n_colors=len(ordered_for_palette))
    model_colors = {m: c for m, c in zip(ordered_for_palette, palette)}  # type: ignore

    # Track scores of comparison models to adjust linestyle for overlaps
    plotted_comparison_scores: set[float] = set()

    for model in comparison_last + trained_first:
        model_df = agg_df.filter(pl.col("model") == model).sort("step")
        x = model_df["step"].to_list()
        y = model_df[full_metric_col].to_list()

        if len(x) == 1:
            # Prompted comparison model – draw a horizontal line across the full range
            score = y[0]
            linestyle = "--"  # Default to dashed

            # If this score was already plotted by another comparison model, use dotted
            if score in plotted_comparison_scores:
                linestyle = ":"
            else:
                plotted_comparison_scores.add(score)  # Record this score

            ax.plot(
                [min_step, max_step],
                [score, score],
                linestyle=linestyle,  # Use determined style
                linewidth=2,
                color=model_colors[model],
                label=model,
            )
        else:
            # Trained model
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                color=model_colors[model],
                label=model,
            )

    if title is None:
        title = f"{split}/{metric_name}"
    ax.set_title(title, color=tick_col)

    # Axis labels

    # X‑axis label
    if x_label is None:
        ax.set_xlabel("Step")
    else:
        ax.set_xlabel(x_label)

    # Y‑axis label
    if y_label is not None:
        ax.set_ylabel(y_label)

    # Assemble the legend: trained models first, comparison models afterwards
    # so that the legend mirrors the visual layering (trained on top).

    handles, labels = ax.get_legend_handles_labels()
    handle_map = {lbl: h for lbl, h in zip(labels, handles)}
    ordered_labels = trained_first + comparison_last
    ordered_handles = [handle_map[lbl] for lbl in ordered_labels]

    # Use a transparent legend background so that the charts look good on
    # dark-themed slide decks.  This relies on the caller (or a global
    # ``rcParams`` change) to set suitable text / edge colours for the legend
    # so that it remains readable on the chosen background.
    legend = ax.legend(
        handles=ordered_handles,
        labels=ordered_labels,
        frameon=True,
        facecolor="none",  # fully transparent so it inherits background
        loc=legend_loc,
        borderaxespad=0.5,
    )
    # Make sure legend text colour matches the rest of the plot
    for text in legend.get_texts():
        text.set_color(tick_col)

    # Remove the top and right spines for a cleaner presentation
    sns.despine()

    fig.tight_layout(pad=1.0)
    return fig


def comparison_models_bar_chart(
    df: pl.DataFrame,
    split: str,
    metric_name: str,
    models: list[str | tuple[str, str]] | None = None,
    title: str | None = None,
    y_label: str | None = None,
    perfect_score: float | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Create a bar chart visualising *metric_name* for all *models* at the
    **first** and **last** available training step.

    For *trained* models (those that have been evaluated across multiple training
    steps) the bar is *stacked*: the base segment represents the score at the
    **first** step while the hatched segment on top encodes the improvement (or
    decline) by the **last** step.  Prompted comparison models that were only
    evaluated once are rendered as a single, solid bar.

    Parameters
    ----------
    df : pl.DataFrame
        Table produced by :func:`load_trajectories`.
    split : str
        Which evaluation split to visualise (e.g. ``"train"`` / ``"val"``).
    metric_name : str
        Name of the metric (without the ``"metric_"`` prefix) or the full
        column name as it appears in *df*.
    models : list[str | tuple[str, str]] | None, optional
        Same semantics as in :func:`training_progress_chart` – accepts either
        plain model identifiers or ``(internal, display)`` tuples.
    title : str | None, optional
        Custom figure title.  If *None*, a sensible default is generated.
    y_label : str | None, optional
        Label for the *y*‑axis.  If *None* (default) the axis remains
        unlabeled.
    perfect_score : float | None, optional
        If provided, draws a horizontal reference line at this *y*‑value and
        annotates it with the text "Perfect score".
    figsize : tuple[float, float] | None, optional
        Figure size in inches (width, height). If *None* (default), uses
        ``(6, 4)``.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure so that it can be displayed inline in IPython /
        Jupyter notebooks.
    """
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Resolve column name (support both raw + prefixed)
    full_metric_col = (
        metric_name if metric_name in df.columns else f"metric_{metric_name}"
    )
    if full_metric_col not in df.columns:
        raise ValueError(
            f"Column '{full_metric_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    df = df.filter(pl.col("split") == split)

    if models is not None:
        internal_names: list[str] = []
        display_names: list[str] = []
        rename_pairs: list[tuple[str, str]] = []

        for entry in models:
            if isinstance(entry, tuple):
                internal, display = entry
            else:
                internal = display = entry

            internal_names.append(internal)
            display_names.append(display)

            if internal != display:
                rename_pairs.append((internal, display))

        df = df.filter(pl.col("model").is_in(internal_names))

        for internal, display in rename_pairs:
            df = df.with_columns(
                pl.when(pl.col("model") == internal)
                .then(pl.lit(display))
                .otherwise(pl.col("model"))
                .alias("model")
            )

        plot_models = display_names
    else:
        step_counts = (
            df.group_by("model")
            .agg(pl.col("step").n_unique().alias("n_steps"))
            .to_dict(as_series=False)
        )
        trained_models, comparison_models = [], []
        for mdl, n in zip(step_counts["model"], step_counts["n_steps"]):
            (trained_models if n > 1 else comparison_models).append(mdl)
        plot_models = sorted(trained_models) + sorted(comparison_models)

    if df.is_empty():
        raise ValueError("No rows left after filtering – check 'split' or model list.")

    agg = (df.group_by(["model", "step"]).agg(pl.col(full_metric_col).mean())).sort(
        ["model", "step"]
    )

    model_stats: list[tuple[str, float, float]] = []  # (model, base, improvement)
    for model in plot_models:
        sub = agg.filter(pl.col("model") == model)
        metric_vals = sub[full_metric_col].to_list()
        if len(metric_vals) == 0:
            continue
        base_score = float(metric_vals[0])
        final_score = float(metric_vals[-1])
        improvement = final_score - base_score if len(metric_vals) > 1 else 0.0
        model_stats.append((str(model), base_score, improvement))

    if not model_stats:
        raise ValueError("No data available for the requested models.")

    # Convert to percentages if the data appears to be in [0, 1] range
    max_val = max(base + imp for _, base, imp in model_stats)
    scale = 100.0 if max_val <= 1.01 else 1.0

    setup_dark_theme()
    fig, ax = plt.subplots(figsize=figsize if figsize is not None else (6, 4))
    ax.grid(False)

    tick_col = plt.rcParams.get("text.color", "white")

    trained_set = {m for m, b, imp in model_stats if imp != 0}

    ORANGE = "#e67a30"
    GREY = "#e0dcd5"

    # Colour assignment: use same approach (trained first) so the most
    # distinctive colours are given to the trained models.
    ordered_for_palette = [m for m in plot_models if m in trained_set] + [
        m for m in plot_models if m not in trained_set
    ]
    palette = sns.color_palette(n_colors=len(ordered_for_palette))
    model_colors = {m: c for m, c in zip(ordered_for_palette, palette)}  # type: ignore

    bar_width = 0.7
    x_positions = np.arange(len(model_stats))

    for idx, (model, base, imp) in enumerate(model_stats):
        base_scaled = base * scale
        imp_scaled = imp * scale

        # Keep the *base* segment grey even when an improvement exists so that
        # the only colour change between baseline and RL slides is the hatched
        # orange add-on.

        # Base segment (always, grey)
        ax.bar(
            x_positions[idx],
            base_scaled,
            width=bar_width,
            color=GREY,
            edgecolor="white",
        )

        # Improvement segment (only for trained models)
        if imp_scaled != 0:
            ax.bar(
                x_positions[idx],
                imp_scaled,
                width=bar_width,
                bottom=base_scaled,
                color=ORANGE,
                hatch="///",
                edgecolor="white",
                alpha=0.95,
            )
            # Annotate improvement in the middle of the improvement bar
            ax.text(
                x_positions[idx],
                base_scaled + imp_scaled / 2,
                f"RL +{imp_scaled:.0f}{'%' if scale == 100 else ''}",
                ha="center",
                va="center",
                color="#c05a20",  # slightly darker orange for contrast
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    boxstyle="round,pad=0.3",
                    alpha=0.95,
                    edgecolor="#e67a30",
                    linewidth=1,
                ),
            )

        # Total label on top
        total = base_scaled + imp_scaled
        ax.text(
            x_positions[idx],
            total + (max_val * scale) * 0.02,
            f"{total:.0f}{'%' if scale == 100 else ''}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color=tick_col,
        )

    if perfect_score is not None:
        score_scaled = perfect_score * scale
        ax.plot(
            [-0.5, len(model_stats) - 0.5],
            [score_scaled, score_scaled],
            linestyle="-",
            linewidth=1,
            color="lightgray",
            label="_nolegend_",
        )
        ax.text(
            -0.4,
            score_scaled - (max_val * scale) * 0.05,
            "Perfect score",
            va="top",
            ha="left",
            color="lightgray",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [m for m, _, _ in model_stats],
        rotation=0,
        ha="center",
        color=tick_col,
    )
    ax.tick_params(colors=tick_col)
    # Remove y‑axis label, ticks, and spine as requested
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    if title is None:
        title = f"{split}/{metric_name}"
    ax.set_title(title, pad=15)
    ax.title.set_color(tick_col)

    # Clean up remaining spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


# Define and create the temporary directory early
TMP_DIR = "/tmp/art-e"
os.makedirs(TMP_DIR, exist_ok=True)


# Helper function to handle async calls in both Jupyter and script environments
def load_trajectories_sync():
    """Load trajectories with proper async handling for both Jupyter and script environments."""

    async def _load():
        return await load_trajectories(
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
        )

    try:
        # Check if we're in a running event loop (like Jupyter)
        loop = asyncio.get_running_loop()
        # We're in a running loop, try to use nest_asyncio
        try:
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(_load())
        except ImportError:
            # nest_asyncio not available, use asyncio.create_task with loop.run_until_complete
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _load())
                return future.result()
    except RuntimeError:
        # No running event loop, safe to use asyncio.run()
        return asyncio.run(_load())


df = load_trajectories_sync()

df = df.filter(pl.col("step").lt(510))

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
        ("email-agent-008", "Qwen 2.5 14B\n+RL"),
    ],
    title="Fraction of Questions Answered Correctly",
    y_label="Val set success rate",
)
fig2.savefig(os.path.join(TMP_DIR, "art-e-accuracy-training-progress.png"))

# %%
# --- Create Latency and Cost Charts ---

models_cost_latency = ["o3", "o4-mini", "Qwen 2.5 14B\n+RL"]
latency_values = [5.6, 3.4, 1.1]
cost_values = [55.19, 7.88, 0.85]

ORANGE = "#e67a30"
GREY = "#e0dcd5"
model_colors = {"o4-mini": GREY, "o3": GREY, "Qwen 2.5 14B\n+RL": ORANGE}
bar_colors = [model_colors[model] for model in models_cost_latency]


def plot_latency_chart(ax, models, values, colors):
    """Plots the latency bar chart onto the given Axes."""
    x_pos = np.arange(len(models))
    bar_width = 0.6
    bars_latency = ax.bar(x_pos, values, width=bar_width, color=colors)
    ax.set_title(
        "Full-Run Latency (Seconds)",
        pad=15,
        color=plt.rcParams.get("text.color", "white"),
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        models, rotation=0, ha="center", color=plt.rcParams.get("text.color", "white")
    )
    ax.tick_params(colors=plt.rcParams.get("text.color", "white"))
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.grid(False)

    # Add latency value labels
    max_latency = max(values) if values else 0
    tick_col = plt.rcParams.get("text.color", "white")
    for bar in bars_latency:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max_latency * 0.02,  # Position label slightly above bar
            f"{yval:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            color=tick_col,
        )
    return ax


def plot_cost_chart(ax, models, values, colors):
    """Plots the cost bar chart onto the given Axes."""
    x_pos = np.arange(len(models))
    bar_width = 0.6
    bars_cost = ax.bar(x_pos, values, width=bar_width, color=colors)
    ax.set_title(
        "Cost per 1K Runs", pad=15, color=plt.rcParams.get("text.color", "white")
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        models, rotation=0, ha="center", color=plt.rcParams.get("text.color", "white")
    )
    ax.tick_params(colors=plt.rcParams.get("text.color", "white"))
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.grid(False)

    # Add cost value labels
    max_cost = max(values) if values else 0
    tick_col = plt.rcParams.get("text.color", "white")
    for bar in bars_cost:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max_cost * 0.02,  # Position label slightly above bar
            f"${yval:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color=tick_col,
        )
    return ax


# --- Create and Save Separate Charts ---
# TMP_DIR = "/tmp/art-e" # Moved to top
# os.makedirs(TMP_DIR, exist_ok=True) # Moved to top

# Create and save separate latency chart
setup_dark_theme()
fig_latency, ax_latency_single = plt.subplots(
    figsize=(4.5, 3.8)
)  # Slightly adjusted size
plot_latency_chart(ax_latency_single, models_cost_latency, latency_values, bar_colors)
fig_latency.tight_layout(pad=1.5)
latency_save_path = os.path.join(TMP_DIR, "art-e-latency-comparison.png")
fig_latency.savefig(latency_save_path)
print(f"Saved separate latency chart to {latency_save_path}")
plt.close(fig_latency)

# Create and save separate cost chart
setup_dark_theme()
fig_cost, ax_cost_single = plt.subplots(figsize=(4.5, 3.8))  # Slightly adjusted size
plot_cost_chart(ax_cost_single, models_cost_latency, cost_values, bar_colors)
fig_cost.tight_layout(pad=1.5)
cost_save_path = os.path.join(TMP_DIR, "art-e-cost-comparison.png")
fig_cost.savefig(cost_save_path)
print(f"Saved separate cost chart to {cost_save_path}")
plt.close(fig_cost)


print("Chart generation and conversion finished.")


# %%

fig = training_progress_chart(
    df.filter(pl.col("step").ne(592)),
    "val",
    "num_turns",
    models=[
        ("email-agent-008", "Qwen 2.5 14B\n+RL"),
        ("gpt-4.1", "GPT-4.1"),
        "o3",
        ("o4-mini", "o4-mini"),
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ],
    title="Average Number of Turns to Answer Question",
    y_label="Number of turns",
    legend_loc="upper right",
)

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
        ("email-agent-008", "Qwen 2.5 14B\n+RL"),
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
    os.path.join(TMP_DIR, "art-e-wrong-answer-training-progress.png"),
    bbox_inches="tight",
)


# --- Create single-model ART·E bar charts (baseline vs. RL) ---
# We show two versions back-to-back in the slide deck: the *baseline*
# score (grey bar) and the *post-RL* result including the improvement
# hatch (orange).  Both charts must keep identical y-axis limits so
# that the only visible change is the taller bar for ART·E.

arte_model_id = "email-agent-008"
arte_display_name = "Qwen 2.5 14B"

# Same comparison models as the earlier accuracy chart
comparison_models = [
    (arte_model_id, arte_display_name),  # ART·E / Qwen (first, smallest)
    ("gpt-4.1", "GPT-4.1"),
    ("gemini-2.5-pro", "Gemini\n2.5 Pro"),
    ("o4-mini", "o4-mini"),
    "o3",
]

# 1) Prepare *baseline* DataFrame: keep only the very first evaluation
#    step for ART·E so that it renders as a single grey bar.
first_step_val = int(
    df.filter(pl.col("model") == arte_model_id)
    .select(pl.col("step").min())
    .to_series()[0]
)

df_baseline = df.filter(
    (pl.col("model") != arte_model_id)  # keep all rows for other models
    | (pl.col("step") == first_step_val)  # only first step for ART·E
)

# 2) Create RL chart *first* to capture y-axis range
fig_rl = comparison_models_bar_chart(
    df,
    split="val",
    metric_name="answer_correct",
    models=comparison_models,
    title="Percentage of Questions Answered Correctly",
    figsize=(6, 4),
)

ax_rl = fig_rl.axes[0]
ylim_top = ax_rl.get_ylim()[1]

# Optionally display the figure inside notebooks for quick feedback
try:
    from IPython.display import display as _ip_display

    _ip_display(fig_rl)
except ImportError:
    pass

rl_save_path = os.path.join(TMP_DIR, "art-e-accuracy-comparison-rl.png")
fig_rl.tight_layout(pad=1.5)
fig_rl.savefig(rl_save_path)
print(f"Saved RL chart to {rl_save_path}")

# Close RL figure to free resources after display/save
plt.close(fig_rl)

# 3) Create baseline chart and force its y-axis to match
fig_base = comparison_models_bar_chart(
    df_baseline,
    split="val",
    metric_name="answer_correct",
    models=comparison_models,
    title="Percentage of Questions Answered Correctly",
    figsize=(6, 4),
)

fig_base.axes[0].set_ylim(0, ylim_top)

# Display baseline figure for quick feedback (if in notebook)
try:
    _ip_display(fig_base)
except NameError:
    pass

base_save_path = os.path.join(TMP_DIR, "art-e-accuracy-comparison-baseline.png")
fig_base.tight_layout(pad=1.5)
fig_base.savefig(base_save_path)
print(f"Saved baseline chart to {base_save_path}")
plt.close(fig_base)

# --- End ART·E baseline vs. RL charts ---

print("Chart generation and conversion finished.")
