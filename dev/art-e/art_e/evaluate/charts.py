# %%
from typing import Sequence

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def training_progress_chart(
    df: pl.DataFrame,
    split: str,
    metric_name: str,
    models: Sequence[str | tuple[str, str]] | None = None,
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

    plt.rcParams["figure.dpi"] = 300

    full_metric_col = (
        metric_name if metric_name in df.columns else f"metric_{metric_name}"
    )

    if full_metric_col not in df.columns:
        raise ValueError(
            f"Column '{full_metric_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    df = df.filter(pl.col("split") == split)

    # ------------------------------------------------------------------
    # Determine plotting / legend order
    # ------------------------------------------------------------------
    if models is not None:
        # ------------------------------------------------------------------
        # Support internal→display name mapping via (internal, display) tuples
        # ------------------------------------------------------------------
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

    # Use a clean theme without background grid lines
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=figsize if figsize is not None else (6, 4))

    # Explicitly remove grid lines that might be enabled by default
    ax.grid(False)

    # Determine global x‑range so that we can draw horizontal reference lines
    all_steps = agg_df["step"].unique().to_list()
    if not all_steps:
        raise ValueError(
            "Unable to determine the range of training steps – no data available."
        )
    min_step, max_step = min(all_steps), max(all_steps)

    # ------------------------------------------------------------------
    # Optional: draw a reference line indicating a perfect score
    # ------------------------------------------------------------------
    if perfect_score is not None:
        # Draw the line *before* plotting the model curves so that it sits
        # behind them; exclude it from the legend by using a special label.
        ax.plot(
            [min_step, max_step],
            [perfect_score, perfect_score],
            linestyle="-",
            linewidth=1,
            color="dimgray",
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
            color="dimgray",
        )

    # ------------------------------------------------------------------
    # Draw trained models first so that prompted comparison models (usually
    # baselines) are rendered *on top* and therefore remain visible even if
    # lines overlap.
    # ------------------------------------------------------------------

    # Identify models with a single evaluation step (= prompted / comparison)
    step_counts = (
        df.group_by("model")
        .agg(pl.col("step").n_unique().alias("n_steps"))
        .to_dict(as_series=False)
    )
    step_count_map = dict(zip(step_counts["model"], step_counts["n_steps"]))

    trained_first = [m for m in plot_models if step_count_map.get(m, 2) > 1]
    comparison_last = [m for m in plot_models if step_count_map.get(m, 1) == 1]

    # ------------------------------------------------------------------
    # Colour assignment
    # ------------------------------------------------------------------
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
    ax.set_title(title)

    # ------------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------------

    # X‑axis label
    if x_label is None:
        ax.set_xlabel("Step")
    else:
        ax.set_xlabel(x_label)

    # Y‑axis label
    if y_label is not None:
        ax.set_ylabel(y_label)

    # ------------------------------------------------------------------
    # Assemble the legend: trained models first, comparison models afterwards
    # so that the legend mirrors the visual layering (trained on top).
    # ------------------------------------------------------------------

    handles, labels = ax.get_legend_handles_labels()
    handle_map = {lbl: h for lbl, h in zip(labels, handles)}
    ordered_labels = trained_first + comparison_last
    ordered_handles = [handle_map[lbl] for lbl in ordered_labels]

    ax.legend(
        handles=ordered_handles,
        labels=ordered_labels,
        frameon=True,
        facecolor="white",
        loc=legend_loc,
        borderaxespad=0.5,
    )

    # Remove the top and right spines for a cleaner presentation
    sns.despine()

    fig.tight_layout(pad=1.0)
    return fig


def comparison_models_bar_chart(
    df: pl.DataFrame,
    split: str,
    metric_name: str,
    models: Sequence[str | tuple[str, str]] | None = None,
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

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    plt.rcParams["figure.dpi"] = 300

    # Resolve column name (support both raw + prefixed)
    full_metric_col = (
        metric_name if metric_name in df.columns else f"metric_{metric_name}"
    )
    if full_metric_col not in df.columns:
        raise ValueError(
            f"Column '{full_metric_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # Filter + determine model ordering (same logic as *graph_metric*)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Aggregate first / last metrics per model
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=figsize if figsize is not None else (6, 4))
    ax.grid(False)

    # ------------------------------------------------------------------
    # Colour scheme – match the example shared by the caller
    # ------------------------------------------------------------------
    ORANGE = "#e67a30"
    GREY = "#e0dcd5"

    # Colour assignment: use same approach (trained first) so the most
    # distinctive colours are given to the trained models.

    bar_width = 0.7
    x_positions = np.arange(len(model_stats))

    for idx, (model, base, imp) in enumerate(model_stats):
        base_scaled = base * scale
        imp_scaled = imp * scale
        color = ORANGE if imp_scaled else GREY

        # Base segment (always)
        ax.bar(
            x_positions[idx],
            base_scaled,
            width=bar_width,
            color=color if imp_scaled else "#e0dcd5",
            edgecolor="white",
        )

        # Improvement segment (only for trained models)
        if imp_scaled != 0:
            ax.bar(
                x_positions[idx],
                imp_scaled,
                width=bar_width,
                bottom=base_scaled,
                color=color,
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
        )

    # ------------------------------------------------------------------
    # Optional perfect score reference line
    # ------------------------------------------------------------------
    if perfect_score is not None:
        score_scaled = perfect_score * scale
        ax.plot(
            [-0.5, len(model_stats) - 0.5],
            [score_scaled, score_scaled],
            linestyle="-",
            linewidth=1,
            color="dimgray",
            label="_nolegend_",
        )
        ax.text(
            -0.4,
            score_scaled - (max_val * scale) * 0.05,
            "Perfect score",
            va="top",
            ha="left",
            color="dimgray",
        )

    # ------------------------------------------------------------------
    # Axes & legend
    # ------------------------------------------------------------------
    ax.set_xticks(x_positions)
    ax.set_xticklabels([m for m, _, _ in model_stats], rotation=0, ha="center")
    # Remove y‑axis label, ticks, and spine as requested
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    if title is None:
        title = f"{split}/{metric_name}"
    ax.set_title(title, pad=15)

    # Clean up remaining spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Save + return - Saving is now handled externally
    # ------------------------------------------------------------------
    # Removed saving logic:
    # save_path = Path(save_to)
    # if save_path.suffix == "":
    #     save_path = save_path.with_suffix(".png")
    # os.makedirs(save_path.parent, exist_ok=True)
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
