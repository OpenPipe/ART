try:
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns
except ImportError:
    raise ImportError(
        "Plotting dependencies are not installed. Please install them with: "
        "pip install openpipe-art[plotting]"
    )

from art.utils.benchmarking.filter_model_split import filter_rename_model_split
from art.utils.benchmarking.types import BenchmarkModelKey


def training_progress_chart(
    df: pl.DataFrame,
    metric_name: str,
    models: list[str | BenchmarkModelKey] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    perfect_score: float | None = None,
    legend_loc: str | None = "lower right",
):
    """Plot and save a line chart of *metric_name* over training *step* for every model.

    Parameters
    ----------
    df : pl.DataFrame
        Table produced by :func:`load_trajectories`.
    metric_name : str
        Name of the metric to plot.
    models : list[str | BenchmarkModelKey] | None
        Optional explicit ordering / subset of model names.  Each element can
        either be a plain string (``"gpt-4o"``) or a ``BenchmarkModelKey``
        object.
    title : str | None, optional
        Custom figure title.  If *None* (default) a sensible title will be
        generated automatically.
    x_label : str | None, optional
        Custom label for the *x*-axis.  If *None* (default) the label will be
        set to ``"Step"`` (the previous hard-coded behaviour).  Pass an empty
        string (``""``) to remove the label entirely.
    y_label : str | None, optional
        Custom label for the *y*-axis.  If *None* (default) the *y*-axis will
        remain unlabeled (replicating the previous behaviour).  Pass an empty
        string (``""``) to explicitly clear the label.
    perfect_score : float | None, optional
        If provided, draws a horizontal reference line at this *y*-value and
        annotates it with the text "Perfect score" on the left-hand side of
        the plot.  The reference line is excluded from the legend.
    legend_loc : str | None, optional
        Location of the legend, passed directly to `matplotlib.pyplot.legend`.
        Defaults to ``"lower right"``.

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

    # ------------------------------------------------------------------
    # Determine plotting / legend order
    # ------------------------------------------------------------------
    if models is not None:
        models_keys = [
            entry if isinstance(entry, BenchmarkModelKey) else BenchmarkModelKey(entry)
            for entry in models
        ]

        df = filter_rename_model_split(df, models_keys)

        # Preserve caller‑specified order *after* renaming
        plot_models = [key.display_name for key in models_keys]
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
    fig, ax = plt.subplots(figsize=(6, 4))

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
    model_colors = {m: c for m, c in zip(ordered_for_palette, palette)}

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
        title = f"{metric_name}"
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
