try:
    import polars as pl
except ImportError:
    raise ImportError(
        "Plotting dependencies are not installed. Please install them with: "
        "pip install openpipe-art[plotting]"
    )

from ..filter_model_split import filter_rename_model_split
from ..types import BenchmarkModelKey


def percentage_comparison_bar_chart(
    df: pl.DataFrame,
    metric_name: str,
    models: list[str | BenchmarkModelKey] | None = None,
    title: str | None = None,
    y_label: str | None = None,
    perfect_score: float | None = None,
    num_decimals: int = 0,
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
    metric_name : str
        Name of the metric (without the ``"metric_"`` prefix) or the full
        column name as it appears in *df*.
    models : list[str | BenchmarkModelKey] | None, optional
        Same semantics as in :func:`training_progress_chart` - accepts either
        plain model identifiers or ``(internal, display)`` tuples.
    title : str | None, optional
        Custom figure title.  If *None*, a sensible default is generated.
    y_label : str | None, optional
        Label for the *y*-axis.  If *None* (default) the axis remains
        unlabeled.
    perfect_score : float | None, optional
        If provided, draws a horizontal reference line at this *y*-value and
        annotates it with the text "Perfect score".

    Returns
    -------
    matplotlib.figure.Figure
        The created figure so that it can be displayed inline in IPython /
        Jupyter notebooks.
    """

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError:
        raise ImportError(
            "Plotting dependencies are not installed. Please install them with: "
            "pip install openpipe-art[plotting]"
        )

    # create new copy of df
    df = df.clone()

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

    if models is not None:
        models_keys = [
            entry if isinstance(entry, BenchmarkModelKey) else BenchmarkModelKey(entry)
            for entry in models
        ]

        df = filter_rename_model_split(df, models_keys)

        plot_models = [key.display_name for key in models_keys]
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
        raise ValueError("No rows left after filtering - check 'split' or model list.")

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
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(False)

    # ------------------------------------------------------------------
    # Colour scheme - match the example shared by the caller
    # ------------------------------------------------------------------
    ORANGE = "#e67a30"
    GREY = "#e0dcd5"

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
                f"RL +{imp_scaled:.{num_decimals}f}{'%' if scale == 100 else ''}",
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
            f"{total:.{num_decimals}f}{'%' if scale == 100 else ''}",
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
    # Remove y-axis label, ticks, and spine as requested
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    if title is None:
        title = metric_name
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
