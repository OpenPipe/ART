import polars as pl
from polars.testing import assert_frame_equal
import pytest

from art.utils.benchmarking.filter_model_split import filter_rename_model_split
from art.utils.benchmarking.types import BenchmarkModelKey


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "display_names",
    [
        pytest.param(["B", "C", "D"], id="chain"),
        pytest.param(["B", "A", "C"], id="swap"),
        pytest.param(["B", "C", "A"], id="cycle"),
        pytest.param(["A", "B", "C"], id="identity"),
        pytest.param(["Alpha", "Beta", "Gamma"], id="non-overlapping"),
    ],
)
def test_renames_match_original_model_names(display_names, reverse):
    df = pl.DataFrame(
        {
            "model": ["A", "B", "C"],
            "split": ["val", "val", "val"],
            "metric_win": [0.2, 0.5, 0.8],
        }
    )
    original = df.clone()
    models = [
        BenchmarkModelKey(name, display_name)
        for name, display_name in zip(df["model"], display_names)
    ]
    if reverse:
        models.reverse()

    result = filter_rename_model_split(df, models)

    expected = df.with_columns(pl.Series("model", display_names))
    assert_frame_equal(result, expected)
    assert_frame_equal(df, original)


@pytest.mark.parametrize(
    "display_names, expected_name",
    [
        (["B", "C"], "B"),
        (["A", "B"], "B"),
        (["B", "A"], "B"),
    ],
)
def test_duplicate_keys_use_first_non_identity_rename(display_names, expected_name):
    df = pl.DataFrame({"model": ["A"], "split": ["val"], "metric_win": [0.2]})
    models = [BenchmarkModelKey("A", name) for name in display_names]

    result = filter_rename_model_split(df, models)

    assert_frame_equal(result, df.with_columns(pl.lit(expected_name).alias("model")))


def test_filters_and_renames_model_split_pairs_preserving_row_order():
    df = pl.DataFrame(
        {
            "model": ["B", "A", "A", "C", "B", "A", "B"],
            "split": ["val", "train", "val", "val", "train", "val", "test"],
            "metric_win": [0.8, 0.1, 0.2, 0.9, 0.7, 0.3, 0.6],
        }
    )
    original = df.clone()
    models = [
        BenchmarkModelKey("A", "B"),
        BenchmarkModelKey("B", "C"),
        BenchmarkModelKey("B", "Training B", split="train"),
    ]

    result = filter_rename_model_split(df, models)

    expected = pl.DataFrame(
        {
            "model": ["C", "B", "Training B", "B"],
            "split": ["val", "val", "train", "val"],
            "metric_win": [0.8, 0.2, 0.7, 0.3],
        }
    )
    assert_frame_equal(result, expected)
    assert_frame_equal(df, original)


@pytest.mark.parametrize("models", [[], [BenchmarkModelKey("missing", "Renamed")]])
def test_empty_selection_preserves_schema(models):
    df = pl.DataFrame({"model": ["A"], "split": ["val"], "metric_win": [0.2]})

    result = filter_rename_model_split(df, models)

    assert_frame_equal(result, df.clear())


@pytest.mark.parametrize("df", [pl.DataFrame(), pl.DataFrame({"other": [1]})])
def test_empty_model_list_does_not_require_model_columns(df):
    assert_frame_equal(filter_rename_model_split(df, []), df.clear())


@pytest.mark.parametrize("chart_type", ["bar", "progress"])
def test_charts_keep_colliding_display_names_separate(chart_type):
    plt = pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("seaborn")
    from art.utils.benchmarking.charts import (
        percentage_comparison_bar_chart,
        training_progress_chart,
    )

    df = pl.DataFrame(
        {
            "model": ["A", "B"],
            "split": ["val", "val"],
            "step": [0, 0],
            "metric_win": [0.2, 0.8],
        }
    )
    models: list[str | BenchmarkModelKey] = [
        BenchmarkModelKey("A", "B"),
        BenchmarkModelKey("B", "C"),
    ]
    chart = (
        percentage_comparison_bar_chart
        if chart_type == "bar"
        else training_progress_chart
    )

    with plt.rc_context({"backend": "Agg"}):
        fig = chart(df, "win", models=models)
        try:
            ax = fig.axes[0]
            if chart_type == "bar":
                assert [label.get_text() for label in ax.get_xticklabels()] == [
                    "B",
                    "C",
                ]
                assert [bar.get_height() for bar in ax.patches] == pytest.approx(
                    [20, 80]
                )
            else:
                assert [line.get_label() for line in ax.lines] == ["B", "C"]
                assert list(ax.lines[0].get_ydata()) == pytest.approx([0.2, 0.2])
                assert list(ax.lines[1].get_ydata()) == pytest.approx([0.8, 0.8])
        finally:
            plt.close(fig)
