try:
    import polars as pl
except ImportError:
    raise ImportError(
        "Plotting dependencies are not installed. Please install them with: "
        "pip install openpipe-art[plotting]"
    )

from art.utils.benchmarking.types import BenchmarkModelKey


def filter_rename_model_split(
    df: pl.DataFrame, models: list[BenchmarkModelKey]
) -> pl.DataFrame:
    # filter by combinations of name + split
    z = pl.fold(
        acc=pl.lit(False),
        function=lambda acc, expr: acc | expr,
        exprs=[
            (pl.col("model") == model.name) & (pl.col("split") == model.split)
            for model in models
        ],
    )

    df = df.filter(z)
    if not models:
        return df

    # Evaluate every rename against the original names, not earlier display names.
    # Reverse the order so the first non-identity rename wins for duplicate keys.
    model_name = pl.col("model")
    for model in reversed(models):
        if model.name != model.display_name:
            model_name = (
                pl.when(
                    (pl.col("model") == model.name) & (pl.col("split") == model.split)
                )
                .then(pl.lit(model.display_name))
                .otherwise(model_name)
            )

    return df.with_columns(model_name.alias("model"))
