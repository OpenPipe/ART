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

    for model in models:
        if model.name != model.display_name:
            df = df.with_columns(
                pl.when(
                    (pl.col("model") == model.name) & (pl.col("split") == model.split)
                )
                .then(pl.lit(model.display_name))
                .otherwise(pl.col("model"))
                .alias("model")
            )

    return df
