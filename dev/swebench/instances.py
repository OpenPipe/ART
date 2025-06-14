from datetime import datetime
import polars as pl
from typing import cast, Iterator, TypedDict
from typing_extensions import NotRequired


class Instance(TypedDict):
    instance_id: str
    repo: str
    patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    created_at: datetime
    image_name: str
    base_commit: str
    problem_statement: str
    use_swebench_modal_harness: bool
    test_patch: NotRequired[str]
    hints_text: NotRequired[str]
    version: NotRequired[str]
    environment_setup_commit: NotRequired[str]
    difficulty: NotRequired[str]


def as_instances_iter(df: pl.DataFrame) -> Iterator[Instance]:
    for row in df.iter_rows(named=True):
        yield cast(Instance, row)


def get_filtered_swe_smith_instances_df() -> pl.DataFrame:
    return (
        pl.read_parquet(
            "hf://datasets/bradhiltonendercorp/SWE-smith-filtered/instances.parquet"
        )
        .filter(
            ~pl.col("repo")
            .cast(pl.Utf8)
            .is_in(
                [
                    "swesmith/facebookresearch__hydra.0f03eb60",
                    "swesmith/jawah__charset_normalizer.1fdd6463",
                    "swesmith/marshmallow-code__marshmallow.9716fc62",
                    "swesmith/mido__mido.a0158ff9",
                    "swesmith/pydantic__pydantic.acb0f10f",
                ]
            )
        )
        .with_columns(
            base_commit=pl.col("instance_id"),
            image_name="jyangballin/"
            + pl.col("image_name").cast(pl.Utf8).str.replace("__", "_1776_"),
            use_swebench_modal_harness=False,
        )
    )


def get_swe_bench_verified_instances_df() -> pl.DataFrame:
    return pl.read_parquet(
        "hf://datasets/SWE-bench/SWE-bench_Verified/data/test-00000-of-00001.parquet"
    ).with_columns(
        created_at=pl.col("created_at").str.strptime(pl.Datetime),
        image_name="swebench/sweb.eval.x86_64."
        + pl.col("instance_id").str.replace("__", "_1776_"),
        use_swebench_modal_harness=True,
    )
