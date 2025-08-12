from datetime import datetime
from typing import Iterator, TypedDict, cast

import polars as pl
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
        .sample(fraction=1.0, shuffle=True, seed=42)
        .filter(
            ~pl.col("repo")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .is_in(
                [
                    f"swesmith/{repo}"
                    for repo in [
                        "encode__starlette.db5063c2",
                        "facebookresearch__hydra.0f03eb60",
                        "facelessuser__soupsieve.a8080d97",
                        "graphql-python__graphene.82903263",
                        "jawah__charset_normalizer.1fdd6463",
                        "jd__tenacity.0d40e76f",
                        "knio__dominate.9082227e",
                        "marshmallow-code__marshmallow.9716fc62",
                        "mido__mido.a0158ff9",
                        "pandas-dev__pandas.95280573",
                        "pallets__markupsafe.620c06c9",
                        "pydantic__pydantic.acb0f10f",
                        "pylint-dev__astroid.b114f6b5",
                        "pyupio__safety.7654596b",
                        "tornadoweb__tornado.d5ac65c1",
                        "tox-dev__pipdeptree.c31b6418",
                        "un33k__python-slugify.872b3750",
                    ]
                ]
            )
        )
        .filter(
            ~pl.col("instance_id")
            .cast(pl.Utf8)
            .str.to_lowercase()
            .is_in(
                [
                    "dask__dask.5f61e423.pr_10746",
                    # Tests passing when they should fail (patch doesn't introduce bug)
                    "kayak__pypika.1c9646f0.func_basic__09ty2nm3",
                    "sunpy__sunpy.f8edfd5c.combine_module__cjeldg50",
                    "modin-project__modin.8c7799fd.func_pm_remove_cond__bso0lmmj",
                    "alecthomas__voluptuous.a7a55f83.lm_rewrite__9zfstktu",
                    "alecthomas__voluptuous.a7a55f83.func_basic__s03cvwpk",
                    # pandas: missing compiled C extensions (pandas._libs.pandas_parser)
                    # "pandas-dev__pandas.95280573.lm_rewrite__am6uh57m",
                    # "pandas-dev__pandas.95280573.pr_57116",
                    # "pandas-dev__pandas.95280573.combine_module__f5wnuvus",
                    # "pandas-dev__pandas.95280573.combine_module__bwhtr1n5",
                    # "pandas-dev__pandas.95280573.combine_file__i3dkumyn",
                    # "pandas-dev__pandas.95280573.lm_rewrite__36dwchy4",
                    # "pandas-dev__pandas.95280573.pr_57174",
                    # pypika: no tests collected (same issue as other pypika instances)
                    "kayak__pypika.1c9646f0.lm_rewrite__nqk3tt63",
                    "kayak__pypika.1c9646f0.lm_rewrite__ddbee9oz",
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
