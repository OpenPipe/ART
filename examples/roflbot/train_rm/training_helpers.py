from typing import Optional
import torch  # type: ignore
import polars as pl  # type: ignore
from scipy.stats import pearsonr  # type: ignore
from sklearn.metrics import root_mean_squared_error  # type: ignore
from inference import run_inference_transformers, ModelOrPath, MandT, load_model
import math
import logging
from datetime import datetime
from datasets import Dataset, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
import os
from pydantic import BaseModel


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Convert numpy arrays to torch tensors
    predictions = torch.tensor(predictions).squeeze()
    labels = torch.tensor(labels)

    # Filter out NaN values
    valid_indices = ~torch.isnan(predictions) & ~torch.isnan(labels)
    valid_predictions = predictions[valid_indices]
    valid_labels = labels[valid_indices]

    return {
        "rmse": root_mean_squared_error(valid_labels, valid_predictions),
        "correlation": pearsonr(valid_labels, valid_predictions)[0],
    }


def run_final_inference_and_report_metrics(
    model_or_path: ModelOrPath, dataset: Dataset, output_dir: Optional[str] = None
):
    import wandb

    if output_dir is None:
        if not isinstance(model_or_path, str):
            raise ValueError(
                "output_dir is required when model_or_path is not a path string. Please provide an output directory."
            )
        output_dir = model_or_path

    predictions_path = f"{output_dir}/dataset_predictions.parquet"

    # Convert Hugging Face Dataset to a Polars DataFrame for easier manipulation
    df: pl.DataFrame = dataset.to_polars()  # type: ignore

    # Ensure a `split` column exists so downstream logic works for single-split datasets (e.g. test only)
    if "split" not in df.columns:
        df = df.with_columns(pl.lit("test").alias("split"))

    # Check if predictions file already exists
    if os.path.exists(predictions_path):
        print(f"Loading existing predictions from {predictions_path}")
        existing_predictions = pl.read_parquet(predictions_path)
        df = df.join(existing_predictions, on="id", how="left")
    else:
        mandt = load_model(model_or_path)
        model = mandt.model

        # If we're working with a PEFT wrapper, merge before running inference for speed
        if hasattr(model, "merge_and_unload"):
            print("Merging PEFT model with base model...")
            model = model.merge_and_unload()  # type: ignore

        predictions = run_inference_transformers(df["text"].to_list(), mandt)
        df = df.with_columns(pl.Series(name="predictions", values=predictions))
        df.select(["id", "predictions"]).write_parquet(predictions_path)

    metrics = calculate_metrics_by_split(df)

    print(metrics)

    for row in metrics.iter_rows(named=True):
        split = row["split"]
        wandb.summary.update(
            {
                f"final/{split}/baseline_rmse": row["baseline_rmse"],
                f"final/{split}/model_rmse": row["model_rmse"],
                f"final/{split}/correlation": row["model_correlation"],
            }
        )

    return metrics


def calculate_metrics_by_split(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate correlation and RMSE metrics for each split in the dataset.

    Args:
        df: DataFrame with label, predictions and split columns

    Returns:
        DataFrame with metrics for each split
    """
    metrics = []

    for split in df["split"].unique():
        split_df = df.filter(pl.col("split") == split)

        # Calculate baseline (mean) metrics
        average_score = split_df["label"].mean()
        rmse_baseline = math.sqrt(
            (split_df["label"] - average_score).pow(2).sum() / len(split_df)
        )

        # Calculate model metrics
        rmse_model = math.sqrt(
            (split_df["label"] - split_df["predictions"]).pow(2).sum() / len(split_df)
        )
        correlation_model = split_df.select(pl.corr("label", "predictions"))["label"][0]

        metrics.append(
            {
                "split": split,
                "baseline_rmse": rmse_baseline,
                "model_rmse": rmse_model,
                "model_correlation": correlation_model,
                "num_rows": len(split_df),
            }
        )

    return pl.DataFrame(metrics)


def format_joke_for_rm(
    title: str,
    text: str,
    created_at: datetime = datetime(2025, 1, 1),
    poster: str = "unknown",
) -> str:
    return f"Poster: {poster}\nTimestamp: {created_at.strftime('%A %b %d, %Y at %H:%M')}\n\nTitle: {title}\n\nText: {text}"


def get_dataset(
    split: str,
    tokenizer: PreTrainedTokenizer,
    max_len: int,
    max_rows: int | None = None,
    n_proc: int = 4,
) -> Dataset:
    """
    Loads and prepares the Reddit Jokes dataset for training.

    Args:
        split: The dataset split to load (e.g., "train", "test").
        tokenizer: The tokenizer to use for preparing the text.
        max_len: The maximum sequence length for filtering.
        n_proc: Number of processes for mapping operations.

    Returns:
        A processed Hugging Face Dataset object.
    """
    ds: Dataset = load_dataset("corbt/reddit_jokes", split=split)  # type: ignore

    if max_rows is not None:
        ds = ds.select(
            range(max_rows * 2)
        )  # grab 2x the rows so we can filter them down to the max length

    # Helper function to apply formatting
    def format_example(example):
        return {
            "text": format_joke_for_rm(
                title=example["title"],
                text=example["selftext"],
                created_at=example["created_at"],
                poster=example["author"],
            ),
            "label": example["log_score"],
        }

    # Apply formatting and rename score to label
    ds = ds.map(format_example, num_proc=n_proc)

    # Tokenize the text
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    ds = ds.map(tokenize_function, batched=True)

    ds = ds.filter(lambda example: len(example["input_ids"]) <= max_len)

    if max_rows is not None:
        ds = ds.select(range(max_rows))

    return ds


class RunConfig(BaseModel):
    run_name: str
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 4096
    val_size: int = 500
    base_model: str = "Qwen/Qwen2.5-0.5B"
    accelerator: str = "H100-SXM:1"


def get_s3_model_path(run_name: str) -> str:
    """Constructs the S3 path for storing/retrieving model artifacts."""
    remote_bucket = os.getenv("REMOTE_BUCKET")
    if not remote_bucket:
        raise ValueError("Environment variable REMOTE_BUCKET is not set.")
    return f"s3://{remote_bucket}/roflbot_rm/models/{run_name}"
