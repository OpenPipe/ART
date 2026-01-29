try:
    import polars as pl
except ImportError:
    raise ImportError(
        "Plotting dependencies are not installed. Please install them with: "
        "pip install openpipe-art[plotting]"
    )

import json
from pathlib import Path

import duckdb
from tqdm.auto import tqdm
import yaml

from art.utils.get_repo_root_path import get_repo_root_path
from art.utils.output_dirs import (
    get_default_art_path,
    get_models_dir,
    get_trajectories_dir,
)

cache_path = Path(get_repo_root_path()) / "data" / "cache.db"
cache_path.parent.mkdir(parents=True, exist_ok=True)


async def load_trajectories(
    project_name: str,
    models: list[str] | None = None,
    debug: bool = False,
    art_path: str | None = None,
) -> pl.DataFrame:
    """
    Load and flatten trajectory files (Parquet) into a Polars DataFrame.

    The expected on-disk layout is::

        {art_path}/{project_name}/models/{model_name}/trajectories/{split}/{step_number}.parquet

    Each file contains trajectory data with the following columns:
    - reward: float
    - metrics: JSON string
    - metadata: JSON string
    - tools: JSON string (nullable)
    - logs: list of strings
    - messages: list of message structs

    This function reads all Parquet files efficiently using DuckDB and returns
    a flattened Polars DataFrame with one row per trajectory.

    Fixed columns
    -------------
    model : str
        Name of the model that produced the trajectory.
    split : str
        Split name (e.g., 'train', 'val').
    step : int
        Training / evaluation step extracted from the filename.
    reward : float | None
        Reward associated with the trajectory.
    messages : list[dict] | None
        List of messages and choices for the dialogue.
    logs : list[str] | None
        Internal log lines captured during rollout.

    Dynamic columns
    ---------------
    metric_* : float
        One column for every distinct metric key found in the dataset.
    metadata_* : str
        One column for every distinct metadata key.
    group_metric_* : float
        One column for every distinct group-level metric key.
    group_metadata_* : str
        One column for every distinct group-level metadata key.

    Parameters
    ----------
    project_name : str
        Name of the project to load trajectories from.
    models : list[str] | None, optional
        List of model names to load. If None, loads all models.
    debug : bool, optional
        If True, prints progress information.
    art_path : str | None, optional
        Path to the .art directory. If None, uses default.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing one row per trajectory.
    """
    if art_path is None:
        art_path = get_default_art_path()

    root = Path(get_models_dir(project_name=project_name, art_path=art_path))

    if not root.exists():
        return pl.DataFrame()

    # Determine which models to process
    models_set: set[str] | None = set(models) if models is not None else None
    model_dirs = [
        d
        for d in root.iterdir()
        if d.is_dir() and (models_set is None or d.name in models_set)
    ]

    if not model_dirs:
        return pl.DataFrame()

    # Collect all parquet files
    all_parquet_files: list[
        tuple[str, str, str, int]
    ] = []  # (path, model, split, step)

    for model_dir in tqdm(
        model_dirs, desc="Scanning models", unit="model", disable=not debug
    ):
        model_name = model_dir.name
        traj_root = Path(get_trajectories_dir(str(model_dir)))

        if not traj_root.exists():
            continue

        for split_dir in traj_root.iterdir():
            if not split_dir.is_dir():
                continue

            for trajectory_path in split_dir.glob("*.parquet"):
                try:
                    step = int(trajectory_path.stem)
                    all_parquet_files.append(
                        (
                            str(trajectory_path),
                            model_name,
                            split_dir.name,
                            step,
                        )
                    )
                except ValueError:
                    continue

    if not all_parquet_files:
        return pl.DataFrame()

    # Use DuckDB to read all parquet files efficiently
    rows: list[dict] = []
    metric_cols: set[str] = set()
    metadata_cols: set[str] = set()
    group_metric_cols: set[str] = set()
    group_metadata_cols: set[str] = set()
    # Map (model, split, step, group_index) -> unique group_number
    group_key_to_number: dict[tuple[str, str, int, int], int] = {}
    next_group_number = 1

    con = duckdb.connect(":memory:")

    for file_path, model_name, split_name, step in tqdm(
        all_parquet_files,
        desc="Loading trajectories",
        unit="file",
        disable=not debug,
    ):
        try:
            result = con.execute(f"SELECT * FROM '{file_path}'").fetchall()
            columns = [desc[0] for desc in con.description]
        except Exception as e:
            if debug:
                print(f"Error reading {file_path}: {e}")
            continue

        for row in result:
            row_dict = dict(zip(columns, row))

            # Skip empty rows
            if row_dict.get("reward") is None and row_dict.get("messages") is None:
                continue

            # Get group_index from parquet, default to 0 for backwards compatibility
            group_index = row_dict.get("group_index", 0)
            group_key = (model_name, split_name, step, group_index)
            if group_key not in group_key_to_number:
                group_key_to_number[group_key] = next_group_number
                next_group_number += 1
            group_number = group_key_to_number[group_key]

            # Parse metrics from JSON
            metrics = {}
            if row_dict.get("metrics"):
                try:
                    metrics = json.loads(row_dict["metrics"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Parse metadata from JSON
            metadata = {}
            if row_dict.get("metadata"):
                try:
                    metadata = json.loads(row_dict["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Parse group metrics from JSON (duplicated across group rows)
            group_metrics = {}
            if row_dict.get("group_metrics"):
                try:
                    group_metrics = json.loads(row_dict["group_metrics"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Parse group metadata from JSON (duplicated across group rows)
            group_metadata = {}
            if row_dict.get("group_metadata"):
                try:
                    group_metadata = json.loads(row_dict["group_metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # Prepare metrics and metadata columns
            prepped_metrics = {f"metric_{k}": v for k, v in metrics.items()}
            prepped_metadata = {f"metadata_{k}": str(v) for k, v in metadata.items()}
            prepped_group_metrics = {
                f"group_metric_{k}": v for k, v in group_metrics.items()
            }
            prepped_group_metadata = {
                f"group_metadata_{k}": str(v) for k, v in group_metadata.items()
            }
            metric_cols.update(prepped_metrics.keys())
            metadata_cols.update(prepped_metadata.keys())
            group_metric_cols.update(prepped_group_metrics.keys())
            group_metadata_cols.update(prepped_group_metadata.keys())

            # Process messages
            messages = []
            raw_messages = row_dict.get("messages") or []
            for msg in raw_messages:
                if isinstance(msg, dict):
                    msg_dict = msg
                else:
                    # Handle tuple format from DuckDB struct
                    # New format has 5 fields: role, content, tool_calls, tool_call_id, trainable
                    # Old format has 6 fields: role, content, tool_calls, tool_call_id, finish_reason, choice_index
                    if len(msg) == 5:
                        # New format with trainable
                        msg_dict = {
                            "role": msg[0],
                            "content": msg[1],
                            "tool_calls": msg[2],
                            "tool_call_id": msg[3],
                            "trainable": msg[4],
                        }
                    else:
                        # Old format with finish_reason/choice_index
                        msg_dict = {
                            "role": msg[0],
                            "content": msg[1],
                            "tool_calls": msg[2],
                            "tool_call_id": msg[3],
                            "trainable": msg[4]
                            is not None,  # finish_reason present = trainable
                        }

                # Build processed message
                processed_msg = {
                    "role": msg_dict.get("role"),
                    "content": msg_dict.get("content"),
                    "trainable": msg_dict.get("trainable", False),
                }
                if msg_dict.get("tool_calls"):
                    try:
                        processed_msg["tool_calls"] = json.loads(msg_dict["tool_calls"])  # ty:ignore[invalid-argument-type]
                    except (json.JSONDecodeError, TypeError):
                        pass

                messages.append(processed_msg)

            row_data: dict[str, object] = {
                "model": model_name,
                "split": split_name,
                "step": step,
                "reward": row_dict.get("reward"),
                "group_number": group_number,
                "messages": messages,
                "logs": row_dict.get("logs"),
                **prepped_metrics,
                **prepped_metadata,
                **prepped_group_metrics,
                **prepped_group_metadata,
            }

            rows.append(row_data)

    if not rows:
        return pl.DataFrame()

    # Build schema
    schema = (
        {
            "model": pl.Utf8,
            "split": pl.Utf8,
            "step": pl.Int64,
            "reward": pl.Float64,
            "group_number": pl.Int64,
            "messages": pl.List(
                pl.Struct(
                    {
                        "role": pl.Utf8,
                        "content": pl.Utf8,
                        "tool_calls": pl.List(
                            pl.Struct(
                                {
                                    "function": pl.Struct(
                                        {
                                            "name": pl.Utf8,
                                            "arguments": pl.Utf8,
                                        }
                                    )
                                }
                            )
                        ),
                        "trainable": pl.Boolean,
                    }
                )
            ),
            "logs": pl.List(pl.Utf8),
        }
        | {k: pl.Float64 for k in metric_cols}
        | {k: pl.Utf8 for k in metadata_cols}
        | {k: pl.Float64 for k in group_metric_cols}
        | {k: pl.Utf8 for k in group_metadata_cols}
    )

    return pl.DataFrame(rows, schema=schema)
