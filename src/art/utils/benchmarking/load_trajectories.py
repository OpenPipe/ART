import polars as pl
import yaml
from pathlib import Path
from panza import SQLiteCache

from art.utils.get_repo_root_path import get_repo_root_path
from art.utils.output_dirs import (
    get_default_art_path,
    get_models_dir,
    get_trajectories_dir,
)

cache_path = Path(get_repo_root_path()) / "data" / "cache.db"
cache_path.parent.mkdir(parents=True, exist_ok=True)
cache = SQLiteCache(str(cache_path))


@cache.cache()
async def load_trajectories(
    project_name: str,
    models: list[str] | None = None,
    debug: bool = False,
    art_path: str | None = None,
) -> pl.DataFrame:
    """
      Load and flatten trajectory YAML files into a Polars DataFrame.

      The expected on-disk layout is::

          {api_path}/{project_name}/models/{model_name}/trajectories/{split}/{step_number}.yaml

      Each YAML file contains a list of *TrajectoryGroups* (see `art`), and each
      group in turn contains a list of *Trajectories*.  This helper walks the
      directory tree, reads every YAML file, and converts every single trajectory
      into one row of a tabular dataset.

      For every trajectory we record a handful of fixed columns plus two dynamic
      families of columns:

      Fixed columns
      -------------
      model : str
          Name of the model that produced the trajectory (taken from the folder
          name under ``models/``).
    split : str
          Split name extracted from the folder name under ``trajectories/``.
      step : int
          Training / evaluation step extracted from the YAML filename.
      reward : float | None
          Reward associated with the trajectory.
      group_number : int
          Running counter that uniquely identifies the surrounding trajectory
          group within the current parsing session (useful for debugging / joins).
      messages : list[str] | None
          Raw list of messages & choices for the dialogue.
      logs : list[str] | None
          Internal log lines captured during rollout.

      Dynamic columns
      ---------------
      metric_* : float
          One column for every distinct metric key found in the dataset.  Missing
          values are filled with nulls.
      metadata_* : str
          One column for every distinct metadata key (after merging group- and
          trajectory-level metadata).  Values are coerced to strings so that the
          resulting table is rectangular.

      Parameters
      ----------
      project_path : str
          Path to the ART project on disk. Typically found in `.art/{project_name}`.
      debug : bool, optional
          If *True*, the function prints progress information while parsing.  The
          default is *False*.

      Returns
      -------
      pl.DataFrame
          A Polars DataFrame containing one row per trajectory with the schema
          described above.
    """
    rows: list[dict] = []
    metric_cols: set[str] = set()
    metadata_cols: set[str] = set()

    if art_path is None:
        art_path = get_default_art_path()

    root = Path(get_models_dir(project_name=project_name, art_path=art_path))
    group_number = 0

    # Normalize the optional *models* argument for quick membership tests
    models_set: set[str] | None = set(models) if models is not None else None

    # Walk through all models and their trajectory files
    for model_dir in root.iterdir():
        if debug:
            print(f"Processing {model_dir}")
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # If a subset of models is requested, skip any others early to save time
        if models_set is not None and model_name not in models_set:
            continue

        traj_root = Path(get_trajectories_dir(model_dir))
        if not traj_root.exists():
            continue

        # Iterate over splits (e.g. train/val)
        for split_dir in traj_root.iterdir():
            if not split_dir.is_dir():
                continue

            for yaml_path in split_dir.glob("*.yaml"):
                step = int(yaml_path.stem)
                if debug:
                    print(f"Processing {yaml_path}")

                # Each YAML file is a list of trajectory groups
                trajectory_groups = yaml.safe_load(yaml_path.read_text())

                for group in trajectory_groups:
                    group_number += 1
                    group_meta = group.get("metadata", {})
                    for traj in group.get("trajectories", []):
                        # Merge metadata downward (group metadata < traj metadata wins)
                        merged_meta = {**group_meta, **traj.get("metadata", {})}

                        metrics = traj.get("metrics", {})

                        prepped_metrics = {f"metric_{k}": v for k, v in metrics.items()}
                        prepped_metadata = {
                            f"metadata_{k}": str(v) for k, v in merged_meta.items()
                        }
                        metric_cols.update(prepped_metrics.keys())
                        metadata_cols.update(prepped_metadata.keys())
                        messages = []
                        for message in traj.get("messages_and_choices", []):
                            if "message" in message:
                                messages.append(
                                    {**message["message"], "trainable": True}
                                )
                            else:
                                messages.append({**message, "trainable": False})

                        row: dict[str, object] = {
                            "model": model_name,
                            "split": split_dir.name,
                            "step": step,
                            "reward": traj.get("reward"),
                            "group_number": group_number,
                            "messages": messages,
                            "logs": traj.get("logs"),
                            **prepped_metrics,
                            **prepped_metadata,
                        }

                        rows.append(row)

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
    )

    return pl.DataFrame(rows, schema=schema)
