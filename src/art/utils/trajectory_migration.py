"""
Utilities for migrating trajectory files from JSONL to Parquet format.

This module provides functions to:
1. Migrate individual JSONL files to Parquet
2. Migrate entire model directories
3. Migrate all models in a project
4. Legacy JSONL serialization/deserialization (for backwards compatibility)

The migration provides ~25x compression and ~20x faster query performance.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterator, cast

import yaml

from art.trajectories import History, Trajectory, TrajectoryGroup
from art.types import Choice, Message, MessageOrChoice

# ============================================================================
# Legacy JSONL serialization helpers
# ============================================================================


def serialize_trajectory_groups(trajectory_groups: list[TrajectoryGroup]) -> str:
    """Serialize trajectory groups to a JSONL string."""
    group_dicts = [
        trajectory_group_to_dict(trajectory_group)
        for trajectory_group in trajectory_groups
    ]
    return "\n".join(json.dumps(group_dict) for group_dict in group_dicts)


def trajectory_group_to_dict(trajectory_group: TrajectoryGroup) -> dict[str, Any]:
    trajectory_dicts = []
    for trajectory in trajectory_group.trajectories:
        if not isinstance(trajectory, Trajectory):
            # remove exceptions
            continue
        trajectory_dicts.append(trajectory_to_dict(trajectory))

    return {
        "trajectories": trajectory_dicts,
    }


def history_to_dict(history: History) -> dict[str, Any]:
    messages_and_choices = [
        message_or_choice_to_dict(message_or_choice)
        for message_or_choice in history.messages_and_choices
    ]
    return {"messages_and_choices": messages_and_choices, "tools": history.tools}


def trajectory_to_dict(trajectory: Trajectory) -> dict[str, Any]:
    messages_and_choices = [
        message_or_choice_to_dict(message_or_choice)
        for message_or_choice in trajectory.messages_and_choices
    ]

    return {
        "reward": trajectory.reward,
        "metrics": trajectory.metrics,
        "metadata": trajectory.metadata,
        "messages_and_choices": messages_and_choices,
        "tools": trajectory.tools,
        "additional_histories": (
            [history_to_dict(h) for h in trajectory.additional_histories]
            if trajectory.additional_histories
            else trajectory.additional_histories
        ),
        "logs": trajectory.logs,
    }


def message_or_choice_to_dict(message_or_choice: MessageOrChoice) -> dict[str, Any]:
    # messages are sometimes stored as dicts, so we need to handle both cases
    item_dict = (
        message_or_choice
        if isinstance(message_or_choice, dict)
        else message_or_choice.to_dict()  # ty:ignore[possibly-missing-attribute]
    )

    if "logprobs" in item_dict:
        # item is a choice with logprobs, remove the logprobs
        item_dict.pop("logprobs")  # ty:ignore[invalid-argument-type]

    if "content" in item_dict and isinstance(item_dict["content"], Iterator):
        item_dict["content"] = list(item_dict["content"])  # type: ignore

    return dict(item_dict)  # ty:ignore[no-matching-overload]


def deserialize_trajectory_groups(serialized: str) -> list[TrajectoryGroup]:
    """Deserialize trajectory groups from a JSONL or YAML string."""
    # Try to parse as JSONL first (new format)
    try:
        loaded_groups = [
            json.loads(line) for line in serialized.strip().split("\n") if line
        ]
    except json.JSONDecodeError:
        # Fall back to YAML parsing (old format)
        loaded_groups = yaml.load(serialized, Loader=yaml.SafeLoader)
    return [dict_to_trajectory_group(group) for group in loaded_groups]


def dict_to_trajectory_group(d: dict[str, Any]) -> TrajectoryGroup:
    return TrajectoryGroup(
        trajectories=[
            dict_to_trajectory(trajectory) for trajectory in d["trajectories"]
        ],
        exceptions=[],
    )


def dict_to_trajectory(d: dict[str, Any]) -> Trajectory:
    return Trajectory(
        messages_and_choices=[
            dict_to_message_or_choice(message_or_choice)
            for message_or_choice in d["messages_and_choices"]
        ],
        reward=d["reward"],
        metrics=d["metrics"],
        metadata=d["metadata"],
        logs=d["logs"],
    )


def dict_to_message_or_choice(d: dict[str, Any]) -> MessageOrChoice:
    if "message" in d:
        return Choice(**d)
    else:
        return cast(Message, d)


# ============================================================================
# Migration utilities
# ============================================================================


@dataclass
class MigrationResult:
    """Results from a migration operation."""

    files_migrated: int = 0
    files_skipped: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Return compression ratio (original_size / new_size)."""
        if self.bytes_after == 0:
            return 0.0
        return self.bytes_before / self.bytes_after

    @property
    def space_saved(self) -> int:
        """Return bytes saved by compression."""
        return self.bytes_before - self.bytes_after

    def __add__(self, other: "MigrationResult") -> "MigrationResult":
        """Combine two migration results."""
        return MigrationResult(
            files_migrated=self.files_migrated + other.files_migrated,
            files_skipped=self.files_skipped + other.files_skipped,
            bytes_before=self.bytes_before + other.bytes_before,
            bytes_after=self.bytes_after + other.bytes_after,
            errors=self.errors + other.errors,
        )


def migrate_jsonl_to_parquet(
    jsonl_path: Path | str,
    delete_original: bool = True,
    dry_run: bool = False,
) -> MigrationResult:
    """
    Migrate a single JSONL trajectory file to Parquet format.

    Args:
        jsonl_path: Path to the JSONL file to migrate.
        delete_original: Whether to delete the original JSONL file after successful migration.
        dry_run: If True, only report what would be done without making changes.

    Returns:
        MigrationResult with statistics about the migration.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    jsonl_path = Path(jsonl_path)
    result = MigrationResult()

    if not jsonl_path.exists():
        result.errors.append(f"File not found: {jsonl_path}")
        return result

    if jsonl_path.suffix != ".jsonl":
        result.files_skipped += 1
        return result

    parquet_path = jsonl_path.with_suffix(".parquet")

    # Get original size
    original_size = jsonl_path.stat().st_size
    result.bytes_before = original_size

    if dry_run:
        result.files_migrated = 1
        # Estimate compression (typically ~25x for trajectory data)
        result.bytes_after = original_size // 25
        return result

    try:
        # Read JSONL file
        trajectory_groups_data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    trajectory_groups_data.append(json.loads(line))

        # Convert to flat rows for Parquet
        rows = []
        for group_index, group in enumerate(trajectory_groups_data):
            for traj in group.get("trajectories", []):
                # Flatten messages
                messages = []
                for msg in traj.get("messages_and_choices", []):
                    if "finish_reason" in msg:
                        # Choice format - extract inner message, mark as trainable
                        inner = msg.get("message", {})
                        messages.append(
                            {
                                "role": inner.get("role"),
                                "content": inner.get("content"),
                                "tool_calls": json.dumps(inner.get("tool_calls"))
                                if inner.get("tool_calls")
                                else None,
                                "tool_call_id": None,
                                "trainable": True,
                            }
                        )
                    else:
                        # Regular message
                        messages.append(
                            {
                                "role": msg.get("role"),
                                "content": msg.get("content"),
                                "tool_calls": json.dumps(msg.get("tool_calls"))
                                if msg.get("tool_calls")
                                else None,
                                "tool_call_id": msg.get("tool_call_id"),
                                "trainable": False,
                            }
                        )

                rows.append(
                    {
                        "group_index": group_index,
                        "reward": traj.get("reward"),
                        "metrics": json.dumps(traj.get("metrics"))
                        if traj.get("metrics")
                        else None,
                        "metadata": json.dumps(traj.get("metadata"))
                        if traj.get("metadata")
                        else None,
                        "tools": json.dumps(traj.get("tools"))
                        if traj.get("tools")
                        else None,
                        "logs": traj.get("logs"),
                        "additional_histories": json.dumps(
                            traj.get("additional_histories")
                        )
                        if traj.get("additional_histories")
                        else None,
                        "messages": messages,
                    }
                )

        # Define the message struct schema
        message_type = pa.struct(
            [
                ("role", pa.string()),
                ("content", pa.string()),
                ("tool_calls", pa.string()),
                ("tool_call_id", pa.string()),
                ("trainable", pa.bool_()),
            ]
        )

        schema = pa.schema(
            [
                ("group_index", pa.int64()),
                ("reward", pa.float64()),
                ("metrics", pa.string()),
                ("metadata", pa.string()),
                ("tools", pa.string()),
                ("logs", pa.list_(pa.string())),
                ("additional_histories", pa.string()),
                ("messages", pa.list_(message_type)),
            ]
        )

        # Handle empty case
        if not rows:
            table = pa.table({name: [] for name in schema.names}, schema=schema)
            pq.write_table(table, parquet_path, compression="zstd")
        else:
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, parquet_path, compression="zstd")

        # Get new size
        new_size = parquet_path.stat().st_size
        result.bytes_after = new_size
        result.files_migrated = 1

        # Delete original if requested
        if delete_original:
            jsonl_path.unlink()

    except Exception as e:
        result.errors.append(f"Error migrating {jsonl_path}: {e}")
        # Clean up partial parquet file if it exists
        if parquet_path.exists():
            parquet_path.unlink()

    return result


def migrate_trajectories_dir(
    trajectories_dir: Path | str,
    delete_originals: bool = True,
    dry_run: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> MigrationResult:
    """
    Migrate all JSONL files in a trajectories directory (including subdirectories).

    Args:
        trajectories_dir: Path to the trajectories directory.
        delete_originals: Whether to delete original JSONL files after migration.
        dry_run: If True, only report what would be done.
        progress_callback: Optional callback for progress updates.

    Returns:
        Combined MigrationResult for all files.
    """
    trajectories_dir = Path(trajectories_dir)
    result = MigrationResult()

    if not trajectories_dir.exists():
        result.errors.append(f"Directory not found: {trajectories_dir}")
        return result

    # Find all JSONL files
    jsonl_files = list(trajectories_dir.rglob("*.jsonl"))

    for jsonl_path in jsonl_files:
        if progress_callback:
            progress_callback(str(jsonl_path))

        file_result = migrate_jsonl_to_parquet(
            jsonl_path,
            delete_original=delete_originals,
            dry_run=dry_run,
        )
        result = result + file_result

    return result


def migrate_model_dir(
    model_dir: Path | str,
    delete_originals: bool = True,
    dry_run: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> MigrationResult:
    """
    Migrate all trajectory files for a model.

    Args:
        model_dir: Path to the model directory (containing trajectories/ subdirectory).
        delete_originals: Whether to delete original JSONL files after migration.
        dry_run: If True, only report what would be done.
        progress_callback: Optional callback for progress updates.

    Returns:
        Combined MigrationResult for all files.
    """
    model_dir = Path(model_dir)
    trajectories_dir = model_dir / "trajectories"

    if not trajectories_dir.exists():
        return MigrationResult()

    return migrate_trajectories_dir(
        trajectories_dir,
        delete_originals=delete_originals,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )


def migrate_all_models(
    art_path: Path | str,
    project_name: str,
    delete_originals: bool = True,
    dry_run: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> MigrationResult:
    """
    Migrate all trajectory files for all models in a project.

    Args:
        art_path: Path to the .art directory.
        project_name: Name of the project.
        delete_originals: Whether to delete original JSONL files after migration.
        dry_run: If True, only report what would be done.
        progress_callback: Optional callback for progress updates.

    Returns:
        Combined MigrationResult for all files.
    """
    art_path = Path(art_path)
    models_dir = art_path / project_name / "models"

    result = MigrationResult()

    if not models_dir.exists():
        result.errors.append(f"Models directory not found: {models_dir}")
        return result

    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            if progress_callback:
                progress_callback(f"Processing model: {model_dir.name}")

            model_result = migrate_model_dir(
                model_dir,
                delete_originals=delete_originals,
                dry_run=dry_run,
                progress_callback=progress_callback,
            )
            result = result + model_result

    return result


def auto_migrate_on_register(model_dir: Path | str) -> MigrationResult:
    """
    Automatically migrate any JSONL files found when a model is registered.

    This is called by the backend during model registration to ensure
    all trajectories are in the new Parquet format. Prints a summary
    if any files were migrated.

    Args:
        model_dir: Path to the model directory.

    Returns:
        MigrationResult with statistics (empty if no migration needed).
    """
    result = migrate_model_dir(
        model_dir,
        delete_originals=True,
        dry_run=False,
    )

    if result.files_migrated > 0:
        print(
            f"Migrated {result.files_migrated} trajectory files to Parquet "
            f"(saved {result.space_saved / 1024 / 1024:.1f} MB)"
        )

    return result
