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
import warnings

import pydantic
import yaml

from art.trajectories import History, Trajectory, TrajectoryGroup
from art.types import Choice, Message, MessageOrChoice
from art.utils.trajectory_logging import write_trajectory_groups_parquet

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
    return {
        "trajectories": [
            trajectory_to_dict(trajectory)
            for trajectory in trajectory_group.trajectories
        ],
        **trajectory_group.model_dump(
            mode="json", exclude={"trajectories"}, warnings="error"
        ),
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
        **trajectory.model_dump(
            mode="json",
            exclude={
                "reward",
                "metrics",
                "metadata",
                "messages_and_choices",
                "tools",
                "additional_histories",
                "logs",
            },
            exclude_computed_fields=True,
            warnings="error",
        ),
    }


def message_or_choice_to_dict(message_or_choice: MessageOrChoice) -> dict[str, Any]:
    # messages are sometimes stored as dicts, so we need to handle both cases
    item_dict = (
        message_or_choice.to_dict()
        if isinstance(message_or_choice, Choice)
        else dict(message_or_choice)
    )

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
    return TrajectoryGroup.model_validate(
        {
            **d,
            "trajectories": [
                dict_to_trajectory(trajectory) for trajectory in d["trajectories"]
            ],
        }
    )


def dict_to_trajectory(d: dict[str, Any]) -> Trajectory:
    return Trajectory.model_validate(
        {
            **d,
            "messages_and_choices": [
                dict_to_message_or_choice(message_or_choice)
                for message_or_choice in d.get("messages_and_choices", [])
            ],
            "additional_histories": [
                dict_to_history(history)
                for history in d.get("additional_histories", [])
            ],
        }
    )


def dict_to_history(d: dict[str, Any]) -> History:
    return History.model_validate(
        {
            **d,
            "messages_and_choices": [
                dict_to_message_or_choice(message_or_choice)
                for message_or_choice in d.get("messages_and_choices", [])
            ],
        }
    )


def dict_to_message_or_choice(d: dict[str, Any]) -> MessageOrChoice:
    if "message" in d:
        try:
            return Choice(**d)
        except pydantic.ValidationError:
            return cast(Message, d)
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
        trajectory_groups_data: list[dict[str, Any]] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    trajectory_groups_data.append(json.loads(line))
        write_trajectory_groups_parquet(
            [dict_to_trajectory_group(group) for group in trajectory_groups_data],
            parquet_path,
        )

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
    if result.errors:
        warnings.warn("\n".join(result.errors), RuntimeWarning, stacklevel=2)

    return result
