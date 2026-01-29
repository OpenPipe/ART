"""
Parquet-based trajectory logging.

This module provides efficient Parquet serialization for trajectory data,
offering ~25x compression compared to JSONL and fast columnar queries.

For legacy JSONL support and migration utilities, see trajectory_migration.py.
"""

import json
from pathlib import Path
from typing import Any, cast

from litellm.types.utils import Choices
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from art.trajectories import Trajectory, TrajectoryGroup


def _flatten_message(msg: dict) -> dict:
    """Convert a message or Choice to flat parquet format."""
    if "finish_reason" in msg:
        # Choice format - extract inner message, mark as trainable
        inner = msg.get("message", {})
        tool_calls = inner.get("tool_calls")
        return {
            "role": inner.get("role"),
            "content": inner.get("content"),
            "tool_calls": json.dumps(tool_calls) if tool_calls else None,
            "tool_call_id": None,
            "trainable": True,
        }
    else:
        # Regular message
        tool_calls = msg.get("tool_calls")
        return {
            "role": msg.get("role"),
            "content": msg.get("content"),
            "tool_calls": json.dumps(tool_calls) if tool_calls else None,
            "tool_call_id": msg.get("tool_call_id"),
            "trainable": False,
        }


def _unflatten_message(msg_dict: dict) -> dict:
    """Convert flat parquet format back to message dict."""
    result = {
        "role": msg_dict["role"],
        "content": msg_dict["content"],
    }
    if msg_dict.get("tool_calls"):
        result["tool_calls"] = json.loads(msg_dict["tool_calls"])
    if msg_dict.get("tool_call_id"):
        result["tool_call_id"] = msg_dict["tool_call_id"]
    return result


def write_trajectory_groups_parquet(
    trajectory_groups: list[TrajectoryGroup],
    path: str | Path,
) -> None:
    """
    Write trajectory groups to a Parquet file with ZSTD compression.

    Args:
        trajectory_groups: List of TrajectoryGroup objects to serialize.
        path: Output path for the Parquet file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for group_index, group in enumerate(trajectory_groups):
        group_metadata = json.dumps(group.metadata) if group.metadata else None
        group_metrics = json.dumps(group.metrics) if group.metrics else None
        group_logs = group.logs if group.logs else None
        for trajectory in group.trajectories:
            if not isinstance(trajectory, Trajectory):
                continue

            # Flatten messages
            messages = []
            for message_or_choice in trajectory.messages_and_choices:
                if isinstance(message_or_choice, Choice):
                    message = message_or_choice.to_dict()
                elif isinstance(message_or_choice, Choices):
                    message = {
                        "finish_reason": message_or_choice.finish_reason,
                        "index": message_or_choice.index,
                        "message": message_or_choice.message.to_dict()
                        if hasattr(message_or_choice.message, "to_dict")
                        else message_or_choice.message,
                    }
                else:
                    message = message_or_choice
                messages.append(_flatten_message(message))  # type: ignore

            rows.append(
                {
                    "group_index": group_index,
                    "group_metadata": group_metadata,
                    "group_metrics": group_metrics,
                    "group_logs": group_logs,
                    "reward": trajectory.reward,
                    "metrics": json.dumps(trajectory.metrics)
                    if trajectory.metrics
                    else None,
                    "metadata": json.dumps(trajectory.metadata)
                    if trajectory.metadata
                    else None,
                    "tools": json.dumps(trajectory.tools) if trajectory.tools else None,
                    "logs": trajectory.logs if trajectory.logs else None,
                    "messages": messages,
                }
            )

    # Define schema
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
            ("group_metadata", pa.string()),
            ("group_metrics", pa.string()),
            ("group_logs", pa.list_(pa.string())),
            ("reward", pa.float64()),
            ("metrics", pa.string()),
            ("metadata", pa.string()),
            ("tools", pa.string()),
            ("logs", pa.list_(pa.string())),
            ("messages", pa.list_(message_type)),
        ]
    )

    if not rows:
        table = pa.table({name: [] for name in schema.names}, schema=schema)
        pq.write_table(table, path, compression="zstd")
        return

    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def read_trajectory_groups_parquet(path: str | Path) -> list[TrajectoryGroup]:
    """
    Read trajectory groups from a Parquet file.

    Args:
        path: Path to the Parquet file.

    Returns:
        List of TrajectoryGroup objects.
    """
    import duckdb

    con = duckdb.connect(":memory:")
    rows = con.execute(f"SELECT * FROM '{path}' ORDER BY group_index").fetchall()
    columns = [desc[0] for desc in con.description]

    groups_dict: dict[int, list[Trajectory]] = {}
    group_metadata_by_index: dict[int, dict[str, Any]] = {}
    group_metrics_by_index: dict[int, dict[str, Any]] = {}
    group_logs_by_index: dict[int, list[str]] = {}

    def _load_json_payload(payload: object | None) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        if isinstance(payload, (str, bytes, bytearray)):
            if not payload:
                return {}
            try:
                return json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    for row in rows:
        row_dict = dict(zip(columns, row))

        if row_dict.get("reward") is None and row_dict.get("messages") is None:
            continue

        group_index = row_dict.get("group_index", 0)
        if group_index not in group_metadata_by_index:
            group_metadata_by_index[group_index] = _load_json_payload(
                row_dict.get("group_metadata")
            )
        if group_index not in group_metrics_by_index:
            group_metrics_by_index[group_index] = _load_json_payload(
                row_dict.get("group_metrics")
            )
        if group_index not in group_logs_by_index:
            raw_group_logs = row_dict.get("group_logs")
            if isinstance(raw_group_logs, (list, tuple)):
                group_logs_by_index[group_index] = [
                    str(item) for item in raw_group_logs
                ]
            elif raw_group_logs is None:
                group_logs_by_index[group_index] = []
            else:
                group_logs_by_index[group_index] = [str(raw_group_logs)]

        # Convert messages
        messages_and_choices = []
        for msg in row_dict.get("messages") or []:
            # Handle tuple format from DuckDB
            if not isinstance(msg, dict):
                msg = {
                    "role": msg[0],
                    "content": msg[1],
                    "tool_calls": msg[2],
                    "tool_call_id": msg[3],
                    "trainable": msg[4],
                }
            messages_and_choices.append(_unflatten_message(msg))

        trajectory = Trajectory(
            messages_and_choices=messages_and_choices,
            reward=row_dict["reward"],
            metrics=json.loads(row_dict["metrics"]) if row_dict.get("metrics") else {},
            metadata=json.loads(row_dict["metadata"])
            if row_dict.get("metadata")
            else {},
            logs=row_dict.get("logs") or [],
        )

        if group_index not in groups_dict:
            groups_dict[group_index] = []
        groups_dict[group_index].append(trajectory)

    return [
        TrajectoryGroup(
            trajectories=groups_dict[idx],
            exceptions=[],
            metadata=group_metadata_by_index.get(idx, {}),
            metrics=group_metrics_by_index.get(idx, {}),
            logs=group_logs_by_index.get(idx, []),
        )
        for idx in sorted(groups_dict.keys())
    ]
