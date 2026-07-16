"""Versioned Parquet persistence for trajectory groups."""

from collections.abc import Mapping
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from openai.types.chat.chat_completion import Choice
import pydantic

from art.trajectories import History, Trajectory, TrajectoryGroup

if TYPE_CHECKING:
    import pyarrow as pa

_FORMAT_VERSION = 2
_JSON_VALUE = pydantic.TypeAdapter(Any)


def _flatten_message(message: Mapping[str, object]) -> dict[str, object]:
    """Convert a legacy message or Choice to the query-friendly v1 shape."""
    if "finish_reason" in message:
        inner = message.get("message", {})
        if not isinstance(inner, Mapping):
            inner = {}
        tool_calls = inner.get("tool_calls")
        return {
            "role": inner.get("role"),
            "content": inner.get("content"),
            "tool_calls": json.dumps(tool_calls) if tool_calls else None,
            "tool_call_id": None,
            "trainable": True,
        }
    tool_calls = message.get("tool_calls")
    return {
        "role": message.get("role"),
        "content": message.get("content"),
        "tool_calls": json.dumps(tool_calls) if tool_calls else None,
        "tool_call_id": message.get("tool_call_id"),
        "trainable": False,
    }


def _unflatten_message(message: Mapping[str, object]) -> dict[str, object]:
    """Convert the query-friendly v1 shape back to a legacy message."""
    result = {"role": message["role"], "content": message["content"]}
    if tool_calls := message.get("tool_calls"):
        result["tool_calls"] = json.loads(str(tool_calls))
    if tool_call_id := message.get("tool_call_id"):
        result["tool_call_id"] = tool_call_id
    return result


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _choice_data(item: object) -> dict[str, Any] | None:
    if isinstance(item, Choice):
        raw = item.model_dump(mode="json", warnings="error")
    elif (
        isinstance(item, Mapping)
        and {
            "finish_reason",
            "index",
            "message",
        }
        <= item.keys()
    ):
        raw = dict(item)
        try:
            return Choice.model_validate(raw).model_dump(mode="json", warnings="error")
        except pydantic.ValidationError:
            return None
    else:
        from litellm.types.utils import Choices

        if not isinstance(item, Choices):
            return None
        raw = item.model_dump(mode="json", warnings="error")
    return Choice.model_validate(raw).model_dump(mode="json", warnings="error")


def _history_data(history: History) -> dict[str, Any]:
    data = history.model_dump(
        mode="json", exclude={"messages_and_choices"}, warnings="error"
    )
    if history.messages_and_choices:
        data["messages_and_choices"] = [
            choice
            if (choice := _choice_data(item)) is not None
            else _JSON_VALUE.dump_python(item, mode="json", warnings="error")
            for item in history.messages_and_choices
        ]
    return data


def _trajectory_data(trajectory: Trajectory) -> dict[str, Any]:
    data = trajectory.model_dump(
        mode="json",
        exclude={"messages_and_choices", "additional_histories"},
        exclude_computed_fields=True,
        warnings="error",
    )
    if trajectory.messages_and_choices:
        data["messages_and_choices"] = [
            choice
            if (choice := _choice_data(item)) is not None
            else _JSON_VALUE.dump_python(item, mode="json", warnings="error")
            for item in trajectory.messages_and_choices
        ]
    if trajectory.additional_histories:
        data["additional_histories"] = [
            _history_data(history) for history in trajectory.additional_histories
        ]
    return data


def _restore_history(data: object) -> History:
    if not isinstance(data, dict):
        raise ValueError("Parquet additional history must be a JSON object")
    restored = dict(data)
    messages = restored.get("messages_and_choices", [])
    if not isinstance(messages, list):
        raise ValueError("Parquet history messages must be a list")
    restored["messages_and_choices"] = [
        Choice.model_validate(choice)
        if (choice := _choice_data(item)) is not None
        else item
        for item in messages
    ]
    return History.model_validate(restored)


def _restore_trajectory(payload: object) -> Trajectory:
    data = _json_object(payload, field="trajectory_json")
    messages = data.get("messages_and_choices", [])
    if not isinstance(messages, list):
        raise ValueError("Parquet trajectory messages must be a list")
    data["messages_and_choices"] = [
        Choice.model_validate(choice)
        if (choice := _choice_data(item)) is not None
        else item
        for item in messages
    ]
    histories = data.get("additional_histories", [])
    if not isinstance(histories, list):
        raise ValueError("Parquet additional_histories must be a list")
    data["additional_histories"] = [_restore_history(item) for item in histories]
    return Trajectory.model_validate(data)


def _legacy_messages(trajectory: Trajectory) -> list[dict[str, object]] | None:
    if trajectory.exchanges:
        return None
    messages = []
    for item in trajectory.messages_and_choices:
        if isinstance(item, Choice):
            message = item.to_dict()
        elif isinstance(item, Mapping):
            message = item
        else:
            from litellm.types.utils import Choices

            if not isinstance(item, Choices):
                return None
            message = {
                "finish_reason": item.finish_reason,
                "index": item.index,
                "message": (
                    item.message.to_dict()
                    if hasattr(item.message, "to_dict")
                    else item.message
                ),
            }
        try:
            flattened = _flatten_message(message)
        except (TypeError, ValueError):
            return None
        if flattened["content"] is not None and not isinstance(
            flattened["content"], str
        ):
            return None
        messages.append(flattened)
    return messages


def _schema() -> "pa.Schema":
    import pyarrow as pa

    message_type = pa.struct(
        [
            ("role", pa.string()),
            ("content", pa.string()),
            ("tool_calls", pa.string()),
            ("tool_call_id", pa.string()),
            ("trainable", pa.bool_()),
        ]
    )
    return pa.schema(
        [
            ("format_version", pa.int16()),
            ("group_index", pa.int64()),
            ("trajectory_index", pa.int64()),
            ("group_json", pa.large_string()),
            ("trajectory_json", pa.large_string()),
            # Stable materialized columns remain convenient for analytics. The JSON
            # payloads above are authoritative and preserve the complete models.
            ("group_metadata", pa.string()),
            ("group_metrics", pa.string()),
            ("group_logs", pa.list_(pa.string())),
            ("reward", pa.float64()),
            ("metrics", pa.string()),
            ("metadata", pa.string()),
            ("tools", pa.string()),
            ("logs", pa.list_(pa.string())),
            ("messages", pa.list_(message_type)),
        ],
        metadata={b"art.trajectory_parquet.version": str(_FORMAT_VERSION).encode()},
    )


def write_trajectory_groups_parquet(
    trajectory_groups: list[TrajectoryGroup],
    path: str | Path,
) -> None:
    """Persist complete trajectory groups with Zstd-compressed Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[dict[str, object]] = []
    for group_index, group in enumerate(trajectory_groups):
        group_data = group.model_dump(
            mode="json", exclude={"trajectories"}, warnings="error"
        )
        common: dict[str, object] = {
            "format_version": _FORMAT_VERSION,
            "group_index": group_index,
            "group_json": _compact_json(group_data),
            "group_metadata": (
                _compact_json(group_data["metadata"])
                if group_data.get("metadata")
                else None
            ),
            "group_metrics": (
                _compact_json(group_data["metrics"])
                if group_data.get("metrics")
                else None
            ),
            "group_logs": group_data.get("logs") or None,
        }
        if not group.trajectories:
            rows.append(
                {
                    **common,
                    "trajectory_index": None,
                    "trajectory_json": None,
                    "reward": None,
                    "metrics": None,
                    "metadata": None,
                    "tools": None,
                    "logs": None,
                    "messages": None,
                }
            )
            continue

        for trajectory_index, trajectory in enumerate(group.trajectories):
            trajectory_data = _trajectory_data(trajectory)
            rows.append(
                {
                    **common,
                    "trajectory_index": trajectory_index,
                    "trajectory_json": _compact_json(trajectory_data),
                    "reward": trajectory.reward,
                    "metrics": (
                        _compact_json(trajectory_data["metrics"])
                        if trajectory_data.get("metrics")
                        else None
                    ),
                    "metadata": (
                        _compact_json(trajectory_data["metadata"])
                        if trajectory_data.get("metadata")
                        else None
                    ),
                    "tools": (
                        _compact_json(trajectory_data["tools"])
                        if trajectory_data.get("tools")
                        else None
                    ),
                    "logs": trajectory_data.get("logs") or None,
                    "messages": _legacy_messages(trajectory),
                }
            )

    schema = _schema()
    table = (
        pa.Table.from_pylist(rows, schema=schema)
        if rows
        else pa.table({name: [] for name in schema.names}, schema=schema)
    )
    pq.write_table(table, path, compression="zstd")


def _json_object(payload: object, *, field: str) -> dict[str, Any]:
    if not isinstance(payload, (str, bytes, bytearray)):
        raise ValueError(f"Parquet {field} must be JSON text")
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Parquet {field} contains invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Parquet {field} must contain a JSON object")
    return value


def _read_v2(rows: list[dict[str, object]]) -> list[TrajectoryGroup]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        version = row.get("format_version")
        if version != _FORMAT_VERSION:
            raise ValueError(f"Unsupported trajectory Parquet version: {version!r}")
        group_index = row.get("group_index")
        if not isinstance(group_index, int) or isinstance(group_index, bool):
            raise ValueError("Parquet group_index must be an integer")
        grouped.setdefault(group_index, []).append(row)

    result: list[TrajectoryGroup] = []
    if sorted(grouped) != list(range(len(grouped))):
        raise ValueError("Parquet group indexes are duplicate or non-contiguous")
    for group_index in sorted(grouped):
        group_rows = grouped[group_index]
        group_payloads = {row.get("group_json") for row in group_rows}
        if len(group_payloads) != 1:
            raise ValueError(f"Parquet group {group_index} has conflicting metadata")
        group_data = _json_object(group_payloads.pop(), field="group_json")
        if "trajectories" in group_data:
            raise ValueError("Parquet group_json must not contain trajectories")

        indexed: list[tuple[int, Trajectory]] = []
        sentinel_count = 0
        for row in group_rows:
            trajectory_index = row.get("trajectory_index")
            trajectory_json = row.get("trajectory_json")
            if trajectory_index is None:
                if trajectory_json is not None:
                    raise ValueError("Parquet group sentinel contains a trajectory")
                sentinel_count += 1
                continue
            if not isinstance(trajectory_index, int) or isinstance(
                trajectory_index, bool
            ):
                raise ValueError("Parquet trajectory_index must be an integer")
            if not isinstance(trajectory_json, (str, bytes, bytearray)):
                raise ValueError("Parquet trajectory_json must be JSON text")
            indexed.append((trajectory_index, _restore_trajectory(trajectory_json)))

        indexed.sort(key=lambda item: item[0])
        if [index for index, _ in indexed] != list(range(len(indexed))):
            raise ValueError(
                f"Parquet group {group_index} has duplicate or missing trajectories"
            )
        if indexed and sentinel_count:
            raise ValueError(f"Parquet group {group_index} mixes data and a sentinel")
        if not indexed and sentinel_count != 1:
            raise ValueError(f"Parquet group {group_index} has no valid group sentinel")
        result.append(
            TrajectoryGroup.model_validate(
                {**group_data, "trajectories": [item for _, item in indexed]}
            )
        )
    return result


def _load_legacy_object(payload: object | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return cast(dict[str, Any], payload)
    if isinstance(payload, (str, bytes, bytearray)) and payload:
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _read_v1(rows: list[dict[str, object]]) -> list[TrajectoryGroup]:
    groups: dict[int, list[Trajectory]] = {}
    group_metadata: dict[int, dict[str, Any]] = {}
    group_metrics: dict[int, dict[str, Any]] = {}
    group_logs: dict[int, list[str]] = {}

    for row in rows:
        if row.get("reward") is None and row.get("messages") is None:
            continue
        raw_group_index = row.get("group_index", 0)
        if not isinstance(raw_group_index, int) or isinstance(raw_group_index, bool):
            raise ValueError("Legacy Parquet group_index must be an integer")
        group_index = raw_group_index
        group_metadata.setdefault(
            group_index, _load_legacy_object(row.get("group_metadata"))
        )
        group_metrics.setdefault(
            group_index, _load_legacy_object(row.get("group_metrics"))
        )
        raw_logs = row.get("group_logs")
        group_logs.setdefault(
            group_index,
            [str(item) for item in raw_logs]
            if isinstance(raw_logs, (list, tuple))
            else ([] if raw_logs is None else [str(raw_logs)]),
        )

        raw_messages = row.get("messages")
        if raw_messages is None:
            raw_messages = []
        if not isinstance(raw_messages, (list, tuple)):
            raise ValueError("Legacy Parquet messages must be a list")
        messages = []
        for raw_message in raw_messages:
            if isinstance(raw_message, Mapping):
                message = {str(key): value for key, value in raw_message.items()}
            else:
                if not isinstance(raw_message, (list, tuple)):
                    raise ValueError("Legacy Parquet message must be a struct")
                message = {
                    "role": raw_message[0],
                    "content": raw_message[1],
                    "tool_calls": raw_message[2],
                    "tool_call_id": raw_message[3],
                    "trainable": raw_message[4],
                }
            messages.append(_unflatten_message(message))
        reward = row.get("reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise ValueError("Legacy Parquet reward must be numeric")
        raw_trajectory_logs = row.get("logs")
        if raw_trajectory_logs is None:
            trajectory_logs = []
        elif isinstance(raw_trajectory_logs, (list, tuple)):
            trajectory_logs = [str(item) for item in raw_trajectory_logs]
        else:
            raise ValueError("Legacy Parquet logs must be a list")
        groups.setdefault(group_index, []).append(
            Trajectory(
                messages_and_choices=messages,
                reward=float(reward),
                metrics=_load_legacy_object(row.get("metrics")),
                metadata=_load_legacy_object(row.get("metadata")),
                logs=trajectory_logs,
            )
        )

    return [
        TrajectoryGroup(
            trajectories=groups[index],
            exceptions=[],
            metadata=group_metadata.get(index, {}),
            metrics=group_metrics.get(index, {}),
            logs=group_logs.get(index, []),
        )
        for index in sorted(groups)
    ]


def read_trajectory_groups_parquet(path: str | Path) -> list[TrajectoryGroup]:
    """Read exact v2 trajectory groups or legacy v1 files."""
    import duckdb
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    metadata = schema.metadata or {}
    raw_version = metadata.get(b"art.trajectory_parquet.version")
    try:
        file_version = int(raw_version) if raw_version is not None else None
    except ValueError as exc:
        raise ValueError("Invalid trajectory Parquet version metadata") from exc
    if file_version not in {None, _FORMAT_VERSION}:
        raise ValueError(f"Unsupported trajectory Parquet version: {file_version}")
    has_version_column = "format_version" in schema.names
    has_canonical_columns = bool(
        {"group_json", "trajectory_json"}.intersection(schema.names)
    )
    if file_version == _FORMAT_VERSION and not has_version_column:
        raise ValueError("Versioned trajectory Parquet is missing format_version")
    if has_canonical_columns and not has_version_column:
        raise ValueError("Canonical trajectory Parquet is missing its version marker")
    if (
        has_version_column
        and file_version is None
        and not pq.read_metadata(path).num_rows
    ):
        raise ValueError(
            "Empty versioned trajectory Parquet is missing version metadata"
        )

    with duckdb.connect(":memory:") as connection:
        result = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
        rows = [
            dict(zip((column[0] for column in result.description), row))
            for row in result.fetchall()
        ]
    if not rows:
        return []
    return _read_v2(rows) if has_version_column else _read_v1(rows)
