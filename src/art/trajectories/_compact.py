from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import re
from typing import TypeVar, cast

import pydantic

from . import (
    CompactTrajectoryKind,
    CompactTrajectoryPayload,
    Trajectory,
    TrajectoryGroup,
)
from ._serialization import _StringPool

_FORMAT = "art.trajectories"
_VERSION = 1
_FIELDS = {"format", "version", "kind", "strings", "data"}
_REFERENCE = re.compile(r"\$(?:0|[1-9][0-9]*)\Z")
_TRAJECTORIES = pydantic.TypeAdapter(list[Trajectory])
_TRAJECTORY_GROUPS = pydantic.TypeAdapter(list[TrajectoryGroup])
_TrajectoryT = TypeVar("_TrajectoryT", bound=Trajectory)
_TrajectoryGroupT = TypeVar("_TrajectoryGroupT", bound=TrajectoryGroup)


def dump_trajectory(trajectory: Trajectory) -> CompactTrajectoryPayload:
    trajectory._intern_strings()
    return _encode("trajectory", _dump_model(trajectory))


def validate_trajectory(
    payload: Mapping[str, object], cls: type[_TrajectoryT]
) -> _TrajectoryT:
    value = cls.model_validate(_decode(payload, "trajectory"))
    value._intern_strings()
    return value


def dump_trajectory_group(group: TrajectoryGroup) -> CompactTrajectoryPayload:
    group._intern_strings()
    return _encode("trajectory_group", _dump_model(group))


def validate_trajectory_group(
    payload: Mapping[str, object], cls: type[_TrajectoryGroupT]
) -> _TrajectoryGroupT:
    value = cls.model_validate(_decode(payload, "trajectory_group"))
    value._intern_strings()
    return value


def dump_trajectories(
    trajectories: Iterable[Trajectory],
) -> CompactTrajectoryPayload:
    values = list(trajectories)
    pool: _StringPool = {}
    for trajectory in values:
        trajectory._intern_strings(pool)
    return _encode("trajectories", [_dump_model(value) for value in values])


def validate_trajectories(
    payload: Mapping[str, object],
) -> list[Trajectory]:
    values = _TRAJECTORIES.validate_python(_decode(payload, "trajectories"))
    pool: _StringPool = {}
    for value in values:
        value._intern_strings(pool)
    return values


def dump_trajectory_groups(
    groups: Iterable[TrajectoryGroup],
) -> CompactTrajectoryPayload:
    values = list(groups)
    pool: _StringPool = {}
    for group in values:
        group._intern_strings(pool)
    return _encode("trajectory_groups", [_dump_model(value) for value in values])


def validate_trajectory_groups(
    payload: Mapping[str, object],
) -> list[TrajectoryGroup]:
    values = _TRAJECTORY_GROUPS.validate_python(_decode(payload, "trajectory_groups"))
    pool: _StringPool = {}
    for value in values:
        value._intern_strings(pool)
    return values


def _dump_model(model: pydantic.BaseModel) -> pydantic.JsonValue:
    return cast(
        pydantic.JsonValue,
        model.model_dump(mode="json", warnings="error"),
    )


def _encode(
    kind: CompactTrajectoryKind, data: pydantic.JsonValue
) -> CompactTrajectoryPayload:
    counts: dict[str, int] = {}
    order: list[str] = []
    _count_strings(data, counts, order)
    encoded_lengths = {value: _encoded_length(value) for value in counts}

    strings: dict[str, str] = {}
    replacements: dict[str, str] = {}
    pending, reference = _next_reference(len(strings), counts, replacements)
    pending_cost = _mapping_cost(pending, len(strings), encoded_lengths)
    for value in order:
        mapping_cost = (
            pending_cost
            + _encoded_length(reference)
            + 1
            + encoded_lengths[value]
            + int(bool(strings) or bool(pending))
        )
        literal_cost = counts[value] * encoded_lengths[value]
        reference_cost = counts[value] * _encoded_length(reference)
        if reference_cost + mapping_cost >= literal_cost:
            continue

        for key, item in pending:
            strings[key] = item
            replacements[item] = key
        strings[reference] = value
        replacements[value] = reference
        pending, reference = _next_reference(len(strings), counts, replacements)
        pending_cost = _mapping_cost(pending, len(strings), encoded_lengths)

    encoded_data = _replace_strings(data, replacements)
    candidate = _payload(kind, strings, encoded_data)
    plain = _payload(kind, {}, data)
    return candidate if _json_size(candidate) < _json_size(plain) else plain


def _next_reference(
    index: int, counts: dict[str, int], replacements: dict[str, str]
) -> tuple[list[tuple[str, str]], str]:
    pending: list[tuple[str, str]] = []
    reference = f"${index}"
    while reference in counts and replacements.get(reference, reference) == reference:
        pending.append((reference, reference))
        index += 1
        reference = f"${index}"
    return pending, reference


def _mapping_cost(
    entries: list[tuple[str, str]],
    existing_entries: int,
    encoded_lengths: dict[str, int],
) -> int:
    return sum(
        encoded_lengths[key]
        + 1
        + encoded_lengths[value]
        + int(existing_entries > 0 or index > 0)
        for index, (key, value) in enumerate(entries)
    )


def _payload(
    kind: CompactTrajectoryKind,
    strings: dict[str, str],
    data: pydantic.JsonValue,
) -> CompactTrajectoryPayload:
    return {
        "format": _FORMAT,
        "version": _VERSION,
        "kind": kind,
        "strings": strings,
        "data": data,
    }


def _count_strings(
    value: pydantic.JsonValue, counts: dict[str, int], order: list[str]
) -> None:
    if isinstance(value, str):
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1
    elif isinstance(value, list):
        for item in value:
            _count_strings(item, counts, order)
    elif isinstance(value, dict):
        for key, item in value.items():
            _count_strings(key, counts, order)
            _count_strings(item, counts, order)


def _replace_strings(
    value: pydantic.JsonValue, replacements: dict[str, str]
) -> pydantic.JsonValue:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            replacements.get(key, key): _replace_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def _decode(
    payload: Mapping[str, object], expected_kind: CompactTrajectoryKind
) -> pydantic.JsonValue:
    if set(payload) != _FIELDS:
        raise ValueError(
            "Compact trajectory payload must contain exactly "
            "format, version, kind, strings, and data"
        )
    if payload["format"] != _FORMAT:
        raise ValueError("Unsupported compact trajectory format")
    version = payload["version"]
    if type(version) is not int or version != _VERSION:
        raise ValueError("Unsupported compact trajectory version")
    if payload["kind"] != expected_kind:
        raise ValueError(
            f"Expected compact trajectory kind {expected_kind!r}, "
            f"got {payload['kind']!r}"
        )
    raw_strings = payload["strings"]
    if not isinstance(raw_strings, dict):
        raise ValueError("Compact trajectory strings must be a dictionary")
    strings: dict[str, str] = {}
    for key, value in raw_strings.items():
        if not isinstance(key, str) or _REFERENCE.fullmatch(key) is None:
            raise ValueError(f"Invalid compact trajectory string reference: {key!r}")
        if not isinstance(value, str):
            raise ValueError("Compact trajectory string table values must be strings")
        strings[key] = value
    return _decode_value(payload["data"], strings)


def _decode_value(value: object, strings: dict[str, str]) -> pydantic.JsonValue:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return strings.get(value, value)
    if isinstance(value, list):
        return [_decode_value(item, strings) for item in value]
    if isinstance(value, dict):
        decoded: dict[str, pydantic.JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Compact trajectory data keys must be strings")
            decoded_key = strings.get(key, key)
            if decoded_key in decoded:
                raise ValueError(
                    f"Compact trajectory decoding creates duplicate key {decoded_key!r}"
                )
            decoded[decoded_key] = _decode_value(item, strings)
        return decoded
    raise ValueError(f"Compact trajectory data is not JSON-compatible: {type(value)!r}")


def _encoded_length(value: str) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _json_size(value: object) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
