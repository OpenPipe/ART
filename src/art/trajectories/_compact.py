from __future__ import annotations

import builtins
from collections.abc import Iterable, Mapping
import json
import re
from typing import TypeGuard, cast, get_args, get_origin

import pydantic

from . import (
    CompactDumpable,
    CompactTrajectoryKind,
    CompactTrajectoryPayload,
    TokenizedHistory,
    TokenizedMultiHistoryTrajectory,
    TokenizedTrajectory,
    TokenizedTrajectoryGroup,
    Trajectory,
    TrajectoryGroup,
    _load_tensors,
)
from ._serialization import _rebind_history_sources

_FORMAT = "art.trajectories"
_VERSION = 1
_FIELDS = {"format", "version", "kind", "strings", "data"}
_REFERENCE = re.compile(r"\$(?:0|[1-9][0-9]*)\Z")
_SOURCE_MARKERS = {
    "request_index",
    "choice_index",
    "output_index",
    "output_indices",
    "generation_index",
    "prompt_index",
}
type _CompactValidated = CompactDumpable | list[CompactDumpable]

_PLURAL_BY_SINGULAR: dict[CompactTrajectoryKind, CompactTrajectoryKind] = {
    "trajectory": "trajectories",
    "trajectory_group": "trajectory_groups",
    "tokenized_history": "tokenized_histories",
    "tokenized_trajectory": "tokenized_trajectories",
    "tokenized_multi_history_trajectory": "tokenized_multi_history_trajectories",
    "tokenized_trajectory_group": "tokenized_trajectory_groups",
    "tensorized_history": "tensorized_histories",
    "tensorized_trajectory": "tensorized_trajectories",
    "tensorized_multi_history_trajectory": "tensorized_multi_history_trajectories",
    "tensorized_trajectory_group": "tensorized_trajectory_groups",
}
_SINGULAR_BY_PLURAL = {
    plural: singular for singular, plural in _PLURAL_BY_SINGULAR.items()
}
_SINGULAR_KINDS = _PLURAL_BY_SINGULAR.keys()
_PLURAL_KINDS = _SINGULAR_BY_PLURAL.keys()
_KINDS = _SINGULAR_KINDS | _PLURAL_KINDS


def dump(
    value: CompactDumpable | Iterable[CompactDumpable],
) -> CompactTrajectoryPayload:
    """Serialize one supported value or a homogeneous iterable."""

    if _is_dumpable(value):
        kind = _kind(value)
        return _encode(kind, _dump_value(value, kind))
    values = list(cast(Iterable[CompactDumpable], value))
    if not values:
        raise ValueError("compact_dump() cannot infer the kind of an empty iterable")
    if any(not _is_dumpable(item) for item in values):
        raise TypeError("compact_dump() received an unsupported value")
    kinds = [_kind(item) for item in values]
    if len(set(kinds)) != 1:
        raise TypeError("compact_dump() requires a homogeneous iterable")
    return _encode(
        _plural_kind(kinds[0]),
        [_dump_value(item, kind) for item, kind in zip(values, kinds, strict=True)],
    )


def dump_trajectory(trajectory: Trajectory) -> CompactTrajectoryPayload:
    return dump(trajectory)


def dump_trajectory_group(group: TrajectoryGroup) -> CompactTrajectoryPayload:
    return dump(group)


def dump_tokenized_history(value: TokenizedHistory) -> CompactTrajectoryPayload:
    return dump(value)


def dump_tokenized_trajectory(
    value: TokenizedTrajectory,
) -> CompactTrajectoryPayload:
    return dump(value)


def dump_tokenized_multi_history_trajectory(
    value: TokenizedMultiHistoryTrajectory,
) -> CompactTrajectoryPayload:
    return dump(value)


def dump_tokenized_trajectory_group(
    value: TokenizedTrajectoryGroup,
) -> CompactTrajectoryPayload:
    return dump(value)


def validate(
    payload: Mapping[str, object], *, type: object = None, device: object = None
) -> _CompactValidated:
    """Decode one compact value, inferring or checking its requested type."""

    kind = _payload_kind(payload)
    target_model: builtins.type[pydantic.BaseModel] | None = None
    if type is not None:
        expected_kind, target_model = _target(type)
        if kind != expected_kind:
            raise ValueError(
                f"Requested compact type expects kind {expected_kind!r}, got {kind!r}"
            )
    if device is not None and not kind.startswith("tensorized_"):
        raise ValueError("device is only valid for tensorized compact payloads")
    data = _decode(payload, kind)
    if kind in _PLURAL_KINDS:
        if not isinstance(data, list):
            raise ValueError("Compact collection payload data must be a list")
        singular = _singular_kind(kind)
        values = [
            _validate_value(item, singular, target_model, device=device)
            for item in data
        ]
        for value in values:
            _finish(value)
        return cast(_CompactValidated, values)
    return _finish(_validate_value(data, kind, target_model, device=device))


def _validate_value(
    value: object,
    kind: CompactTrajectoryKind,
    cls: type[pydantic.BaseModel] | None,
    *,
    device: object,
) -> CompactDumpable:
    if kind == "trajectory":
        return Trajectory.model_validate(value)
    if kind == "trajectory_group":
        return TrajectoryGroup.model_validate(value)
    if kind == "tokenized_history":
        return _validate_history(value, cls or TokenizedHistory)
    if kind == "tokenized_trajectory":
        return _validate_trajectory(value, cls or TokenizedTrajectory)
    if kind == "tokenized_multi_history_trajectory":
        return _validate_multi(value, cls or TokenizedMultiHistoryTrajectory)
    if kind == "tokenized_trajectory_group":
        return _validate_group(value, cls or TokenizedTrajectoryGroup)
    if kind.startswith("tensorized_"):
        tensors = _tensors()
        models: dict[CompactTrajectoryKind, type[pydantic.BaseModel]] = {
            "tensorized_history": tensors.TensorizedHistory,
            "tensorized_trajectory": tensors.TensorizedTrajectory,
            "tensorized_multi_history_trajectory": (
                tensors.TensorizedMultiHistoryTrajectory
            ),
            "tensorized_trajectory_group": tensors.TensorizedTrajectoryGroup,
        }
        model = cls or models[kind]
        if kind == "tensorized_history":
            result = _validate_history(value, model)
        elif kind == "tensorized_trajectory":
            result = _validate_trajectory(value, model)
        elif kind == "tensorized_multi_history_trajectory":
            result = _validate_multi(value, model)
        else:
            result = _validate_group(value, model)
        if device is not None:
            move = getattr(result, "to", None)
            if not callable(move):
                raise AssertionError("Tensorized compact value has no device mover")
            move(device)
        return cast(CompactDumpable, result)
    raise ValueError(f"Compact kind {kind!r} does not identify one value")


def _plural_kind(kind: CompactTrajectoryKind) -> CompactTrajectoryKind:
    try:
        return _PLURAL_BY_SINGULAR[kind]
    except KeyError as error:
        raise ValueError(f"Compact kind {kind!r} is already plural") from error


def _singular_kind(kind: CompactTrajectoryKind) -> CompactTrajectoryKind:
    try:
        return _SINGULAR_BY_PLURAL[kind]
    except KeyError as error:
        raise ValueError(f"Compact kind {kind!r} is already singular") from error


def _tensors():
    return _load_tensors()


def _is_tensorized(value: object) -> bool:
    return (
        value.__class__.__module__ == "art.trajectories.tensors"
        and value.__class__.__name__.split("[", 1)[0]
        in {
            "TensorizedHistory",
            "TensorizedTrajectory",
            "TensorizedMultiHistoryTrajectory",
            "TensorizedTrajectoryGroup",
        }
    )


def _target(value: object) -> tuple[CompactTrajectoryKind, type[pydantic.BaseModel]]:
    origin = get_origin(value)
    plural = origin is list
    if plural:
        arguments = get_args(value)
        if len(arguments) != 1:
            raise TypeError("Compact collection type must have exactly one item type")
        value = arguments[0]
    if not isinstance(value, type) or not issubclass(value, pydantic.BaseModel):
        raise TypeError("type must be a supported compact model or list[model]")
    kind = _model_kind(value)
    return (_plural_kind(kind) if plural else kind), value


def _model_kind(value: type[pydantic.BaseModel]) -> CompactTrajectoryKind:
    if issubclass(value, TokenizedTrajectory):
        return "tokenized_trajectory"
    if issubclass(value, TokenizedHistory):
        return "tokenized_history"
    if issubclass(value, TokenizedMultiHistoryTrajectory):
        return "tokenized_multi_history_trajectory"
    if issubclass(value, TokenizedTrajectoryGroup):
        return "tokenized_trajectory_group"
    if issubclass(value, TrajectoryGroup):
        return "trajectory_group"
    if issubclass(value, Trajectory):
        return "trajectory"
    if value.__module__ == "art.trajectories.tensors":
        tensors = _tensors()
        if issubclass(value, tensors.TensorizedTrajectory):
            return "tensorized_trajectory"
        if issubclass(value, tensors.TensorizedHistory):
            return "tensorized_history"
        if issubclass(value, tensors.TensorizedMultiHistoryTrajectory):
            return "tensorized_multi_history_trajectory"
        if issubclass(value, tensors.TensorizedTrajectoryGroup):
            return "tensorized_trajectory_group"
    raise TypeError(f"Unsupported compact type: {value!r}")


def _payload_kind(payload: Mapping[str, object]) -> CompactTrajectoryKind:
    if set(payload) != _FIELDS:
        raise ValueError(
            "Compact trajectory payload must contain exactly "
            "format, version, kind, strings, and data"
        )
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ValueError(f"Unsupported compact trajectory kind: {kind!r}")
    return kind


def _is_dumpable(value: object) -> TypeGuard[CompactDumpable]:
    return _is_tensorized(value) or isinstance(
        value,
        (
            Trajectory,
            TrajectoryGroup,
            TokenizedHistory,
            TokenizedMultiHistoryTrajectory,
            TokenizedTrajectoryGroup,
        ),
    )


def _kind(value: CompactDumpable) -> CompactTrajectoryKind:
    if _is_tensorized(value):
        return _model_kind(cast(type[pydantic.BaseModel], type(value)))
    if isinstance(value, TokenizedTrajectory):
        return "tokenized_trajectory"
    if isinstance(value, TokenizedHistory):
        return "tokenized_history"
    if isinstance(value, TokenizedMultiHistoryTrajectory):
        return "tokenized_multi_history_trajectory"
    if isinstance(value, TokenizedTrajectoryGroup):
        return "tokenized_trajectory_group"
    if isinstance(value, TrajectoryGroup):
        return "trajectory_group"
    if isinstance(value, Trajectory):
        return "trajectory"
    raise TypeError(f"Unsupported compact value: {type(value).__name__}")


def _dump_value(
    value: CompactDumpable, kind: CompactTrajectoryKind
) -> pydantic.JsonValue:
    if kind in {"trajectory", "trajectory_group"}:
        if not isinstance(value, pydantic.BaseModel):
            raise AssertionError("Plain compact values must be Pydantic models")
        return _dump_model(value)
    data = _dump_model(cast(pydantic.BaseModel, value))
    if kind in {"tokenized_history", "tensorized_history"}:
        registry = _ExchangeDataRegistry()
        _replace_source_exchanges(data["history"], registry, encode=True)
        return {"value": data, "exchanges": registry.exchanges}
    if kind in {"tokenized_trajectory", "tensorized_trajectory"}:
        _compact_trajectory_data(data)
        return data
    if kind in {
        "tokenized_multi_history_trajectory",
        "tensorized_multi_history_trajectory",
    }:
        _compact_multi_data(data)
        return data
    if kind in {"tokenized_trajectory_group", "tensorized_trajectory_group"}:
        _compact_group_data(data)
        return data
    raise AssertionError("Compact kind and value disagree")


class _ExchangeDataRegistry:
    def __init__(
        self,
        exchanges: Iterable[dict[str, pydantic.JsonValue]] = (),
        *,
        fixed: bool = False,
    ):
        self.exchanges = list(exchanges)
        self.fixed = fixed
        self._indices = {
            _json_key(exchange): index for index, exchange in enumerate(self.exchanges)
        }

    @classmethod
    def from_trajectory_data(cls, value: object) -> _ExchangeDataRegistry:
        data = _mapping(value, "Compact source trajectory")
        exchanges = _mapping(data.get("exchanges"), "Compact trajectory exchanges")
        values: list[dict[str, pydantic.JsonValue]] = []
        for protocol in ("chat_completions", "completions", "responses", "messages"):
            for exchange in _list(exchanges.get(protocol, []), "Trajectory exchanges"):
                values.append(
                    cast(
                        dict[str, pydantic.JsonValue],
                        dict(_mapping(exchange, "Exchange")),
                    )
                )
        return cls(values, fixed=True)

    def reference(self, value: object) -> int:
        exchange = cast(
            dict[str, pydantic.JsonValue],
            dict(_mapping(value, "History source exchange")),
        )
        key = _json_key(exchange)
        if (index := self._indices.get(key)) is not None:
            return index
        if self.fixed:
            raise ValueError("History source exchange is not present in its trajectory")
        index = len(self.exchanges)
        self.exchanges.append(exchange)
        self._indices[key] = index
        return index

    def resolve(self, value: object) -> dict[str, pydantic.JsonValue]:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < len(self.exchanges)
        ):
            raise ValueError("History source exchange reference is invalid")
        return self.exchanges[value]


def _compact_trajectory_data(data: dict[str, pydantic.JsonValue]) -> None:
    registry = _ExchangeDataRegistry.from_trajectory_data(data.get("trajectory"))
    _replace_source_exchanges(data.get("history"), registry, encode=True)


def _compact_multi_data(data: dict[str, pydantic.JsonValue]) -> None:
    registry = _ExchangeDataRegistry.from_trajectory_data(data.get("trajectory"))
    for history in _list(data.get("histories"), "Tokenized histories"):
        _replace_source_exchanges(
            _mapping(history, "Tokenized history").get("history"),
            registry,
            encode=True,
        )


def _compact_group_data(data: dict[str, pydantic.JsonValue]) -> None:
    group = _mapping(data.get("trajectory_group"), "Compact source group")
    source_trajectories = _list(group.get("trajectories"), "Source trajectories")
    tokenized = _list(data.get("trajectories"), "Tokenized trajectories")
    if len(source_trajectories) != len(tokenized):
        raise ValueError("Tokenized group differs in length from its source group")
    for index, (item, source) in enumerate(
        zip(tokenized, source_trajectories, strict=True)
    ):
        child = cast(
            dict[str, pydantic.JsonValue],
            dict(_mapping(item, "Tokenized trajectory")),
        )
        if child.pop("trajectory", None) != source:
            raise ValueError("Tokenized trajectory does not match its source group")
        registry = _ExchangeDataRegistry.from_trajectory_data(source)
        if "history" in child:
            _replace_source_exchanges(child["history"], registry, encode=True)
        else:
            for history in _list(child.get("histories"), "Tokenized histories"):
                _replace_source_exchanges(
                    _mapping(history, "Tokenized history").get("history"),
                    registry,
                    encode=True,
                )
        cast(list[pydantic.JsonValue], tokenized)[index] = child


def _validate_history[ValueT: pydantic.BaseModel](
    value: object, cls: type[ValueT]
) -> ValueT:
    wrapper = _mapping(value, "Compact tokenized history")
    if set(wrapper) != {"value", "exchanges"}:
        raise ValueError("Compact tokenized history has invalid fields")
    data = dict(_mapping(wrapper["value"], "Tokenized history data"))
    registry = _ExchangeDataRegistry(
        (
            cast(dict[str, pydantic.JsonValue], dict(_mapping(item, "Exchange")))
            for item in _list(wrapper["exchanges"], "History source exchanges")
        ),
        fixed=True,
    )
    _replace_source_exchanges(data.get("history"), registry, encode=False)
    return cls.model_validate(data)


def _validate_trajectory[ValueT: pydantic.BaseModel](
    value: object, cls: type[ValueT]
) -> ValueT:
    data = dict(_mapping(value, "Compact tokenized trajectory"))
    registry = _ExchangeDataRegistry.from_trajectory_data(data.get("trajectory"))
    _replace_source_exchanges(data.get("history"), registry, encode=False)
    return cls.model_validate(data)


def _validate_multi[ValueT: pydantic.BaseModel](
    value: object, cls: type[ValueT]
) -> ValueT:
    data = dict(_mapping(value, "Compact multi-history trajectory"))
    registry = _ExchangeDataRegistry.from_trajectory_data(data.get("trajectory"))
    for history in _list(data.get("histories"), "Tokenized histories"):
        _replace_source_exchanges(
            _mapping(history, "Tokenized history").get("history"),
            registry,
            encode=False,
        )
    return cls.model_validate(data)


def _validate_group[ValueT: pydantic.BaseModel](
    value: object, cls: type[ValueT]
) -> ValueT:
    data = dict(_mapping(value, "Compact tokenized group"))
    group = _mapping(data.get("trajectory_group"), "Compact source group")
    source_trajectories = _list(group.get("trajectories"), "Source trajectories")
    tokenized = _list(data.get("trajectories"), "Tokenized trajectories")
    if len(source_trajectories) != len(tokenized):
        raise ValueError("Tokenized group differs in length from its source group")
    multi: bool | None = None
    for index, (item, source) in enumerate(
        zip(tokenized, source_trajectories, strict=True)
    ):
        child = cast(
            dict[str, pydantic.JsonValue],
            dict(_mapping(item, "Tokenized trajectory")),
        )
        child["trajectory"] = cast(pydantic.JsonValue, source)
        registry = _ExchangeDataRegistry.from_trajectory_data(source)
        child_is_multi = "histories" in child
        if multi is not None and child_is_multi != multi:
            raise ValueError("Compact tokenized group mixes trajectory types")
        multi = child_is_multi
        if child_is_multi:
            for history in _list(child["histories"], "Tokenized histories"):
                _replace_source_exchanges(
                    _mapping(history, "Tokenized history").get("history"),
                    registry,
                    encode=False,
                )
        else:
            _replace_source_exchanges(child.get("history"), registry, encode=False)
        cast(list[pydantic.JsonValue], tokenized)[index] = child
    model: type[pydantic.BaseModel]
    if cls is TokenizedTrajectoryGroup:
        model = cast(
            type[TokenizedTrajectoryGroup],
            (
                TokenizedTrajectoryGroup[TokenizedMultiHistoryTrajectory]
                if multi
                else TokenizedTrajectoryGroup[TokenizedTrajectory]
            ),
        )
    elif cls.__module__ == "art.trajectories.tensors" and cls.__name__ == (
        "TensorizedTrajectoryGroup"
    ):
        tensors = _tensors()
        model = (
            tensors.TensorizedTrajectoryGroup[tensors.TensorizedMultiHistoryTrajectory]
            if multi
            else tensors.TensorizedTrajectoryGroup[tensors.TensorizedTrajectory]
        )
    else:
        model = cls
    return cast(ValueT, model.model_validate(data))


def _replace_source_exchanges(
    value: object, registry: _ExchangeDataRegistry, *, encode: bool
) -> None:
    if isinstance(value, list):
        for item in value:
            _replace_source_exchanges(item, registry, encode=encode)
        return
    if not isinstance(value, dict):
        return
    mutable = cast(dict[str, object], value)
    if "exchange" in mutable and _SOURCE_MARKERS.intersection(mutable):
        source = mutable["exchange"]
        mutable["exchange"] = (
            registry.reference(source) if encode else registry.resolve(source)
        )
        return
    for key, item in mutable.items():
        if key in {"system_source", "instructions_source"} and item is not None:
            mutable[key] = (
                registry.reference(item) if encode else registry.resolve(item)
            )
        else:
            _replace_source_exchanges(item, registry, encode=encode)


def _finish[ValueT](value: ValueT) -> ValueT:
    if isinstance(value, TokenizedHistory):
        _rebind_history_sources(value.history)
    elif _is_tensorized(value) and hasattr(value, "history"):
        _rebind_history_sources(value.history)
    return value


def _dump_model(model: pydantic.BaseModel) -> dict[str, pydantic.JsonValue]:
    return cast(
        dict[str, pydantic.JsonValue],
        model.model_dump(mode="json", warnings="error"),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a string-keyed dictionary")
    return cast(Mapping[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return cast(list[object], value)


def _json_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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
