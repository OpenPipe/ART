from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import Any, Literal, cast

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
import pydantic
from pydantic import BaseModel
from pydantic.main import IncEx

from ..openai import ART_MOE_ROUTING_METADATA_KEY

type _StringPool = dict[str, str]


def _intern_strings(value: object, pool: _StringPool | None = None) -> None:
    """Share equal strings inside supported model and built-in container graphs."""

    _intern_value(value, {} if pool is None else pool, {})


def _intern_value(value: object, pool: _StringPool, memo: dict[int, object]) -> object:
    if isinstance(value, str):
        return pool.setdefault(value, value)
    if isinstance(value, (bytes, bytearray, memoryview)) or value is None:
        return value
    if type(value) in (bool, float, int):
        return value
    if isinstance(value, list) and all(
        item is None or type(item) in (bool, float, int) for item in value
    ):
        return value

    value_id = id(value)
    if value_id in memo:
        return memo[value_id]

    if isinstance(value, BaseModel):
        memo[value_id] = value
        for name, item in value.__dict__.items():
            value.__dict__[name] = _intern_value(item, pool, memo)
        extra = value.__pydantic_extra__
        if extra is not None and id(extra) not in memo:
            memo[id(extra)] = extra
            _intern_mapping(cast(dict[object, object], extra), pool, memo)
        return value
    if is_dataclass(value) and type(value).__module__.startswith("art.trajectories"):
        memo[value_id] = value
        for field in fields(value):
            object.__setattr__(
                value,
                field.name,
                _intern_value(getattr(value, field.name), pool, memo),
            )
        return value
    if isinstance(value, dict):
        memo[value_id] = value
        _intern_mapping(cast(dict[object, object], value), pool, memo)
        return value
    if isinstance(value, list):
        memo[value_id] = value
        items = cast(list[object], value)
        for index, item in enumerate(items):
            items[index] = _intern_value(item, pool, memo)
        return value
    if isinstance(value, tuple):
        memo[value_id] = value
        result = tuple(_intern_value(item, pool, memo) for item in value)
        memo[value_id] = result
        return result
    if isinstance(value, set):
        memo[value_id] = value
        values = cast(set[object], value)
        items = [_intern_value(item, pool, memo) for item in values]
        values.clear()
        values.update(items)
        return value
    if isinstance(value, frozenset):
        memo[value_id] = value
        result = frozenset(_intern_value(item, pool, memo) for item in value)
        memo[value_id] = result
        return result
    return value


def _intern_mapping(
    value: dict[object, object], pool: _StringPool, memo: dict[int, object]
) -> None:
    items = [
        (
            _intern_value(key, pool, memo) if isinstance(key, str) else key,
            _intern_value(item, pool, memo),
        )
        for key, item in value.items()
    ]
    value.clear()
    value.update(items)


def serialize_messages_and_choices(items: list[Any]) -> list[dict[str, Any]]:
    return [
        item.model_dump(mode="json", exclude={ART_MOE_ROUTING_METADATA_KEY})
        if isinstance(item, Choice)
        else dict(item)
        for item in items
    ]


def serialize_chat_completion(response: ChatCompletion) -> dict[str, Any]:
    return response.model_dump(
        mode="json",
        exclude={
            "choices": {"__all__": {ART_MOE_ROUTING_METADATA_KEY}},
        },
    )


def serialize_history(history: object) -> dict[str, pydantic.JsonValue]:
    """Serialize a concrete history without ambiguous union inference."""

    from . import (
        AnthropicMessagesHistory,
        ChatCompletionsHistory,
        CompletionsStringHistory,
        CompletionsTokenHistory,
        LegacyHistory,
        ResponsesHistory,
    )

    kinds = {
        LegacyHistory: "legacy",
        ChatCompletionsHistory: "chat_completions",
        AnthropicMessagesHistory: "messages",
        ResponsesHistory: "responses",
        CompletionsTokenHistory: "completions_token",
        CompletionsStringHistory: "completions_string",
    }
    kind = kinds.get(type(history))
    if kind is None:
        raise TypeError(f"Unsupported history type: {type(history).__name__}")
    if not isinstance(history, BaseModel):
        raise TypeError(f"Unsupported history type: {type(history).__name__}")
    return cast(
        dict[str, pydantic.JsonValue],
        {
            "kind": kind,
            "data": history.model_dump(mode="json", warnings="error"),
        },
    )


def validate_history(value: object) -> object:
    """Restore the concrete history selected by its serialized kind."""

    from . import (
        AnthropicMessagesHistory,
        ChatCompletionsHistory,
        CompletionsStringHistory,
        CompletionsTokenHistory,
        LegacyHistory,
        ResponsesHistory,
    )

    history_types = (
        LegacyHistory,
        ChatCompletionsHistory,
        AnthropicMessagesHistory,
        ResponsesHistory,
        CompletionsTokenHistory,
        CompletionsStringHistory,
    )
    if isinstance(value, history_types):
        return value
    if not isinstance(value, dict) or set(value) != {"kind", "data"}:
        raise ValueError("Serialized tokenized history must identify its kind")
    serialized = cast(dict[str, object], value)
    if not isinstance(serialized["data"], dict):
        raise ValueError("Serialized tokenized history data must be a dictionary")
    data = cast(dict[str, Any], dict(serialized["data"]))

    kind = serialized["kind"]
    models: dict[object, type[BaseModel]] = {
        "legacy": LegacyHistory,
        "chat_completions": ChatCompletionsHistory,
        "messages": AnthropicMessagesHistory,
        "responses": ResponsesHistory,
        "completions_token": CompletionsTokenHistory,
        "completions_string": CompletionsStringHistory,
    }
    try:
        model = models[kind]
    except KeyError as error:
        raise ValueError(f"Unknown serialized history kind: {kind!r}") from error
    return model.model_validate(data)


def _rebind_history_sources(history: object, trajectory: object | None = None) -> None:
    """Restore history sidecars to canonical exchange objects after validation."""

    from . import (
        ChatCompletionsExchange,
        CompletionsExchange,
        MessagesExchange,
        ResponsesExchange,
        Trajectory,
    )

    exchange_types = (
        ChatCompletionsExchange,
        CompletionsExchange,
        ResponsesExchange,
        MessagesExchange,
    )
    canonical = (
        [
            *trajectory.exchanges.chat_completions,
            *trajectory.exchanges.completions,
            *trajectory.exchanges.responses,
            *trajectory.exchanges.messages,
        ]
        if isinstance(trajectory, Trajectory)
        else []
    )
    fixed = trajectory is not None
    identities = {id(exchange): exchange for exchange in canonical}

    def visit(value: object) -> None:
        if isinstance(value, BaseModel) or is_dataclass(value):
            items = (
                value.__dict__.items()
                if isinstance(value, BaseModel)
                else (
                    (field.name, getattr(value, field.name)) for field in fields(value)
                )
            )
            for name, item in items:
                if isinstance(item, exchange_types):
                    if id(item) in identities:
                        continue
                    matches = [
                        exchange
                        for exchange in canonical
                        if type(exchange) is type(item) and exchange == item
                    ]
                    if matches:
                        object.__setattr__(value, name, matches[0])
                        identities[id(item)] = matches[0]
                    elif fixed:
                        raise ValueError(
                            "Tokenized history source is absent from its trajectory"
                        )
                    else:
                        canonical.append(item)
                        identities[id(item)] = item
                else:
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(history)


class _CompactModel(BaseModel):
    """Pydantic model whose default dump omits fields equal to their defaults."""

    def model_dump(
        self,
        *,
        mode: Literal["json", "python"] | str = "python",
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = True,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
        polymorphic_serialization: bool | None = None,
    ) -> dict[str, Any]:
        if polymorphic_serialization is not None:
            return super().model_dump(
                mode=mode,
                include=include,
                exclude=exclude,
                context=context,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                exclude_computed_fields=exclude_computed_fields,
                round_trip=round_trip,
                warnings=warnings,
                fallback=fallback,
                serialize_as_any=serialize_as_any,
                polymorphic_serialization=polymorphic_serialization,
            )
        return super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_computed_fields=exclude_computed_fields,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
        )

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        ensure_ascii: bool = False,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = True,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
        polymorphic_serialization: bool | None = None,
    ) -> str:
        if polymorphic_serialization is not None:
            return super().model_dump_json(
                indent=indent,
                ensure_ascii=ensure_ascii,
                include=include,
                exclude=exclude,
                context=context,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                exclude_computed_fields=exclude_computed_fields,
                round_trip=round_trip,
                warnings=warnings,
                fallback=fallback,
                serialize_as_any=serialize_as_any,
                polymorphic_serialization=polymorphic_serialization,
            )
        return super().model_dump_json(
            indent=indent,
            ensure_ascii=ensure_ascii,
            include=include,
            exclude=exclude,
            context=context,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            exclude_computed_fields=exclude_computed_fields,
            round_trip=round_trip,
            warnings=warnings,
            fallback=fallback,
            serialize_as_any=serialize_as_any,
        )
