from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
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
