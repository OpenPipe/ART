from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.main import IncEx


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
