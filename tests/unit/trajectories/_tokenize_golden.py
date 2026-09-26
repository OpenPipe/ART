"""Canonical digests for tokenization goldens.

Shared by ``conftest.py`` (trace-mode recording of every ``tokenize_history``
call made by the existing tests) and ``test_tokenize_golden.py`` (explicit
named cases). Digests are byte-exact: floats use ``float.hex()`` so ``NaN``,
signed zero and infinities are preserved, and flags are recorded as integers.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, TypedDict

MODE_ENV = "ART_TOKENIZE_GOLDEN"
FIELDS = ("model", "tokens", "logprobs", "flags")


def mode() -> str:
    """``""`` (inactive), ``"check"`` or ``"update"``."""

    value = os.environ.get(MODE_ENV, "").strip().lower()
    if value in ("", "0", "off", "false"):
        return ""
    if value in ("check", "update"):
        return value
    raise ValueError(f"{MODE_ENV} must be 'check' or 'update', not {value!r}")


def _hex_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return float(value).hex()


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=repr
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class FieldValues(TypedDict):
    model: str
    tokens: list[int]
    logprobs: list[str]
    flags: list[int]


class Digest(TypedDict):
    """Digest of one ``TokenizedHistory``."""

    kind: str
    n: int
    input: str
    output: str
    fields: dict[str, str]


def field_values(tokenized: Any) -> FieldValues:
    return FieldValues(
        model=str(tokenized.model),
        tokens=[int(token) for token in tokenized.tokens],
        logprobs=[_hex_float(logprob) for logprob in tokenized.logprobs],
        flags=[int(flag) for flag in tokenized.flags],
    )


def tokenizer_identity(tokenizer: object) -> str:
    if tokenizer is None:
        return "None"
    kind = type(tokenizer)
    identity = f"{kind.__module__}.{kind.__qualname__}"
    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name:
        identity += f"({name})"
    return identity


def input_digest(
    history: object,
    *,
    model: str | None,
    base_model: str | None,
    tokenizer: object,
    chat_template: str | None,
    chat_template_kwargs: Mapping[str, object] | None,
) -> str:
    from art.trajectories._serialization import serialize_history

    try:
        serialized: object = serialize_history(history)
    except Exception as error:  # noqa: BLE001 - never let the harness break a test
        serialized = {
            "unserializable": type(history).__qualname__,
            "error": f"{type(error).__name__}: {error}",
        }
    return _sha256(
        {
            "history": serialized,
            "model": model,
            "base_model": base_model,
            "tokenizer": tokenizer_identity(tokenizer),
            "chat_template": chat_template,
            "chat_template_kwargs": (
                None if chat_template_kwargs is None else dict(chat_template_kwargs)
            ),
        }
    )


def output_digest(tokenized: Any, *, input_hash: str) -> Digest:
    values = field_values(tokenized)
    fields = {name: _sha256(values[name])[:16] for name in FIELDS}  # type: ignore[literal-required]
    return Digest(
        kind=type(tokenized.history).__qualname__,
        n=len(values["tokens"]),
        input=input_hash[:16],
        output=_sha256(values),
        fields=fields,
    )


def describe_mismatch(label: str, expected: Digest, actual: Digest) -> str:
    """Explain the first differing field between two digests."""

    lines = [f"{label}: tokenization output changed"]
    if expected["input"] != actual["input"]:
        lines.append(
            "  the INPUT digest also changed (test inputs or tokenizer identity "
            "differ); regenerate with ART_TOKENIZE_GOLDEN=update if intended"
        )
    else:
        lines.append("  input digest unchanged -> tokenizer behavior changed")
    if expected["kind"] != actual["kind"]:
        lines.append(
            f"  history kind: golden {expected['kind']} != now {actual['kind']}"
        )
    if expected["n"] != actual["n"]:
        lines.append(f"  token count: golden {expected['n']} != now {actual['n']}")
    for name in FIELDS:
        if expected["fields"].get(name) != actual["fields"].get(name):
            lines.append(
                f"  first differing field: {name} "
                f"(golden {expected['fields'].get(name)} != now {actual['fields'].get(name)})"
            )
            break
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, sort_keys=True, indent=0, separators=(",", ":"))
    path.write_text(text + "\n")
