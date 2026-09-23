"""Execution-assigned cumulative report allowances; no transport or quota refund."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterator

_UUID = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LEDGER_LIMIT = 512 * 1024


@dataclass(frozen=True)
class RetentionLimits:
    spool_dir: Path
    max_bytes: int
    max_reports: int
    allowance_id: str
    execution_id: str
    producer_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.spool_dir, Path) or not self.spool_dir.is_absolute():
            raise ValueError("assigned planner spool must be an absolute path")
        if (
            type(self.max_bytes) is not int
            or not 0 <= self.max_bytes <= 256 * 1024 * 1024
            or type(self.max_reports) is not int
            or not 0 <= self.max_reports <= 1024
            or any(
                not isinstance(value, str) or not _UUID.fullmatch(value)
                for value in (self.allowance_id, self.execution_id, self.producer_id)
            )
        ):
            raise ValueError("invalid assigned planner allowance")

    def identity(self) -> dict[str, Any]:
        return {
            "format": 1,
            "allowance_id": self.allowance_id,
            "execution_id": self.execution_id,
            "producer_id": self.producer_id,
            "max_bytes": self.max_bytes,
            "max_reports": self.max_reports,
        }


_current: ContextVar[RetentionLimits | None] = ContextVar(
    "planner_retention", default=None
)


@contextmanager
def report_retention_scope(limits: RetentionLimits | None) -> Iterator[None]:
    if limits is not None and not isinstance(limits, RetentionLimits):
        raise ValueError("invalid assigned planner retention scope")
    token = _current.set(limits)
    try:
        yield
    finally:
        _current.reset(token)


def current_limits() -> RetentionLimits | None:
    return _current.get()


def _encode(value: dict[str, Any]) -> bytes:
    raw = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if len(raw) > _LEDGER_LIMIT:
        raise ValueError("planner charge ledger exceeds limit")
    return raw


def _write(path: Path, value: dict[str, Any]) -> None:
    raw = _encode(value)
    descriptor, name = tempfile.mkstemp(prefix=".retention-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(raw)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def charge(
    limits: RetentionLimits,
    event_id: str,
    raw: bytes,
    *,
    count_limit: int,
    byte_limit: int,
) -> None:
    """Commit a charge before the payload write; ambiguous writes never refund it.

    Payload reclamation does not change this ledger. The caller reserves bounded
    ledger/lock metadata separately from cumulative captured payload bytes.
    """
    if not _UUID.fullmatch(event_id):
        raise ValueError("invalid planner charge identity")
    root = limits.spool_dir
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("assigned planner spool must be a private directory")
    lock = os.open(
        root / ".retention.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        if not stat.S_ISREG(os.fstat(lock).st_mode):
            raise ValueError("planner retention lock is not regular")
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / ".retention.json"
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            if any(entry.name != ".retention.lock" for entry in root.iterdir()):
                raise ValueError("assigned spool contains unaccounted evidence")
            ledger = {
                "allowance": limits.identity(),
                "charges": {},
                "omitted": 0,
                "omitted_bytes": 0,
            }
        else:
            with os.fdopen(descriptor, "rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("planner charge ledger is not regular")
                saved = source.read(_LEDGER_LIMIT + 1)
            ledger = json.loads(saved)
            if (
                _encode(ledger) != saved
                or set(ledger) != {"allowance", "charges", "omitted", "omitted_bytes"}
                or _encode(ledger["allowance"]) != _encode(limits.identity())
                or not isinstance(ledger["charges"], dict)
                or type(ledger["omitted"]) is not int
                or not 0 <= ledger["omitted"] <= 2**63 - 1
                or type(ledger["omitted_bytes"]) is not int
                or not 0 <= ledger["omitted_bytes"] <= 2**63 - 1
            ):
                raise ValueError("planner charge ledger differs from enrollment")
        charges = ledger["charges"]
        total = 0
        for identifier, item in charges.items():
            if (
                not _UUID.fullmatch(identifier)
                or not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not _SHA256.fullmatch(item[0])
                or type(item[1]) is not int
                or not 0 < item[1] <= 16 * 1024 * 1024
            ):
                raise ValueError("invalid retained planner charge")
            total += item[1]
        if len(charges) > limits.max_reports or total > limits.max_bytes:
            raise ValueError("planner charge ledger exceeds enrollment")
        identity = [hashlib.sha256(raw).hexdigest(), len(raw)]
        if event_id in charges:
            if charges[event_id] != identity:
                raise ValueError("planner report ID has conflicting charged bytes")
            return
        if len(charges) >= min(limits.max_reports, count_limit) or total + len(
            raw
        ) > min(limits.max_bytes, byte_limit):
            ledger["omitted"] = min(ledger["omitted"] + 1, 2**63 - 1)
            ledger["omitted_bytes"] = min(ledger["omitted_bytes"] + len(raw), 2**63 - 1)
            _write(path, ledger)
            raise ValueError("assigned planner retention exhausted")
        charges[event_id] = identity
        _write(path, ledger)
    finally:
        os.close(lock)
