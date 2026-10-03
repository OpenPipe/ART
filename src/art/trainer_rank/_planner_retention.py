"""Execution-assigned cumulative report allowances; no transport or quota refund.

``omitted_bytes`` counts serialized failed attempts only. ``omitted_unmeasured``
counts omissions before construction; their unknown sizes are never fabricated.
Both contribute to ``omitted``. Existing ledgers need not contain the new counter.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterator

_UUID = re.compile(r"[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LEDGER_LIMIT = 512 * 1024
# Fixed event keys and one scalar observation per key, inside reserved metadata.
_SUMMARY_LIMIT = 16 * 1024
_EVENTS = frozenset({"estimate_miss", "oom", "admission_refused", "planning_error"})
_SUMMARY_FIELDS = frozenset(
    "id occurred_at phase oom threshold_pct predicted_peak_bytes "
    "admission_peak_bytes observed_peak_bytes partial_peak_bytes error_pct".split()
)


def _validate_summaries(value: Any) -> None:
    if not isinstance(value, dict) or set(value) - _EVENTS:
        raise ValueError("invalid planner event summaries")
    for event in value.values():
        if not isinstance(event, dict) or set(event) != {
            "retained",
            "omitted",
            "latest",
        }:
            raise ValueError("invalid planner event summary")
        if any(
            type(event[k]) is not int or not 0 <= event[k] < 2**63
            for k in ("retained", "omitted")
        ):
            raise ValueError("invalid planner summary counter")
        latest = event["latest"]
        if not isinstance(latest, dict) or set(latest) != _SUMMARY_FIELDS | {
            "retained",
            "failure_type",
        }:
            raise ValueError("invalid planner summary observation")
        for key, item in latest.items():
            if key in {"retained", "oom"}:
                valid = type(item) is bool
            elif key in {"id", "occurred_at", "phase", "failure_type"}:
                valid = isinstance(item, str) and len(item) <= 80
            else:
                valid = item is None or (
                    type(item) in (int, float) and math.isfinite(item) and item >= 0
                )
            if not valid:
                raise ValueError("invalid planner summary scalar")
    if len(_encode(value)) > _SUMMARY_LIMIT:
        raise ValueError("planner event summaries exceed limit")


class RetentionLimitReached(ValueError):
    """Expected refusal when a bounded report allowance is exhausted."""


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


_current: ContextVar[tuple[RetentionLimits | None, bool]] = ContextVar(
    "planner_retention", default=(None, True)
)


@contextmanager
def report_retention_scope(
    limits: RetentionLimits | None, *, capture: bool = True
) -> Iterator[None]:
    if type(capture) is not bool:
        raise ValueError("planner capture must be a boolean")
    if limits is not None and not isinstance(limits, RetentionLimits):
        raise ValueError("invalid assigned planner retention scope")
    token = _current.set((limits, capture and capture_enabled()))
    try:
        yield
    finally:
        _current.reset(token)


def current_limits() -> RetentionLimits | None:
    return _current.get()[0]


def capture_enabled() -> bool:
    return _current.get()[1]


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


@contextmanager
def _ledger(limits: RetentionLimits) -> Iterator[tuple[Path, dict[str, Any], int]]:
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
                or set(ledger) - {"omitted_unmeasured", "event_summaries"}
                != {"allowance", "charges", "omitted", "omitted_bytes"}
                or _encode(ledger["allowance"]) != _encode(limits.identity())
                or not isinstance(ledger["charges"], dict)
                or type(ledger["omitted"]) is not int
                or not 0 <= ledger["omitted"] <= 2**63 - 1
                or type(ledger["omitted_bytes"]) is not int
                or not 0 <= ledger["omitted_bytes"] <= 2**63 - 1
                or type(ledger.get("omitted_unmeasured", 0)) is not int
                or not 0 <= ledger.get("omitted_unmeasured", 0) <= ledger["omitted"]
            ):
                raise ValueError("planner charge ledger differs from enrollment")
        _validate_summaries(ledger.get("event_summaries", {}))
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
        yield path, ledger, total
    finally:
        os.close(lock)


def remaining(limits: RetentionLimits, *, count_limit: int, byte_limit: int) -> int:
    """Snapshot only; concurrent producers must still charge their final bytes."""
    with _ledger(limits) as (_, ledger, total):
        return (
            0
            if len(ledger["charges"]) >= min(limits.max_reports, count_limit)
            else max(0, min(limits.max_bytes, byte_limit) - total)
        )


def omit_unmeasured(limits: RetentionLimits) -> None:
    """Count a skipped construction without inventing its serialized size."""
    with _ledger(limits) as (path, ledger, _):
        ledger["omitted"] = min(ledger["omitted"] + 1, 2**63 - 1)
        ledger["omitted_unmeasured"] = min(
            ledger.get("omitted_unmeasured", 0) + 1, 2**63 - 1
        )
        _write(path, ledger)


@contextmanager
def charge(
    limits: RetentionLimits,
    event_id: str,
    raw: bytes,
    *,
    count_limit: int,
    byte_limit: int,
) -> Iterator[None]:
    """Commit a charge before the payload write; ambiguous writes never refund it.

    Payload reclamation does not change this ledger. The caller reserves bounded
    ledger/lock metadata separately from cumulative captured payload bytes.
    """
    if not _UUID.fullmatch(event_id):
        raise ValueError("invalid planner charge identity")
    with _ledger(limits) as (path, ledger, total):
        charges = ledger["charges"]
        identity = [hashlib.sha256(raw).hexdigest(), len(raw)]
        if event_id in charges and charges[event_id] != identity:
            raise ValueError("planner report ID has conflicting charged bytes")
        try:
            if event_id not in charges:
                if len(charges) >= min(limits.max_reports, count_limit) or total + len(
                    raw
                ) > min(limits.max_bytes, byte_limit):
                    raise RetentionLimitReached("assigned planner retention exhausted")
                charges[event_id] = identity
                _write(path, ledger)
            yield
        except Exception:
            # Count failed attempts even after charging (including duplicate
            # retries blocked by crash leftovers); never refund uncertain writes.
            ledger["omitted"] = min(ledger["omitted"] + 1, 2**63 - 1)
            ledger["omitted_bytes"] = min(ledger["omitted_bytes"] + len(raw), 2**63 - 1)
            with suppress(Exception):
                _write(path, ledger)
            raise


def summarize(
    limits: RetentionLimits, record: dict[str, Any], *, retained: bool
) -> None:
    """Keep counts and the latest scalar event, never replay inputs or tracebacks.

    This is producer capture coverage, not delivery coverage. Historical attempts
    and failed summary writes are not reconstructed. Counters saturate; storage
    remains bounded even after the cumulative payload allowance is exhausted.
    """
    event = record["event"]
    latest = {key: record[key] for key in _SUMMARY_FIELDS}
    latest["retained"] = retained
    latest["phase"] = latest["phase"][:80]
    failure = record.get("failure")
    latest["failure_type"] = str(failure.get("type", ""))[:80] if failure else ""
    update = {"retained": int(retained), "omitted": int(not retained), "latest": latest}
    _validate_summaries({event: update})
    with _ledger(limits) as (path, ledger, _):
        summaries = ledger.setdefault("event_summaries", {})
        previous = summaries.get(event, {})
        for key in ("retained", "omitted"):
            update[key] = min(previous.get(key, 0) + update[key], 2**63 - 1)
        summaries[event] = update
        _write(path, ledger)


def coverage(limits: RetentionLimits) -> dict[str, Any]:
    """Bounded capture snapshot. Charges include uncertain writes, not ACKs.

    Remaining bytes do not promise the next full report fits. Event summaries
    cover supported Reporter calls only, not all training batches or historical
    calls. Missing snapshots must never be interpreted as complete coverage.
    """
    with _ledger(limits) as (_, ledger, total):
        summaries = ledger.get("event_summaries", {})
        return {
            "allowance": limits.identity(),
            "charged_reports": len(ledger["charges"]),
            "charged_bytes": total,
            "remaining_reports": limits.max_reports - len(ledger["charges"]),
            "remaining_bytes": limits.max_bytes - total,
            "omitted": ledger["omitted"],
            "omitted_bytes": ledger["omitted_bytes"],
            "omitted_unmeasured": ledger.get("omitted_unmeasured", 0),
            "event_summaries": summaries,
        }


def validate_coverage(value: dict[str, Any], *, allowance: dict[str, Any]) -> None:
    """Validate the bounded producer snapshot separately from transport ledgers."""
    counters = {
        "charged_reports",
        "charged_bytes",
        "remaining_reports",
        "remaining_bytes",
        "omitted",
        "omitted_bytes",
        "omitted_unmeasured",
    }
    if (
        not isinstance(value, dict)
        or set(value) != counters | {"allowance", "event_summaries"}
        or value["allowance"] != allowance
        or any(
            type(value[key]) is not int or not 0 <= value[key] < 2**63
            for key in counters
        )
        or value["charged_reports"] + value["remaining_reports"]
        != allowance["max_reports"]
        or value["charged_bytes"] + value["remaining_bytes"] != allowance["max_bytes"]
        or value["omitted_unmeasured"] > value["omitted"]
        or len(_encode(value)) > _SUMMARY_LIMIT
    ):
        raise ValueError("invalid planner capture coverage")
    _validate_summaries(value["event_summaries"])
