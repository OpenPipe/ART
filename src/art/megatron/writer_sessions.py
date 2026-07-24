"""Durable model-scoped writer sessions for revision-bound Megatron updates.

The store is intentionally independent of the backend and service call paths.
It owns one OS file lock, fencing counter, and fsync'd session journal per model.
An ``open`` session can be abandoned safely before it binds an optimizer
operation. A ``bound`` session is ambiguous after owner loss and remains
fail-closed until a higher layer reconciles its durable training receipt.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import time
from typing import Any, Literal, cast
import uuid

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
SessionKind = Literal["current_step", "ordinary"]
SessionState = Literal["open", "bound", "committed", "closed"]


class BoundOperationResolution(StrEnum):
    """Durable disposition proved by the owner of a bound operation."""

    COMMITTED = "committed"
    ABANDONED = "abandoned"
    UNRESOLVED = "unresolved"


BoundOperationResolver = Callable[
    [dict[str, JsonValue], int],
    BoundOperationResolution | bool,
]

_SCHEMA_VERSION = 1
_CAPABILITY_BYTES = 32


class WriterSessionError(RuntimeError):
    """Base class for durable writer-session failures."""


class WriterBusyError(WriterSessionError):
    """Another writer or unexpired durable session owns the model."""


class WriterSessionValidationError(WriterSessionError):
    """A lease does not match the durable session that issued it."""


class WriterBindingMismatchError(WriterSessionError):
    """A single-use session was presented with a different operation."""


class AmbiguousWriterSessionError(WriterSessionError):
    """A bound operation lost its owner before a committed result was recorded."""


class WriterJournalCorruptionError(WriterSessionError):
    """Durable writer state is unreadable or violates its schema."""


@dataclass(frozen=True, slots=True)
class WriterLease:
    """Opaque bearer capability for one model revision and fencing epoch."""

    model_identity: str
    revision: int
    session_id: str
    fence: int
    expires_at: float
    kind: SessionKind
    capability: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class WriterRecovery:
    """Outcome observed while recovering durable state before acquisition."""

    action: Literal[
        "none",
        "abandoned_before_submit",
        "closed_committed",
        "closed_abandoned",
        "ambiguous_after_bind",
        "busy_open",
    ]
    session_id: str | None = None
    revision: int | None = None
    fence: int | None = None


class WriterSessionStore:
    """Service-owned durable writer gate for one canonical model identity."""

    def __init__(
        self,
        *,
        root: Path,
        model_identity: str,
        clock: Callable[[], float] = time.time,
        secret_factory: Callable[[int], bytes] = secrets.token_bytes,
        committed_operation_resolver: BoundOperationResolver | None = None,
    ) -> None:
        if not model_identity:
            raise ValueError("model_identity must not be empty")
        self.model_identity = model_identity
        self._clock = clock
        self._secret_factory = secret_factory
        # The constructor name is retained while service callers migrate from
        # the original committed-or-not boolean resolver. New resolvers return
        # an explicit three-way disposition so missing evidence cannot be
        # confused with proof that a submitted operation was abandoned.
        self._bound_operation_resolver = committed_operation_resolver
        digest = hashlib.sha256(model_identity.encode()).hexdigest()
        self._directory = root / "writer_sessions" / digest
        self._lock_path = self._directory / "writer.lock"
        self._journal_path = self._directory / "session.json"
        self._fence_path = self._directory / "fence"
        self._held_fd: int | None = None
        self._held_session_id: str | None = None

    @property
    def journal_path(self) -> Path:
        return self._journal_path

    @property
    def fence_path(self) -> Path:
        return self._fence_path

    def acquire_current_step(
        self,
        *,
        revision: int,
        ttl_s: float,
    ) -> WriterLease:
        """Acquire a long writer lease before rollout, preparation, and training."""

        return self._acquire(
            revision=revision,
            ttl_s=ttl_s,
            kind="current_step",
            operation_identity=None,
        )

    def acquire_ordinary(
        self,
        *,
        revision: int,
        ttl_s: float,
        operation_identity: Mapping[str, Any],
    ) -> WriterLease:
        """Acquire the same gate and immediately bind a short ordinary update."""

        operation = _canonical_operation(operation_identity)
        lease = self._acquire(
            revision=revision,
            ttl_s=ttl_s,
            kind="ordinary",
            operation_identity=operation,
        )
        return lease

    def acquire_ordinary_open(
        self,
        *,
        revision: int,
        ttl_s: float,
    ) -> WriterLease:
        """Acquire a short ordinary writer without declaring submission.

        Services use this after read-only preflight but before shared staging,
        then call :meth:`bind` immediately before publishing the durable job.
        An exception between acquisition and binding is therefore safely
        classified as ``abandoned_before_submit`` instead of an ambiguous
        optimizer update.
        """

        return self._acquire(
            revision=revision,
            ttl_s=ttl_s,
            kind="ordinary",
            operation_identity=None,
        )

    @contextmanager
    def current_step(
        self,
        *,
        revision: int,
        ttl_s: float,
    ) -> Iterator[WriterLease]:
        """Context wrapper that safely abandons only never-bound work."""

        lease = self.acquire_current_step(revision=revision, ttl_s=ttl_s)
        try:
            yield lease
        except BaseException as exc:
            recovery = self.release(lease)
            if recovery.action == "ambiguous_after_bind":
                raise BaseExceptionGroup(
                    "Current-step work failed after binding an optimizer operation.",
                    [
                        exc,
                        AmbiguousWriterSessionError(
                            "bound writer operation requires receipt reconciliation"
                        ),
                    ],
                ) from None
            raise
        else:
            recovery = self.release(lease)
            if recovery.action == "ambiguous_after_bind":
                raise AmbiguousWriterSessionError(
                    "current-step context exited with a bound but uncommitted "
                    "optimizer operation"
                )

    @contextmanager
    def ordinary(
        self,
        *,
        revision: int,
        ttl_s: float,
        operation_identity: Mapping[str, Any],
    ) -> Iterator[WriterLease]:
        """Context wrapper for a short writer already bound to one operation."""

        lease = self.acquire_ordinary(
            revision=revision,
            ttl_s=ttl_s,
            operation_identity=operation_identity,
        )
        try:
            yield lease
        except BaseException as exc:
            recovery = self.release(lease)
            if recovery.action == "ambiguous_after_bind":
                raise BaseExceptionGroup(
                    "Ordinary writer failed after binding an optimizer operation.",
                    [
                        exc,
                        AmbiguousWriterSessionError(
                            "bound writer operation requires receipt reconciliation"
                        ),
                    ],
                ) from None
            raise
        else:
            recovery = self.release(lease)
            if recovery.action == "ambiguous_after_bind":
                raise AmbiguousWriterSessionError(
                    "ordinary writer exited without a committed optimizer operation"
                )

    def heartbeat(self, lease: WriterLease, *, ttl_s: float) -> WriterLease:
        """Renew an unexpired open or bound lease and return its new value."""

        ttl = _positive_finite_ttl(ttl_s)
        journal = self._validated_active_journal(lease, require_unexpired=True)
        if journal["state"] not in {"open", "bound"}:
            raise WriterSessionValidationError(
                "only open or bound writer sessions may heartbeat"
            )
        now = self._now()
        expires_at = now + ttl
        journal["heartbeat_at"] = now
        journal["expires_at"] = expires_at
        self._write_journal(journal)
        return WriterLease(
            model_identity=lease.model_identity,
            revision=lease.revision,
            session_id=lease.session_id,
            fence=lease.fence,
            expires_at=expires_at,
            kind=lease.kind,
            capability=lease.capability,
        )

    def bind(
        self,
        lease: WriterLease,
        *,
        operation_identity: Mapping[str, Any],
    ) -> None:
        """Bind exactly one immutable operation; identical binding is idempotent."""

        operation = _canonical_operation(operation_identity)
        journal = self._validated_active_journal(lease, require_unexpired=True)
        state = cast(SessionState, journal["state"])
        if state == "open":
            journal["state"] = "bound"
            journal["operation_identity"] = operation
            journal["bound_at"] = self._now()
            self._write_journal(journal)
            return
        if state in {"bound", "committed"}:
            self._require_same_operation(journal, operation)
            return
        raise WriterSessionValidationError("closed writer capability cannot be reused")

    def commit(
        self,
        lease: WriterLease,
        *,
        operation_identity: Mapping[str, Any],
        result_revision: int,
    ) -> None:
        """Record the sole valid transition ``R -> R + 1`` durably."""

        operation = _canonical_operation(operation_identity)
        # Expiry protects admission and pre-submit binding. Once a service has
        # bound the operation while retaining the OS writer gate, a delayed
        # heartbeat cannot make a known successful R -> R+1 result ambiguous.
        journal = self._validated_active_journal(lease, require_unexpired=False)
        self._require_same_operation(journal, operation)
        expected_result = lease.revision + 1
        if result_revision != expected_result:
            raise WriterSessionValidationError(
                "writer commit must advance exactly one revision"
            )
        state = cast(SessionState, journal["state"])
        if state == "bound":
            journal["state"] = "committed"
            journal["result_revision"] = result_revision
            journal["committed_at"] = self._now()
            self._write_journal(journal)
            return
        if state == "committed" and journal["result_revision"] == result_revision:
            return
        raise WriterSessionValidationError(
            "writer session must be bound before it can commit"
        )

    def release(self, lease: WriterLease) -> WriterRecovery:
        """Release the OS gate without hiding an ambiguous bound operation."""

        journal = self._validated_active_journal(
            lease,
            require_unexpired=False,
        )
        try:
            state = cast(SessionState, journal["state"])
            if state == "open":
                journal["state"] = "closed"
                journal["closed_at"] = self._now()
                journal["close_reason"] = "abandoned_before_submit"
                self._write_journal(journal)
                return WriterRecovery(
                    action="abandoned_before_submit",
                    session_id=lease.session_id,
                    revision=lease.revision,
                    fence=lease.fence,
                )
            if state == "committed":
                journal["state"] = "closed"
                journal["closed_at"] = self._now()
                journal["close_reason"] = "committed"
                self._write_journal(journal)
                return WriterRecovery(
                    action="closed_committed",
                    session_id=lease.session_id,
                    revision=lease.revision,
                    fence=lease.fence,
                )
            if state == "bound":
                return WriterRecovery(
                    action="ambiguous_after_bind",
                    session_id=lease.session_id,
                    revision=lease.revision,
                    fence=lease.fence,
                )
            raise WriterSessionValidationError(
                "closed writer capability cannot be reused"
            )
        finally:
            self._release_gate(lease.session_id)

    def inspect(self) -> dict[str, JsonValue] | None:
        """Return validated durable state without exposing the capability secret."""

        journal = self._read_journal()
        if journal is None:
            return None
        return cast(dict[str, JsonValue], json.loads(_canonical_json(journal)))

    def recovery_status(self) -> WriterRecovery:
        """Inspect whether durable state is safe, busy, or ambiguous."""

        journal = self._read_journal()
        if journal is None or journal["state"] == "closed":
            return WriterRecovery(action="none")
        return self._classify_recovery(journal)

    def reconcile(self) -> WriterRecovery:
        """Reconcile recoverable durable state while holding the model gate.

        A bound session is closed only when the configured resolver proves that
        its exact immutable operation either committed ``R + 1`` or was abandoned
        while durable state remained at ``R``. Missing evidence returns
        ``ambiguous_after_bind``; resolver errors, including identity or revision
        mismatches, propagate and fail closed.
        """

        self._acquire_gate()
        try:
            journal = self._read_journal()
            if journal is None or journal["state"] == "closed":
                return WriterRecovery(action="none")
            recovery = self._classify_recovery(journal)
            if recovery.action in {"abandoned_before_submit", "busy_open"}:
                self._close_abandoned(journal)
                return WriterRecovery(
                    action="abandoned_before_submit",
                    session_id=recovery.session_id,
                    revision=recovery.revision,
                    fence=recovery.fence,
                )
            if recovery.action == "closed_committed":
                self._close_committed(journal, recovered=True)
                return recovery
            if recovery.action == "ambiguous_after_bind":
                resolution = self._reconcile_bound(journal)
                if resolution is BoundOperationResolution.UNRESOLVED:
                    return recovery
                action = (
                    "closed_committed"
                    if resolution is BoundOperationResolution.COMMITTED
                    else "closed_abandoned"
                )
                return WriterRecovery(
                    action=action,
                    session_id=recovery.session_id,
                    revision=recovery.revision,
                    fence=recovery.fence,
                )
            return recovery
        finally:
            self._release_gate(None)

    def _acquire(
        self,
        *,
        revision: int,
        ttl_s: float,
        kind: SessionKind,
        operation_identity: Mapping[str, Any] | None,
    ) -> WriterLease:
        if revision < 0:
            raise ValueError("revision must be non-negative")
        ttl = _positive_finite_ttl(ttl_s)
        self._acquire_gate()
        try:
            recovery = self._recover_before_acquire()
            if (
                recovery.action == "closed_committed"
                and recovery.revision is not None
                and revision != recovery.revision + 1
            ):
                raise WriterSessionValidationError(
                    "writer revision is stale after recovering a committed update; "
                    f"authoritative revision is {recovery.revision + 1}"
                )
            fence = self._next_fence()
            session_id = uuid.uuid4().hex
            capability = self._secret_factory(_CAPABILITY_BYTES)
            if not isinstance(capability, bytes) or len(capability) < _CAPABILITY_BYTES:
                raise RuntimeError(
                    "writer capability source must return at least 32 random bytes"
                )
            now = self._now()
            expires_at = now + ttl
            journal: dict[str, JsonValue] = {
                "schema_version": _SCHEMA_VERSION,
                "model_identity": self.model_identity,
                "kind": kind,
                "state": "open",
                "revision": revision,
                "session_id": session_id,
                "fence": fence,
                "capability_sha256": hashlib.sha256(capability).hexdigest(),
                "created_at": now,
                "heartbeat_at": now,
                "expires_at": expires_at,
                "operation_identity": None,
                "bound_at": None,
                "result_revision": None,
                "committed_at": None,
                "closed_at": None,
                "close_reason": None,
            }
            self._write_journal(journal)
            self._held_session_id = session_id
            lease = WriterLease(
                model_identity=self.model_identity,
                revision=revision,
                session_id=session_id,
                fence=fence,
                expires_at=expires_at,
                kind=kind,
                capability=capability,
            )
            if operation_identity is not None:
                self.bind(lease, operation_identity=operation_identity)
            return lease
        except BaseException:
            self._release_gate(None)
            raise

    def _recover_before_acquire(self) -> WriterRecovery:
        journal = self._read_journal()
        if journal is None:
            return WriterRecovery(action="none")
        if journal["state"] == "closed":
            if journal["close_reason"] in {"committed", "recovered_committed"}:
                return WriterRecovery(
                    action="closed_committed",
                    session_id=cast(str, journal["session_id"]),
                    revision=cast(int, journal["revision"]),
                    fence=cast(int, journal["fence"]),
                )
            return WriterRecovery(action="none")
        recovery = self._classify_recovery(journal)
        if recovery.action in {"abandoned_before_submit", "busy_open"}:
            self._close_abandoned(journal)
            return WriterRecovery(
                action="abandoned_before_submit",
                session_id=recovery.session_id,
                revision=recovery.revision,
                fence=recovery.fence,
            )
        if recovery.action == "closed_committed":
            self._close_committed(journal, recovered=True)
            return recovery
        if recovery.action == "ambiguous_after_bind":
            resolution = self._reconcile_bound(journal)
            if resolution is not BoundOperationResolution.UNRESOLVED:
                action = (
                    "closed_committed"
                    if resolution is BoundOperationResolution.COMMITTED
                    else "closed_abandoned"
                )
                return WriterRecovery(
                    action=action,
                    session_id=recovery.session_id,
                    revision=recovery.revision,
                    fence=recovery.fence,
                )
            raise AmbiguousWriterSessionError(
                "bound writer operation has an ambiguous outcome and requires "
                "receipt reconciliation"
            )
        raise AssertionError(f"unexpected writer recovery action: {recovery.action}")

    def _reconcile_bound(
        self,
        journal: dict[str, JsonValue],
    ) -> BoundOperationResolution:
        resolver = self._bound_operation_resolver
        if resolver is None:
            return BoundOperationResolution.UNRESOLVED
        operation = cast(dict[str, JsonValue], journal["operation_identity"])
        result_revision = cast(int, journal["revision"]) + 1
        resolution = _normalize_bound_operation_resolution(
            resolver(operation, result_revision)
        )
        if resolution is BoundOperationResolution.UNRESOLVED:
            return resolution
        now = self._now()
        journal["state"] = "closed"
        journal["closed_at"] = now
        if resolution is BoundOperationResolution.COMMITTED:
            journal["result_revision"] = result_revision
            journal["committed_at"] = now
            journal["close_reason"] = "recovered_committed"
        else:
            journal["result_revision"] = None
            journal["committed_at"] = None
            journal["close_reason"] = "recovered_abandoned"
        self._write_journal(journal)
        return resolution

    def _close_abandoned(self, journal: dict[str, JsonValue]) -> None:
        journal["state"] = "closed"
        journal["closed_at"] = self._now()
        journal["close_reason"] = "abandoned_before_submit"
        self._write_journal(journal)

    def _close_committed(
        self,
        journal: dict[str, JsonValue],
        *,
        recovered: bool,
    ) -> None:
        journal["state"] = "closed"
        journal["closed_at"] = self._now()
        journal["close_reason"] = "recovered_committed" if recovered else "committed"
        self._write_journal(journal)

    def _classify_recovery(
        self,
        journal: dict[str, JsonValue],
    ) -> WriterRecovery:
        state = cast(SessionState, journal["state"])
        common = {
            "session_id": cast(str, journal["session_id"]),
            "revision": cast(int, journal["revision"]),
            "fence": cast(int, journal["fence"]),
        }
        if state == "open":
            if self._now() > cast(float, journal["expires_at"]):
                return WriterRecovery(action="abandoned_before_submit", **common)
            return WriterRecovery(action="busy_open", **common)
        if state == "bound":
            return WriterRecovery(action="ambiguous_after_bind", **common)
        if state == "committed":
            return WriterRecovery(action="closed_committed", **common)
        return WriterRecovery(action="none")

    def _validated_active_journal(
        self,
        lease: WriterLease,
        *,
        require_unexpired: bool,
    ) -> dict[str, JsonValue]:
        if self._held_fd is None or self._held_session_id != lease.session_id:
            raise WriterSessionValidationError(
                "writer capability is not active in this service owner"
            )
        journal = self._read_journal()
        if journal is None:
            raise WriterSessionValidationError("writer session journal is missing")
        expected = {
            "model_identity": lease.model_identity,
            "revision": lease.revision,
            "session_id": lease.session_id,
            "fence": lease.fence,
            "kind": lease.kind,
            "expires_at": lease.expires_at,
        }
        for field_name, expected_value in expected.items():
            if journal[field_name] != expected_value:
                raise WriterSessionValidationError(
                    f"writer capability {field_name} does not match durable session"
                )
        if lease.model_identity != self.model_identity:
            raise WriterSessionValidationError(
                "writer capability belongs to a different model"
            )
        actual_hash = hashlib.sha256(lease.capability).hexdigest()
        if not hmac.compare_digest(
            actual_hash,
            cast(str, journal["capability_sha256"]),
        ):
            raise WriterSessionValidationError("writer capability secret is invalid")
        if require_unexpired and self._now() > lease.expires_at:
            raise WriterSessionValidationError("writer capability has expired")
        return journal

    @staticmethod
    def _require_same_operation(
        journal: dict[str, JsonValue],
        operation: dict[str, JsonValue],
    ) -> None:
        existing = journal["operation_identity"]
        if existing != operation:
            raise WriterBindingMismatchError(
                "writer session is already bound to a different operation"
            )

    def _acquire_gate(self) -> None:
        if self._held_fd is not None:
            raise WriterBusyError("this service owner already holds the writer gate")
        self._ensure_directory()
        descriptor = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise WriterBusyError("another process owns the model writer gate") from exc
        self._held_fd = descriptor

    def _release_gate(self, session_id: str | None) -> None:
        if self._held_fd is None:
            return
        if (
            session_id is not None
            and self._held_session_id is not None
            and session_id != self._held_session_id
        ):
            raise WriterSessionValidationError(
                "cannot release a writer gate owned by another session"
            )
        descriptor = self._held_fd
        self._held_fd = None
        self._held_session_id = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _next_fence(self) -> int:
        current = 0
        if self._fence_path.exists():
            try:
                encoded = self._fence_path.read_text(encoding="ascii")
                current = int(encoded)
            except (OSError, ValueError) as exc:
                raise WriterJournalCorruptionError(
                    "writer fencing counter is unreadable"
                ) from exc
            if current < 0 or encoded != str(current):
                raise WriterJournalCorruptionError(
                    "writer fencing counter is not canonical"
                )
        next_fence = current + 1
        _write_bytes_atomic(self._fence_path, str(next_fence).encode("ascii"))
        return next_fence

    def _read_journal(self) -> dict[str, JsonValue] | None:
        if not self._journal_path.exists():
            return None
        try:
            encoded = self._journal_path.read_text(encoding="utf-8")
            raw = json.loads(encoded)
        except (OSError, ValueError) as exc:
            raise WriterJournalCorruptionError(
                "writer session journal is unreadable"
            ) from exc
        if not isinstance(raw, dict):
            raise WriterJournalCorruptionError(
                "writer session journal must contain a JSON object"
            )
        journal = cast(dict[str, JsonValue], raw)
        _validate_journal(journal, expected_model=self.model_identity)
        if _canonical_json(journal) != encoded:
            raise WriterJournalCorruptionError(
                "writer session journal is not canonical"
            )
        return journal

    def _write_journal(self, journal: dict[str, JsonValue]) -> None:
        _validate_journal(journal, expected_model=self.model_identity)
        _write_bytes_atomic(
            self._journal_path,
            _canonical_json(journal).encode(),
        )

    def _ensure_directory(self) -> None:
        parent = self._directory.parent
        parent.mkdir(parents=True, exist_ok=True)
        created = not self._directory.exists()
        self._directory.mkdir(parents=False, exist_ok=True)
        if created:
            _fsync_directory(parent)

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise RuntimeError("writer-session clock returned a non-finite value")
        return now


def _positive_finite_ttl(value: float) -> float:
    ttl = float(value)
    if not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("writer-session ttl_s must be finite and positive")
    return ttl


def _normalize_bound_operation_resolution(
    value: BoundOperationResolution | bool,
) -> BoundOperationResolution:
    if isinstance(value, BoundOperationResolution):
        return value
    # Temporary compatibility for the existing service callback: the old
    # boolean API could prove only committed (True) or unresolved (False).
    if value is True:
        return BoundOperationResolution.COMMITTED
    if value is False:
        return BoundOperationResolution.UNRESOLVED
    raise WriterSessionValidationError(
        "bound-operation resolver returned an invalid disposition"
    )


def _canonical_operation(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    try:
        encoded = _canonical_json(dict(value))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "writer operation identity must be a finite canonical JSON object"
        ) from exc
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError("writer operation identity must be a non-empty JSON object")
    return cast(dict[str, JsonValue], decoded)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_journal(
    journal: dict[str, JsonValue],
    *,
    expected_model: str,
) -> None:
    expected_fields = {
        "schema_version",
        "model_identity",
        "kind",
        "state",
        "revision",
        "session_id",
        "fence",
        "capability_sha256",
        "created_at",
        "heartbeat_at",
        "expires_at",
        "operation_identity",
        "bound_at",
        "result_revision",
        "committed_at",
        "closed_at",
        "close_reason",
    }
    if set(journal) != expected_fields:
        raise WriterJournalCorruptionError(
            "writer session journal has unexpected fields"
        )
    if journal["schema_version"] != _SCHEMA_VERSION:
        raise WriterJournalCorruptionError(
            "writer session journal has an unsupported schema"
        )
    if journal["model_identity"] != expected_model:
        raise WriterJournalCorruptionError(
            "writer session journal belongs to another model"
        )
    if journal["kind"] not in {"current_step", "ordinary"}:
        raise WriterJournalCorruptionError("writer session journal has an invalid kind")
    if journal["state"] not in {"open", "bound", "committed", "closed"}:
        raise WriterJournalCorruptionError(
            "writer session journal has an invalid state"
        )
    for field_name in ("revision", "fence"):
        value = journal[field_name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < (1 if field_name == "fence" else 0)
        ):
            raise WriterJournalCorruptionError(
                f"writer session journal has an invalid {field_name}"
            )
    if not isinstance(journal["session_id"], str) or not journal["session_id"]:
        raise WriterJournalCorruptionError(
            "writer session journal has an invalid session_id"
        )
    capability_hash = journal["capability_sha256"]
    if (
        not isinstance(capability_hash, str)
        or len(capability_hash) != 64
        or any(character not in "0123456789abcdef" for character in capability_hash)
    ):
        raise WriterJournalCorruptionError(
            "writer session journal has an invalid capability hash"
        )
    for field_name in ("created_at", "heartbeat_at", "expires_at"):
        value = journal[field_name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise WriterJournalCorruptionError(
                f"writer session journal has an invalid {field_name}"
            )
        if not math.isfinite(float(value)):
            raise WriterJournalCorruptionError(
                f"writer session journal has a non-finite {field_name}"
            )
    for field_name in ("bound_at", "committed_at", "closed_at"):
        value = journal[field_name]
        if value is None:
            continue
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise WriterJournalCorruptionError(
                f"writer session journal has an invalid {field_name}"
            )
    if cast(float, journal["created_at"]) > cast(float, journal["heartbeat_at"]):
        raise WriterJournalCorruptionError("writer session heartbeat precedes creation")
    if cast(float, journal["expires_at"]) < cast(float, journal["heartbeat_at"]):
        raise WriterJournalCorruptionError(
            "writer session expiry precedes its heartbeat"
        )

    state = cast(SessionState, journal["state"])
    operation = journal["operation_identity"]
    result_revision = journal["result_revision"]
    if result_revision is not None and (
        not isinstance(result_revision, int)
        or isinstance(result_revision, bool)
        or result_revision < 1
    ):
        raise WriterJournalCorruptionError(
            "writer session journal has an invalid result revision"
        )
    if operation is not None:
        if not isinstance(operation, dict) or not operation:
            raise WriterJournalCorruptionError(
                "writer operation identity must be a non-empty JSON object"
            )
        try:
            _canonical_json(operation)
        except (TypeError, ValueError) as exc:
            raise WriterJournalCorruptionError(
                "writer operation identity is not finite JSON"
            ) from exc
    if state == "open":
        if (
            operation is not None
            or result_revision is not None
            or journal["bound_at"] is not None
            or journal["committed_at"] is not None
            or journal["closed_at"] is not None
            or journal["close_reason"] is not None
        ):
            raise WriterJournalCorruptionError(
                "open writer session contains operation or result state"
            )
    elif state == "bound":
        if (
            operation is None
            or result_revision is not None
            or journal["bound_at"] is None
            or journal["committed_at"] is not None
            or journal["closed_at"] is not None
            or journal["close_reason"] is not None
        ):
            raise WriterJournalCorruptionError(
                "bound writer session lacks exactly one operation"
            )
    elif state == "committed":
        if (
            operation is None
            or journal["bound_at"] is None
            or journal["committed_at"] is None
            or journal["closed_at"] is not None
            or journal["close_reason"] is not None
        ):
            raise WriterJournalCorruptionError(
                "committed writer session lacks an operation"
            )
        if result_revision != cast(int, journal["revision"]) + 1:
            raise WriterJournalCorruptionError(
                "committed writer session has an invalid result revision"
            )
    else:
        reason = journal["close_reason"]
        if reason not in {
            "abandoned_before_submit",
            "committed",
            "recovered_committed",
            "recovered_abandoned",
        }:
            raise WriterJournalCorruptionError(
                "closed writer session has an invalid close reason"
            )
        if reason == "abandoned_before_submit" and (
            operation is not None
            or result_revision is not None
            or journal["bound_at"] is not None
            or journal["committed_at"] is not None
        ):
            raise WriterJournalCorruptionError(
                "abandoned writer session contains submitted operation state"
            )
        if reason in {"committed", "recovered_committed"} and (
            operation is None
            or result_revision != cast(int, journal["revision"]) + 1
            or journal["bound_at"] is None
            or journal["committed_at"] is None
        ):
            raise WriterJournalCorruptionError(
                "closed committed writer session lacks its committed transition"
            )
        if reason == "recovered_abandoned" and (
            operation is None
            or result_revision is not None
            or journal["bound_at"] is None
            or journal["committed_at"] is not None
        ):
            raise WriterJournalCorruptionError(
                "closed abandoned writer session lacks its bound transition"
            )
        if journal["closed_at"] is None:
            raise WriterJournalCorruptionError(
                "closed writer session lacks a close timestamp"
            )


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
