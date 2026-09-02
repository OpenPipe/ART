"""Bounded terminal usage journal for the paired vLLM frontend."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal
import math
from threading import RLock
import time
from typing import Any, Callable

_NANOSECONDS_PER_SECOND = 1_000_000_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000


class RuntimeUsageCapacityError(RuntimeError):
    pass


class RuntimeUsageCursorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeRequestContext:
    tenant_id: str
    run_id: str
    operation_id: str
    service_tier: str
    model: str


@dataclass(slots=True)
class _PendingRequest:
    context: RuntimeRequestContext
    gpu_service_ns: int = 0
    gpu_complete: bool = False
    finished: Any | None = None
    observed_unix_s: float | None = None


@dataclass(slots=True)
class _KVResidency:
    owner: Any
    started_monotonic_ns: int
    last_monotonic_ns: int
    byte_count: int
    byte_ns: int = 0


class RuntimeUsageJournal:
    """Retain exact request facts until the durable consumer acknowledges them."""

    def __init__(
        self,
        source_id: str,
        source_epoch: int,
        *,
        capacity: int = 4096,
        clock: Callable[[], float] = time.time,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.source_id = _identity(source_id, "source_id", 512)
        self.source_epoch = _nonnegative_int(source_epoch, "source_epoch")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("runtime usage capacity must be a positive integer")
        self.capacity = capacity
        self._clock = clock
        self._monotonic_ns = monotonic_ns
        self._started_unix_s = _clock_now(clock)
        self._started_monotonic_ns = _monotonic_now(monotonic_ns)
        self._last_observed_unix_s = 0.0
        self._pending: dict[str, _PendingRequest] = {}
        self._kv_residencies: dict[tuple[int, Any], _KVResidency] = {}
        self._last_kv_ns_by_engine: dict[int, int] = {}
        self._next_kv_receipt = 0
        self._receipts: OrderedDict[int, dict[str, object]] = OrderedDict()
        self._next_sequence = 1
        self._acknowledged_through = 0
        self._lock = RLock()

    def reserve(self, request_identity: str, context: RuntimeRequestContext) -> None:
        request_identity = _identity(request_identity, "request_identity", 255)
        _validate_context(context)
        with self._lock:
            if request_identity in self._pending:
                raise RuntimeError("runtime usage request identity is already active")
            if self._retained_count() >= self.capacity:
                raise RuntimeUsageCapacityError("runtime usage journal is full")
            self._pending[request_identity] = _PendingRequest(context)

    def discard(self, request_identity: str) -> None:
        with self._lock:
            self._pending.pop(request_identity, None)

    def record_gpu_service(self, request_identity: Any, gpu_service_ns: Any) -> None:
        request_identity = _identity(request_identity, "request_identity", 255)
        gpu_service_ns = _nonnegative_int(gpu_service_ns, "gpu_service_ns")
        with self._lock:
            pending = self._pending.get(request_identity)
            if pending is None:
                raise RuntimeError("GPU service belongs to an inactive request")
            pending.gpu_service_ns += gpu_service_ns

    def record_gpu_complete(self, request_identity: Any) -> None:
        request_identity = _identity(request_identity, "request_identity", 255)
        with self._lock:
            pending = self._pending.get(request_identity)
            if pending is None:
                raise RuntimeError("GPU completion belongs to an inactive request")
            if pending.gpu_complete:
                raise RuntimeError("GPU completion was reported more than once")
            pending.gpu_complete = True
            self._finalize_request(request_identity, pending)

    def record_kv_residency(
        self,
        engine_index: Any,
        owner: Any,
        *,
        byte_count: Any,
        monotonic_ns: Any,
    ) -> None:
        engine_index = _bounded_engine_index(engine_index)
        byte_count = _nonnegative_int(byte_count, "KV byte_count")
        monotonic_ns = _nonnegative_int(monotonic_ns, "KV monotonic_ns")
        _validate_kv_owner(owner)
        key = (engine_index, owner)
        with self._lock:
            if monotonic_ns < self._started_monotonic_ns:
                raise RuntimeError("KV residency predates the usage source")
            prior_engine_ns = self._last_kv_ns_by_engine.get(engine_index)
            if prior_engine_ns is not None and monotonic_ns < prior_engine_ns:
                raise RuntimeError("KV residency updates arrived out of order")
            residency = self._kv_residencies.get(key)
            if residency is None:
                if byte_count == 0:
                    raise RuntimeError("KV residency released an unknown owner")
                if self._retained_count() >= self.capacity:
                    raise RuntimeUsageCapacityError("runtime usage journal is full")
                self._kv_residencies[key] = _KVResidency(
                    owner=owner,
                    started_monotonic_ns=monotonic_ns,
                    last_monotonic_ns=monotonic_ns,
                    byte_count=byte_count,
                )
            else:
                residency.byte_ns += residency.byte_count * (
                    monotonic_ns - residency.last_monotonic_ns
                )
                residency.last_monotonic_ns = monotonic_ns
                residency.byte_count = byte_count
                if byte_count == 0:
                    self._kv_residencies.pop(key)
                    self._append_kv_receipt(engine_index, residency)
            self._last_kv_ns_by_engine[engine_index] = monotonic_ns

    def record_finished(
        self, finished: Any, *, observed_unix_s: float | None = None
    ) -> bool:
        request_id = getattr(finished, "request_id", None)
        if not isinstance(request_id, str):
            return False
        with self._lock:
            receipt_id = request_id
            pending = self._pending.get(receipt_id)
            if pending is None and request_id.startswith("chatcmpl-"):
                receipt_id = request_id.removeprefix("chatcmpl-")
                pending = self._pending.get(receipt_id)
            if pending is None:
                return False
            if pending.finished is not None:
                return False
            observed = max(
                self._last_observed_unix_s,
                _clock_now(
                    self._clock if observed_unix_s is None else lambda: observed_unix_s
                ),
            )
            pending.finished = finished
            pending.observed_unix_s = observed
            self._finalize_request(receipt_id, pending)
            return True

    def _finalize_request(self, request_id: str, pending: _PendingRequest) -> None:
        if not pending.gpu_complete or pending.finished is None:
            return
        observed = pending.observed_unix_s
        if observed is None:
            raise RuntimeError("terminal usage observation is unavailable")
        if self._pending.pop(request_id, None) is not pending:
            raise RuntimeError("terminal usage request changed before completion")
        sequence = self._next_sequence
        self._next_sequence += 1
        self._receipts[sequence] = self._receipt(
            request_id, pending, pending.finished, sequence, observed
        )
        self._last_observed_unix_s = observed

    def read(self, *, after_sequence: int, limit: int = 64) -> dict[str, object]:
        after_sequence = _nonnegative_int(after_sequence, "after_sequence")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 64
        ):
            raise ValueError("runtime usage page limit must be between 1 and 64")
        with self._lock:
            produced_through = self._next_sequence - 1
            if after_sequence < self._acknowledged_through:
                raise RuntimeUsageCursorError("runtime usage cursor has expired")
            if after_sequence > produced_through:
                raise RuntimeUsageCursorError("runtime usage cursor is ahead")
            receipts = tuple(
                receipt
                for sequence, receipt in self._receipts.items()
                if sequence > after_sequence
            )[:limit]
            high_watermark = after_sequence + len(receipts)
            if receipts:
                observed_at = receipts[-1]["interval_ended_at"]
            elif not self._pending and not self._kv_residencies:
                observed = max(self._last_observed_unix_s, _clock_now(self._clock))
                self._last_observed_unix_s = observed
                observed_at = _utc_timestamp(observed)
            else:
                observed_at = _utc_timestamp(
                    self._last_observed_unix_s or self._started_unix_s
                )
            return {
                "source_id": self.source_id,
                "source_epoch": self.source_epoch,
                "requested_after_sequence": after_sequence,
                "high_watermark_sequence": high_watermark,
                "dropped_through_sequence": self._acknowledged_through,
                "complete_through_high_watermark": True,
                "observed_at": observed_at,
                "receipts": receipts,
            }

    def acknowledge(self, through_sequence: int) -> int:
        through_sequence = _nonnegative_int(through_sequence, "through_sequence")
        with self._lock:
            produced_through = self._next_sequence - 1
            if not self._acknowledged_through <= through_sequence <= produced_through:
                raise RuntimeUsageCursorError(
                    "runtime usage acknowledgement is invalid"
                )
            for sequence in range(self._acknowledged_through + 1, through_sequence + 1):
                if sequence not in self._receipts:
                    raise RuntimeUsageCursorError(
                        "runtime usage acknowledgement is not contiguous"
                    )
            for sequence in tuple(self._receipts):
                if sequence > through_sequence:
                    break
                self._receipts.pop(sequence)
            self._acknowledged_through = through_sequence
            return through_sequence

    def state(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "source_id": self.source_id,
                "source_epoch": self.source_epoch,
                "produced_through_sequence": self._next_sequence - 1,
                "acknowledged_through_sequence": self._acknowledged_through,
                "active_requests": len(self._pending),
                "active_kv_residencies": len(self._kv_residencies),
                "retained_receipts": len(self._receipts),
                "capacity": self.capacity,
            }

    def _receipt(
        self,
        request_id: str,
        pending: _PendingRequest,
        finished: Any,
        sequence: int,
        observed_unix_s: float,
    ) -> dict[str, object]:
        context = pending.context
        try:
            latency = _finite_nonnegative(
                getattr(finished, "e2e_latency"), "e2e_latency"
            )
            prompt = _nonnegative_int(
                getattr(finished, "num_prompt_tokens"), "num_prompt_tokens"
            )
            cached = _nonnegative_int(
                getattr(finished, "num_cached_tokens"), "num_cached_tokens"
            )
            decode = _nonnegative_int(
                getattr(finished, "num_generation_tokens"),
                "num_generation_tokens",
            )
            if cached > prompt:
                raise ValueError("cached tokens exceed prompt tokens")
            reason = str(getattr(finished, "finish_reason"))
            corrupted = bool(getattr(finished, "is_corrupted", False))
            status, attribution = _terminal_status(reason, corrupted)
            live_request_ns = (
                Decimal(str(latency)) * _NANOSECONDS_PER_SECOND
            ).to_integral_value(rounding=ROUND_HALF_EVEN)
            measurements: tuple[dict[str, object], ...] = (
                {
                    "metric": "live_request_ms",
                    "quantity": str(live_request_ns / _NANOSECONDS_PER_MILLISECOND),
                },
                {
                    "metric": "inference_gpu_ms",
                    "quantity": str(Decimal(pending.gpu_service_ns) / 1_000_000),
                },
                {"metric": "cached_prefill_tokens", "quantity": cached},
                {
                    "metric": "uncached_prefill_tokens",
                    "quantity": prompt - cached,
                },
                {"metric": "decode_tokens", "quantity": decode},
            )
            coverage = "exact"
            started_at = _utc_timestamp(observed_unix_s - latency)
            ended_at = _utc_timestamp(observed_unix_s)
        except (AttributeError, OSError, OverflowError, TypeError, ValueError):
            status, attribution = "failed", "unknown"
            measurements = ()
            coverage = "unknown"
            started_at = ended_at = _utc_timestamp(observed_unix_s)
        return {
            "tenant_id": context.tenant_id,
            "run_id": context.run_id,
            "operation_id": context.operation_id,
            "producer": "inference_request",
            "receipt_id": request_id,
            "status": status,
            "failure_attribution": attribution,
            "interval_started_at": started_at,
            "interval_ended_at": ended_at,
            "dimensions": {
                "model": context.model,
                "runtime_class": "art_vllm",
                "service_tier": context.service_tier,
            },
            "coverage": coverage,
            "measurements": measurements,
            "source_id": self.source_id,
            "source_epoch": self.source_epoch,
            "source_sequence": sequence,
            "retry_index": 0,
        }

    def _append_kv_receipt(self, engine_index: int, residency: _KVResidency) -> None:
        sequence = self._next_sequence
        self._next_sequence += 1
        receipt_index = self._next_kv_receipt
        self._next_kv_receipt += 1
        started = self._monotonic_to_unix_s(residency.started_monotonic_ns)
        ended = self._monotonic_to_unix_s(residency.last_monotonic_ns)
        self._receipts[sequence] = {
            "tenant_id": residency.owner.tenant_id,
            "run_id": residency.owner.run_id,
            "operation_id": None,
            "producer": "residency",
            "receipt_id": f"kv:{engine_index}:{receipt_index}",
            "status": "succeeded",
            "failure_attribution": None,
            "interval_started_at": _utc_timestamp(started),
            "interval_ended_at": _utc_timestamp(ended),
            "dimensions": {
                "model": residency.owner.model,
                "runtime_class": "art_vllm",
                "service_tier": residency.owner.service_tier,
            },
            "coverage": "exact",
            "measurements": (
                {
                    "metric": "kv_byte_ms",
                    "quantity": str(Decimal(residency.byte_ns) / 1_000_000),
                },
            ),
            "source_id": self.source_id,
            "source_epoch": self.source_epoch,
            "source_sequence": sequence,
            "retry_index": 0,
        }
        self._last_observed_unix_s = max(self._last_observed_unix_s, ended)

    def _monotonic_to_unix_s(self, monotonic_ns: int) -> float:
        return (
            self._started_unix_s
            + (monotonic_ns - self._started_monotonic_ns) / 1_000_000_000
        )

    def _retained_count(self) -> int:
        return len(self._pending) + len(self._kv_residencies) + len(self._receipts)


_JOURNAL: RuntimeUsageJournal | None = None


def configure_runtime_usage(source_id: str, source_epoch: int) -> RuntimeUsageJournal:
    global _JOURNAL
    _JOURNAL = RuntimeUsageJournal(source_id, source_epoch)
    return _JOURNAL


def runtime_usage_journal() -> RuntimeUsageJournal:
    if _JOURNAL is None:
        raise RuntimeError("runtime usage journal is not configured")
    return _JOURNAL


def record_finished_requests(iteration_stats: Any | None) -> None:
    journal = _JOURNAL
    if journal is None or iteration_stats is None:
        return
    observed = getattr(iteration_stats, "iteration_timestamp", None)
    for finished in getattr(iteration_stats, "finished_requests", ()):
        journal.record_finished(finished, observed_unix_s=observed)


def _validate_context(context: RuntimeRequestContext) -> None:
    _identity(context.tenant_id, "tenant_id", 255)
    _identity(context.run_id, "run_id", 255)
    _identity(context.operation_id, "operation_id", 255)
    _identity(context.service_tier, "service_tier", 128)
    _identity(context.model, "model", 128)


def _terminal_status(reason: str, corrupted: bool) -> tuple[str, str | None]:
    if corrupted or reason == "error":
        return "failed", "infrastructure"
    if reason == "abort":
        return "cancelled", "unknown"
    if reason in {"stop", "length", "repetition"}:
        return "succeeded", None
    raise ValueError(f"unknown finish reason {reason!r}")


def _identity(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must contain 1-{maximum} characters")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _bounded_engine_index(value: Any) -> int:
    value = _nonnegative_int(value, "engine_index")
    if value >= 4_096:
        raise ValueError("engine_index exceeds its fixed bound")
    return value


def _validate_kv_owner(owner: Any) -> None:
    for name, maximum in (
        ("tenant_id", 255),
        ("run_id", 255),
        ("service_tier", 128),
        ("model", 128),
    ):
        _identity(getattr(owner, name, None), f"KV owner {name}", maximum)


def _utc_timestamp(unix_s: float) -> str:
    return datetime.fromtimestamp(unix_s, timezone.utc).isoformat()


def _clock_now(clock: Callable[[], float]) -> float:
    try:
        result = _finite_nonnegative(clock(), "clock")
        _utc_timestamp(result)
        return result
    except (OSError, OverflowError, TypeError, ValueError):
        return max(0.0, time.time())


def _monotonic_now(clock: Callable[[], int]) -> int:
    try:
        return _nonnegative_int(clock(), "monotonic clock")
    except (OSError, TypeError, ValueError):
        return max(0, time.monotonic_ns())
