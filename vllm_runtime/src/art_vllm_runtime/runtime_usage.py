"""Bounded terminal usage journal for the paired vLLM frontend."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import math
from threading import RLock
import time
from typing import Any, Callable


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


class RuntimeUsageJournal:
    """Retain exact request facts until the durable consumer acknowledges them."""

    def __init__(
        self,
        source_id: str,
        source_epoch: int,
        *,
        capacity: int = 4096,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.source_id = _identity(source_id, "source_id", 512)
        self.source_epoch = _nonnegative_int(source_epoch, "source_epoch")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("runtime usage capacity must be a positive integer")
        self.capacity = capacity
        self._clock = clock
        self._started_unix_s = _clock_now(clock)
        self._last_observed_unix_s = 0.0
        self._pending: dict[str, RuntimeRequestContext] = {}
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
            if len(self._pending) + len(self._receipts) >= self.capacity:
                raise RuntimeUsageCapacityError("runtime usage journal is full")
            self._pending[request_identity] = context

    def discard(self, request_identity: str) -> None:
        with self._lock:
            self._pending.pop(request_identity, None)

    def record_finished(
        self, finished: Any, *, observed_unix_s: float | None = None
    ) -> bool:
        request_id = getattr(finished, "request_id", None)
        if not isinstance(request_id, str):
            return False
        with self._lock:
            context = self._pending.pop(request_id, None)
            if context is None:
                return False
            sequence = self._next_sequence
            self._next_sequence += 1
            observed = max(
                self._last_observed_unix_s,
                _clock_now(
                    self._clock if observed_unix_s is None else lambda: observed_unix_s
                ),
            )
            self._receipts[sequence] = self._receipt(
                request_id, context, finished, sequence, observed
            )
            self._last_observed_unix_s = observed
            return True

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
            observed_at = (
                receipts[-1]["interval_ended_at"]
                if receipts
                else _utc_timestamp(self._last_observed_unix_s or self._started_unix_s)
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
                "retained_receipts": len(self._receipts),
                "capacity": self.capacity,
            }

    def _receipt(
        self,
        request_id: str,
        context: RuntimeRequestContext,
        finished: Any,
        sequence: int,
        observed_unix_s: float,
    ) -> dict[str, object]:
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
            measurements: tuple[dict[str, object], ...] = (
                {
                    "metric": "live_request_ms",
                    "quantity": str(Decimal(str(latency)) * 1000),
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


def _identity(value: str, name: str, maximum: int) -> str:
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


def _utc_timestamp(unix_s: float) -> str:
    return datetime.fromtimestamp(unix_s, timezone.utc).isoformat()


def _clock_now(clock: Callable[[], float]) -> float:
    try:
        result = _finite_nonnegative(clock(), "clock")
        _utc_timestamp(result)
        return result
    except (OSError, OverflowError, TypeError, ValueError):
        return max(0.0, time.time())
