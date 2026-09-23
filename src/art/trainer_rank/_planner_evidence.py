"""Bounded scalar evidence; no CUDA calls, file I/O, or retained traceback objects."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, fields
import json
import math
import re
import time
from typing import Any, Iterator
import uuid

MAX_TRACE = 64
MAX_SUMMARY_BYTES = 64 * 1024
MAX_FAILURE_BYTES = 8 * 1024
EVENTS = frozenset({"estimate_miss", "oom", "admission_refused", "planning_error"})
_current: ContextVar[Decision | None] = ContextVar("planner_decision", default=None)


@dataclass(frozen=True)
class MemorySample:
    attempt_id: str
    ordinal: int
    monotonic_ns: int
    scope: str
    local_required_bytes: int | None
    required_operand_bytes: int
    required_source: str
    required_from_ordinal: int | None
    local_available_bytes: int
    reduced_required_bytes: int
    reduced_available_bytes: int
    safety_factor: float
    reserve_fraction: float
    allocator: str | None = None
    physical_free_bytes: int | None = None
    physical_total_bytes: int | None = None
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    inactive_split_bytes: int | None = None
    reserve_bytes: int | None = None
    test_cap_bytes: int | None = None


class Decision:
    def __init__(
        self, operation: str, *, sync_across_dp: bool, owner: object | None = None
    ) -> None:
        self.id = uuid.uuid4().hex
        self.operation = operation
        self.sync_across_dp = sync_across_dp
        self.owner = owner
        self.outcome = "planning"
        self.first: MemorySample | None = None
        self.selected: MemorySample | None = None
        self.ordinal = 0
        self.events = 0
        self.trace: deque[dict[str, Any]] = deque(maxlen=MAX_TRACE)
        self.omitted = 0
        self.refusal: Any = None
        self.refresh_source: tuple[MemorySample | None] | None = None

    def record(self, kind: str, status: str, **values: Any) -> None:
        unavailable = [
            key
            for key, value in values.items()
            if isinstance(value, float) and not math.isfinite(value)
        ]
        # Invalid budget operands are evidence, not JSON NaNs or invented zeroes.
        values = {
            key: None if key in unavailable else value for key, value in values.items()
        }
        if len(self.trace) == MAX_TRACE:
            self.omitted += 1
        self.trace.append(
            {
                "kind": kind,
                "status": status,
                "values": values,
                "ordinal": self.events,
                "monotonic_ns": time.monotonic_ns(),
                "unavailable_fields": unavailable,
            }
        )
        self.events += 1

    def observe(self, **values: Any) -> MemorySample:
        operand = values.pop("local_required_bytes")
        parent = None if self.refresh_source is None else self.refresh_source[0]
        if parent is not None and parent.attempt_id != self.id:
            parent = None
        sample = MemorySample(
            attempt_id=self.id,
            ordinal=self.ordinal,
            monotonic_ns=time.monotonic_ns(),
            required_operand_bytes=operand,
            required_source="initial"
            if self.refresh_source is None
            else "previous_check",
            required_from_ordinal=None if parent is None else parent.ordinal,
            local_required_bytes=operand
            if self.refresh_source is None
            else (None if parent is None else parent.local_required_bytes),
            **values,
        )
        self.ordinal += 1
        if self.first is None:
            self.first = sample
        self.record("memory_check", "observed", **asdict(sample))
        return sample

    @contextmanager
    def refresh_of(self, parent: MemorySample | None) -> Iterator[None]:
        previous, self.refresh_source = self.refresh_source, (parent,)
        try:
            yield
        finally:
            self.refresh_source = previous

    def snapshot(self) -> dict[str, Any]:
        return {
            "attempt_id": self.id,
            "operation": self.operation,
            "reduction_scope": "world" if self.sync_across_dp else "tp_cp_or_local",
            "outcome": self.outcome,
            "first_sample": None if self.first is None else asdict(self.first),
            "selected_sample": None if self.selected is None else asdict(self.selected),
            "trace": list(self.trace),
            "omitted_trace_entries": self.omitted,
        }


def current(owner: object | None = None) -> Decision | None:
    value = _current.get()
    return (
        value if owner is None or value is not None and value.owner is owner else None
    )


@contextmanager
def scope(decision: Decision) -> Iterator[None]:
    token = _current.set(decision)
    try:
        yield
    finally:
        _current.reset(token)


def record(owner: object, kind: str, status: str, **values: Any) -> None:
    """Best effort at an existing decision point; never change its outcome."""
    try:
        decision = current(owner)
        if decision is not None:
            decision.record(kind, status, **values)
    except Exception:
        pass


def failure(error: BaseException, *, phase: str) -> dict[str, Any]:
    """Read frame identities eagerly, without text, paths, locals, or frame retention."""
    frames: deque[dict[str, Any]] = deque(maxlen=32)
    omitted = 0
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        module = frame.f_globals.get("__name__")
        name = frame.f_code.co_name
        item = {
            "module": module
            if isinstance(module, str) and re.fullmatch(r"[A-Za-z0-9_.]{1,128}", module)
            else None,
            "function": name[:128],
            "line": trace.tb_lineno
            if type(trace.tb_lineno) is int and trace.tb_lineno > 0
            else None,
        }
        if len(frames) == 32:
            omitted += 1
        frames.append(item)
        trace = trace.tb_next
    result: dict[str, Any] = {
        "type": type(error).__name__[:128],
        "phase": phase,
        "frames": list(frames),
        "omitted_frames": omitted,
    }
    while len(json.dumps(result, separators=(",", ":")).encode()) > MAX_FAILURE_BYTES:
        result["frames"].pop(0)
        result["omitted_frames"] += 1
    return result


def bounded(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    """Trim only the optional trace, preserving first/selected decision facts."""
    if decision is None:
        return None
    value: dict[str, Any] = {**decision, "trace": list(decision["trace"])}
    while (
        len(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())
        > MAX_SUMMARY_BYTES
    ):
        if not value["trace"]:
            raise ValueError("decision summary exceeds limit")
        value["trace"].pop(0)
        value["omitted_trace_entries"] += 1
    return value


def validate(decision: Any, failed: Any) -> None:
    if decision is not None:
        if (
            not isinstance(decision, dict)
            or set(decision)
            != {
                "attempt_id",
                "operation",
                "reduction_scope",
                "outcome",
                "first_sample",
                "selected_sample",
                "trace",
                "omitted_trace_entries",
            }
            or not isinstance(decision["attempt_id"], str)
            or re.fullmatch("[0-9a-f]{32}", decision["attempt_id"]) is None
            or decision["operation"] not in {"forward_micro_batches", "dp_rank_forward"}
            or decision["reduction_scope"] not in {"world", "tp_cp_or_local"}
            or decision["outcome"]
            not in {"admitted", "admitted_oversized", "refused", "planning_error"}
            or not isinstance(decision["trace"], list)
            or len(decision["trace"]) > MAX_TRACE
            or type(decision["omitted_trace_entries"]) is not int
            or decision["omitted_trace_entries"] < 0
            or len(
                json.dumps(decision, separators=(",", ":"), allow_nan=False).encode()
            )
            > MAX_SUMMARY_BYTES
        ):
            raise ValueError("invalid planner decision summary")
        for key in ("first_sample", "selected_sample"):
            sample = decision[key]
            if sample is None:
                continue
            if (
                not isinstance(sample, dict)
                or set(sample) != {field.name for field in fields(MemorySample)}
                or sample["attempt_id"] != decision["attempt_id"]
                or sample["scope"] not in {"world", "tp_cp", "local"}
                or sample["required_source"] not in {"initial", "previous_check"}
                or not (
                    sample["allocator"] is None
                    or isinstance(sample["allocator"], str)
                    and len(sample["allocator"]) <= 128
                )
            ):
                raise ValueError("invalid planner memory sample")
            for name, value in sample.items():
                if name in {"attempt_id", "scope", "allocator", "required_source"}:
                    continue
                if name in {"safety_factor", "reserve_fraction"}:
                    valid = (
                        type(value) in (int, float)
                        and math.isfinite(value)
                        and value >= 0
                    )
                else:
                    optional = name in {
                        field.name
                        for field in fields(MemorySample)
                        if field.default is None
                    } | {"local_required_bytes", "required_from_ordinal"}
                    valid = (
                        value is None
                        and optional
                        or type(value) is int
                        and (value >= 0 or name == "test_cap_bytes")
                    )
                if not valid:
                    raise ValueError("invalid planner memory value")
            if (
                sample["required_source"] == "initial"
                and (
                    sample["local_required_bytes"] != sample["required_operand_bytes"]
                    or sample["required_from_ordinal"] is not None
                )
                or sample["required_from_ordinal"] is not None
                and sample["required_from_ordinal"] >= sample["ordinal"]
            ):
                raise ValueError("invalid refreshed requirement provenance")
        previous = -1
        for item in decision["trace"]:
            if (
                not isinstance(item, dict)
                or set(item)
                != {
                    "kind",
                    "status",
                    "values",
                    "ordinal",
                    "monotonic_ns",
                    "unavailable_fields",
                }
                or item["kind"] not in {"memory_check", "recovery", "cache_release"}
                or item["status"]
                not in {
                    "observed",
                    "accounted",
                    "budget_observed",
                    "sampled",
                    "skipped",
                    "failed",
                    "refused",
                    "attempted",
                    "completed",
                }
                or type(item["ordinal"]) is not int
                or item["ordinal"] <= previous
                or type(item["monotonic_ns"]) is not int
                or item["monotonic_ns"] < 0
                or not isinstance(item["values"], dict)
                or not isinstance(item["unavailable_fields"], list)
                or any(
                    key not in item["values"] or item["values"][key] is not None
                    for key in item["unavailable_fields"]
                )
            ):
                raise ValueError("invalid planner recovery trace")
            for name, value in item["values"].items():
                if (
                    not isinstance(name, str)
                    or not re.fullmatch(r"[a-z_]{1,64}", name)
                    or not (
                        value is None
                        or type(value) in (int, bool)
                        or type(value) is float
                        and math.isfinite(value)
                        or isinstance(value, str)
                        and len(value) <= 128
                    )
                ):
                    raise ValueError("invalid planner recovery operand")
            previous = item["ordinal"]
    if failed is not None:
        if (
            not isinstance(failed, dict)
            or set(failed) != {"type", "phase", "frames", "omitted_frames"}
            or not isinstance(failed["frames"], list)
            or len(failed["frames"]) > 32
            or type(failed["omitted_frames"]) is not int
            or failed["omitted_frames"] < 0
            or any(
                not isinstance(failed[name], str) or not 0 < len(failed[name]) <= 128
                for name in ("type", "phase")
            )
            or len(json.dumps(failed, separators=(",", ":"), allow_nan=False).encode())
            > MAX_FAILURE_BYTES
        ):
            raise ValueError("invalid planner failure summary")
        for frame in failed["frames"]:
            if (
                not isinstance(frame, dict)
                or set(frame) != {"module", "function", "line"}
                or not (
                    frame["module"] is None
                    or isinstance(frame["module"], str)
                    and re.fullmatch(r"[A-Za-z0-9_.]{1,128}", frame["module"])
                )
                or not isinstance(frame["function"], str)
                or not 0 < len(frame["function"]) <= 128
                or frame["line"] is not None
                and (type(frame["line"]) is not int or frame["line"] < 1)
            ):
                raise ValueError("invalid planner failure frame")
