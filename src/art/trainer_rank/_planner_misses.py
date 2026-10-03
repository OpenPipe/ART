"""Opt-in, local planner diagnostics. Importing this module does no I/O.

The sink must enqueue paths, not upload on the training thread. Reports survive
sink failure and are JSON only; CPU replay never loads a checkpoint or executes
training. In particular an OOM's partial peak is not a completed measurement.

Grouped plans can freeze bounded primitive model/slot eligibility and
head/checkpoint/GDN inputs at selection. CPU replay recomputes their shared
arithmetic and verifies request/layout provenance. Unsupported or historical
reports without those facts remain incomplete. Recorded costs are comparison
targets, never replacements for missing estimator inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import logging
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from typing import Any
import uuid

from . import _planner_evidence, _planner_retention
from ._planner_retention import RetentionLimits as RetentionLimits
from ._planner_retention import report_retention_scope as report_retention_scope

ALLOW_OVERSIZED_ENV = "ART_TRAINER_RANK_ALLOW_OVERSIZED_BATCHES"
MISS_THRESHOLD_ENV = "ART_TRAINER_RANK_PLANNER_MISS_THRESHOLD_PCT"
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_SPOOL_BYTES = 256 * 1024 * 1024
MAX_SPOOL_REPORTS = 1024
MAX_PLANNING_REPORT_BYTES = 256 * 1024
MAX_PLANNING_REPORTS = 64
MAX_PLANNING_SPOOL_BYTES = 16 * 1024 * 1024
_SOURCE_NAMES = (
    "_impl.py",
    "_memory.py",
    "_micro_batch_planner.py",
    "_planner_cost.py",
    "_prefix_tree_planner.py",
    "_prefix_tree_performance_search.py",
    "_planner_misses.py",
    "_gdn_memory.py",
    "_memory_policy.py",
    "_options.py",
    "_planner_replay.py",
    "_planner_evidence.py",
    "_planner_retention.py",
)
_SOURCE_BYTE_LIMIT = 1024 * 1024
_REPORT_KEYS = frozenset(
    "format kind id occurred_at phase oom threshold_pct predicted_peak_bytes "
    "admission_peak_bytes observed_peak_bytes partial_peak_bytes error_pct "
    "replay_complete incomplete_reasons replay_scope replay".split()
)
_sink: Callable[[Path], None] | None = None
_spool_lock = threading.Lock()
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Options:
    allow_oversized_batches: bool = False
    threshold_pct: float | None = None


def parse_options(environ: Mapping[str, str]) -> Options:
    allow = environ.get(ALLOW_OVERSIZED_ENV, "0")
    if allow not in {"0", "1"}:
        raise ValueError(f"{ALLOW_OVERSIZED_ENV} must be 0 or 1")
    threshold = environ.get(MISS_THRESHOLD_ENV)
    value = None if threshold is None else float(threshold)
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{MISS_THRESHOLD_ENV} must be finite and nonnegative")
    return Options(allow == "1", value)


def set_report_sink(sink: Callable[[Path], None] | None) -> None:
    """Install an optional process-local enqueue callback after durable writes."""
    global _sink
    _sink = sink


def _warn(reason: str) -> None:
    # Logging handlers and warning filters must not replace a training error.
    try:
        _logger.warning("Planner miss reporting incomplete: %s", reason)
    except Exception:
        pass


class _ReportTooLarge(ValueError):
    pass


def _json_chunks(value: Any, encoder: json.JSONEncoder, active: set[int]):
    # Token inventories dominate reports. Encode small native-int blocks in C,
    # keeping bounded incremental rejection instead of allocating a whole report.
    sequence = type(value) in (list, tuple)
    mapping = type(value) is dict and all(type(key) is str for key in value)
    if not sequence and not mapping:
        if type(value) in (str, int, float, bool, type(None)):
            yield encoder.encode(value)
        else:
            yield from encoder.iterencode(value)
        return
    identity = id(value)
    if identity in active:
        raise ValueError("Circular reference detected")
    active.add(identity)
    try:
        if mapping:
            yield "{"
            for index, (key, item) in enumerate(sorted(value.items())):
                if index:
                    yield ","
                yield encoder.encode(key)
                yield ":"
                yield from _json_chunks(item, encoder, active)
            yield "}"
        else:
            yield "["
            start = 0
            while start < len(value):
                block = value[start : start + 1024]
                if all(
                    type(item) is int and -(1 << 63) <= item < 1 << 63 for item in block
                ):
                    if start:
                        yield ","
                    yield encoder.encode(block)[1:-1]
                    start += len(block)
                else:
                    # Subclass iterators can mutate later native-list entries.
                    stop = start + len(block)
                    while start < stop and start < len(value):
                        if start:
                            yield ","
                        yield from _json_chunks(value[start], encoder, active)
                        start += 1
            yield "]"
    finally:
        active.remove(identity)


def _encode(record: dict[str, Any], *, limit: int = MAX_REPORT_BYTES) -> bytes:
    chunks = bytearray()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in _json_chunks(record, encoder, set()):
        encoded = chunk.encode("utf-8")
        if len(chunks) + len(encoded) + 1 > limit:
            raise _ReportTooLarge("report exceeds byte limit")
        chunks.extend(encoded)
    return bytes(chunks) + b"\n"


def _compact_planning_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep whole compact facts before bulk inputs; never claim partial replay."""
    payload = record["replay"]
    reasons = [
        *record["incomplete_reasons"],
        "planning replay exceeds report limit",
    ]
    omitted: list[str] = []
    compact: dict[str, Any] = {
        "incomplete_reasons": reasons,
        "omitted_fields": omitted,
        "unlisted_fields": 0,
        "omitted_field_names_truncated": 0,
    }
    result = {
        **record,
        "replay": compact,
        "replay_complete": False,
        "incomplete_reasons": reasons,
    }
    priority = (
        "source_files",
        "source_scope",
        "rank",
        "device",
        "model",
        "model_identity",
        "candidate_matches_check",
        "memory_replay",
    )
    # The maintained snapshot has fewer than 32 fields. Bound optional factory
    # fields too: neither omission names nor repeated encoding may grow without
    # limit just because the report itself exceeds its cap.
    selected = dict.fromkeys((*priority, *islice(payload, 64)))
    compact["unlisted_fields"] = len(payload) - sum(key in payload for key in selected)

    def omit(key: str) -> None:
        if len(key) > 32:
            compact["omitted_field_names_truncated"] += 1
            key = "<field name exceeds limit>"
        omitted.append(key)

    for key in selected:
        if key not in payload or key == "incomplete_reasons":
            continue
        if key in {
            "requests",
            "layouts",
            "omitted_fields",
            "unlisted_fields",
            "omitted_field_names_truncated",
        }:
            omit(key)
            continue
        compact[key] = payload[key]
        try:
            # At most 72 names, each <=32 characters (<=12 JSON bytes/character).
            _encode(result, limit=MAX_PLANNING_REPORT_BYTES - 32 * 1024)
        except _ReportTooLarge:
            del compact[key]
            omit(key)
    return result


def _source_files() -> dict[str, dict[str, str | int]]:
    """Fingerprint current module files, not an attestation of loaded bytecode."""
    result = {}
    remaining = _SOURCE_BYTE_LIMIT
    for name in _SOURCE_NAMES:
        path = Path(__file__).parent / name
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            raw = source.read(remaining + 1)
            after = os.fstat(source.fileno())
        if (
            len(raw) > remaining
            or len(raw) != before.st_size
            or (before.st_size, before.st_mtime_ns)
            != (after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("planner source changed or exceeds byte limit")
        result[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        remaining -= len(raw)
    return result


def _decode_report(raw: bytes, *, generated: bool = False) -> dict[str, Any]:
    """Parse and check schema; generated bytes must come directly from _encode."""
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds byte limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        previous: str | None = None
        for key, value in items:
            if key in result:
                raise ValueError("duplicate report key")
            # Encoder keys may be nonstrings/custom strings: their encoded or
            # Unicode-normalized order can differ from the decoded string order.
            if generated and previous is not None and key < previous:
                raise ValueError("report must be canonical JSON with a final newline")
            result[key] = value
            previous = key
        return result

    record = json.loads(raw, object_pairs_hook=pairs)
    if (
        not isinstance(record, dict)
        or set(record)
        != (
            _REPORT_KEYS | {"event", "decision", "failure"}
            if record.get("format") == 2
            else _REPORT_KEYS
        )
        or type(record.get("format")) is not int
        or record["format"] not in (1, 2)
        or record.get("kind") != "art-planner-miss"
        or not isinstance(record.get("id"), str)
        or re.fullmatch("[0-9a-f]{32}", record["id"]) is None
        or type(record.get("oom")) is not bool
        or type(record.get("replay_complete")) is not bool
        or not isinstance(record.get("incomplete_reasons"), list)
        or not isinstance(record.get("phase"), str)
        or not isinstance(record.get("occurred_at"), str)
        or not isinstance(record.get("replay_scope"), str)
        or any(not isinstance(reason, str) for reason in record["incomplete_reasons"])
        or not (record.get("replay") is None or isinstance(record["replay"], dict))
    ):
        raise ValueError("invalid planner report schema")
    occurred = datetime.fromisoformat(record["occurred_at"])
    offset = occurred.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError("report occurrence time must be UTC")
    for key in (
        "predicted_peak_bytes",
        "admission_peak_bytes",
        "observed_peak_bytes",
        "partial_peak_bytes",
    ):
        value = record[key]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("invalid report memory value")
    event = record.get("event", "oom" if record["oom"] else "estimate_miss")
    if event not in _planner_evidence.EVENTS or record["oom"] != (event == "oom"):
        raise ValueError("invalid planner event")
    if record["format"] == 2:
        _planner_evidence.validate(record["decision"], record["failure"])
    if record["predicted_peak_bytes"] is None and event not in {
        "planning_error",
        "admission_refused",
    }:
        raise ValueError("missing prediction")
    for key in ("threshold_pct", "error_pct"):
        value = record[key]
        if value is None and key == "error_pct":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid report percentage")
    if record["oom"] and (
        record["error_pct"] is not None or record["observed_peak_bytes"] is not None
    ):
        raise ValueError("OOM cannot claim a completed observation")
    if event in {"admission_refused", "planning_error"} and any(
        record[key] is not None
        for key in ("observed_peak_bytes", "partial_peak_bytes", "error_pct")
    ):
        raise ValueError("planning failure cannot claim an execution measurement")
    if event == "estimate_miss":
        predicted, observed = (
            record["predicted_peak_bytes"],
            record["observed_peak_bytes"],
        )
        if observed is None or record["partial_peak_bytes"] is not None:
            raise ValueError("ordinary miss requires a completed observation")
        expected = 100 * abs(observed - predicted) / predicted if predicted else None
        if (
            record["error_pct"] != expected
            or abs(observed - predicted) * 100 <= record["threshold_pct"] * predicted
        ):
            raise ValueError("report percentage/threshold does not match measurements")
    return record


def validate_report(raw: bytes) -> dict[str, Any]:
    """Validate the bounded, canonical transport without loading code/tensors."""
    record = _decode_report(raw)
    if _encode(record) != raw:
        raise ValueError("report must be canonical JSON with a final newline")
    return record


def persist_report(
    raw: bytes,
    spool_dir: Path,
    *,
    planning_budget: bool = False,
    retention: RetentionLimits | None = None,
) -> Path:
    """Durably retain exact bytes; duplicate delivery is safe, conflicts refuse."""
    return _persist_report(
        raw,
        validate_report(raw),
        spool_dir,
        planning_budget=planning_budget,
        retention=retention,
    )


def _persist_report(
    raw: bytes,
    record: dict[str, Any],
    spool_dir: Path,
    *,
    planning_budget: bool,
    retention: RetentionLimits | None,
) -> Path:
    if retention is not None and spool_dir != retention.spool_dir:
        raise ValueError("planner report spool differs from assigned allowance")
    charged = (
        _planner_retention.charge(
            retention,
            record["id"],
            raw,
            count_limit=MAX_PLANNING_REPORTS if planning_budget else MAX_SPOOL_REPORTS,
            byte_limit=MAX_PLANNING_SPOOL_BYTES if planning_budget else MAX_SPOOL_BYTES,
        )
        if retention is not None
        else nullcontext()
    )
    with _spool_lock, charged:
        spool_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = spool_dir.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("report spool must be a private directory")
        path = spool_dir / f"{record['id']}.json"
        if path.exists() or path.is_symlink():
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as existing:
                info = os.fstat(existing.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_size != len(raw)
                    or existing.read(len(raw) + 1) != raw
                ):
                    raise ValueError("existing report identity has different bytes")
                # A prior attempt may have linked the file but failed its
                # durability barrier. Visibility alone cannot acknowledge it.
                os.fsync(existing.fileno())
            directory = os.open(spool_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return path
        # Rank-side ordinary planning failures may use only the first small
        # part of the spool. OOMs/misses and delivery keep their original limit.
        count_limit = (
            min(MAX_SPOOL_REPORTS, MAX_PLANNING_REPORTS)
            if planning_budget
            else MAX_SPOOL_REPORTS
        )
        byte_limit = (
            min(MAX_SPOOL_BYTES, MAX_PLANNING_SPOOL_BYTES)
            if planning_budget
            else MAX_SPOOL_BYTES
        )
        if retention is not None:
            count_limit = min(count_limit, retention.max_reports)
            byte_limit = min(byte_limit, retention.max_bytes)
        size = count = 0
        for entry in spool_dir.iterdir():
            if retention is not None and entry.name in {
                ".retention.lock",
                ".retention.json",
            }:
                # Assigned metadata has its separate reserved allowance.
                continue
            try:
                item = entry.lstat()
            except FileNotFoundError:
                # The uploader may prune an acknowledged report in another
                # process after directory enumeration. It consumes no quota.
                continue
            if not stat.S_ISREG(item.st_mode):
                raise ValueError("unexpected nonregular spool entry")
            count += 1
            size += item.st_size
            if count >= count_limit or size + len(raw) > byte_limit:
                raise _planner_retention.RetentionLimitReached(
                    "report spool is full; preserve and export reports"
                )
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=spool_dir)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            os.link(temporary, path)
            directory = os.open(spool_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(temporary)
        return path


class Reporter:
    def __init__(
        self, threshold_pct: float | None = None, *, spool_dir: Path | None = None
    ) -> None:
        if threshold_pct is not None and (
            not math.isfinite(threshold_pct) or threshold_pct < 0
        ):
            raise ValueError("threshold_pct must be finite and nonnegative")
        self.threshold_pct = threshold_pct
        self.spool_dir = spool_dir or Path(tempfile.gettempdir()) / (
            f"art-planner-misses-{os.getuid()}-{os.getpid()}"
        )
        self.failures = 0

    def report(
        self,
        *,
        predicted_peak_bytes: int | None,
        observed_peak_bytes: int | None,
        phase: str,
        replay_factory: Callable[[], dict[str, Any]],
        oom: bool = False,
        admission_peak_bytes: int | None = None,
        partial_peak_bytes: int | None = None,
        event: str | None = None,
        decision: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
    ) -> Path | None:
        """Serialize only a miss; ordinary observation failures never escape."""
        threshold = self.threshold_pct
        if threshold is None or not _planner_retention.capture_enabled():
            return None
        event = event or ("oom" if oom else "estimate_miss")
        planning = event in {"admission_refused", "planning_error"}
        if event == "estimate_miss" and (
            predicted_peak_bytes is None
            or observed_peak_bytes is None
            or abs(observed_peak_bytes - predicted_peak_bytes) * 100
            <= threshold * predicted_peak_bytes
        ):
            return None
        retention = None
        record = None
        path = None
        try:
            if (predicted_peak_bytes is not None and predicted_peak_bytes < 0) or (
                observed_peak_bytes is not None and observed_peak_bytes < 0
            ):
                raise ValueError("negative memory measurement")
            retention = _planner_retention.current_limits()
            planning_budget = planning and not (
                failure is not None and failure["type"] == "OutOfMemoryError"
            )
            record = {
                "format": 2,
                "kind": "art-planner-miss",
                "event": event,
                "decision": _planner_evidence.bounded(decision),
                "failure": failure,
                "id": uuid.uuid4().hex,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "oom": oom,
                "threshold_pct": threshold,
                "predicted_peak_bytes": predicted_peak_bytes,
                "admission_peak_bytes": admission_peak_bytes,
                "observed_peak_bytes": None if oom else observed_peak_bytes,
                "partial_peak_bytes": partial_peak_bytes if oom else None,
                "error_pct": (
                    100
                    * abs(observed_peak_bytes - predicted_peak_bytes)
                    / predicted_peak_bytes
                    if not oom
                    and observed_peak_bytes is not None
                    and predicted_peak_bytes is not None
                    and predicted_peak_bytes > 0
                    else None
                ),
                "replay_complete": False,
                "incomplete_reasons": [],
                "replay_scope": "cpu-memory-estimator; GPU execution requires checkpoint/runtime",
            }
            if retention is not None and not _planner_retention.remaining(
                retention,
                count_limit=MAX_PLANNING_REPORTS
                if planning_budget
                else MAX_SPOOL_REPORTS,
                byte_limit=MAX_PLANNING_SPOOL_BYTES
                if planning_budget
                else MAX_SPOOL_BYTES,
            ):
                # Only definitive exhaustion can skip construction without changing
                # which smaller reports or static-cap fallbacks remain retainable.
                _planner_retention.omit_unmeasured(retention)
                raise _planner_retention.RetentionLimitReached(
                    "assigned planner retention exhausted before construction"
                )
            try:
                record["replay"] = dict(replay_factory())
                record["replay"]["source_files"] = _source_files()
                record["replay"]["source_scope"] = (
                    "current module files; not loaded-bytecode attestation"
                )
                record["replay_complete"] = bool(
                    record["replay"].get("memory_replay")
                ) and not record["replay"].get("incomplete_reasons")
                record["incomplete_reasons"] = record["replay"].get(
                    "incomplete_reasons", []
                )
                if not record["replay_complete"] and not record["incomplete_reasons"]:
                    record["incomplete_reasons"] = ["memory replay inputs unavailable"]
                try:
                    raw = _encode(
                        record,
                        limit=MAX_PLANNING_REPORT_BYTES
                        if planning
                        else MAX_REPORT_BYTES,
                    )
                except _ReportTooLarge:
                    if not planning:
                        raise
                    record = _compact_planning_record(record)
                    raw = _encode(record, limit=MAX_PLANNING_REPORT_BYTES)
            except Exception as exc:
                record["replay"] = None
                record["replay_complete"] = False
                record["incomplete_reasons"] = [
                    f"replay unavailable: {'ValueError' if isinstance(exc, _ReportTooLarge) else type(exc).__name__}"
                ]
                raw = _encode(record)
            # Only this producer bypasses the canonical re-encode: _encode
            # already fixed scalar spelling/escaping/whitespace and the newline.
            # Decode still checks key normalization, duplicates and full schema.
            path = _persist_report(
                raw,
                _decode_report(raw, generated=True),
                self.spool_dir if retention is None else retention.spool_dir,
                planning_budget=planning_budget,
                retention=retention,
            )
        except Exception as exc:
            self.failures += 1
            _warn(
                "report retention limit reached; report omitted"
                if isinstance(exc, _planner_retention.RetentionLimitReached)
                else f"local persistence failed ({type(exc).__name__})"
            )
            return None
        finally:
            if retention is not None and record is not None:
                try:
                    _planner_retention.summarize(
                        retention, record, retained=path is not None
                    )
                except Exception as exc:
                    _warn(f"capture summary unavailable ({type(exc).__name__})")
        if not record["replay_complete"]:
            _warn(f"partial replay retained at {path}")
        if _sink is not None:
            try:
                _sink(path)
            except Exception as exc:
                self.failures += 1
                _warn(f"sink failed ({type(exc).__name__}); retained at {path}")
        return path


_RANK_FIELDS = frozenset(
    "num_layers hidden_size param_dtype_size recompute_granularity "
    "one_layer_recompute sequence_parallel attention_output_gate "
    "mlp_activation_factor gdn_layers "
    "checkpointed_moe_layers recompute_modules moe_output_bytes_per_token "
    "moe_forward_stages".split()
)
# TP x SP floor facts; reports captured before them replay conservatively.
_OPTIONAL_RANK_FIELDS = frozenset({"mixer_top_gaps", "lora_modules_per_layer"})


def _adapter_ranks_unavailable(
    topology: tuple[int, ...], facts: dict[str, Any], signature: Any
) -> bool:
    """A TP x SP floor estimate trains a named adapter whose ranks (the LoRA
    intermediates' price) were not recorded, as in reports from before
    sequence-parallel signatures carried slot shapes."""
    return (
        topology[1] > 1
        and bool(facts["checkpoint_layers"])
        and any(
            group["grad"] and group["adapter"] is not None for group in facts["groups"]
        )
        and not any(
            len(shape) == 5 and shape[0] == 2
            for grad, shapes in signature.slot_shapes
            if grad
            for shape in shapes
        )
    )


def _signature_values(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    for name in ("topology", "planner_coefficients", "request_mix", "grad_modes"):
        values[name] = tuple(values[name])
    values["memory_placement"] = tuple(
        tuple(placement) for placement in values.get("memory_placement", ())
    )
    slots = []
    raw_slots = values.get("slot_shapes", ())
    if not isinstance(raw_slots, (list, tuple)):
        raise ValueError("invalid slot_shapes container")
    for entry in raw_slots:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("invalid slot_shapes entry")
        enabled, shapes = entry
        if type(enabled) is not bool or not isinstance(shapes, (list, tuple)):
            raise ValueError("invalid slot_shapes types")
        normalized = []
        for shape in shapes:
            if not isinstance(shape, (list, tuple)) or any(
                type(dim) is not int or dim < 0 for dim in shape
            ):
                raise ValueError("invalid slot_shapes dimensions")
            normalized.append(tuple(shape))
        slots.append((enabled, tuple(normalized)))
    values["slot_shapes"] = tuple(slots)
    return values


def replay(
    report: dict[str, Any], *, allow_source_drift: bool = False
) -> dict[str, Any]:
    """Rerun the maintained memory estimator, never arbitrary serialized code.

    This reconstructs the observed candidate's estimate and optionally its
    prefix layouts. It does not rerun distributed admission or reproduce GPU
    execution, allocator fragmentation, or an OOM without the referenced model.
    """
    if report.get("format") not in (1, 2) or report.get("kind") != "art-planner-miss":
        raise ValueError("unsupported planner report")
    if not report.get("replay_complete"):
        raise ValueError(
            "report has incomplete replay inputs: "
            + "; ".join(report.get("incomplete_reasons", []))
        )
    from . import _impl, _planner_replay
    from ._planner_cost import ModelGeometry
    from ._prefix_tree_planner import (
        build_canonical_prefix_tree,
        plan_prefix_tree_layout,
    )

    payload = report["replay"]
    source_matches = payload["source_files"] == _source_files()
    if not source_matches and not allow_source_drift:
        raise ValueError(
            "planner source differs; use --allow-source-drift for comparison"
        )
    state = payload["memory_replay"]
    if not state["estimates"]:
        raise ValueError("memory replay has no candidate estimates")
    values = state["rank"]
    if set(values) - _OPTIONAL_RANK_FIELDS != _RANK_FIELDS | {"geometry", "topology"}:
        raise ValueError(
            "incomplete replay: immutable rank fields differ (including MoE stages)"
        )
    layers = values["num_layers"]
    if type(layers) is not int or not 0 < layers <= _planner_replay.MAX_LAYERS:
        raise ValueError("incomplete replay: recorded layer count out of bounds")
    rank = _planner_replay.ReplayRank.__new__(_planner_replay.ReplayRank)
    for name in _RANK_FIELDS - {"one_layer_recompute"}:
        setattr(rank, "_" + name, values[name])
    if type(values["one_layer_recompute"]) is not bool:
        raise ValueError("incomplete replay: recompute mode is not recorded")
    rank._recorded_one_layer_recompute = values["one_layer_recompute"]
    rank._moe_forward_stages = tuple(tuple(row) for row in values["moe_forward_stages"])
    gaps = values.get("mixer_top_gaps")
    if gaps is not None and (
        type(gaps) is not list
        or len(gaps) != 2
        or any(
            gap is not None and (type(gap) is not int or not 0 <= gap < layers)
            for gap in gaps
        )
    ):
        raise ValueError("incomplete replay: invalid mixer layer order")
    rank._mixer_top_gaps = None if gaps is None else tuple(gaps)
    modules = values.get("lora_modules_per_layer")
    if modules is not None and (type(modules) is not int or not 0 <= modules < 2**16):
        raise ValueError("incomplete replay: invalid LoRA module count")
    rank._lora_modules_per_layer = modules
    rank._geometry = ModelGeometry(**values["geometry"])
    dp, tp, cp, pp = values["topology"]
    rank._topology_key = lambda: (dp, tp, cp, pp)
    # Check the same shared subforward inventories as capture, including the
    # independently retained layout inputs, before constructing any prefix tree.
    group_offset = request_offset = 0
    for index, item in enumerate(state["estimates"]):
        rows = item["arguments"].get("group_rows")
        if rows in ([], ()):
            continue
        facts = item.get("runtime_facts")
        if type(rows) not in (list, tuple) or facts is None:
            raise ValueError(
                "incomplete replay: immutable runtime estimator facts unavailable"
            )
        _planner_replay.validate(facts)
        groups = facts["groups"]
        count = len(groups)
        if (
            len(rows) != count
            or type(payload.get("subforward_group_counts")) is not list
            or index >= len(payload["subforward_group_counts"])
            or payload["subforward_group_counts"][index] != count
            or any(
                type(payload.get(name)) is not list
                or len(payload[name]) < group_offset + count
                for name in ("layouts", "checkpoint_slots", "group_request_indices")
            )
        ):
            raise ValueError("invalid grouped replay inventory")
        requests = sum(len(g["request_indices"]) for g in groups)
        if (
            type(payload.get("requests")) is not list
            or len(payload["requests"]) < request_offset + requests
        ):
            raise ValueError("invalid grouped replay request inventory")
        _planner_replay.validate_tokens(
            value
            for record in payload["requests"][
                request_offset : request_offset + requests
            ]
            for value in (record["input_tokens"], record["target_tokens"])
            if value is not None
        )
        _planner_replay.validate_tokens(
            layout["input_tokens"]
            for layout in payload["layouts"][group_offset : group_offset + count]
        )
        group_offset += count
        request_offset += requests
    if group_offset and (
        len(payload["subforward_group_counts"]) != len(state["estimates"])
        or sum(payload["subforward_group_counts"]) != group_offset
        or any(
            len(payload[name]) != group_offset
            for name in ("checkpoint_slots", "group_request_indices", "layouts")
        )
        or request_offset != len(payload["requests"])
    ):
        raise ValueError("unused grouped replay metadata")
    selected_layouts = [
        plan_prefix_tree_layout(
            build_canonical_prefix_tree(item["input_tokens"]),
            frozenset(item["selected_decisions"]),
        )
        for item in payload.get("layouts", [])
    ]
    estimates = []
    costs = []
    group_cursor = 0
    request_cursor = 0
    for estimate_index, item in enumerate(state["estimates"]):
        if "cost_components" not in item:
            raise ValueError("incomplete replay: expected cost components unavailable")
        arguments = item["arguments"]
        grouped = arguments.get("group_rows") not in ([], ())
        if (
            item.get("missing_inputs")
            or (grouped and item.get("runtime_facts") is None)
            or (
                grouped
                and any(
                    name in arguments
                    for name in (
                        "slot_refs",
                        "head_workspace_bytes",
                        "checkpoint_floor",
                    )
                )
            )
            or arguments.get("slot_refs")
            or arguments.get("head_workspace_bytes", 0)
            or any(arguments.get("checkpoint_floor", (0, 0)))
        ):
            raise ValueError(
                "incomplete replay: immutable runtime estimator facts unavailable"
            )
        rank._facts = None
        if grouped:
            arguments = rank.runtime_arguments(item["runtime_facts"], arguments)
            groups = item["runtime_facts"]["groups"]
            if len(groups) != payload["subforward_group_counts"][estimate_index]:
                raise ValueError("runtime facts disagree with selected subforward")
            for group in groups:
                if (
                    group["slot"] != payload["checkpoint_slots"][group_cursor]
                    or group["request_indices"]
                    != payload["group_request_indices"][group_cursor]
                    or group["layout_fingerprint"]
                    != payload["layouts"][group_cursor]["expected_fingerprint"]
                    or group["packed_rows"]
                    != payload["layouts"][group_cursor]["expected_packed_tokens"]
                ):
                    raise ValueError(
                        "runtime facts disagree with selected group/layout"
                    )
                count = len(group["request_indices"])
                rank.verify_group(
                    group,
                    selected_layouts[group_cursor],
                    payload["requests"][request_cursor : request_cursor + count],
                )
                request_cursor += count
                group_cursor += 1
        key = _impl._MemorySignature(**_signature_values(item["signature"]))
        if grouped and _adapter_ranks_unavailable(
            rank._topology_key(), item["runtime_facts"], key
        ):
            raise ValueError("incomplete replay: TP x SP adapter ranks unavailable")
        rank._memory_profiles = (
            {key: _impl._MemoryProfile(**item["profile"])}
            if item["profile"] is not None
            else {}
        )
        cost = rank._subforward_cost(signature=key, **arguments)
        costs.append(cost)
        estimates.append(
            {
                "required_bytes": cost.required,
                "retained_bytes": cost.retained,
                "matches": cost.required == item["expected_required_bytes"]
                and cost.retained == item["retained_bytes"]
                and asdict(cost) == item["cost_components"],
            }
        )
    safety = _impl._MEMORY_SAFETY_FACTOR
    required = max(
        rank._split_required_memory(costs),
        int(payload["split_memory_floor_bytes"] * safety),
    )
    predicted = round(required / safety)
    aggregate = {
        "local_admission_peak_bytes": required,
        "predicted_peak_bytes": predicted,
        # Reduced admission is a separately recorded cross-rank result. CPU
        # replay verifies that join, not the absent peers' measurements.
        "matches": required == payload["local_admission_peak_bytes"]
        and predicted == report["predicted_peak_bytes"]
        and safety == payload["safety_factor"]
        and report["admission_peak_bytes"] == payload["reduced_admission_peak_bytes"],
    }
    layouts = []
    for item, layout in zip(payload.get("layouts", []), selected_layouts, strict=True):
        layouts.append(
            {
                "fingerprint": layout.fingerprint,
                "packed_tokens": layout.packed_tokens,
                "matches": layout.fingerprint == item["expected_fingerprint"]
                and layout.packed_tokens == item["expected_packed_tokens"],
            }
        )
    return {
        "scope": "cpu-memory-estimator",
        "source_matches": source_matches,
        "estimates": estimates,
        "aggregate": aggregate,
        "layouts": layouts,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--allow-source-drift", action="store_true")
    args = parser.parse_args()
    with args.report.open("rb") as source:
        raw = source.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds byte limit")
    result = replay(validate_report(raw), allow_source_drift=args.allow_source_drift)
    print(json.dumps(result, sort_keys=True))
    if not result["aggregate"]["matches"] or not all(
        item["matches"] for item in result["estimates"] + result["layouts"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
