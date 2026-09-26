"""Opt-in, local planner diagnostics. Importing this module does no I/O.

The sink must enqueue paths, not upload on the training thread. Reports survive
sink failure and are JSON only; CPU replay never loads a checkpoint or executes
training. In particular an OOM's partial peak is not a completed measurement.

Grouped-plan reports retain observed cost components and plan provenance, but
estimator replay is incomplete until immutable model/slot eligibility and
head/checkpoint/GDN inputs can be reconstructed. Scalar ungrouped estimates use
the maintained arithmetic and a frozen MoE stage inventory; recorded totals are
comparison targets, never replacements for missing estimator inputs.
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
    "_planner_cost.py",
    "_prefix_tree_planner.py",
    "_prefix_tree_performance_search.py",
    "_planner_misses.py",
    "_gdn_memory.py",
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


def _encode(record: dict[str, Any], *, limit: int = MAX_REPORT_BYTES) -> bytes:
    chunks = bytearray()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(record):
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


def validate_report(raw: bytes) -> dict[str, Any]:
    """Validate the bounded, canonical transport without loading code/tensors."""
    if len(raw) > MAX_REPORT_BYTES:
        raise ValueError("report exceeds byte limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate report key")
            result[key] = value
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
    record = validate_report(raw)
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
                raise ValueError("report spool is full; preserve and export reports")
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
        try:
            if (predicted_peak_bytes is not None and predicted_peak_bytes < 0) or (
                observed_peak_bytes is not None and observed_peak_bytes < 0
            ):
                raise ValueError("negative memory measurement")
            record: dict[str, Any] = {
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
            retention = _planner_retention.current_limits()
            path = persist_report(
                raw,
                self.spool_dir if retention is None else retention.spool_dir,
                planning_budget=planning
                and not (failure is not None and failure["type"] == "OutOfMemoryError"),
                retention=retention,
            )
        except Exception as exc:
            self.failures += 1
            _warn(f"local persistence failed ({type(exc).__name__})")
            return None
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
# Recorded by newer ranks; reports from before it replay with 0 (no dense stage).
_OPTIONAL_RANK_FIELDS = frozenset({"dense_recompute_bytes_per_token"})


def _signature_values(values: dict[str, Any]) -> dict[str, Any]:
    values = dict(values)
    for name in ("topology", "planner_coefficients", "request_mix", "grad_modes"):
        values[name] = tuple(values[name])
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
    from . import _impl
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
    rank = _impl.TrainerRank.__new__(_impl.TrainerRank)
    for name in _RANK_FIELDS - {"one_layer_recompute"}:
        setattr(rank, "_" + name, values[name])
    for name in _OPTIONAL_RANK_FIELDS:
        setattr(rank, "_" + name, values.get(name, 0))
    if type(values["one_layer_recompute"]) is not bool:
        raise ValueError("incomplete replay: recompute mode is not recorded")
    rank._recorded_one_layer_recompute = values["one_layer_recompute"]
    rank._moe_forward_stages = tuple(tuple(row) for row in values["moe_forward_stages"])
    rank._geometry = ModelGeometry(**values["geometry"])
    dp, tp, cp, pp = values["topology"]
    rank._topology_key = lambda: (dp, tp, cp, pp)
    estimates = []
    costs = []
    for item in state["estimates"]:
        if "cost_components" not in item:
            raise ValueError("incomplete replay: expected cost components unavailable")
        arguments = item["arguments"]
        # Only the scalar, ungrouped estimator is currently reconstructible.
        # Even an observed zero floor cannot prove runtime eligibility declined.
        if (
            item.get("missing_inputs")
            or arguments.get("group_rows") not in ([], ())
            or arguments.get("slot_refs")
            or arguments.get("head_workspace_bytes", 0)
            or any(arguments.get("checkpoint_floor", (0, 0)))
        ):
            raise ValueError(
                "incomplete replay: immutable runtime estimator facts unavailable"
            )
        key = _impl._MemorySignature(**_signature_values(item["signature"]))
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
    for item in payload.get("layouts", []):
        tree = build_canonical_prefix_tree(item["input_tokens"])
        layout = plan_prefix_tree_layout(tree, frozenset(item["selected_decisions"]))
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
