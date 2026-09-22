"""Opt-in, local planner diagnostics. Importing this module does no I/O.

The sink must enqueue paths, not upload on the training thread. Reports survive
sink failure and are JSON only; CPU replay never loads a checkpoint or executes
training. In particular an OOM's partial peak is not a completed measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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

ALLOW_OVERSIZED_ENV = "ART_TRAINER_RANK_ALLOW_OVERSIZED_BATCHES"
MISS_THRESHOLD_ENV = "ART_TRAINER_RANK_PLANNER_MISS_THRESHOLD_PCT"
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_SPOOL_BYTES = 256 * 1024 * 1024
MAX_SPOOL_REPORTS = 1024
_SOURCE_NAMES = (
    "_impl.py",
    "_planner_cost.py",
    "_prefix_tree_planner.py",
    "_prefix_tree_performance_search.py",
    "_planner_misses.py",
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


def _encode(record: dict[str, Any]) -> bytes:
    chunks = bytearray()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(record):
        encoded = chunk.encode("utf-8")
        if len(chunks) + len(encoded) + 1 > MAX_REPORT_BYTES:
            raise ValueError("report exceeds byte limit")
        chunks.extend(encoded)
    return bytes(chunks) + b"\n"


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
        or set(record) != _REPORT_KEYS
        or type(record.get("format")) is not int
        or record["format"] != 1
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
    if record["predicted_peak_bytes"] is None:
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
    if not record["oom"]:
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


def persist_report(raw: bytes, spool_dir: Path) -> Path:
    """Durably retain exact bytes; duplicate delivery is safe, conflicts refuse."""
    record = validate_report(raw)
    with _spool_lock:
        spool_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = spool_dir.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("report spool must be a private directory")
        path = spool_dir / f"{record['id']}.json"
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size != len(raw)
                or path.read_bytes() != raw
            ):
                raise ValueError("existing report identity has different bytes")
            return path
        size = count = 0
        for entry in spool_dir.iterdir():
            item = entry.lstat()
            if not stat.S_ISREG(item.st_mode):
                raise ValueError("unexpected nonregular spool entry")
            count += 1
            size += item.st_size
            if count >= MAX_SPOOL_REPORTS or size + len(raw) > MAX_SPOOL_BYTES:
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
        predicted_peak_bytes: int,
        observed_peak_bytes: int | None,
        phase: str,
        replay_factory: Callable[[], dict[str, Any]],
        oom: bool = False,
        admission_peak_bytes: int | None = None,
        partial_peak_bytes: int | None = None,
    ) -> Path | None:
        """Serialize only a miss; ordinary observation failures never escape."""
        threshold = self.threshold_pct
        if threshold is None:
            return None
        if not oom and (
            observed_peak_bytes is None
            or abs(observed_peak_bytes - predicted_peak_bytes) * 100
            <= threshold * predicted_peak_bytes
        ):
            return None
        try:
            if predicted_peak_bytes < 0 or (
                observed_peak_bytes is not None and observed_peak_bytes < 0
            ):
                raise ValueError("negative memory measurement")
            record: dict[str, Any] = {
                "format": 1,
                "kind": "art-planner-miss",
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
                raw = _encode(record)
            except Exception as exc:
                record["replay"] = None
                record["replay_complete"] = False
                record["incomplete_reasons"] = [
                    f"replay unavailable: {type(exc).__name__}"
                ]
                raw = _encode(record)
            path = persist_report(raw, self.spool_dir)
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
    "sequence_parallel attention_output_gate mlp_activation_factor gdn_layers "
    "checkpointed_moe_layers recompute_modules moe_output_bytes_per_token".split()
)


def replay(
    report: dict[str, Any], *, allow_source_drift: bool = False
) -> dict[str, Any]:
    """Rerun the maintained memory estimator, never arbitrary serialized code.

    This reconstructs the observed candidate's estimate and optionally its
    prefix layouts. It does not rerun distributed admission or reproduce GPU
    execution, allocator fragmentation, or an OOM without the referenced model.
    """
    from . import _impl
    from ._planner_cost import ModelGeometry
    from ._prefix_tree_planner import (
        build_canonical_prefix_tree,
        plan_prefix_tree_layout,
    )

    if report.get("format") != 1 or report.get("kind") != "art-planner-miss":
        raise ValueError("unsupported planner report")
    if not report.get("replay_complete"):
        raise ValueError("report has incomplete replay inputs")
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
    if set(values) != _RANK_FIELDS | {"geometry", "topology"}:
        raise ValueError("memory replay rank fields differ")
    rank = _impl.TrainerRank.__new__(_impl.TrainerRank)
    for name in _RANK_FIELDS:
        setattr(rank, "_" + name, values[name])
    rank._geometry = ModelGeometry(**values["geometry"])
    dp, tp, cp, pp = values["topology"]
    rank._topology_key = lambda: (dp, tp, cp, pp)
    estimates = []
    for item in state["estimates"]:
        signature = dict(item["signature"])
        for name in ("topology", "planner_coefficients", "request_mix", "grad_modes"):
            signature[name] = tuple(signature[name])
        key = _impl._MemorySignature(**signature)
        rank._memory_profiles = (
            {key: _impl._MemoryProfile(**item["profile"])}
            if item["profile"] is not None
            else {}
        )
        actual = rank._estimate_required_memory_bytes_from_values(
            signature=key, **item["arguments"]
        )
        estimates.append(
            {
                "required_bytes": actual,
                "matches": actual == item["expected_required_bytes"],
            }
        )
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
    if not all(item["matches"] for item in result["estimates"] + result["layouts"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
