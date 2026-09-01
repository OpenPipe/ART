"""Bounded private request evidence collected at the vLLM scheduler boundary."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping

_PRIVATE_REQUEST_ID = re.compile(r"^[0-9a-f]{64}$")
_REPORTS_FIELD = "_art_private_request_reports"
_REPORT_OVERFLOW_FIELD = "_art_private_request_report_overflow"
_REPORT_CAPACITY = 4096
_EVENT_CAPACITY = 32


def request_snapshot(scheduler: Any, request: Any) -> dict[str, Any] | None:
    request_id = str(getattr(request, "request_id", ""))
    if _PRIVATE_REQUEST_ID.fullmatch(request_id) is None:
        return None
    prompt_ids = getattr(request, "prompt_token_ids", ()) or ()
    prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or len(prompt_ids))
    computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
    output_tokens = len(getattr(request, "output_token_ids", ()) or ())
    if computed_tokens <= 0:
        phase = "queued"
    elif computed_tokens < prompt_tokens:
        phase = "prefill"
    else:
        phase = "decode"
    return {
        "request_identity": request_id,
        "phase": phase,
        "prompt_tokens": prompt_tokens,
        "computed_tokens": computed_tokens,
        "output_tokens": output_tokens,
        "preemptions": int(getattr(request, "num_preemptions", 0) or 0),
        "policy": _policy_identity(request),
        "kv": _kv_snapshot(scheduler, request_id),
    }


def observe_scheduler_step(
    scheduler: Any,
    scheduler_output: Any,
    requests: Mapping[str, Any],
) -> None:
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", {})
    for request_id, request in requests.items():
        if request is None or _PRIVATE_REQUEST_ID.fullmatch(request_id) is None:
            continue
        snapshot = request_snapshot(scheduler, request)
        if snapshot is None:
            continue
        snapshot["scheduled_tokens"] = int(scheduled.get(request_id, 0) or 0)
        active = request_id in scheduler.requests
        report = _report(scheduler, request_id)
        report["state"] = "active" if active else "terminal"
        report["snapshot"] = snapshot
        if not active and not any(
            event["kind"] == "terminal" for event in report["events"]
        ):
            _append_event(report, {"kind": "terminal", "snapshot": snapshot})


def observe_policy_transition(
    scheduler: Any,
    request: Any,
    *,
    before: dict[str, Any] | None,
    previous_policy: Mapping[str, Any] | None,
    next_policy: Mapping[str, Any],
) -> None:
    if before is None:
        return
    request_id = before["request_identity"]
    after = request_snapshot(scheduler, request)
    if after is None:
        return
    before_kv = before["kv"]
    after_kv = after["kv"]
    report = _report(scheduler, request_id)
    report["state"] = "active"
    report["snapshot"] = after
    _append_event(
        report,
        {
            "kind": "policy_transition",
            "phase": before["phase"],
            "computed_tokens": before["computed_tokens"],
            "prompt_tokens": before["prompt_tokens"],
            "output_tokens": before["output_tokens"],
            "preemptions": before["preemptions"],
            "previous_policy": (
                None if previous_policy is None else dict(previous_policy)
            ),
            "next_policy": dict(next_policy),
            "kv_before": before_kv,
            "kv_after": after_kv,
        },
    )


def observe_preemption(
    scheduler: Any,
    request: Any,
    *,
    before: dict[str, Any] | None,
) -> None:
    if before is None:
        return
    after = request_snapshot(scheduler, request)
    if after is None:
        return
    report = _report(scheduler, before["request_identity"])
    report["state"] = "active"
    report["snapshot"] = after
    _append_event(
        report,
        {
            "kind": "preemption",
            "before": before,
            "after": after,
        },
    )


def request_runtime_report(scheduler: Any, request_id: str) -> dict[str, Any]:
    reports = _reports(scheduler)
    request = scheduler.requests.get(request_id)
    report = reports.get(request_id)
    if request is not None:
        snapshot = request_snapshot(scheduler, request)
        if snapshot is not None:
            report = _report(scheduler, request_id)
            report["state"] = "active"
            report["snapshot"] = snapshot
    active_identities = {
        identity
        for identity in scheduler.requests
        if _PRIVATE_REQUEST_ID.fullmatch(identity) is not None
    }
    active = sum(identity in active_identities for identity in reports)
    return {
        "schema_version": 1,
        "request_identity": request_id,
        "report": None if report is None else deepcopy(report),
        "registry": {
            "capacity": _REPORT_CAPACITY,
            "max_events_per_request": _EVENT_CAPACITY,
            "retained_requests": len(reports),
            "active_requests": active,
            "terminal_requests": len(reports) - active,
            "unretained_active_requests": len(active_identities - reports.keys()),
            "evicted_active_reports": int(
                getattr(scheduler, _REPORT_OVERFLOW_FIELD, 0)
            ),
        },
    }


def _reports(scheduler: Any) -> OrderedDict[str, dict[str, Any]]:
    reports = getattr(scheduler, _REPORTS_FIELD, None)
    if reports is None:
        reports = OrderedDict()
        setattr(scheduler, _REPORTS_FIELD, reports)
    return reports


def _report(scheduler: Any, request_id: str) -> dict[str, Any]:
    reports = _reports(scheduler)
    report = reports.get(request_id)
    if report is not None:
        reports.move_to_end(request_id)
        return report
    if len(reports) >= _REPORT_CAPACITY:
        terminal = next(
            (identity for identity in reports if identity not in scheduler.requests),
            None,
        )
        if terminal is not None:
            reports.pop(terminal)
        else:
            # Admission already bounds active requests. Keep serving if a future
            # runtime raises that bound without updating this private observer.
            reports.popitem(last=False)
            setattr(
                scheduler,
                _REPORT_OVERFLOW_FIELD,
                int(getattr(scheduler, _REPORT_OVERFLOW_FIELD, 0)) + 1,
            )
    report = {
        "state": "active",
        "snapshot": None,
        "events": [],
        "dropped_events": 0,
    }
    reports[request_id] = report
    return report


def _append_event(report: dict[str, Any], event: dict[str, Any]) -> None:
    events = report["events"]
    event["ordinal"] = report["dropped_events"] + len(events)
    if len(events) >= _EVENT_CAPACITY:
        events.pop(0)
        report["dropped_events"] += 1
    events.append(event)


def _policy_identity(request: Any) -> dict[str, Any] | None:
    lora = getattr(request, "lora_request", None)
    if lora is None:
        return None
    generation_id = getattr(lora, "generation_id", None)
    policy_version = getattr(lora, "policy_version", None)
    update_seq = getattr(lora, "update_seq", None)
    if generation_id is None or policy_version is None or update_seq is None:
        return None
    return {
        "lora_slot": str(getattr(lora, "lora_name", "")),
        "generation_id": str(generation_id),
        "policy_version": int(policy_version),
        "update_seq": int(update_seq),
    }


def _kv_snapshot(scheduler: Any, request_id: str) -> dict[str, Any]:
    try:
        groups = scheduler.kv_cache_manager.get_block_ids(request_id)
        normalized = tuple(tuple(int(block) for block in group) for group in groups)
    except Exception as error:
        return {"available": False, "error_type": type(error).__name__}
    return {
        "available": True,
        "groups": [
            {
                "group_index": index,
                "block_count": len(group),
                "block_ids_sha256": hashlib.sha256(
                    json.dumps(group, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for index, group in enumerate(normalized)
        ],
        "total_blocks": sum(len(group) for group in normalized),
    }
