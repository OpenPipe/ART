from types import SimpleNamespace

from art_vllm_runtime.resource_usage import KVUsageOwner
from art_vllm_runtime.runtime_usage import (
    RuntimeRequestContext,
    RuntimeUsageCapacityError,
    RuntimeUsageCursorError,
    RuntimeUsageJournal,
)
import pytest


def _context(tenant: str) -> RuntimeRequestContext:
    return RuntimeRequestContext(
        tenant_id=tenant,
        run_id=f"run-{tenant}",
        operation_id=f"operation-{tenant}",
        service_tier="standard",
        model="test-model",
    )


def _finished(request_id: str, reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        finish_reason=reason,
        e2e_latency=0.125,
        num_prompt_tokens=12,
        num_cached_tokens=5,
        num_generation_tokens=3,
        is_corrupted=False,
    )


def test_runtime_usage_pages_are_gapless_bounded_and_acknowledged() -> None:
    now = [20.0]
    journal = RuntimeUsageJournal("paired-runtime", 4, capacity=2, clock=lambda: now[0])
    journal.reserve("request-a", _context("tenant-a"))
    journal.reserve("request-b", _context("tenant-b"))
    journal.record_gpu_complete("request-a")
    journal.record_gpu_complete("request-b")
    assert journal.record_finished(
        _finished("chatcmpl-request-a"), observed_unix_s=10.0
    )
    assert journal.record_finished(
        _finished("request-b", reason="abort"), observed_unix_s=11.0
    )

    with pytest.raises(RuntimeUsageCapacityError):
        journal.reserve("request-c", _context("tenant-c"))

    page = journal.read(after_sequence=0, limit=1)
    assert page["high_watermark_sequence"] == 1
    assert page["dropped_through_sequence"] == 0
    first = page["receipts"][0]
    assert first["tenant_id"] == "tenant-a"
    assert first["receipt_id"] == "request-a"
    assert first["source_sequence"] == 1
    assert first["status"] == "succeeded"
    assert first["failure_attribution"] is None
    assert first["measurements"] == (
        {"metric": "live_request_ms", "quantity": "125"},
        {"metric": "inference_gpu_ms", "quantity": "0"},
        {"metric": "cached_prefill_tokens", "quantity": 5},
        {"metric": "uncached_prefill_tokens", "quantity": 7},
        {"metric": "decode_tokens", "quantity": 3},
    )
    now[0] = 30.0
    assert journal.read(after_sequence=0, limit=1) == page

    assert journal.acknowledge(1) == 1
    journal.reserve("request-c", _context("tenant-c"))
    second = journal.read(after_sequence=1)["receipts"][0]
    assert second["source_sequence"] == 2
    assert second["status"] == "cancelled"
    assert second["failure_attribution"] == "unknown"
    with pytest.raises(RuntimeUsageCursorError, match="expired"):
        journal.read(after_sequence=0)


def test_runtime_usage_rounds_live_latency_to_service_precision() -> None:
    journal = RuntimeUsageJournal("paired-runtime", 0)
    journal.reserve("request", _context("tenant"))
    journal.record_gpu_complete("request")
    finished = _finished("request")
    finished.e2e_latency = 4.219876543678901

    assert journal.record_finished(finished, observed_unix_s=10.0)
    receipt = journal.read(after_sequence=0)["receipts"][0]
    quantity = next(
        item["quantity"]
        for item in receipt["measurements"]
        if item["metric"] == "live_request_ms"
    )

    assert quantity == "4219.876544"


def test_runtime_usage_preserves_unknown_terminal_measurements() -> None:
    journal = RuntimeUsageJournal("paired-runtime", 0)
    journal.reserve("request", _context("tenant"))
    journal.record_gpu_complete("request")
    malformed = _finished("request")
    malformed.num_cached_tokens = 13

    assert journal.record_finished(malformed, observed_unix_s=10.0)
    receipt = journal.read(after_sequence=0)["receipts"][0]
    assert receipt["status"] == "failed"
    assert receipt["failure_attribution"] == "unknown"
    assert receipt["coverage"] == "unknown"
    assert receipt["measurements"] == ()


def test_runtime_usage_emits_exact_gpu_and_closed_kv_usage() -> None:
    journal = RuntimeUsageJournal(
        "paired-runtime",
        0,
        clock=lambda: 100.0,
        monotonic_ns=lambda: 1_000,
    )
    journal.reserve("request", _context("tenant"))
    journal.record_gpu_service("request", 3_000_000)
    journal.record_gpu_complete("request")
    assert journal.record_finished(_finished("request"), observed_unix_s=100.125)
    owner = KVUsageOwner("tenant", "run-tenant", "standard", "test-model")
    journal.record_kv_residency(0, owner, byte_count=8, monotonic_ns=1_000)
    journal.record_kv_residency(0, owner, byte_count=0, monotonic_ns=2_000)

    request, residency = journal.read(after_sequence=0)["receipts"]
    assert {item["metric"]: item["quantity"] for item in request["measurements"]}[
        "inference_gpu_ms"
    ] == "3"
    assert residency["producer"] == "residency"
    assert residency["measurements"] == ({"metric": "kv_byte_ms", "quantity": "0.008"},)


def test_runtime_usage_waits_for_late_gpu_service_before_sealing() -> None:
    journal = RuntimeUsageJournal("paired-runtime", 0)
    journal.reserve("request", _context("tenant"))
    journal.record_gpu_service("request", 1_000_000)

    assert journal.record_finished(_finished("request"), observed_unix_s=100.125)
    assert journal.read(after_sequence=0)["receipts"] == ()

    journal.record_gpu_service("request", 2_000_000)
    journal.record_gpu_complete("request")
    receipt = journal.read(after_sequence=0)["receipts"][0]
    assert {item["metric"]: item["quantity"] for item in receipt["measurements"]}[
        "inference_gpu_ms"
    ] == "3"


def test_empty_page_observation_advances_only_after_activity_drains() -> None:
    wall = iter((100.0, 110.0, 120.0, 130.0))
    journal = RuntimeUsageJournal(
        "paired-runtime",
        0,
        clock=lambda: next(wall),
        monotonic_ns=lambda: 1_000,
    )

    first = journal.read(after_sequence=0)["observed_at"]
    journal.reserve("request", _context("tenant"))
    assert journal.read(after_sequence=0)["observed_at"] == first
    journal.discard("request")
    drained = journal.read(after_sequence=0)["observed_at"]
    assert drained != first
    owner = KVUsageOwner("tenant", "run-tenant", "standard", "test-model")
    journal.record_kv_residency(0, owner, byte_count=8, monotonic_ns=1_000)
    assert journal.read(after_sequence=0)["observed_at"] == drained
    journal.record_kv_residency(0, owner, byte_count=0, monotonic_ns=2_000)
    journal.acknowledge(1)
    assert journal.read(after_sequence=1)["observed_at"] != drained
