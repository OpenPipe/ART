from types import SimpleNamespace

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
    assert journal.record_finished(_finished("request-a"), observed_unix_s=10.0)
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
    assert first["source_sequence"] == 1
    assert first["status"] == "succeeded"
    assert first["failure_attribution"] is None
    assert first["measurements"] == (
        {"metric": "live_request_ms", "quantity": "125.000"},
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


def test_runtime_usage_preserves_unknown_terminal_measurements() -> None:
    journal = RuntimeUsageJournal("paired-runtime", 0)
    journal.reserve("request", _context("tenant"))
    malformed = _finished("request")
    malformed.num_cached_tokens = 13

    assert journal.record_finished(malformed, observed_unix_s=10.0)
    receipt = journal.read(after_sequence=0)["receipts"][0]
    assert receipt["status"] == "failed"
    assert receipt["failure_attribution"] == "unknown"
    assert receipt["coverage"] == "unknown"
    assert receipt["measurements"] == ()
