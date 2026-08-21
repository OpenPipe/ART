from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import statistics
import time
from typing import Any

import pytest
import torch

from art.trainer_rank import _impl


def _plan(*, packed_tokens: int = 4, output_bytes: int = 100) -> _impl._FlatForwardPlan:
    return _impl._FlatForwardPlan(
        request_count=1,
        groups=(),
        packed_tokens=packed_tokens,
        logical_tokens=packed_tokens,
        output_bytes=output_bytes,
        signature=_impl._MemorySignature(
            topology=(1, 1, 1, 1),
            shared_prefix_max_depth=1,
            slot_group_count=1,
            request_mix=("target_logprobs",),
            grad_enabled=True,
        ),
    )


def _trainer(
    execute: Callable[[_impl._FlatForwardPlan], list[Any]],
) -> _impl.TrainerRank:
    trainer = object.__new__(_impl.TrainerRank)
    trainer.device = torch.device("cuda")
    trainer._memory_profiles = {}
    trainer._execute_flat_plan = execute  # type: ignore[method-assign]
    return trainer


@contextmanager
def _phase(_name: str, _signature: object, **kwargs: object) -> Iterator[None]:
    assert kwargs.get("synchronized", False) is False
    yield


def _mock_cuda_allocator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline: int = 1_000,
    peak: int = 1_700,
) -> list[str]:
    calls: list[str] = []

    def memory_allocated(device: torch.device) -> int:
        assert device.type == "cuda"
        calls.append("baseline")
        return baseline

    def reset_peak(device: torch.device) -> None:
        assert device.type == "cuda"
        calls.append("reset_peak")

    def max_memory_allocated(device: torch.device) -> int:
        assert device.type == "cuda"
        calls.append("peak")
        return peak

    def synchronize(*_args: object, **_kwargs: object) -> None:
        calls.append("synchronize")
        raise AssertionError("memory tracking synchronized CUDA")

    monkeypatch.setattr(_impl, "_telemetry_phase", _phase)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", memory_allocated)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", max_memory_allocated)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return calls


def test_allocator_peak_updates_profile_without_synchronizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_cuda_allocator(monkeypatch)

    def execute(_plan: _impl._FlatForwardPlan) -> list[str]:
        calls.append("execute")
        return ["output"]

    trainer = _trainer(execute)
    plan = _plan()

    assert trainer._run_flat_plan_with_memory_tracking(plan, context="probe") == [
        "output"
    ]
    assert calls == ["baseline", "reset_peak", "execute", "peak"]
    assert trainer._memory_profiles[plan.signature] == _impl._MemoryProfile(
        bytes_per_token=150.0,
        packed_tokens=4,
    )


def test_allocator_oom_keeps_context_and_does_not_publish_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_cuda_allocator(monkeypatch)

    def execute(_plan: _impl._FlatForwardPlan) -> list[Any]:
        calls.append("execute")
        raise torch.cuda.OutOfMemoryError("mock allocator OOM")

    trainer = _trainer(execute)
    trainer._memory_check = lambda _plan: _impl._MemoryCheck(  # type: ignore[method-assign]
        estimated_required_bytes=2_048,
        available_bytes=1_024,
        fits=False,
    )
    plan = _plan()

    with pytest.raises(
        _impl.TrainerRankMemoryError,
        match="probe_oom: CUDA OOM occurred despite the planner estimate",
    ) as exc_info:
        trainer._run_flat_plan_with_memory_tracking(plan, context="probe_oom")
    assert calls == ["baseline", "reset_peak", "execute"]
    assert plan.signature not in trainer._memory_profiles
    assert isinstance(exc_info.value.__context__, torch.cuda.OutOfMemoryError)


def test_repeated_forwards_only_queue_modeled_cuda_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_cuda_allocator(monkeypatch, baseline=1_000, peak=1_000)
    modeled_seconds = 0.01
    queued = 0

    def execute(_plan: _impl._FlatForwardPlan) -> list[Any]:
        nonlocal queued
        queued += 1
        calls.append("execute")
        return []

    trainer = _trainer(execute)
    plan = _plan(output_bytes=0)
    iterations = 32
    started = time.perf_counter()
    for _ in range(iterations):
        trainer._run_flat_plan_with_memory_tracking(plan, context="probe_queue")
    submission_seconds = time.perf_counter() - started

    assert queued == iterations
    assert "synchronize" not in calls
    assert submission_seconds < iterations * modeled_seconds
    print(
        "mock_launch "
        f"forwards={iterations} "
        f"host_ms={submission_seconds * 1_000:.3f} "
        f"per_forward_us={submission_seconds * 1_000_000 / iterations:.3f} "
        f"modeled_device_ms={iterations * modeled_seconds * 1_000:.1f}"
    )


def test_cuda_launch_side_latency_under_4_gib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; exact mock launch evidence ran instead")

    monkeypatch.setattr(_impl, "_telemetry_phase", _phase)
    device = torch.device("cuda")
    trainer = _trainer(lambda _plan: [])
    trainer.device = device
    plan = _plan(packed_tokens=1, output_bytes=0)
    sleep_cycles = 20_000_000
    iterations = 16

    def execute(_plan: _impl._FlatForwardPlan) -> list[Any]:
        torch.cuda._sleep(sleep_cycles)
        return []

    trainer._execute_flat_plan = execute  # type: ignore[method-assign]
    torch.cuda.synchronize(device)
    latencies = []
    started = time.perf_counter()
    for _ in range(iterations):
        launched = time.perf_counter()
        trainer._run_flat_plan_with_memory_tracking(plan, context="cuda_probe")
        latencies.append(time.perf_counter() - launched)
    submission_seconds = time.perf_counter() - started
    torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - started
    peak_reserved = torch.cuda.max_memory_reserved(device)

    assert submission_seconds < total_seconds / 2
    assert peak_reserved <= 4 * 1024**3
    print(
        "cuda_launch "
        f"forwards={iterations} "
        f"median_us={statistics.median(latencies) * 1_000_000:.3f} "
        f"p95_us={sorted(latencies)[int(iterations * 0.95) - 1] * 1_000_000:.3f} "
        f"submission_ms={submission_seconds * 1_000:.3f} "
        f"drain_ms={total_seconds * 1_000:.3f} "
        f"peak_reserved_gib={peak_reserved / 1024**3:.3f}"
    )
