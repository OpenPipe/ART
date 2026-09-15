"""Native allocator budget contracts; CPU counters, not CUDA qualification."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from art.trainer_rank import TrainerRank, _impl


@pytest.fixture
def budget(monkeypatch):
    rank = object.__new__(TrainerRank)
    rank.device = torch.device("cuda")
    stats = {
        "allocated_bytes.all.current": 10,
        "active_bytes.all.current": 30,
        "reserved_bytes.all.current": 80,
        "inactive_split_bytes.all.current": 20,
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (100, 1000))
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "memory_stats", Mock(side_effect=lambda _: stats))
    monkeypatch.setattr(torch.cuda, "memory_allocated", Mock(return_value=10))
    monkeypatch.setattr(torch.cuda, "memory_reserved", Mock(return_value=80))
    monkeypatch.setattr(_impl.dist, "is_available", lambda: False)
    for name in (
        "empty_cache",
        "synchronize",
        "memory_snapshot",
        "reset_peak_memory_stats",
    ):
        monkeypatch.setattr(torch.cuda, name, Mock(side_effect=AssertionError(name)))
    monkeypatch.delenv("ART_TRAINER_RANK_TEST_HOOKS", raising=False)
    monkeypatch.delenv("ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES", raising=False)
    return rank, stats


@pytest.mark.parametrize("active,available", [(10, 140), (30, 120), (80, 70)])
def test_native_pending_is_not_reusable_credit(budget, active, available):
    rank, stats = budget
    stats["active_bytes.all.current"] = active
    assert rank._available_memory_bytes() == available
    torch.cuda.memory_stats.assert_called_once_with(rank.device)
    torch.cuda.memory_allocated.assert_not_called()
    torch.cuda.memory_reserved.assert_not_called()


def test_pending_credit_changes_admission_before_execution(budget):
    rank, stats = budget
    check = rank._memory_check_required(130)
    assert check.available_bytes == 120
    assert not check.fits
    # Normal allocator collection can later make the same bytes inactive.
    # The budget itself neither polls events nor forces collection.
    stats["active_bytes.all.current"] = 10
    assert rank._memory_check_required(130).fits


def test_split_and_private_credit_remain_explicit_residuals(budget):
    rank, stats = budget
    stats["active_bytes.all.current"] = 10
    stats["inactive_split_bytes.all.current"] = 70
    assert rank._available_memory_bytes() == 140
    stats["inactive_split_bytes.all.current"] = 0
    # A whole inactive retained private pool has this same scalar geometry.
    # This partial correction does not establish pool compatibility.
    assert rank._available_memory_bytes() == 140


@pytest.mark.parametrize(
    "field,value",
    [
        ("allocated", None),
        ("active", None),
        ("reserved", None),
        ("allocated", True),
        ("active", True),
        ("reserved", 80.0),
        ("allocated", -1),
        ("active", 9),
        ("active", 81),
        ("reserved", -1),
    ],
)
def test_incomplete_native_counters_grant_no_cache_credit(budget, field, value):
    rank, stats = budget
    key = field + "_bytes.all.current"
    if value is None:
        stats.pop(key)
    else:
        stats[key] = value
    assert rank._available_memory_bytes() == 70
    torch.cuda.memory_allocated.assert_called_once_with(rank.device)


@pytest.mark.parametrize("backend", ["cudaMallocAsync", "unrecognized"])
def test_other_backends_retain_legacy_unqualified_credit(budget, monkeypatch, backend):
    rank, _ = budget
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: backend)
    assert rank._available_memory_bytes() == 140
    torch.cuda.memory_stats.assert_not_called()
    torch.cuda.memory_allocated.assert_called_once_with(rank.device)
    torch.cuda.memory_reserved.assert_called_once_with(rank.device)


@pytest.mark.parametrize("missing", [False, True])
def test_existing_test_limit_stays_relative_to_allocated(budget, monkeypatch, missing):
    rank, stats = budget
    if missing:
        stats.pop("active_bytes.all.current")
    monkeypatch.setenv("ART_TRAINER_RANK_TEST_HOOKS", "1")
    monkeypatch.setenv(_impl._TEST_MEMORY_LIMIT_ENV, "50")
    assert rank._available_memory_bytes() == 40
    monkeypatch.setenv(_impl._TEST_MEMORY_LIMIT_ENV, "5")
    assert rank._available_memory_bytes() == 0
    monkeypatch.setenv("ART_TRAINER_RANK_TEST_HOOKS", "0")
    assert rank._available_memory_bytes() == (70 if missing else 120)


@pytest.mark.parametrize(
    "api", ["mem_get_info", "get_allocator_backend", "memory_stats"]
)
def test_actual_api_error_identity_is_not_swallowed(budget, monkeypatch, api):
    rank, _ = budget
    error = RuntimeError("native API failed")
    monkeypatch.setattr(torch.cuda, api, Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        rank._available_memory_bytes()
    assert caught.value is error


def test_missing_counter_fallback_api_error_is_preserved(budget, monkeypatch):
    rank, stats = budget
    stats.pop("active_bytes.all.current")
    error = RuntimeError("allocated read failed")
    monkeypatch.setattr(torch.cuda, "memory_allocated", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        rank._available_memory_bytes()
    assert caught.value is error


def test_cpu_budget_avoids_all_new_cuda_api_calls(budget, monkeypatch):
    rank, _ = budget
    rank.device = torch.device("cpu")
    monkeypatch.setattr(
        torch.cuda, "get_allocator_backend", Mock(side_effect=AssertionError)
    )
    assert rank._available_memory_bytes() == 1 << 60
    torch.cuda.memory_stats.assert_not_called()


def test_required_max_then_available_min_collectives_unchanged(budget, monkeypatch):
    rank, _ = budget
    monkeypatch.setattr(_impl.dist, "is_available", lambda: True)
    monkeypatch.setattr(_impl.dist, "is_initialized", lambda: True)
    group = object()
    monkeypatch.setattr(rank, "_forward_memory_group", lambda: group)
    tensor = torch.tensor
    monkeypatch.setattr(
        torch, "tensor", lambda values, **kwargs: tensor(values, dtype=kwargs["dtype"])
    )
    calls = []

    def reduce(value, op, group):
        calls.append((float(value.item()), op, group))
        value.fill_(150 if op == _impl.dist.ReduceOp.MAX else 110)

    monkeypatch.setattr(_impl.dist, "all_reduce", reduce)
    check = rank._memory_check_required(130)
    assert calls == [
        (130, _impl.dist.ReduceOp.MAX, group),
        (120, _impl.dist.ReduceOp.MIN, group),
    ]
    assert (check.estimated_required_bytes, check.available_bytes, check.fits) == (
        150,
        110,
        False,
    )


@pytest.mark.parametrize("failed_locally", [False, True])
@pytest.mark.parametrize(
    "error_type", [RuntimeError, KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_admission_failure_sentinel_stops_healthy_peers(
    budget, monkeypatch, failed_locally, error_type
):
    rank, stats = budget
    stats["active_bytes.all.current"] = 80
    monkeypatch.setattr(_impl.dist, "is_available", lambda: True)
    monkeypatch.setattr(_impl.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rank, "_forward_memory_group", lambda: None)
    tensor = torch.tensor
    monkeypatch.setattr(
        torch, "tensor", lambda values, **kwargs: tensor(values, dtype=kwargs["dtype"])
    )
    original = error_type("recoverable local budget read failure")
    events = []

    def read(_):
        events.append("sample_failure")
        raise original

    monkeypatch.setattr(torch.cuda, "mem_get_info", read)
    if not failed_locally:
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (200, 1000))

    def reduce(value, op, group):
        events.append((op, float(value.item())))
        value.fill_(150 if op == _impl.dist.ReduceOp.MAX else -1)

    monkeypatch.setattr(_impl.dist, "all_reduce", reduce)
    with pytest.raises(error_type if failed_locally else RuntimeError) as caught:
        for required in (0, 75):
            rank._memory_check_required(required)
    assert (
        caught.value is original
        if failed_locally
        else str(caught.value) == "Memory admission failed on another rank"
    )
    assert events == [
        (_impl.dist.ReduceOp.MAX, 0),
        *(["sample_failure"] if failed_locally else []),
        (_impl.dist.ReduceOp.MIN, -1 if failed_locally else 170),
    ]


@pytest.mark.parametrize("stage", ["assignment", "min", "result"])
@pytest.mark.parametrize("failed_locally", [False, True])
def test_secondary_collective_failure_preserves_local_primary(
    budget, monkeypatch, stage, failed_locally
):
    rank, stats = budget
    stats["active_bytes.all.current"] = 80
    monkeypatch.setattr(_impl.dist, "is_available", lambda: True)
    monkeypatch.setattr(_impl.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rank, "_forward_memory_group", lambda: None)
    original = RuntimeError("original local budget read error")
    secondary = RuntimeError("secondary CUDA or communicator failure")

    class Values:
        def __init__(self, values, **kwargs):
            self.values = values

        def __getitem__(self, index):
            def item():
                if stage == "result" and index == 1:
                    raise secondary
                return self.values[index]

            return SimpleNamespace(
                item=item, fill_=lambda x: self.values.__setitem__(index, x)
            )

        def __setitem__(self, index, value):
            if stage == "assignment":
                raise secondary
            self.values[index] = value

    monkeypatch.setattr(torch, "tensor", Values)
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(side_effect=original))
    if not failed_locally:
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (200, 1000))

    def reduce(value, op, group):
        if op == _impl.dist.ReduceOp.MAX:
            value.fill_(150)
        elif stage == "min":
            raise secondary

    monkeypatch.setattr(_impl.dist, "all_reduce", reduce)
    with pytest.raises(RuntimeError) as caught:
        rank._memory_check_required(0)
    assert caught.value is (original if failed_locally else secondary)


@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("available", [10, 210])
def test_final_selection_uses_pure_fresh_budget_and_original_demand(
    budget, monkeypatch, distributed, available
):
    rank, stats = budget
    stats["active_bytes.all.current"] = 80
    plan = SimpleNamespace(packed_tokens=64, logical_tokens=64)
    stale = _impl._MemoryCheck(
        estimated_required_bytes=192, available_bytes=256, fits=True
    )
    candidate = _impl._CandidateMicroBatch([], (), plan, stale, 64, 1, False)
    monkeypatch.setattr(rank, "_search_next_micro_batch", lambda *a, **k: candidate)
    monkeypatch.setattr(
        rank,
        "_estimate_required_memory_bytes_from_values",
        Mock(side_effect=AssertionError("must keep original selected demand")),
    )
    monkeypatch.setattr(rank, "_snapshot_planning_telemetry", Mock())
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", Mock(return_value=(available + 30, 1000))
    )
    monkeypatch.setattr(_impl.dist, "is_available", lambda: distributed)
    monkeypatch.setattr(_impl.dist, "is_initialized", lambda: distributed)
    tensor = torch.tensor
    monkeypatch.setattr(
        torch, "tensor", lambda values, **kwargs: tensor(values, dtype=kwargs["dtype"])
    )
    calls = []
    monkeypatch.setattr(
        _impl.dist,
        "all_reduce",
        lambda value, op, group: calls.append((op, value.item(), group)),
    )
    if available < 192:
        with pytest.raises(_impl.TrainerRankMemoryError) as caught:
            rank._select_next_micro_batch([], 0)
        assert caught.value.predicted_peak_bytes == 192
        assert caught.value.usable_limit_bytes == available
    else:
        selected = rank._select_next_micro_batch([], 0)
        assert selected.check.estimated_required_bytes == 192
        assert selected.check.available_bytes == available and selected.check.fits
        assert selected.plan is plan
    assert stale.available_bytes == 256
    assert calls == (
        [
            (_impl.dist.ReduceOp.MAX, 192, None),
            (_impl.dist.ReduceOp.MIN, available, None),
        ]
        if distributed
        else []
    )
    torch.cuda.empty_cache.assert_not_called()
    torch.cuda.mem_get_info.assert_called_once_with(rank.device)


def test_available_sample_follows_required_collective(budget, monkeypatch):
    rank, stats = budget
    stats["active_bytes.all.current"] = 80
    free = [100]
    events = []

    def read(_):
        events.append("sample")
        return free[0], 1000

    monkeypatch.setattr(torch.cuda, "mem_get_info", read)
    monkeypatch.setattr(_impl.dist, "is_available", lambda: True)
    monkeypatch.setattr(_impl.dist, "is_initialized", lambda: True)
    tensor = torch.tensor
    monkeypatch.setattr(
        torch, "tensor", lambda values, **kwargs: tensor(values, dtype=kwargs["dtype"])
    )

    def reduce(value, op, group):
        events.append(op)
        if op == _impl.dist.ReduceOp.MAX:
            value.fill_(60)
            free[0] = 50

    monkeypatch.setattr(_impl.dist, "all_reduce", reduce)
    check = rank._memory_check_required(0, sync_across_dp=True)
    assert (check.estimated_required_bytes, check.available_bytes, check.fits) == (
        60,
        20,
        False,
    )
    assert events == [_impl.dist.ReduceOp.MAX, "sample", _impl.dist.ReduceOp.MIN]
    torch.cuda.empty_cache.assert_not_called()


def test_execution_failure_retains_final_admission_without_resampling(
    budget, monkeypatch
):
    rank, stats = budget
    stats["active_bytes.all.current"] = 80
    free = [100]
    read = Mock(side_effect=lambda _: (free[0], 1000))
    monkeypatch.setattr(torch.cuda, "mem_get_info", read)
    admitted = rank._memory_check_required(50)
    assert admitted.fits and admitted.available_bytes == 70
    original = torch.cuda.OutOfMemoryError("synthetic later execution failure")

    def execute(_):
        free[0] = 10
        raise original

    monkeypatch.setattr(rank, "_execute_flat_plan", execute)
    monkeypatch.setattr(rank, "_telemetry_signature", lambda _: {})
    monkeypatch.setattr(rank, "_telemetry_plan_signature", lambda _: {})
    monkeypatch.setattr(_impl, "_telemetry_phase", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", Mock())
    with pytest.raises(_impl.TrainerRankMemoryError) as caught:
        rank._run_flat_plan_with_memory_tracking(
            SimpleNamespace(packed_tokens=1, logical_tokens=1),
            check=admitted,
            context="CPU final-admission witness",
        )
    assert caught.value.__cause__ is original
    assert caught.value.usable_limit_bytes == admitted.available_bytes == 70
    read.assert_called_once_with(rank.device)
    assert rank._available_memory_bytes() == 0
    torch.cuda.empty_cache.assert_not_called()
