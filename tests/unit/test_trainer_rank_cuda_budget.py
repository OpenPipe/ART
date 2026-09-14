"""Native allocator budget contracts; CPU counters, not CUDA qualification."""

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
