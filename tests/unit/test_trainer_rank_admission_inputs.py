"""Seven additive estimator joins; scalar CPU plans, not distributed/CUDA bounds."""

from dataclasses import replace
from types import SimpleNamespace

from test_trainer_rank_head_memory import rank as head_rank
from test_trainer_rank_head_memory import request as head_request
from test_trainer_rank_pending_memory import layer, pending_rank
from test_trainer_rank_recompute_memory import _hybrid_rank
import torch

from art.trainer_rank import ForwardInput, _gdn_memory
from art.trainer_rank._impl import Unset


def record_prices(monkeypatch, rank):
    original = rank._estimate_required_memory_bytes_from_values
    calls = []

    def observed(**values):
        calls.append(dict(values))
        return original(**values)

    monkeypatch.setattr(rank, "_estimate_required_memory_bytes_from_values", observed)
    return calls


def assert_plan_values(rank, plan, values):
    assert values["packed_tokens"] == plan.packed_tokens
    assert values["output_bytes"] == plan.output_bytes
    assert values["logical_tokens"] == plan.active_logical_tokens
    assert values["signature"] == plan.signature
    assert values["gdn_segments"] == plan.grad_segment_count
    assert values["group_rows"] == rank._plan_group_rows(plan)
    assert values["head_workspace_bytes"] == rank._plan_head_workspace_bytes(plan)
    assert values["checkpoint_floor"] == _gdn_memory.plan_floor(rank, plan)
    assert values["retained_tokens"] == rank._plan_retained_tokens(plan)


def test_shared_head_lower_and_plan_keep_all_keywords(monkeypatch):
    rank = head_rank()
    a = replace(head_request(2), target_tokens=torch.tensor([1, -100]))
    b = replace(a, target_tokens=torch.tensor([-100, 1]))
    requests = [a, a, b, b]
    plan = rank._plan_flat_forward(requests, memory_minimal=True)
    assert plan.packed_tokens == 2 and plan.active_logical_tokens == 8
    assert rank._plan_group_rows(plan) == ((2, False),)
    assert rank._checkpoint_memory_floor(((2, False),)) == (
        0,
        2 * (188416 + 4 * 2048 * 2),
    )
    assert rank._plan_head_workspace_bytes(plan) == 2 * 248320 * 2
    calls = record_prices(monkeypatch, rank)
    lower = rank._split_chunk_lower_cost(
        requests, tuple(item.input_tokens for item in requests), checkpoint=Unset
    )
    assert len(calls) == 1
    assert calls[0]["group_rows"] == ((2, False),)
    assert calls[0]["retained_tokens"] == 2
    assert calls[0]["head_workspace_bytes"] == 248320 * 2
    assert calls[0]["checkpoint_floor"] == (0, 0)
    calls.clear()
    cost = rank._plan_cost(plan)
    check = rank._memory_check(plan)
    assert cost.required == check.estimated_required_bytes
    assert lower.required <= cost.required
    assert len(calls) == 2
    for values in calls:
        assert_plan_values(rank, plan, values)
    # All shape/floor inputs remain available together, even though the dense
    # head dominates this two-row source floor and sharing changes logical rows.
    assert cost.required == int((plan.output_bytes + 2 * 248320 * 2) * 1.1)


def test_cp_gdn_segments_groups_and_retained_tokens_reach_exact_search(monkeypatch):
    rank = _hybrid_rank(monkeypatch, 2)
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 2, 4, 1))
    monkeypatch.setattr(rank, "_topology", lambda: SimpleNamespace(cp=4, tp=2))
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 1 << 60)
    monkeypatch.setattr(
        rank, "_max_rank_model_tokens", lambda batch, **_: batch.tokens.numel() * 3 // 4
    )
    requests = [
        ForwardInput(input_tokens=torch.arange(64) + offset, hidden_states=True)
        for offset in (0, 100)
    ] + [
        ForwardInput(
            input_tokens=torch.arange(80) + 200, hidden_states=True, no_grad=True
        ),
        ForwardInput(input_tokens=torch.arange(1000)),
    ]
    assert _gdn_memory.model_shapes(rank) is None  # This exercises the CP fallback.
    plan = rank._plan_flat_forward(requests)
    assert plan.grad_segment_count == 2
    assert rank._plan_group_rows(plan) == ((128, True), (80, False))
    assert rank._plan_retained_tokens(plan) == 156
    for exact in (False, True):
        for memory_minimal in (False, True):
            assert (
                rank._estimate_flat_forward(
                    requests, exact=exact, memory_minimal=memory_minimal
                )
                is None
            )
    calls = record_prices(monkeypatch, rank)
    lower = rank._split_chunk_lower_cost(
        requests, tuple(item.input_tokens for item in requests), checkpoint=Unset
    )
    assert len(calls) == 1
    assert calls[0]["group_rows"] == ((128, True), (80, False))
    assert calls[0]["retained_tokens"] == 52  # Optimistic CP average only here.
    calls.clear()
    cost = rank._plan_cost(plan)
    check = rank._memory_check(plan)
    assert cost.required == check.estimated_required_bytes
    assert lower.required < cost.required
    assert len(calls) == 2
    for values in calls:
        assert_plan_values(rank, plan, values)
    calls.clear()
    # The outer sequence contains one multi-request wave. A flat list would
    # instead let width search select separate top-level requests.
    selected = rank._search_next_micro_batch([requests], 0)
    assert selected.check.fits and selected.plan.grad_segment_count == 2
    assert selected.check.estimated_required_bytes == cost.required
    assert calls
    for values in calls:
        assert_plan_values(rank, selected.plan, values)


def test_full_gdn_nonzero_floor_survives_separate_exact_fallback(
    monkeypatch, pending_rank
):
    rank = pending_rank
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 1 << 60)
    requests = [
        ForwardInput(input_tokens=torch.arange(64) + offset, hidden_states=True)
        for offset in (0, 100)
    ]
    assert rank._topology_key()[2] == 1
    assert _gdn_memory.model_shapes(rank) is not None
    plan = rank._plan_flat_forward(requests)
    assert plan.grad_segment_count == 2
    floor = _gdn_memory.plan_floor(rank, plan)
    assert floor[0] > 0 and floor[1] > 0
    for exact in (False, True):
        for memory_minimal in (False, True):
            assert (
                rank._estimate_flat_forward(
                    requests, exact=exact, memory_minimal=memory_minimal
                )
                is None
            )
    calls = record_prices(monkeypatch, rank)
    cost = rank._plan_cost(plan)
    check = rank._memory_check(plan)
    assert cost.required == check.estimated_required_bytes
    assert len(calls) == 2
    for values in calls:
        assert_plan_values(rank, plan, values)
        assert values["checkpoint_floor"] == floor
    calls.clear()
    selected = rank._search_next_micro_batch([requests], 0)
    assert selected.check.fits
    assert selected.check.estimated_required_bytes == cost.required
    assert calls
    for values in calls:
        assert_plan_values(rank, selected.plan, values)
        assert values["checkpoint_floor"] == floor
