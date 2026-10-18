"""Real CPU slot/model guards, with no model forward or CUDA initialization."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
from threading import get_ident
from types import MethodType

import pytest
from test_trainer_rank_converted_memory import weights
from test_trainer_rank_moe_memory import layer as layer
from test_trainer_rank_pending_memory import rank_with_moe
from test_trainer_rank_slot_memory import load_slot, request
import torch

from art.trainer_rank import _impl, _memory


def selected_rank(layer):
    rank, _ = rank_with_moe(weights(layer, 1))
    load_slot(rank, "selected", 1)
    plan = rank._plan_flat_forward(
        [request("selected", rows=65, grad=True)], ensure_slots=False
    )
    return rank, plan


@pytest.mark.parametrize("method", ["_plan_cost", "_memory_check"])
def test_each_evaluation_reuses_only_same_slot_and_mode(layer, monkeypatch, method):
    rank, plan = selected_rank(layer)
    scan = _impl._moe_output_bytes_per_token
    calls = []

    def counted(*args, **kwargs):
        calls.append((kwargs["slot_ref"], kwargs["checkpoint_grad"]))
        return scan(*args, **kwargs)

    monkeypatch.setattr(_impl, "_moe_output_bytes_per_token", counted)
    evaluate = getattr(rank, method)
    first = evaluate(plan)
    assert len(calls) == 2
    assert {grad for _, grad in calls} == {False, True}
    assert _memory._moe_evaluation.get() is None
    assert evaluate(plan) == first
    assert len(calls) == 4  # No persistent constructor/slot cache.
    load_slot(rank, "selected", 64)
    changed = evaluate(plan)
    required = lambda value: (
        getattr(value, "required", None) or value.estimated_required_bytes
    )
    assert required(changed) > required(first)
    assert len(calls) == 6
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kind", ["hook", "dispatcher", "dtype", "storage"])
def test_next_evaluation_rechecks_live_owner_and_tensor_guards(
    layer, monkeypatch, kind
):
    rank, plan = selected_rank(layer)
    first = rank._plan_cost(plan)
    original_workspace = rank._moe_workspace_bytes(
        65, checkpoint_grad=True, slot_ref=plan.groups[0].slot_ref
    )
    slot = layer.experts.linear_fc2.lora._slot(plan.groups[0].slot_ref)
    if kind == "hook":
        layer.register_forward_pre_hook(lambda *args: None)
    elif kind == "dispatcher":
        layer.token_dispatcher.dispatch_preprocess = lambda *args: None
    elif kind == "dtype":
        slot.A_T = torch.nn.Parameter(slot.A_T.float())
    else:
        slot.B_T = torch.nn.Parameter(slot.B_T[..., :1].expand_as(slot.B_T))
    fresh = rank._plan_cost(plan)
    # Compare the entire result with explicitly uncached row-independent pricing.
    with monkeypatch.context() as patch:
        for name in ("_subforward_cost", "_estimate_required_memory_bytes_from_values"):
            patch.setattr(rank, name, MethodType(getattr(rank, name).__wrapped__, rank))
        uncached = rank._plan_cost.__wrapped__(rank, plan)
    assert fresh == uncached
    assert _memory._moe_evaluation.get() is None
    assert (
        rank._moe_workspace_bytes(
            65, checkpoint_grad=True, slot_ref=plan.groups[0].slot_ref
        )
        != original_workspace
    )
    assert first.checkpoint_workspace > 0
    assert not torch.cuda.is_initialized()


def test_fresh_available_memory_and_distinct_slots(layer, monkeypatch):
    rank, _ = selected_rank(layer)
    load_slot(rank, "other", 64)
    plan = rank._plan_flat_forward(
        [request("selected", rows=65, grad=True), request("other", rows=17, grad=True)],
        ensure_slots=False,
    )
    calls = []
    scan = _impl._moe_output_bytes_per_token

    def counted(*args, **kwargs):
        calls.append((kwargs["slot_ref"].name, kwargs["checkpoint_grad"]))
        return scan(*args, **kwargs)

    monkeypatch.setattr(_impl, "_moe_output_bytes_per_token", counted)
    check = rank._memory_check(plan)
    assert len(calls) == 4 and len(set(calls)) == 4
    samples = iter([check.estimated_required_bytes, check.estimated_required_bytes - 1])
    rank._available_memory_bytes = lambda: next(samples)
    assert rank._memory_check(plan).fits
    assert not rank._memory_check(plan).fits
    assert len(calls) == 12


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failed_evaluation_drops_all_terms(layer, monkeypatch, error):
    rank, plan = selected_rank(layer)
    original = rank._estimate_required_memory_bytes_from_values

    def fail(**kwargs):
        state = _memory._moe_evaluation.get()
        assert state is not None and state.terms  # Earlier GDN floor was evaluated.
        raise error("inert failure after pricing")

    monkeypatch.setattr(rank, "_estimate_required_memory_bytes_from_values", fail)
    with pytest.raises(error, match="inert failure"):
        rank._memory_check(plan)
    assert _memory._moe_evaluation.get() is None
    monkeypatch.setattr(rank, "_estimate_required_memory_bytes_from_values", original)
    load_slot(rank, "selected", 64)
    assert rank._memory_check(plan).fits


def test_replay_and_instance_workspace_override_remain_authoritative(layer):
    rank, plan = selected_rank(layer)
    calls = []

    def price(rows, *, checkpoint_grad=False, slot_ref=None):
        calls.append((rows, checkpoint_grad, slot_ref))
        return rows * (300000 if checkpoint_grad else 200000)

    rank._moe_workspace_bytes = price
    first = rank._plan_cost(plan)
    assert len(calls) == 3  # The override is not memoized or bypassed.
    assert rank._plan_cost(replace(plan)) == first
    assert len(calls) == 6


def test_copied_context_cannot_reuse_finished_evaluation(layer):
    rank, plan = selected_rank(layer)
    contexts = []

    def available():
        contexts.append(copy_context())
        return 1 << 60

    rank._available_memory_bytes = available
    before = rank._memory_check(plan)
    assert contexts[0].get(_memory._moe_evaluation).terms == {}
    assert contexts[0].get(_memory._moe_evaluation).rank is None
    assert not contexts[0].get(_memory._moe_evaluation).active
    load_slot(rank, "selected", 64)
    after = contexts[0].run(rank._memory_check, plan)
    assert after.estimated_required_bytes > before.estimated_required_bytes


def test_copied_context_on_another_thread_gets_its_own_live_evaluation(
    layer, monkeypatch
):
    rank, plan = selected_rank(layer)
    owner = get_ident()
    calls = []
    scan = _impl._moe_output_bytes_per_token

    def counted(*args, **kwargs):
        calls.append(get_ident())
        return scan(*args, **kwargs)

    monkeypatch.setattr(_impl, "_moe_output_bytes_per_token", counted)

    def available():
        if get_ident() == owner:
            context = copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                check = pool.submit(context.run, rank._memory_check, plan).result()
                assert isinstance(check, _impl._MemoryCheck) and check.fits
        return 1 << 60

    rank._available_memory_bytes = available
    assert rank._memory_check(plan).fits
    assert len(calls) == 4 and calls.count(owner) == 2
    assert _memory._moe_evaluation.get() is None
