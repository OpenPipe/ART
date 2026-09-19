"""Partial input-gradient extents: CPU admission math, not peak/overlap proof."""

from dataclasses import replace

import pytest
from test_trainer_rank_checkpoint_memory import price, rank, requests
from test_trainer_rank_moe_memory import layer  # noqa: F401
from test_trainer_rank_pending_memory import full_requests, pending_rank  # noqa: F401
import torch

from art.trainer_rank import ForwardInput
from art.trainer_rank._impl import Unset, _MemoryProfile, _SplitForwardPlan


def profile(r, plan, *, rate=1, retained_rate=1):
    r._memory_profiles[plan.signature] = _MemoryProfile(
        bytes_per_token=rate,
        packed_tokens=plan.packed_tokens,
        logical_per_packed=1,
        retained_compute_bytes_per_token=retained_rate,
    )


def test_pending_cold_peak_does_not_become_forward_retention(pending_rank):
    r = pending_rank
    plan = r._plan_flat_forward(full_requests())
    cost = r._plan_cost(plan)
    gradient = 8 * 6330 * 40 * 2048 * 2
    assert cost.checkpoint_input_gradient == gradient
    # Exact previous cold estimate, including outputs and its one safety factor.
    assert cost.retained == 23102959299
    assert cost.required == int(
        (plan.output_bytes + 2 * gradient + 12705630112) * 1.1
    )
    assert r._memory_check(plan).estimated_required_bytes == cost.required
    profile(r, plan)
    warm = r._plan_cost(plan)
    assert warm.retained == int((plan.output_bytes + gradient) * 1.1)
    assert warm.required == cost.required


@pytest.mark.parametrize("rows", [1, 67, 1024])
def test_attention_only_extent_scales_with_gradient_rows(rows):
    r = rank()
    values = r._estimate_flat_forward(requests(rows, 4096))
    cost = price(r, values)
    assert r._gdn_layers == 0
    assert cost.checkpoint_input_gradient == rows * 40 * 2048 * 2
    assert cost.required >= int(
        (
            cost.checkpoint_retained
            + cost.checkpoint_workspace
            + cost.checkpoint_input_gradient
        )
        * 1.1
    )


def test_gradient_is_not_absorbed_by_larger_head_workspace():
    r = rank()
    n, out, sig, groups, _head = r._estimate_flat_forward(requests(67, 4096))
    head = 10**10
    cost = price(r, (n, out, sig, groups, head))
    gradient = 67 * 40 * 2048 * 2
    assert cost.checkpoint_workspace == head
    assert cost.required == int((out + head + 2 * gradient) * 1.1)
    assert cost.retained == int((out + head + gradient) * 1.1)
    r._memory_profiles[sig] = _MemoryProfile(
        bytes_per_token=10**9,
        packed_tokens=n,
        logical_per_packed=1,
        retained_compute_bytes_per_token=1,
    )
    assert price(r, (n, out, sig, groups, head)).required == int(
        (out + n * 10**9) * 1.1
    )


@pytest.mark.parametrize("unsupported", [False, True])
def test_no_grad_and_unsupported_checkpoint_do_not_gain_extent(unsupported):
    r = rank()
    req = [
        ForwardInput(
            input_tokens=torch.arange(67),
            hidden_states=True,
            no_grad=not unsupported,
        )
    ]
    values = r._estimate_flat_forward(req)
    if unsupported:
        r.runtime.model[0].decoder.config.recompute_num_layers = 2
    cost = price(r, values)
    assert cost.checkpoint_input_gradient == 0
    assert cost.required == cost.retained


def test_split_sums_all_gradient_children_outside_workspace_max():
    r = rank()
    children = [
        r._plan_flat_forward(
            [
                ForwardInput(
                    input_tokens=torch.arange(rows),
                    hidden_states=True,
                    no_grad=no_grad,
                )
            ]
        )
        for rows, no_grad in ((17, False), (29, False), (83, True))
    ]
    for child in children:
        profile(r, child)
    costs = [r._plan_cost(child) for child in children]
    split = _SplitForwardPlan(tuple(children), ((0,), (1,), (2,)), 3)
    assert [c.checkpoint_input_gradient for c in costs] == [
        17 * 40 * 4096,
        29 * 40 * 4096,
        0,
    ]
    expected = int(
        (
            sum(c.checkpoint_retained + c.checkpoint_input_gradient for c in costs)
            + max(c.checkpoint_workspace for c in costs)
        )
        * 1.1
    )
    old_child_max = sum(c.retained for c in costs) + max(c.ephemeral for c in costs)
    assert expected > old_child_max
    r._available_memory_bytes = lambda: expected - 1
    assert not r._split_rung_check(costs).fits
    check = r._split_plan_memory_check(split, costs)
    assert not check.fits and check.estimated_required_bytes == expected
    r._available_memory_bytes = lambda: expected
    assert r._split_plan_memory_check(split, costs).fits
    key = r._split_memory_key(split)
    assert key is not None
    r._split_memory_floors[key] = 10**9
    assert r._split_plan_memory_check(split, costs).estimated_required_bytes == int(
        10**9 * 1.1
    )


def test_lower_bound_profile_cliff_preserves_separate_peak_component():
    r = rank()
    req = [
        ForwardInput(
            input_tokens=torch.arange(128),
            target_tokens=torch.arange(128),
            no_grad=False,
        )
        for _ in range(16)
    ]
    full = r._plan_flat_forward(req, memory_minimal=True)
    profile(r, full)
    r._memory_profiles[full.signature] = replace(
        r._memory_profiles[full.signature], packed_tokens=256
    )
    lower = r._split_chunk_lower_cost(
        req, tuple(q.input_tokens for q in req), checkpoint=Unset
    )
    assert lower.checkpoint_input_gradient == 128 * 40 * 4096
    assert lower.required <= r._plan_cost(full).required
    assert lower.checkpoint_retained == full.output_bytes + 128 * 40 * 4096


def test_finite_backward_profile_is_not_added_to_static_gradient_extent():
    r = rank()
    plan = r._plan_flat_forward(requests(17, 19))
    static = r._plan_cost(plan).required
    rate = static * 4
    profile(r, plan, rate=rate)
    assert r._plan_cost(plan).required == int(
        (plan.output_bytes + plan.packed_tokens * rate) * 1.1
    )
