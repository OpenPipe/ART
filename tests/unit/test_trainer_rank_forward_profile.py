"""Cold fixed scratch must not poison known single-root hidden-only inference."""

from dataclasses import replace

import pytest
from test_trainer_rank_moe_memory import layer as layer
from test_trainer_rank_pending_memory import pending_rank as pending_rank
import torch

from art.trainer_rank import ForwardInput


def request(tokens):
    return ForwardInput(
        input_tokens=torch.full((tokens,), 198, dtype=torch.long),
        hidden_states=True,
        no_grad=True,
    )


def observe(rank, plan, compute):
    rank._update_memory_profile(
        plan, plan.output_bytes + compute, retained_bytes=plan.output_bytes
    )


def test_three_token_cold_peak_does_not_price_81920_linearly(pending_rank):
    rank = pending_rank
    short = rank._plan_flat_forward([request(3)])
    long = rank._plan_flat_forward([request(81920)])
    assert short.signature == long.signature
    assert short.signature.bounded_forward_peak
    cold_required = rank._memory_check(long).estimated_required_bytes
    # The real H200 three-token reproduction observed this compute delta.
    fixed = 470221312
    observe(rank, short, fixed)
    profile = rank._memory_profiles[short.signature]
    assert profile.caller_plans == 0 and profile.warm_bytes_per_token is None
    assert profile.forward_compute_peak_bytes == fixed
    assert profile.bytes_per_token * 81920 > 10_000 * 2**30
    assert rank._memory_check(long).estimated_required_bytes == cold_required
    assert cold_required < 63 * 2**30


def test_observed_peak_still_refuses_insufficient_memory(pending_rank, monkeypatch):
    rank = pending_rank
    short = rank._plan_flat_forward([request(3)])
    fixed = 470221312
    observe(rank, short, fixed)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: fixed - 1)
    for tokens in (3, 24, 25, 81920):
        check = rank._memory_check(rank._plan_flat_forward([request(tokens)]))
        assert not check.fits
        assert check.estimated_required_bytes >= fixed


def test_costs_are_monotone_and_static_floors_survive(pending_rank):
    rank = pending_rank
    short = rank._plan_flat_forward([request(3)])
    observe(rank, short, 470221312)
    required = []
    for tokens in (1, 2, 3, 4, 8, 23, 24, 25, 128, 8192, 81920):
        plan = rank._plan_flat_forward([request(tokens)])
        profiled = rank._memory_check(plan).estimated_required_bytes
        saved, rank._memory_profiles = rank._memory_profiles, {}
        unprofiled = rank._memory_check(plan).estimated_required_bytes
        rank._memory_profiles = saved
        assert profiled >= unprofiled
        required.append(profiled)
    assert required == sorted(required)
    long = rank._plan_flat_forward([request(81920)])
    observe(rank, long, 32 * 2**30)
    prior = rank._memory_check(long).estimated_required_bytes
    observe(rank, short, 1)
    assert (
        rank._memory_profiles[long.signature].forward_compute_peak_bytes == 32 * 2**30
    )
    assert rank._memory_check(long).estimated_required_bytes >= prior


@pytest.mark.parametrize(
    "mode",
    ["grad", "target", "logits", "topk", "multiple", "matrix", "cp", "unsupported"],
)
def test_other_requests_keep_existing_profile_growth(pending_rank, mode):
    rank = pending_rank
    observed = rank._plan_flat_forward([request(3)])
    value = request(3)
    if mode == "grad":
        value = replace(value, no_grad=False)
    elif mode == "target":
        value = replace(value, target_tokens=torch.ones(3, dtype=torch.long))
    elif mode == "logits":
        value = replace(value, logits=True)
    elif mode == "topk":
        value = replace(value, top_k=1)
    elif mode == "matrix":
        value = replace(value, input_tokens=value.input_tokens.reshape(1, 3))
    elif mode == "cp":
        rank._topology_key = lambda: (1, 1, 2, 1)
    elif mode == "unsupported":
        rank.runtime.model[0].decoder.config.recompute_num_layers = 2
    signature = rank._memory_signature_from_requests(
        [value, value] if mode == "multiple" else [value],
        slot_group_count=1,
        grad_modes=(mode == "grad",),
    )
    assert not signature.bounded_forward_peak
    observe(rank, replace(observed, signature=signature), 470221312)
    assert rank._memory_profiles[signature].forward_compute_peak_bytes == 0
    required = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=81920,
        output_bytes=0,
        signature=signature,
        group_rows=((81920, mode == "grad"),),
    )
    assert required > 10_000 * 2**30
