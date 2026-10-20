"""Cold fixed scratch must not poison known single-root hidden-only inference."""

from dataclasses import replace

import pytest
from test_trainer_rank_moe_memory import layer as layer
from test_trainer_rank_pending_memory import pending_rank as pending_rank
import torch

from art.trainer_rank import ForwardInput, _memory


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


def test_direct_gradient_profile_keeps_static_and_retained_graph_floors(pending_rank):
    rank = pending_rank
    inputs = [replace(request(3), no_grad=False)]
    yielded = rank._plan_flat_forward(inputs)
    assert not yielded.signature.bounded_forward_peak
    with _memory.direct_forward_profile():
        short = rank._plan_flat_forward(inputs)
        long = rank._plan_flat_forward([replace(request(81920), no_grad=False)])
    assert short.signature == long.signature and short.signature.bounded_forward_peak
    assert short.signature != yielded.signature
    # Existing policy drops a one-time recompute allowance once profiled;
    # compare the token-dependent floors with that same observation state.
    observe(rank, short, 0)
    static = rank._memory_check(long).estimated_required_bytes
    rank._memory_profiles.clear()
    rank._update_memory_profile(
        short,
        short.output_bytes + 4058112,
        retained_bytes=short.output_bytes + 516608,
    )
    required = rank._memory_check(long).estimated_required_bytes
    assert static <= required < 63 * 2**30
    assert (
        rank._plan_cost(long).retained
        >= rank._checkpoint_memory_floor(((81920, True),))[0]
    )
    assert rank._memory_profiles[short.signature].caller_plans == 0
    assert rank._plan_flat_forward(inputs).signature == yielded.signature


def test_direct_scope_resets_after_nested_scope_and_exception(pending_rank):
    rank = pending_rank
    inputs = [replace(request(3), no_grad=False)]
    with pytest.raises(ValueError, match="scope"):
        with _memory.direct_forward_profile():
            with _memory.direct_forward_profile():
                assert rank._plan_flat_forward(inputs).signature.bounded_forward_peak
            assert rank._plan_flat_forward(inputs).signature.bounded_forward_peak
            raise ValueError("scope")
    assert not rank._plan_flat_forward(inputs).signature.bounded_forward_peak


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
