"""Borrowed caller coverage must not expire into cheaper admission."""

from dataclasses import replace

import pytest
from test_trainer_rank_active_memory import _rank
from test_trainer_rank_fixed_slope import signature
from test_trainer_rank_unanchored_floor import observe, required
import torch

from art.trainer_rank import ForwardInput
from art.trainer_rank import _memory as m


def test_borrowed_reference_price_does_not_drop_at_root_extent(monkeypatch):
    rank = _rank()
    monkeypatch.setattr(
        rank, "_available_memory_bytes", lambda sample=None: 4_000_000_000
    )
    requests = [
        ForwardInput(
            input_tokens=torch.arange(4096),
            target_tokens=torch.arange(4096),
            no_grad=False,
        ),
        ForwardInput(
            input_tokens=torch.arange(4096) + 100000,
            target_tokens=torch.arange(4096),
            no_grad=False,
        ),
    ]
    short = rank._plan_flat_forward(requests)
    longer_requests = [
        requests[0],
        replace(
            requests[1],
            input_tokens=torch.arange(4097) + 100000,
            target_tokens=torch.arange(4097),
        ),
    ]
    long = rank._plan_flat_forward(longer_requests)
    assert (short.packed_tokens, long.packed_tokens) == (8192, 8193)
    assert short.signature == long.signature
    root = m._ScopedMemorySignature(**vars(short.signature), bounded_caller_peak=True)
    anchor = replace(short.signature, request_mix=("hidden",))
    with m.fixed_slope_memory_profiles():
        observe(rank, root, 8192, 8_000_000_000)
        observe(rank, anchor, 32768, 1_000_000_000)
        observe(rank, short.signature, 3, 300_000_000)
        checks = [rank._memory_check(p) for p in (short, long)]
        assert not checks[0].fits
        assert checks[1].estimated_required_bytes >= checks[0].estimated_required_bytes
        assert not checks[1].fits
        assert checks[1].estimated_required_bytes >= m._MEMORY_DEFERRED_BYTES


@pytest.mark.parametrize("cp", [1, 2, 8])
@pytest.mark.parametrize("anchor_peak", [1_000_000_000, 16_000_000_000])
def test_expired_root_defers_with_any_model_anchor(cp, anchor_peak):
    rank = _rank()
    key = replace(signature((0, "forward_backward", 8)), topology=(1, 1, cp, 1))
    root = m._ScopedMemorySignature(**vars(key), bounded_caller_peak=True)
    work = ("test-family", ((64,),) * cp)
    with m.fixed_slope_memory_profiles():
        observe(rank, root, 8192, 8_000_000_000, cp_workload=work)
        observe(rank, replace(key, request_mix=("hidden",)), 32768, anchor_peak)
        observe(rank, key, 3, 300_000_000)
        prices = [
            required(rank, key, n, cp_workload=work)
            for n in (8191, 8192, 8193, 16384, 32768)
        ]
        assert prices == sorted(prices)
        assert all(price >= m._MEMORY_DEFERRED_BYTES for price in prices[2:])
        assert (
            m._rate_reference(
                rank,
                key,
                rank._memory_profiles[key],
                packed_tokens=8193,
                caller_cp_workload=work,
            )
            is None
        )


@pytest.mark.parametrize("cp", [1, 2, 8])
def test_expired_model_anchor_cannot_leave_only_a_cheaper_contributor(cp):
    rank = _rank()
    base = replace(signature((0, "forward_backward", 8)), topology=(1, 1, cp, 1))
    key = m._ScopedMemorySignature(**vars(base), bounded_caller_peak=True)
    work = ("test-family", ((64,),) * cp)
    with m.fixed_slope_memory_profiles():
        observe(
            rank,
            replace(key, request_mix=("hidden",)),
            8192,
            8_000_000_000,
            cp_workload=work,
        )
        # Both contributors start at the same qualified floor; only the cheap
        # contributor extends beyond it. This isolates expiry from minimums.
        observe(
            rank,
            replace(key, request_mix=("logits",)),
            8192,
            250_000_000,
            cp_workload=work,
        )
        observe(
            rank,
            replace(key, request_mix=("logits",)),
            32768,
            1_000_000_000,
            cp_workload=work,
        )
        reference = m._rate_reference(
            rank,
            key,
            None,
            packed_tokens=8192,
            caller_cp_workload=work,
        )
        assert reference is not None and reference.source == "model_anchor"
        before = required(rank, key, 8192, cp_workload=work)
        after = required(rank, key, 8193, cp_workload=work)
        assert after >= before
        assert after >= m._MEMORY_DEFERRED_BYTES
