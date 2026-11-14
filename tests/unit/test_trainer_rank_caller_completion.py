"""Forward observations cannot qualify a caller's unmeasured backward."""

from dataclasses import asdict, replace
import json
import math
import pickle
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_moe_long_root_caller import memory_rank, signature

from art.trainer_rank import _memory as m
from art.trainer_rank import _planner_misses as reports

GIB = 2**30


def observe(rank, sig, tokens, peak, complete):
    plan = SimpleNamespace(
        signature=sig,
        packed_tokens=tokens,
        active_logical_tokens=tokens,
        output_bytes=0,
        caller_cp_workload=("test-family", ((1,),) * sig.topology[2]),
    )
    m._update_memory_profile(
        rank, cast(Any, plan), peak, retained_bytes=None, caller_phase=complete
    )


def required(rank, sig, tokens):
    return m._estimate_required_memory_bytes_from_values(
        rank,
        packed_tokens=tokens,
        logical_tokens=tokens,
        output_bytes=0,
        signature=sig,
        group_rows=((tokens, True),),
        caller_cp_workload=("test-family", ((1,),) * sig.topology[2]),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_forward_only_measurement_cannot_qualify_backward(enabled):
    rank, sig = memory_rank(), signature()
    with m.fixed_slope_memory_profiles(enabled):
        observe(rank, sig, 8192, 16 * GIB, True)
        before = required(rank, sig, 32768)
        assert before > 40 * GIB
        # Direct forwards and abandoned iterators supply this incomplete sample.
        observe(rank, sig, 32768, 20 * GIB, False)
        assert rank._memory_profiles[sig].caller_compute_peaks == ((8192, 16 * GIB),)
        after = required(rank, sig, 32768)
        assert after >= before
        assert after > 40 * GIB


@pytest.mark.parametrize("cp", [1, 2])
def test_forward_only_sample_cannot_lower_covered_caller_fit(cp):
    rank = memory_rank()
    sig = replace(signature(), topology=(1, 1, cp, 1))
    with m.fixed_slope_memory_profiles():
        observe(rank, sig, 8192, 16 * GIB, True)
        observe(rank, sig, 16384, 20 * GIB, True)
        before = [required(rank, sig, n) for n in (8192, 16384, 32768)]
        observe(rank, sig, 32768, 24 * GIB, False)
        assert all(
            required(rank, sig, n) >= old
            for n, old in zip((8192, 16384, 32768), before, strict=True)
        )
        profile = rank._memory_profiles[sig]
        assert m._cold_extrapolation_factor(profile, 32768) == 1.75


def test_forward_only_long_root_does_not_anchor_caller_or_remove_cold_margin():
    rank, sig = memory_rank(), signature()
    with m.fixed_slope_memory_profiles():
        observe(rank, sig, 32768, 20 * GIB, False)
        profile = rank._memory_profiles[sig]
        assert m._unanchored(profile)
        assert m._cold_extrapolation_factor(profile, 8192) == 1.75
        observe(rank, sig, 8192, 16 * GIB, True)
        assert m._cold_extrapolation_factor(rank._memory_profiles[sig], 8192) == 1


@pytest.mark.parametrize("cp", [1, 2])
@pytest.mark.parametrize("tokens", [1, 4, 16, 72])
def test_tiny_caller_peak_only_raises_fixed_cost(cp, tokens):
    rank = memory_rank()
    sig = replace(signature(), topology=(1, 1, cp, 1))
    with m.fixed_slope_memory_profiles():
        observe(rank, sig, 8192, 16 * GIB, True)
        profile = rank._memory_profiles[sig]
        slope, fixed = profile.compute_slope, profile.fixed_compute_bytes
        before = required(rank, sig, 8192)
        observe(rank, sig, tokens, 60 * GIB, True)
        profile = rank._memory_profiles[sig]
        assert profile.compute_slope == slope
        # An enveloped covering curve can smear the tiny peak into the long
        # root. Its token rate must still come from the actual long reading.
        assert (
            required(rank, sig, 8192)
            <= before + math.ceil((profile.fixed_compute_bytes - fixed) * 1.1) + 1
        )


@pytest.mark.parametrize("cp", [1, 2])
def test_incomplete_only_caller_roundtrip_has_no_qualified_rate(cp):
    rank = memory_rank()
    sig = replace(signature(), topology=(1, 1, cp, 1))
    with m.fixed_slope_memory_profiles():
        observe(rank, sig, 32768, 20 * GIB, False)
        profile = rank._memory_profiles[sig]
        if getattr(profile, "profile_format_version", 1) < 3:
            assert profile.compute_observations == ()
        else:
            assert profile.rate_observations == ()
        assert profile.compute_slope == (0, 1)
        assert profile.compute_peak_floor == 20 * GIB
        restored = reports._memory_profile(json.loads(json.dumps(asdict(profile))))
        assert restored == profile == pickle.loads(pickle.dumps(profile))
        assert m._unanchored(restored)
        assert m._cold_extrapolation_factor(restored, 8192) == 1.75
