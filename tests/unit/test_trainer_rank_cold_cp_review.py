"""Coverage revocations cannot enter the uncovered short-CP fallback."""

from dataclasses import replace

import pytest
from test_trainer_rank_calibration_seed import fake_device
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_cp_dominance import seed

from art.trainer_rank import CalibrationCoverageViolation
from art.trainer_rank import _memory as memory


def test_revoked_subfloor_pair_stays_fail_closed(monkeypatch):
    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        plan = rank._plan_flat_forward([request(72, grad=True)])
        profile = rank._memory_profiles[plan.signature]
        point = profile.covering_cp_observation(72, plan.caller_cp_workload)
        assert point is not None
        before = rank._memory_check(plan)
        measured = point[1] + plan.output_bytes + 1_000_000
        fake_device(monkeypatch, rank, measured)
        with rank.calibration_rung([request(72, grad=True)], no_grad=False) as rung:
            assert rung.decision.fits
            with pytest.raises(CalibrationCoverageViolation):
                list(rung.forward_micro_batches())
        profile = rank._memory_profiles[plan.signature]
        assert profile.non_dominant_cp_pairs
        assert profile.covering_cp_peak(72, plan.caller_cp_workload) is None
        # Budget below the disproving measurement, above the new cold estimate.
        rank._available_memory_bytes = lambda sample=None: measured - 1
        after = rank._memory_check(plan)
        print(
            dict(
                before=before.estimated_required_bytes,
                measured=measured,
                after=after.estimated_required_bytes,
                fits=after.fits,
                revoked=profile.non_dominant_cp_pairs,
            )
        )
        assert not after.fits


def test_disabling_all_dominance_does_not_lower_subfloor_requirement(monkeypatch):
    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        plan = rank._plan_flat_forward([request(72, grad=True)])
        before = rank._memory_check(plan).estimated_required_bytes
        profile = rank._memory_profiles[plan.signature]
        rank._memory_profiles[plan.signature] = replace(
            profile, dominance_disabled=True
        )
        after = rank._memory_check(plan).estimated_required_bytes
        print(dict(before=before, after_disabling_dominance=after))
        assert after >= before
