"""Legacy equality survives updates, persistence, bounded eviction and merging."""

from dataclasses import asdict, replace
import json
import pickle

import pytest
from test_trainer_rank_calibration_seed import metadata_rank
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_cp_kernel_identity import legacy_profile

from art.trainer_rank import _memory as memory
from art.trainer_rank import _planner_misses as reports

TAG = "memory_observation_format"


def observation_version(point):
    return json.loads(point[2]).get(TAG)


def restored_rank():
    signature, profile = legacy_profile()
    rank = metadata_rank("moe")
    rank._memory_profiles[signature] = profile
    rank._available_memory_bytes = lambda sample=None: 71_000_000_000
    return rank, signature


def smaller_measurement(rank):
    plan = rank._plan_flat_forward([request(8192, grad=True)])
    rank._update_memory_profile(
        plan, 1_000_000_000, retained_bytes=None, caller_phase=True
    )
    return rank._memory_profiles[plan.signature]


def merge(left, right):
    # Same exact-ownership union and bounded eviction as profile consumers.
    points = {}
    for n, peak, family, loads in (*left.caller_cp_peaks, *right.caller_cp_peaks):
        key = n, family, loads
        points[key] = max(peak, points.get(key, 0))
    evidence = sorted(
        (n, peak, family, loads) for (n, family, loads), peak in points.items()
    )
    fields = dict(
        asdict(left), **memory._merge_memory_profile_model_fields(left, right)
    )
    fields["caller_cp_peaks"] = tuple(
        [evidence[0], *evidence[-7:]] if len(evidence) > 8 else evidence
    )
    return memory._memory_profile_from_values(fields)


def test_legacy_larger_equality_survives_smaller_caller_measurement():
    rank, signature = restored_rank()
    profile = rank._memory_profiles[signature]
    with memory.fixed_slope_memory_profiles():
        larger = rank._plan_flat_forward([request(16384, grad=True)])
        smaller = rank._plan_flat_forward([request(8192, grad=True)])
        assert larger.signature == smaller.signature == signature
        before = profile.covering_cp_peak(16384, larger.caller_cp_workload)
        assert before == 2_499_934_464
        assert rank._memory_check(larger).fits
        assert rank._memory_check(smaller).fits
        rank._update_memory_profile(
            smaller, 1_000_000_000, retained_bytes=None, caller_phase=True
        )
        updated = rank._memory_profiles[signature]
        after = updated.covering_cp_peak(16384, larger.caller_cp_workload)
        assert after == before
        assert rank._memory_check(larger).fits
        assert updated.profile_format_version == 7
        assert {observation_version(p) for p in updated.caller_cp_peaks} == {None, 1}
        with rank.calibration_rung([request(16384, grad=True)], no_grad=False) as rung:
            assert rung._coverage_guard is None


def assert_legacy_only_refuses_subset(rank, signature, profile):
    subset = rank._plan_flat_forward([request(512, grad=True)])
    # Isolate the retained legacy observation: the new qualified 8K reading
    # can independently cover this request and must not hide the legacy gate.
    legacy = replace(
        profile,
        caller_cp_peaks=tuple(
            p for p in profile.caller_cp_peaks if observation_version(p) is None
        ),
    )
    assert legacy.caller_cp_peaks
    assert legacy.covering_cp_peak(512, subset.caller_cp_workload) is None
    rank._memory_profiles[signature] = legacy
    assert not rank._memory_check(subset).fits
    rank._memory_profiles[signature] = profile


def test_legacy_observation_never_gains_dominance_after_promotion():
    with memory.fixed_slope_memory_profiles():
        rank, signature = restored_rank()
        profile = smaller_measurement(rank)
        assert_legacy_only_refuses_subset(rank, signature, profile)
        # Fresh qualified evidence still grants the supported dominance.
        plan = rank._plan_flat_forward([request(512, grad=True)])
        covered = profile.covering_cp_observation(512, plan.caller_cp_workload)
        assert covered is not None and observation_version(covered) == 1
        assert rank._memory_check(plan).fits


@pytest.mark.parametrize("encoding", ["pickle", "json"])
@pytest.mark.parametrize("stage", ["legacy", "promoted", "merged"])
def test_per_observation_provenance_survives_roundtrip_and_further_updates(
    encoding, stage
):
    with memory.fixed_slope_memory_profiles():
        rank, signature = restored_rank()
        original = rank._memory_profiles[signature]
        if stage != "legacy":
            smaller_measurement(rank)
        profile = rank._memory_profiles[signature]
        if stage == "merged":
            profile = merge(profile, original)
        points = profile.caller_cp_peaks
        if encoding == "pickle":
            profile = pickle.loads(pickle.dumps(profile))
        else:
            profile = reports._memory_profile(json.loads(json.dumps(asdict(profile))))
        assert profile.caller_cp_peaks == points
        memory._validate_memory_profile_model(profile)
        rank._memory_profiles[signature] = profile
        updated = smaller_measurement(rank)
        plan = rank._plan_flat_forward([request(16384, grad=True)])
        assert updated.covering_cp_peak(16384, plan.caller_cp_workload) == 2_499_934_464
        assert rank._memory_check(plan).fits
        assert_legacy_only_refuses_subset(rank, signature, updated)


@pytest.mark.parametrize("encoding", ["pickle", "json"])
def test_format6_without_observation_tags_is_legacy_even_when_qualified(encoding):
    with memory.fixed_slope_memory_profiles():
        rank = metadata_rank("moe")
        rank._available_memory_bytes = lambda sample=None: 71_000_000_000
        larger = rank._plan_flat_forward([request(16384, grad=True)])
        rank._update_memory_profile(
            larger, 2_500_000_000, retained_bytes=None, caller_phase=True
        )
        points = []
        for n, peak, family, loads in rank._memory_profiles[
            larger.signature
        ].caller_cp_peaks:
            data = json.loads(family)
            data.pop(TAG, None)
            points.append(
                (
                    n,
                    peak,
                    json.dumps(data, sort_keys=True, separators=(",", ":")),
                    loads,
                )
            )
        old = replace(
            rank._memory_profiles[larger.signature],
            profile_format_version=6,
            caller_cp_peaks=tuple(points),
        )
        if encoding == "pickle":
            old = pickle.loads(pickle.dumps(old))
        else:
            old = reports._memory_profile(json.loads(json.dumps(asdict(old))))
        rank._memory_profiles[larger.signature] = old
        assert old.covering_cp_peak(16384, larger.caller_cp_workload) == 2_499_934_464
        subset = rank._plan_flat_forward([request(512, grad=True)])
        assert old.covering_cp_peak(512, subset.caller_cp_workload) is None
        updated = smaller_measurement(rank)
        assert (
            updated.covering_cp_peak(16384, larger.caller_cp_workload) == 2_499_934_464
        )
        assert_legacy_only_refuses_subset(rank, larger.signature, updated)


def test_merges_preserve_mixed_provenance_in_both_orders():
    with memory.fixed_slope_memory_profiles():
        rank, signature = restored_rank()
        original = rank._memory_profiles[signature]
        updated = smaller_measurement(rank)
        a, b = merge(original, updated), merge(updated, original)
        assert a.caller_cp_peaks == b.caller_cp_peaks == updated.caller_cp_peaks
        assert {observation_version(p) for p in a.caller_cp_peaks} == {None, 1}
        for profile in (a, b):
            memory._validate_memory_profile_model(profile)
            rank._memory_profiles[signature] = profile
            plan = rank._plan_flat_forward([request(16384, grad=True)])
            assert (
                profile.covering_cp_peak(16384, plan.caller_cp_workload)
                == 2_499_934_464
            )
            assert rank._memory_check(plan).fits
            assert_legacy_only_refuses_subset(rank, signature, profile)


def test_multiple_updates_and_bounded_eviction_keep_retained_legacy_equality():
    with memory.fixed_slope_memory_profiles():
        rank, signature = restored_rank()
        for n in (8192, 4096, 2048, 1024, 512, 384, 256, 128, 64):
            plan = rank._plan_flat_forward([request(n, grad=True)])
            rank._update_memory_profile(
                plan, 1_000_000_000, retained_bytes=None, caller_phase=True
            )
            profile = rank._memory_profiles[signature]
            assert len(profile.caller_cp_peaks) <= 8
            larger = rank._plan_flat_forward([request(16384, grad=True)])
            assert (
                profile.covering_cp_peak(16384, larger.caller_cp_workload)
                == 2_499_934_464
            )
            assert rank._memory_check(larger).fits


def test_revocation_identity_ignores_observation_provenance_tags():
    with memory.fixed_slope_memory_profiles():
        rank, signature = restored_rank()
        profile = smaller_measurement(rank)
        plan = rank._plan_flat_forward([request(512, grad=True)])
        point = profile.covering_cp_observation(512, plan.caller_cp_workload)
        assert point is not None and observation_version(point) == 1
        data = json.loads(point[2])
        data.pop(TAG)
        original = json.dumps(data, sort_keys=True, separators=(",", ":"))
        assert memory._cp_pair(plan.caller_cp_workload[0], original) == memory._cp_pair(
            plan.caller_cp_workload[0], point[2]
        )
        blocked = memory._block_cp_pair(profile, plan.caller_cp_workload[0], original)
        assert blocked.covering_cp_peak(512, plan.caller_cp_workload) is None
