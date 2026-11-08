"""Qualified CP mask subsets and per-release revocation on measured violations."""

from dataclasses import asdict, replace
import json
import pickle
from unittest.mock import patch

import pytest
from test_trainer_rank_calibration_seed import fake_device, metadata_rank
from test_trainer_rank_calibration_trust_region import request

from art.megatron.flex_attn import compiled
from art.trainer_rank import CalibrationCoverageViolation, TrainerRankMemoryError
from art.trainer_rank import _memory as memory
from art.trainer_rank import _planner_misses as reports


def seed(monkeypatch, kind="moe"):
    rank = metadata_rank(kind)
    monkeypatch.setattr(
        rank, "_available_memory_bytes", lambda sample=None: 71_000_000_000
    )
    fake_device(monkeypatch, rank, 2_500_000_000)
    with rank.calibration_rung([request(16384, grad=True)], no_grad=False) as rung:
        list(rung.forward_micro_batches())
    return rank


def change_family(plan, edit):
    family, loads = plan.caller_cp_workload
    values = json.loads(family)
    edit(values)
    return replace(
        plan,
        caller_cp_workload=(
            json.dumps(values, sort_keys=True, separators=(",", ":")),
            loads,
        ),
    )


@pytest.mark.parametrize("n", [512, 2048, 8192])
def test_moe_mask_subsets_are_covered_after_only_one_16k_seed(monkeypatch, n):
    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        plan = rank._plan_flat_forward([request(n, grad=True)])
        profile = rank._memory_profiles[plan.signature]
        point = profile.covering_cp_observation(n, plan.caller_cp_workload)
        assert point is not None and rank._memory_check(plan).fits
        if n < 8192:
            assert point[2] != plan.caller_cp_workload[0]
            assert memory._cp_family_dominates(plan.caller_cp_workload[0], point[2])


@pytest.mark.parametrize(
    "negative",
    [
        "kind",
        "component",
        "rank_class",
        "backend",
        "kernel",
        "model",
        "missing_qualification",
    ],
)
def test_incompatible_cp_evidence_still_refuses(monkeypatch, negative):
    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        original = rank._plan_flat_forward([request(2048, grad=True)])
        profile = rank._memory_profiles[original.signature]
        plan = original
        if negative == "kind":
            entries = []
            for n, peak, family, loads in profile.caller_cp_peaks:
                data = json.loads(family)
                data["masks"][0] = ["causal"]
                entries.append(
                    (
                        n,
                        peak,
                        json.dumps(data, sort_keys=True, separators=(",", ":")),
                        loads,
                    )
                )
            rank._memory_profiles[plan.signature] = replace(
                profile, caller_cp_peaks=tuple(entries)
            )
            plan = change_family(plan, lambda d: d["masks"].__setitem__(0, ["full"]))
        elif negative == "component":
            family, loads = plan.caller_cp_workload
            rows = [list(row) for row in loads]
            rows[-1][-1] = profile.caller_cp_peaks[0][3][-1][-1] + 1
            plan = replace(
                plan, caller_cp_workload=(family, tuple(tuple(row) for row in rows))
            )
        elif negative == "rank_class":
            monkeypatch.setattr(
                rank, "_slot_memory_shapes", lambda ref: ((2, 2048, 32, 32, 2048),)
            )
            plan = rank._plan_flat_forward([request(2048, grad=True)])
            assert plan.signature != original.signature
        elif negative == "backend":
            monkeypatch.setattr(compiled, "_FORCED_FLEX_BACKEND", "TRITON")
            plan = rank._plan_flat_forward([request(2048, grad=True)])
        elif negative == "kernel":
            monkeypatch.setitem(compiled._FORCED_FLEX_KERNEL_OPTIONS, "num_stages", 9)
            plan = rank._plan_flat_forward([request(2048, grad=True)])
        elif negative == "model":
            plan = change_family(
                plan, lambda d: d["execution_qualification"].__setitem__("geometry", {})
            )
        else:
            plan = change_family(plan, lambda d: d.pop("execution_qualification"))
        assert not rank._memory_check(plan).fits
        if plan.signature == original.signature:
            assert (
                rank._memory_profiles[plan.signature].covering_cp_peak(
                    plan.packed_tokens, plan.caller_cp_workload
                )
                is None
            )


@pytest.mark.parametrize("n", [512, 2048])
def test_calibration_violation_revokes_pair_persists_and_signals_once(monkeypatch, n):
    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        plan = rank._plan_flat_forward([request(n, grad=True)])
        profile = rank._memory_profiles[plan.signature]
        covered = profile.covering_cp_observation(n, plan.caller_cp_workload)
        assert covered is not None
        covering = covered[1] + plan.output_bytes
        executed, state = fake_device(monkeypatch, rank, covering + 1_000_000)
        with rank.calibration_rung([request(n, grad=True)], no_grad=False) as rung:
            assert rung.decision.fits and not rung.decision.seed
            with pytest.raises(CalibrationCoverageViolation) as raised:
                list(rung.forward_micro_batches())
            error = raised.value
            assert rung.coverage_violation is error
            assert error.requested_family == plan.caller_cp_workload[0]
            assert error.observed_family == covered[2]
            assert (
                error.measured_peak_bytes
                == state["peak"]
                > error.covering_peak_bytes
                == covering
            )
            assert pickle.loads(pickle.dumps(error)).__dict__ == error.__dict__
        assert len(executed) == 1
        assert rank._suppressed_caller_memory_samples == 0
        assert "_calibration_coverage_guard" not in rank.__dict__
        profile = rank._memory_profiles[plan.signature]
        pair = memory._cp_pair(error.requested_family, error.observed_family)
        assert (
            profile.profile_format_version == 7
            and pair in profile.non_dominant_cp_pairs
        )
        # The unsafe reading tightens floors but never creates equality coverage.
        assert all(
            not profile.cp_family_equal(error.requested_family, family)
            for _, _, family, _ in profile.caller_cp_peaks
        )
        assert profile.covering_cp_peak(n, plan.caller_cp_workload) is None
        for restored in (
            pickle.loads(pickle.dumps(profile)),
            reports._memory_profile(json.loads(json.dumps(asdict(profile)))),
        ):
            assert restored.non_dominant_cp_pairs == profile.non_dominant_cp_pairs
            assert restored.covering_cp_peak(n, plan.caller_cp_workload) is None
            memory._validate_memory_profile_model(restored)
        merged = memory._merge_memory_profile_model_fields(
            profile, replace(profile, non_dominant_cp_pairs=())
        )
        assert pair in merged["non_dominant_cp_pairs"]
        with rank.calibration_rung([request(n, grad=True)], no_grad=False) as retry:
            assert not retry.decision.fits
            with pytest.raises(TrainerRankMemoryError):
                retry.forward_micro_batches()
            assert retry.coverage_violation is None
        assert len(executed) == 1


def test_safe_dominance_rung_learns_equality_and_preserves_caller_error(monkeypatch):
    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        executed, _ = fake_device(monkeypatch, rank, 1_000_000_000)
        with rank.calibration_rung([request(2048, grad=True)], no_grad=False) as rung:
            list(rung.forward_micro_batches())
            assert rung.coverage_violation is None
        plan = executed[0]
        profile = rank._memory_profiles[plan.signature]
        assert any(
            profile.cp_family_equal(plan.caller_cp_workload[0], family)
            for _, _, family, _ in profile.caller_cp_peaks
        )
        assert rank._memory_check(plan).fits
        with rank.calibration_rung(
            [request(512, grad=True)], no_grad=False
        ) as interrupted:
            iterator = interrupted.forward_micro_batches()
            next(iterator)
            error = ValueError("caller interrupted")
            with pytest.raises(ValueError) as raised:
                iterator.throw(error)
            assert raised.value is error and interrupted.coverage_violation is None
        assert "_calibration_coverage_guard" not in rank.__dict__


@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_equality_dense_and_unanchored_floor_are_preserved(monkeypatch, kind):
    with memory.fixed_slope_memory_profiles():
        rank = metadata_rank(kind)
        monkeypatch.setattr(
            rank, "_available_memory_bytes", lambda sample=None: 71_000_000_000
        )
        cold = rank._plan_flat_forward([request(16384, grad=True)])
        assert rank._memory_check(cold).estimated_required_bytes >= 1 << 62
        rank = seed(monkeypatch, kind)
        plan = rank._plan_flat_forward([request(16384, grad=True)])
        assert rank._memory_check(plan).fits
        if kind == "moe":
            profile = rank._memory_profiles[plan.signature]
            disabled = replace(profile, dominance_disabled=True)
            assert disabled.covering_cp_peak(16384, plan.caller_cp_workload) is not None
        else:
            assert not isinstance(
                rank._memory_profiles[plan.signature], memory._CallerMemoryProfile
            )


def test_revocation_budget_disables_dominance_without_forgetting_unsafe_pairs(
    monkeypatch,
):
    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        plan = rank._plan_flat_forward([request(512, grad=True)])
        profile = rank._memory_profiles[plan.signature]
        for i in range(100):
            profile = memory._block_cp_pair(profile, f"request-{i}", f"observed-{i}")
        assert len(profile.non_dominant_cp_pairs) == memory._CP_BLOCKED_PAIR_BUDGET
        assert profile.dominance_disabled
        assert profile.covering_cp_peak(512, plan.caller_cp_workload) is None
        memory._validate_memory_profile_model(profile)


def test_early_v2_caller_evidence_remains_readable_and_mergeable(monkeypatch):
    from art.trainer_rank import _memory_model

    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        profile = next(iter(rank._memory_profiles.values()))
        fixed, slope = _memory_model.fit_points(profile.compute_observations)
        old = replace(
            profile,
            profile_format_version=2,
            fixed_compute_bytes=fixed,
            compute_slope=slope,
            rate_floor_tokens=0,
            rate_observations=None,
            absolute_compute_floor=0,
            eligible_packed_tokens=0,
            subfloor_packed_tokens=0,
            retained_rate_tokens=0,
            retained_compute_peaks=(),
        )
        memory._validate_memory_profile_model(old)
        values = {**asdict(old), **memory._merge_memory_profile_model_fields(old, old)}
        restored = memory._memory_profile_from_values(values)
        assert restored.profile_format_version == 2
        assert restored.compute_observations == old.compute_observations
        memory._validate_memory_profile_model(restored)
