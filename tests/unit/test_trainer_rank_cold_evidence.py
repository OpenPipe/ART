"""CPU review witnesses; synthetic readings, no model or GPU execution."""

import pytest
from test_trainer_rank_cp_output_families import anchors, native_rank, output_requests
import torch

from art.trainer_rank import _memory as memory


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("grad", [False, True])
def test_cold_fallback_preserves_its_own_observed_peak(kind, grad):
    rank = native_rank(kind)
    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        anchors(rank, grad)
        inputs = output_requests(2971, grad, "none", True, False, None, False)
        plan = rank._plan_flat_forward(inputs)
        before = rank._memory_check(plan)
        assert before.fits
        measured_compute = max(10_000_000_000, 5 * before.estimated_required_bytes)
        rank._update_memory_profile(
            plan,
            measured_compute + plan.output_bytes,
            retained_bytes=None,
            caller_phase=True,
            rate_eligible=False,
        )
        repeated = rank._plan_flat_forward(inputs)
        assert repeated.signature == plan.signature
        profile = rank._memory_profiles[repeated.signature]
        assert profile.absolute_compute_floor >= measured_compute
        rank._available_memory_bytes = lambda sample=None: measured_compute // 2
        after = rank._memory_check(repeated)
        print(
            dict(
                kind=kind,
                grad=grad,
                before=before.estimated_required_bytes,
                after=after.estimated_required_bytes,
                measured=measured_compute,
                budget=measured_compute // 2,
                fits=after.fits,
                profile=type(profile).__name__,
            )
        )
        assert not after.fits, "a measured peak must remain a binding admission floor"


def test_completed_ordinary_caller_peak_stays_binding(monkeypatch):
    from test_trainer_rank_calibration_seed import fake_device

    rank = native_rank("moe")
    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        anchors(rank, False)
        inputs = output_requests(2971, False, "none", True, False, None, False)
        fake_device(monkeypatch, rank, 10_000_000_000)
        assert len(list(rank.forward_micro_batches(inputs, no_grad=True))) == 1
        plan = rank._plan_flat_forward(inputs)
        profile = rank._memory_profiles[plan.signature]
        assert profile.absolute_compute_floor > 9_000_000_000
        rank._available_memory_bytes = lambda sample=None: 5_000_000_000
        check = rank._memory_check(plan)
        assert not check.fits, (profile.absolute_compute_floor, check)


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("grad", [False, True])
def test_new_cold_family_report_replays(kind, grad, tmp_path):
    from types import MethodType

    from test_grouped_planner_replay import emitted

    from art.trainer_rank import _impl
    from art.trainer_rank import _planner_misses as reports

    rank = native_rank(kind)
    # Stock cost reader, with CP8 topology supplied by the CPU fixture.
    rank._topology_key = MethodType(_impl.TrainerRank._topology_key, rank)
    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        anchors(rank, True)
        inputs = output_requests(2971, grad, "none", True, False, None, False)
        rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
        plan = rank._plan_flat_forward(inputs)
        report, costs = emitted(rank, plan, tmp_path)
        assert report["replay_complete"], report["incomplete_reasons"]
        replayed = reports.replay(report)
        print(
            dict(
                kind=kind,
                live=costs[0].required,
                replay=replayed["estimates"][0],
                argument_names=tuple(
                    report["replay"]["memory_replay"]["estimates"][0]["arguments"]
                ),
            )
        )
        assert replayed["aggregate"]["matches"]


def test_hidden_family_seed_composes_with_bounded_forward_scope(monkeypatch):
    from test_trainer_rank_calibration_seed import fake_device

    rank = native_rank("moe")
    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        anchors(rank, False)
        inputs = output_requests(4095, False, "none", True, False, None, False)
        fake_device(monkeypatch, rank, 1_000_000_000)
        with memory.bounded_forward_peaks():
            with rank.calibration_rung(
                inputs,
                no_grad=True,
                seed_uncovered_cp_family=True,
            ) as rung:
                assert rung.decision.seed and rung.decision.fits
                list(rung.forward_micro_batches())


def test_profile_construction_failure_precedes_seed_attempt(monkeypatch):
    rank = native_rank("moe")
    with (
        memory.fixed_slope_memory_profiles(),
        torch.no_grad(),
        memory.bounded_forward_peaks(),
    ):
        anchors(rank, False)
        inputs = output_requests(4095, False, "none", True, False, None, False)
        before = dict(rank._memory_profiles)

        def fail(*args, **kwargs):
            raise TypeError("profile constructor unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(memory._CallerMemoryProfileV2, "__init__", fail)
            with pytest.raises(TypeError, match="profile constructor"):
                with rank.calibration_rung(
                    inputs, no_grad=True, seed_uncovered_cp_family=True
                ):
                    pass
        assert rank._memory_profiles == before
        assert not rank.__dict__.get("_calibration_seeded_cp_families", ())
        with rank.calibration_rung(
            inputs, no_grad=True, seed_uncovered_cp_family=True
        ) as rung:
            assert rung.decision.seed and rung.decision.fits


@pytest.mark.parametrize(
    "floor,budget,required",
    [
        (200_000_000, 5_000_000_000, 10_000_000_000),
        (20_000_000_000, 15_000_000_000, 20_000_000_000),
    ],
)
def test_empirical_and_minimal_family_floors_both_bind(floor, budget, required):
    rank = native_rank("moe")
    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        anchors(rank, False)
        inputs = output_requests(2971, False, "none", True, False, None, False)
        plan = rank._plan_flat_forward(inputs)
        rank._update_memory_profile(
            plan,
            10_000_000_000 + plan.output_bytes,
            retained_bytes=None,
            caller_phase=True,
            rate_eligible=False,
        )
        token = memory._CP_FAMILY_CALIBRATION.set(True)
        try:
            minimum = rank._plan_flat_forward(
                output_requests(1, False, "none", True, False, None, False)
            )
        finally:
            memory._CP_FAMILY_CALIBRATION.reset(token)
        minimum.__dict__["_calibration_cp_family_seed"] = True
        rank._update_memory_profile(
            minimum,
            floor + minimum.output_bytes,
            retained_bytes=None,
            caller_phase=True,
            rate_eligible=False,
        )
        rank._available_memory_bytes = lambda sample=None: budget
        repeated = rank._plan_flat_forward(inputs)
        assert not rank._memory_check(repeated).fits
        assert rank._memory_check(repeated).estimated_required_bytes >= required
        assert rank._plan_cost(repeated).required >= required


@pytest.mark.parametrize("grad", [False, True])
def test_report_keeps_minimum_floor_and_optimizer_phase(grad, tmp_path):
    from types import MethodType

    from test_grouped_planner_replay import emitted
    from test_trainer_rank_calibration_trust_region import request
    from test_trainer_rank_cp_request_floor import prepared_rank, record_minimal_caller

    from art.trainer_rank import _impl
    from art.trainer_rank import _planner_misses as reports

    with memory.fixed_slope_memory_profiles(), torch.set_grad_enabled(grad):
        rank = prepared_rank("moe")
        rank._topology_key = MethodType(_impl.TrainerRank._topology_key, rank)
        record_minimal_caller(rank, grad, 100_000_000)
        rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
        plan = rank._plan_flat_forward([request(72, grad=grad)])
        report, costs = emitted(rank, plan, tmp_path)
        assert report["replay_complete"], report["incomplete_reasons"]
        result = reports.replay(report)
        assert result["aggregate"]["matches"]
        assert result["estimates"][0]["required_bytes"] == costs[0].required
