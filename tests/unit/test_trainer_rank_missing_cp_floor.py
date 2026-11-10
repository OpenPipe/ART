"""Missing CP request measurements retain conservative fixed admission costs."""

from contextlib import nullcontext

import pytest
from test_trainer_rank_calibration_seed import fake_device
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_cp_output_families import output_modes, output_requests
from test_trainer_rank_cp_request_floor import prepared_rank
from test_trainer_rank_subfloor_cp import short_kernel
import torch

from art.trainer_rank import _memory as memory


def calibrate_targets(monkeypatch, rank):
    for grad in (True, False):
        with torch.set_grad_enabled(grad):
            fake_device(monkeypatch, rank, 100_000_000)
            with rank.calibration_rung(
                [request(1, grad=grad)],
                no_grad=not grad,
                seed_uncovered_cp_family=True,
            ) as rung:
                assert rung.decision.fits
                list(rung.forward_micro_batches())
            plan = rank._plan_flat_forward([request(1, grad=grad)])
            assert (
                memory._cp_request_floor(rank, plan.signature, plan.caller_cp_workload)
                >= 99_000_000
            )


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_hidden_request_after_other_minimum_trials_has_a_fixed_bound(monkeypatch, kind):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank(kind)
        calibrate_targets(monkeypatch, rank)
        with torch.no_grad():
            for n in (1, 4, 72):
                plan = rank._plan_flat_forward(
                    output_requests(n, False, "none", True, False, None, False)
                )
                measured = memory._cp_request_floor(
                    rank, plan.signature, plan.caller_cp_workload
                )
                phases = memory._cp_fixed_phase_bytes(
                    rank,
                    plan.signature,
                    plan.output_bytes,
                    plan.caller_cp_workload,
                    packed_tokens=plan.packed_tokens,
                )
                check = rank._memory_check(plan)
                # Another output class cannot manufacture a hidden measurement.
                assert measured == 0
                print(
                    dict(
                        kind=kind,
                        tokens=n,
                        measured_floor=measured,
                        fixed_phases=phases,
                        required=check.estimated_required_bytes,
                    )
                )
                assert not check.fits or phases[0] > 0
                assert check.estimated_required_bytes >= phases[0]


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("grad", [False, True])
def test_every_cp_output_family_has_a_fixed_bound_or_refuses(monkeypatch, kind, grad):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank(kind)
        calibrate_targets(monkeypatch, rank)
        rates = {
            key: profile.rate_observations
            for key, profile in rank._memory_profiles.items()
        }
        # Local/chained recurrence, the short-signature boundary, and both
        # ordinary and short-probe attention kernels must retain a fixed cost.
        for collect in (True, False):
            with (
                memory.fixed_slope_memory_profiles(collect),
                torch.set_grad_enabled(grad),
            ):
                for alternate in (False, True):
                    with short_kernel() if alternate else nullcontext():
                        for mode in output_modes(kind):
                            for n in (1, 72, 4095, 4096, 8191):
                                plan = rank._plan_flat_forward(
                                    output_requests(
                                        n, grad, *mode, layers=rank._num_layers
                                    )
                                )
                                floor, _ = memory._cp_fixed_phase_bytes(
                                    rank,
                                    plan.signature,
                                    plan.output_bytes,
                                    plan.caller_cp_workload,
                                    packed_tokens=plan.packed_tokens,
                                )
                                check = rank._memory_check(plan)
                                assert floor > 0 or not check.fits, (
                                    kind,
                                    grad,
                                    collect,
                                    alternate,
                                    mode,
                                    n,
                                )
                                assert check.estimated_required_bytes >= floor
        assert {
            key: p.rate_observations for key, p in rank._memory_profiles.items()
        } == rates
