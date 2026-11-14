from unittest.mock import patch

import pytest
from test_trainer_rank_calibration_seed import fake_device, metadata_rank
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_cp_dominance import seed
from test_trainer_rank_subfloor_cp import requests, short_kernel

from art.trainer_rank import CalibrationCoverageViolation, _impl
from art.trainer_rank import _memory as memory


def test_unrelated_revocation_must_not_prune_cold_split(monkeypatch):
    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        revoked = rank._plan_flat_forward([request(72, grad=True)])
        profile = rank._memory_profiles[revoked.signature]
        covering = profile.covering_cp_observation(72, revoked.caller_cp_workload)
        fake_device(monkeypatch, rank, covering[1] + revoked.output_bytes + 1_000_000)
        with rank.calibration_rung([request(72, grad=True)], no_grad=False) as rung:
            with pytest.raises(CalibrationCoverageViolation):
                list(rung.forward_micro_batches())

        with short_kernel():
            inputs = requests([256, 256], True)
            children = [rank._plan_flat_forward([r]) for r in inputs]
            profile = rank._memory_profiles[children[0].signature]
            assert all(
                memory._uncovered_subfloor_cp(
                    profile, p.signature, p.packed_tokens, p.caller_cp_workload
                )
                for p in children
            )
            costs = [rank._plan_cost(p) for p in children]
            exact_required = rank._split_required_memory(costs)
            budget = 300_000_000
            assert exact_required < budget
            rank._available_memory_bytes = lambda sample=None: budget
            assert not rank._memory_check(rank._plan_flat_forward(inputs)).fits
            lower = [
                rank._split_chunk_lower_cost([r], [r.input_tokens], checkpoint=None)
                for r in inputs
            ]
            lower_required = rank._split_required_memory(lower)
            result = rank._find_admissible_forward(
                inputs, checkpoint=None, refusal_prefix="review"
            )
            # Exact child pricing preserves all current fail-closed checks.
            with patch.object(
                rank,
                "_split_chunk_lower_cost",
                side_effect=lambda rs, rows, **kw: rank._plan_cost(
                    rank._plan_flat_forward(rs, checkpoint=kw["checkpoint"])
                ),
            ):
                exact_result = rank._find_admissible_forward(
                    inputs, checkpoint=None, refusal_prefix="review"
                )
            print(
                dict(
                    exact_required=exact_required,
                    lower_required=lower_required,
                    budget=budget,
                    result=type(result).__name__,
                    exact_result=type(exact_result).__name__,
                )
            )
            assert isinstance(exact_result, tuple)
            assert isinstance(exact_result[0], _impl._SplitForwardPlan)
            assert not isinstance(result, _impl._ForwardRefusal)


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("growth", [0, 17_000_000])
def test_cold_split_matches_no_profile_for_all_cost_fields(
    monkeypatch, kind, grad, collect, growth
):
    with memory.fixed_slope_memory_profiles():
        rank = metadata_rank(kind)
        rank._available_memory_bytes = lambda sample=None: 71_000_000_000
        fake_device(monkeypatch, rank, 2_500_000_000)
        with rank.calibration_rung(
            [request(16384, grad=grad)], no_grad=not grad
        ) as rung:
            list(rung.forward_micro_batches())
        with short_kernel(), memory.fixed_slope_memory_profiles(collect):
            child = rank._plan_flat_forward([request(72, grad=grad)])
            profile = rank._memory_profiles[child.signature]
            assert memory._uncovered_subfloor_cp(
                profile, child.signature, child.packed_tokens, child.caller_cp_workload
            )
            monkeypatch.setattr(rank, "_plan_hybridep_growth_bytes", lambda p: growth)
            warm_cost = rank._plan_cost(child)
            with (
                patch.object(rank, "_memory_profiles", {}),
                memory.fixed_slope_memory_profiles(),
            ):
                cold_cost = rank._plan_cost(child)
                cold_single = rank._memory_check(child).estimated_required_bytes
            assert warm_cost == cold_cost
            assert rank._split_required_memory([warm_cost, warm_cost]) >= cold_single


@pytest.mark.parametrize("length", [72, 32768], ids=["revoked", "unknown_extent"])
def test_nonbinding_preview_still_requires_exact_family_admission(monkeypatch, length):
    with memory.fixed_slope_memory_profiles():
        rank = seed(monkeypatch)
        revoked = rank._plan_flat_forward([request(72, grad=True)])
        profile = rank._memory_profiles[revoked.signature]
        point = profile.covering_cp_observation(72, revoked.caller_cp_workload)
        assert point is not None
        fake_device(monkeypatch, rank, point[1] + revoked.output_bytes + 1_000_000)
        with rank.calibration_rung([request(72, grad=True)], no_grad=False) as rung:
            with pytest.raises(CalibrationCoverageViolation):
                list(rung.forward_micro_batches())
        inputs = requests([length, length], True)
        children = [rank._plan_flat_forward([r]) for r in inputs]
        rank._available_memory_bytes = lambda sample=None: 300_000_000
        assert all(p.caller_cp_workload is not None for p in children)
        assert all(not rank._memory_check(p).fits for p in children)
        with patch.object(rank, "_plan_cost", wraps=rank._plan_cost) as exact:
            split, check = rank._admit_split_rung(
                ((0,), (1,)), inputs, [r.input_tokens for r in inputs], checkpoint=None
            )
        assert split is None and not check.fits
        assert check.estimated_required_bytes >= memory._MEMORY_DEFERRED_BYTES
        assert exact.call_count >= 2
