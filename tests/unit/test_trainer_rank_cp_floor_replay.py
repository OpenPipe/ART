"""Local review witnesses; all peak readings are synthetic CPU fixtures."""

from dataclasses import replace
from types import MethodType

import pytest
from test_grouped_planner_replay import emitted
from test_trainer_rank_cp_output_families import anchors, native_rank, output_requests
import torch

from art.trainer_rank import ForwardInput, _impl
from art.trainer_rank import _memory as memory
from art.trainer_rank import _planner_misses as reports


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("tokens", [1, 4, 72])
def test_unmeasured_family_after_collection_replays(kind, tokens, tmp_path):
    rank = native_rank(kind)
    rank._topology_key = MethodType(_impl.TrainerRank._topology_key, rank)
    with memory.fixed_slope_memory_profiles():
        anchors(rank, True)
    with torch.no_grad():
        rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
        plan = rank._plan_flat_forward(
            output_requests(tokens, False, "none", True, False, None, False)
        )
        assert plan.signature not in rank._memory_profiles
        report, costs = emitted(rank, plan, tmp_path)
        assert report["replay_complete"], report["incomplete_reasons"]
        replayed = reports.replay(report)
        print(
            {
                "kind": kind,
                "tokens": tokens,
                "live": costs[0].required,
                "replayed": replayed["estimates"][0],
                "pool": report["replay"]["memory_replay"].get(
                    "fixed_slope_profile_pool"
                ),
            }
        )
        assert replayed["aggregate"]["matches"]


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_inactive_child_has_zero_cost(kind):
    rank = native_rank(kind)
    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        anchors(rank, False)
        memory._FIXED_SLOPE_PROFILES.set(False)
        rank._available_memory_bytes = lambda sample=None: 300_000_000
        active = output_requests(272, False, "none", True, False, None, False)[0]
        inputs = [active, replace(active, input_tokens=active.input_tokens + 1024)]
        inactive = ForwardInput(input_tokens=torch.arange(8192), no_grad=True)
        zero = rank._plan_flat_forward([inactive])
        result_before = rank._find_admissible_forward(
            inputs, checkpoint=None, refusal_prefix="review"
        )
        result_after = rank._find_admissible_forward(
            inputs + [inactive], checkpoint=None, refusal_prefix="review"
        )
        print(
            {
                "kind": kind,
                "inactive_tokens": zero.packed_tokens,
                "inactive_check": rank._memory_check(zero),
                "inactive_cost": rank._plan_cost(zero),
                "before": type(result_before).__name__,
                "after": type(result_after).__name__,
                "after_check": result_after.check
                if isinstance(result_after, _impl._ForwardRefusal)
                else result_after[1],
            }
        )
        assert rank._plan_cost(zero).required == 0


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_inactive_request_does_not_break_covered_split(monkeypatch, kind):
    from test_trainer_rank_calibration_seed import fake_device
    from test_trainer_rank_calibration_trust_region import request

    rank = native_rank(kind)
    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        anchors(rank, True)
        fake_device(monkeypatch, rank, 100_000_000)
        with rank.calibration_rung(
            [request(2048, grad=True)],
            no_grad=False,
            seed_uncovered_cp_family=True,
        ) as rung:
            assert rung.decision.fits
            list(rung.forward_micro_batches())
        memory._FIXED_SLOPE_PROFILES.set(False)
        active = request(272, grad=True)
        inputs = [active, replace(active, input_tokens=active.input_tokens + 1024)]
        inactive = ForwardInput(input_tokens=torch.arange(8192), no_grad=True)
        costs = [rank._plan_cost(rank._plan_flat_forward([r])) for r in inputs]
        zero = rank._plan_cost(rank._plan_flat_forward([inactive]))
        base = rank._split_required_memory(costs)
        augmented = rank._split_required_memory(costs + [zero])
        budget = base + max(1000, (augmented - base) // 2)
        rank._available_memory_bytes = lambda sample=None: budget
        before = rank._find_admissible_forward(
            inputs, checkpoint=None, refusal_prefix="review"
        )
        after = rank._find_admissible_forward(
            inputs + [inactive], checkpoint=None, refusal_prefix="review"
        )
        print(
            {
                "kind": kind,
                "base": base,
                "with_inactive": augmented,
                "budget": budget,
                "before": type(before).__name__,
                "after": type(after).__name__,
                "inactive_cost": zero,
            }
        )
        assert not isinstance(before, _impl._ForwardRefusal)
        assert not isinstance(after, _impl._ForwardRefusal)
        _, base_check = rank._admit_split_rung(
            ((0,), (1,)), inputs, [r.input_tokens for r in inputs], checkpoint=None
        )
        assert base_check.fits
        _, augmented_check = rank._admit_split_rung(
            ((0,), (1,), (2,)),
            inputs + [inactive],
            [r.input_tokens for r in inputs + [inactive]],
            checkpoint=None,
        )
        print(
            {
                "kind": kind,
                "base_check": base_check,
                "with_inactive_check": augmented_check,
            }
        )
        assert augmented_check.fits


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("sibling", [False, True])
def test_sibling_max_and_floorless_refusal_replay_after_collection(
    monkeypatch, kind, sibling, tmp_path
):
    from test_trainer_rank_cp_request_floor import record_minimal_caller

    rank = native_rank(kind)
    rank._topology_key = MethodType(_impl.TrainerRank._topology_key, rank)
    with memory.fixed_slope_memory_profiles():
        anchors(rank, True, include_floor=False)
        if sibling:
            record_minimal_caller(rank, True, 333_333_333)
    with torch.no_grad():
        rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
        plan = rank._plan_flat_forward(
            output_requests(1, False, "none", True, False, None, False)
        )
        own = memory._cp_measured_request_floor(
            rank, plan.signature, plan.caller_cp_workload
        )
        assert own == 0
        floor = memory._cp_request_floor(rank, plan.signature, plan.caller_cp_workload)
        assert floor == (333_333_333 if sibling else memory._MEMORY_DEFERRED_BYTES)
        report, _ = emitted(rank, plan, tmp_path)
        assert report["replay_complete"], report["incomplete_reasons"]
        assert report["replay"]["memory_replay"]["fixed_slope_profile_pool"]
        assert reports.replay(report)["aggregate"]["matches"]
        if not sibling:
            assert not rank._memory_check(plan).fits


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("collect", [False, True])
def test_sibling_fixed_phases_skip_inactive_children(kind, collect):
    from test_trainer_rank_cp_request_floor import prepared_rank, record_minimal_caller

    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank(kind, include_floor=False)
        record_minimal_caller(rank, False, 333_333_333)
    with memory.fixed_slope_memory_profiles(collect), torch.no_grad():
        active = output_requests(1, False, "none", True, False, None, False)[0]
        inactive = ForwardInput(input_tokens=torch.arange(8192), no_grad=True)
        child = rank._plan_flat_forward([active])
        zero = rank._plan_flat_forward([inactive])
        assert (
            memory._cp_request_floor(rank, child.signature, child.caller_cp_workload)
            == 333_333_333
        )
        assert rank._plan_cost(child).required >= 333_333_333
        phases = memory._cp_phase_bounds(
            rank,
            zero.signature,
            zero.output_bytes,
            zero.caller_cp_workload,
            packed_tokens=zero.packed_tokens,
        )
        assert phases == (0, 0, 0)
        assert rank._plan_cost(zero).required == 0
        costs = [rank._plan_cost(child)]
        assert rank._split_required_memory(
            costs + [rank._plan_cost(zero)]
        ) == rank._split_required_memory(costs)
