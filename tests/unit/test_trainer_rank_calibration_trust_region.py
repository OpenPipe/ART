"""Calibration-only bounded extrapolation; scalar peaks are CPU fixtures."""

from dataclasses import asdict, replace
from datetime import timedelta
import itertools
import json
import math
from pathlib import Path
import pickle
from unittest.mock import patch

import pytest
from test_trainer_rank_active_memory import _rank
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.context_parallel.types import ParallelTopology
from art.trainer_rank import ForwardInput, ForwardOutput, TrainerRankMemoryError, _impl
from art.trainer_rank import _memory as memory


def request(n, *, grad=False):
    return ForwardInput(
        input_tokens=torch.arange(n), target_tokens=torch.arange(n), no_grad=not grad
    )


def observe(rank, n, peak, *, grad=False, caller=True, eligible=True):
    plan = rank._plan_flat_forward([request(n, grad=grad)])
    rank._update_memory_profile(
        plan,
        peak + plan.output_bytes,
        retained_bytes=None,
        caller_phase=caller,
        rate_eligible=eligible,
    )
    return plan


def harness(monkeypatch, *, budget=15_000_000_000):
    rank = _rank()
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda sample=None: budget)
    return rank


def seed(rank):
    # Rounded compute peaks: the longer measured rung is 4.1 GB; the
    # unobserved next rung's 8.7 GB is the held-out check, never fitted.
    observe(rank, 8192, 1_300_000_000)
    return observe(rank, 32768, 4_100_000_000)


def test_increment_margin_admits_held_out_65k_and_exposes_intent(monkeypatch):
    rank = harness(monkeypatch)
    with memory.fixed_slope_memory_profiles():
        seed(rank)
        with rank.calibration_rung([request(65536)], no_grad=True) as rung:
            d = rung.decision
            assert d.fits and d.eligible and d.extrapolated
            assert d.measured_base_bytes is not None
            assert d.slope_bytes_per_token is not None
            assert d.predicted_peak_bytes is not None
            assert d.observed_rungs == 2
            assert d.candidate_packed_tokens == 65536
            assert d.largest_observed_packed_tokens == 32768
            assert d.measured_base_bytes == 4_100_000_000
            assert d.slope_bytes_per_token == math.ceil(4_100_000_000 / 32768)
            assert d.delta_tokens == 32768 and d.margin == 1.5
            assert d.predicted_peak_bytes == (
                d.measured_base_bytes
                + math.ceil(d.margin * d.slope_bytes_per_token * d.delta_tokens)
                + d.output_bytes
            )
            assert d.predicted_peak_bytes >= 8_700_000_000
            assert (
                d.estimated_required_bytes >= d.predicted_peak_bytes * d.safety_factor
            )
            assert d.estimated_required_bytes < d.available_bytes
            assert len(d.signature_id) == 64
            assert json.loads(json.dumps(asdict(d))) == asdict(d)
            # Fixed resident weights are outside this incremental budget. A
            # hypothetical 70 GB resident allocation is never multiplied by 1.5.
            assert 70_000_000_000 + d.predicted_peak_bytes < 85_000_000_000


@pytest.mark.parametrize("n,admit", [(65536, True), (65537, False), (81920, False)])
def test_step_cannot_exceed_twice_observed_extent(monkeypatch, n, admit):
    rank = harness(monkeypatch)
    with memory.fixed_slope_memory_profiles():
        seed(rank)
        with rank.calibration_rung([request(n)], no_grad=True) as rung:
            assert rung.decision.fits is admit
            assert rung.decision.eligible is admit
            if not admit:
                with pytest.raises(TrainerRankMemoryError, match="refused"):
                    rung.forward_micro_batches()


@pytest.mark.parametrize(
    "peaks", [(), ((32768, 4_100_000_000),), ((32768, 1_000_000_000),) * 2]
)
def test_two_distinct_observed_rungs_required(monkeypatch, peaks):
    rank = harness(monkeypatch)
    with memory.fixed_slope_memory_profiles():
        for n, peak in peaks:
            observe(rank, n, peak)
        with rank.calibration_rung([request(65536)], no_grad=True) as rung:
            assert not rung.decision.eligible and not rung.decision.fits


def test_superlinear_last_two_rungs_stop_even_when_hull_prunes_middle(monkeypatch):
    rank = harness(monkeypatch, budget=100_000_000_000)
    with memory.fixed_slope_memory_profiles():
        observe(rank, 8192, 100_000_000)
        middle = observe(rank, 16384, 200_000_000)
        last = observe(rank, 32768, 1_000_000_000)
        assert len(rank._memory_profiles[last.signature].compute_observations) == 2
        assert middle.packed_tokens == 16384
        with rank.calibration_rung([request(65536)], no_grad=True) as rung:
            assert rung.decision.observed_rungs == 3
            assert not rung.decision.eligible and not rung.decision.fits


@pytest.mark.parametrize("ratio,admit", [(1.25, True), (1.250000002, False)])
def test_superlinear_limit_is_strictly_greater_than_25_percent(
    monkeypatch, ratio, admit
):
    rank = harness(monkeypatch, budget=100_000_000_000)
    with memory.fixed_slope_memory_profiles():
        # At the boundary: marginal = 1.25 * (1 GB / 32768).
        observe(rank, 16384, 1_000_000_000 - round(500_000_000 * ratio))
        observe(rank, 32768, 1_000_000_000)
        with rank.calibration_rung([request(65536)], no_grad=True) as rung:
            assert rung.decision.eligible is admit


def test_ordinary_admission_never_receives_calibration_override(monkeypatch):
    rank = harness(monkeypatch)
    with memory.fixed_slope_memory_profiles():
        plan = seed(rank)
        future = rank._plan_flat_forward([request(65536)])
        normal = rank._memory_check(future)
        profile_bytes = pickle.dumps(rank._memory_profiles)
        estimator = rank._estimate_required_memory_bytes_from_values
        overrides = []

        def record(**kwargs):
            assert ("_calibration_compute_bytes" in kwargs) == (
                kwargs.get("_calibration_compute_bytes") is not None
            )
            overrides.append(kwargs.get("_calibration_compute_bytes"))
            return estimator(**kwargs)

        monkeypatch.setattr(rank, "_estimate_required_memory_bytes_from_values", record)
        with rank.calibration_rung([request(65536)], no_grad=True) as rung:
            assert rung.decision.extrapolated
            assert rank._memory_check(future) == normal
            assert rank._memory_check(plan).fits
        assert rank._memory_check(future) == normal
        assert overrides[0] is not None and overrides[1:] == [None, None, None]
        assert pickle.dumps(rank._memory_profiles) == profile_bytes


def test_preparation_executes_nothing_and_execution_uses_exact_plan_once(monkeypatch):
    rank = harness(monkeypatch)
    executed = []

    def run(plan, **kwargs):
        executed.append((plan, kwargs["check"]))
        return [ForwardOutput(torch.zeros(plan.packed_tokens), None, None, None)], None

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", run)
    with memory.fixed_slope_memory_profiles():
        seed(rank)
        with rank.calibration_rung([request(65536)], no_grad=True) as rung:
            assert not executed
            intent = asdict(rung.decision)  # Available before execution.
            monkeypatch.setattr(
                rank,
                "_select_next_micro_batch",
                lambda *a, **k: pytest.fail("replanned a prepared rung"),
            )
            batches = list(rung.forward_micro_batches())
            assert len(batches) == len(executed) == 1
            assert batches[0].stats.packed_tokens == intent["candidate_packed_tokens"]
            assert (
                batches[0].stats.estimated_required_bytes
                == intent["estimated_required_bytes"]
            )
            assert executed[0][0] is rung._candidate.plan
            with pytest.raises(RuntimeError, match="already executed"):
                rung.forward_micro_batches()
        with pytest.raises(RuntimeError, match="expired"):
            rung.forward_micro_batches()


def test_observed_rung_keeps_normal_admission(monkeypatch):
    rank = harness(monkeypatch)
    with memory.fixed_slope_memory_profiles():
        plan = seed(rank)
        normal = rank._memory_check(plan)
        with rank.calibration_rung([request(32768)], no_grad=True) as rung:
            assert not rung.decision.extrapolated
            assert rung.decision.predicted_peak_bytes is None
            assert (
                rung.decision.estimated_required_bytes
                == normal.estimated_required_bytes
            )


def test_deferred_iterator_cannot_execute_outside_calibration_scope(monkeypatch):
    rank = harness(monkeypatch)
    with memory.fixed_slope_memory_profiles():
        seed(rank)
        with rank.calibration_rung([request(65536)], no_grad=True) as rung:
            iterator = rung.forward_micro_batches()
        with pytest.raises(RuntimeError, match="expired"):
            next(iterator)


def test_existing_oversized_opt_in_does_not_override_calibration_limits(monkeypatch):
    rank = harness(monkeypatch)
    rank._allow_oversized_batches = True
    with memory.fixed_slope_memory_profiles():
        seed(rank)
        with rank.calibration_rung([request(81920)], no_grad=True) as rung:
            assert not rung.decision.fits
            with pytest.raises(TrainerRankMemoryError, match="refused"):
                rung.forward_micro_batches()


def test_complete_observed_rungs_advance_the_ladder_and_cover_held_out_peaks(
    monkeypatch,
):
    rank = harness(monkeypatch, budget=71_000_000_000)
    with memory.fixed_slope_memory_profiles():
        seed(rank)
        for n, measured in (
            (65536, 8_700_000_000),
            (131072, 18_200_000_000),
            (262144, 37_200_000_000),
        ):
            with rank.calibration_rung([request(n)], no_grad=True) as rung:
                assert rung.decision.fits
                assert rung.decision.predicted_peak_bytes is not None
                assert rung.decision.predicted_peak_bytes >= measured
                assert rung.decision.estimated_required_bytes >= measured
            observe(rank, n, measured)


def test_cp8_caller_fixture_discount_is_confined_to_calibration():
    from test_moe_caller_cp_coverage import add, plan, required, workload
    from test_moe_long_root_caller import memory_rank

    rank = memory_rank()
    # Small analytic floors isolate the measured compute law; production's
    # analytic floors still run through the identical estimator below.
    rank._checkpoint_memory_floor = lambda *args: (0, 0)
    with memory.fixed_slope_memory_profiles():
        for n, peak in ((8192, 1_300_000_000), (32768, 4_100_000_000)):
            add(rank, plan(8, n, workload(8, n)), peak)
        candidate = plan(8, 65536, workload(8, 65536))
        normal = required(rank, candidate)
        estimate = memory._calibration_estimate(rank, candidate)
        assert estimate.eligible and estimate.extrapolated
        extrapolated = memory._estimate_required_memory_bytes_from_values(
            rank,
            packed_tokens=candidate.packed_tokens,
            logical_tokens=candidate.active_logical_tokens,
            output_bytes=candidate.output_bytes,
            signature=candidate.signature,
            group_rows=((8192, True),),
            caller_cp_workload=candidate.caller_cp_workload,
            _calibration_compute_bytes=estimate.peak - candidate.output_bytes,
        )
        assert 8_700_000_000 <= extrapolated < 15_000_000_000 < normal
        assert required(rank, candidate) == normal


def test_failed_update_rolls_back_the_live_rung_journal_too(monkeypatch):
    rank = harness(monkeypatch)
    with memory.fixed_slope_memory_profiles():
        plan = seed(rank)
        profiles = dict(rank._memory_profiles)
        journal = dict(rank._calibration_memory_rungs)

        def fail(*args, **kwargs):
            rank._update_memory_profile(
                plan, 9_000_000_000, retained_bytes=None, caller_phase=True
            )
            raise ValueError("sample unavailable")

        monkeypatch.setattr(rank, "_update_peak_memory_profile", fail)
        assert not memory._try_update_caller_memory_profile(rank, plan, None, None, 0)
        assert rank._memory_profiles == profiles
        assert rank._calibration_memory_rungs == journal


def test_calibration_keeps_an_explicit_legacy_rate_floor(monkeypatch):
    rank = harness(monkeypatch)
    with memory.fixed_slope_memory_profiles():
        plan = seed(rank)
        rank._memory_profiles[plan.signature] = replace(
            rank._memory_profiles[plan.signature], legacy_bytes_per_token=1_000_000
        )
        seed(rank)  # Two fresh complete observations after profile replacement.
        with rank.calibration_rung([request(65536)], no_grad=True) as rung:
            assert rung.decision.eligible and rung.decision.extrapolated
            assert not rung.decision.fits
            assert rung.decision.estimated_required_bytes >= 65536 * 1_000_000 * 1.1


@pytest.mark.parametrize(
    "variant",
    ["different_signature", "ineligible", "forward_only", "legacy", "restored"],
)
def test_unqualified_observations_do_not_enable_extrapolation(monkeypatch, variant):
    rank = harness(monkeypatch)
    with memory.fixed_slope_memory_profiles(variant != "legacy"):
        if variant == "different_signature":
            observe(rank, 8192, 1_300_000_000, grad=True)
            observe(rank, 32768, 4_100_000_000)
        elif variant == "ineligible":
            observe(rank, 8192, 1_300_000_000, eligible=False)
            observe(rank, 32768, 4_100_000_000)
        elif variant == "forward_only":
            observe(rank, 8192, 1_300_000_000, grad=True, caller=False)
            observe(rank, 32768, 4_100_000_000, grad=True, caller=False)
        else:
            seed(rank)
            if variant == "restored":
                rank._memory_profiles = pickle.loads(
                    pickle.dumps(rank._memory_profiles)
                )
        with rank.calibration_rung(
            [request(65536, grad=variant == "forward_only")],
            no_grad=variant != "forward_only",
        ) as rung:
            assert not rung.decision.eligible and not rung.decision.fits


def test_journal_is_bounded_and_order_independent(monkeypatch):
    points = ((8192, 1_300_000_000), (16384, 2_200_000_000), (32768, 4_100_000_000))
    decisions = []
    with memory.fixed_slope_memory_profiles():
        for order in itertools.permutations(points):
            rank = harness(monkeypatch)
            for n, peak in order:
                observe(rank, n, peak)
            with rank.calibration_rung([request(65536)], no_grad=True) as rung:
                decisions.append(rung.decision)
        assert all(d == decisions[0] for d in decisions)
        for n in range(40000, 40050):
            observe(rank, n, 4_100_000_000 + n)
        assert len(next(iter(rank._calibration_memory_rungs.values()))[1]) == 8


def _agreement_worker(index, directory, variant):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=8),
    )
    rank = _rank()
    rank._dp_rank_and_size = lambda: (0, 1)
    rank._topology_key = lambda: (1, 1, 2, 1)
    rank._topology = lambda: ParallelTopology(cp=2)
    if variant == "different_signature":
        rank._coefficient_version = 7 + index
    rank._plan_group_rows = lambda plan: tuple(
        ((g.packed.tokens.numel() + 1) // 2, g.grad_enabled) for g in plan.groups
    )
    rank._plan_retained_tokens = lambda plan: (plan.packed_tokens + 1) // 2
    rank._plan_head_workspace_bytes = lambda plan: 0
    rank._plan_hybridep_growth_bytes = lambda plan: 0
    rank._available_memory_bytes = lambda sample=None: (
        12_000_000_000 if variant == "refuse" and index == 0 else 20_000_000_000
    )
    operations = []
    original, original_gather = dist.all_reduce, dist.all_gather_object

    def reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
        operations.append((str(op), list(tensor.shape), str(tensor.dtype)))
        return original(tensor, op=op, **kwargs)

    def gather(objects, value, **kwargs):
        operations.append(("ALL_GATHER_OBJECT", [len(objects)], "object"))
        return original_gather(objects, value, **kwargs)

    try:
        with memory.fixed_slope_memory_profiles():
            observe(rank, 8192, 1_300_000_000 + index * 200_000_000)
            observe(
                rank,
                32768,
                4_100_000_000 + index * 1_000_000_000,
                eligible=variant != "unqualified" or index == 0,
            )
            with (
                patch.object(dist, "all_reduce", reduce),
                patch.object(dist, "all_gather_object", gather),
            ):
                plan = rank._plan_flat_forward(
                    [request(65536)], sync_planning_errors=True
                )
                rank._memory_check(plan, sync_across_dp=True, sync_planning_errors=True)
                ordinary = list(operations)
                operations.clear()
                with rank.calibration_rung([request(65536)], no_grad=True) as rung:
                    decision = rung.decision
            Path(directory, f"rank{index}.json").write_text(
                json.dumps(
                    dict(
                        decision=asdict(decision),
                        operations=operations,
                        ordinary=ordinary,
                    )
                )
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "variant,admit",
    [
        ("admit", True),
        ("refuse", False),
        ("unqualified", False),
        ("different_signature", False),
    ],
)
def test_gloo_replicates_decision_and_components_without_another_exchange(
    tmp_path, variant, admit
):
    mp.start_processes(
        _agreement_worker,
        args=(str(tmp_path), variant),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    a, b = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    assert a["decision"] == b["decision"]
    assert a["decision"]["fits"] is admit
    assert a["operations"] == b["operations"]
    # Existing planning status/checkpoint exchanges and admission MAX/MIN:
    # only the MAX payload grows. There is no additional collective round.
    assert len(a["operations"]) == len(a["ordinary"])
    assert [op[0] for op in a["operations"]] == [op[0] for op in a["ordinary"]]
    changed = [
        (before, after)
        for before, after in zip(a["ordinary"], a["operations"], strict=True)
        if before != after
    ]
    assert len(changed) == 1
    assert changed[0][0][1:] == [[], "torch.float64"]
    assert changed[0][1][1:] == [[29], "torch.float64"]
    if variant == "admit":
        assert a["decision"]["measured_base_bytes"] == 5_100_000_000
        assert a["decision"]["estimated_required_bytes"] > 14_000_000_000
