"""Generic packed rate measurements; synthetic CPU peaks are not GPU evidence."""

from dataclasses import asdict
import pickle

import pytest
from test_trainer_rank_calibration_seed import fake_device
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_cp_request_floor import prepared_rank
import torch

from art.trainer_rank import ForwardInput, ForwardOutput, TrainerRankMemoryError
from art.trainer_rank import _memory as memory
from art.trainer_rank import _planner_misses as reports

PEAK = 1_684_410_368


def packed_requests(count, *, shared=False, length=577, prefix_length=64):
    return [
        ForwardInput(
            input_tokens=torch.cat(
                (
                    torch.arange(prefix_length)
                    if shared
                    else torch.arange(prefix_length) + i * 2048,
                    torch.arange(prefix_length, length) + i * 2048,
                )
            ),
            target_tokens=torch.zeros(length, dtype=torch.long),
            no_grad=False,
        )
        for i in range(count)
    ]


def marker_device(monkeypatch, rank):
    calls, _ = fake_device(monkeypatch, rank, PEAK)
    original = rank._run_flat_plan_with_memory_tracking

    def run(plan, **kwargs):
        original(plan, **kwargs)
        return [
            ForwardOutput(
                torch.zeros(len(item.input_ids), requires_grad=True), None, None, None
            )
            for group in plan.groups
            for item in group.items
        ], 0

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", run)
    return calls


def finish(rank, *, execute=None, pending=lambda: None):
    def complete(rung):
        assert rung.family_id is None  # Ordinary rates, not floor-only records.
        assert rung.decision.fits
        assert len(rung.inputs) == 2 and not rung.no_grad
        assert rung.inputs[0].input_tokens[0] != rung.inputs[1].input_tokens[0]
        for batch in rung.forward_micro_batches():
            sum(o.target_logprobs.sum() for o in batch.outputs).backward()

    return rank.calibrate_packed_target_rates(
        request(1),
        execute=complete if execute is None else execute,
        check_pending=pending,
    )


def test_floor_covered_packs_use_one_generic_measured_rate_scope(monkeypatch):
    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        rank = prepared_rank("moe")
        rank._available_memory_bytes = lambda sample=None: 71_000_000_000
        calls = marker_device(monkeypatch, rank)
        before = dict(rank._memory_profiles)
        plans = [rank._plan_flat_forward(packed_requests(k)) for k in (1, 2, 9)]
        assert (
            len(
                {
                    memory._cp_request_floor_key(p.signature, p.caller_cp_workload)
                    for p in plans
                }
            )
            == 1
        )
        assert len({str(memory._cp_sibling_scope(p.signature)) for p in plans}) == 1
        assert (
            memory._cp_request_floor(
                rank, plans[1].signature, plans[1].caller_cp_workload
            )
            > 0
        )
        assert (
            rank._memory_check(plans[1]).estimated_required_bytes
            >= memory._MEMORY_DEFERRED_BYTES
        )
        # The semantic assertion below is also red on a floor-only runtime.
        if hasattr(rank, "calibrate_packed_target_rates"):
            decisions = finish(rank)
            assert [d.candidate_packed_tokens for d in decisions] == [16384, 8192]
            assert decisions[0].seed and decisions[0].seed_margin == 2
            assert len(calls) == 2
        signatures = set()
        for shared in (False, True):
            for count in range(2, 17):
                inputs = packed_requests(count, shared=shared)
                plan = rank._plan_flat_forward(inputs)
                check = rank._memory_check(plan)
                assert (
                    check.fits
                    and check.estimated_required_bytes < memory._MEMORY_DEFERRED_BYTES
                )
                signatures.add(plan.signature)
                # Modeled marker requirement, including output and a measured
                # complete interval. This is not a measured packed GPU bound.
                assert check.estimated_required_bytes >= PEAK + plan.output_bytes
                batches = list(rank.forward_micro_batches(inputs))
                assert len(batches) == 1
                assert tuple(batches[0].indices) == tuple(range(count))
        assert len(signatures) == 1
        for signature, profile in before.items():
            assert rank._memory_profiles[signature] == profile
        generic = rank._memory_profiles[signatures.pop()]
        assert not isinstance(generic, memory._CallerMemoryProfileV2)
        assert min(dict(generic.rate_observations)) <= 8192
        assert max(dict(generic.rate_observations)) >= 16384
        assert getattr(generic, "cp_request_floors", ()) == ()
        assert reports._memory_profile(asdict(generic)) == generic
        assert pickle.loads(pickle.dumps(generic)) == generic
        executed = len(calls)
        assert finish(rank) == ()
        assert len(calls) == executed


@pytest.mark.parametrize("partial", [False, True])
def test_callback_must_complete_rate_interval(monkeypatch, partial):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank("moe")
        marker_device(monkeypatch, rank)

        def incomplete(rung):
            if partial:
                next(rung.forward_micro_batches())

        with pytest.raises(RuntimeError, match="completed rate measurement"):
            finish(rank, execute=incomplete)
        plan = rank._plan_flat_forward(packed_requests(2))
        assert (
            rank._memory_check(plan).estimated_required_bytes
            >= memory._MEMORY_DEFERRED_BYTES
        )


def test_pending_intent_blocks_before_profile_skip(monkeypatch):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank("moe")
        calls = marker_device(monkeypatch, rank)
        finish(rank)
        executed = len(calls)

        def pending():
            raise RuntimeError("unresolved rate intent")

        with pytest.raises(RuntimeError, match="unresolved rate intent"):
            finish(rank, pending=pending)
        assert len(calls) == executed


def test_refusal_does_not_execute_or_learn(monkeypatch):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank("moe")
        calls = marker_device(monkeypatch, rank)
        rank._available_memory_bytes = lambda sample=None: 1
        before = dict(rank._memory_profiles)
        with pytest.raises(TrainerRankMemoryError):
            finish(rank)
        assert not calls and rank._memory_profiles == before


def test_dense_finishing_has_no_trials_or_votes(monkeypatch):
    rank = prepared_rank("dense")
    before = dict(rank._memory_profiles)

    def unexpected(*args, **kwargs):
        raise AssertionError("dense must retain its existing path")

    monkeypatch.setattr(rank, "_recovery_reduce", unexpected)
    assert finish(rank, execute=unexpected, pending=unexpected) == ()
    assert rank._memory_profiles == before


def test_long_shared_packs_use_startup_rates_without_serving_measurements(monkeypatch):
    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        rank = prepared_rank("moe")
        calls = marker_device(monkeypatch, rank)
        finish(rank)
        before = dict(rank._memory_profiles)
        executed = len(calls)
        for length in (16384, 18432, 20480):
            for count in (9, 16):
                inputs = packed_requests(
                    count,
                    shared=True,
                    length=length,
                    prefix_length=length - 2048,
                )
                plan = rank._plan_flat_forward(inputs)
                check = rank._memory_check(plan)
                assert (
                    check.fits
                    and check.estimated_required_bytes < memory._MEMORY_DEFERRED_BYTES
                )
                assert (
                    check.estimated_required_bytes
                    >= PEAK * plan.packed_tokens // 8192 + plan.output_bytes
                )
                selected, selected_check = rank._find_admissible_forward(
                    inputs,
                    checkpoint=None,
                    refusal_prefix="synthetic packed calibration",
                )
                assert not hasattr(selected, "subforwards") and selected_check.fits
                assert selected.request_count == count
        assert rank._memory_profiles == before and len(calls) == executed
