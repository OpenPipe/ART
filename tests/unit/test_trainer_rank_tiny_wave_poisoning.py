"""Tiny-wave regressions using real ART packing, search, learning and admission.

CP8 topology/row metadata and scalar peaks are injected on CPU; no model or GPU
peak is measured. Flag-off numbers document the current residual poisoning,
not desired behavior. Unanchored v2 admission needs qualifying rate evidence;
these cases without references defer large requests rather than price them fixed-only.
"""

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import (
    ForwardInput,
    ForwardOutput,
    TrainerRank,
    TrainerRankMemoryError,
    _impl,
    _memory,
    _memory_model,
)

RATE = 85_000
WORKSPACE = 300_000_000
# These are deliberately inflated flag-off admission operands after one tiny
# wave, ordered as 8K, 32K, shared-prompt plan. Each row starts a fresh ladder.
OFF_WARM_BEFORE = (2_085_988_044, 8_343_952_179, 1_845_610_516)
OFF_WARM_AFTER = {
    8: (338_685_952_000, 1_354_743_808_000, 299_657_688_000),
    16: (169_725_952_000, 678_903_808_000, 150_167_688_000),
    32: (85_245_952_000, 340_983_808_000, 75_422_688_000),
}


def _requests(lengths):
    return [
        ForwardInput(
            input_tokens=torch.arange(1, n + 1),
            target_tokens=torch.zeros(n, dtype=torch.long),
            no_grad=True,
        )
        for n in lengths
    ]


@dataclass
class _Harness:
    rank: TrainerRank
    executed: list[_impl._FlatForwardPlan]
    observations: list[tuple[_impl._FlatForwardPlan, int]]
    peak: int = 0

    def plan(self, lengths):
        return self.rank._plan_flat_forward(_requests(lengths), checkpoint=None)

    def run(self, lengths):
        return list(
            self.rank.forward_micro_batches(
                _requests(lengths), checkpoint=None, no_grad=True
            )
        )

    def required(self, plan):
        return self.rank._memory_check(plan).estimated_required_bytes


def _harness(monkeypatch):
    runtime = SimpleNamespace(
        model=[torch.nn.Linear(1, 1).bfloat16()],
        optimizer=None,
        provider=SimpleNamespace(
            hidden_size=2048,
            num_layers=40,
            context_parallel_size=8,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        ),
        model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
    )
    rank = TrainerRank(cast(Any, runtime))
    harness = _Harness(rank, [], [])
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 1, 8, 1))
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(
        rank,
        "_plan_group_rows",
        lambda plan: tuple((plan.packed_tokens, g.grad_enabled) for g in plan.groups),
    )
    monkeypatch.setattr(rank, "_plan_head_workspace_bytes", lambda plan: 0)
    monkeypatch.setattr(rank, "_plan_retained_tokens", lambda plan: plan.packed_tokens)
    monkeypatch.setattr(rank, "_plan_hybridep_growth_bytes", lambda plan: 0)
    monkeypatch.setattr(
        rank, "_available_memory_bytes", lambda sample=None: 20_000_000_000
    )

    def execute(plan: _impl._FlatForwardPlan, **kwargs):
        harness.executed.append(plan)
        forward_peak = plan.output_bytes + 240_000_000 + RATE * plan.packed_tokens
        # The caller reading includes outputs. ART must subtract their bytes
        # when fitting compute, and charge them explicitly during admission.
        harness.peak = WORKSPACE + RATE * plan.packed_tokens
        harness.observations.extend(((plan, forward_peak), (plan, harness.peak)))
        rank._peak_resets += 1
        caller_peak = harness.peak
        harness.peak = forward_peak
        rank._update_peak_memory_profile(plan, 0, retained_after=plan.output_bytes)
        harness.peak = caller_peak
        return [
            ForwardOutput(torch.zeros(len(item.request.input_tokens)), None, None, None)
            for group in plan.groups
            for item in group.items
        ], 0

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", execute)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: harness.peak)
    return harness


def _ladder(harness, lengths=(512, 2048, 8192, 32768)):
    for n in lengths:
        plan = harness.plan([n])
        peak = plan.output_bytes + WORKSPACE + RATE * plan.packed_tokens
        for caller in (False, True):
            harness.rank._update_memory_profile(
                plan,
                peak,
                retained_bytes=None if caller else plan.output_bytes,
                caller_phase=caller,
            )
            harness.observations.append((plan, peak))


def _shared_prompt_plan(harness):
    # Sixteen scored continuations share a 240-token prompt and have distinct
    # 213-token tails: the small, reproducible unit of a shared-prompt RL wave.
    requests = [
        ForwardInput(
            input_tokens=torch.cat(
                (torch.full((240,), 1000), torch.arange(213) + 10_000 + 1000 * i)
            ),
            target_tokens=torch.zeros(453, dtype=torch.long),
            no_grad=True,
        )
        for i in range(16)
    ]
    plan = harness.rank._plan_flat_forward(requests, checkpoint=None)
    assert (plan.logical_tokens, plan.packed_tokens) == (7248, 3648)
    return plan


def _covered(harness, enabled):
    assert len(harness.rank._memory_profiles) == 1
    for plan, peak in harness.observations:
        assert harness.required(plan) >= peak
        if enabled:
            profile = harness.rank._memory_profiles[plan.signature]
            assert (
                _memory_model.predict(
                    profile.fixed_compute_bytes,
                    profile.compute_slope,
                    plan.packed_tokens,
                )
                + plan.output_bytes
                >= peak
            )


def _fixed_term_only(harness, before, plans, predictions):
    after = harness.rank._memory_profiles[plans[0].signature]
    assert after.compute_slope == before.compute_slope
    increase = after.fixed_compute_bytes - before.fixed_compute_bytes
    assert increase >= 0
    for plan, previous in zip(plans, predictions, strict=True):
        factor = _memory._cold_extrapolation_factor(after, plan.packed_tokens)
        bound = math.ceil(increase * _impl._MEMORY_SAFETY_FACTOR * factor) + 1
        assert previous <= harness.required(plan) <= previous + bound


def _cold_fit_is_only_a_fixed_floor(profile):
    assert profile.compute_slope == (0, 1)
    assert profile.fixed_compute_bytes == 300_254_988
    assert [_memory._profile_compute_peak(profile, n) for n in (8192, 32768)] == [
        profile.fixed_compute_bytes
    ] * 2


@pytest.mark.parametrize("enabled", [False, True])
def test_cold_three_then_8192_preserves_floor_and_documents_cold_guard(
    monkeypatch, enabled
):
    with _memory.fixed_slope_memory_profiles(enabled):
        harness = _harness(monkeypatch)
        large = harness.plan([8192])
        before = harness.required(large)
        with pytest.raises(TrainerRankMemoryError, match="predicted to exceed"):
            harness.run([3, 8192])
        assert [p.packed_tokens for p in harness.executed] == [3]
        assert harness.executed[0].signature == large.signature
        _covered(harness, enabled)
        if enabled:
            profile = harness.rank._memory_profiles[large.signature]
            _cold_fit_is_only_a_fixed_floor(profile)
            assert harness.required(large) >= _memory._MEMORY_DEFERRED_BYTES
        else:
            # Documentation of current flag-off behavior, not a desired bound.
            assert (before, harness.required(large)) == (516_810_342, 901_885_952_000)


@pytest.mark.parametrize("enabled", [False, True])
def test_171_shared_three_token_roots_use_three_token_waves(monkeypatch, enabled):
    with _memory.fixed_slope_memory_profiles(enabled):
        harness = _harness(monkeypatch)
        selected = harness.plan([3] * 171)
        assert (selected.logical_tokens, selected.packed_tokens) == (513, 3)
        large = harness.plan([8192])
        before = harness.required(large)
        batches = harness.run([3] * 171)
        assert sum(batch.stats.global_count for batch in batches) == 171
        assert harness.executed and all(p.packed_tokens == 3 for p in harness.executed)
        assert all(p.signature == large.signature for p in harness.executed)
        _covered(harness, enabled)
        if enabled:
            profile = harness.rank._memory_profiles[large.signature]
            _cold_fit_is_only_a_fixed_floor(profile)
            assert harness.required(large) >= _memory._MEMORY_DEFERRED_BYTES
        else:
            # Aggregate logical rows do not prevent a short packed observation.
            assert (before, harness.required(large)) == (516_810_342, 901_883_969_536)


@pytest.mark.parametrize("enabled", [False, True])
def test_256_token_workspace_observation_refuses_250mb(monkeypatch, enabled):
    with _memory.fixed_slope_memory_profiles(enabled):
        harness = _harness(monkeypatch)
        _ladder(harness, (512, 2048, 8192))
        small = harness.plan([256])
        plans = [harness.plan([n]) for n in (8192, 32768)]
        predictions = [harness.required(plan) for plan in plans]
        before = harness.rank._memory_profiles[small.signature]
        small_before = harness.required(small)
        harness.run([256])
        assert [p.packed_tokens for p in harness.executed] == [256]
        assert harness.observations[-1][1] == 321_760_000
        _covered(harness, enabled)
        if enabled:
            _fixed_term_only(harness, before, plans, predictions)
        else:
            assert (small_before, harness.required(small)) == (188_937_126, 353_936_000)
        monkeypatch.setattr(
            harness.rank, "_available_memory_bytes", lambda sample=None: 250_000_000
        )
        check = harness.rank._memory_check(small)
        assert check.estimated_required_bytes >= 321_760_000
        assert not check.fits
        with pytest.raises(TrainerRankMemoryError, match="predicted to exceed"):
            harness.run([256])
        assert len(harness.executed) == 1


@pytest.mark.parametrize("tokens", [8, 16, 32])
@pytest.mark.parametrize("enabled", [False, True])
def test_warm_tiny_waves_only_raise_fixed_term_after_anchoring(
    monkeypatch, enabled, tokens
):
    with _memory.fixed_slope_memory_profiles(enabled):
        harness = _harness(monkeypatch)
        _ladder(harness)
        plans = [harness.plan([n]) for n in (8192, 32768)] + [
            _shared_prompt_plan(harness)
        ]
        before = harness.rank._memory_profiles[plans[0].signature]
        predictions = [harness.required(plan) for plan in plans]
        harness.run([tokens])
        assert [p.packed_tokens for p in harness.executed] == [tokens]
        assert harness.executed[0].signature == plans[0].signature
        _covered(harness, enabled)
        if enabled:
            _fixed_term_only(harness, before, plans, predictions)
        else:
            # Documentation of current main behavior, not desired behavior.
            assert tuple(predictions) == OFF_WARM_BEFORE
            assert (
                tuple(harness.required(plan) for plan in plans)
                == OFF_WARM_AFTER[tokens]
            )
