"""Rate qualification, fallback and absolute-floor CPU admission contracts."""

from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_trainer_rank_active_memory import _rank
from test_trainer_rank_fixed_slope import GROUPS, signature
from test_trainer_rank_tiny_wave_poisoning import RATE, WORKSPACE, _harness
import torch

from art.trainer_rank import TrainerRankMemoryError, _impl
from art.trainer_rank import _memory as m
from art.trainer_rank import _planner_misses as reports


def observe(rank, key, n, peak, *, eligible=True, local_tokens=0):
    plan = SimpleNamespace(
        signature=key,
        packed_tokens=n,
        active_logical_tokens=n,
        output_bytes=0,
        local_packed_tokens=local_tokens,
    )
    m._update_memory_profile(
        rank,
        cast(Any, plan),
        peak,
        retained_bytes=None,
        caller_phase=True,
        rate_eligible=eligible,
    )
    return plan


def required(rank, key, n, *, raw=False):
    from unittest.mock import patch

    with patch.object(_impl, "_MEMORY_SAFETY_FACTOR", 1 if raw else 1.1):
        return rank._estimate_required_memory_bytes_from_values(
            packed_tokens=n,
            logical_tokens=n,
            output_bytes=0,
            signature=key,
            include_cold_extrapolation=not raw,
        )


def seed_model_reference(harness):
    key = replace(harness.plan([8192]).signature, request_mix=("hidden",))
    observe(harness.rank, key, 8192, WORKSPACE + RATE * 8192)
    return key


@pytest.mark.parametrize(
    "cp,floor", [(1, 64), (2, 128), (8, 512), (16, 1024), (128, 8192), (256, 8192)]
)
def test_rate_floor_uses_global_cp_extent_and_never_exceeds_anchor(cp, floor):
    key = replace(signature(), topology=(1, 1, cp, 1))
    assert m._rate_floor_tokens(key) == floor
    rank = _rank()
    with m.fixed_slope_memory_profiles():
        observe(rank, key, floor - 1, 300_000_000, local_tokens=floor * 2)
        profile = rank._memory_profiles[key]
        assert profile.rate_observations == ()
        assert profile.absolute_compute_floor == 300_000_000
        observe(rank, key, floor, 400_000_000, local_tokens=0)
        profile = rank._memory_profiles[key]
        assert profile.rate_observations == ((floor, 400_000_000),)
        assert profile.absolute_compute_floor == 300_000_000
        assert profile.compute_slope == (
            (400_000_000, 8192) if floor == 8192 else (0, 1)
        )


def test_clean_own_rate_ignores_tiny_max_rate_and_sharing():
    rank, key = _rank(), signature()
    with m.fixed_slope_memory_profiles():
        observe(rank, key, 512, 343_520_000)
        p = observe(rank, key, 3, 300_255_000)
        p.active_logical_tokens = 513
        m._update_memory_profile(
            rank, cast(Any, p), 300_255_000, retained_bytes=1000, caller_phase=True
        )
        profile = rank._memory_profiles[key]
        reference = m._rate_reference(rank, key, profile)
        assert reference is not None and reference.source == "signature"
        assert reference.rate == (343_520_000, 512)
        assert profile.logical_per_packed == 1
        assert profile.warm_bytes_per_token is None
        assert profile.retained_compute_bytes_per_token is None
        assert profile.absolute_compute_floor == 300_255_000
        assert required(rank, key, 8192, raw=True) >= 300_255_000 + 343_520_000 * 16
        assert required(rank, key, 8192) < 12_000_000_000


def test_root_counterpart_precedes_larger_model_anchor():
    rank, key = _rank(), signature((1, "forward_backward", 8))
    root = m._ScopedMemorySignature(**vars(key), bounded_caller_peak=True)
    other = replace(key, request_mix=("hidden",))
    with m.fixed_slope_memory_profiles():
        observe(rank, root, 8192, 1_000_000_000)
        observe(rank, other, 8192, 10_000_000_000)
        observe(rank, key, 3, 300_000_000)
        reference = m._rate_reference(rank, key, rank._memory_profiles[key])
        assert reference is not None and reference.source == "calibration_root"
        assert reference.rate == (1_000_000_000, 8192)
        assert required(rank, key, 8192, raw=True) >= 1_300_000_000
        assert required(rank, key, 8192) < 3_000_000_000


def test_model_fallback_is_the_max_clean_anchored_rate():
    rank, key = _rank(), signature()
    with m.fixed_slope_memory_profiles():
        observe(rank, replace(key, request_mix=("hidden",)), 8192, 1_000_000_000)
        observe(rank, replace(key, request_mix=("logits",)), 8192, 2_000_000_000)
        observe(rank, key, 3, 300_000_000)
        reference = m._rate_reference(rank, key, rank._memory_profiles[key])
        assert reference is not None and reference.source == "model_anchor"
        assert reference.rate == (2_000_000_000, 8192)
        assert required(rank, key, 8192, raw=True) >= 2_300_000_000
        assert required(rank, key, 8192) < 5_000_000_000


def test_fallback_inventory_order_and_predictions_are_monotone():
    key = signature()
    entries = (
        (replace(key, request_mix=("hidden",)), 8192, 1_000_000_000),
        (replace(key, request_mix=("logits",)), 32768, 3_000_000_000),
        (replace(key, slot_group_count=2), 8192, 2_000_000_000),
    )
    expected = None
    with m.fixed_slope_memory_profiles():
        for order in itertools.permutations(entries):
            rank = _rank()
            for other, n, peak in order:
                observe(rank, other, n, peak)
            observe(rank, key, 3, 300_000_000)
            prices = tuple(
                required(rank, key, n)
                for n in (1, 3, 4, 511, 512, 8192, 8193, 32768, 131072)
            )
            assert prices == tuple(sorted(prices))
            assert expected is None or prices == expected
            expected = prices


def test_sub_floor_wave_cannot_remove_a_long_anchor_downward_floor():
    rank, key = _rank(), signature()
    with m.fixed_slope_memory_profiles():
        observe(rank, key, 32768, 2_000_000_000)
        before = required(rank, key, 8192, raw=True)
        observe(rank, key, 3, 300_000_000)
        assert required(rank, key, 8192, raw=True) >= before
        assert (
            m._profile_compute_peak(rank._memory_profiles[key], 8192) >= 2_000_000_000
        )


@pytest.mark.parametrize(
    "bad", ["topology", "operation", "unanchored", "v1", "other_model"]
)
def test_no_compatible_evidence_defers_instead_of_using_fixed_only(bad):
    rank, key = _rank(), signature()
    other = replace(key, request_mix=("hidden",))
    owner = rank
    if bad == "topology":
        other = replace(other, topology=(1, 1, 1, 1))
    elif bad == "operation":
        other = replace(other, grad_enabled=True, grad_modes=(True,))
    elif bad == "other_model":
        owner = _rank()
    with m.fixed_slope_memory_profiles():
        if bad == "v1":
            owner._memory_profiles[other] = _impl._MemoryProfile(64_600_000, 81920)
        else:
            observe(owner, other, 512 if bad == "unanchored" else 8192, 1_000_000_000)
        observe(rank, key, 3, 300_000_000)
        assert m._rate_reference(rank, key, rank._memory_profiles[key]) is None
        for n in (512, 8192, 32768):
            assert required(rank, key, n, raw=True) >= m._MEMORY_DEFERRED_BYTES
            assert required(rank, key, n) >= m._MEMORY_DEFERRED_BYTES
        assert required(rank, key, 3, raw=True) >= 300_000_000


@pytest.mark.parametrize("n,effective", [(8192, 2_495_969_931), (32768, 8_249_907_166)])
def test_cold_tiny_then_large_uses_bounded_reference_and_learns(
    monkeypatch, n, effective
):
    with m.fixed_slope_memory_profiles():
        h = _harness(monkeypatch)
        seed_model_reference(h)
        demands = []
        original = h.rank._run_flat_plan_with_memory_tracking

        def record(plan, **kwargs):
            demands.append(h.required(plan))
            return original(plan, **kwargs)

        monkeypatch.setattr(h.rank, "_run_flat_plan_with_memory_tracking", record)
        h.run([3, n])
        assert [p.packed_tokens for p in h.executed] == [3, n]
        # Split retained/ephemeral integer accounting may differ by one byte
        # from the direct admission operand. Both cover the injected peak.
        assert effective - 1 <= demands[-1] <= effective
        assert demands[-1] >= WORKSPACE + RATE * n
        profile = h.rank._memory_profiles[h.plan([n]).signature]
        assert profile.rate_observations and profile.rate_observations[-1][0] == n
        assert not m._unanchored(profile)
        for plan, peak in h.observations:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(_impl, "_MEMORY_SAFETY_FACTOR", 1)
                assert (
                    h.rank._estimate_required_memory_bytes_from_values(
                        packed_tokens=plan.packed_tokens,
                        output_bytes=plan.output_bytes,
                        logical_tokens=plan.active_logical_tokens,
                        signature=plan.signature,
                        include_cold_extrapolation=False,
                    )
                    >= peak
                )
            assert h.required(plan) >= peak


@pytest.mark.parametrize("seed", [False, True])
def test_tiny_only_sequence_retains_every_peak_without_learning_a_rate(
    monkeypatch, seed
):
    with m.fixed_slope_memory_profiles():
        h = _harness(monkeypatch)
        if seed:
            seed_model_reference(h)
        for n in (3, 8, 16, 32, 256) * 3:
            h.run([n])
            p = h.rank._memory_profiles[h.plan([n]).signature]
            assert p.rate_observations == () and p.compute_slope == (0, 1)
            assert p.absolute_compute_floor >= WORKSPACE + RATE * n - 4 * n
            assert h.required(h.plan([n])) >= WORKSPACE + RATE * n
            plan = h.plan([n])
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(_impl, "_MEMORY_SAFETY_FACTOR", 1)
                assert (
                    h.rank._estimate_required_memory_bytes_from_values(
                        packed_tokens=n,
                        logical_tokens=n,
                        output_bytes=plan.output_bytes,
                        signature=plan.signature,
                        include_cold_extrapolation=False,
                    )
                    >= WORKSPACE + RATE * n
                )
        for n in (8192, 32768):
            demand = h.required(h.plan([n]))
            assert (
                demand < 9_000_000_000 if seed else demand >= m._MEMORY_DEFERRED_BYTES
            )


def test_global_floor_and_qualified_rate_merge_are_order_independent():
    key = signature()
    points = (
        (3, 300_255_000),
        (256, 321_760_000),
        (512, 343_520_000),
        (8192, 996_320_000),
    )
    expected = None
    with m.fixed_slope_memory_profiles():
        for order in itertools.permutations(points):
            rank = _rank()
            for n, peak in order:
                observe(rank, key, n, peak)
            p = rank._memory_profiles[key]
            state = (
                p.compute_observations,
                p.rate_observations,
                p.absolute_compute_floor,
                p.fixed_compute_bytes,
                p.compute_slope,
                tuple(required(rank, key, n) for n in (3, 256, 8192, 32768)),
            )
            assert p.rate_observations == ((512, 343_520_000), (8192, 996_320_000))
            assert p.absolute_compute_floor == 321_760_000
            assert expected is None or state == expected
            expected = state
        profiles = []
        for half in (points[:2], points[2:]):
            rank = _rank()
            for n, peak in half:
                observe(rank, key, n, peak)
            profiles.append(rank._memory_profiles[key])
        a = m._merge_memory_profile_model_fields(*profiles)
        b = m._merge_memory_profile_model_fields(*reversed(profiles))
        assert a == b
        assert expected is not None
        assert a["rate_observations"] == expected[1]
        assert a["absolute_compute_floor"] == expected[2]


@pytest.mark.parametrize("key,rows", list(GROUPS.items()))
def test_all_measured_fixture_peaks_remain_raw_and_effective_bounds(key, rows):
    rank, sig = _rank(), signature(key)
    with m.fixed_slope_memory_profiles():
        observe(rank, sig, 3, 300_000_000)
        for row in rows:
            observe(rank, sig, row["length"], row["measured_increment_bytes"])
        for row in rows:
            assert (
                required(rank, sig, row["length"], raw=True)
                >= row["measured_increment_bytes"]
            )
            assert required(rank, sig, row["length"]) >= max(row["rank_peak_bytes"])


def test_format3_roundtrips_and_old_v2_defaults_remain_readable():
    rank, key = _rank(), signature()
    with m.fixed_slope_memory_profiles():
        observe(rank, key, 3, 300_000_000)
        observe(rank, key, 8192, 1_000_000_000)
    p = rank._memory_profiles[key]
    assert p.profile_format_version == 3
    assert reports._memory_profile(json.loads(json.dumps(asdict(p)))) == p
    assert pickle.loads(pickle.dumps(p)) == p
    old = asdict(p)
    for name in (
        "profile_format_version",
        "rate_floor_tokens",
        "rate_observations",
        "absolute_compute_floor",
    ):
        old.pop(name)
    from art.trainer_rank import _memory_model

    old["fixed_compute_bytes"], old["compute_slope"] = _memory_model.fit_points(
        p.compute_observations
    )
    restored = m._memory_profile_from_values(old)
    assert restored.profile_format_version == 2 and restored.rate_observations is None


def test_fallback_pool_is_frozen_for_report_replay(tmp_path):
    from test_grouped_planner_replay import emitted

    from art.trainer_rank import ForwardInput

    rank = _rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward(
        [ForwardInput(input_tokens=torch.arange(128), hidden_states=True, no_grad=True)]
    )
    key = plan.signature
    with m.fixed_slope_memory_profiles():
        observe(rank, replace(key, request_mix=("target:single",)), 8192, 1_000_000_000)
        observe(rank, key, 3, 300_000_000)
        report, _ = emitted(rank, plan, tmp_path)
    assert report["replay"]["memory_replay"]["fixed_slope_profile_pool"]
    result = reports.replay(report)
    assert result["aggregate"]["matches"]


def test_one_rank_without_a_reference_collectively_defers(tmp_path):
    from test_moe_caller_cp_ownership import admission_worker
    import torch.multiprocessing as mp

    key = signature()
    demands = []
    with m.fixed_slope_memory_profiles():
        for i in range(8):
            rank = _rank()
            if i != 6:
                observe(
                    rank, replace(key, request_mix=("hidden",)), 8192, 1_000_000_000
                )
            observe(rank, key, 3, 300_000_000 + i)
            assert rank._memory_profiles[key].rate_observations == ()
            demands.append(required(rank, key, 8192))
    mp.start_processes(
        admission_worker,
        args=(8, str(tmp_path / "gloo"), str(tmp_path), demands, [20_000_000_000] * 8),
        nprocs=8,
        join=True,
        start_method="fork",
    )
    results = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(8)]
    assert not any(row["fits"] for row in results)
    assert len({row["required"] for row in results}) == 1
