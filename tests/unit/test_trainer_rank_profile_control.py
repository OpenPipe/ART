"""Independent learning control and bounded conservative retention evidence."""

from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path
import pickle
import random
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from test_trainer_rank_active_memory import _rank
from test_trainer_rank_fixed_slope import signature
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import ForwardInput, ForwardOutput, _impl
from art.trainer_rank import _memory as m
from art.trainer_rank import _planner_misses as reports


def _cohort_worker(index, rendezvous, output, variant):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=8),
    )
    try:
        rank = _rank()
        rank._dp_rank_and_size = lambda: (0, 1)
        # Explicitly oversized CPU collection, with no real model allocation.
        rank._available_memory_bytes = lambda sample=None: 9_000_000_000_000_000_000
        requests = [
            ForwardInput(
                input_tokens=torch.arange(8192), hidden_states=True, no_grad=False
            )
        ]
        plan = rank._plan_flat_forward(requests)
        source = plan.signature
        source_peak = 1_000_000_000 + index * 100_000_000
        trace = []
        original_reduce = rank._recovery_reduce

        def reduce(values, **kwargs):
            trace.append((kwargs["op"], values))
            return original_reduce(values, **kwargs)

        rank._recovery_reduce = reduce
        current_peak = [source_peak + plan.output_bytes]
        sampling_error = [False]

        def peak(*_):
            if sampling_error[0]:
                raise ValueError("sample unavailable")
            return current_peak[0]

        def execute(candidate, **kwargs):
            rank._peak_resets = 1
            current_peak[0] = (
                source_peak if candidate.signature == source else 500_000_000
            ) + candidate.output_bytes
            baseline = None if variant == "missing" and index == 1 else 0
            rank._update_peak_memory_profile(
                candidate, baseline, retained_after=candidate.output_bytes
            )
            values = [
                ForwardOutput(
                    None
                    if item.request.hidden_states
                    else torch.zeros(len(item.input_ids)),
                    None,
                    None,
                    torch.zeros((len(item.input_ids), 8))
                    if item.request.hidden_states
                    else None,
                )
                for group in candidate.groups
                for item in group.items
            ]
            return values, baseline

        rank._run_flat_plan_with_memory_tracking = execute
        with (
            patch.object(torch.cuda, "max_memory_allocated", peak),
            m.fixed_slope_memory_profiles(),
        ):
            rank._update_memory_profile(
                plan,
                source_peak + plan.output_bytes,
                retained_bytes=None,
                caller_phase=False,
            )
            batches = rank.forward_micro_batches(requests, no_grad=False)
            next(batches)
            if index == 1:
                if variant == "reset":
                    rank._peak_resets += 1
                elif variant == "decrease":
                    current_peak[0] -= 1
                elif variant == "error":
                    sampling_error[0] = True
            trace.clear()
            error = None
            try:
                list(batches)
            except Exception as exc:
                error = type(exc).__name__
            resume_trace = list(trace)
            sampling_error[0] = False
            result = dict(
                error=error,
                trace=resume_trace,
                callers=rank._memory_profiles[source].caller_plans,
                peak=rank._memory_profiles[source].compute_observations[-1][1],
                suppressed=rank._suppressed_caller_memory_samples,
                warned=rank._caller_memory_sample_warning_logged,
            )
            if error is None and variant != "error":
                target_requests = [
                    ForwardInput(
                        input_tokens=torch.arange(512),
                        target_tokens=torch.arange(512),
                        no_grad=False,
                    )
                ]
                target_plan = rank._plan_flat_forward(target_requests)
                tiny = replace(
                    target_plan, packed_tokens=3, logical_tokens=3, output_bytes=0
                )
                rank._update_memory_profile(tiny, 300_000_000, retained_bytes=None)
                # This CPU diagnostic intentionally collects an untrusted wave
                # to test learning isolation. Deferred estimates are now policy
                # refusals even under the synthetic oversized budget; supply
                # explicit fixture execution permission, not an admission bypass
                # in the production estimator. The final check below is real.
                synthetic = _impl._CandidateMicroBatch(
                    inputs=tuple(target_requests),
                    indices=(0,),
                    plan=target_plan,
                    check=replace(rank._memory_check(target_plan), fits=True),
                    stats_global_count=1,
                    rejected_candidates=0,
                    cold_start=True,
                )
                list(
                    cast(Any, _impl.TrainerRank.forward_micro_batches)(
                        rank,
                        target_requests,
                        no_grad=False,
                        _calibration_candidate=synthetic,
                    )
                )
                profile = rank._memory_profiles[target_plan.signature]
                demand = rank._estimate_required_memory_bytes_from_values(
                    packed_tokens=32768,
                    logical_tokens=32768,
                    output_bytes=0,
                    signature=target_plan.signature,
                )
                rank._available_memory_bytes = lambda sample=None: 70_000_000_000
                check = rank._memory_check_required(demand)
                result.update(
                    rates=profile.rate_observations,
                    target_callers=profile.caller_plans,
                    trusted=m._trusted_extent(profile),
                    demand=demand,
                    fits=check.fits,
                )
            Path(output, f"rank{index}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("variant", ["reset", "decrease", "missing", "error", "valid"])
def test_borrowed_learning_needs_no_caller_agreement(tmp_path, variant):
    _rank()  # Prime optional runtime imports before starting the short timeout.
    context = mp.start_processes(
        _cohort_worker,
        args=(str(tmp_path / "gloo"), str(tmp_path), variant),
        nprocs=2,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 20
    try:
        while not context.join(timeout=0.1):
            if time.monotonic() >= deadline:
                pytest.fail("cohort did not complete within the timeout")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
    rows = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    # Local readings/complete-caller counts may differ. Neither can extend a
    # different signature's learning window, so no control vote is needed.
    assert [row["peak"] for row in rows] == [1_000_000_000, 1_100_000_000]
    assert all(row["trace"] == [] for row in rows)
    if variant == "error":
        assert [row["error"] for row in rows] == [None, None]
        assert [row["suppressed"] for row in rows] == [0, 1]
        assert [row["warned"] for row in rows] == [False, True]
        assert [row["callers"] for row in rows] == [1, 0]
    else:
        assert not any(row["error"] for row in rows)
        assert all(row["rates"] == [] and row["target_callers"] == 0 for row in rows)
        assert all(row["trusted"] == 3 for row in rows)
        assert [row["callers"] for row in rows] == (
            [1, 1] if variant == "valid" else [1, 0]
        )
        if variant == "valid":
            assert all(row["demand"] < m._MEMORY_DEFERRED_BYTES for row in rows)
            assert all(row["fits"] for row in rows)
        else:
            # Admission still uses its existing MAX: an uncovered peer refuses.
            assert rows[1]["demand"] >= m._MEMORY_DEFERRED_BYTES
            assert not any(row["fits"] for row in rows)


def test_missing_sample_is_local_and_does_not_learn(monkeypatch):
    rank = _rank()
    monkeypatch.setattr(
        rank, "_recovery_reduce", lambda *a, **k: pytest.fail("profiling exchange")
    )
    monkeypatch.setattr(
        torch.cuda, "max_memory_allocated", lambda *a: pytest.fail("missing sample")
    )
    plan = SimpleNamespace(signature=signature(), packed_tokens=512)
    rank._update_peak_memory_profile(cast(Any, plan), None)
    assert rank._memory_profiles == {}


def test_own_token_bridge_still_grows_into_an_anchor(monkeypatch):
    rank, key = _rank(), signature((0, "forward_backward", 8))
    anchor = replace(key, request_mix=("hidden",))

    def plan(key, n):
        return cast(
            Any,
            SimpleNamespace(
                signature=key,
                packed_tokens=n,
                active_logical_tokens=n,
                output_bytes=0,
            ),
        )

    with m.fixed_slope_memory_profiles():
        rank._update_memory_profile(
            plan(anchor, 8192), 1_000_000_000, retained_bytes=None, caller_phase=True
        )
        for n, peak in ((3, 300_000_000), (512, 500_000_000)):
            monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: peak)
            rank._update_peak_memory_profile(
                plan(key, n), 0, caller_phase=True, interval=(0, 0)
            )
        profile = rank._memory_profiles[key]
        assert profile.rate_observations == ()
        assert m._trusted_extent(profile) == 3
        # Borrowing may price the request, but only this signature's own
        # sub-floor bridge can extend learning into floor-qualified waves.
        for n in (256, 512, 4096, 8192):
            monkeypatch.setattr(
                torch.cuda, "max_memory_allocated", lambda *a: 1_000_000_000
            )
            rank._update_peak_memory_profile(
                plan(key, n), 0, caller_phase=True, interval=(0, 0)
            )
            if n == 256:
                assert rank._memory_profiles[key].rate_observations == ()
            else:
                assert rank._memory_profiles[key].eligible_packed_tokens == n
        assert not m._unanchored(rank._memory_profiles[key])


def _floor(points, n):
    if hasattr(m, "_retained_floor_peak"):
        return m._retained_floor_peak(points, n)
    return max((peak for tokens, peak in points if tokens <= n), default=0)


def test_retained_history_and_serialization_have_a_fixed_budget():
    rank, key = _rank(), signature((0, "forward_backward", 8))
    sizes = []
    with m.fixed_slope_memory_profiles():
        for n in range(8193, 16385):
            plan = SimpleNamespace(
                signature=key, packed_tokens=n, active_logical_tokens=n, output_bytes=0
            )
            rank._update_memory_profile(plan, n * 100_000, retained_bytes=n * 10_000)
            if n in (8704, 10240, 16384):
                profile = rank._memory_profiles[key]
                assert len(profile.retained_compute_peaks) <= 64
                sizes.append(len(json.dumps(asdict(profile))))
        profile = rank._memory_profiles[key]
        assert max(sizes) < 6000
        assert pickle.loads(pickle.dumps(profile)) == profile
        assert (
            reports._memory_profile(json.loads(json.dumps(asdict(profile)))) == profile
        )
        assert all(
            _floor(profile.retained_compute_peaks, n) >= n * 10_000
            for n in range(8193, 16385)
        )


def test_compaction_preserves_every_observed_and_previous_bound():
    rng = random.Random(119)
    sequences = [
        [(n, n * 17) for n in range(1, 1025)],
        [(n, 10**9 if n == 513 else n) for n in range(1, 1025)],
        [(2**n + 1, 10**12 + n) for n in range(90)],
        [(512, 512_000), (8192, 8_000_000_000), (2048, 1_000_000)],
    ] + [
        [(rng.randint(1, 10**6), rng.randint(0, 10**12)) for _ in range(128)]
        for _ in range(20)
    ]
    probes = (1, 2, 3, 511, 512, 8192, 32768, 10**6, 2**100)
    for sequence in sequences:
        state = ()
        for point in sequence:
            before = [_floor(state, n) for n in probes]
            state = m._merge_retained_peaks(state, (point,))
            assert len(state) <= 64
            assert _floor(state, point[0]) >= point[1]
            assert all(_floor(state, n) >= value for n, value in zip(probes, before))
        assert all(_floor(state, n) >= peak for n, peak in sequence)
        assert state == m._merge_retained_peaks(sequence)
        shuffled = list(sequence)
        rng.shuffle(shuffled)
        assert state == m._merge_retained_peaks(
            shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2 :]
        )


def test_old_retained_encoding_loads_into_a_bounded_format():
    rank, key = _rank(), signature((0, "forward_backward", 8))
    with m.fixed_slope_memory_profiles():
        plan = SimpleNamespace(
            signature=key,
            packed_tokens=8193,
            active_logical_tokens=8193,
            output_bytes=0,
        )
        rank._update_memory_profile(plan, 8193 * 100_000, retained_bytes=8193 * 10_000)
    values = asdict(rank._memory_profiles[key])
    values.update(
        profile_format_version=4,
        packed_tokens=16384,
        retained_compute_peaks=tuple((n, n * 10_000) for n in range(8193, 16385)),
    )
    restored = m._memory_profile_from_values(values)
    assert len(restored.retained_compute_peaks) <= 64
    assert len(json.dumps(asdict(restored))) < 6000
    assert all(
        _floor(restored.retained_compute_peaks, n) >= n * 10_000
        for n in range(8193, 16385)
    )
