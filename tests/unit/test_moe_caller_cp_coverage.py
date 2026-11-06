"""CP caller reuse covers actual ownership and workspace, including uneven roots."""

from dataclasses import asdict, replace
import json
import pickle
import random
from types import SimpleNamespace
from typing import Any

import pytest
from test_moe_caller_cp_invariants import sig
from test_moe_long_root_caller import memory_rank
from test_trainer_rank_pending_memory import layer as layer
from test_trainer_rank_pending_memory import pending_rank as pending_rank
import torch

from art.megatron.context_parallel.runtime import (
    context_parallel_memory_workload,
    context_parallel_rank_model_token_counts,
)
from art.megatron.context_parallel.types import ContextParallelConfig, ParallelTopology
from art.trainer_rank import _impl as tr
from art.trainer_rank import _memory as m
from art.trainer_rank import _planner_misses as reports

GIB = 2**30


def workload(cp, length, *, gdn=False, config=None):
    metadata = torch.ones((1, length), dtype=torch.int64)
    args: dict[str, Any] = dict(
        group_ids=metadata,
        parent_ids=metadata,
        topology=ParallelTopology(cp=cp),
        config=config or ContextParallelConfig(planner_owned_token_ms=0.0003),
        original_seq_len=length,
        build_gdn_execution_spec=gdn,
    )
    result = context_parallel_memory_workload(**args)
    assert tuple(
        row[0] for row in result[1]
    ) == context_parallel_rank_model_token_counts(**args)
    return result


def plan(cp, length, work):
    return SimpleNamespace(
        signature=sig(cp),
        packed_tokens=length,
        active_logical_tokens=length,
        output_bytes=4 * length,
        caller_cp_workload=work,
    )


def add(rank, plan, peak, *, complete=True):
    m._update_memory_profile(
        rank, plan, peak + plan.output_bytes, retained_bytes=None, caller_phase=complete
    )


def required(rank, plan):
    return m._estimate_required_memory_bytes_from_values(
        rank,
        packed_tokens=plan.packed_tokens,
        logical_tokens=plan.active_logical_tokens,
        output_bytes=plan.output_bytes,
        signature=plan.signature,
        group_rows=((max(row[0] for row in plan.caller_cp_workload[1]), True),),
        caller_cp_workload=plan.caller_cp_workload,
    )


@pytest.mark.parametrize("gdn", (False, True))
@pytest.mark.parametrize("length", (8500, 9216, 12000))
def test_warm_ladder_covers_variable_roots(gdn, length):
    rank = memory_rank()
    for n, peak in ((128, GIB), (8192, 4 * GIB), (32768, 6 * GIB)):
        add(rank, plan(8, n, workload(8, n, gdn=gdn)), peak)
    request = plan(8, length, workload(8, length, gdn=gdn))
    profile = rank._memory_profiles[request.signature]
    assert (
        profile.compute_peak(
            length, context_parallel=True, cp_workload=request.caller_cp_workload
        )
        == 6 * GIB
    )
    assert required(rank, request) < 40 * GIB
    rank._available_memory_bytes = lambda: 40 * GIB
    assert m._memory_check_required(rank, required(rank, request)).fits
    # Drop just one measured rank/dimension below the real request. Every
    # measurement's global length still covers: ownership/geometry must veto it.
    family, loads = request.caller_cp_workload
    measured = profile.caller_cp_peaks[-1]
    altered = list(measured[3])
    altered[-1] = (loads[-1][0] - 1, *altered[-1][1:])
    bad = replace(
        profile, caller_cp_peaks=((measured[0], measured[1], family, tuple(altered)),)
    )
    rank._memory_profiles[request.signature] = bad
    assert required(rank, request) >= int(profile.bytes_per_token * length * 1.1)
    assert not m._memory_check_required(rank, required(rank, request)).fits


def test_smallest_covering_measurement_and_each_workspace_dimension():
    rank = memory_rank()
    work = ("attention+chain", ((100, 80, 40), (80, 60, 30)))
    for n, peak in ((8192, GIB), (32768, 3 * GIB), (65536, 4 * GIB)):
        add(rank, plan(2, n, work), peak)
    p = rank._memory_profiles[sig(2)]
    # Global floors still win, but a zero floor exposes next-higher selection.
    p = replace(p, compute_peak_floor=0)
    assert p.compute_peak(9000, context_parallel=True, cp_workload=work) == 3 * GIB
    for r in range(2):
        for d in range(3):
            loads = [list(row) for row in work[1]]
            loads[r][d] += 1
            request = (work[0], tuple(tuple(row) for row in loads))
            assert (
                p.compute_peak(9000, context_parallel=True, cp_workload=request)
                == p.bytes_per_token * 9000
            )
    assert (
        p.compute_peak(
            9000, context_parallel=True, cp_workload=("other-layout", work[1])
        )
        == p.bytes_per_token * 9000
    )
    assert p.compute_peak(9000, context_parallel=True) == p.bytes_per_token * 9000
    assert (
        p.compute_peak(65537, context_parallel=True, cp_workload=work)
        == p.bytes_per_token * 65537
    )
    # Equal max rank loads cannot substitute for rank-wise coverage.
    moved = (work[0], tuple(reversed(work[1])))
    assert (
        p.compute_peak(9000, context_parallel=True, cp_workload=moved)
        == p.bytes_per_token * 9000
    )


@pytest.mark.parametrize("cp", (2, 4, 8))
def test_random_histories_with_rank_loads(cp):
    for seed in range(8):
        rng = random.Random(seed)
        rank = memory_rank()
        history = []
        for _ in range(60):
            n = rng.choice((128, 8192, 9216, 32768, 65536))
            work = (
                rng.choice(("attention", "chain")),
                tuple(
                    tuple(rng.randrange(1, 100) for _ in range(4)) for _ in range(cp)
                ),
            )
            peak = rng.randrange(1, 10) * GIB
            complete = bool(rng.randrange(2))
            add(rank, plan(cp, n, work), peak, complete=complete)
            history.append((n, work, peak, complete))
            profile = rank._memory_profiles[sig(cp)]
            for _ in range(8):
                length = rng.choice((512, 8500, 9216, 12000, 32768, 70000))
                if profile.caller_cp_peaks and rng.randrange(2):
                    n, _, family, loads = rng.choice(profile.caller_cp_peaks)
                    length = min(length, n)
                    request = (
                        family,
                        tuple(tuple(v // 2 for v in row) for row in loads),
                    )
                else:
                    request = (
                        rng.choice(("attention", "chain")),
                        tuple(
                            tuple(rng.randrange(1, 100) for _ in range(4))
                            for _ in range(cp)
                        ),
                    )
                covering = [
                    p
                    for n, p, family, loads in profile.caller_cp_peaks
                    if n >= length
                    and family == request[0]
                    and all(
                        w <= h
                        for want, have in zip(request[1], loads, strict=True)
                        for w, h in zip(want, have, strict=True)
                    )
                ]
                expected = max(
                    covering[0] if covering else profile.bytes_per_token * length,
                    profile.compute_peak_floor,
                )
                assert (
                    profile.compute_peak(
                        length, context_parallel=True, cp_workload=request
                    )
                    == expected
                )
                for n, peak, family, loads in profile.caller_cp_peaks:
                    assert (n, (family, loads), peak, True) in history
            restored = reports._memory_profile(json.loads(json.dumps(asdict(profile))))
            assert restored == profile
            assert pickle.loads(pickle.dumps(profile)) == profile


def test_real_flat_plan_carries_workload_to_admission_cost_and_report(
    pending_rank, monkeypatch, tmp_path
):
    from test_grouped_planner_replay import emitted

    rank = pending_rank
    rank._parallel_shape = replace(rank._parallel_shape, cp=8)
    monkeypatch.setattr(
        tr.TrainerRank, "_topology", lambda self: ParallelTopology(cp=8)
    )
    rank.runtime.model_support_handler.context_parallel_workload_profile = (
        lambda provider: None
    )
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)

    def root(n):
        return rank._plan_flat_forward(
            [
                tr.ForwardInput(
                    input_tokens=torch.arange(n), target_tokens=torch.arange(n)
                )
            ]
        )

    for n, peak in ((128, 256 * GIB), (8192, 256 * GIB), (32768, 256 * GIB)):
        measured = root(n)
        assert measured.signature.bounded_caller_peak
        assert measured.caller_cp_workload is not None
        add(rank, measured, peak)
    request = root(9216)
    p = rank._memory_profiles[request.signature]
    assert (
        p.compute_peak(
            9216, context_parallel=True, cp_workload=request.caller_cp_workload
        )
        == 256 * GIB
    )
    cost = rank._plan_cost(request)
    assert cost.required < p.bytes_per_token * 9216
    rank._available_memory_bytes = lambda: cost.required * 2
    assert rank._memory_check(request).fits
    report, _ = emitted(rank, request, tmp_path)
    result = reports.replay(report)
    assert result["aggregate"]["matches"]
    assert all(item["matches"] for item in result["estimates"] + result["layouts"])
