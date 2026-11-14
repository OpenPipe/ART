"""CPU regressions for CP qualification and legacy report compatibility."""

import base64
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_trainer_rank_active_memory import _rank
import torch

from art.trainer_rank import _impl
from art.trainer_rank import _memory as m
from art.trainer_rank import _memory_model as model
from art.trainer_rank import _planner_misses as reports

FIXTURES = Path(__file__).parent / "fixtures/memory_fixed_slope"


def observe(rank, sig, n, peak, **extra):
    p = SimpleNamespace(
        signature=sig, packed_tokens=n, active_logical_tokens=n, output_bytes=0, **extra
    )
    m._update_memory_profile(
        rank, cast(Any, p), peak, retained_bytes=None, caller_phase=True
    )


def test_cp_uncovered_rank_load_must_refuse(tmp_path):
    from test_moe_caller_cp_coverage import add, plan, required, workload
    from test_moe_caller_cp_ownership import admission_worker
    from test_moe_long_root_caller import memory_rank
    import torch.multiprocessing as mp

    cp, n, anchor = 8, 8192, 8320
    measured, wanted = workload(cp, anchor), workload(cp, n)
    assert measured[1][6][0] == 512 and wanted[1][6][0] == 1024
    demands = []
    with m.fixed_slope_memory_profiles():
        for row in measured[1]:
            rank = memory_rank()
            rank._topology_key = lambda: (1, 1, cp, 1)
            add(rank, plan(cp, 128, workload(cp, 128)), 8 * 2**30)
            add(rank, plan(cp, anchor, measured), max(8 * 2**30, row[0] * 32 * 2**20))
            demands.append(required(rank, plan(cp, n, wanted)))
    budgets = [80 * 2**30] * cp
    budgets[6] = 27 * 2**30
    actual = wanted[1][6][0] * 32 * 2**20 + 4 * n
    assert actual > budgets[6]
    mp.start_processes(
        admission_worker,
        args=(cp, str(tmp_path / "gloo"), str(tmp_path), demands, budgets),
        nprocs=cp,
        join=True,
        start_method="fork",
    )
    results = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(cp)]
    assert not any(r["fits"] for r in results), (
        "admitted rank 6 at 27 GiB with a 32 GiB workspace load"
    )


def test_single_larger_anchor_covers_measured_smaller_peak():
    from test_trainer_rank_fixed_slope import signature

    rows = [
        r
        for r in json.loads((FIXTURES / "sweeps.json").read_text())
        if (r["model"], r["operation"], r["lora_rank"]) == (1, "forward_backward", 32)
    ]
    rank = _rank()
    key = signature((1, "forward_backward", 32))
    anchor = next(r for r in rows if r["length"] == 32768)
    target = next(r for r in rows if r["length"] == 8192)
    with m.fixed_slope_memory_profiles():
        observe(rank, key, anchor["length"], anchor["measured_increment_bytes"])
        assert (
            m._profile_compute_peak(rank._memory_profiles[key], target["length"])
            >= target["measured_increment_bytes"]
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "record", json.loads((FIXTURES / "legacy_serialization.json").read_text())
)
def test_json_report_roundtrip_is_valid_and_equal(record, enabled):
    values = dict(record["profile"])
    if enabled:
        values.update(
            peak_model_version=2,
            compute_observations=((8192, 2**30),),
            fixed_compute_bytes=0,
            compute_slope=(2**30, 8192),
        )
    profile = reports._memory_profile(values)
    assert pickle.loads(pickle.dumps(profile)) == profile
    restored = reports._memory_profile(json.loads(json.dumps(asdict(profile))))
    m._validate_memory_profile_model(restored)
    assert restored == profile


@pytest.mark.parametrize("mismatch", ["family", "rank", "dimension", "missing"])
def test_v2_caller_requires_componentwise_cp_qualification(mismatch):
    from test_moe_caller_cp_coverage import add, plan
    from test_moe_long_root_caller import memory_rank

    rank = memory_rank()
    loads = ("test-layout", ((512, 128), (512, 128)))
    with m.fixed_slope_memory_profiles():
        add(rank, plan(2, 128, loads), 2**30)
        add(rank, plan(2, 8192, loads), 2 * 2**30)
    profile = rank._memory_profiles[plan(2, 8192, loads).signature]
    discounted = profile.compute_peak(8192, context_parallel=True, cp_workload=loads)
    assert 2 * 2**30 <= discounted < profile.bytes_per_token * 8192
    request = {
        "family": ("other-layout", loads[1]),
        "rank": (loads[0], ((513, 128), (512, 128))),
        "dimension": (loads[0], ((512, 129), (512, 128))),
        "missing": None,
    }[mismatch]
    assert (
        profile.compute_peak(8192, context_parallel=True, cp_workload=request)
        >= profile.bytes_per_token * 8192
    )
    restored = replace(profile, legacy_bytes_per_token=2 * profile.bytes_per_token)
    assert (
        restored.compute_peak(8192, context_parallel=True, cp_workload=request)
        >= restored.legacy_bytes_per_token * 8192
    )


def test_off_profile_serialization_matches_base(tmp_path):
    from test_grouped_planner_replay import emitted

    from art.trainer_rank import ForwardInput

    rank = _rank()
    request = ForwardInput(
        input_tokens=torch.arange(8192), target_tokens=torch.arange(8192), no_grad=True
    )
    plan_ = rank._plan_flat_forward([request])
    expected = json.loads((FIXTURES / "legacy_snapshot.json").read_text())
    with m.fixed_slope_memory_profiles(False):
        rank._update_memory_profile(plan_, 2**30, retained_bytes=None)
        p = rank._memory_profiles[plan_.signature]
        current = dict(
            signature=asdict(plan_.signature),
            profile=asdict(p),
            cost=asdict(rank._plan_cost(plan_)),
            demands=[
                rank._estimate_required_memory_bytes_from_values(
                    packed_tokens=n, output_bytes=4 * n, signature=plan_.signature
                )
                for n in (1, 128, 8192, 8193, 32768, 65536, 65537)
            ],
        )
        assert reports._encode(current) == reports._encode(expected)
        report, _ = emitted(rank, plan_, tmp_path)
    assert "fixed_slope_profiles" not in report["replay"]["memory_replay"]
    assert reports._encode(
        report["replay"]["memory_replay"]["estimates"][0]["profile"]
    ) == reports._encode(expected["profile"])


@pytest.mark.parametrize(
    "record", json.loads((FIXTURES / "legacy_serialization.json").read_text())
)
def test_off_profile_pickle_and_report_json_are_byte_identical(record, tmp_path):
    from test_grouped_planner_replay import emitted

    from art.trainer_rank import ForwardInput

    baseline_bytes = base64.b64decode(record["pickle_protocol4"])
    historical = pickle.loads(baseline_bytes)
    # Re-run today's constructor with the exact baseline object graph, including
    # shared tuples whose pickle memoization would be lost through JSON.
    profile = type(historical)(**vars(historical))
    assert type(profile).__name__ == record["type"]
    assert pickle.dumps(profile, protocol=4) == baseline_bytes
    assert reports._encode(asdict(profile)) == reports._encode(record["profile"])
    rank = _rank()
    plan_ = rank._plan_flat_forward(
        [
            ForwardInput(
                input_tokens=torch.arange(8192),
                target_tokens=torch.arange(8192),
                no_grad=True,
            )
        ]
    )
    rank._memory_profiles[plan_.signature] = profile
    with m.fixed_slope_memory_profiles(False):
        report, _ = emitted(rank, plan_, tmp_path)
    assert reports._encode(
        report["replay"]["memory_replay"]["estimates"][0]["profile"]
    ) == reports._encode(record["profile"])


@pytest.mark.parametrize(
    "record", json.loads((FIXTURES / "legacy_serialization.json").read_text())
)
def test_earlier_v2_pickle_promotes_without_losing_model_fields(record):
    old = reports._memory_profile(record["profile"])
    for name, value in dict(
        peak_model_version=2,
        compute_observations=((8192, 2**30),),
        fixed_compute_bytes=0,
        compute_slope=(2**30, 8192),
        legacy_bytes_per_token=0.0,
    ).items():
        object.__setattr__(old, name, value)
    restored = pickle.loads(pickle.dumps(old, protocol=4))
    assert type(restored).__name__ == record["type"] + "V2"
    assert asdict(restored) == vars(old)
    m._validate_memory_profile_model(restored)


def test_new_model_is_in_report_source_fingerprint():
    assert "_memory_model.py" in reports._source_files()


def test_unprofiled_enabled_report_replays_outside_collection_context(tmp_path):
    from test_grouped_planner_replay import emitted

    from art.trainer_rank import ForwardInput

    rank = _rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward(
        [
            ForwardInput(
                input_tokens=torch.arange(128),
                target_tokens=torch.arange(128),
                no_grad=True,
            )
        ]
    )
    with m.fixed_slope_memory_profiles():
        report, _ = emitted(rank, plan, tmp_path)
    assert report["replay_complete"]
    result = reports.replay(report)
    assert result["aggregate"]["matches"] and all(
        r["matches"] for r in result["estimates"]
    )


def test_random_hull_preserves_constraints_and_partition_merges():
    rng = random.Random(1173)
    for _ in range(1000):
        points = tuple(
            (rng.randint(1, 131072), rng.randint(0, 10000000000)) for _ in range(30)
        )
        hull = model.merge_points(points)
        fixed, slope = model.fit_points(hull)
        assert all(model.predict(fixed, slope, n) >= y for n, y in points)
        assert (
            model.merge_points(
                model.merge_points(points[:10]),
                model.merge_points(points[10:20]),
                model.merge_points(points[20:]),
            )
            == hull
        )
        assert model.merge_points(hull, hull) == hull


def test_merge_is_not_prediction_monotone():
    rank = _rank()
    sig = _impl._MemorySignature((1, 1, 1, 1), (1, None), 1, (), False, (False,))
    with m.fixed_slope_memory_profiles():
        observe(rank, sig, 8192, 8 * 2**30)
        a = rank._memory_profiles[sig]
        rank._memory_profiles.clear()
        observe(rank, sig, 32768, 8 * 2**30)
        b = rank._memory_profiles[sig]
    fields = m._merge_memory_profile_model_fields(a, b)
    assert fields == m._merge_memory_profile_model_fields(b, a)
    merged = _impl._MemoryProfileV2(
        bytes_per_token=max(a.bytes_per_token, b.bytes_per_token),
        packed_tokens=32768,
        **fields,
    )
    # Preserving observed constraints is different from preserving every
    # previously predicted bound, even before the cold margin is considered.
    assert m._profile_compute_peak(merged, 32768) < m._profile_compute_peak(a, 32768)
    assert all(
        m._profile_compute_peak(merged, n) >= y
        for n, y in (*a.compute_observations, *b.compute_observations)
    )
