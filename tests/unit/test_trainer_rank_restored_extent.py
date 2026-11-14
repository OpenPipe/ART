from dataclasses import asdict, replace
import json
import pickle
from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest
from test_trainer_rank_fixed_slope import signature

from art.trainer_rank import _impl
from art.trainer_rank import _memory as m
from art.trainer_rank import _planner_misses as reports


@pytest.mark.parametrize("codec", ["json", "pickle", "merge"])
def test_filtered_restored_caller_rates_still_roundtrip(monkeypatch, codec):
    rank = SimpleNamespace(
        _memory_profiles={},
        _peak_resets=0,
        _peak_reading=None,
        _planner_observation=None,
        device="cpu",
    )
    rank._update_memory_profile = MethodType(m._update_memory_profile, rank)
    key = replace(signature((0, "forward_backward", 8)), topology=(1, 1, 1, 1))
    root = m._ScopedMemorySignature(**vars(key), bounded_caller_peak=True)
    points = ((8192, 16 * 2**30), (32768, 20 * 2**30))
    fixed, slope = m._memory_model.fit_points(points)
    previous = m._CallerMemoryProfileV2(
        bytes_per_token=16 * 2**30 / 8192,
        packed_tokens=32768,
        compute_observations=points,
        fixed_compute_bytes=fixed,
        compute_slope=slope,
        compute_peak_floor=20 * 2**30,
        profile_format_version=5,
        rate_floor_tokens=64,
        rate_observations=points,
        eligible_packed_tokens=32768,
        retained_rate_tokens=32768,
        caller_compute_peaks=((8192, 16 * 2**30),),
        caller_plans=1,
    )
    rank._memory_profiles[root] = reports._memory_profile(
        json.loads(json.dumps(asdict(previous)))
    )
    plan = SimpleNamespace(
        signature=root, packed_tokens=8192, active_logical_tokens=8192, output_bytes=0
    )
    monkeypatch.setattr(_impl.torch.cuda, "max_memory_allocated", lambda *_: 16 * 2**30)
    with m.fixed_slope_memory_profiles():
        m._update_peak_memory_profile(
            cast(Any, rank),
            cast(Any, plan),
            0,
            caller_phase=True,
            interval=(0, 16 * 2**30),
        )
    current = rank._memory_profiles[root]
    if codec == "json":
        assert (
            reports._memory_profile(json.loads(json.dumps(asdict(current)))) == current
        )
    elif codec == "pickle":
        assert pickle.loads(pickle.dumps(current)) == current
    else:
        merged = replace(previous, **m._merge_memory_profile_model_fields(previous))
        m._validate_memory_profile_model(merged)
        current = merged
    assert current.eligible_packed_tokens == current.retained_rate_tokens == 8192
    assert current.absolute_compute_floor >= 20 * 2**30
    assert current.compute_peak_floor >= 20 * 2**30
    assert max(peak for _, peak in current.compute_observations) >= 20 * 2**30
