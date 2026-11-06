"""Report replay must continue to work for newly eligible CP callers."""

from dataclasses import replace

import pytest
from test_grouped_planner_replay import emitted
from test_trainer_rank_pending_memory import layer as layer
from test_trainer_rank_pending_memory import pending_rank as pending_rank
import torch

from art.megatron.context_parallel.types import ParallelTopology
from art.trainer_rank import _impl as tr
from art.trainer_rank import _planner_misses as reports


@pytest.mark.parametrize("cp", (1, 2, 4, 8))
@pytest.mark.parametrize("calibrated", (False, True))
def test_cp_caller_report_replays(cp, calibrated, pending_rank, monkeypatch, tmp_path):
    rank = pending_rank
    rank._parallel_shape = replace(rank._parallel_shape, cp=cp)
    monkeypatch.setattr(
        tr.TrainerRank, "_topology", lambda self: ParallelTopology(cp=cp)
    )
    rank.runtime.model_support_handler.context_parallel_workload_profile = (
        lambda provider: None
    )
    item = tr.ForwardInput(
        input_tokens=torch.arange(2048), target_tokens=torch.arange(2048)
    )
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    plan = rank._plan_flat_forward([item])
    if calibrated:
        rank._update_memory_profile(
            plan, 256 * 2**30, retained_bytes=None, caller_phase=True
        )
        rank._update_memory_profile(
            plan, 300 * 2**30, retained_bytes=None, caller_phase=False
        )
    report, costs = emitted(rank, plan, tmp_path)
    assert report["replay_complete"], report["incomplete_reasons"]
    result = reports.replay(report)
    assert result["aggregate"]["matches"]
    assert all(item["matches"] for item in result["estimates"] + result["layouts"])


@pytest.mark.parametrize("cp", (1, 2, 4, 8))
@pytest.mark.parametrize("scope", ("caller", "forward", "checkpoint"))
def test_scoped_memory_json_roundtrip(cp, scope):
    from dataclasses import asdict
    import json

    from art.trainer_rank import _memory as memory

    signature = memory._ScopedMemorySignature(
        topology=(1, 1, cp, 1),
        planner_coefficients=(2, "test"),
        slot_group_count=1,
        request_mix=("target:single",),
        grad_enabled=True,
        grad_modes=(True,),
        bounded_caller_peak=scope == "caller",
        bounded_forward_peak=scope == "forward",
        checkpoint_partition=(2, (2048, 1, 128)) if scope == "checkpoint" else None,
    )
    profile = (
        memory._CallerMemoryProfile(
            bytes_per_token=10,
            packed_tokens=8192,
            caller_compute_peaks=((128, 1000), (8192, 2000)),
            compute_peak_floor=3000,
        )
        if scope == "caller"
        else memory._BoundedMemoryProfile(
            bytes_per_token=10,
            packed_tokens=8192,
            forward_compute_peaks=((128, 1000), (8192, 2000)),
        )
        if scope == "forward"
        else tr._MemoryProfile(bytes_per_token=10, packed_tokens=8192)
    )
    restored_signature = reports._memory_signature(
        json.loads(json.dumps(asdict(signature)))
    )
    restored_profile = reports._memory_profile(json.loads(json.dumps(asdict(profile))))
    assert type(restored_signature) is type(signature)
    assert restored_signature == signature
    assert hash(restored_signature) == hash(signature)
    assert type(restored_profile) is type(profile)
    assert restored_profile == profile
