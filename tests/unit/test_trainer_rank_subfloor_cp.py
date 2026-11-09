"""Cold sub-floor CP admission and isolation from larger-wave learning."""

from contextlib import contextmanager
from datetime import timedelta
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from test_trainer_rank_calibration_seed import fake_device, metadata_rank
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.flex_attn import compiled
from art.trainer_rank import ForwardInput, _impl
from art.trainer_rank import _memory as memory

SHAPE = (4023, 2971, 2971, *([2970] * 13))
SIZES = (1, 4, 16, 72)


def requests(lengths, backward):
    values = []
    for index, n in enumerate(lengths):
        tokens = torch.arange(n) % 1024
        tokens[0] = 1024 + index
        values.append(
            ForwardInput(
                input_tokens=tokens, target_tokens=tokens, no_grad=not backward
            )
        )
    return values


def rank_metadata(kind, layout="seed"):
    rank = (
        metadata_rank(kind, head_dim=128)
        if layout == "acceptance"
        else metadata_rank(kind)
    )
    rank._available_memory_bytes = lambda sample=None: 71_000_000_000
    return rank


@contextmanager
def short_kernel():
    entry = torch.compile(
        compiled._sparse_flex_attention_with_options(
            {"BACKEND": "FLASH", "num_stages": 1}
        )
    )
    with patch.object(
        compiled, "get_sparse_compiled_flex_attention", lambda **kw: entry
    ):
        yield


def calibrate(monkeypatch, rank):
    for backward in (True, False):
        with torch.set_grad_enabled(backward):
            for n, compute in (
                (16384, 2_200_000_000),
                (8192, 1_300_000_000),
                (512, 1_300_000_000),
                (2048, 1_300_000_000),
                (4096, 1_300_000_000),
            ):
                inputs = requests([n], backward)
                plan = rank._plan_flat_forward(inputs)
                fake_device(monkeypatch, rank, compute + plan.output_bytes)
                if n == 16384:
                    with rank.calibration_rung(inputs, no_grad=not backward) as rung:
                        assert rung.decision.seed and rung.decision.fits
                        list(rung.forward_micro_batches())
                else:
                    assert rank._memory_check(plan).fits
                    list(rank.forward_micro_batches(inputs, no_grad=not backward))
            # Complete larger fixture observations, as in metadata replay.
            for n, compute in (
                (32768, 4_100_000_000),
                (65536, 8_700_000_000),
                (131072, 18_200_000_000),
                (262144, 37_200_000_000),
            ):
                plan = rank._plan_flat_forward(requests([n], backward))
                rank._update_memory_profile(
                    plan,
                    compute + plan.output_bytes,
                    retained_bytes=None,
                    caller_phase=True,
                )


def cold_price(rank, plan):
    with patch.object(rank, "_memory_profiles", {}):
        return rank._memory_check(plan).estimated_required_bytes


def prediction(rank, lengths, backward):
    items = tuple(_impl._materialize(r) for r in requests(lengths, backward))
    waves, start = [], 0
    with torch.set_grad_enabled(backward):
        while start < len(items):
            selected = rank._select_next_micro_batch(items, start, checkpoint=None)
            assert selected.check.fits and selected.stats_global_count > 0
            waves.append(
                (selected.plan.signature, selected.check.estimated_required_bytes)
            )
            start += selected.stats_global_count
    return waves


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("layout", ["seed", "acceptance"])
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("n", SIZES)
def test_tiny_cp_wave_after_ladder_equals_unprofiled_price(
    monkeypatch, kind, layout, backward, n
):
    with memory.fixed_slope_memory_profiles():
        rank = rank_metadata(kind, layout)
        calibrate(monkeypatch, rank)
        with torch.set_grad_enabled(backward):
            with short_kernel():
                plan = rank._plan_flat_forward(requests([n], backward))
            assert plan.signature.short_requests == (n < 64)
            assert plan.packed_tokens < 512
            old = rank._memory_profiles[plan.signature]
            if plan.caller_cp_workload is not None:
                assert old.covering_cp_peak(n, plan.caller_cp_workload) is None
                masks = json.loads(plan.caller_cp_workload[0])["masks"]
                if n < 8:
                    assert any(not kinds for kinds in masks)
            expected = cold_price(rank, plan)
            check = rank._memory_check(plan)
            assert check.fits and check.estimated_required_bytes == expected
            # V2 profiles retain this rule outside the collection context too.
            with memory.fixed_slope_memory_profiles(False):
                assert rank._memory_check(plan).estimated_required_bytes == expected


def test_distinct_selected_short_kernel_uses_the_same_cold_path(monkeypatch):
    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        rank = rank_metadata("moe", "acceptance")
        calibrate(monkeypatch, rank)
        short_entry = torch.compile(
            compiled._sparse_flex_attention_with_options(
                {"BACKEND": "FLASH", "num_stages": 1}
            )
        )
        with patch.object(
            compiled, "get_sparse_compiled_flex_attention", lambda **kw: short_entry
        ):
            plan = rank._plan_flat_forward(requests([16], True))
        profile = rank._memory_profiles[plan.signature]
        assert profile.covering_cp_peak(16, plan.caller_cp_workload) is None
        assert rank._memory_check(plan).estimated_required_bytes == cold_price(
            rank, plan
        )
        assert rank._memory_check(plan).fits


def test_native_short_class_is_cold_but_covered_72_keeps_caller_pricing(monkeypatch):
    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        rank = rank_metadata("moe", "acceptance")
        calibrate(monkeypatch, rank)
        tiny = rank._plan_flat_forward(requests([16], True))
        profile = rank._memory_profiles[tiny.signature]
        assert profile.covering_cp_peak(16, tiny.caller_cp_workload) is None
        assert rank._memory_check(tiny).estimated_required_bytes == cold_price(
            rank, tiny
        )
        covered = rank._plan_flat_forward(requests([72], True))
        assert profile.covering_cp_peak(72, covered.caller_cp_workload) is not None
        assert not memory._uncovered_subfloor_cp(
            profile,
            covered.signature,
            covered.packed_tokens,
            covered.caller_cp_workload,
        )
        check = rank._memory_check(covered)
        assert check.fits and check.estimated_required_bytes > cold_price(rank, covered)


def test_completed_cold_sample_is_reported_without_changing_large_profile(monkeypatch):
    with memory.fixed_slope_memory_profiles(), torch.enable_grad():
        rank = rank_metadata("moe", "acceptance")
        calibrate(monkeypatch, rank)
        saved = dict(rank._memory_profiles)
        executed, _ = fake_device(monkeypatch, rank, 100_000_000)
        list(rank.forward_micro_batches(requests([16], True), no_grad=False))
        assert len(executed) == 1
        assert rank._peak_reading[1] == 100_000_000
        assert rank._memory_profiles == saved
        assert rank._suppressed_caller_memory_samples == 0


@pytest.mark.parametrize("n", [512, 1024, 8192])
def test_uncovered_cp_at_or_above_global_floor_still_refuses(monkeypatch, n):
    with memory.fixed_slope_memory_profiles():
        rank = rank_metadata("moe")
        calibrate(monkeypatch, rank)
        with patch.dict(compiled._FORCED_FLEX_KERNEL_OPTIONS, num_stages=9):
            plan = rank._plan_flat_forward(requests([n], True))
        profile = rank._memory_profiles[plan.signature]
        assert profile.covering_cp_peak(n, plan.caller_cp_workload) is None
        check = rank._memory_check(plan)
        assert not check.fits and check.estimated_required_bytes >= 1 << 62


@pytest.mark.parametrize("layout", ["seed", "acceptance"])
def test_subfloor_samples_cannot_inflate_multi_root_step_or_large_forward(
    monkeypatch, layout
):
    assert len(SHAPE) == 16 and sum(SHAPE) == 48575
    with memory.fixed_slope_memory_profiles():
        rank = rank_metadata("moe", layout)
        calibrate(monkeypatch, rank)
        clean = prediction(rank, SHAPE, True)
        large = prediction(rank, [131072], False)
        saved = dict(rank._memory_profiles)
        for backward in (True, False):
            with torch.set_grad_enabled(backward):
                for n in SIZES:
                    with short_kernel():
                        plan = rank._plan_flat_forward(requests([n], backward))
                    assert rank._memory_check(plan).fits
                    # Adversarial synthetic readings stress every learned floor;
                    # these values are not claimed to fit the cold prediction.
                    rank._update_memory_profile(
                        plan,
                        60_000_000_000,
                        retained_bytes=50_000_000_000,
                        caller_phase=True,
                    )
        assert rank._memory_profiles == saved
        assert all(rank._memory_profiles[s] is p for s, p in saved.items())
        assert prediction(rank, SHAPE, True) == clean
        assert prediction(rank, [131072], False) == large


def _agreement_worker(index, directory):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        rank = rank_metadata("moe")
        rank._plan_head_workspace_bytes = lambda plan: (index + 1) * 100_000_000
        operations = []
        original = dist.all_reduce

        def reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
            operations.append((str(op), list(tensor.shape), str(tensor.dtype)))
            return original(tensor, op=op, **kwargs)

        with (
            memory.fixed_slope_memory_profiles(),
            patch.object(dist, "all_reduce", reduce),
        ):
            observed = rank._plan_flat_forward(requests([16384], True))
            rank._update_memory_profile(
                observed, 2_200_000_000, retained_bytes=None, caller_phase=True
            )
            tiny = rank._plan_flat_forward(requests([16], True))
            check = rank._memory_check(tiny)
            Path(directory, f"rank{index}.json").write_text(
                json.dumps(
                    dict(
                        required=check.estimated_required_bytes,
                        fits=check.fits,
                        operations=operations,
                    )
                )
            )
    finally:
        dist.destroy_process_group()


def test_cold_subfloor_decision_agrees_with_different_rank_local_static_terms(tmp_path):
    mp.start_processes(
        _agreement_worker,
        args=(str(tmp_path),),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    a, b = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    assert a == b and a["fits"]
    assert a["operations"][-2:] == [
        ["RedOpType.MAX", [], "torch.float64"],
        ["RedOpType.MIN", [2], "torch.float64"],
    ]
